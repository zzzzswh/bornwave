"""Multi-frequency synthesis: wavelet -> band of Helmholtz solves -> traces.

Convention (empirically pinned by tests/test_homogeneous_hankel.py): the
solver realizes the +iwt branch, which coincides with the numpy/torch FFT
transfer-function convention, so

    d(t) = irfft( W(f_k) * H(x_r, f_k) ,  n = nt )

with W = rfft(wavelet) and H the monochromatic pressure response to a
unit point source -- no conjugation anywhere.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .solver import CBSSolver2D, point_source_2d

__all__ = ["ricker", "band_indices", "synthesize_shot"]


def ricker(f0: float, nt: int, dt: float, t0: float | None = None):
    """Ricker wavelet with peak frequency f0, delayed by t0 (default 1.5/f0).

    Returns (w, t0) with w a float64 numpy array of length nt.
    """
    if t0 is None:
        t0 = 1.5 / f0
    t = np.arange(nt) * dt - t0
    a = (np.pi * f0 * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a), t0


def band_indices(spectrum: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    """Indices k>0 of rfft bins with |W_k| >= threshold * max|W|."""
    keep = np.nonzero(np.abs(spectrum) >= threshold * np.abs(spectrum).max())[0]
    return keep[keep > 0]


def synthesize_shot(
    c0,
    rho0,
    dx: float,
    src: tuple[int, int],
    rec_z,
    rec_x,
    nt: int,
    dt: float,
    f0: float = 15.0,
    t0: float | None = None,
    wavelet: np.ndarray | None = None,
    alpha=0.0,
    amp_threshold: float = 1e-3,
    tol: float = 2e-4,
    max_iter: int = 60000,
    abs_points: int = 40,
    freq_batch: int = 16,
    dtype: torch.dtype = torch.complex64,
    device=None,
    verbose: bool = True,
):
    """Synthesize a time-domain shot gather on an arbitrary (c0, rho0, alpha)
    2D model by solving one Helmholtz problem per retained frequency bin.

    src = (iz, ix) grid indices of the point pressure source (interior frame).
    rec_z, rec_x: receiver grid indices (rec_z may be a scalar).
    Returns a dict with the gather (nt, nrec), time axis, spectra and
    per-frequency diagnostics.
    """
    c0 = torch.as_tensor(c0, dtype=torch.float64)
    nz, nx = c0.shape
    rec_x = np.atleast_1d(np.asarray(rec_x, dtype=int))
    rec_z = np.broadcast_to(np.asarray(rec_z, dtype=int), rec_x.shape).copy()
    nrec = rec_x.size

    if wavelet is None:
        wavelet, t0 = ricker(f0, nt, dt, t0)
    W = np.fft.rfft(wavelet)
    freqs = np.fft.rfftfreq(nt, dt)
    keep = band_indices(W, amp_threshold)

    H = np.zeros((freqs.size, nrec), dtype=np.complex128)
    iters, resid = [], []
    if verbose:
        dev = torch.device(device) if device is not None else torch.device("cpu")
        print(f"  [synthesize_shot] device={dev}, freq_batch={freq_batch}, "
              f"{keep.size} freqs in [{freqs[keep[0]]:.2f}, {freqs[keep[-1]]:.2f}] Hz",
              flush=True)
    t_start = time.time()

    if freq_batch > 1:
        from .multifreq import CBSFreqBatch2D
        sp = point_source_2d(nz, nx, src[0], src[1], dx, dtype=dtype,
                             device=device)
        for c0_idx in range(0, keep.size, freq_batch):
            chunk = keep[c0_idx:c0_idx + freq_batch]
            omegas = 2.0 * np.pi * freqs[chunk]
            solver = CBSFreqBatch2D(
                c0, rho0, omegas, dx, alpha=alpha, abs_points=abs_points,
                dtype=dtype, device=device)
            fields, its, rs = solver.solve(sp, tol=tol, max_iter=max_iter)
            p = fields[:, 2].cpu().numpy()               # (Fc, nz, nx)
            H[chunk] = p[:, rec_z, rec_x]
            iters.extend(its.tolist())
            resid.extend(rs.tolist())
            if verbose:
                print(f"  chunk {chunk[0]:3d}-{chunk[-1]:3d} "
                      f"({freqs[chunk[0]]:5.2f}-{freqs[chunk[-1]]:5.2f} Hz)  "
                      f"iters<= {int(its.max()):5d}  "
                      f"max residual={rs.max():.1e}  "
                      f"elapsed={time.time() - t_start:6.1f}s", flush=True)
        gather = np.fft.irfft(W[:, None] * H, n=nt, axis=0)
        return {
            "gather": gather, "t": np.arange(nt) * dt, "t0": t0,
            "wavelet": wavelet, "W": W, "H": H, "freqs": freqs, "band": keep,
            "iterations": np.asarray(iters), "residuals": np.asarray(resid),
            "wall_time": time.time() - t_start,
        }
    for j, k in enumerate(keep):
        omega = 2.0 * np.pi * freqs[k]
        solver = CBSSolver2D(
            c0, rho0, omega, dx, alpha=alpha, abs_points=abs_points,
            dtype=dtype, device=device,
        )
        sp = point_source_2d(nz, nx, src[0], src[1], dx, dtype=dtype,
                             device=device)
        res = solver.solve(sp=sp, tol=tol, max_iter=max_iter)
        if not res.converged:
            raise RuntimeError(
                f"frequency {freqs[k]:.2f} Hz did not converge "
                f"(increment {res.rel_increment:.2e})")
        H[k] = res.p[rec_z, rec_x].cpu().numpy()
        iters.append(res.iterations)
        resid.append(res.rel_residual)
        if verbose:
            print(f"  [{j + 1:3d}/{keep.size}] f={freqs[k]:6.2f} Hz  "
                  f"iters={res.iterations:5d}  residual={res.rel_residual:.1e}  "
                  f"elapsed={time.time() - t_start:6.1f}s", flush=True)

    gather = np.fft.irfft(W[:, None] * H, n=nt, axis=0)
    return {
        "gather": gather,                       # (nt, nrec) float64
        "t": np.arange(nt) * dt,
        "t0": t0,
        "wavelet": wavelet,
        "W": W,
        "H": H,                                 # (nfreq_all, nrec) transfer fn
        "freqs": freqs,
        "band": keep,
        "iterations": np.asarray(iters),
        "residuals": np.asarray(resid),
        "wall_time": time.time() - t_start,
    }