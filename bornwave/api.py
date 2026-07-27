"""One-call forward-modeling engine API.

    from bornwave import acoustic2d

    res = acoustic2d(vp, rho, dh, dt, nt, f0,
                     sx=[200], sz=[10],
                     rx=range(0, 400, 2), rz=10,
                     nbc=60, snap_interval=25)

    res.seis_p / res.seis_vx / res.seis_vz    # (nt, nrec) time-domain records
    res.snaps                                  # (nsnap, nz, nx) wavefield movie
    res.stats.kernel_time_s                    # timing / iteration diagnostics

Under the hood: Ricker (or user) wavelet -> retained rfft band -> chunks of
adjacent frequencies solved jointly with all shots as ONE batched CBS
iteration (CUDA-graph accelerated on GPU) -> records by irfft of W*H,
wavefield snapshots by exact band-limited time-slice synthesis (never
materializes the full (nt, nz, nx) cube).

Conventions
-----------
* Arrays are (nz, nx): vp[z, x]. All indices 0-based; sx/rx are x-indices,
  sz/rz are z-indices.
* vx/vz records are the staggered u_x (x + dh/2) / u_z (z + dh/2) fields at
  the receiver cells; for dh at seismic scales the half-cell offset is far
  below a wavelength.
* Vacuum cells (vp == 0 or rho == 0) are NOT representable: the CBS
  contraction requires bounded material contrast. A pressure-free flat
  surface can be emulated with the image method (planned); until then keep
  the model water/rock-filled and rely on the absorbing sponge.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import torch

from .engine import CBSFreqShotBatch2D
from .grid import next_fast_len
from .synthesis import ricker, band_indices
from .timesynth import time_slices_from_band

__all__ = ["acoustic2d", "AcousticResult"]


@dataclass
class AcousticResult:
    """Everything acoustic2d produces. Shapes below are for nshot == 1
    (a leading nshot axis is prepended when several shots are batched)."""

    seis_p: np.ndarray            # (nt, nrec) pressure records
    seis_vx: np.ndarray           # (nt, nrec) u_x records (staggered +dh/2 in x)
    seis_vz: np.ndarray           # (nt, nrec) u_z records (staggered +dh/2 in z)
    t: np.ndarray                 # (nt,) time axis [s]
    t0: float                     # wavelet delay [s]
    wavelet: np.ndarray           # (nt,) source time function
    freqs: np.ndarray             # (nt//2+1,) rfft frequency axis [Hz]
    band: np.ndarray              # retained rfft bin indices
    H_p: np.ndarray               # (nfreq, nrec) unit-source transfer function:
                                  # re-synthesize with ANY wavelet at zero cost
    snaps: np.ndarray | None      # (nsnap, nz, nx) pressure snapshots
    snap_times: np.ndarray | None # (nsnap,) snapshot times [s]
    iterations: np.ndarray        # (n_band,) CBS iterations per solved frequency
    residuals: np.ndarray         # (n_band, nshot) true relative residuals
    stats: SimpleNamespace = field(repr=False, default=None)

    def resynthesize(self, wavelet: np.ndarray) -> np.ndarray:
        """Records for a different source wavelet, reusing H_p (no solves)."""
        W = np.fft.rfft(np.asarray(wavelet), n=self.t.size)
        H = self.H_p if self.H_p.ndim == 3 else self.H_p[:, None, :]
        g = np.fft.irfft(W[:, None, None] * H, n=self.t.size, axis=0)
        g = np.moveaxis(g, 0, 1)
        return g[0] if self.H_p.ndim == 2 else g

    def __repr__(self):
        s = "AcousticResult(seis_p %s" % (self.seis_p.shape,)
        if self.snaps is not None:
            s += ", snaps %s" % (self.snaps.shape,)
        if self.stats is not None:
            s += ", kernel %.2fs, %d freqs, %d iters" % (
                self.stats.kernel_time_s, self.stats.n_freqs,
                self.stats.total_iterations)
        return s + ")"


def _as_index_array(v, n=None, name=""):
    a = np.atleast_1d(np.asarray(v, dtype=np.int64)).ravel()
    if n is not None and a.size == 1 and n > 1:
        a = np.full(n, a[0], dtype=np.int64)
    return a


def acoustic2d(
    vp,
    rho,
    dh: float,
    dt: float,
    nt: int,
    f0: float = 15.0,
    *,
    sx,
    sz,
    rx,
    rz,
    wavelet: np.ndarray | None = None,
    t0: float | None = None,
    alpha=0.0,
    Q=None,
    nbc: int | None = None,
    abs_points: int = 60,
    tol: float = 2e-4,
    amp_threshold: float = 1e-3,
    freq_batch: int = 16,
    max_iter: int = 60000,
    check_every: int = 50,
    snap_interval: int | None = None,
    cuda_graph="auto",
    dtype: torch.dtype = torch.complex64,
    device=None,
    squeeze: bool = True,
    verbose: bool = True,
) -> AcousticResult:
    """2D acoustic forward modeling on a heterogeneous (vp, rho[, alpha|Q])
    model: time-domain records at receivers, optional wavefield snapshots.

    Parameters
    ----------
    vp, rho : (nz, nx) arrays — velocity [m/s] and density [kg/m^3]
        (rho may be a scalar). Must be strictly positive (no vacuum).
    dh, dt, nt : grid spacing [m], sample interval [s], number of samples.
    f0 : Ricker peak frequency [Hz] (ignored if `wavelet` is given).
    sx, sz : shot x/z grid indices — scalars or equal-length sequences;
        several shots are solved as ONE batch (nearly free in frequency
        domain).
    rx, rz : receiver x/z grid indices (rz may be a scalar, broadcast to rx).
    alpha : absorption [Np/m] map/scalar (frequency-independent), OR
    Q : constant-Q quality factor map/scalar (mutually exclusive with alpha).
    nbc : sponge thickness in cells (alias of abs_points; 40-60 is typical —
        the sponge is a polynomial gamma ramp, not an FD boundary, so it
        does not need FD-style 100-cell widths).
    tol : stopping tolerance on the relative increment. Rule of thumb:
        2e-4 gives ~0.1-1%% amplitude accuracy (see README validation).
    snap_interval : store a full pressure wavefield every this many time
        samples (synthesized exactly from the frequency band).
    cuda_graph : True | False | "auto" — capture the CBS iteration as a CUDA
        graph (large speedup on GPU: removes per-kernel launch latency).
    device : torch device; default = "cuda" if available else "cpu".

    Returns
    -------
    AcousticResult
    """
    wall0 = time.time()

    # ---- inputs & validation ------------------------------------------
    vp = np.asarray(vp, dtype=np.float64)
    if vp.ndim != 2:
        raise ValueError("vp must be a 2D (nz, nx) array — note (nz, nx), "
                         "i.e. vp[z, x]")
    nz, nx = vp.shape
    rho = np.broadcast_to(np.asarray(rho, dtype=np.float64), (nz, nx)).copy()
    if np.any(vp <= 0) or np.any(rho <= 0):
        raise ValueError(
            "vp/rho must be strictly positive: vacuum cells put the medium "
            "contrast outside the CBS convergence domain (||V|| < 1). "
            "For a pressure-free flat surface use the image method "
            "(roadmap); meanwhile remove the zeroed cells and rely on the "
            "absorbing sponge.")
    if alpha is not None and Q is not None and np.any(np.asarray(alpha) != 0):
        raise ValueError("give either alpha or Q, not both")

    sx = _as_index_array(sx); sz = _as_index_array(sz, sx.size, "sz")
    if sx.size != sz.size:
        raise ValueError("sx and sz must have the same length")
    rx = _as_index_array(rx); rz = _as_index_array(rz, rx.size, "rz")
    if rx.size != rz.size:
        raise ValueError("rx and rz must have the same length (rz may be scalar)")
    for name, a, hi in (("sx", sx, nx), ("rx", rx, nx),
                        ("sz", sz, nz), ("rz", rz, nz)):
        if a.size and (a.min() < 0 or a.max() >= hi):
            raise ValueError(f"{name} indices out of range [0, {hi})")
    nshot, nrec = sx.size, rx.size

    if nbc is not None:
        abs_points = int(nbc)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # ---- wavelet & band --------------------------------------------------
    if wavelet is None:
        wavelet, t0 = ricker(f0, nt, dt, t0)
    else:
        wavelet = np.asarray(wavelet, dtype=np.float64)
        if wavelet.size != nt:
            raise ValueError("custom wavelet must have length nt")
        t0 = 0.0 if t0 is None else t0
    W = np.fft.rfft(wavelet)
    freqs = np.fft.rfftfreq(nt, dt)
    keep = band_indices(W, amp_threshold)
    if keep.size == 0:
        raise ValueError("empty frequency band — check wavelet/amp_threshold")

    # ---- sources ---------------------------------------------------------
    sp = torch.zeros((nshot, nz, nx), dtype=dtype, device=device)
    for b in range(nshot):
        sp[b, sz[b], sx[b]] = 1.0 / dh**2      # unit-integral delta (README #2)
    rz_t = torch.as_tensor(rz, dtype=torch.long, device=device)
    rx_t = torch.as_tensor(rx, dtype=torch.long, device=device)

    # ---- accumulators ------------------------------------------------------
    H_p = np.zeros((freqs.size, nshot, nrec), dtype=np.complex128)
    H_ux = np.zeros_like(H_p)
    H_uz = np.zeros_like(H_p)
    hf_dtype = np.complex64 if dtype == torch.complex64 else np.complex128
    Hfull = (np.zeros((keep.size, nshot, nz, nx), dtype=hf_dtype)
             if snap_interval else None)
    iters_all = np.zeros(keep.size, dtype=np.int64)
    resid_all = np.zeros((keep.size, nshot), dtype=np.float64)

    if verbose:
        npad = (next_fast_len(nz + 2 * abs_points),
                next_fast_len(nx + 2 * abs_points))
        est = (min(freq_batch, keep.size) * nshot * 3 * npad[0] * npad[1]
               * (8 if dtype == torch.complex64 else 16) * 5 / 2**30)
        print(f"[acoustic2d] device={device}  padded grid={npad}  "
              f"shots={nshot}  {keep.size} freqs in "
              f"[{freqs[keep[0]]:.2f}, {freqs[keep[-1]]:.2f}] Hz  "
              f"freq_batch={freq_batch} (~{est:.1f} GB working set/chunk)",
              flush=True)

    # ---- chunked frequency sweep -----------------------------------------
    kernel_time = 0.0
    setup_time = 0.0
    for i0 in range(0, keep.size, freq_batch):
        chunk = keep[i0:i0 + freq_batch]
        omegas = 2.0 * np.pi * freqs[chunk]

        ts = time.time()
        eng = CBSFreqShotBatch2D(
            vp, rho, omegas, dh, alpha=alpha, Q=Q, abs_points=abs_points,
            dtype=dtype, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        setup_time += time.time() - ts

        tk = time.time()
        fields, its, rs = eng.solve(sp, tol=tol, max_iter=max_iter,
                                    check_every=check_every,
                                    cuda_graph=cuda_graph)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        kernel_time += time.time() - tk

        # fields: (Fc, B, 3, nz, nx) = [ux, uz, p]
        H_ux[chunk] = fields[:, :, 0, rz_t, rx_t].cpu().numpy()
        H_uz[chunk] = fields[:, :, 1, rz_t, rx_t].cpu().numpy()
        H_p[chunk] = fields[:, :, 2, rz_t, rx_t].cpu().numpy()
        if Hfull is not None:
            Hfull[i0:i0 + chunk.size] = fields[:, :, 2].cpu().numpy()
        iters_all[i0:i0 + chunk.size] = its.numpy()
        resid_all[i0:i0 + chunk.size] = rs.numpy()
        del fields, eng

        if verbose:
            print(f"  chunk {i0:3d}-{i0 + chunk.size - 1:3d}  "
                  f"({freqs[chunk[0]]:5.2f}-{freqs[chunk[-1]]:5.2f} Hz)  "
                  f"iters<={int(its.max()):5d}  "
                  f"max residual={rs.max().item():.1e}  "
                  f"elapsed={time.time() - wall0:6.1f}s", flush=True)

    # ---- records: d(t) = irfft(W * H) (README convention #1) --------------
    def gathers(H):
        g = np.fft.irfft(W[:, None, None] * H, n=nt, axis=0)   # (nt, B, nrec)
        return np.moveaxis(g, 0, 1)                            # (B, nt, nrec)

    seis_p, seis_vx, seis_vz = gathers(H_p), gathers(H_ux), gathers(H_uz)

    # ---- wavefield snapshots ----------------------------------------------
    snaps = snap_times = None
    if Hfull is not None:
        samples = np.arange(0, nt, int(snap_interval))
        Wc = W[keep].astype(hf_dtype)
        X = Wc[:, None, None, None] * Hfull                    # (K, B, nz, nx)
        snaps = time_slices_from_band(X, keep, nt, samples)    # (S, B, nz, nx)
        snaps = np.moveaxis(snaps, 0, 1)                       # (B, S, nz, nx)
        snap_times = samples * dt

    stats = SimpleNamespace(
        kernel_time_s=kernel_time,
        setup_time_s=setup_time,
        wall_time_s=time.time() - wall0,
        device=str(device),
        n_freqs=int(keep.size),
        band_hz=(float(freqs[keep[0]]), float(freqs[keep[-1]])),
        n_shots=nshot,
        total_iterations=int(iters_all.sum()),
        max_residual=float(resid_all.max()),
        freq_batch=freq_batch,
        tol=tol,
    )
    if verbose:
        print(f"[acoustic2d] done: kernel {kernel_time:.1f}s  "
              f"setup {setup_time:.1f}s  wall {stats.wall_time_s:.1f}s  "
              f"total iterations {stats.total_iterations}", flush=True)

    if squeeze and nshot == 1:
        seis_p, seis_vx, seis_vz = seis_p[0], seis_vx[0], seis_vz[0]
        H_p = H_p[:, 0]
        if snaps is not None:
            snaps = snaps[0]

    return AcousticResult(
        seis_p=seis_p, seis_vx=seis_vx, seis_vz=seis_vz,
        t=np.arange(nt) * dt, t0=float(t0), wavelet=wavelet,
        freqs=freqs, band=keep, H_p=H_p,
        snaps=snaps, snap_times=snap_times,
        iterations=iters_all, residuals=resid_all, stats=stats,
    )
