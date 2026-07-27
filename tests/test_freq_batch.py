"""Frequency batch vs serial consistency.

Same two-layer model, 8 adjacent frequency bins solved (a) one-by-one with
CBSSolver2D and (b) as a single compacting batch with CBSFreqBatch2D. The
transfer functions at a line of receivers must agree to iteration accuracy,
and per-frequency shifts/scales must match exactly.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from bornwave import CBSSolver2D, point_source_2d
from bornwave.multifreq import CBSFreqBatch2D


def run(device=None, verbose=True):
    nz, nx, dx = 96, 128, 7.5
    c0 = torch.full((nz, nx), 1500.0, dtype=torch.float64)
    rho0 = torch.full((nz, nx), 1000.0, dtype=torch.float64)
    c0[56:], rho0[56:] = 2500.0, 2200.0

    freqs = np.linspace(18.0, 25.0, 8)
    omegas = 2 * np.pi * freqs
    src = (12, 64)
    rec_z, rec_x = 12, np.arange(8, 121, 4)
    tol = 1e-5

    # ---- serial reference --------------------------------------------
    H_serial = np.zeros((freqs.size, rec_x.size), np.complex128)
    a1s, lam1s = [], []
    t0 = time.time()
    for i, om in enumerate(omegas):
        s = CBSSolver2D(c0, rho0, om, dx, abs_points=40, device=device)
        res = s.solve(sp=point_source_2d(nz, nx, *src, dx, device=device),
                      tol=tol, max_iter=60000)
        assert res.converged
        H_serial[i] = res.p[rec_z, rec_x].cpu().numpy()
        a1s.append(s.a1); lam1s.append(s.lam1)
    t_serial = time.time() - t0

    # ---- batched ------------------------------------------------------
    t0 = time.time()
    solver = CBSFreqBatch2D(c0, rho0, omegas, dx, abs_points=40, device=device)
    fields, iters, resid = solver.solve(
        point_source_2d(nz, nx, *src, dx, device=device), tol=tol,
        max_iter=60000)
    t_batch = time.time() - t0
    H_batch = fields[:, 2].cpu().numpy()[:, rec_z, rec_x]

    # shifts/scales must match the serial construction exactly
    da = max(abs(complex(solver.a1[i].item()) - a1s[i]) / abs(a1s[i])
             for i in range(freqs.size))
    dl = max(abs(solver.lam1[i].item() - lam1s[i]) / lam1s[i]
             for i in range(freqs.size))

    err = (np.linalg.norm(H_batch - H_serial, axis=1)
           / np.linalg.norm(H_serial, axis=1))
    if verbose:
        print(f"serial {t_serial:.1f}s vs batched {t_batch:.1f}s "
              f"(iters per bin: {iters.tolist()})")
        print(f"max residual (batched): {resid.max():.1e}")
        print(f"shift/scale agreement: da1={da:.2e}, dlam1={dl:.2e}")
        print(f"per-frequency |H_batch - H_serial|/|H_serial|: "
              f"max={err.max():.2e}")
    assert da < 1e-10 and dl < 1e-10
    assert err.max() < 5e-3, f"batched/serial mismatch: {err.max():.2e}"
    return err


if __name__ == "__main__":
    run()
    print("frequency-batch consistency test passed")