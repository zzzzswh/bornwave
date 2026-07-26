"""Homogeneous medium vs the analytic 2D free-space Green's function.

Point pressure source in constant (c0, rho0); compare p on an annulus
1.2*lambda < r < 6.5*lambda (excluding the source-sinc region and the
sponge, as in the paper's Fig. 2 protocol) against

    p(r) = (w/(4 c^2)) H0^(2)(k r)   [+iwt / torch-FFT convention]

and against its conjugate (H0^(1), -iwt) to pin down the convention.
Also reports the complex least-squares scale numeric/analytic to expose
any amplitude or phase bias in the source normalization.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from bornwave import CBSSolver2D, point_source_2d, pressure_point_source_2d


def run(dtype=torch.complex64, verbose=True):
    c0, rho0 = 1500.0, 1000.0
    f = 50.0
    omega = 2 * np.pi * f
    lam = c0 / f                      # 30 m
    dx = lam / 10                     # 10 points per wavelength
    n_int = 160                       # interior: 16 lambda across

    c0_map = torch.full((n_int, n_int), c0, dtype=torch.float64)
    solver = CBSSolver2D(c0_map, rho0, omega, dx, abs_points=40, dtype=dtype)
    if verbose:
        print(f"padded grid {solver.shape}, a1={solver.a1:.3e}, a2={solver.a2:.3e}, "
              f"lam1={solver.lam1:.3e}, lam2={solver.lam2:.3e}")

    isz = isx = n_int // 2
    sp = point_source_2d(n_int, n_int, isz, isx, dx, dtype=dtype)

    t0 = time.time()
    res = solver.solve(sp=sp, tol=1e-6, max_iter=30000, check_every=25)
    if verbose:
        print(f"iterations={res.iterations}  rel_increment={res.rel_increment:.2e}  "
              f"rel_residual={res.rel_residual:.2e}  wall={time.time()-t0:.1f}s")
    assert res.converged, "solver did not converge"
    assert res.rel_residual < 1e-4, f"true residual too large: {res.rel_residual}"

    # ---- compare on annulus -------------------------------------------
    p = res.p.cpu().numpy()
    zz, xx = np.meshgrid(np.arange(n_int), np.arange(n_int), indexing="ij")
    r = dx * np.hypot(zz - isz, xx - isx)
    mask = (r > 1.2 * lam) & (r < 6.5 * lam)

    errs = {}
    for conv in ("+iwt", "-iwt"):
        ref = pressure_point_source_2d(r[mask], omega, c0, convention=conv)
        errs[conv] = np.linalg.norm(p[mask] - ref) / np.linalg.norm(ref)
    best = min(errs, key=errs.get)

    ref = pressure_point_source_2d(r[mask], omega, c0, convention=best)
    scale = np.vdot(ref, p[mask]) / np.vdot(ref, ref)  # complex lsq p ~ scale*ref
    if verbose:
        print(f"rel L2 error: +iwt(H2)={errs['+iwt']:.3e}  -iwt(H1)={errs['-iwt']:.3e}")
        print(f"best convention: {best};  lsq scale |s|={abs(scale):.4f} "
              f"arg(s)={np.degrees(np.angle(scale)):.2f} deg")

    assert errs[best] < 0.02, f"analytic mismatch: {errs[best]:.3e}"
    assert abs(abs(scale) - 1.0) < 0.02, f"amplitude bias: |s|={abs(scale):.4f}"
    return best, errs, res


if __name__ == "__main__":
    best, errs, _ = run()
    print(f"homogeneous Hankel test passed (convention: {best})")