"""Strong-contrast disc inclusion: convergence + acoustic reciprocity.

For the variable-density Helmholtz operator div(rho^{-1} grad p) +
omega^2/(rho c^2) p, the Green's function is symmetric, so for two point
pressure sources sp = delta at A and B:

    p(B | source A) * rho(A) c^2(A) = p(A | source B) * rho(B) c^2(B).

Both sources sit in the background here, so the fields must match directly.
The two shots are solved as ONE batched iteration (shape (2, 3, Nz, Nx)),
exercising the shot-batch dimension: operators are shared, only y differs.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from bornwave import CBSSolver2D, point_source_2d


def run(verbose=True):
    c_bg, rho_bg = 1500.0, 1000.0
    c_in, rho_in = 3000.0, 2500.0     # 2x speed, 2.5x density contrast
    f = 50.0
    omega = 2 * np.pi * f
    dx = (c_bg / f) / 10.0            # 10 ppw in the background
    n = 128

    zz, xx = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    disc = ((zz - n / 2) ** 2 + (xx - n / 2) ** 2) < 20**2
    c0 = torch.full((n, n), c_bg, dtype=torch.float64)
    rho0 = torch.full((n, n), rho_bg, dtype=torch.float64)
    c0[disc], rho0[disc] = c_in, rho_in

    solver = CBSSolver2D(c0, rho0, omega, dx, abs_points=40)

    A = (20, n // 2)   # above the disc
    B = (108, n // 2)  # below the disc
    sp = torch.stack([
        point_source_2d(n, n, *A, dx),
        point_source_2d(n, n, *B, dx),
    ])                                  # batched shots: (2, n, n)

    t0 = time.time()
    res = solver.solve(sp=sp, tol=1e-6, max_iter=40000, check_every=50)
    if verbose:
        print(f"iterations={res.iterations}  rel_increment={res.rel_increment:.2e}  "
              f"rel_residual={res.rel_residual:.2e}  wall={time.time()-t0:.1f}s  "
              f"batch p shape={tuple(res.p.shape)}")
    assert res.converged
    assert res.rel_residual < 1e-4

    p_B_from_A = res.p[0, B[0], B[1]].item()
    p_A_from_B = res.p[1, A[0], A[1]].item()
    rec_err = abs(p_B_from_A - p_A_from_B) / abs(p_B_from_A)
    if verbose:
        print(f"p(B|A) = {p_B_from_A:.6e}")
        print(f"p(A|B) = {p_A_from_B:.6e}")
        print(f"reciprocity relative error = {rec_err:.3e}")
    assert rec_err < 1e-2, f"reciprocity violated: {rec_err:.3e}"
    return res


if __name__ == "__main__":
    run()
    print("heterogeneous reciprocity test passed")