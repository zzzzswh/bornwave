"""Implicit-function autograd: three levels of verification.

(1) adjoint machinery: A^H solve residual + adjointness <Au,v> = <u,A^Hv>
(2) torch.autograd.gradcheck on the Function (complex128, unit-scaled
    wrapper -- raw physical diagonals span ~1e12 in magnitude and
    gradcheck's ABSOLUTE eps would destroy accretivity of d_p)
(3) directional finite differences through the full material chain
    (c0, rho0, alpha, sp) -> loss.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from bornwave.autograd import _ScaledCBS, HelmholtzDiagSolve, solve_helmholtz


def test_adjoint_machinery():
    torch.manual_seed(0)
    nz, nx, dx = 60, 72, 5.0
    w = 2 * torch.pi * 30
    d_ux = ((1000 + 400 * torch.rand(nz, nx))
            * (1j * w + 50 * torch.rand(nz, nx))).to(torch.complex128)
    d_uz = ((1000 + 400 * torch.rand(nz, nx))
            * (1j * w + 50 * torch.rand(nz, nx))).to(torch.complex128)
    d_p = ((1j * w + 50 * torch.rand(nz, nx))
           / (2.25e9 * (1 + torch.rand(nz, nx)))).to(torch.complex128)
    op = _ScaledCBS(d_ux, d_uz, d_p, dx)

    def apply_with(Msym, Vd, x):
        X = torch.fft.fftn(x, dim=(-2, -1))
        Y = torch.stack([Msym(0, 0) * X[0] + Msym(0, 1) * X[1] + Msym(0, 2) * X[2],
                         Msym(1, 0) * X[0] + Msym(1, 1) * X[1] + Msym(1, 2) * X[2],
                         Msym(2, 0) * X[0] + Msym(2, 1) * X[1] + Msym(2, 2) * X[2]])
        return torch.fft.ifftn(Y, dim=(-2, -1)) - x + Vd * x

    A = lambda x: apply_with(lambda i, j: op.M_fwd[i, j], op.V, x)
    AH = lambda x: apply_with(lambda i, j: op.M_fwd[j, i].conj(), op.V.conj(), x)

    y = torch.randn(3, nz, nx, dtype=torch.complex128)
    x, _ = op.solve(y, tol=1e-12, max_iter=60000)
    xh, _ = op.solve_adjoint(y, tol=1e-12, max_iter=60000)
    r = ((y - A(x)).norm() / y.norm()).item()
    rh = ((y - AH(xh)).norm() / y.norm()).item()
    u, v = torch.randn_like(y), torch.randn_like(y)
    adj = abs(torch.vdot(A(u).flatten(), v.flatten())
              - torch.vdot(u.flatten(), AH(v).flatten()))
    adj = (adj / abs(torch.vdot(A(u).flatten(), v.flatten()))).item()
    print(f"forward residual {r:.2e}, adjoint residual {rh:.2e}, "
          f"adjointness {adj:.2e}")
    assert r < 1e-10 and rh < 1e-10 and adj < 1e-12


def test_gradcheck_function():
    torch.manual_seed(1)
    nz, nx, dx = 24, 28, 5.0
    w = 2 * torch.pi * 30
    Su, Sp = 1200.0 * w, w / 2.25e9        # unit-scaling wrappers
    du_n = ((1000 + 400 * torch.rand(nz, nx)) * (1j * w + 40 * torch.rand(nz, nx))
            / Su).to(torch.complex128).requires_grad_(True)
    dz_n = ((1000 + 400 * torch.rand(nz, nx)) * (1j * w + 40 * torch.rand(nz, nx))
            / Su).to(torch.complex128).requires_grad_(True)
    dp_n = (((1j * w + 40 * torch.rand(nz, nx))
             / (2.25e9 * (1 + torch.rand(nz, nx)))) / Sp
            ).to(torch.complex128).requires_grad_(True)
    s = torch.randn(3, nz, nx, dtype=torch.complex128, requires_grad=True)
    fn = lambda a, b, c, y: HelmholtzDiagSolve.apply(
        a * Su, b * Su, c * Sp, y, dx, 1e-13, 60000)
    assert torch.autograd.gradcheck(fn, (du_n, dz_n, dp_n, s), fast_mode=True,
                                    eps=1e-6, atol=1e-5, rtol=1e-4)
    print("gradcheck passed")


def test_full_chain_fd():
    torch.manual_seed(2)
    ni_z, ni_x = 20, 24
    c0 = (1500 + 300 * torch.rand(ni_z, ni_x)).to(torch.float64).requires_grad_(True)
    rho0 = (1000 + 200 * torch.rand(ni_z, ni_x)).to(torch.float64).requires_grad_(True)
    alpha = (1e-3 * torch.rand(ni_z, ni_x)).to(torch.float64).requires_grad_(True)
    sp = torch.randn(ni_z, ni_x, dtype=torch.complex128).requires_grad_(True)
    omega, dxg = 2 * torch.pi * 40, 6.0
    target = torch.randn(ni_z, ni_x, dtype=torch.complex128)

    def J(c0_, rho0_, alpha_, sp_):
        p, _, _ = solve_helmholtz(c0_, rho0_, omega, dxg, sp_, alpha=alpha_,
                                  abs_points=8, tol=1e-13, max_iter=60000,
                                  cdtype=torch.complex128)
        return (p - target).abs().pow(2).sum()

    J(c0, rho0, alpha, sp).backward()
    base = [c0.detach(), rho0.detach(), alpha.detach(), sp.detach()]
    for i, (name, scale, ten) in enumerate(
            zip(("c0", "rho0", "alpha", "sp"), (1500., 1000., 1e-3, 1.0),
                (c0, rho0, alpha, sp))):
        de = torch.randn_like(base[i])
        eps = 1e-6 * scale
        ap = [t.clone() for t in base]; am = [t.clone() for t in base]
        ap[i] = ap[i] + eps * de; am[i] = am[i] - eps * de
        with torch.no_grad():
            fd = (J(*ap) - J(*am)).item() / (2 * eps)
        an = torch.real(torch.vdot(ten.grad.flatten(), de.flatten())).item()
        rd = abs(fd - an) / max(abs(fd), 1e-30)
        print(f"{name:6s} FD={fd:+.6e}  autograd={an:+.6e}  rel diff={rd:.2e}")
        assert rd < 3e-5, name


if __name__ == "__main__":
    test_adjoint_machinery()
    test_gradcheck_function()
    test_full_chain_fd()
    print("autograd tests passed")