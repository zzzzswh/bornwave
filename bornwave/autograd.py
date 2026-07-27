"""Differentiable Helmholtz solve via the implicit function theorem.

The physical staggered first-order system is

    D(m) w = s_hat,     D = Diag(d_ux, d_uz, d_p) + Ell,

where Ell (staggered grad/div, incl. stagger phases) is model-independent
and the model enters ONLY through the three diagonal fields. For a scalar
loss J, the adjoint-state gradients are

    D^H lam = g_w                 (one adjoint solve, same CBS machinery:
                                   V -> conj(V), symbol -> per-k conj-transpose)
    dJ/dd_ch = - lam_ch * conj(w_ch)     (pointwise zero-lag correlation)
    dJ/ds_hat = lam

(cotangents in torch's conj-Wirtinger convention). The preconditioner
internals (shifts a, scales lam, symbols) are built from DETACHED
diagonals: the converged solution does not depend on them, so this is
exact, not an approximation. Backward stores only the forward solution
-> O(1) memory in the iteration count.

Chain rules d(c0, rho0, alpha) -> (d_ux, d_uz, d_p) and s_hat(sp) are
plain pointwise torch ops handled by native autograd; see
`solve_helmholtz` for the user-facing differentiable entry point.
"""
from __future__ import annotations

import torch

from .grid import next_fast_len, pad_replicate_2d, sponge_profile_1d, stagger_avg
from .operators import assemble_symbols, complex_median

__all__ = ["solve_helmholtz", "HelmholtzDiagSolve"]


class _ScaledCBS:
    """Preconditioner + fixed-point loops for given PADDED diagonals.

    Built from detached tensors; provides solve() for A x = y and
    solve_adjoint() for A^H x = y in the scaled variables, plus the
    C^{1/2} channel scaling."""

    def __init__(self, d_ux, d_uz, d_p, dx, beta=0.95, nu=0.9):
        self.nu = float(nu)
        nz, nx = d_p.shape
        self.shape = (nz, nx)
        cdtype = d_p.dtype
        device = d_p.device

        d_u_all = torch.cat([d_ux.flatten(), d_uz.flatten()])
        a1 = complex_median(d_u_all)
        a2 = complex_median(d_p.flatten())
        lam1 = (d_u_all - a1).abs().max().item() / beta
        lam2 = (d_p.flatten() - a2).abs().max().item() / beta
        lam1 = max(lam1, 1e-12 * abs(a1))
        lam2 = max(lam2, 1e-12 * abs(a2))

        self.V = torch.stack(
            [(d_ux - a1) / lam1, (d_uz - a1) / lam1, (d_p - a2) / lam2]
        ).to(cdtype)
        self.B = 1.0 - self.V
        self.M_inv, self.M_fwd = assemble_symbols(
            nz, nx, dx, a1, a2, lam1, lam2, cdtype=cdtype, device=device,
            compute_forward=True,
        )
        self.sqrt_lam = torch.tensor(
            [lam1**0.5, lam1**0.5, lam2**0.5], dtype=cdtype, device=device
        ).view(3, 1, 1)

    # -- symbol application: M and M^H, both unrolled --------------------
    def _sym(self, x):
        M = self.M_inv
        X = torch.fft.fftn(x, dim=(-2, -1))
        x0, x1, x2 = X[..., 0, :, :], X[..., 1, :, :], X[..., 2, :, :]
        Y = torch.stack(
            [M[0, 0] * x0 + M[0, 1] * x1 + M[0, 2] * x2,
             M[1, 0] * x0 + M[1, 1] * x1 + M[1, 2] * x2,
             M[2, 0] * x0 + M[2, 1] * x1 + M[2, 2] * x2], dim=-3)
        return torch.fft.ifftn(Y, dim=(-2, -1))

    def _sym_H(self, x):
        """Per-k conjugate TRANSPOSE of M_inv (validated adjoint symbol):
        Y_i = sum_j conj(M[j, i]) X_j. conj() is a zero-copy view."""
        M = self.M_inv
        X = torch.fft.fftn(x, dim=(-2, -1))
        x0, x1, x2 = X[..., 0, :, :], X[..., 1, :, :], X[..., 2, :, :]
        Y = torch.stack(
            [M[0, 0].conj() * x0 + M[1, 0].conj() * x1 + M[2, 0].conj() * x2,
             M[0, 1].conj() * x0 + M[1, 1].conj() * x1 + M[2, 1].conj() * x2,
             M[0, 2].conj() * x0 + M[1, 2].conj() * x1 + M[2, 2].conj() * x2],
            dim=-3)
        return torch.fft.ifftn(Y, dim=(-2, -1))

    # -- fixed-point loops ------------------------------------------------
    @torch.no_grad()
    def _iterate(self, y, B, sym, tol, max_iter, check_every):
        x = torch.zeros_like(y)
        nu = self.nu
        for it in range(1, max_iter + 1):
            z = sym(B * x + y)
            upd = nu * (B * (z - x))
            x += upd
            if it % check_every == 0:
                rel = (upd.norm() / x.norm().clamp_min(1e-30)).item()
                if rel != rel:
                    raise FloatingPointError("iteration diverged (NaN)")
                if rel < tol:
                    return x, it
        raise RuntimeError(f"CBS did not converge in {max_iter} iterations")

    def solve(self, y, tol, max_iter, check_every=25):
        return self._iterate(y, self.B, self._sym, tol, max_iter, check_every)

    def solve_adjoint(self, y, tol, max_iter, check_every=25):
        """A^H x = y: split (L^H, V^H); V^H = conj(V) pointwise, and
        ((L+I)^H)^{-1} = per-k conj-transpose of M_inv."""
        return self._iterate(y, self.B.conj(), self._sym_H, tol, max_iter,
                             check_every)


class HelmholtzDiagSolve(torch.autograd.Function):
    """w = D^{-1} s_hat with D = Diag(d_ux, d_uz, d_p) + Ell.

    All tensors PADDED (Nz, Nx) complex; s_hat may carry leading batch
    dims (..., 3, Nz, Nx). Returns w with the same shape as s_hat."""

    @staticmethod
    def forward(ctx, d_ux, d_uz, d_p, s_hat, dx, tol, max_iter):
        op = _ScaledCBS(d_ux.detach(), d_uz.detach(), d_p.detach(), dx)
        y = s_hat.detach() / op.sqrt_lam            # C^{-1/2} s_hat
        x, _ = op.solve(y, tol, max_iter)
        w = x / op.sqrt_lam                          # C^{-1/2} x
        ctx.save_for_backward(w)
        ctx.op, ctx.tol, ctx.max_iter = op, tol, max_iter
        return w

    @staticmethod
    def backward(ctx, g_w):
        (w,) = ctx.saved_tensors
        op, tol, max_iter = ctx.op, ctx.tol, ctx.max_iter
        # D^H lam = g_w  <=>  lam = C^{-1/2} A^{-H} C^{-1/2} g_w
        g = g_w / op.sqrt_lam
        lam_s, _ = op.solve_adjoint(g, tol, max_iter)
        lam = lam_s / op.sqrt_lam

        grad_d = -lam * w.conj()                     # (..., 3, Nz, Nx)
        if grad_d.ndim > 3:                          # sum over source batch
            grad_d = grad_d.reshape(-1, *grad_d.shape[-3:]).sum(0)
        g_dux, g_duz, g_dp = grad_d[0], grad_d[1], grad_d[2]
        return g_dux, g_duz, g_dp, lam, None, None, None


def solve_helmholtz(
    c0,
    rho0,
    omega: float,
    dx: float,
    sp,
    alpha=0.0,
    abs_points: int = 40,
    gamma_max_factor: float = 1.0,
    gamma_power: float = 3.0,
    tol: float = 1e-6,
    max_iter: int = 60000,
    cdtype: torch.dtype = torch.complex64,
):
    """Differentiable single-frequency solve: returns (p, ux, uz) on the
    interior grid, with gradients w.r.t. c0, rho0, alpha and sp.

    c0, rho0, alpha: (nz, nx) real tensors (requires_grad as desired).
    sp: (*batch, nz, nx) complex pressure source on the interior grid.
    The material -> diagonal chain below is plain differentiable torch;
    only the linear solve itself goes through HelmholtzDiagSolve.
    """
    rdtype = torch.float32 if cdtype == torch.complex64 else torch.float64
    c0 = torch.as_tensor(c0).to(rdtype)
    nz_int, nx_int = c0.shape
    rho0 = torch.as_tensor(rho0).to(rdtype).expand(nz_int, nx_int)
    alpha = torch.as_tensor(alpha).to(rdtype).expand(nz_int, nx_int)

    pt = int(abs_points)
    nz_pad, nx_pad = nz_int + 2 * pt, nx_int + 2 * pt
    nz_tot, nx_tot = next_fast_len(nz_pad), next_fast_len(nx_pad)
    pb = pt + (nz_tot - nz_pad)
    pr = pt + (nx_tot - nx_pad)
    interior = (slice(pt, pt + nz_int), slice(pt, pt + nx_int))

    c0p = pad_replicate_2d(c0, (pt, pb, pt, pr))
    rhop = pad_replicate_2d(rho0.contiguous(), (pt, pb, pt, pr))
    alp = pad_replicate_2d(alpha.contiguous(), (pt, pb, pt, pr))

    gmax = gamma_max_factor * float(omega)
    gp = dict(gamma_max=gmax, power=gamma_power, dtype=rdtype,
              device=c0p.device)
    gx_c = sponge_profile_1d(nx_tot, pt, pr, offset=0.0, **gp)
    gx_s = sponge_profile_1d(nx_tot, pt, pr, offset=0.5, **gp)
    gz_c = sponge_profile_1d(nz_tot, pt, pb, offset=0.0, **gp)
    gz_s = sponge_profile_1d(nz_tot, pt, pb, offset=0.5, **gp)
    gam_p = gz_c[:, None] + gx_c[None, :]
    gam_ux = gz_c[:, None] + gx_s[None, :]
    gam_uz = gz_s[:, None] + gx_c[None, :]

    w = float(omega)
    c2 = (c0p.to(cdtype)) ** 2 / (1.0 - 2.0j * (alp * c0p).to(cdtype) / w)
    rho_sx = stagger_avg(rhop, dim=1)
    rho_sz = stagger_avg(rhop, dim=0)
    d_ux = rho_sx.to(cdtype) * (1j * w + gam_ux.to(cdtype))
    d_uz = rho_sz.to(cdtype) * (1j * w + gam_uz.to(cdtype))
    d_p = (gam_p.to(cdtype) + 1j * w) / (rhop.to(cdtype) * c2)

    sp = torch.as_tensor(sp).to(cdtype)
    batch = tuple(sp.shape[:-2])
    s_hat = torch.zeros(batch + (3, nz_tot, nx_tot), dtype=cdtype,
                        device=sp.device)
    iz, ix = interior
    s_hat[..., 2, iz, ix] = sp / (rhop.to(cdtype) * c2)[iz, ix]

    wfield = HelmholtzDiagSolve.apply(d_ux, d_uz, d_p, s_hat, dx, tol,
                                      max_iter)
    return (wfield[..., 2, iz, ix], wfield[..., 0, iz, ix],
            wfield[..., 1, iz, ix])