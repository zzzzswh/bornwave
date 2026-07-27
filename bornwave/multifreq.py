"""Frequency-batched CBS solver (roadmap item: (B_freq, 3, Nz, Nx)).

A chunk of F nearby frequencies is iterated as ONE tensor of shape
(F, 3, Nz, Nx). Everything frequency-dependent gets a leading F dim:
sponge gamma (gamma_max ~ omega), complex c^2 (if alpha != 0), shifts
a1/a2, scales lam1/lam2, V/B, and the (L+I)^{-1} symbols.

Frequencies that converge are finalized immediately and REMOVED from the
working tensors (compaction), so the cost of a chunk decays as bins
finish. Group ADJACENT frequencies (iteration count grows monotonically
with f) and the waste stays small.
"""
from __future__ import annotations

import torch

from .grid import next_fast_len, pad_replicate_2d, sponge_profile_1d, stagger_avg
from .operators import assemble_symbols_batched

__all__ = ["CBSFreqBatch2D"]


def _median_complex_rows(z: torch.Tensor) -> torch.Tensor:
    """(F, N) complex -> (F,) complex, component-wise median per row."""
    return torch.complex(z.real.median(dim=1).values, z.imag.median(dim=1).values)


class CBSFreqBatch2D:
    """CBS iteration over a batch of frequencies on a shared (c0, rho0,
    alpha) model. See CBSSolver2D for parameter meanings; `omegas` is a
    1D tensor/array of angular frequencies (a chunk of adjacent bins)."""

    def __init__(
        self,
        c0,
        rho0,
        omegas,
        dx: float,
        alpha=0.0,
        abs_points: int = 40,
        gamma_max_factor: float = 1.0,
        gamma_power: float = 3.0,
        beta: float = 0.95,
        nu: float = 0.9,
        dtype: torch.dtype = torch.complex64,
        device=None,
        fft_friendly: bool = True,
        keep_forward_symbol: bool = True,
    ):
        self.cdtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dx = float(dx)
        self.nu = float(nu)

        omegas = torch.as_tensor(omegas, dtype=torch.float64).flatten()
        self.omegas = omegas
        F = omegas.numel()
        w = omegas.view(F, 1, 1)                                  # (F,1,1)

        # ---- shared geometry (identical to CBSSolver2D) -------------------
        c0 = torch.as_tensor(c0, dtype=torch.float64)
        nz_int, nx_int = c0.shape
        rho0 = torch.as_tensor(rho0, dtype=torch.float64).expand(nz_int, nx_int)
        alpha = torch.as_tensor(alpha, dtype=torch.float64).expand(nz_int, nx_int)

        pt = pl = int(abs_points)
        nz_pad, nx_pad = nz_int + 2 * pt, nx_int + 2 * pl
        if fft_friendly:
            nz_tot, nx_tot = next_fast_len(nz_pad), next_fast_len(nx_pad)
        else:
            nz_tot, nx_tot = nz_pad, nx_pad
        pb = pt + (nz_tot - nz_pad)
        pr = pl + (nx_tot - nx_pad)
        self.interior = (slice(pt, pt + nz_int), slice(pl, pl + nx_int))
        self.shape = (nz_tot, nx_tot)

        c0p = pad_replicate_2d(c0, (pt, pb, pl, pr))
        rhop = pad_replicate_2d(rho0.contiguous(), (pt, pb, pl, pr))
        alp = pad_replicate_2d(alpha.contiguous(), (pt, pb, pl, pr))

        # unit-amplitude sponge ramps; gamma = gamma_max_factor * omega * base
        gp = dict(gamma_max=1.0, power=gamma_power, dtype=torch.float64)
        gx_c = sponge_profile_1d(nx_tot, pl, pr, offset=0.0, **gp)
        gx_s = sponge_profile_1d(nx_tot, pl, pr, offset=0.5, **gp)
        gz_c = sponge_profile_1d(nz_tot, pt, pb, offset=0.0, **gp)
        gz_s = sponge_profile_1d(nz_tot, pt, pb, offset=0.5, **gp)
        base_p = gz_c[:, None] + gx_c[None, :]
        base_ux = gz_c[:, None] + gx_s[None, :]
        base_uz = gz_s[:, None] + gx_c[None, :]

        gam_p = gamma_max_factor * w * base_p[None]                # (F,Nz,Nx)
        gam_ux = gamma_max_factor * w * base_ux[None]
        gam_uz = gamma_max_factor * w * base_uz[None]

        # ---- per-frequency material diagonals -----------------------------
        c2 = c0p[None] ** 2 / (1.0 - 2.0j * alp[None] * c0p[None] / w)  # (F,Nz,Nx)
        rho_sx = stagger_avg(rhop, dim=1)
        rho_sz = stagger_avg(rhop, dim=0)

        d_ux = rho_sx[None] * (1j * w + gam_ux)
        d_uz = rho_sz[None] * (1j * w + gam_uz)
        d_p = (gam_p + 1j * w) / (rhop[None] * c2)

        d_u_all = torch.cat([d_ux.reshape(F, -1), d_uz.reshape(F, -1)], dim=1)
        a1 = _median_complex_rows(d_u_all)                          # (F,)
        a2 = _median_complex_rows(d_p.reshape(F, -1))
        lam1 = (d_u_all - a1[:, None]).abs().amax(dim=1) / beta     # (F,)
        lam2 = (d_p.reshape(F, -1) - a2[:, None]).abs().amax(dim=1) / beta
        lam1 = torch.maximum(lam1, 1e-12 * a1.abs())
        lam2 = torch.maximum(lam2, 1e-12 * a2.abs())
        self.a1, self.a2, self.lam1, self.lam2 = a1, a2, lam1, lam2

        V = torch.stack(
            [
                (d_ux - a1.view(F, 1, 1)) / lam1.view(F, 1, 1),
                (d_uz - a1.view(F, 1, 1)) / lam1.view(F, 1, 1),
                (d_p - a2.view(F, 1, 1)) / lam2.view(F, 1, 1),
            ],
            dim=1,
        ).to(self.cdtype).to(self.device)                           # (F,3,Nz,Nx)
        self.V = V
        self.B = 1.0 - V

        M_inv, M_fwd = assemble_symbols_batched(
            nz_tot, nx_tot, self.dx, a1, a2, lam1, lam2,
            cdtype=self.cdtype, device=self.device,
            compute_forward=keep_forward_symbol,
        )
        self.M_inv, self.M_fwd = M_inv, M_fwd                       # (F,3,3,Nz,Nx)

        sq1 = lam1.sqrt()
        sq2 = lam2.sqrt()
        self.sqrt_lam = torch.stack([sq1, sq1, sq2], dim=1) \
            .to(self.cdtype).to(self.device).view(F, 3, 1, 1)
        self.rho_c2 = (rhop[None] * c2).to(self.cdtype).to(self.device)
        self.rho_sx = rho_sx.to(self.device)
        self.rho_sz = rho_sz.to(self.device)

    # ---------------------------------------------------------------- #
    def _apply_symbol(self, M, x):
        """Unrolled (see CBSSolver2D._apply_symbol): M (F,3,3,H,W), x (F,3,H,W)."""
        X = torch.fft.fftn(x, dim=(-2, -1))
        x0 = X[:, 0]
        x1 = X[:, 1]
        x2 = X[:, 2]
        Y = torch.stack(
            [
                M[:, 0, 0] * x0 + M[:, 0, 1] * x1 + M[:, 0, 2] * x2,
                M[:, 1, 0] * x0 + M[:, 1, 1] * x1 + M[:, 1, 2] * x2,
                M[:, 2, 0] * x0 + M[:, 2, 1] * x1 + M[:, 2, 2] * x2,
            ],
            dim=1,
        )
        return torch.fft.ifftn(Y, dim=(-2, -1))

    def build_rhs(self, sp) -> torch.Tensor:
        """sp: (nz_int, nx_int) complex pressure source, shared by all
        frequencies (unit transfer-function source). Returns y (F,3,Nz,Nx)."""
        F = self.omegas.numel()
        y = torch.zeros((F, 3) + self.shape, dtype=self.cdtype, device=self.device)
        iz, ix = self.interior
        sp = torch.as_tensor(sp, dtype=self.cdtype, device=self.device)
        y[:, 2, iz, ix] = sp[None] / self.rho_c2[:, iz, ix]
        return y / self.sqrt_lam

    @torch.no_grad()
    def solve(self, sp, tol=2e-4, max_iter=60000, check_every=50):
        """Masked + compacting fixed-point iteration.

        Frequencies that converge are finalized immediately (true residual,
        crop, store) and REMOVED from the working tensors, so the cost of a
        chunk decays as bins finish instead of running everyone to the
        slowest bin. Returns (fields, iters, resid): fields
        (F,3,nz_int,nx_int) unscaled [ux,uz,p], iters (F,), resid (F,).
        """
        F = self.omegas.numel()
        iz, ix = self.interior
        nz_i = iz.stop - iz.start
        nx_i = ix.stop - ix.start

        out = torch.empty((F, 3, nz_i, nx_i), dtype=self.cdtype, device=self.device)
        iters = torch.zeros(F, dtype=torch.long)
        resid = torch.zeros(F, dtype=torch.float64)

        y = self.build_rhs(sp)
        x = torch.zeros_like(y)
        B, V, Mi, Mf, sqlam = self.B, self.V, self.M_inv, self.M_fwd, self.sqrt_lam
        idx = torch.arange(F, device=self.device)
        nu = self.nu

        def finalize(sel_done):
            """Residual + crop + store for finished bins."""
            xd, yd = x[sel_done], y[sel_done]
            Ax = self._apply_symbol(Mf[sel_done], xd) - xd + V[sel_done] * xd
            r = yd - Ax
            rn = (torch.linalg.vector_norm(r, dim=(1, 2, 3))
                  / torch.linalg.vector_norm(yd, dim=(1, 2, 3)))
            gid = idx[sel_done]
            resid[gid.cpu()] = rn.to(torch.float64).cpu()
            out[gid] = (xd / sqlam[sel_done])[..., iz, ix]

        for it in range(1, max_iter + 1):
            z = self._apply_symbol(Mi, B * x + y)
            upd = nu * (B * (z - x))
            x += upd
            if it % check_every == 0:
                un = torch.linalg.vector_norm(upd, dim=(1, 2, 3))
                xn = torch.linalg.vector_norm(x, dim=(1, 2, 3)).clamp_min(1e-30)
                rel = un / xn
                if torch.isnan(rel).any():
                    raise FloatingPointError("iteration diverged (NaN)")
                done = rel < tol
                if bool(done.any()):
                    iters[idx[done].cpu()] = it
                    finalize(done)
                    keep = ~done
                    if not bool(keep.any()):
                        return out, iters, resid
                    x, y, B, V, sqlam = x[keep], y[keep], B[keep], V[keep], sqlam[keep]
                    Mi, Mf, idx = Mi[keep], Mf[keep], idx[keep]

        raise RuntimeError(f"{idx.numel()} frequencies not converged "
                           f"after {max_iter} iterations")