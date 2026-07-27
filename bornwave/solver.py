"""CBS-type iterative Helmholtz solver for the first-order acoustic system
with heterogeneous sound speed, density and absorption (2D, single frequency).

Implements Stanziola et al. 2025 (arXiv:2507.16087): the universal
split-preconditioner of Vettenburg & Vellekoop applied to

    [ (S+rho0)(S+gamma + i w)     S+ grad  ] [u]   [ s_u_hat ]
    [ div S-                (gamma+i w)/(rho0 c^2)] [p] = [ s_p_hat ]

with the fixed-point iteration (paper eq. (13) / Algorithm 1)

    x <- x + nu * B * ( (L+I)^{-1} (B x + y) - x ),     B = I - V.

Time convention: the first-order system is written with +i*omega, which
matches the exp(+i w t) / numpy-torch FFT transfer-function convention
(see README). Field layout: x has shape (*batch, 3, Nz, Nx) with channel
order [u_x, u_z, p]; u_x lives at (x + dx/2), u_z at (z + dx/2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .grid import next_fast_len, pad_replicate_2d, sponge_profile_1d, stagger_avg
from .operators import assemble_symbols, complex_median

__all__ = ["CBSSolver2D", "SolveResult", "point_source_2d", "alpha_from_Q"]


def alpha_from_Q(omega: float, c0, Q) -> torch.Tensor:
    """Seismic quality factor Q -> absorption coefficient alpha [Np/m]."""
    return omega / (2.0 * torch.as_tensor(c0, dtype=torch.float64) * Q)


def point_source_2d(nz, nx, iz, ix, dx, amplitude=1.0, dtype=torch.complex64, device=None):
    """Monopole source field approximating amplitude * delta(x - x_s).

    A value q on a single node of a spectral grid represents a band-limited
    sinc of integral q*dx^2, so the node is set to amplitude/dx^2 to model a
    unit-integral continuous delta.
    """
    s = torch.zeros((nz, nx), dtype=dtype, device=device)
    s[iz, ix] = amplitude / dx**2
    return s


@dataclass
class SolveResult:
    p: torch.Tensor          # pressure, interior grid (collocated)
    ux: torch.Tensor         # x-velocity, interior grid (staggered +dx/2 in x)
    uz: torch.Tensor         # z-velocity, interior grid (staggered +dx/2 in z)
    iterations: int
    rel_increment: float     # last ||x_{k+1}-x_k|| / ||x_k||
    rel_residual: float | None  # ||y - A x|| / ||y|| in scaled variables
    converged: bool
    history: list = field(default_factory=list)  # (iteration, rel_increment)
    x_scaled: torch.Tensor | None = None  # full scaled state (for warm starts)


class CBSSolver2D:
    """Single-frequency 2D solver. Batch-ready: sources may carry leading
    batch dimensions; the operator tensors are shared across the batch
    (this is the 'shots are almost free' property of the frequency domain).

    Parameters
    ----------
    c0 : (nz, nx) array-like, sound speed [m/s]
    rho0 : scalar or (nz, nx), density [kg/m^3]
    omega : angular frequency [rad/s]
    dx : grid spacing [m] (uniform)
    alpha : scalar or (nz, nx), absorption [Np/m] at omega
    abs_points : sponge thickness in grid points (per side)
    gamma_max_factor : gamma_max = factor * omega
    gamma_power : polynomial ramp exponent
    beta : ||V|| <= beta < 1 (paper eq. (22))
    nu : relaxation 0 < nu < 1 (paper eq. (13))
    """

    def __init__(
        self,
        c0,
        rho0,
        omega: float,
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
        self.rdtype = torch.float32 if dtype == torch.complex64 else torch.float64
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.omega = float(omega)
        self.dx = float(dx)
        self.nu = float(nu)
        self.beta = float(beta)

        # ---- Stage 0: geometry -------------------------------------------
        c0 = torch.as_tensor(c0, dtype=torch.float64)
        if c0.ndim != 2:
            raise ValueError("c0 must be a 2D (nz, nx) map; broadcast scalars yourself")
        nz_int, nx_int = c0.shape
        rho0 = torch.as_tensor(rho0, dtype=torch.float64).expand(nz_int, nx_int)
        alpha = torch.as_tensor(alpha, dtype=torch.float64).expand(nz_int, nx_int)

        pt = pl = int(abs_points)
        nz_pad = nz_int + 2 * pt
        nx_pad = nx_int + 2 * pl
        if fft_friendly:
            nz_tot, nx_tot = next_fast_len(nz_pad), next_fast_len(nx_pad)
        else:
            nz_tot, nx_tot = nz_pad, nx_pad
        pb = pt + (nz_tot - nz_pad)  # extra rows go to the bottom sponge
        pr = pl + (nx_tot - nx_pad)  # extra cols go to the right sponge
        self.interior = (slice(pt, pt + nz_int), slice(pl, pl + nx_int))
        self.shape_interior = (nz_int, nx_int)
        self.shape = (nz_tot, nx_tot)

        c0p = pad_replicate_2d(c0, (pt, pb, pl, pr))
        rhop = pad_replicate_2d(rho0.contiguous(), (pt, pb, pl, pr))
        alp = pad_replicate_2d(alpha.contiguous(), (pt, pb, pl, pr))

        # ---- Stage 0/1: sponge gamma (collocated + staggered) ------------
        gmax = gamma_max_factor * self.omega
        gp = dict(gamma_max=gmax, power=gamma_power, dtype=torch.float64)
        gx_c = sponge_profile_1d(nx_tot, pl, pr, offset=0.0, **gp)
        gx_s = sponge_profile_1d(nx_tot, pl, pr, offset=0.5, **gp)
        gz_c = sponge_profile_1d(nz_tot, pt, pb, offset=0.0, **gp)
        gz_s = sponge_profile_1d(nz_tot, pt, pb, offset=0.5, **gp)
        gam_p = gz_c[:, None] + gx_c[None, :]
        gam_ux = gz_c[:, None] + gx_s[None, :]
        gam_uz = gz_s[:, None] + gx_c[None, :]

        # ---- Stage 1: material preprocessing -----------------------------
        w = self.omega
        c2 = c0p**2 / (1.0 - 2.0j * alp * c0p / w)          # complex c^2, eq. (4)
        rho_sx = stagger_avg(rhop, dim=1)                    # S+_x rho0, eq. (39)
        rho_sz = stagger_avg(rhop, dim=0)                    # S+_z rho0

        d_ux = rho_sx * (1j * w + gam_ux)                    # velocity diagonals
        d_uz = rho_sz * (1j * w + gam_uz)
        d_p = (gam_p + 1j * w) / (rhop * c2)                 # pressure diagonal

        # ---- Stage 2: shifts, scaling, V/B, Fourier symbols --------------
        d_u_all = torch.cat([d_ux.flatten(), d_uz.flatten()])
        a1 = complex_median(d_u_all)
        a2 = complex_median(d_p.flatten())
        lam1 = ( (d_u_all - a1).abs().max().item() ) / beta
        lam2 = ( (d_p.flatten() - a2).abs().max().item() ) / beta
        # Degenerate (perfectly homogeneous block) -> tiny positive scale
        lam1 = max(lam1, 1e-12 * abs(a1))
        lam2 = max(lam2, 1e-12 * abs(a2))
        self.a1, self.a2, self.lam1, self.lam2 = a1, a2, lam1, lam2

        V = torch.stack(
            [(d_ux - a1) / lam1, (d_uz - a1) / lam1, (d_p - a2) / lam2]
        ).to(self.cdtype).to(self.device)                    # (3, Nz, Nx)
        self.V = V
        self.B = (1.0 - V)

        M_inv, M_fwd = assemble_symbols(
            nz_tot, nx_tot, self.dx, a1, a2, lam1, lam2,
            cdtype=self.cdtype, device=self.device,
            compute_forward=keep_forward_symbol,
        )
        self.M_inv, self.M_fwd = M_inv, M_fwd

        # source / output scaling
        self.sqrt_lam = torch.tensor(
            [lam1**0.5, lam1**0.5, lam2**0.5], dtype=self.cdtype, device=self.device
        ).view(3, 1, 1)
        self.rho_sx = rho_sx.to(self.rdtype).to(self.device)
        self.rho_sz = rho_sz.to(self.rdtype).to(self.device)
        self.rho_c2 = (rhop * c2).to(self.cdtype).to(self.device)

    # ------------------------------------------------------------------ #
    def _apply_symbol(self, M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """IFFT( M(k) @ FFT(x) ), batched over leading dims of x.

        The 3x3 product is UNROLLED into elementwise multiply-adds: on CUDA
        the equivalent einsum lowers to permute+bmm and copies the full
        symbol tensor every iteration, which dominates the runtime."""
        X = torch.fft.fftn(x, dim=(-2, -1))
        x0 = X[..., 0, :, :]
        x1 = X[..., 1, :, :]
        x2 = X[..., 2, :, :]
        Y = torch.stack(
            [
                M[0, 0] * x0 + M[0, 1] * x1 + M[0, 2] * x2,
                M[1, 0] * x0 + M[1, 1] * x1 + M[1, 2] * x2,
                M[2, 0] * x0 + M[2, 1] * x1 + M[2, 2] * x2,
            ],
            dim=-3,
        )
        return torch.fft.ifftn(Y, dim=(-2, -1))

    def apply_A(self, x: torch.Tensor) -> torch.Tensor:
        """Scaled forward operator A = L + V (for true residuals)."""
        if self.M_fwd is None:
            raise RuntimeError("solver built with keep_forward_symbol=False")
        Lx_plus_x = self._apply_symbol(self.M_fwd, x)
        return Lx_plus_x - x + self.V * x

    # ------------------------------------------------------------------ #
    def build_rhs(self, sp=None, su=None) -> torch.Tensor:
        """Assemble scaled RHS y = C^{-1/2} [rho_s * s_u, s_p/(rho0 c^2)].

        sp: (*batch, nz_int, nx_int) complex pressure source (interior grid)
        su: optional tuple (su_x, su_z), same layout, on staggered points
        """
        if sp is None and su is None:
            raise ValueError("provide sp and/or su")
        ref = sp if sp is not None else su[0]
        batch = tuple(ref.shape[:-2])
        y = torch.zeros(batch + (3,) + self.shape, dtype=self.cdtype, device=self.device)
        iz, ix = self.interior
        if su is not None:
            su_x = torch.as_tensor(su[0], dtype=self.cdtype, device=self.device)
            su_z = torch.as_tensor(su[1], dtype=self.cdtype, device=self.device)
            y[..., 0, iz, ix] = self.rho_sx[iz, ix] * su_x
            y[..., 1, iz, ix] = self.rho_sz[iz, ix] * su_z
        if sp is not None:
            sp = torch.as_tensor(sp, dtype=self.cdtype, device=self.device)
            y[..., 2, iz, ix] = sp / self.rho_c2[iz, ix]
        return y / self.sqrt_lam

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def solve(
        self,
        sp=None,
        su=None,
        y=None,
        x0=None,
        tol: float = 1e-6,
        max_iter: int = 20000,
        check_every: int = 25,
        compute_residual: bool = True,
        crop: bool = True,
        verbose: bool = False,
    ) -> SolveResult:
        """Run the fixed-point iteration until the relative increment
        ||x_{k+1} - x_k|| / ||x_k|| drops below `tol`."""
        if y is None:
            y = self.build_rhs(sp=sp, su=su)
        x = torch.zeros_like(y) if x0 is None else x0.clone()
        B, nu, M_inv = self.B, self.nu, self.M_inv

        history: list[tuple[int, float]] = []
        rel = float("inf")
        it = 0
        converged = False
        for it in range(1, max_iter + 1):
            z = self._apply_symbol(M_inv, B * x + y)
            upd = nu * (B * (z - x))
            x += upd
            if it % check_every == 0:
                rel = (upd.norm() / x.norm().clamp_min(1e-30)).item()
                history.append((it, rel))
                if verbose:
                    print(f"iter {it:6d}  rel_increment {rel:.3e}")
                if rel != rel:  # NaN guard
                    raise FloatingPointError("iteration diverged (NaN)")
                if rel < tol:
                    converged = True
                    break

        rel_res = None
        if compute_residual and self.M_fwd is not None:
            r = y - self.apply_A(x)
            rel_res = (r.norm() / y.norm()).item()

        # [u, p]^T = C^{-1/2} x
        fields = x / self.sqrt_lam
        if crop:
            iz, ix = self.interior
            ux = fields[..., 0, iz, ix]
            uz = fields[..., 1, iz, ix]
            p = fields[..., 2, iz, ix]
        else:
            ux, uz, p = fields[..., 0, :, :], fields[..., 1, :, :], fields[..., 2, :, :]

        return SolveResult(
            p=p, ux=ux, uz=uz,
            iterations=it, rel_increment=rel, rel_residual=rel_res,
            converged=converged, history=history, x_scaled=x,
        )