"""High-performance CBS engine: joint (frequency x shot) batching + CUDA Graphs.

Extends the validated CBSFreqBatch2D iteration to a working state of shape

    x : (F, B, 3, Nz, Nx)      F = frequencies in the chunk, B = shots

with all frequency-dependent operator tensors broadcast over the shot axis
(shots are free in the frequency domain: same symbols, same V).

Performance model: the per-iteration work is ~15 small elementwise kernels
plus 2 batched FFTs. On a GPU with grids of this size the wall time is
dominated by KERNEL LAUNCH LATENCY (thousands of iterations x ~15 launches
x ~5-20 us), not by FLOPs. CUDA Graphs capture `check_every - 1` iterations
once and replay them as a single launch, which removes that overhead
entirely. The graph is re-captured whenever converged frequencies are
compacted out (shape change); capture cost is milliseconds and is amortized
over hundreds of replayed iterations.

Convergence bookkeeping is identical to CBSFreqBatch2D: relative increment of
the LAST single iteration, checked every `check_every` iterations (the graph
replays check_every - 1 iterations, then ONE eager iteration produces the
increment used for the check, so the stopping criterion is unchanged).
A frequency is finalized when ALL its shots pass; since iteration count is
governed by frequency, shots at the same frequency converge together and the
compaction waste stays negligible.
"""
from __future__ import annotations

import warnings

import torch

from .grid import next_fast_len, pad_replicate_2d, sponge_profile_1d, stagger_avg
from .operators import assemble_symbols_batched

__all__ = ["CBSFreqShotBatch2D"]


def _median_complex_rows(z: torch.Tensor) -> torch.Tensor:
    """(F, N) complex -> (F,) complex, component-wise median per row."""
    return torch.complex(z.real.median(dim=1).values, z.imag.median(dim=1).values)


class CBSFreqShotBatch2D:
    """CBS iteration over a batch of frequencies x a batch of shots on a
    shared (c0, rho0, alpha|Q) model.

    Parameters are those of CBSSolver2D / CBSFreqBatch2D, plus:

    Q : scalar or (nz, nx), optional
        Constant-Q attenuation. Equivalent to the frequency-DEPENDENT
        alpha(w) = w / (2 c0 Q); implemented exactly as the
        frequency-independent complex modulus c^2 = c0^2 / (1 - i/Q).
        Mutually exclusive with `alpha`.
    """

    def __init__(
        self,
        c0,
        rho0,
        omegas,
        dx: float,
        alpha=0.0,
        Q=None,
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
        w = omegas.view(F, 1, 1)                                   # (F,1,1)

        # ---- shared geometry (identical to CBSFreqBatch2D) ----------------
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

        # ---- per-frequency material diagonals ------------------------------
        if Q is not None:
            Qm = torch.as_tensor(Q, dtype=torch.float64).expand(nz_int, nx_int)
            Qp = pad_replicate_2d(Qm.contiguous(), (pt, pb, pl, pr))
            # constant-Q: alpha(w) = w/(2 c0 Q)  =>  c^2 = c0^2 / (1 - i/Q)
            c2 = c0p[None] ** 2 / (1.0 - 1j / Qp[None])            # (1,Nz,Nx) bc
            c2 = c2.expand(F, nz_tot, nx_tot)
        else:
            c2 = c0p[None] ** 2 / (1.0 - 2.0j * alp[None] * c0p[None] / w)
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
        self.Bop = (1.0 - V).unsqueeze(1)                           # (F,1,3,Nz,Nx)

        M_inv, M_fwd = assemble_symbols_batched(
            nz_tot, nx_tot, self.dx, a1, a2, lam1, lam2,
            cdtype=self.cdtype, device=self.device,
            compute_forward=keep_forward_symbol,
        )
        self.M_inv, self.M_fwd = M_inv, M_fwd                       # (F,3,3,Nz,Nx)

        sq1 = lam1.sqrt()
        sq2 = lam2.sqrt()
        self.sqrt_lam = torch.stack([sq1, sq1, sq2], dim=1) \
            .to(self.cdtype).to(self.device).view(F, 1, 3, 1, 1)
        self.rho_c2 = (rhop[None] * c2).to(self.cdtype).to(self.device)  # (F,Nz,Nx)

    # ---------------------------------------------------------------- #
    @staticmethod
    def _apply_symbol(M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """IFFT( M(k) @ FFT(x) ): M (F,3,3,H,W), x (F,B,3,H,W).

        The 3x3 product is UNROLLED into elementwise multiply-adds (an einsum
        lowers to permute+bmm on CUDA and copies the symbol every iteration)."""
        X = torch.fft.fftn(x, dim=(-2, -1))
        x0 = X[:, :, 0]
        x1 = X[:, :, 1]
        x2 = X[:, :, 2]

        def m(i, j):
            return M[:, i, j].unsqueeze(1)                          # (F,1,H,W)

        Y = torch.stack(
            [
                m(0, 0) * x0 + m(0, 1) * x1 + m(0, 2) * x2,
                m(1, 0) * x0 + m(1, 1) * x1 + m(1, 2) * x2,
                m(2, 0) * x0 + m(2, 1) * x1 + m(2, 2) * x2,
            ],
            dim=2,
        )
        return torch.fft.ifftn(Y, dim=(-2, -1))

    def _iter_once(self, Mi, Bop, x, y):
        """One fixed-point step, in place on x. Returns upd (fresh tensor)."""
        z = self._apply_symbol(Mi, Bop * x + y)
        z.sub_(x).mul_(Bop).mul_(self.nu)   # z <- nu * B * (z - x) == upd
        x.add_(z)
        return z

    def _capture_graph(self, Mi, Bop, x, y, n_inner):
        """Warm up (2 real iterations) then capture n_inner iterations as one
        CUDA graph operating in place on the static tensors x/y/Bop/Mi.
        Returns (graph, n_warmup_iters_consumed)."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._iter_once(Mi, Bop, x, y)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_inner):
                self._iter_once(Mi, Bop, x, y)
        return g, 2

    # ---------------------------------------------------------------- #
    def build_rhs(self, sp) -> torch.Tensor:
        """sp: (B, nz_int, nx_int) complex pressure sources (one per shot),
        shared by all frequencies. Returns y of shape (F, B, 3, Nz, Nx)."""
        sp = torch.as_tensor(sp, dtype=self.cdtype, device=self.device)
        if sp.ndim == 2:
            sp = sp[None]
        F = self.omegas.numel()
        Bs = sp.shape[0]
        y = torch.zeros((F, Bs, 3) + self.shape, dtype=self.cdtype,
                        device=self.device)
        iz, ix = self.interior
        y[:, :, 2, iz, ix] = sp[None] / self.rho_c2[:, None, iz, ix]
        return y / self.sqrt_lam

    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def solve(self, sp, tol=2e-4, max_iter=60000, check_every=50,
              cuda_graph="auto"):
        """Masked + compacting fixed-point iteration over (F, B).

        cuda_graph : True | False | "auto"
            "auto" enables graph capture on CUDA devices. Any capture failure
            falls back to the eager loop with a warning (results identical).

        Returns (fields, iters, resid):
            fields (F, B, 3, nz_int, nx_int) unscaled [ux, uz, p]
            iters  (F,) long, resid (F, B) float64 true relative residuals.
        """
        if self.M_fwd is None:
            raise RuntimeError("engine built with keep_forward_symbol=False; "
                               "true residuals require the forward symbol")
        F = self.omegas.numel()
        iz, ix = self.interior
        nz_i = iz.stop - iz.start
        nx_i = ix.stop - ix.start

        y = self.build_rhs(sp)
        Bs = y.shape[1]
        out = torch.empty((F, Bs, 3, nz_i, nx_i), dtype=self.cdtype,
                          device=self.device)
        iters = torch.zeros(F, dtype=torch.long)
        resid = torch.zeros(F, Bs, dtype=torch.float64)

        x = torch.zeros_like(y)
        Bop, V, Mi, Mf, sqlam = self.Bop, self.V, self.M_inv, self.M_fwd, self.sqrt_lam
        idx = torch.arange(F, device=self.device)

        n_inner = check_every - 1
        use_graph = (cuda_graph is True) or (cuda_graph == "auto" and x.is_cuda)
        use_graph = bool(use_graph and x.is_cuda and n_inner >= 1)
        graph = None

        def finalize(sel_done):
            """True residual + crop + store for finished frequencies."""
            xd, yd = x[sel_done], y[sel_done]
            Ax = self._apply_symbol(Mf[sel_done], xd) - xd \
                + V[sel_done].unsqueeze(1) * xd
            r = yd - Ax
            rn = (torch.linalg.vector_norm(r, dim=(-3, -2, -1))
                  / torch.linalg.vector_norm(yd, dim=(-3, -2, -1)))
            gid = idx[sel_done]
            resid[gid.cpu()] = rn.to(torch.float64).cpu()
            out[gid] = (xd / sqlam[sel_done])[..., iz, ix]

        it = 0
        while it < max_iter:
            if use_graph:
                if graph is None:
                    try:
                        graph, warm = self._capture_graph(Mi, Bop, x, y, n_inner)
                        it += warm
                    except Exception as e:  # pragma: no cover - device specific
                        warnings.warn(
                            f"CUDA graph capture failed ({type(e).__name__}: {e}); "
                            "falling back to eager iteration.")
                        use_graph = False
                        continue
                graph.replay()
                it += n_inner
                upd = self._iter_once(Mi, Bop, x, y)   # eager last step -> upd
                it += 1
            else:
                for _ in range(check_every):
                    upd = self._iter_once(Mi, Bop, x, y)
                it += check_every

            # ---- convergence check (per frequency x shot) ----------------
            un = torch.linalg.vector_norm(upd, dim=(-3, -2, -1))    # (F',B)
            xn = torch.linalg.vector_norm(x, dim=(-3, -2, -1)).clamp_min(1e-30)
            rel = un / xn
            if torch.isnan(rel).any():
                raise FloatingPointError("iteration diverged (NaN)")
            done = (rel < tol).all(dim=1)                           # (F',)
            if bool(done.any()):
                iters[idx[done].cpu()] = it
                finalize(done)
                keep = ~done
                if not bool(keep.any()):
                    return out, iters, resid
                x, y = x[keep], y[keep]
                Bop, V, sqlam = Bop[keep], V[keep], sqlam[keep]
                Mi, Mf, idx = Mi[keep], Mf[keep], idx[keep]
                graph = None    # shapes changed -> re-capture on next round

        raise RuntimeError(f"{idx.numel()} frequencies not converged "
                           f"after {max_iter} iterations "
                           f"(worst rel increment {rel.max().item():.2e})")
