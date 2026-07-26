"""Stage 0: geometry & grid utilities.

- FFT-friendly padded sizes (small-prime products, cuFFT/pocketfft friendly)
- Polynomial-ramp sponge absorbing layer gamma, evaluable at collocated
  (integer) and staggered (integer + 1/2) coordinates
- Edge-replicate padding of material maps
- Linear interpolation of rho0 onto forward-staggered half-grid points
  (paper eq. (39))
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "next_fast_len",
    "sponge_profile_1d",
    "pad_replicate_2d",
    "stagger_avg",
]


def next_fast_len(n: int, primes=(2, 3, 5, 7)) -> int:
    """Smallest m >= n whose prime factors are all in `primes`."""
    if n <= 1:
        return 1

    def smooth(m: int) -> bool:
        for p in primes:
            while m % p == 0:
                m //= p
        return m == 1

    m = n
    while not smooth(m):
        m += 1
    return m


def sponge_profile_1d(
    n_total: int,
    n_left: int,
    n_right: int,
    gamma_max: float,
    power: float = 3.0,
    offset: float = 0.0,
    dtype: torch.dtype = torch.float32,
    device=None,
) -> torch.Tensor:
    """1D absorbing ramp gamma(i) evaluated at coordinates i + offset.

    gamma rises polynomially from 0 at the inner edge of the layer to
    gamma_max at the domain boundary. `offset=0.5` evaluates the profile at
    forward-staggered points (this is the analytic S+ gamma of the paper —
    gamma is generated algorithmically, so we can evaluate it anywhere).
    """
    i = torch.arange(n_total, dtype=dtype, device=device) + offset
    g = torch.zeros_like(i)
    if n_left > 0:
        d = torch.clamp((n_left - i) / n_left, min=0.0, max=1.0)
        g = torch.maximum(g, d)
    if n_right > 0:
        start = n_total - 1 - n_right  # inner edge of the right layer
        d = torch.clamp((i - start) / n_right, min=0.0, max=1.0)
        g = torch.maximum(g, d)
    return gamma_max * g**power


def pad_replicate_2d(a: torch.Tensor, pad_tblr: tuple[int, int, int, int]) -> torch.Tensor:
    """Edge-replicate pad a real 2D map. pad = (top, bottom, left, right)."""
    t, b, l, r = pad_tblr
    return F.pad(a[None, None], (l, r, t, b), mode="replicate")[0, 0]


def stagger_avg(a: torch.Tensor, dim: int) -> torch.Tensor:
    """Linear interpolation onto forward-staggered (+Delta/2) points, edge-clamped.

    Implements paper eq. (39): S+ rho0 ~ 0.5*(rho0(x) + rho0(x + Delta)).
    """
    n = a.shape[dim]
    a_next = torch.cat(
        [a.narrow(dim, 1, n - 1), a.narrow(dim, n - 1, 1)], dim=dim
    )
    return 0.5 * (a + a_next)