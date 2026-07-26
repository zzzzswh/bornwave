"""Stage 2: operator assembly in the Fourier domain (2D, staggered grid).

Field ordering everywhere: [u_x, u_z, p]  (u_x staggered +dx/2 in x,
u_z staggered +dx/2 in z, p collocated).

The 2D staggered symbol of (L+I)^{-1} is the 2D reduction of paper
Appendix B eq. (44)/(45), with

    mu  = k^2 + (a1 + lam1)(a2 + lam2)
    S_e = exp(+i k_e dx/2)          (forward-stagger phase, e in {x,z})

    M_inv[u_e, u_f] = lam1 (mu d_ef - k_e k_f e^{i(k_e-k_f)dx/2}) / ((a1+lam1) mu)
    M_inv[u_e, p ]  = -i k_e sqrt(lam1 lam2) S_e        / mu
    M_inv[p , u_e]  = -i k_e sqrt(lam1 lam2) conj(S_e)  / mu
    M_inv[p , p ]   =  lam2 (a1 + lam1) / mu

The forward symbol of (L+I) (used for true residuals and for the unit test
M_fwd @ M_inv = I) is

    M_fwd[u_e, u_e] = (a1 + lam1)/lam1
    M_fwd[p , p ]   = (a2 + lam2)/lam2
    M_fwd[u_e, p ]  = +i k_e S_e        / sqrt(lam1 lam2)
    M_fwd[p , u_e]  = +i k_e conj(S_e)  / sqrt(lam1 lam2)
"""
from __future__ import annotations

import math

import torch

__all__ = ["k_vectors", "assemble_symbols", "complex_median"]


def k_vectors(nz, nx, dx, dtype=torch.float64, device=None):
    """Angular wavenumber meshgrids (KZ, KX), each of shape (nz, nx)."""
    kx = 2.0 * math.pi * torch.fft.fftfreq(nx, d=dx, dtype=dtype, device=device)
    kz = 2.0 * math.pi * torch.fft.fftfreq(nz, d=dx, dtype=dtype, device=device)
    KZ, KX = torch.meshgrid(kz, kx, indexing="ij")
    return KZ, KX


def complex_median(z: torch.Tensor) -> complex:
    """Cheap approximation of the minimum-bounding-circle centre:
    component-wise median. Adequate per the paper's Algorithm 1."""
    return complex(z.real.median().item(), z.imag.median().item())


def assemble_symbols(
    nz: int,
    nx: int,
    dx: float,
    a1: complex,
    a2: complex,
    lam1: float,
    lam2: float,
    cdtype: torch.dtype = torch.complex64,
    device=None,
    compute_forward: bool = True,
):
    """Assemble (L+I)^{-1} (and optionally (L+I)) Fourier symbols.

    Returns (M_inv, M_fwd_or_None), each of shape (3, 3, nz, nx), complex.
    Assembly is done in complex128 and cast down at the end.
    """
    KZ, KX = k_vectors(nz, nx, dx, dtype=torch.float64, device=device)
    KX = KX.to(torch.complex128)
    KZ = KZ.to(torch.complex128)

    a1 = complex(a1)
    a2 = complex(a2)
    au = a1 + lam1  # velocity-block shift + scale
    ap = a2 + lam2  # pressure-block shift + scale
    sll = math.sqrt(lam1 * lam2)

    Sx = torch.exp(0.5j * KX * dx)  # forward-stagger phase e^{+i kx dx/2}
    Sz = torch.exp(0.5j * KZ * dx)

    k2 = KX * KX + KZ * KZ
    mu = k2 + au * ap

    Mi = torch.empty((3, 3, nz, nx), dtype=torch.complex128, device=device)
    # velocity-velocity block: lam1/(au*mu) * S+(mu I - K)S-
    Mi[0, 0] = lam1 * (mu - KX * KX) / (au * mu)
    Mi[1, 1] = lam1 * (mu - KZ * KZ) / (au * mu)
    Mi[0, 1] = -lam1 * KX * KZ * (Sx * Sz.conj()) / (au * mu)
    Mi[1, 0] = -lam1 * KX * KZ * (Sz * Sx.conj()) / (au * mu)
    # velocity-pressure couplings
    Mi[0, 2] = -1j * KX * sll * Sx / mu
    Mi[1, 2] = -1j * KZ * sll * Sz / mu
    Mi[2, 0] = -1j * KX * sll * Sx.conj() / mu
    Mi[2, 1] = -1j * KZ * sll * Sz.conj() / mu
    # pressure-pressure
    Mi[2, 2] = lam2 * au / mu

    Mf = None
    if compute_forward:
        Mf = torch.zeros((3, 3, nz, nx), dtype=torch.complex128, device=device)
        Mf[0, 0] = au / lam1
        Mf[1, 1] = au / lam1
        Mf[2, 2] = ap / lam2
        Mf[0, 2] = 1j * KX * Sx / sll
        Mf[1, 2] = 1j * KZ * Sz / sll
        Mf[2, 0] = 1j * KX * Sx.conj() / sll
        Mf[2, 1] = 1j * KZ * Sz.conj() / sll
        Mf = Mf.to(cdtype)

    return Mi.to(cdtype), Mf