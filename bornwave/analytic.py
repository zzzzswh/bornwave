"""Analytic references for validation.

2D homogeneous medium, unit-integral monopole pressure source sp = delta(x):
the first-order system reduces (Appendix A) to

    (nabla^2 + k^2) p = -(i w / c^2) sp

For the paper's +i*omega convention (time factor exp(+i w t), which matches
the numpy/torch FFT transfer-function convention) the outgoing free-space
solution is

    p(r) = + (w / (4 c^2)) * H0^(2)(k r)

and for the exp(-i w t) convention it is the complex conjugate,

    p(r) = + (w / (4 c^2)) * H0^(1)(k r).
"""
from __future__ import annotations

import numpy as np
from scipy.special import hankel1, hankel2

__all__ = ["pressure_point_source_2d"]


def pressure_point_source_2d(r, omega, c0, convention: str = "+iwt"):
    """Pressure field of a unit continuous delta pressure source sp.

    r : array of distances [m] (r=0 excluded by the caller)
    convention : '+iwt' (paper / torch-FFT, H0^(2)) or '-iwt' (H0^(1))
    """
    r = np.asarray(r, dtype=np.float64)
    k = omega / c0
    amp = omega / (4.0 * c0**2)
    if convention == "+iwt":
        return amp * hankel2(0, k * r)
    elif convention == "-iwt":
        return amp * hankel1(0, k * r)
    raise ValueError("convention must be '+iwt' or '-iwt'")


def fluid_reflection_coefficient(theta_inc, c1, rho1, c2, rho2):
    """Plane-wave pressure reflection coefficient at a fluid-fluid interface
    (acoustic Zoeppritz), incidence angle theta_inc [rad] in medium 1.
    Complex-safe past the critical angle."""
    theta_inc = np.asarray(theta_inc, dtype=np.float64)
    s1 = np.sin(theta_inc)
    cos1 = np.cos(theta_inc)
    cos2 = np.sqrt(1.0 - (c2 / c1 * s1) ** 2 + 0j)
    Z1 = rho1 * c1 / cos1
    Z2 = rho2 * c2 / cos2
    return (Z2 - Z1) / (Z2 + Z1)