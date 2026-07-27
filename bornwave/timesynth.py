"""Torch-free spectral -> time-slice synthesis (pure numpy).

Given band-limited rfft spectra X_k on retained bins k (0 < k <= nt//2),
evaluate the inverse real FFT at SELECTED time samples only:

    x[n] = (1/nt) * Re( sum_k w_k * X_k * exp(+2*pi*i*k*n/nt) ),
    w_k = 2 for 0 < k < nt/2,   w_k = 1 for k == nt/2 (even nt).

This equals np.fft.irfft(X_dense, n=nt) sampled at `samples` when X_dense is
zero outside the retained band (the k=0 bin is excluded by band_indices and a
Ricker wavelet has no DC). One (S, K) @ (K, ...) product instead of a full
irfft over the whole wavefield: this is how wavefield snapshots/animations
are synthesized without ever materializing (nt, nz, nx).

Validated against np.fft.irfft to machine precision (tests/test_timesynth.py).
"""
from __future__ import annotations

import numpy as np

__all__ = ["time_slices_from_band"]


def time_slices_from_band(Xk: np.ndarray, kidx, nt: int, samples) -> np.ndarray:
    """Evaluate the real time-domain signal at `samples` from band spectra.

    Parameters
    ----------
    Xk : (K, ...) complex — spectra at the retained rfft bins, leading axis
        must correspond one-to-one with `kidx`.
    kidx : (K,) int — rfft bin indices, all in [1, nt//2].
    nt : int — full transform length (number of time samples).
    samples : (S,) int — time-sample indices n at which to evaluate.

    Returns
    -------
    (S, ...) real array; x[j] = irfft(X_dense, nt)[samples[j]].
    """
    Xk = np.asarray(Xk)
    kidx = np.asarray(kidx, dtype=np.int64).ravel()
    samples = np.asarray(samples, dtype=np.int64).ravel()
    if Xk.shape[0] != kidx.size:
        raise ValueError(f"Xk leading axis {Xk.shape[0]} != len(kidx) {kidx.size}")
    if kidx.size and (kidx.min() < 1 or kidx.max() > nt // 2):
        raise ValueError("kidx must lie in [1, nt//2] (k=0 / DC is excluded)")

    w = np.full(kidx.shape, 2.0)
    if nt % 2 == 0:
        w[kidx == nt // 2] = 1.0  # Nyquist bin appears once in the full FFT

    E = np.exp((2j * np.pi / nt) * np.outer(samples, kidx)) * w  # (S, K)
    E = E.astype(Xk.dtype, copy=False)
    return np.tensordot(E, Xk, axes=(1, 0)).real / nt
