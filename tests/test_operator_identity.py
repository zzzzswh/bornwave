"""M_fwd(k) @ M_inv(k) must equal I at every k point.

This directly validates the 2D staggered reduction of Appendix B eq. (44):
any sign/phase transcription error shows up here immediately.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from bornwave.operators import assemble_symbols


def test_symbol_inverse():
    torch.manual_seed(0)
    nz, nx, dx = 48, 60, 3.0
    # deliberately awkward complex shifts / scales
    a1 = 1.2e5 + 3.1e5j
    a2 = 2.4e-7 + 1.3e-7j
    lam1, lam2 = 3.3e5, 1.5e-7
    Mi, Mf = assemble_symbols(nz, nx, dx, a1, a2, lam1, lam2,
                              cdtype=torch.complex128)
    prod = torch.einsum("ikhw,kjhw->ijhw", Mf, Mi)
    I = torch.zeros_like(prod)
    I[0, 0] = I[1, 1] = I[2, 2] = 1.0
    err = (prod - I).abs().max().item()
    print(f"max |M_fwd @ M_inv - I| = {err:.3e}")
    assert err < 1e-12


def test_adjoint_symbol():
    """Adjoint machinery for the *staggered* formulation.

    Note: paper eq. (34) ('adjoint inverse = elementwise conjugate') holds
    for the collocated symbol, which is per-k symmetric. The staggered
    symbol is NOT per-k symmetric (S vs conj(S) phases), so the adjoint
    inverse is the per-k conjugate TRANSPOSE — equally free to compute.
    We verify (a) that it inverts the adjoint forward symbol, and (b) that
    the differential (off-diagonal) block of L is skew-Hermitian per k.
    """
    nz, nx, dx = 40, 40, 2.0
    a1 = 1.0e5 + 2.0e5j
    a2 = 1.0e-7 + 2.0e-7j
    lam1, lam2 = 2.2e5, 2.1e-7
    Mi, Mf = assemble_symbols(nz, nx, dx, a1, a2, lam1, lam2,
                              cdtype=torch.complex128)
    Mf_H = Mf.conj().transpose(0, 1)          # adjoint forward symbol
    Mi_H = Mi.conj().transpose(0, 1)          # adjoint inverse symbol
    prod = torch.einsum("ikhw,kjhw->ijhw", Mf_H, Mi_H)
    I = torch.zeros_like(prod)
    I[0, 0] = I[1, 1] = I[2, 2] = 1.0
    err = (prod - I).abs().max().item()
    print(f"max |M_fwd^H @ M_inv^H - I| = {err:.3e}")
    assert err < 1e-12

    # skew-Hermitian differential block: ell = Mf - diag part
    ell = Mf.clone()
    ell[0, 0] = ell[1, 1] = ell[2, 2] = 0.0
    skew = (ell.conj().transpose(0, 1) + ell).abs().max().item()
    print(f"max |ell^H + ell| = {skew:.3e}  (skew-Hermitian check)")
    assert skew < 1e-12


if __name__ == "__main__":
    test_symbol_inverse()
    test_adjoint_symbol()
    print("operator identity tests passed")