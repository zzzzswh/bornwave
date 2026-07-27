"""time_slices_from_band must equal np.fft.irfft sampled at the same
indices, to machine precision, for even/odd nt, with/without the Nyquist
bin, and for multi-dimensional trailing axes."""
import importlib.util
import pathlib

import numpy as np

# import the module by file path: this test must run without torch installed
_p = pathlib.Path(__file__).resolve().parents[1] / "bornwave" / "timesynth.py"
_spec = importlib.util.spec_from_file_location("timesynth", _p)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
time_slices_from_band = _mod.time_slices_from_band


def _check(nt, kidx, shape_tail=(), seed=0):
    rng = np.random.default_rng(seed)
    kidx = np.asarray(kidx, dtype=np.int64)
    Xd = np.zeros((nt // 2 + 1,) + shape_tail, dtype=np.complex128)
    vals = rng.standard_normal((kidx.size,) + shape_tail) \
        + 1j * rng.standard_normal((kidx.size,) + shape_tail)
    if nt % 2 == 0:
        vals[kidx == nt // 2] = vals[kidx == nt // 2].real  # Nyquist is real
    Xd[kidx] = vals
    ref = np.fft.irfft(Xd, n=nt, axis=0)
    samples = rng.choice(nt, size=min(nt, 17), replace=False)
    got = time_slices_from_band(Xd[kidx], kidx, nt, samples)
    err = np.max(np.abs(got - ref[samples]))
    scale = max(np.max(np.abs(ref)), 1e-30)
    assert err / scale < 1e-12, f"nt={nt} kidx={kidx[:5]}... err={err:.3e}"


def test_even_nt_interior_band():
    _check(240, np.arange(3, 60))


def test_even_nt_with_nyquist():
    _check(64, np.array([1, 5, 17, 32]))          # 32 == nt//2


def test_odd_nt():
    _check(241, np.arange(2, 50, 3), seed=1)


def test_multidim_tail():
    _check(128, np.arange(4, 40), shape_tail=(3, 5), seed=2)


def test_rejects_dc_bin():
    try:
        time_slices_from_band(np.zeros((1,), np.complex128), [0], 64, [0])
    except ValueError:
        return
    raise AssertionError("k=0 must be rejected")


if __name__ == "__main__":
    test_even_nt_interior_band()
    test_even_nt_with_nyquist()
    test_odd_nt()
    test_multidim_tail()
    test_rejects_dc_bin()
    print("time-slice synthesis == irfft to machine precision: all passed")
