"""Engine & API verification — RUN THIS FIRST on your machine.

(1) CBSFreqShotBatch2D (new engine, B=1) must reproduce the validated
    CBSFreqBatch2D fields.
(2) A batch of shots must equal the shots solved one by one.
(3) acoustic2d end-to-end on a homogeneous medium:
    - snapshots must equal the receiver records at the shared samples
      (both are synthesized from the same transfer functions),
    - the direct-wave lag between two receivers must match offset / c
      (kinematic check of the whole time convention chain).

All in complex128 with tight tolerances so agreement is at solver-accuracy
level, not luck. Runs on CPU in ~1-2 min; on GPU it also exercises the
CUDA-graph path (any capture failure falls back with a warning, results
must not change).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from bornwave.engine import CBSFreqShotBatch2D
from bornwave.multifreq import CBSFreqBatch2D
from bornwave.solver import point_source_2d
from bornwave import acoustic2d

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _two_layer():
    nz, nx, dx = 60, 72, 10.0
    c = torch.full((nz, nx), 2000.0, dtype=torch.float64)
    r = torch.full((nz, nx), 1500.0, dtype=torch.float64)
    c[36:], r[36:] = 2600.0, 2000.0
    return c, r, dx


def test_engine_matches_multifreq():
    c, r, dx = _two_layer()
    om = 2 * np.pi * np.array([8.0, 12.0, 16.0])
    sp = point_source_2d(60, 72, 8, 36, dx, dtype=torch.complex128)

    ref_fields, ref_it, ref_rs = CBSFreqBatch2D(
        c, r, om, dx, abs_points=30, dtype=torch.complex128, device=DEV
    ).solve(sp, tol=1e-7)

    new_fields, new_it, new_rs = CBSFreqShotBatch2D(
        c, r, om, dx, abs_points=30, dtype=torch.complex128, device=DEV
    ).solve(sp[None], tol=1e-7)

    err = ((new_fields[:, 0] - ref_fields).norm()
           / ref_fields.norm()).item()
    assert err < 1e-5, f"engine vs multifreq mismatch: {err:.2e}"
    assert float(new_rs.max()) < 1e-5
    print(f"  engine == multifreq: rel diff {err:.2e}, "
          f"iters {new_it.tolist()} vs {ref_it.tolist()}")


def test_batched_shots_equal_sequential():
    c, r, dx = _two_layer()
    om = 2 * np.pi * np.array([10.0, 14.0])
    s1 = point_source_2d(60, 72, 8, 24, dx, dtype=torch.complex128)
    s2 = point_source_2d(60, 72, 8, 48, dx, dtype=torch.complex128)

    eng = lambda: CBSFreqShotBatch2D(c, r, om, dx, abs_points=30,
                                     dtype=torch.complex128, device=DEV)
    batch, _, _ = eng().solve(torch.stack([s1, s2]), tol=1e-7)
    solo1, _, _ = eng().solve(s1[None], tol=1e-7)
    solo2, _, _ = eng().solve(s2[None], tol=1e-7)

    e1 = ((batch[:, 0] - solo1[:, 0]).norm() / solo1.norm()).item()
    e2 = ((batch[:, 1] - solo2[:, 0]).norm() / solo2.norm()).item()
    assert max(e1, e2) < 1e-5, f"shot batching mismatch: {e1:.2e}, {e2:.2e}"
    print(f"  batched == sequential shots: {e1:.2e}, {e2:.2e}")


def test_warm_start_engine():
    """x0 must not change the solution, only the iteration count."""
    c, r, dx = _two_layer()
    sp = point_source_2d(60, 72, 8, 36, dx, dtype=torch.complex128)
    e1 = CBSFreqShotBatch2D(c, r, 2 * np.pi * np.array([10.0]), dx,
                            abs_points=30, dtype=torch.complex128, device=DEV)
    _, _, _, st = e1.solve(sp[None], tol=1e-7, return_state=True)

    e2 = CBSFreqShotBatch2D(c, r, 2 * np.pi * np.array([11.0]), dx,
                            abs_points=30, dtype=torch.complex128, device=DEV)
    cold, itc, _ = e2.solve(sp[None], tol=1e-7)
    warm, itw, _ = e2.solve(sp[None], tol=1e-7, x0=st)
    err = ((warm - cold).norm() / cold.norm()).item()
    assert err < 1e-5, f"warm start changed the solution: {err:.2e}"
    assert int(itw[0]) <= int(itc[0])
    print(f"  warm start: same solution ({err:.1e}), "
          f"iters {int(itw[0])} <= {int(itc[0])}")


def test_api_warm_start_consistency():
    """acoustic2d(warm_start=True) == acoustic2d(warm_start=False)."""
    nz, nx, dh = 64, 80, 10.0
    vp = np.full((nz, nx), 2000.0); rho = np.full((nz, nx), 1000.0)
    kw = dict(sx=[40], sz=[10], rx=np.array([20, 60]), rz=10, nbc=30,
              tol=1e-6, freq_batch=8, dtype=torch.complex128,
              device=DEV, verbose=False)
    a = acoustic2d(vp, rho, dh, 2e-3, 192, 12.0, warm_start=False, **kw)
    b = acoustic2d(vp, rho, dh, 2e-3, 192, 12.0, warm_start=True, **kw)
    err = np.abs(a.seis_p - b.seis_p).max() / np.abs(a.seis_p).max()
    assert err < 1e-4, f"warm-start sweep diverged from cold: {err:.2e}"
    print(f"  api warm vs cold sweep: rel diff {err:.1e}, iterations "
          f"{b.stats.total_iterations} vs {a.stats.total_iterations}")


def test_acoustic2d_end_to_end():
    nz, nx, dh = 64, 96, 10.0
    nt, dt, f0 = 256, 2e-3, 10.0
    c0 = 2000.0
    vp = np.full((nz, nx), c0)
    rho = np.full((nz, nx), 1000.0)

    src_x, src_z = 30, 10
    rx = np.array([45, 65])                 # offsets 150 m and 350 m
    res = acoustic2d(vp, rho, dh, dt, nt, f0,
                     sx=[src_x], sz=[src_z], rx=rx, rz=10,
                     nbc=30, tol=1e-6, freq_batch=8, snap_interval=32,
                     dtype=torch.complex128, device=DEV, verbose=False)

    # shapes / stats
    assert res.seis_p.shape == (nt, 2) and res.seis_vz.shape == (nt, 2)
    assert res.snaps.shape[0] == nt // 32
    assert res.stats.total_iterations > 0
    assert np.isfinite(res.seis_p).all() and np.abs(res.seis_p).max() > 0

    # snapshots vs records: identical spectra, two synthesis paths
    js = np.arange(res.snaps.shape[0])
    snap_at_rec = res.snaps[:, 10, rx[1]]
    rec_at_snap = res.seis_p[js * 32, 1]
    err = (np.abs(snap_at_rec - rec_at_snap).max()
           / np.abs(res.seis_p).max())
    assert err < 1e-8, f"snapshot/record inconsistency: {err:.2e}"
    print(f"  snapshots == records at shared samples: {err:.2e}")

    # kinematics: direct-wave lag between the two receivers = d_offset / c
    tr1, tr2 = res.seis_p[:, 0], res.seis_p[:, 1]
    lag = int(np.argmax(np.correlate(tr2, tr1, "full"))) - (nt - 1)
    lag_true = (350.0 - 150.0) / c0 / dt     # = 50 samples
    assert abs(lag - lag_true) <= 2, f"lag {lag} vs {lag_true}"
    print(f"  direct-wave lag {lag} samples (theory {lag_true:.1f})")

    # resynthesis with the same wavelet reproduces the records
    re = res.resynthesize(res.wavelet)
    assert np.abs(re - res.seis_p).max() / np.abs(res.seis_p).max() < 1e-12
    print("  resynthesize(H_p) == records")


if __name__ == "__main__":
    print(f"device: {DEV}")
    test_engine_matches_multifreq()
    test_batched_shots_equal_sequential()
    test_acoustic2d_end_to_end()
    test_warm_start_engine()
    test_api_warm_start_consistency()
    print("engine & API tests passed")
