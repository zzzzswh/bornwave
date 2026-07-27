"""Engine demo — the target workflow, end to end.

Two-layer model with a low-velocity lens, one shot, a surface receiver
line: records + wavefield movie in one call. On a GPU this runs in
seconds-to-minutes (CUDA graphs on); on CPU expect much longer — shrink
nt / the model or raise tol for a quick look.

    python examples/demo_engine.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from bornwave import acoustic2d, trace_norm, plot_shot, plot_wavefield_video

OUT = pathlib.Path(__file__).resolve().parent

# ---- grid & acquisition (NOTE: arrays are (nz, nx), i.e. vp[z, x]) --------
nz, nx = 300, 400
dh, dt, nt, f0 = 10.0, 1e-3, 2000, 15.0

vp = np.full((nz, nx), 2500.0, dtype=np.float32)
rho = np.full((nz, nx), 2000.0, dtype=np.float32)
vp[180:], rho[180:] = 3200.0, 2300.0                 # interface at 1800 m

zz, xx = np.mgrid[0:nz, 0:nx]
lens = (zz - 90) ** 2 + (xx - 260) ** 2 < 35 ** 2     # low-velocity lens
vp[lens] = 2000.0

sx, sz = [nx // 2], [10]
rx = np.arange(0, nx, 2)
rz = 10

# ---- forward modeling ------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("WARNING: no GPU found — this demo size is slow on CPU. "
          "Consider nt=800, nz,nx=(150,200) for a first run.")

res = acoustic2d(vp, rho, dh, dt, nt, f0,
                 sx=sx, sz=sz, rx=rx, rz=rz,
                 nbc=60, tol=2e-4, freq_batch=16,
                 snap_interval=25, device=device)

print(res)
print(f"kernel {res.stats.kernel_time_s:.1f}s on {res.stats.device}, "
      f"{res.stats.n_freqs} freqs in "
      f"[{res.stats.band_hz[0]:.1f}, {res.stats.band_hz[1]:.1f}] Hz, "
      f"{res.stats.total_iterations} CBS iterations")

# ---- outputs ---------------------------------------------------------------
offsets = (rx - sx[0]) * dh
plot_shot(trace_norm(res.seis_p), OUT / "demo_shot_p.png",
          dt=dt, offsets=offsets, title="pressure record (trace-normalized)")
plot_shot(trace_norm(res.seis_vz), OUT / "demo_shot_vz.png",
          dt=dt, offsets=offsets, title="vz record (trace-normalized)")

plot_wavefield_video(res.snaps, OUT / "demo_wavefield.mp4",
                     fps=12, dh=dh, snap_times=res.snap_times,
                     adaptive_clims=True, model=vp,
                     title="pressure wavefield")

print(f"wrote {OUT / 'demo_shot_p.png'}")
print(f"wrote {OUT / 'demo_shot_vz.png'}")
print(f"wrote {OUT / 'demo_wavefield.mp4'}")
