"""Frequency-sweep movie: the SAME wavefield data, unfolded along omega.

The time-domain movie (demo_engine.py) and this one are two views of one
3D cube: u(x, z, t) = irfft over omega of W(omega) * H(x, z, omega).
Here each frame is the monochromatic steady-state field Re H(x, z, f_k)
(unit source, source always on); the inset shows the Ricker weight |W(f)|
that each frame carries into the time-domain synthesis.

    python examples/freq_sweep_movie.py         # GPU recommended
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, gridspec
import numpy as np
import torch

from bornwave.engine import CBSFreqShotBatch2D
from bornwave.synthesis import band_indices, ricker

OUT = pathlib.Path(__file__).resolve().parent

# ---- same model as demo_engine.py -----------------------------------------
nz, nx = 300, 400
dh, dt, nt, f0 = 10.0, 1e-3, 2000, 15.0
vp = np.full((nz, nx), 2500.0, dtype=np.float32)
rho = np.full((nz, nx), 2000.0, dtype=np.float32)
vp[180:], rho[180:] = 3200.0, 2300.0
zz, xx = np.mgrid[0:nz, 0:nx]
vp[(zz - 90) ** 2 + (xx - 260) ** 2 < 35 ** 2] = 2000.0
sz, sx = 10, nx // 2

device = "cuda" if torch.cuda.is_available() else "cpu"
dev = torch.device(device)

# ---- solve the retained band, keep the FULL field per frequency -----------
wavelet, t0 = ricker(f0, nt, dt)
W = np.fft.rfft(wavelet)
freqs = np.fft.rfftfreq(nt, dt)
keep = band_indices(W, 1e-3)
print(f"{keep.size} freqs in [{freqs[keep[0]]:.2f}, {freqs[keep[-1]]:.2f}] Hz "
      f"on {device}")

sp = torch.zeros((1, nz, nx), dtype=torch.complex64, device=dev)
sp[0, sz, sx] = 1.0 / dh**2

H = np.empty((keep.size, nz, nx), dtype=np.complex64)
for i0 in range(0, keep.size, 16):
    chunk = keep[i0:i0 + 16]
    eng = CBSFreqShotBatch2D(vp, rho, 2 * np.pi * freqs[chunk], dh,
                             abs_points=60, dtype=torch.complex64,
                             device=device)
    fields, its, rs = eng.solve(sp, tol=2e-4)
    H[i0:i0 + chunk.size] = fields[:, 0, 2].cpu().numpy()
    print(f"  {freqs[chunk[0]]:5.2f}-{freqs[chunk[-1]]:5.2f} Hz  "
          f"iters<={int(its.max())}", flush=True)

# ---- animate along omega ----------------------------------------------------
fig = plt.figure(figsize=(7.6, 8.2))
gs = gridspec.GridSpec(2, 1, height_ratios=[4.2, 1.0], hspace=0.28)
ax = fig.add_subplot(gs[0])
axs = fig.add_subplot(gs[1])
ext = [0, nx * dh, nz * dh, 0]

ax.imshow(vp, cmap="gray", extent=ext, interpolation="bilinear")
im = ax.imshow(H[0].real, cmap="seismic", extent=ext, alpha=0.7,
               interpolation="bilinear")
ax.plot(sx * dh, sz * dh, "k*", ms=10)
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
ax.set_title(r"monochromatic field  Re $H(x,z,f)$   (source always on)")
lab = ax.text(0.02, 0.05, "", transform=ax.transAxes, fontsize=11,
              bbox=dict(fc="w", alpha=0.75, ec="none"))

Wn = np.abs(W[keep]) / np.abs(W).max()
axs.plot(freqs[keep], Wn, "k-", lw=1.2)
axs.fill_between(freqs[keep], 0, Wn, alpha=0.15)
mark = axs.axvline(freqs[keep[0]], color="crimson", lw=2)
dot, = axs.plot([], [], "o", color="crimson", ms=6)
axs.set_xlabel("frequency (Hz)")
axs.set_ylabel(r"$|W(f)|$")
axs.set_ylim(0, 1.08)
axs.set_title("Ricker spectrum — weight of this frame in the time movie",
              fontsize=9)


def update(j):
    fr = H[j].real
    v = np.percentile(np.abs(fr), 99.3)
    im.set_data(fr)
    im.set_clim(-v, v)
    f = freqs[keep[j]]
    lab.set_text(f"f = {f:5.2f} Hz    $\\lambda_0$ = c$_0$/f = "
                 f"{2500.0 / f:5.0f} m")
    mark.set_xdata([f, f])
    dot.set_data([f], [Wn[j]])
    return im, lab, mark, dot


ani = animation.FuncAnimation(fig, update, frames=keep.size, blit=False)
out = OUT / "freq_sweep.mp4"
ani.save(out, writer=animation.FFMpegWriter(fps=6, bitrate=2400), dpi=115)
plt.close(fig)
print(f"wrote {out}")
