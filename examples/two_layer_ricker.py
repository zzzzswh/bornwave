"""Two-layer velocity/density model + 15 Hz Ricker source -> shot gather.

The gather is synthesized frequency-by-frequency (Stage 4 pipeline) and
validated quantitatively against analytic references built from the SAME
truncated wavelet band (so band-limiting cancels exactly):

  1. direct wave      p_d(t; x) = irfft( W_k * (w/4c1^2) H0^(2)(k r) )
  2. reflected wave   image source at distance r_m = sqrt(x^2 + 4h^2),
                      amplitude ~ plane-wave fluid Zoeppritz R(theta)
  3. AVO curve        projection coefficient alpha(x) vs R(theta(x))
  4. moveout          envelope peak times vs t0 + r_m / c1

Geometry: dx = 7.5 m, interior 176 x 240 (1320 x 1800 m), interface between
rows 99/100 -> z_int = 746.25 m; source (150 m, 900 m); 73 receivers at the
source depth (line depth 150 m keeps the top-sponge grazing ghost weak and
separable from the direct wave; see README). All boundaries absorbing (no
free surface -> no ghosts and no surface multiples: direct + one primary).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from bornwave import (synthesize_shot, pressure_point_source_2d,
                      fluid_reflection_coefficient)

OUT = pathlib.Path(__file__).resolve().parent

# ----------------------------------------------------------------- model
c1, rho1 = 1500.0, 1000.0
c2, rho2 = 2500.0, 2200.0
nz, nx, dx = 176, 240, 7.5
iz_int = 100                              # first row of medium 2
z_int = (iz_int - 0.5) * dx               # effective reflector: 746.25 m

c0 = torch.full((nz, nx), c1, dtype=torch.float64)
rho0 = torch.full((nz, nx), rho1, dtype=torch.float64)
c0[iz_int:], rho0[iz_int:] = c2, rho2

src = (20, 120)                           # (150 m, 900 m)
rec_iz = 20
rec_ix = np.arange(12, 229, 3)            # 73 receivers, offsets -810..810 m
offsets = (rec_ix - src[1]) * dx
h = z_int - src[0] * dx                   # source-to-interface: 596.25 m

f0, nt, dt = 15.0, 768, 0.002

# ----------------------------------------------------------------- solve
print(f"two-layer model {nz}x{nx}, dx={dx} m, interface z={z_int:.2f} m, "
      f"h={h:.2f} m, {rec_ix.size} receivers")
out = synthesize_shot(c0, rho0, dx, src, rec_iz, rec_ix, nt, dt, f0=f0,
                      tol=2e-4, abs_points=60, verbose=True)
gather, t, t0 = out["gather"], out["t"], out["t0"]
W, freqs, band = out["W"], out["freqs"], out["band"]
print(f"{band.size} frequencies in [{freqs[band[0]]:.2f}, {freqs[band[-1]]:.2f}] Hz, "
      f"total iterations {out['iterations'].sum()}, wall {out['wall_time']:.0f}s")

# ------------------------------------------------- analytic reference gathers
def hankel_gather(r):
    """Band-limited analytic gather for propagation distances r (nrec,)."""
    D = np.zeros((freqs.size, r.size), dtype=np.complex128)
    for k in band:
        om = 2 * np.pi * freqs[k]
        D[k] = W[k] * pressure_point_source_2d(r, om, c1)
    return np.fft.irfft(D, n=nt, axis=0)

r_dir = np.abs(offsets)
r_mir = np.hypot(offsets, 2 * h)
nonzero = r_dir > 0
direct_ref = np.zeros_like(gather)
direct_ref[:, nonzero] = hankel_gather(r_dir[nonzero])
mirror_ref = hankel_gather(r_mir)         # image source, R = 1

theta = np.arctan2(np.abs(offsets), 2 * h)
R_theta = fluid_reflection_coefficient(theta, c1, rho1, c2, rho2).real
R0 = fluid_reflection_coefficient(0.0, c1, rho1, c2, rho2).real

# ------------------------------------------------------------ 1. direct wave
sel = (np.abs(offsets) >= 150) & (np.abs(offsets) <= 800)
errs = []
for j in np.nonzero(sel)[0]:
    win = (t >= r_dir[j] / c1) & (t <= r_dir[j] / c1 + 2 * t0)
    errs.append(np.linalg.norm(gather[win, j] - direct_ref[win, j])
                / np.linalg.norm(direct_ref[win, j]))
errs = np.asarray(errs)
print(f"direct-wave window rel L2: mean={errs.mean():.3e} max={errs.max():.3e}")

# ----------------------------------------------- 2-3. reflection amplitude/AVO
scattered = gather - direct_ref           # analytic direct removed
alpha = np.zeros(offsets.size)
t_pk = np.zeros(offsets.size)
for j in range(offsets.size):
    tm = r_mir[j] / c1 + t0
    win = (t >= tm - 0.08) & (t <= tm + 0.12)
    num = scattered[win, j] if nonzero[j] else gather[win, j]
    ref = mirror_ref[win, j]
    alpha[j] = np.dot(num, ref) / np.dot(ref, ref)
    env = np.abs(hilbert(num))
    t_pk[j] = t[win][np.argmax(env)]

avo_sel = np.abs(offsets) <= 600          # comfortably subcritical
avo_err = np.abs(alpha[avo_sel] - R_theta[avo_sel]) / R0
j0 = np.argmin(np.abs(offsets))
print(f"zero-offset reflection: alpha={alpha[j0]:.4f} vs R0={R0:.4f} "
      f"({abs(alpha[j0]-R0)/R0*100:.2f}% off)")
print(f"AVO |alpha - R(theta)|/R0 over |x|<=600 m: mean={avo_err.mean():.3e} "
      f"max={avo_err.max():.3e}")

# ------------------------------------------------------------ 4. moveout + h fit
t_theory = r_mir / c1 + t0
dt_err = t_pk - t_theory
h_fit = np.sqrt(((c1 * (t_pk - t0)) ** 2 - offsets ** 2).clip(0).mean()) / 2
print(f"moveout |dt|: mean={np.abs(dt_err).mean()*1e3:.2f} ms "
      f"max={np.abs(dt_err).max()*1e3:.2f} ms;  fitted h={h_fit:.1f} m "
      f"(model {h:.2f} m)")

# ----------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))

a = ax[0, 0]
im = a.imshow(c0.numpy(), cmap="viridis", extent=[0, nx * dx, nz * dx, 0])
a.plot(src[1] * dx, src[0] * dx, "r*", ms=14, label="source")
a.plot(rec_ix * dx, np.full(rec_ix.size, rec_iz * dx), "cv", ms=3, label="receivers")
a.set(title=f"(a) model: {c1:.0f}/{rho1:.0f} over {c2:.0f}/{rho2:.0f}",
      xlabel="x [m]", ylabel="z [m]")
a.legend(loc="lower left", frameon=False, labelcolor="w")
plt.colorbar(im, ax=a, fraction=0.04, label="c [m/s]")

a = ax[0, 1]
clip = 0.25 * np.abs(gather).max()
a.imshow(gather, cmap="gray", aspect="auto", vmin=-clip, vmax=clip,
         extent=[offsets[0], offsets[-1], t[-1], 0])
a.plot(offsets, r_dir / c1 + t0, "y--", lw=1, label="direct (analytic)")
a.plot(offsets, t_theory, "r--", lw=1, label="reflection (analytic)")
a.set(title="(b) synthesized shot gather", xlabel="offset [m]", ylabel="t [s]",
      ylim=(1.35, 0))
a.legend(loc="lower right", frameon=False, labelcolor="w")

a = ax[1, 0]
a.plot(t, gather[:, j0], "r", lw=1.6, label="bornwave, offset 0")
a.plot(t, direct_ref[:, j0] + R0 * mirror_ref[:, j0], "k--", lw=1.1,
       label=r"analytic direct + $R_0\,\cdot$ image")
shift = 1.6 * np.abs(gather[t > 0.7, j0]).max()
a.plot(t, gather[:, j0] - R0 * mirror_ref[:, j0] - direct_ref[:, j0] - shift,
       "0.5", lw=0.8, label="difference (shifted)")
a.set(title="(c) zero-offset trace", xlabel="t [s]", xlim=(0.7, 1.35))
a.legend(frameon=False)

a = ax[1, 1]
a.plot(offsets, alpha, "ro", ms=4, label=r"bornwave $\alpha(x)$")
xs = np.linspace(0, offsets.max(), 300)
Rs = fluid_reflection_coefficient(np.arctan2(xs, 2 * h), c1, rho1, c2, rho2).real
a.plot(xs, Rs, "k-", lw=1.2, label="fluid Zoeppritz $R(\\theta)$")
a.plot(-xs, Rs, "k-", lw=1.2)
a.axhline(R0, color="0.7", lw=0.8, ls=":")
a.set(title="(d) reflection AVO", xlabel="offset [m]", ylabel="R")
a.legend(frameon=False)

fig.tight_layout()
fig.savefig(OUT / "two_layer_ricker.png", dpi=140)
np.savez(OUT / "two_layer_ricker.npz", gather=gather, t=t, offsets=offsets,
         alpha=alpha, R_theta=R_theta, t_pk=t_pk, t_theory=t_theory)
print("saved", OUT / "two_layer_ricker.png")

# ----------------------------------------------------------------- asserts
assert errs.max() < 0.02, "direct wave mismatch"
assert abs(alpha[j0] - R0) / R0 < 0.04, "zero-offset reflection amplitude"
assert avo_err.max() < 0.06, "AVO mismatch"
assert np.abs(dt_err).max() < 0.008, "moveout mismatch"
print("two-layer Ricker validation passed")