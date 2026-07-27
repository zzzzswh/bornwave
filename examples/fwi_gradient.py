"""FWI-gradient sanity: does dJ/dc0 image a hidden interface?

Observed data: two-layer model. Initial model: homogeneous background.
Single 12 Hz frequency, 3 shots solved as one batch through the
differentiable solve_helmholtz; the misfit gradient w.r.t. c0 must focus
at the (unknown to the initial model) interface depth -- the classic
single-frequency migration-isochrone image.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bornwave import solve_helmholtz

OUT = pathlib.Path(__file__).resolve().parent

torch.manual_seed(0)
nz, nx, dx = 96, 144, 10.0
iz_int = 56                                   # true interface at 555 m
c_true = torch.full((nz, nx), 1500., dtype=torch.float64)
r_true = torch.full((nz, nx), 1000., dtype=torch.float64)
c_true[iz_int:], r_true[iz_int:] = 2200., 1800.

f = 12.0
omega = 2 * np.pi * f
src_iz, rec_iz = 10, 10
shots_ix = torch.tensor([36, 72, 108])
rec_ix = torch.arange(8, 137, 4)

def sources():
    sp = torch.zeros(len(shots_ix), nz, nx, dtype=torch.complex128)
    for b, ix in enumerate(shots_ix):
        sp[b, src_iz, ix] = 1.0 / dx**2
    return sp

t0 = time.time()
with torch.no_grad():                          # observed data (true model)
    p_obs, _, _ = solve_helmholtz(c_true, r_true, omega, dx, sources(),
                                  tol=1e-6, cdtype=torch.complex128)
    d_obs = p_obs[:, rec_iz, rec_ix]

c_init = torch.full((nz, nx), 1500., dtype=torch.float64, requires_grad=True)
p, _, _ = solve_helmholtz(c_init, r_true, omega, dx, sources(),
                          tol=1e-6, cdtype=torch.complex128)
res = p[:, rec_iz, rec_ix] - d_obs
J = 0.5 * res.abs().pow(2).sum()
J.backward()                                   # ONE adjoint solve, O(1) memory
g = c_init.grad.numpy()
print(f"J = {J.item():.3e},  wall = {time.time()-t0:.1f}s "
      f"(3 shots, forward + adjoint)")

zpk = np.abs(g[20:]).sum(axis=1).argmax() + 20
print(f"peak-|g| depth row = {zpk} (true interface rows {iz_int-1}/{iz_int})")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].imshow(c_true, cmap="viridis", extent=[0, nx*dx, nz*dx, 0])
ax[0].set(title="true c0 (init: homogeneous 1500)", xlabel="x [m]",
          ylabel="z [m]")
cl = np.percentile(np.abs(g), 99.5)
ax[1].imshow(g, cmap="seismic", vmin=-cl, vmax=cl,
             extent=[0, nx*dx, nz*dx, 0])
ax[1].axhline((iz_int - 0.5)*dx, color="k", ls="--", lw=0.8)
ax[1].set(title=f"dJ/dc0, {f:.0f} Hz, 3 shots (dashed: true interface)",
          xlabel="x [m]")
fig.tight_layout()
fig.savefig(OUT / "fwi_gradient.png", dpi=140)
print("saved", OUT / "fwi_gradient.png")

assert zpk in range(iz_int - 3, iz_int + 4), "gradient not focused at interface"
print("FWI gradient sanity OK")