"""Decision experiment #2: frequency-continuation warm starts for CBS sweeps.

Solve the same frequency band four ways and count iterations:

  cold        x0 = 0 for every frequency               (current baseline)
  naive       x0 = physical field of previous freq
  rot         x0 = previous field x exp(-i dw tau(x)),
              tau = straight-ray travel time from the source
  rot+extrap  linear extrapolation of the DEPHASED envelope
              E_k = w_k exp(+i w_k tau):  E_pred = 2 E_k - E_{k-1}

All strategies must converge to the same solution (checked); only the
iteration count may differ. The scaled solver state is C^{1/2}-weighted
with frequency-dependent lambda, so carrying a field across frequencies
means: unscale (old sqrt_lam) -> rotate -> rescale (new sqrt_lam).

Sign convention: this solver realizes +iwt / H0^(2), i.e. outgoing phase
exp(-i w tau) — pinned by tests/test_homogeneous_hankel.py. If you flip
the convention, flip the rotation sign.

    python examples/exp_freq_continuation.py       # GPU recommended
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import map_coordinates

from bornwave import CBSSolver2D, point_source_2d, ricker, band_indices
from bornwave.grid import pad_replicate_2d

OUT = pathlib.Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TOL, ABS_POINTS = 2e-4, 50

# ---- model: two-layer + low-velocity lens (mini demo model) ---------------
nz, nx, dh = 150, 200, 10.0
vp = np.full((nz, nx), 2500.0)
rho = np.full((nz, nx), 2000.0)
vp[90:], rho[90:] = 3200.0, 2300.0
zz, xx = np.mgrid[0:nz, 0:nx]
vp[(zz - 45) ** 2 + (xx - 130) ** 2 < 18 ** 2] = 2000.0
src_z, src_x = 10, 75

# ---- frequency band --------------------------------------------------------
nt, dt, f0 = 1000, 1.5e-3, 15.0
w_ric, _ = ricker(f0, nt, dt)
W = np.fft.rfft(w_ric)
freqs = np.fft.rfftfreq(nt, dt)
keep = band_indices(W, 1e-3)
fs = freqs[keep]
print(f"device={DEV}  {fs.size} freqs in [{fs[0]:.2f}, {fs[-1]:.2f}] Hz  "
      f"tol={TOL}")

# ---- straight-ray travel time tau(x) on the padded grid --------------------
def straight_ray_tau(c0, src, n_samp=64):
    """tau(z,x) = |r| * mean slowness along the straight segment src->(z,x)."""
    nz_, nx_ = c0.shape
    s = 1.0 / c0
    t = np.linspace(0.0, 1.0, n_samp)[:, None, None]
    pz = src[0] + t * (np.arange(nz_)[None, :, None] - src[0])   # (S,nz,1)
    px = src[1] + t * (np.arange(nx_)[None, None, :] - src[1])   # (S,1,nx)
    pz = np.broadcast_to(pz, (n_samp, nz_, nx_)).ravel()
    px = np.broadcast_to(px, (n_samp, nz_, nx_)).ravel()
    s_line = map_coordinates(s, [pz, px], order=1).reshape(n_samp, nz_, nx_)
    r = np.hypot(zz - src[0], xx - src[1]) * dh
    return r * s_line.mean(axis=0)

tau_int = straight_ray_tau(vp, (src_z, src_x))

# probe solver just to learn the padded geometry (any frequency works)
_probe = CBSSolver2D(vp, rho, 2 * np.pi * fs[0], dh, abs_points=ABS_POINTS,
                     dtype=torch.complex64, device=DEV)
pt = _probe.interior[0].start
pads = (pt, _probe.shape[0] - pt - nz,
        _probe.interior[1].start, _probe.shape[1] - _probe.interior[1].start - nx)
tau_pad = pad_replicate_2d(torch.as_tensor(tau_int), pads).to(torch.float64)
tau_t = tau_pad.to(DEV)
del _probe

# ---- the sweep, four ways ---------------------------------------------------
STRATS = ["cold", "naive", "rot", "rot+extrap"]
iters = {s: np.zeros(fs.size, dtype=int) for s in STRATS}
walls = {}
p_cold = np.empty((fs.size, nz, nx), dtype=np.complex64)
maxdiff = {s: 0.0 for s in STRATS}

for strat in STRATS:
    t0 = time.time()
    w_prev = w_prev2 = None          # physical (unscaled) padded states
    om_prev = om_prev2 = None
    for j, f in enumerate(fs):
        om = 2 * np.pi * f
        solver = CBSSolver2D(vp, rho, om, dh, abs_points=ABS_POINTS,
                             dtype=torch.complex64, device=DEV)
        sp = point_source_2d(nz, nx, src_z, src_x, dh, device=DEV)

        x0 = None
        if w_prev is not None and strat != "cold":
            if strat == "naive":
                w0 = w_prev
            elif strat == "rot" or w_prev2 is None:
                rot = torch.exp(-1j * (om - om_prev) * tau_t)
                w0 = w_prev * rot.to(w_prev.dtype)
            else:  # rot+extrap on the dephased envelope
                E1 = w_prev * torch.exp(1j * om_prev * tau_t).to(w_prev.dtype)
                E2 = w_prev2 * torch.exp(1j * om_prev2 * tau_t).to(w_prev.dtype)
                w0 = (2.0 * E1 - E2) * torch.exp(-1j * om * tau_t).to(w_prev.dtype)
            x0 = w0 * solver.sqrt_lam            # rescale into new coordinates

        res = solver.solve(sp=sp, x0=x0, tol=TOL, max_iter=60000,
                           compute_residual=False)
        iters[strat][j] = res.iterations
        w_phys = res.x_scaled / solver.sqrt_lam  # physical padded state
        w_prev2, om_prev2 = w_prev, om_prev
        w_prev, om_prev = w_phys, om

        if strat == "cold":
            p_cold[j] = res.p.cpu().numpy()
        else:                                     # same solution check
            d = (res.p.cpu().numpy() - p_cold[j])
            maxdiff[strat] = max(maxdiff[strat],
                                 float(np.abs(d).max() / np.abs(p_cold[j]).max()))
        del solver, res
    walls[strat] = time.time() - t0
    tot = int(iters[strat].sum())
    print(f"  {strat:<11s} total iters {tot:7d}   wall {walls[strat]:6.1f}s"
          + ("" if strat == "cold" else
             f"   speedup x{iters['cold'].sum() / tot:.2f}"
             f"   max field diff vs cold {maxdiff[strat]:.1e}"))

# ---- report -----------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={"width_ratios": [2.4, 1]})
for s, c in zip(STRATS, ["k", "tab:blue", "tab:orange", "tab:red"]):
    a1.plot(fs, iters[s], c, lw=1.6, label=s)
a1.set_xlabel("frequency (Hz)"); a1.set_ylabel("CBS iterations")
a1.legend(); a1.set_title(f"warm-start strategies, tol={TOL}")
tots = [iters[s].sum() for s in STRATS]
a2.bar(range(4), tots, color=["k", "tab:blue", "tab:orange", "tab:red"])
a2.set_xticks(range(4)); a2.set_xticklabels(STRATS, rotation=20)
for i, t in enumerate(tots):
    a2.text(i, t, f"x{tots[0] / t:.2f}", ha="center", va="bottom", fontsize=9)
a2.set_title("total iterations (label = speedup)")
fig.tight_layout()
fig.savefig(OUT / "exp_freq_continuation.png", dpi=140)
print(f"wrote {OUT / 'exp_freq_continuation.png'}")
