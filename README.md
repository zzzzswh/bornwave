# bornwave

**English** · [简体中文](README.zh-CN.md)

[![Python](https://img.shields.io/badge/python-3.10%2B-3776ab.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-optional-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)

Frequency-domain GPU acoustic wave simulation.

`bornwave` solves the 2-D acoustic Helmholtz equation in media with arbitrary
heterogeneous sound speed, density and absorption. It is a matrix-free PyTorch
implementation of the convergent Born series (CBS) solver of Stanziola,
Arridge, Treeby & Cox (JASA, 2026), extended here with joint frequency × shot
batching, CUDA-graph execution, exact band-limited wavefield synthesis, and an
adjoint-state autograd interface. One call produces shot records, full
wavefield movies and FWI gradients.

Time-domain results are obtained by solving the wavelet's frequency components
in parallel and superposing them.

Despite the name this is a **full-wave** solver. "Born" refers to the form of
the iterative series, not to the first-order Born approximation: the converged
field satisfies the heterogeneous Helmholtz system to the measured true
residual, internal multiples and diffractions included.

<p align="center">
  <img src="docs/demo_wavefield.gif" width="70%" alt="Pressure wavefield movie for a two-layer model with a low-velocity lens"/>
</p>
<p align="center">
  <img src="docs/demo_shot_p.png" width="45%" alt="Trace-normalized pressure shot record"/>
  <br/>
  <em>Two-layer model with a low-velocity lens: pressure wavefield movie and shot
  record, both produced by a single call in <code>examples/demo_engine.py</code>.</em>
</p>

---

## Installation

Requires Python ≥ 3.10 and PyTorch ≥ 2.4 (CUDA optional but strongly
recommended), plus NumPy, SciPy and Matplotlib. `ffmpeg` is needed for MP4
export; without it, movies fall back to animated GIF.

```bash
git clone https://github.com/zzzzswh/bornwave.git
cd bornwave
uv sync                 # or: pip install -e .
```

Verify the installation — this cross-validates the engine against the
reference solver and runs an end-to-end consistency check (~1–2 min on CPU):

```bash
uv run tests/test_engine_api.py
```

Then reproduce the figures above, and the full physics validation suite:

```bash
uv run examples/demo_engine.py        # records + wavefield movie
uv run examples/two_layer_ricker.py   # analytic validation (Hankel, Zoeppritz, moveout)
uv run examples/fwi_gradient.py       # single-frequency FWI gradient
```

## Quick start

```python
import numpy as np
from bornwave import acoustic2d, trace_norm, plot_shot, plot_wavefield_video

nz, nx = 300, 400                        # arrays are (nz, nx), i.e. vp[z, x]
dh, dt, nt, f0 = 10.0, 1e-3, 2000, 15.0

vp  = np.full((nz, nx), 2500.0); vp[180:]  = 3200.0
rho = np.full((nz, nx), 2000.0); rho[180:] = 2300.0

res = acoustic2d(
    vp, rho, dh, dt, nt, f0,
    sx=[nx // 2], sz=[10],               # several shots: pass lists,
    rx=np.arange(0, nx, 2), rz=10,       # they are solved as one batch
    nbc=60, snap_interval=25,
)

res.seis_p, res.seis_vx, res.seis_vz     # (nt, nrec) time-domain records
res.snaps                                # (nsnap, nz, nx) wavefield movie
res.H_p                                  # unit-source transfer functions
res.resynthesize(other_wavelet)          # swap wavelets at zero cost
res.stats.kernel_time_s                  # timing and iteration diagnostics

plot_shot(trace_norm(res.seis_p), "shot.png", dt=dt)
plot_wavefield_video(res.snaps, "wavefield.mp4", fps=12, dh=dh,
                     snap_times=res.snap_times, adaptive_clims=True, model=vp)
```

## Method

### Governing system

The solver works on the first-order acoustic system in the frequency domain,
discretized on a staggered Fourier grid with field ordering
$[u_x, u_z, p]$ ($u_x$ offset by $+\Delta/2$ in $x$, $u_z$ by $+\Delta/2$ in
$z$, $p$ collocated):

```math
\begin{pmatrix}
\rho_0^{+}\,(i\omega + \gamma^{+}) & \nabla^{+} \\
\nabla^{-}\cdot & \dfrac{i\omega + \gamma}{\rho_0 c^2}
\end{pmatrix}
\begin{pmatrix}\mathbf{u}\\ p\end{pmatrix}
=
\begin{pmatrix}\hat{s}_u\\ \hat{s}_p\end{pmatrix}
```

Superscripts $\pm$ denote forward/backward staggering; $\rho_0^{+}$ is the
density linearly interpolated onto the half-grid. Free-space radiation is
imposed by the absorbing term $\gamma$, a polynomial ramp evaluated
analytically at both collocated and staggered coordinates. Absorption enters
through a complex squared sound speed,

```math
c^2 = \frac{c_0^2}{1 - 2i\alpha c_0/\omega}
\qquad\text{or, for constant }Q,\qquad
c^2 = \frac{c_0^2}{1 - i/Q},
```

the latter being exactly equivalent to the frequency-dependent
$\alpha(\omega) = \omega/(2c_0Q)$ while remaining frequency-independent to
assemble.

### Split preconditioner and fixed-point iteration

Write the system as $D w = \hat{s}$ with $D = \mathrm{Diag} + \mathcal{L}$,
where the differential block $\mathcal{L}$ is model-independent and the medium
enters only through the three diagonal fields. Each diagonal block is shifted
by $a$ (a component-wise median, the cheap approximation to the
minimum-bounding-circle centre used in the paper's Algorithm 1) and scaled by
$\lambda = \max|d - a| / \beta$ with $\beta = 0.95$, giving a scattering
potential $V = (\mathrm{Diag} - a)/\lambda$ with $\|V\| \le \beta < 1$. In the
scaled variables $x = C^{1/2} w$, $y = C^{-1/2}\hat{s}$,
$C^{1/2} = \mathrm{diag}(\sqrt{\lambda_1}, \sqrt{\lambda_1}, \sqrt{\lambda_2})$,
the system becomes $Ax = y$ with $A = L + V$, and the CBS iteration is

```math
x \leftarrow x + \nu\,B\left[(L+I)^{-1}(Bx + y) - x\right],
\qquad B = I - V,\quad \nu = 0.9 .
```

The contraction $\|V\| < 1$ is what guarantees convergence; the medium contrast
may be arbitrary as long as it is bounded.

### Cost per iteration

$(L+I)^{-1}$ is diagonal in the Fourier domain — a $3\times3$ matrix per
wavenumber. With $\mu = |k|^2 + (a_1+\lambda_1)(a_2+\lambda_2)$ and stagger
phases $S_e = e^{+i k_e \Delta/2}$, one iteration is:

> pointwise multiply-add → one batched FFT → an unrolled per-$k$ $3\times3$
> product → IFFT → pointwise multiply-add

No matrix assembly, no decomposition, no inner solver, and nothing that grows
with the number of shots: all operator tensors are shared across the shot
batch, which is why additional shots cost almost nothing beyond field memory.
The $3\times3$ product is deliberately unrolled into elementwise operations —
the equivalent `einsum` lowers to `permute + bmm` on CUDA and copies the full
symbol tensor every iteration.

### From frequency domain to time domain

A source wavelet is decomposed with `rfft`; bins above an amplitude threshold
are retained and solved in chunks of adjacent frequencies. Records follow from
$d(t) = \mathrm{irfft}(W \cdot H)$, and wavefield snapshots are evaluated as
exact band-limited inverse-transform samples of the stored full-field transfer
functions, so the $(n_t, n_z, n_x)$ cube is never materialized. Because the
transfer functions $H$ are stored, re-synthesizing with a different wavelet
costs one FFT and no solves.

## Features

- **Arbitrary heterogeneous models** — sound speed, density (interpolated onto
  the staggered half-grid) and absorption in Np/m or constant-$Q$.
- **Spectral spatial accuracy** — a Fourier pseudospectral discretization, so
  there is no grid dispersion to accumulate; the validation suite runs at
  10 points per wavelength and the Nyquist floor is 2.
- **Frequency × shot batching** — the engine iterates a single
  `(F, B, 3, Nz, Nx)` tensor, with converged frequencies finalized and
  compacted out of the working set on the fly.
- **CUDA Graphs** — at these grid sizes and iteration counts the wall time is
  dominated by kernel launch latency, not FLOPs. The fixed-point loop is
  captured and replayed as a single launch; the graph is re-captured after
  each compaction. Capture failure falls back to eager execution with a
  warning, and results are unchanged.
- **Wavefield movies without extra solves** — snapshots are synthesized from
  the same transfer functions as the records, and agree with `np.fft.irfft` to
  machine precision.
- **Differentiable** — `solve_helmholtz` implements the adjoint-state gradient
  via the implicit function theorem: exact, and $O(1)$ in memory with respect
  to the iteration count.

## API

### `acoustic2d` — one-call forward modeling

| Argument | Meaning |
|---|---|
| `vp`, `rho` | `(nz, nx)` velocity [m/s] and density [kg/m³]; strictly positive. `rho` may be a scalar. |
| `dh`, `dt`, `nt` | Grid spacing [m], sample interval [s], number of time samples. |
| `f0` | Ricker peak frequency [Hz]; ignored if `wavelet` is supplied. |
| `sx`, `sz` | Shot x/z grid indices — scalars or equal-length sequences, solved as one batch. |
| `rx`, `rz` | Receiver x/z grid indices; `rz` may be a scalar and is broadcast. |
| `alpha` / `Q` | Absorption [Np/m], or constant-$Q$ quality factor. Mutually exclusive. |
| `nbc` | Sponge thickness in cells; 40–60 is typical. This is a polynomial $\gamma$ ramp, not an FD boundary, so it does not need FD-style widths. |
| `tol` | Stopping tolerance on the relative increment. `2e-4` gives roughly 0.1–1 % amplitude accuracy (see [Validation](#validation)). |
| `freq_batch` | Frequencies solved jointly per chunk; the main memory/throughput knob. |
| `snap_interval` | Store a full pressure wavefield every this many time samples. |
| `cuda_graph` | `True` / `False` / `"auto"`. |

Returns an `AcousticResult` holding `seis_p`, `seis_vx`, `seis_vz`, `snaps`,
`snap_times`, transfer functions `H_p`, per-frequency `iterations` and
`residuals`, a `stats` namespace, and `resynthesize(wavelet)`.

Note that `vx`/`vz` records are the staggered $u_x$, $u_z$ fields at the
receiver cells; at seismic grid spacings the half-cell offset is far below a
wavelength.

### Lower-level entry points

```python
from bornwave import CBSSolver2D, CBSFreqShotBatch2D, synthesize_shot, solve_helmholtz
```

| Object | Use |
|---|---|
| `CBSSolver2D` | Single frequency, shot-batched. The reference implementation. |
| `CBSFreqBatch2D` | Frequency batch with convergence compaction. |
| `CBSFreqShotBatch2D` | Joint (frequency × shot) batch, CUDA-graph accelerated. Backs `acoustic2d`. |
| `synthesize_shot` | Wavelet → frequency band → time-domain gather, without the engine layer. |
| `solve_helmholtz` | Differentiable single-frequency solve; gradients w.r.t. $c_0$, $\rho_0$, $\alpha$ and the source. |

### Differentiable solves

`solve_helmholtz` is a `torch.autograd.Function` built on the implicit function
theorem. The backward pass runs one adjoint solve on the same CBS machinery
($V \to \bar V$, symbol → per-$k$ conjugate transpose) and forms the gradient
as a pointwise zero-lag correlation, so only the forward solution is stored.
The preconditioner internals are built from detached diagonals: the converged
solution does not depend on them, which makes the gradient exact rather than
approximate. `examples/fwi_gradient.py` shows a three-shot, single-frequency
FWI gradient imaging an interface absent from the starting model.

## Validation

Every layer is checked against something it did not produce itself — operator
identities, analytic Green's functions, plane-wave reflection theory,
reciprocity, and cross-validation between implementations.

| Check | Reference | Result | Script |
|---|---|---|---|
| $(L+I)(L+I)^{-1} = I$ per wavenumber | exact identity | 9.0e-16 | `test_operator_identity.py` |
| Adjoint symbol; skew-Hermitian differential block | exact identity | 1.4e-15 / 0 | `test_operator_identity.py` |
| Homogeneous medium, 10 ppw | analytic 2-D Hankel Green's function | rel. L2 **8.9e-5**, amplitude ratio 1.0000, phase 0.00° | `test_homogeneous_hankel.py` |
| Disk with 2× speed and 2.5× density contrast | acoustic reciprocity | **3.2e-6** | `test_heterogeneous_reciprocity.py` |
| Two-layer + 15 Hz Ricker, direct-wave window | band-limited Hankel | mean 0.075 %, max **0.18 %** | `examples/two_layer_ricker.py` |
| Zero-offset reflection amplitude | fluid Zoeppritz $R_0 = 0.5714$ | 0.5710 (**0.08 %**) | `examples/two_layer_ricker.py` |
| AVO curve, offsets ≤ 600 m | fluid Zoeppritz $R(\theta)$ | mean 0.13 %, max 0.54 % | `examples/two_layer_ricker.py` |
| Reflection moveout / inverted interface depth | ray theory, $h = 596.25$ m | ≤ 1.3 ms (< 1 sample); fitted 596.3 m | `examples/two_layer_ricker.py` |
| Engine (freq × shot batch) | `CBSFreqBatch2D` reference solver | 2.6e-7 | `test_engine_api.py` |
| Batched shots | the same shots solved sequentially | 0.0 | `test_engine_api.py` |
| Wavefield snapshots at shared samples | receiver records | 3e-16 | `test_engine_api.py` |
| Direct-wave lag between receivers | offset / $c$ | exact (50 / 50 samples) | `test_engine_api.py` |
| Band-limited time-slice synthesis | `np.fft.irfft` | machine precision | `test_timesynth.py` |
| Frequency batch | serial `CBSSolver2D` | shifts/scales exact; $H$ to iteration tolerance | `test_freq_batch.py` |
| Autograd through $(c_0, \rho_0, \alpha, s)$ | `torch.autograd.gradcheck` + directional finite differences | pass, rel. diff < 3e-5 | `test_autograd.py` |

Residuals quoted for the solver are **true** residuals $\|y - Ax\|/\|y\|$,
computed with an independently assembled forward symbol rather than the
iteration's own increment.

A full measured log of these runs is in `tests/test-log-260726.txt`.

## Performance

Reference problem — `examples/demo_engine.py`:

| | |
|---|---|
| Grid | 300 × 400, padded to 420 × 525 |
| Time samples | 2000 |
| Frequencies | 95, covering 0.5–47.5 Hz |
| CBS iterations | 76,456 total |
| Kernel time | **27 s**, CUDA Graphs enabled |
| Hardware | single CUDA GPU (`NVIDIA <model>`) |

Additional shots share all operator tensors and cost almost nothing beyond the
extra field memory. The per-chunk working-set size is printed at startup and is
controlled by `freq_batch`; iteration count grows with frequency, so grouping
adjacent bins keeps compaction waste small.

## Implementation notes

Four conventions that the source papers leave implicit, or that differ on a
staggered grid. Each was determined by experiment and is locked in by a test —
worth reading before modifying the internals.

1. **Time convention is $+i\omega t$** (the $H_0^{(2)}$ branch). Implemented
   literally, it coincides with the NumPy/PyTorch FFT transfer-function
   convention, so synthesis is $d(t) = \mathrm{irfft}(W \cdot H)$ with no
   conjugation anywhere.

2. **The 2-D point-source normalization is amplitude/$\Delta^2$.** A single
   node on a spectral grid represents a band-limited sinc of unit integral; the
   $2c_0/\Delta$ correction given in the paper is specific to 1-D. The measured
   amplitude ratio against the analytic Hankel solution is 1.0000.

3. **On a staggered grid the per-$k$ $(L+I)^{-1}$ matrix is not symmetric**
   (the stagger phases $e^{\pm ik\Delta/2}$ break it). The adjoint symbol is the
   per-$k$ **conjugate transpose**, not the elementwise conjugate — the latter
   holds only for the non-staggered variant. Both cost the same to apply.

4. **Grazing-incidence sponge ghosts.** Sponge absorption is inefficient for
   near-horizontally propagating energy. Keep sources and receivers roughly one
   dominant wavelength away from the absorbing layer; closer than that, residual
   grazing reflections are not separable from the direct wave and long-offset
   errors reach several percent. This is the same failure mode as absorbing
   boundaries in time-domain finite differences.

## Limitations

- **Time wraparound.** Frequency sampling $\Delta f = 1/(n_t \Delta t)$ makes
  the synthesized response periodic with period $T = n_t \Delta t$: any coda
  still ringing at $t = T$ aliases back to $t = 0$ and appears *before the
  source fires*. Increase `nt`, or window/damp, when late energy matters. The
  residual noise floor of the iterative solve is likewise non-causal — uniform
  in time — and is set by `tol`.
- **No free surface yet.** Vacuum cells (`vp = 0`) lie outside the CBS
  convergence domain by construction, since the contraction requires bounded
  contrast. All four boundaries are absorbing sponges, so there are no surface
  multiples; internal multiples are fully present.
- **2-D only**, with a single uniform grid spacing.

## Repository layout

```
bornwave/
  solver.py      CBSSolver2D — single frequency, shot-batched
  multifreq.py   CBSFreqBatch2D — frequency batch + convergence compaction
  engine.py      CBSFreqShotBatch2D — (freq × shot) batch + CUDA Graphs
  api.py         acoustic2d — one-call forward-modeling entry point
  autograd.py    differentiable solve (implicit function theorem / adjoint)
  operators.py   per-k Fourier symbols of (L+I)^-1 and (L+I), staggered
  grid.py        FFT-friendly sizes, sponge profiles, staggered averaging
  synthesis.py   Ricker wavelet, band selection, per-frequency synthesis
  timesynth.py   band spectra → time slices (torch-free, machine precision)
  viz.py         shot plots, wavefield movies (torch-free)
  analytic.py    Hankel Green's function, fluid Zoeppritz (validation only)
examples/        demo_engine.py, two_layer_ricker.py, fwi_gradient.py
tests/           validation suite + measured log
```

## Roadmap

- Free surface via the image method
- Complex-frequency damping to suppress time wraparound
- Anderson fluid-cylinder analytic benchmark for the variable-density path
- True-residual, frequency-adaptive stopping criterion
- Frequency-bucket scheduling
- Osnabrugge (2021) ultra-thin absorbing boundary layer
- 3-D (four fields, eight FFTs per iteration)

## References

This repository is an independent implementation. All algorithmic credit
belongs to the following papers; the implementation, the engine layer and any
bugs are ours.

1. A. Stanziola, S. R. Arridge, B. E. Treeby, B. T. Cox,
   *Iterative Born solver for the acoustic Helmholtz equation with
   heterogeneous sound speed and density*,
   J. Acoust. Soc. Am. **159** (2026) 1457–1470.
   [doi:10.1121/10.0042259](https://doi.org/10.1121/10.0042259) ·
   [arXiv:2507.16087](https://arxiv.org/abs/2507.16087)
2. T. Vettenburg, I. M. Vellekoop,
   *A universal matrix-free split preconditioner for the fixed-point iterative
   solution of non-symmetric linear systems*,
   [arXiv:2207.14222](https://arxiv.org/abs/2207.14222) (2022).
3. G. Osnabrugge, S. Leedumrongwatthanakun, I. M. Vellekoop,
   *A convergent Born series for solving the inhomogeneous Helmholtz equation
   in arbitrarily large media*,
   J. Comput. Phys. **322** (2016) 113–124.
   [doi:10.1016/j.jcp.2016.06.034](https://doi.org/10.1016/j.jcp.2016.06.034)

<details>
<summary>BibTeX</summary>

```bibtex
@article{stanziola2026iterativeborn,
  title   = {Iterative {Born} solver for the acoustic {Helmholtz} equation
             with heterogeneous sound speed and density},
  author  = {Stanziola, Antonio and Arridge, Simon R. and
             Treeby, Bradley E. and Cox, Benjamin T.},
  journal = {The Journal of the Acoustical Society of America},
  volume  = {159},
  number  = {2},
  pages   = {1457--1470},
  year    = {2026},
  doi     = {10.1121/10.0042259}
}

@misc{vettenburg2022universal,
  title         = {A universal matrix-free split preconditioner for the
                   fixed-point iterative solution of non-symmetric linear systems},
  author        = {Vettenburg, Tom and Vellekoop, Ivo M.},
  year          = {2022},
  eprint        = {2207.14222},
  archivePrefix = {arXiv},
  primaryClass  = {math.NA}
}

@article{osnabrugge2016convergent,
  title   = {A convergent {Born} series for solving the inhomogeneous
             {Helmholtz} equation in arbitrarily large media},
  author  = {Osnabrugge, Gerwin and Leedumrongwatthanakun, Saroch and
             Vellekoop, Ivo M.},
  journal = {Journal of Computational Physics},
  volume  = {322},
  pages   = {113--124},
  year    = {2016},
  doi     = {10.1016/j.jcp.2016.06.034}
}
```
</details>

## License

Not yet declared. Please open an issue if you need a specific license for your
use case.