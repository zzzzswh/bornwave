"""bornwave: PyTorch Convergent-Born-Series Helmholtz solver for acoustics
with heterogeneous sound speed, density and absorption.

Reference: Stanziola, Arridge, Treeby, Cox — "Iterative Born solver for the
acoustic Helmholtz equation with heterogeneous sound speed and density"
(arXiv:2507.16087), built on the universal split-preconditioner of
Vettenburg & Vellekoop (arXiv:2207.14222).
"""
from .solver import CBSSolver2D, SolveResult, point_source_2d, alpha_from_Q
from .grid import next_fast_len
from .analytic import pressure_point_source_2d, fluid_reflection_coefficient
from .synthesis import ricker, band_indices, synthesize_shot
from .autograd import solve_helmholtz, HelmholtzDiagSolve
from .engine import CBSFreqShotBatch2D
from .api import acoustic2d, AcousticResult
from .viz import trace_norm, plot_shot, plot_wavefield_video
from .timesynth import time_slices_from_band

__all__ = [
    "CBSSolver2D",
    "SolveResult",
    "point_source_2d",
    "alpha_from_Q",
    "next_fast_len",
    "pressure_point_source_2d",
    "fluid_reflection_coefficient",
    "ricker",
    "band_indices",
    "synthesize_shot",
    "CBSFreqShotBatch2D",
    "acoustic2d",
    "AcousticResult",
    "trace_norm",
    "plot_shot",
    "plot_wavefield_video",
    "time_slices_from_band",
]
__version__ = "0.3.0"