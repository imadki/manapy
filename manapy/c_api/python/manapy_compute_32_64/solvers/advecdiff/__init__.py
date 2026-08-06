"""Advection-diffusion (advecdiff) solver kernels.

Ported from src/solvers/to_convert.py (see src/solvers/Steps.md); kernels are
added batch by batch.
"""

from ._core import (
    explicitscheme_convective_2d,
    explicitscheme_convective_2d_cuda,
    explicitscheme_convective_3d,
    explicitscheme_convective_3d_cuda,
    explicitscheme_dissipative,
    explicitscheme_dissipative_cuda,
    time_step,
    time_step_cuda,
)

__all__ = [
    "explicitscheme_convective_2d",
    "explicitscheme_convective_2d_cuda",
    "explicitscheme_convective_3d",
    "explicitscheme_convective_3d_cuda",
    "explicitscheme_dissipative",
    "explicitscheme_dissipative_cuda",
    "time_step",
    "time_step_cuda",
]
