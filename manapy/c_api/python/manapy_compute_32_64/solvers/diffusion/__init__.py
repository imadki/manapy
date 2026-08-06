"""Pure-diffusion (diffusion) solver kernels.

Ported from src/solvers/to_convert.py: the dissipative (diffusion) residual and
the diffusion-only CFL time step. The forward-Euler update is not duplicated
here -- use `manapy_compute_32_64.solvers.utils.update_new_value`.
"""

from ._core import (
    explicitscheme_dissipative,
    explicitscheme_dissipative_cuda,
    time_step,
    time_step_cuda,
)

__all__ = [
    "explicitscheme_dissipative",
    "explicitscheme_dissipative_cuda",
    "time_step",
    "time_step_cuda",
]
