"""Solver utility kernels (common to all solvers) for float64 data and int32 indices."""

from ._core import (
    initialisation_gaussian_2d,
    initialisation_gaussian_3d,
    update_new_value,
    update_new_value_cuda,
)

__all__ = [
    "initialisation_gaussian_2d",
    "initialisation_gaussian_3d",
    "update_new_value",
    "update_new_value_cuda",
]
