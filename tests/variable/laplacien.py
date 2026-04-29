"""
Unit tests for the Laplacian operator (Δu = 0, diamond scheme).

These tests solve a stationary diffusion problem with Dirichlet boundary
conditions and compare the numerical solution against an analytical one.

Test functions:
  - u(x,y) = x           →  Δu = 0, BCs: u = x on all boundaries
  - u(x,y) = x + y       →  Δu = 0
  - u(x,y) = x² - y²     →  Δu = 0  (harmonic)

MUMPS is the default sparse direct solver.  The test is skipped if it is
not installed.
"""
import numpy as np
import pytest
from manapy.helpers import get_test_mesh, get_mesh
from manapy.core import Variable
from manapy.domain import Domain
import sys
import os
from manapy.solvers.ls import MUMPSSolver

def _solve_laplacian_2d(domain, func):
    """
    Build BC dicts for a 2D domain where every boundary is Dirichlet with
    value given by func(x, y, z).

    Solve Δu = 0 with Dirichlet BCs given by func(x, y).
    Returns the Variable after solving.
    """
    boundaries = {loc: "dirichlet" for loc in ("in", "out", "upper", "bottom")}
    values = {loc: lambda x, y, z: func(x, y) for loc in ("in", "out", "upper", "bottom")}

    var = Variable(domain=domain, BC=boundaries, values_dict=values)
    solver = MUMPSSolver(domain=domain, var=var, reuse_mtx=False, scheme="diamond")
    solver()
    var.update_halo_value()
    var.update_ghost_value()
    return var

def error_metrics(values, exact, epsilon=1e-12):
  values = np.asarray(values)
  exact = np.asarray(exact)

  abs_err = np.abs(values - exact)

  # Avoid division by zero
  rel_err = abs_err / (np.abs(exact) + epsilon)

  metrics = {
    "sum_abs_error": np.sum(abs_err),
    "avg_abs_error": np.mean(abs_err),
    "sum_rel_error": np.sum(rel_err),
    "avg_rel_error": np.mean(rel_err),
  }

  return metrics

def _get_functions():
  functions = {
    "linear_x": {
      "f": lambda x, y: 2 * x + 5,
      "expr": "2x + 5"
    },
    "linear_y": {
      "f": lambda x, y: -3 * y + 1,
      "expr": "-3y + 1"
    },
    "bilinear": {
      "f": lambda x, y: x * y,
      "expr": "xy"
    },
    "quadratic_harmonic": {
      "f": lambda x, y: x ** 2 - y ** 2,
      "expr": "x^2 - y^2"
    },
    "cubic_harmonic": {
      "f": lambda x, y: x ** 3 - 3 * x * y ** 2,
      "expr": "x^3 - 3xy^2"
    },
    "trig_exp": {
      "f": lambda x, y: np.exp(x) * np.cos(y),
      "expr": "e^x cos(y)"
    },
    "trig_exp_2": {
      "f": lambda x, y: np.exp(x) * np.sin(y),
      "expr": "e^x sin(y)"
    },
    # "logarithmic": {
    #   "f": lambda x, y: np.log(x ** 2 + y ** 2),
    #   "expr": "log(x^2 + y^2)  (undefined at (0,0))"
    # },
    # "arctan": {
    #   "f": lambda x, y: np.atan2(y, x),
    #   "expr": "arctan(y/x) (multi-valued / branch cut)"
    # }
  }
  return functions

def main():
  # Domain
  for fun_name in _get_functions():
    meshes = ["carre.msh", "hybrid.msh", "rectangles2d.msh", "triangles2d.msh"]
    for mesh in meshes:
      dim, mesh_path, mesh_name = get_mesh(f"big/var/{mesh}", 2)
      old_stdout = sys.stdout
      sys.stdout = open(os.devnull, 'w')
      domain = Domain.create_domain(mesh_path, dim, Domain.PartitioningClass.Par_Nodal)
      sys.stdout.close()
      sys.stdout = old_stdout

      # Functions
      function = _get_functions()[fun_name]
      fun = function["f"]
      fun_expr = f"fun={function['expr']}"

      # Variable
      var = _solve_laplacian_2d(domain, fun)
      u = var.cell

      # Reference
      exact_u = fun(domain.cells.center[:, 0], domain.cells.center[:, 1])

      metrics = error_metrics(u, exact_u)
      print(fun_expr)
      try:
        np.testing.assert_allclose(u, exact_u, rtol=1e-3, atol=1e-3)
        print(f'{mesh}: nb_cells={domain.nbcells}, Ok')
        print(metrics)
      except AssertionError as e:
        print(f'{mesh}: nb_cells={domain.nbcells}, Not Ok')
        print(metrics)
        print(str(e))
      print("----------------------------------------------------------------\n")
    print("\n\n\n\n")


if __name__ == "__main__":
  main()
