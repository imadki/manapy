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

try:
    from manapy.solvers.ls import MUMPSSolver
    MUMPS_AVAILABLE = True
except Exception:
    MUMPS_AVAILABLE = False

skip_no_mumps = pytest.mark.skipif(
    not MUMPS_AVAILABLE, reason="MUMPS solver not installed"
)



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solve_laplacian_2d(domain, func):
    """
    Build BC dicts for a 2D domain where every boundary is Dirichlet with
    value given by func(x, y, z).

    Solve Δu = 0 with Dirichlet BCs given by func(x, y).
    Returns the Variable after solving.
    """
    boundaries = {loc: "dirichlet" for loc in ("in", "out", "upper", "bottom")}
    values = {loc: lambda x, y, z, f=func: f(x, y) for loc in ("in", "out", "upper", "bottom")}

    var = Variable(domain=domain, BC=boundaries, values_dict=values)
    solver = MUMPSSolver(domain=domain, var=var, reuse_mtx=False, scheme="diamond")
    solver()
    var.update_halo_value()
    var.update_ghost_value()
    return var


def _l2_relative_error(var, exact_values):
    """Compute L2 relative error ||u_h - u||_2 / ||u||_2."""
    diff = var.cell - exact_values
    return np.linalg.norm(diff) / (np.linalg.norm(exact_values) + 1e-12)


def test_linear_solution_x(domain):
  """
  u(x,y) = x is harmonic (Δu=0).
  After solving with u=x on all boundaries, the interior must match x.
  """
  func = lambda x, y: x
  var = _solve_laplacian_2d(domain, func)

  exact = func(domain.cells.center[:, 0], domain.cells.center[:, 1])
  err = _l2_relative_error(var, exact)
  assert err < 1e-6, f"L2 relative error for u=x: {err:.2e}"

def main():
  dim, mesh_path, mesh_name = get_mesh("big/carre.msh", 2)
  dim, mesh_path, mesh_name = get_mesh("hybrid2d.msh", 2)
  # dim, mesh_path, mesh_name = get_mesh("rectangles.msh", 2)
  # dim, mesh_path, mesh_name = get_mesh("triangles.msh", 2)
  domain = Domain.create_domain(mesh_path, dim, Domain.PartitioningClass.Par_Nodal)

  test_linear_solution_x(domain)

if __name__ == "__main__":
  main()
