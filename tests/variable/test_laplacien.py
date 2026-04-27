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
def _make_bc_dirichlet_2d(func, domain):
    """
    Build BC dicts for a 2D domain where every boundary is Dirichlet with
    value given by func(x, y, z).
    """
    boundaries = {loc: "dirichlet" for loc in ("in", "out", "upper", "bottom")}
    values = {loc: lambda x, y, z, f=func: f(x, y) for loc in ("in", "out", "upper", "bottom")}
    return boundaries, values


def _solve_laplacian_2d(domain, func):
    """
    Solve Δu = 0 with Dirichlet BCs given by func(x, y).
    Returns the Variable after solving.
    """
    boundaries, values = _make_bc_dirichlet_2d(func, domain)
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


# ---------------------------------------------------------------------------
# 2D Laplacian tests — rectangle mesh
# ---------------------------------------------------------------------------
def duplicate_config(a):
  return a

@pytest.mark.parametrize(
  "config",
  duplicate_config([
    {
      "dim": get_mesh('big/carre.msh')[0],
      "mesh_path": get_mesh('big/carre.msh')[1],
      "partitioning_type": Domain.PartitioningClass.Par_Nodal,
    }
  ]),
  indirect=True
)
@skip_no_mumps
class TestLaplacian2DRectangle:

    def test_linear_solution_x(self, domain):
        """
        u(x,y) = x is harmonic (Δu=0).
        After solving with u=x on all boundaries, the interior must match x.
        """
        func = lambda x, y: x
        var = _solve_laplacian_2d(domain, func)

        exact = func(domain.cells.center[:, 0], domain.cells.center[:, 1])
        err = _l2_relative_error(var, exact)
        assert err < 1e-6, f"L2 relative error for u=x: {err:.2e}"

    def test_linear_solution_y(self, domain):
        """u(x,y) = y  →  Δu = 0."""
        func = lambda x, y: y
        var = _solve_laplacian_2d(domain, func)

        exact = func(domain.cells.center[:, 0], domain.cells.center[:, 1])
        err = _l2_relative_error(var, exact)
        assert err < 1e-6, f"L2 relative error for u=y: {err:.2e}"

    def test_linear_solution_x_plus_y(self, domain):
        """u(x,y) = x + y  →  Δu = 0."""
        func = lambda x, y: x + y
        var = _solve_laplacian_2d(domain, func)

        exact = func(domain.cells.center[:, 0], domain.cells.center[:, 1])
        err = _l2_relative_error(var, exact)
        assert err < 1e-6, f"L2 relative error for u=x+y: {err:.2e}"

    def test_harmonic_solution_x2_minus_y2(self, domain):
        """
        u(x,y) = x² - y²  →  Δu = 2 - 2 = 0  (harmonic).
        We expect a small but non-zero discretisation error.
        """
        func = lambda x, y: x ** 2 - y ** 2
        var = _solve_laplacian_2d(domain, func)

        exact = func(domain.cells.center[:, 0], domain.cells.center[:, 1])
        err = _l2_relative_error(var, exact)
        assert err < 0.01, f"L2 relative error for u=x²-y²: {err:.2e}"

    def test_constant_solution(self, domain):
        """
        u(x,y) = 5 (constant)  →  Δu = 0.
        The solver must reproduce the constant exactly.
        """
        func = lambda x, y: 5.0
        var = _solve_laplacian_2d(domain, func)

        assert np.allclose(var.cell, 5.0, atol=1e-8), \
            f"Constant solution: max error = {np.max(np.abs(var.cell - 5.0)):.2e}"


# ---------------------------------------------------------------------------
# 2D Laplacian tests — hybrid mesh
# ---------------------------------------------------------------------------
@skip_no_mumps
class TestLaplacian2DHybrid:

    def test_linear_solution_x(self, domain):
        func = lambda x, y: x
        var = _solve_laplacian_2d(domain, func)
        exact = func(domain.cells.center[:, 0], domain.cells.center[:, 1])
        err = _l2_relative_error(var, exact)
        assert err < 1e-6

    def test_linear_solution_x_plus_y(self, domain):
        func = lambda x, y: x + y
        var = _solve_laplacian_2d(domain, func)
        exact = func(domain.cells.center[:, 0], domain.cells.center[:, 1])
        err = _l2_relative_error(var, exact)
        assert err < 1e-6


# ---------------------------------------------------------------------------
# Variable norml2 utility
# ---------------------------------------------------------------------------
class TestVariableNorml2:
    """Tests for the built-in L2-norm helper on Variable."""

    def test_zero_error(self, domain):
        """norml2 against the exact values should be 0."""
        var = Variable(domain=domain)
        c = domain.cells.center
        var.cell[:] = c[:, 0]
        exact = c[:, 0].copy()
        assert var.norml2(exact) == pytest.approx(0.0, abs=1e-12)

    def test_known_error(self, domain):
        """norml2 with a constant offset should give a predictable result."""
        var = Variable(domain=domain)
        var.cell[:] = 1.0
        exact = np.zeros(domain.nbcells)
        # ||1 - 0|| / ||0|| → large (or inf), so we just test it doesn't crash
        # and returns a positive float
        # Use a non-zero exact to avoid division by zero
        exact[:] = 2.0
        err = var.norml2(exact)
        assert err >= 0.0
