"""
Unit tests for the 3D gradient operator.

Same strategy as for 2D: linear functions must be reconstructed exactly
by the FVM cell-centred gradient.
"""
import numpy as np
import pytest

from manapy.ast import Variable

ATOL_LINEAR = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_linear_3d(domain, a, b, c_coef):
    """f(x,y,z) = a*x + b*y + c*z"""
    var = Variable(domain=domain)
    cen = domain.cells.center        # (ncells, 3)
    ghost = domain.faces.ghostcenter  # (nfaces, >=3)

    var.cell[:] = a * cen[:, 0] + b * cen[:, 1] + c_coef * cen[:, 2]
    var.ghost[:] = a * ghost[:, 0] + b * ghost[:, 1] + c_coef * ghost[:, 2]
    return var


def _check_cell_gradient_3d(domain, a, b, c_coef, atol=ATOL_LINEAR):
    var = _set_linear_3d(domain, a, b, c_coef)
    var.compute_cell_gradient()

    assert np.allclose(var.gradcellx, a, atol=atol), \
        f"gradcellx expected {a}, max err {np.max(np.abs(var.gradcellx - a)):.2e}"
    assert np.allclose(var.gradcelly, b, atol=atol), \
        f"gradcelly expected {b}, max err {np.max(np.abs(var.gradcelly - b)):.2e}"
    assert np.allclose(var.gradcellz, c_coef, atol=atol), \
        f"gradcellz expected {c_coef}, max err {np.max(np.abs(var.gradcellz - c_coef)):.2e}"


# ---------------------------------------------------------------------------
# Cell gradient — cube mesh (hexahedra)
# ---------------------------------------------------------------------------
class TestCellGradientCube3D:

    def test_grad_f_equals_x(self, domain_cube_3d):
        """f = x  →  grad = (1, 0, 0)"""
        _check_cell_gradient_3d(domain_cube_3d, a=1.0, b=0.0, c_coef=0.0)

    def test_grad_f_equals_y(self, domain_cube_3d):
        """f = y  →  grad = (0, 1, 0)"""
        _check_cell_gradient_3d(domain_cube_3d, a=0.0, b=1.0, c_coef=0.0)

    def test_grad_f_equals_z(self, domain_cube_3d):
        """f = z  →  grad = (0, 0, 1)"""
        _check_cell_gradient_3d(domain_cube_3d, a=0.0, b=0.0, c_coef=1.0)

    def test_grad_f_equals_x_plus_y_plus_z(self, domain_cube_3d):
        """f = x + y + z  →  grad = (1, 1, 1)"""
        _check_cell_gradient_3d(domain_cube_3d, a=1.0, b=1.0, c_coef=1.0)

    def test_grad_f_general_linear(self, domain_cube_3d):
        """f = 2x - y + 3z  →  grad = (2, -1, 3)"""
        _check_cell_gradient_3d(domain_cube_3d, a=2.0, b=-1.0, c_coef=3.0)

    def test_grad_constant_is_zero(self, domain_cube_3d):
        """f = 7 (constant)  →  grad = (0, 0, 0)"""
        var = Variable(domain=domain_cube_3d)
        var.cell[:] = 7.0
        var.ghost[:] = 7.0
        var.compute_cell_gradient()
        assert np.allclose(var.gradcellx, 0.0, atol=ATOL_LINEAR)
        assert np.allclose(var.gradcelly, 0.0, atol=ATOL_LINEAR)
        assert np.allclose(var.gradcellz, 0.0, atol=ATOL_LINEAR)

    def test_grad_quadratic_l2_error(self, domain_cube_3d):
        """
        f = x² + y² + z²  →  exact grad = (2x, 2y, 2z).
        FVM gives an approximation; we check the L2 relative error is small.
        """
        domain = domain_cube_3d
        var = Variable(domain=domain)
        c = domain.cells.center
        g = domain.faces.ghostcenter

        var.cell[:] = c[:, 0] ** 2 + c[:, 1] ** 2 + c[:, 2] ** 2
        var.ghost[:] = g[:, 0] ** 2 + g[:, 1] ** 2 + g[:, 2] ** 2
        var.compute_cell_gradient()

        exact_gx = 2.0 * c[:, 0]
        exact_gy = 2.0 * c[:, 1]
        exact_gz = 2.0 * c[:, 2]

        for grad, exact, name in [
            (var.gradcellx, exact_gx, "x"),
            (var.gradcelly, exact_gy, "y"),
            (var.gradcellz, exact_gz, "z"),
        ]:
            err = np.linalg.norm(grad - exact) / (np.linalg.norm(exact) + 1e-12)
            assert err < 0.05, f"L2 relative error on grad{name} too large: {err:.3e}"


# ---------------------------------------------------------------------------
# Cell gradient — alternative cube mesh
# ---------------------------------------------------------------------------
class TestCellGradientCubeBis3D:

    def test_grad_f_equals_x(self, domain_cube_bis_3d):
        _check_cell_gradient_3d(domain_cube_bis_3d, a=1.0, b=0.0, c_coef=0.0)

    def test_grad_f_equals_y(self, domain_cube_bis_3d):
        _check_cell_gradient_3d(domain_cube_bis_3d, a=0.0, b=1.0, c_coef=0.0)

    def test_grad_f_equals_z(self, domain_cube_bis_3d):
        _check_cell_gradient_3d(domain_cube_bis_3d, a=0.0, b=0.0, c_coef=1.0)

    def test_grad_f_general_linear(self, domain_cube_bis_3d):
        _check_cell_gradient_3d(domain_cube_bis_3d, a=1.0, b=-2.0, c_coef=0.5)
