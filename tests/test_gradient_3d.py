"""
Unit tests for the 3D gradient operator.

Same strategy as for 2D: linear functions must be reconstructed exactly
by the FVM cell-centred gradient.

For sinusoidal functions the reconstruction is approximate; we check
convergence in L2 norm against the analytical gradient.
"""
import numpy as np
import pytest

from manapy.ast import Variable

ATOL_LINEAR = 1e-6
RTOL_SMOOTH = 0.05


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


def _set_sinusoidal_3d(domain, kx, ky, kz):
    """
    f(x,y,z) = sin(kx*x) * sin(ky*y) * sin(kz*z)
    Analytical gradient:
      df/dx = kx * cos(kx*x) * sin(ky*y) * sin(kz*z)
      df/dy = ky * sin(kx*x) * cos(ky*y) * sin(kz*z)
      df/dz = kz * sin(kx*x) * sin(ky*y) * cos(kz*z)
    """
    var = Variable(domain=domain)
    c = domain.cells.center
    g = domain.faces.ghostcenter

    var.cell[:] = (np.sin(kx * c[:, 0]) * np.sin(ky * c[:, 1]) * np.sin(kz * c[:, 2]))
    var.ghost[:] = (np.sin(kx * g[:, 0]) * np.sin(ky * g[:, 1]) * np.sin(kz * g[:, 2]))
    return var


def _l2_relative_error(computed, exact):
    return np.linalg.norm(computed - exact) / (np.linalg.norm(exact) + 1e-12)


def _check_sinusoidal_gradient_3d(domain, kx, ky, kz, rtol=RTOL_SMOOTH):
    var = _set_sinusoidal_3d(domain, kx, ky, kz)
    var.compute_cell_gradient()

    c = domain.cells.center
    exact_gx = kx * np.cos(kx * c[:, 0]) * np.sin(ky * c[:, 1]) * np.sin(kz * c[:, 2])
    exact_gy = ky * np.sin(kx * c[:, 0]) * np.cos(ky * c[:, 1]) * np.sin(kz * c[:, 2])
    exact_gz = kz * np.sin(kx * c[:, 0]) * np.sin(ky * c[:, 1]) * np.cos(kz * c[:, 2])

    for grad, exact, name in [
        (var.gradcellx, exact_gx, "x"),
        (var.gradcelly, exact_gy, "y"),
        (var.gradcellz, exact_gz, "z"),
    ]:
        err = _l2_relative_error(grad, exact)
        assert err < rtol, \
            f"sin grad L2 err on {name}: {err:.3e} (kx={kx}, ky={ky}, kz={kz})"


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


# ---------------------------------------------------------------------------
# Sinusoidal gradient — cell gradient
# f(x,y,z) = sin(kx*x) * sin(ky*y) * sin(kz*z)
# ---------------------------------------------------------------------------
class TestSinusoidalGradientCube3D:

    def test_sin_x_only(self, domain_cube_3d):
        """f = sin(pi*x)  →  df/dx = pi*cos(pi*x), df/dy = df/dz = 0"""
        _check_sinusoidal_gradient_3d(domain_cube_3d, kx=np.pi, ky=0.0, kz=0.0)

    def test_sin_y_only(self, domain_cube_3d):
        """f = sin(pi*y)"""
        _check_sinusoidal_gradient_3d(domain_cube_3d, kx=0.0, ky=np.pi, kz=0.0)

    def test_sin_z_only(self, domain_cube_3d):
        """f = sin(pi*z)"""
        _check_sinusoidal_gradient_3d(domain_cube_3d, kx=0.0, ky=0.0, kz=np.pi)

    def test_sin_xyz(self, domain_cube_3d):
        """f = sin(pi*x) * sin(pi*y) * sin(pi*z)"""
        _check_sinusoidal_gradient_3d(domain_cube_3d, kx=np.pi, ky=np.pi, kz=np.pi)

    def test_sin_xyz_different_wavenumbers(self, domain_cube_3d):
        """f = sin(pi*x) * sin(2*pi*y) * sin(pi*z) — asymmetric"""
        _check_sinusoidal_gradient_3d(domain_cube_3d, kx=np.pi, ky=2 * np.pi, kz=np.pi)

    def test_sin_low_wavenumber(self, domain_cube_3d):
        """Low wavenumber: easier to resolve, expect tighter error."""
        _check_sinusoidal_gradient_3d(domain_cube_3d, kx=1.0, ky=1.0, kz=1.0, rtol=0.02)


class TestSinusoidalGradientCubeBis3D:

    def test_sin_xyz(self, domain_cube_bis_3d):
        """f = sin(pi*x) * sin(pi*y) * sin(pi*z) on alternative cube mesh."""
        _check_sinusoidal_gradient_3d(domain_cube_bis_3d, kx=np.pi, ky=np.pi, kz=np.pi)

    def test_sin_x_only(self, domain_cube_bis_3d):
        _check_sinusoidal_gradient_3d(domain_cube_bis_3d, kx=np.pi, ky=0.0, kz=0.0)

    def test_sin_low_wavenumber(self, domain_cube_bis_3d):
        _check_sinusoidal_gradient_3d(domain_cube_bis_3d, kx=1.0, ky=1.0, kz=1.0, rtol=0.02)
