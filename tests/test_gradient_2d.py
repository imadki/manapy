"""
Unit tests for the 2D gradient operator.

Strategy
--------
For a linear function f(x, y) = a*x + b*y the FVM cell-centred gradient
reconstruction is exact regardless of mesh geometry or element type.
We verify this property on three mesh types:
  - rectangle.msh    (quads)
  - carre_hybrid.msh (triangles + quads)
  - carre_structure.msh (structured quads)

We also test the face gradient (diamond scheme) for the same functions.

For sinusoidal functions the reconstruction is approximate; we check
convergence in L2 norm against the analytical gradient.

Ghost cell values are initialised from the analytical function so that
boundary cells are not penalised.
"""
import numpy as np
import pytest

from manapy.ast import Variable

# Absolute tolerance for "exact" gradient reconstruction on linear functions
ATOL_LINEAR = 1e-6

# Relative L2 tolerance for smooth (sinusoidal) functions
RTOL_SMOOTH = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_linear(domain, a, b):
    """
    Create a Variable with f(x,y) = a*x + b*y.
    Cell and boundary ghost values are initialised analytically.
    Returns the Variable ready for gradient computation.
    """
    var = Variable(domain=domain)
    c = domain.cells.center          # (ncells, >=3)
    g = domain.faces.ghostcenter     # (nfaces, >=2)

    var.cell[:] = a * c[:, 0] + b * c[:, 1]
    var.ghost[:] = a * g[:, 0] + b * g[:, 1]
    return var


def _set_sinusoidal(domain, kx, ky):
    """
    Create a Variable with f(x,y) = sin(kx*x) * sin(ky*y).
    Analytical gradient:
      df/dx = kx * cos(kx*x) * sin(ky*y)
      df/dy = ky * sin(kx*x) * cos(ky*y)
    """
    var = Variable(domain=domain)
    c = domain.cells.center
    g = domain.faces.ghostcenter

    var.cell[:] = np.sin(kx * c[:, 0]) * np.sin(ky * c[:, 1])
    var.ghost[:] = np.sin(kx * g[:, 0]) * np.sin(ky * g[:, 1])
    return var


def _l2_relative_error(computed, exact):
    return np.linalg.norm(computed - exact) / (np.linalg.norm(exact) + 1e-12)


def _check_sinusoidal_gradient(domain, kx, ky, rtol=RTOL_SMOOTH):
    """
    Compute cell gradient of sin(kx*x)*sin(ky*y) and compare with
    the analytical gradient in L2 norm.
    """
    var = _set_sinusoidal(domain, kx, ky)
    var.compute_cell_gradient()

    c = domain.cells.center
    exact_gx = kx * np.cos(kx * c[:, 0]) * np.sin(ky * c[:, 1])
    exact_gy = ky * np.sin(kx * c[:, 0]) * np.cos(ky * c[:, 1])

    err_x = _l2_relative_error(var.gradcellx, exact_gx)
    err_y = _l2_relative_error(var.gradcelly, exact_gy)

    assert err_x < rtol, f"sin grad L2 err on x: {err_x:.3e} (kx={kx}, ky={ky})"
    assert err_y < rtol, f"sin grad L2 err on y: {err_y:.3e} (kx={kx}, ky={ky})"


def _check_cell_gradient(domain, a, b, atol=ATOL_LINEAR):
    var = _set_linear(domain, a, b)
    var.compute_cell_gradient()
    assert np.allclose(var.gradcellx, a, atol=atol), \
        f"gradcellx expected {a}, got max deviation {np.max(np.abs(var.gradcellx - a)):.2e}"
    assert np.allclose(var.gradcelly, b, atol=atol), \
        f"gradcelly expected {b}, got max deviation {np.max(np.abs(var.gradcelly - b)):.2e}"


def _check_face_gradient(domain, a, b, atol=1e-4):
    """
    Face gradients use the diamond scheme and require node interpolation first.
    We accept a slightly larger tolerance than the cell gradient.
    """
    var = _set_linear(domain, a, b)
    # Interpolate to nodes before computing the face gradient
    var.interpolate_celltonode()
    var.compute_face_gradient()

    # Only check inner faces (boundary faces can differ due to ghost handling)
    inner = domain.innerfaces  # array of inner face indices
    assert np.allclose(var.gradfacex[inner], a, atol=atol), \
        f"gradfacex (inner) expected {a}, max err {np.max(np.abs(var.gradfacex[inner] - a)):.2e}"
    assert np.allclose(var.gradfacey[inner], b, atol=atol), \
        f"gradfacey (inner) expected {b}, max err {np.max(np.abs(var.gradfacey[inner] - b)):.2e}"


# ---------------------------------------------------------------------------
# Cell gradient — rectangle mesh
# ---------------------------------------------------------------------------
class TestCellGradientRectangle2D:

    def test_grad_f_equals_x(self, domain_rectangle_2d):
        """f = x  →  grad = (1, 0)"""
        _check_cell_gradient(domain_rectangle_2d, a=1.0, b=0.0)

    def test_grad_f_equals_y(self, domain_rectangle_2d):
        """f = y  →  grad = (0, 1)"""
        _check_cell_gradient(domain_rectangle_2d, a=0.0, b=1.0)

    def test_grad_f_equals_x_plus_y(self, domain_rectangle_2d):
        """f = x + y  →  grad = (1, 1)"""
        _check_cell_gradient(domain_rectangle_2d, a=1.0, b=1.0)

    def test_grad_f_general_linear(self, domain_rectangle_2d):
        """f = 3x - 2y  →  grad = (3, -2)"""
        _check_cell_gradient(domain_rectangle_2d, a=3.0, b=-2.0)

    def test_grad_constant_is_zero(self, domain_rectangle_2d):
        """f = 5 (constant)  →  grad = (0, 0)"""
        var = Variable(domain=domain_rectangle_2d)
        var.cell[:] = 5.0
        var.ghost[:] = 5.0
        var.compute_cell_gradient()
        assert np.allclose(var.gradcellx, 0.0, atol=ATOL_LINEAR)
        assert np.allclose(var.gradcelly, 0.0, atol=ATOL_LINEAR)

    def test_grad_quadratic_l2_error(self, domain_rectangle_2d):
        """
        f = x² + y²  →  exact grad = (2x, 2y).
        FVM gives an approximation; we check the L2 relative error is small.
        """
        domain = domain_rectangle_2d
        var = Variable(domain=domain)
        c = domain.cells.center
        g = domain.faces.ghostcenter

        var.cell[:] = c[:, 0] ** 2 + c[:, 1] ** 2
        var.ghost[:] = g[:, 0] ** 2 + g[:, 1] ** 2
        var.compute_cell_gradient()

        exact_gx = 2.0 * c[:, 0]
        exact_gy = 2.0 * c[:, 1]

        err_x = np.linalg.norm(var.gradcellx - exact_gx) / (np.linalg.norm(exact_gx) + 1e-12)
        err_y = np.linalg.norm(var.gradcelly - exact_gy) / (np.linalg.norm(exact_gy) + 1e-12)

        assert err_x < 0.05, f"L2 relative error on gradx too large: {err_x:.3e}"
        assert err_y < 0.05, f"L2 relative error on grady too large: {err_y:.3e}"


# ---------------------------------------------------------------------------
# Cell gradient — hybrid mesh
# ---------------------------------------------------------------------------
class TestCellGradientHybrid2D:

    def test_grad_f_equals_x(self, domain_hybrid_2d):
        _check_cell_gradient(domain_hybrid_2d, a=1.0, b=0.0)

    def test_grad_f_equals_y(self, domain_hybrid_2d):
        _check_cell_gradient(domain_hybrid_2d, a=0.0, b=1.0)

    def test_grad_f_equals_x_plus_y(self, domain_hybrid_2d):
        _check_cell_gradient(domain_hybrid_2d, a=1.0, b=1.0)

    def test_grad_f_general_linear(self, domain_hybrid_2d):
        _check_cell_gradient(domain_hybrid_2d, a=2.0, b=-3.0)

    def test_grad_constant_is_zero(self, domain_hybrid_2d):
        var = Variable(domain=domain_hybrid_2d)
        var.cell[:] = 1.0
        var.ghost[:] = 1.0
        var.compute_cell_gradient()
        assert np.allclose(var.gradcellx, 0.0, atol=ATOL_LINEAR)
        assert np.allclose(var.gradcelly, 0.0, atol=ATOL_LINEAR)

    def test_grad_quadratic_l2_error(self, domain_hybrid_2d):
        """f = x² + y²  →  exact grad = (2x, 2y), L2 err < 5%"""
        domain = domain_hybrid_2d
        var = Variable(domain=domain)
        c = domain.cells.center
        g = domain.faces.ghostcenter
        var.cell[:] = c[:, 0] ** 2 + c[:, 1] ** 2
        var.ghost[:] = g[:, 0] ** 2 + g[:, 1] ** 2
        var.compute_cell_gradient()
        err_x = _l2_relative_error(var.gradcellx, 2.0 * c[:, 0])
        err_y = _l2_relative_error(var.gradcelly, 2.0 * c[:, 1])
        assert err_x < 0.05, f"grad x L2 err: {err_x:.3e}"
        assert err_y < 0.05, f"grad y L2 err: {err_y:.3e}"


# ---------------------------------------------------------------------------
# Cell gradient — carre mesh (pure quads)
# ---------------------------------------------------------------------------
class TestCellGradientCarre2D:

    def test_grad_f_equals_x(self, domain_carre_2d):
        """f = x  →  grad = (1, 0)"""
        _check_cell_gradient(domain_carre_2d, a=1.0, b=0.0)

    def test_grad_f_equals_y(self, domain_carre_2d):
        """f = y  →  grad = (0, 1)"""
        _check_cell_gradient(domain_carre_2d, a=0.0, b=1.0)

    def test_grad_f_equals_x_plus_y(self, domain_carre_2d):
        _check_cell_gradient(domain_carre_2d, a=1.0, b=1.0)

    def test_grad_f_general_linear(self, domain_carre_2d):
        _check_cell_gradient(domain_carre_2d, a=3.0, b=-2.0)

    def test_grad_constant_is_zero(self, domain_carre_2d):
        var = Variable(domain=domain_carre_2d)
        var.cell[:] = 4.0
        var.ghost[:] = 4.0
        var.compute_cell_gradient()
        assert np.allclose(var.gradcellx, 0.0, atol=ATOL_LINEAR)
        assert np.allclose(var.gradcelly, 0.0, atol=ATOL_LINEAR)

    def test_sin_xy(self, domain_carre_2d):
        """f = sin(pi*x) * sin(pi*y) on pure quad mesh."""
        _check_sinusoidal_gradient(domain_carre_2d, kx=np.pi, ky=np.pi)

    def test_sin_x_only(self, domain_carre_2d):
        _check_sinusoidal_gradient(domain_carre_2d, kx=np.pi, ky=0.0)

    def test_sin_low_wavenumber(self, domain_carre_2d):
        _check_sinusoidal_gradient(domain_carre_2d, kx=1.0, ky=1.0, rtol=0.02)

    def test_grad_quadratic_l2_error(self, domain_carre_2d):
        """f = x² + y²  →  exact grad = (2x, 2y), L2 err < 5%"""
        domain = domain_carre_2d
        var = Variable(domain=domain)
        c = domain.cells.center
        g = domain.faces.ghostcenter
        var.cell[:] = c[:, 0] ** 2 + c[:, 1] ** 2
        var.ghost[:] = g[:, 0] ** 2 + g[:, 1] ** 2
        var.compute_cell_gradient()
        err_x = _l2_relative_error(var.gradcellx, 2.0 * c[:, 0])
        err_y = _l2_relative_error(var.gradcelly, 2.0 * c[:, 1])
        assert err_x < 0.05, f"grad x L2 err: {err_x:.3e}"
        assert err_y < 0.05, f"grad y L2 err: {err_y:.3e}"


# ---------------------------------------------------------------------------
# Cell gradient — structured mesh
# ---------------------------------------------------------------------------
class TestCellGradientStructured2D:

    def test_grad_f_equals_x(self, domain_structured_2d):
        _check_cell_gradient(domain_structured_2d, a=1.0, b=0.0)

    def test_grad_f_equals_y(self, domain_structured_2d):
        _check_cell_gradient(domain_structured_2d, a=0.0, b=1.0)

    def test_grad_f_general_linear(self, domain_structured_2d):
        _check_cell_gradient(domain_structured_2d, a=-1.0, b=4.0)

    def test_grad_quadratic_l2_error(self, domain_structured_2d):
        """f = x² + y²  →  exact grad = (2x, 2y), L2 err < 5%"""
        domain = domain_structured_2d
        var = Variable(domain=domain)
        c = domain.cells.center
        g = domain.faces.ghostcenter
        var.cell[:] = c[:, 0] ** 2 + c[:, 1] ** 2
        var.ghost[:] = g[:, 0] ** 2 + g[:, 1] ** 2
        var.compute_cell_gradient()
        err_x = _l2_relative_error(var.gradcellx, 2.0 * c[:, 0])
        err_y = _l2_relative_error(var.gradcelly, 2.0 * c[:, 1])
        assert err_x < 0.05, f"grad x L2 err: {err_x:.3e}"
        assert err_y < 0.05, f"grad y L2 err: {err_y:.3e}"


# ---------------------------------------------------------------------------
# Face gradient (diamond scheme) — rectangle mesh
# ---------------------------------------------------------------------------
class TestFaceGradientRectangle2D:

    def test_face_grad_f_equals_x(self, domain_rectangle_2d):
        _check_face_gradient(domain_rectangle_2d, a=1.0, b=0.0)

    def test_face_grad_f_equals_y(self, domain_rectangle_2d):
        _check_face_gradient(domain_rectangle_2d, a=0.0, b=1.0)

    def test_face_grad_f_equals_x_plus_y(self, domain_rectangle_2d):
        _check_face_gradient(domain_rectangle_2d, a=1.0, b=1.0)


# ---------------------------------------------------------------------------
# Sinusoidal gradient — cell gradient
# f(x,y) = sin(kx*x) * sin(ky*y)
# df/dx  = kx * cos(kx*x) * sin(ky*y)
# df/dy  = ky * sin(kx*x) * cos(ky*y)
# ---------------------------------------------------------------------------
class TestSinusoidalGradientRectangle2D:

    def test_sin_x_only(self, domain_rectangle_2d):
        """f = sin(pi*x)  →  df/dx = pi*cos(pi*x), df/dy = 0"""
        _check_sinusoidal_gradient(domain_rectangle_2d, kx=np.pi, ky=0.0)

    def test_sin_y_only(self, domain_rectangle_2d):
        """f = sin(pi*y)  →  df/dx = 0, df/dy = pi*cos(pi*y)"""
        _check_sinusoidal_gradient(domain_rectangle_2d, kx=0.0, ky=np.pi)

    def test_sin_xy(self, domain_rectangle_2d):
        """f = sin(pi*x) * sin(pi*y)"""
        _check_sinusoidal_gradient(domain_rectangle_2d, kx=np.pi, ky=np.pi)

    def test_sin_xy_different_wavenumbers(self, domain_rectangle_2d):
        """f = sin(2*pi*x) * sin(pi*y)  — asymmetric wavenumbers"""
        _check_sinusoidal_gradient(domain_rectangle_2d, kx=2 * np.pi, ky=np.pi)

    def test_sin_low_wavenumber(self, domain_rectangle_2d):
        """Low wavenumber: easier to resolve, expect tighter error."""
        _check_sinusoidal_gradient(domain_rectangle_2d, kx=1.0, ky=1.0, rtol=0.02)


class TestSinusoidalGradientHybrid2D:

    def test_sin_xy(self, domain_hybrid_2d):
        """f = sin(pi*x) * sin(pi*y) on hybrid mesh."""
        _check_sinusoidal_gradient(domain_hybrid_2d, kx=np.pi, ky=np.pi)

    def test_sin_x_only(self, domain_hybrid_2d):
        _check_sinusoidal_gradient(domain_hybrid_2d, kx=np.pi, ky=0.0)

    def test_sin_y_only(self, domain_hybrid_2d):
        _check_sinusoidal_gradient(domain_hybrid_2d, kx=0.0, ky=np.pi)


class TestSinusoidalGradientStructured2D:

    def test_sin_xy(self, domain_structured_2d):
        """f = sin(pi*x) * sin(pi*y) on structured mesh."""
        _check_sinusoidal_gradient(domain_structured_2d, kx=np.pi, ky=np.pi)

    def test_sin_low_wavenumber(self, domain_structured_2d):
        """Structured mesh is regular: expect better accuracy at low wavenumber."""
        _check_sinusoidal_gradient(domain_structured_2d, kx=1.0, ky=1.0, rtol=0.02)
