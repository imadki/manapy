from manapy.core import Variable
from manapy.domain import Domain
from manapy.helpers import get_test_mesh, get_mesh
from manapy.testing.test_domain_helper import make_test_config
import numpy as np
import pytest

# jax is only needed here to build autodiff reference gradients. It is installed
# best-effort in the CI image; if its wheel is unavailable (e.g. very recent
# Python), skip this module cleanly rather than fail collection.
pytest.importorskip("jax")
import jax.numpy as jnp
from jax import grad, vmap

# Absolute tolerance for "exact" gradient reconstruction on linear functions
ATOL_LINEAR = 1e-6

# Relative L2 tolerance for smooth (sinusoidal) functions
RTOL_SMOOTH = 0.05

def _reference(values, fun):
  grad_x = grad(fun, argnums=0)
  grad_y = grad(fun, argnums=1)

  x = jnp.array(values[:, 0])
  y = jnp.array(values[:, 1])

  g_x = vmap(grad_x)(x, y)
  g_y = vmap(grad_y)(x, y)
  return g_x, g_y



def _get_var(domain: Domain, fun):
  var = Variable(domain=domain)
  c = domain.cells.center
  ghost_center = domain.ghost.info_flt[:, 0:2]
  g = np.zeros(shape=(domain.nbfaces, 2), dtype=ghost_center.dtype)
  g[domain.ghost.faceid] = ghost_center[:]
  h = domain.halos.centvol[:, 0:2]
  hg = domain.ghost.ext_info_flt[:, 0:2]

  var.cell[:] = fun(c[:, 0], c[:, 1])
  var.ghost[:] = fun(g[:, 0], g[:, 1])
  if domain.size > 1:
    var.halo[:] = fun(h[:, 0], h[:, 1])
    var.haloghost[:] = fun(hg[:, 0], hg[:, 1])
  var.interpolate_celltonode() # for face gradient
  return var



def _set_linear(domain: Domain, a, b, p):
  """
  Create a Variable with f(x,y) = a*x + b*y + p.
    df/dx = a
    df/dy = b
  Cell and boundary ghost values are initialised analytically.
  Returns the Variable ready for gradient computation.
  """
  var = Variable(domain=domain)
  c = domain.cells.center
  ghost_center = domain.ghost.info_flt[:, 0:3]
  g = np.zeros(shape=(domain.nbfaces, 3), dtype=ghost_center.dtype)
  g[domain.ghost.faceid] = ghost_center[:]

  var.cell[:] = a * c[:, 0] + b * c[:, 1] + p
  var.ghost[:] = a * g[:, 0] + b * g[:, 1] + p
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
  ghost_center = domain.ghost.info_flt[:, 0:3]
  g = np.zeros(shape=(domain.nbfaces, 3), dtype=ghost_center.dtype)
  g[domain.ghost.faceid] = ghost_center[:]

  var.cell[:] = np.sin(kx * c[:, 0]) * np.sin(ky * c[:, 1])
  var.ghost[:] = np.sin(kx * g[:, 0]) * np.sin(ky * g[:, 1])
  return var

def _l2_relative_error(computed, exact):
  return np.linalg.norm(computed - exact) / (np.linalg.norm(exact) + 1e-12)

def _check_cell_gradient(domain, fun, atol=ATOL_LINEAR):
  var = _get_var(domain, fun)
  var.compute_cell_gradient()

  grad_cell_x = var.gradcellx
  grad_cell_y = var.gradcelly
  # Analytic gradient (via jax autodiff) evaluated at the cell centres, not at
  # the field values: _reference expects (x, y) coordinates.
  x, y = _reference(domain.cells.center, fun)

  np.testing.assert_almost_equal(grad_cell_x, x, decimal=4)
  np.testing.assert_almost_equal(grad_cell_y, y, decimal=4)
  # assert np.allclose(var.gradcellx, a, atol=atol), \
  #   f"gradcellx expected {a}, got max deviation {np.max(np.abs(var.gradcellx - a)):.2e}"
  # assert np.allclose(var.gradcelly, b, atol=atol), \
  #   f"gradcelly expected {b}, got max deviation {np.max(np.abs(var.gradcelly - b)):.2e}"

def _check_sinusoidal_gradient(domain, kx, ky, rtol=RTOL_SMOOTH):
  """
  Compute cell gradient of sin(kx*x)*sin(ky*y) and compare with
  the analytical gradient in L2 norm.
  """
  var = _set_sinusoidal(domain, kx, ky)
  var.compute_cell_gradient()
  print("var.ghost", var.gradcellx)

  c = domain.cells.center
  exact_gx = kx * np.cos(kx * c[:, 0]) * np.sin(ky * c[:, 1])
  exact_gy = ky * np.sin(kx * c[:, 0]) * np.cos(ky * c[:, 1])
  print("=>", exact_gx)

  err_x = _l2_relative_error(var.gradcellx, exact_gx)
  err_y = _l2_relative_error(var.gradcelly, exact_gy)

  assert err_x < rtol, f"sin grad L2 err on x: {err_x:.3e} (kx={kx}, ky={ky})"
  assert err_y < rtol, f"sin grad L2 err on y: {err_y:.3e} (kx={kx}, ky={ky})"

def _check_face_gradient(domain, a, b, p, atol=1e-4):
  """
  Face gradients use the diamond scheme and require node interpolation first.
  We accept a slightly larger tolerance than the cell gradient.
  """
  var = _set_linear(domain, a, b, p)
  # Interpolate to nodes before computing the face gradient
  var.interpolate_celltonode()
  var.compute_face_gradient()

  # Only check inner faces (boundary faces can differ due to ghost handling)
  inner = domain.innerfaces  # array of inner face indices
  np.testing.assert_allclose(var.gradfacex[inner], a, atol=atol)
  np.testing.assert_allclose(var.gradfacey[inner], b, atol=atol)
  # assert np.allclose(var.gradfacex[inner], a, atol=atol), \
  #   f"gradfacex (inner) expected {a}, max err {np.max(np.abs(var.gradfacex[inner] - a)):.2e}"
  # assert np.allclose(var.gradfacey[inner], b, atol=atol), \
  #   f"gradfacey (inner) expected {b}, max err {np.max(np.abs(var.gradfacey[inner] - b)):.2e}"

def duplicate_config(a):
  return a

@pytest.mark.parametrize(
  "config",
  duplicate_config([
    {
      "dim": get_test_mesh('carre.msh')[0],
      "mesh_path": get_test_mesh('carre.msh')[1],
      "partitioning_type": Domain.PartitioningClass.Par_Nodal,
    }
  ]),
  indirect=True
)
class TestGradient:
  def test_linear(self, domain):
    """
    grad(a*x + b*y + p) = (a, b)
    """
    # _check_cell_gradient takes the field function fun(x, y), not (a, b, p).
    for a, b, p in [(0, 1, 0), (1, 0, 0), (0, 0, 0), (0, 0, 6), (3, -1, 0), (7, -10, 0)]:
      _check_cell_gradient(domain, lambda x, y, a=a, b=b, p=p: a * x + b * y + p)

  # Only work with carre.msh, does not work with kx=2*np.pi
  def test_sinusoidal(self, domain):
    """
    f(x,y) = sin(kx*x) * sin(ky*y).
    df/dx = kx * cos(kx*x) * sin(ky*y)
    df/dy = ky * sin(kx*x) * cos(ky*y)
    grad(f) = kx * cos(kx*x) * sin(ky*y) + ky * sin(kx*x) * cos(ky*y)
    """
    _check_sinusoidal_gradient(domain, kx=np.pi, ky=np.pi)
    _check_sinusoidal_gradient(domain, kx=np.pi, ky=0.0)
    _check_sinusoidal_gradient(domain, kx=1.0, ky=1.0, rtol=0.02)
    _check_sinusoidal_gradient(domain, kx=0.0, ky=np.pi)
    # _check_sinusoidal_gradient(domain, kx=2 * np.pi, ky=np.pi)



  def test_face_gradient(self, domain):
    _check_face_gradient(domain, a=0, b=1, p=0)
    _check_face_gradient(domain, a=1, b=0, p=0)
    _check_face_gradient(domain, a=0, b=0, p=0)
    _check_face_gradient(domain, a=0, b=0, p=6)
    _check_face_gradient(domain, a=3, b=-1, p=0)
    _check_face_gradient(domain, a=7, b=-10, p=0)
    

  def test_grad_quadratic_l2_error(self, domain):
    """f = x² + y²  →  exact grad = (2x, 2y), L2 err < 5%"""
    var = Variable(domain=domain)
    c = domain.cells.center
    ghost_center = domain.ghost.info_flt[:, 0:3]
    g = np.zeros(shape=(domain.nbfaces, 3), dtype=ghost_center.dtype)
    g[domain.ghost.faceid] = ghost_center[:]

    var.cell[:] = c[:, 0] ** 2 + c[:, 1] ** 2
    var.ghost[:] = g[:, 0] ** 2 + g[:, 1] ** 2
    var.compute_cell_gradient()
    err_x = _l2_relative_error(var.gradcellx, 2.0 * c[:, 0])
    err_y = _l2_relative_error(var.gradcelly, 2.0 * c[:, 1])
    assert err_x < 0.05, f"grad x L2 err: {err_x:.3e}"
    assert err_y < 0.05, f"grad y L2 err: {err_y:.3e}"

def main():
  dim, mesh_path, mesh_name = get_mesh("big/carre.msh")
  # dim, mesh_path, mesh_name = get_mesh("rectangles.msh")
  # dim, mesh_path, mesh_name = get_mesh("triangles.msh")
  domain = Domain.create_domain(mesh_path, dim, make_test_config(), Domain.PartitioningClass.Par_Nodal)


  fun = lambda x, y: x + y
  var = _get_var(domain, fun)
  var.compute_cell_gradient()
  var.compute_face_gradient()

  grad_cell_x = var.gradcellx
  grad_cell_y = var.gradcelly
  grad_halocell_x = var.gradhalocellx
  grad_halocell_y = var.gradhalocelly
  grad_face_x = var.gradfacex
  grad_face_y = var.gradfacey
  cx, cy = _reference(domain.cells.center, fun)
  hcx, hcy = _reference(domain.halos.centvol[:, 0:2], fun)
  fx, fy = _reference(domain.faces.center, fun)

  # np.testing.assert_almost_equal(grad_cell_x, cx, decimal=4)
  # np.testing.assert_almost_equal(grad_cell_y, cy, decimal=4)

  # np.testing.assert_almost_equal(grad_face_x, fx, decimal=4)
  # np.testing.assert_almost_equal(grad_face_y, fy, decimal=4)

  # np.testing.assert_almost_equal(grad_halocell_x, hcx)
  # np.testing.assert_almost_equal(grad_halocell_x, hcx)

  # _check_face_gradient(domain, a=0, b=1, p=0)
  # _check_face_gradient(domain, a=1, b=0, p=0)
  # _check_face_gradient(domain, a=0, b=0, p=0)
  # _check_face_gradient(domain, a=0, b=0, p=6)
  # _check_face_gradient(domain, a=3, b=-1, p=0)
  # _check_face_gradient(domain, a=7, b=-10, p=0)
  # _check_sinusoidal_gradient(domain, kx=2*np.pi, ky=np.pi)

if __name__ == "__main__":
  main()