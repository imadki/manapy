from manapy.core import Variable
from manapy.domain import Domain
from manapy.helpers import get_mesh
import jax.numpy as jnp
from jax import grad, vmap
import numpy as np


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
  var.interpolate_celltonode()  # for face gradient
  return var

def main():
  # You can add mesh using absolute path with dimension
  # You can add mesh using relative path from mesh folder at root of the project with dimension of not listed at get_mesh
  dim, mesh_path, mesh_name = get_mesh("big/carre.msh", 2) # OK
  dim, mesh_path, mesh_name = get_mesh("hybrid2d.msh", 2) # Not OK
  # dim, mesh_path, mesh_name = get_mesh("rectangles.msh", 2) # Not OK
  # dim, mesh_path, mesh_name = get_mesh("triangles.msh", 2) # Not Ok
  domain = Domain.create_domain(mesh_path, dim, Domain.PartitioningClass.Par_Nodal)

  fun = lambda x, y: x + y # Ok
  fun = lambda x, y: 2*x + 0*y + 5 # Ok
  fun = lambda x, y: 0*x + 2*y + 3 # Ok
  fun = lambda x, y: 2*x + 2*y + 3 # Ok
  fun = lambda x, y: 0*x + 0*y + 8 # Ok
  fun = lambda x, y: x ** 2 + y ** 2 # Ok with rtol=1e-1, atol=1e-1
  fun = lambda x, y: jnp.sin(x) * jnp.sin(y) # Ok with rtol=1e-1, atol=1e-1
  fun = lambda x, y: jnp.sin(3*x) * jnp.sin(2*y) # Ok with rtol=1e-1, atol=1e-1
  var = _get_var(domain, fun)
  var.compute_cell_gradient()
  var.compute_face_gradient()

  # Tested attributes
  grad_cell_x = var.gradcellx
  grad_cell_y = var.gradcelly
  grad_halocell_x = var.gradhalocellx
  grad_halocell_y = var.gradhalocelly
  grad_face_x = var.gradfacex
  grad_face_y = var.gradfacey

  # Reference
  cx, cy = _reference(domain.cells.center, fun)
  hcx, hcy = _reference(domain.halos.centvol[:, 0:2], fun)
  fx, fy = _reference(domain.faces.center, fun)

  # Check cell gradient
  np.testing.assert_allclose(grad_cell_x, cx, rtol=1e-1, atol=1e-1)
  np.testing.assert_allclose(grad_cell_y, cy, rtol=1e-1, atol=1e-1)

  # Check face gradient
  np.testing.assert_allclose(grad_face_x, fx, rtol=1e-1, atol=1e-1)
  np.testing.assert_allclose(grad_face_y, fy, rtol=1e-1, atol=1e-1)

  # Check halo cell gradient (MPI communication)
  # This test is only to test communication. the actual values are already tests in cell gradient
  np.testing.assert_allclose(grad_halocell_x, hcx, rtol=1e-1, atol=1e-1)
  np.testing.assert_allclose(grad_halocell_y, hcy, rtol=1e-1, atol=1e-1)



if __name__ == "__main__":
  main()