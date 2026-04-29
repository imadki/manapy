from manapy.core import Variable
from manapy.domain import Domain
from manapy.helpers import get_mesh
import numpy as np
from manapy.backends import FLOAT_TYPE
import sys
import os

def _reference(values, div_fun_x, div_fun_y):
  x = values[:, 0]
  y = values[:, 1]

  if x.shape[0] == 0:
    return np.zeros_like(x), np.zeros_like(y)
  g_x = np.vectorize(div_fun_x)(x, y).astype(FLOAT_TYPE)
  g_y = np.vectorize(div_fun_y)(x, y).astype(FLOAT_TYPE)
  return g_x, g_y


def _get_var(domain: Domain, fun):
  var = Variable(domain=domain)

  # Cell center
  c = domain.cells.center

  # Ghost center
  ghost_center = domain.ghost.info_flt[:, 0:2]
  g = np.zeros(shape=(domain.nbfaces, 2), dtype=ghost_center.dtype)
  g[domain.ghost.faceid] = ghost_center[:]

  # Haloghost center
  h = domain.halos.centvol[:, 0:2]
  hg = domain.ghost.ext_info_flt[:, 0:2]

  var.cell[:] = fun(c[:, 0], c[:, 1])
  var.ghost[:] = fun(g[:, 0], g[:, 1])
  if domain.size > 1:
    var.halo[:] = fun(h[:, 0], h[:, 1])
    var.haloghost[:] = fun(hg[:, 0], hg[:, 1])
  var.interpolate_celltonode()  # for face gradient
  return var

def _get_functions():
  functions = {
    "linear_x": {
      "f": lambda x, y: 2 * x + 5,
      "df_dx": lambda x, y: 2,
      "df_dy": lambda x, y: 0,
      "expr": "2x + 5"
    },
    "linear_y": {
      "f": lambda x, y: 2 * y + 3,
      "df_dx": lambda x, y: 0,
      "df_dy": lambda x, y: 2,
      "expr": "2y + 3"
    },
    "linear_xy": {
      "f": lambda x, y: 2 * x + 2 * y + 3,
      "df_dx": lambda x, y: 2,
      "df_dy": lambda x, y: 2,
      "expr": "2x + 2y + 3"
    },
    "constant": {
      "f": lambda x, y: 8,
      "df_dx": lambda x, y: 0,
      "df_dy": lambda x, y: 0,
      "expr": "8"
    },
    "quadratic": {
      "f": lambda x, y: x ** 2 + y ** 2,
      "df_dx": lambda x, y: 2 * x,
      "df_dy": lambda x, y: 2 * y,
      "expr": "x² + y²"
    },
    "sin_xy": {
      "f": lambda x, y: np.sin(x) * np.sin(y),
      "df_dx": lambda x, y: np.cos(x) * np.sin(y),
      "df_dy": lambda x, y: np.sin(x) * np.cos(y),
      "expr": "sin(x).sin(y)"
    },
    "sin_scaled": {
      "f": lambda x, y: np.sin(3 * x) * np.sin(2 * y),
      "df_dx": lambda x, y: 3 * np.cos(3 * x) * np.sin(2 * y),
      "df_dy": lambda x, y: 2 * np.sin(3 * x) * np.cos(2 * y),
      "expr": "sin(3x).sin(2y)"
    },
    "poly_xy_square": {
      "f": lambda x, y: 2 * (x ** 2) * (y ** 2),
      "df_dx": lambda x, y: 4 * x * (y ** 2),
      "df_dy": lambda x, y: 4 * y * (x ** 2),
      "expr": "2x²y²"
    },
    "poly_xy": {
      "f": lambda x, y: x * y,
      "df_dx": lambda x, y: y,
      "df_dy": lambda x, y: x,
      "expr": "xy"
    }
  }
  return functions

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

def main():
  # Domain
  meshes = ["carre.msh", "hybrid.msh", "rectangles2d.msh", "triangles2d.msh"]
  for mesh in meshes:
    dim, mesh_path, mesh_name = get_mesh(f"big/var/{mesh}", 2)
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    domain = Domain.create_domain(mesh_path, dim, Domain.PartitioningClass.Par_Nodal)
    sys.stdout.close()
    sys.stdout = old_stdout

    # Functions
    function = _get_functions()["sin_xy"]
    fun = function["f"]
    df_dx = function["df_dx"]
    df_dy = function["df_dy"]
    fun_expr = f"fun={function['expr']}"

    # Variable
    var = _get_var(domain, fun)
    var.compute_cell_gradient()
    var.compute_face_gradient()

    # Reference
    cx, cy = _reference(domain.cells.center, df_dx, df_dy)
    hcx, hcy = _reference(domain.halos.centvol[:, 0:2], df_dx, df_dy)
    fx, fy = _reference(domain.faces.center, df_dx, df_dy)

    # Check cell gradient
    # domain.save_on_cell_multi(0, 0, 0, 0, variables=["ne", "exact"],
    #                           values=[var.gradcellx, cx], file_format="vtu")

    metrics = error_metrics(var.gradcellx, cx)
    print(fun_expr)
    try:
      np.testing.assert_allclose(var.gradcellx, cx, rtol=1e-3, atol=1e-3)
      np.testing.assert_allclose(var.gradcelly, cy, rtol=1e-3, atol=1e-3)
      print(f'{mesh}: nb_cells={domain.nbcells}, Ok')
      print(metrics)
    except AssertionError as e:
      print(f'{mesh}: nb_cells={domain.nbcells}, Not Ok')
      print(metrics)
      print(str(e))
      print("----------------------------------------------------------------\n")

  # # Check face gradient
  # np.testing.assert_allclose(var.gradfacex, fx, rtol=1e-1, atol=1e-1)
  # np.testing.assert_allclose(var.gradfacey, fy, rtol=1e-1, atol=1e-1)
  #
  # # Check halo cell gradient (MPI communication)
  # # This test is only to test communication. the actual values are already tested in cell gradient
  # np.testing.assert_allclose(var.gradhalocellx, hcx, rtol=1e-1, atol=1e-1)
  # np.testing.assert_allclose(var.gradhalocelly, hcy, rtol=1e-1, atol=1e-1)



if __name__ == "__main__":
  main()