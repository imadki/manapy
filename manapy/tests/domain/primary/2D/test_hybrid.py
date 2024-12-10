import meshio
import os
import numpy as np
from manapy.ddm import Domain
from manapy.partitions import MeshPartition
from manapy.base.base import Struct

##############################
# Hybrid mesh
##############################

"""
This test checks the functionality of domain construction using a hybrid 2D mesh composed of a mix of triangular and rectangular cells.

    Verify cell centers
    Verify cell areas
    Verify domain construction, specifically (_cells._nodeid, _cells._vertex, _cells._center, _cells._volume)

It is assumed that the domain cells and Meshio cells are in the same order.
"""

def load(path : 'str', dim : 'int', float_precision : 'str'):
  running_conf = Struct(backend="numba", signature=True, cache=True, float_precision=float_precision)
  meshio_mesh = meshio.read(path)
  MeshPartition(path, dim=dim, conf=running_conf, periodic=[0,0,0])
  domain = Domain(dim=dim, conf=running_conf)
  return meshio_mesh, domain

def check(points, c_points, c_cx, c_cy, c_area, decimal_precision):
  """
  :param points: float[:, :] meshio cell nodes
  :param c_points: domain cell nodes
  :param c_cx: domain cell center x coordinate
  :param c_cy: domain cell center y coordinate
  :param c_area: domain cell area
  :param decimal_precision: decimal precision for comparison

  :details Compute the cell center and cell area from `points`, and compare them with the corresponding domain cells using `np.testing.assert_almost_equal`
  """
  cx = sum(points[:, 0]) / len(points)
  cy = sum(points[:, 1]) / len(points)

  # Shoelace Theorem
  area = 0.0
  for i in range(len(points)):
    j = (i + 1) % len(points)
    x_i = points[i][0]
    y_i = points[i][1]
    x_j = points[j][0]
    y_j = points[j][1]
    area += x_i * y_j - x_j * y_i
  area = abs(area) / 2.0

  np.testing.assert_almost_equal(cx, c_cx, decimal=decimal_precision)
  np.testing.assert_almost_equal(cy, c_cy, decimal=decimal_precision)
  np.testing.assert_almost_equal(area, c_area, decimal=decimal_precision)

  for i in range(points.shape[0]):
    np.testing.assert_almost_equal(points[i], c_points[i], decimal=decimal_precision)

def main_test(domain, meshio_mesh, decimal_precision):
  """
  :param domain: manapy domain
  :param meshio_mesh: meshio mesh
  :param decimal_precision: decimal precision for comparison

  :details For each cell in the domain, retrieve the cell vertices, get the corresponding Meshio cell vertices, and check for validity.
  """
  d_cells = domain._cells._nodeid
  d_points = domain._nodes._vertex
  d_center = domain._cells._center
  d_area = domain._cells._volume
  m_triangles = meshio_mesh.cells['triangle']
  m_rectangles = meshio_mesh.cells['quad']
  m_points = meshio_mesh.points

  t_cmp = 0
  r_cmp = 0
  for i in range(d_cells.shape[0]):
    d_cell = d_cells[i]
    m_cell = None
    if d_cell[-1] == 4: #regrangles
      m_cell = m_rectangles[r_cmp]
      r_cmp += 1
    elif d_cell[-1] == 3: # triangles
      m_cell = m_triangles[t_cmp]
      t_cmp += 1
    if m_cell is not None:
      points = np.array([np.array(m_points[m_cell[j]][0:2], dtype=d_points.dtype) for j in range(len(m_cell))])
      c_points = np.array([d_points[d_cell[j]][0:2] for j in range(len(d_cell))])
      c_cx = d_center[i][0]
      c_cy = d_center[i][1]
      c_area = d_area[i]
      check(points, c_points, c_cx, c_cy, c_area, decimal_precision=decimal_precision)

if __name__ == '__main__':
  PATH = os.path.dirname(os.path.realpath(__file__))
  PATH = os.path.join(PATH, 'mesh', 'carre_hybrid.msh')
  meshio_mesh, domain = load(PATH, dim=2, float_precision='float32')
  main_test(domain, meshio_mesh, decimal_precision=4)
