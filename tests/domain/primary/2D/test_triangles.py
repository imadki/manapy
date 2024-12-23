import meshio
import os
import numpy as np
from manapy.ddm import Domain
from manapy.partitions import MeshPartition
from manapy.base.base import Struct

##############################
# Triangles mesh
##############################

"""
This test checks the functionality of domain construction using a uniform 2D mesh composed of triangular cells.

    Verify cell centers
    Verify cell areas
    Verify domain construction, specifically (_cells._nodeid, _cells._vertex, _cells._center, _cells._volume)
"""

def load(path : 'str', dim : 'int', float_precision : 'str'):
  running_conf = Struct(backend="numba", signature=True, cache=True, float_precision=float_precision)
  MeshPartition(path, dim=dim, conf=running_conf, periodic=[0,0,0])
  domain = Domain(dim=dim, conf=running_conf)
  return domain

def main_test(domain, decimal_precision):
  """
  :param domain: manapy domain
  :param decimal_precision: decimal precision for comparison
  
  :details Create a rectangular mesh using loops, retrieve the cell center and area (each rectangle has two triangles), and compare them with the corresponding domain values.
  """
  With = 10.0
  Height = 5.0
  StepX = With / 10.0
  StepY = Height / 10.0

  d_cells = domain._cells._nodeid
  d_points = domain._nodes._vertex
  d_center = domain._cells._center
  d_area = domain._cells._volume

  cmp = 0
  for x in np.arange(0, With, StepX):
    for y in np.arange(0, Height, StepY):
      p1 = np.array([x, y])
      p2 = np.array([x + StepX, y])
      p3 = np.array([x + StepX, y + StepY])
      p4 = np.array([x, y + StepY])
      area = np.abs((p2[0] - p1[0]) * (p3[1] - p1[1])) / 2.0

      # traingle 1
      t1_p1 = p1
      t1_p2 = p2
      t1_p3 = p4
      t1_cx = (t1_p1[0] + t1_p2[0] + t1_p3[0]) / 3.0
      t1_cy = (t1_p1[1] + t1_p2[1] + t1_p3[1]) / 3.0

      # traingle 2
      t2_p1 = p4
      t2_p2 = p2
      t2_p3 = p3
      t2_cx = (t2_p1[0] + t2_p2[0] + t2_p3[0]) / 3.0
      t2_cy = (t2_p1[1] + t2_p2[1] + t2_p3[1]) / 3.0

      # domain traingle 1
      c1_p1 = d_points[d_cells[cmp][0]][0:2]
      c1_p2 = d_points[d_cells[cmp][1]][0:2]
      c1_p3 = d_points[d_cells[cmp][2]][0:2]
      c1_cx = d_center[cmp][0]
      c1_cy = d_center[cmp][1]
      c1_area = d_area[cmp]

      # domain traingle 2
      c2_p1 = d_points[d_cells[cmp + 1][0]][0:2]
      c2_p2 = d_points[d_cells[cmp + 1][1]][0:2]
      c2_p3 = d_points[d_cells[cmp + 1][2]][0:2]
      c2_cx = d_center[cmp + 1][0]
      c2_cy = d_center[cmp + 1][1]
      c2_area = d_area[cmp + 1]

      np.testing.assert_almost_equal(c1_p1, t1_p1, decimal=decimal_precision)
      np.testing.assert_almost_equal(c1_p2, t1_p2, decimal=decimal_precision)
      np.testing.assert_almost_equal(c1_p3, t1_p3, decimal=decimal_precision)
      np.testing.assert_almost_equal(c2_p1, t2_p1, decimal=decimal_precision)
      np.testing.assert_almost_equal(c2_p2, t2_p2, decimal=decimal_precision)
      np.testing.assert_almost_equal(c2_p3, t2_p3, decimal=decimal_precision)
      np.testing.assert_almost_equal(c1_cx, t1_cx, decimal=decimal_precision)
      np.testing.assert_almost_equal(c1_cy, t1_cy, decimal=decimal_precision)
      np.testing.assert_almost_equal(c2_cx, t2_cx, decimal=decimal_precision)
      np.testing.assert_almost_equal(c2_cy, t2_cy, decimal=decimal_precision)
      np.testing.assert_almost_equal(c1_area, area, decimal=decimal_precision)
      np.testing.assert_almost_equal(c2_area, area, decimal=decimal_precision)

      cmp += 2


def test_main():
  PATH = os.path.dirname(os.path.realpath(__file__))
  PATH = os.path.join(PATH, 'mesh', 'triangles.msh')
  domain = load(PATH, dim=2, float_precision='float32')
  main_test(domain,decimal_precision=4)
