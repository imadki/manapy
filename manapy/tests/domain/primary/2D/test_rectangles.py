import meshio
import os
import numpy as np
from manapy.ddm import Domain
from manapy.partitions import MeshPartition
from manapy.base.base import Struct

##############################
# Rectangle mesh
# ############################

"""
This test checks the functionality of domain construction using a uniform 2D mesh composed of rectangular cells.

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
  
  :details Create a rectangular mesh using loops, retrieve the cell center and area, and compare them with the corresponding domain values.
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

      cell = d_cells[cmp]
      c_p1 = d_points[cell[0]][0:2]
      c_p2 = d_points[cell[1]][0:2]
      c_p3 = d_points[cell[2]][0:2]
      c_p4 = d_points[cell[3]][0:2]

      cx = (p1[0] + p2[0] + p3[0] + p4[0]) / 4.0
      cy = (p1[1] + p2[1] + p3[1] + p4[1]) / 4.0
      area = np.abs((p2[0] - p1[0]) * (p3[1] - p1[1]))

      c_cx = d_center[cmp][0]
      c_cy = d_center[cmp][1]
      c_area = d_area[cmp]

      np.testing.assert_almost_equal(p1, c_p1, decimal=decimal_precision)
      np.testing.assert_almost_equal(p2, c_p2, decimal=decimal_precision)
      np.testing.assert_almost_equal(p3, c_p3, decimal=decimal_precision)
      np.testing.assert_almost_equal(p4, c_p4, decimal=decimal_precision)
      np.testing.assert_almost_equal(cx, c_cx, decimal=decimal_precision)
      np.testing.assert_almost_equal(cy, c_cy, decimal=decimal_precision)
      np.testing.assert_almost_equal(area, c_area, decimal=decimal_precision)

      cmp += 1

if __name__ == '__main__':
  PATH = os.path.dirname(os.path.realpath(__file__))
  PATH = os.path.join(PATH, 'mesh', 'rectangles.msh')
  domain = load(PATH, dim=2, float_precision='float32')
  main_test(domain,decimal_precision=4)
