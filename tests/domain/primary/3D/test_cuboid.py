import meshio
import os
import numpy as np
from manapy.ddm import Domain
from manapy.partitions import MeshPartition
from manapy.base.base import Struct

##############################
# Cuboid mesh
# ############################

"""
This test checks the functionality of domain construction using a uniform 3D mesh composed of cuboid cells.

    Verify cell centers
    Verify cell areas
    Verify domain construction, specifically (_cells._nodeid, _cells._vertex, _cells._center, _cells._volume)
"""

def main_test(domain, decimal_precision):
  """
  :param domain: manapy domain
  :param decimal_precision: decimal precision for comparison

  :details Create a cuboid mesh constructed from cuboid cells, retrieve the cell center and area, and compare them with the corresponding domain values.
  """
  With = 10.0
  Height = 5.0
  Depth = 15.0
  StepX = With / 10.0
  StepY = Height / 10.0
  StepZ = Depth / 10.0

  d_cells = domain._cells._nodeid
  d_points = domain._nodes._vertex[:, 0:3]
  d_center = domain._cells._center[:, 0:3]
  d_volume = domain._cells._volume

  cmp = 0
  for x in np.arange(With, 0.0, -StepX):
    for z in np.arange(Depth, 0.0, -StepZ):
      for y in np.arange(Height, 0.0, -StepY):
        points = np.array([
          [x, y, z],
          [x - StepX, y, z],
          [x - StepX, y, z - StepZ],
          [x, y, z - StepZ],

          [x, y - StepY, z],
          [x - StepX, y - StepY, z],
          [x - StepX, y - StepY, z - StepZ],
          [x, y - StepY, z - StepZ],
        ], dtype=d_points.dtype)
        c_points = d_points[d_cells[cmp][:-1]]

        #Center
        center = np.array([
          np.sum(points[:, 0]) / 8.0,
          np.sum(points[:, 1]) / 8.0,
          np.sum(points[:, 2]) / 8.0,
        ], dtype=d_points.dtype)
        c_center = d_center[cmp]

        #Volume
        volume = StepX * StepY * StepZ
        c_volume = d_volume[cmp]

        #Testing
        np.testing.assert_almost_equal(c_points, points, decimal=decimal_precision)
        np.testing.assert_almost_equal(c_center, center, decimal=decimal_precision)
        np.testing.assert_almost_equal(c_volume, volume, decimal=decimal_precision)

        #Next cell
        cmp += 1


def test_main():
  dim = 3
  float_precision = 'float32'
  PATH = os.path.dirname(os.path.realpath(__file__))
  PATH = os.path.join(PATH, 'mesh', 'cuboid.msh')

  running_conf = Struct(backend="numba", signature=True, cache=True, float_precision=float_precision)
  MeshPartition(PATH, dim=dim, conf=running_conf, periodic=[0, 0, 0])
  domain = Domain(dim=dim, conf=running_conf)

  main_test(domain, decimal_precision=4)

