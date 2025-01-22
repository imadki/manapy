import os
import numpy as np
from tests_helper import DomainTest, halo_test

mesh_name = 'rectangles.msh'
float_precision = 'float32'
dim = 2


def get_neighboring_by_face(i, width):
  arr = np.array([i - 1, i + 1, i + width, i - width], dtype=np.int32)
  if i % width - 1 < 0:
    arr[0] = -1
  if i % width + 1 >= width:
    arr[1] = -1
  if i + width >= width * width:
    arr[2] = -1
  if i - width < 0:
    arr[3] = -1
  return np.sort(arr[arr != -1])

def get_neighboring_by_vertex(i, width):
  arr = np.array([
    i - 1,
    i + 1,
    i + width,
    i - width,
    i + width - 1,
    i - width - 1,
    i + width + 1,
    i - width + 1,
  ], dtype=np.int32)
  if i % width - 1 < 0:
    arr[[0, 4, 5]] = -1
  if i % width + 1 >= width:
    arr[[1, 6, 7]] = -1
  if i + width >= width * width:
    arr[[2, 4, 6]] = -1
  if i - width < 0:
    arr[[3, 5, 7]] = -1
  return np.sort(arr[arr != -1])

# Test 1
def test_unified_domain():
  """
  Test unified domain (area, center, neighboring by vertex, neighboring by face)

  :details Create a rectangular mesh using loops, calculate the cell info, and compare them with the corresponding domain values.
  """

  decimal_precision = 4
  domain_tables = DomainTest(nb_partitions=1, mesh_name=mesh_name, float_precision=float_precision, dim=dim)

  Steps = 10
  With = 10.0
  Height = 5.0
  StepX = With / Steps
  StepY = Height / Steps

  d_cells = domain_tables.d_cells[0]
  d_points = domain_tables.d_nodes[0][:, 0:2]
  d_center = domain_tables.d_cell_center[0][:, 0:2]
  d_area = domain_tables.d_cell_volume[0]
  d_cellfid = domain_tables.d_cell_cellfid[0]
  d_cellnid = domain_tables.d_cell_cellnid[0]

  cmp = 0
  for x in np.arange(0, With, StepX):
    for y in np.arange(0, Height, StepY):
      # Points
      points = np.array([
        [x, y],
        [x + StepX, y],
        [x + StepX, y + StepY],
        [x, y + StepY]
      ], dtype=d_points.dtype)
      c_points = d_points[d_cells[cmp][:-1]]

      # Center
      center = np.array([
        np.sum(points[:, 0]) / 4.0,
        np.sum(points[:, 1]) / 4.0,
      ], dtype=d_center.dtype)
      c_center = d_center[cmp]

      # Area
      area = StepX * StepY
      c_area = d_area[cmp]

      # Testing neighboring cells by face
      cellfid = get_neighboring_by_face(cmp, Steps)
      c_cellfid = np.sort(d_cellfid[cmp][0:d_cellfid[cmp][-1]])

      # Testing neighboring cells by nodes
      cellnid = get_neighboring_by_vertex(cmp, Steps)
      c_cellnid = np.sort(d_cellnid[cmp][0:d_cellnid[cmp][-1]])

      # Testing
      np.testing.assert_almost_equal(c_points, points, decimal=decimal_precision)
      np.testing.assert_almost_equal(c_center, center, decimal=decimal_precision)
      np.testing.assert_almost_equal(c_area, area, decimal=decimal_precision)
      np.testing.assert_equal(c_cellfid, cellfid)
      np.testing.assert_equal(cellnid, c_cellnid)

      # Next cell
      cmp += 1

    # Number of cells
  np.testing.assert_equal(cmp, Steps * Steps)

# Test 2
def test_halos():
  """
    # Test halo cells (by vertex and by face)
  """

  # create `nb_partitions` local domains
  domain_tables = DomainTest(nb_partitions=4, mesh_name=mesh_name, float_precision=float_precision, dim=dim)
  halo_test(domain_tables, get_neighboring_by_vertex, get_neighboring_by_face)
  domain_tables = DomainTest(nb_partitions=5, mesh_name=mesh_name, float_precision=float_precision, dim=dim)
  halo_test(domain_tables, get_neighboring_by_vertex, get_neighboring_by_face)
  domain_tables = DomainTest(nb_partitions=6, mesh_name=mesh_name, float_precision=float_precision, dim=dim)
  halo_test(domain_tables, get_neighboring_by_vertex, get_neighboring_by_face)
  domain_tables = DomainTest(nb_partitions=7, mesh_name=mesh_name, float_precision=float_precision, dim=dim)
  halo_test(domain_tables, get_neighboring_by_vertex, get_neighboring_by_face)


if __name__ == '__main__':
  test_unified_domain()
  test_halos()