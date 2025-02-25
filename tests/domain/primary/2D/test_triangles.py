import numpy as np
from helpers.DomainTables import DomainTables

mesh_name = 'triangles.msh'
float_precision = 'float32'
dim = 2

def halo_test(domain_tables, get_neighboring_by_vertex, get_neighboring_by_face):
  """
  Create halo cells by cell vertex and by cell faces using the original mesh and loctoglob, then compare it with local domain partition correspondence.
  """
  nb_partitions = domain_tables.nb_partitions
  # Deps tables (should be valid)
  d_loctoglob = domain_tables.d_cell_loctoglob
  # Test subjects
  d_halonid = domain_tables.d_cell_halonid
  d_halo_halosext = domain_tables.d_halo_halosext
  d_face_halofid = domain_tables.d_face_halofid
  d_cell_faces = domain_tables.d_cell_faces
  d_face_name = domain_tables.d_face_name

  # The mesh is consist of 10 rectangles along the x-axis and y-axis each rectangle has two triangles.
  width = 10
  which_partition = np.ndarray(shape=(width * width * 2), dtype=np.int32)

  # create which_partition (cell partition id)
  for p in range(nb_partitions):
    loctoglob = d_loctoglob[p]
    for j in range(len(loctoglob)):
      global_index = loctoglob[j]
      which_partition[global_index] = p

  for p in range(nb_partitions):
    loctoglob = d_loctoglob[p]

    # remove any face that does not have a halo cell.
    (d_face_halofid[p])[d_face_name[p] != 10] = -1

    for cellid in range(len(loctoglob)):
      g_cellid = loctoglob[cellid]

      # #####################
      # halonid
      # #####################
      g_cellnid = get_neighboring_by_vertex(g_cellid, width)
      # every neighboring cell not in the same partition is a halo cell
      g_halonid = g_cellnid[which_partition[g_cellnid] != which_partition[g_cellid]]
      g_halonid = np.sort(g_halonid)

      halonid = d_halonid[p][cellid]
      # get domain global halo cells
      halonid = d_halo_halosext[p][halonid[0:halonid[-1]]][:, 0]
      halonid = np.sort(halonid)

      # #####################
      # halofid
      # #####################
      g_cellfid = get_neighboring_by_face(g_cellid, width)
      # every neighboring cell not in the same partition is a halo cell
      g_halofid = g_cellfid[which_partition[g_cellfid] != which_partition[g_cellid]]
      g_halofid = np.sort(g_halofid)

      cell_faces = d_cell_faces[p][cellid] # get cell faces
      halofid = d_face_halofid[p][cell_faces[0:cell_faces[-1]]] # get cell halo cells
      halofid = halofid[halofid != -1] # get cell halo cells
      halofid = d_halo_halosext[p][halofid][:, 0] # get halos global index
      halofid = np.sort(halofid)

      # #####################
      # Test
      # #####################
      np.testing.assert_array_equal(g_halonid, halonid)
      np.testing.assert_array_equal(g_halofid, halofid)

def get_neighboring_by_face(i, width):
  sq_id = i // 2
  if i % 2 == 0:
    # all neighboring cells
    arr = np.array([i + 1, i - 1, i - width * 2 + 1], dtype=np.int32)
    # disable non-existing neighboring cells
    if sq_id % width - 1 < 0:
      arr[1] = -1
    if sq_id - width < 0:
      arr[2] = -1
  else:
    # all neighboring cells
    arr = np.array([i - 1, i + 1, i + width * 2 - 1], dtype=np.int32)
    # disable non-existing neighboring cells
    if sq_id % width + 1 >= width:
      arr[1] = -1
    if sq_id + width >= width * width:
      arr[2] = -1
  return np.sort(arr[arr != -1])

def get_neighboring_by_vertex(i, width):
  sq_id = i // 2

  arr = np.array([
    i - width * 2 - 3,
    i - width * 2 - 2,
    i - width * 2 - 1,
    i - width * 2,
    i - width * 2 + 1,
    i - width * 2 + 2,
    i - 3, #6
    i - 2,
    i - 1, # index 8, neighbor in the same cell
    i + 1,
    i + 2,
    i + width * 2 - 3, #11
    i + width * 2 - 2,
    i + width * 2 - 1,
    i + width * 2,
    i + width * 2 + 1,
    i + width * 2 + 2,
  ], dtype=np.int32)
  if i % 2 == 0:
    arr = arr + 1
    arr[8] = i + 1
    arr[[0, 10, 14, 15, 16]] = -1
  else:
    arr[[0, 1, 2, 6, 16]] = -1
  if sq_id % width - 1 < 0:
    arr[[0, 1, 6, 7, 11, 12]] = -1
  if sq_id % width + 1 >= width:
    arr[[4, 5, 9, 10, 15, 16]] = -1
  if sq_id + width >= width * width:
    arr[[11, 12, 13, 14, 15, 16]] = -1
  if sq_id - width < 0:
    arr[[0, 1, 2, 3, 4, 5]] = -1
  return np.sort(arr[arr != -1])

# Test 1
def test_unified_domain():
  """
  Test unified domain (area, center, neighboring by vertex, neighboring by face)

  :details Create a rectangular mesh using loops, calculate the cell info (each rectangle has two triangles), and compare them with the corresponding domain values.
  """

  decimal_precision = 4
  domain_tables = DomainTables(nb_partitions=1, mesh_name=mesh_name, float_precision=float_precision, dim=dim)

  Steps = 10
  With = 10.0
  Height = 5.0
  StepX = With / Steps
  StepY = Height / Steps

  d_cells = domain_tables.d_cells[0]
  d_points = (domain_tables.d_nodes[0])[:, 0:2]
  d_center = (domain_tables.d_cell_center[0])[:, 0:2]
  d_area = domain_tables.d_cell_volume[0]
  d_cellfid = domain_tables.d_cell_cellfid[0]
  d_cellnid = domain_tables.d_cell_cellnid[0]


  cmp = 0
  for x in np.arange(0, With, StepX):
    for y in np.arange(0, Height, StepY):
      p1 = np.array([x, y])
      p2 = np.array([x + StepX, y])
      p3 = np.array([x + StepX, y + StepY])
      p4 = np.array([x, y + StepY])
      area = (StepX * StepY)

      # traingle 1
      t1_points = np.array([p1, p2, p4], dtype=d_points.dtype)
      t1_center = np.array([
        np.sum(t1_points[:, 0]) / 3.0,
        np.sum(t1_points[:, 1]) / 3.0,
      ], dtype=d_center.dtype)
      t1_area = area / 2.0
      t1_nf = get_neighboring_by_face(cmp, Steps)
      t1_nv = get_neighboring_by_vertex(cmp, Steps)

      # traingle 2
      t2_points = np.array([p4, p2, p3])
      t2_center = np.array([
        np.sum(t2_points[:, 0]) / 3.0,
        np.sum(t2_points[:, 1]) / 3.0,
      ], dtype=d_points.dtype)
      t2_area = area / 2.0
      t2_nf = get_neighboring_by_face(cmp + 1, Steps)
      t2_nv = get_neighboring_by_vertex(cmp + 1, Steps)

      # domain traingle 1
      c1_points = d_points[d_cells[cmp][:-1]]
      c1_center = d_center[cmp]
      c1_area = d_area[cmp]
      c1_nf = np.sort(d_cellfid[cmp][0:d_cellfid[cmp][-1]])
      c1_nv = np.sort(d_cellnid[cmp][0:d_cellnid[cmp][-1]])

      # domain traingle 2
      c2_points = d_points[d_cells[cmp + 1][:-1]]
      c2_center = d_center[cmp + 1]
      c2_area = d_area[cmp + 1]
      c2_nf = np.sort(d_cellfid[cmp + 1][0:d_cellfid[cmp + 1][-1]])
      c2_nv = np.sort(d_cellnid[cmp + 1][0:d_cellnid[cmp + 1][-1]])

      np.testing.assert_almost_equal(c1_points, t1_points, decimal=decimal_precision)
      np.testing.assert_almost_equal(c2_points, t2_points, decimal=decimal_precision)
      np.testing.assert_almost_equal(c1_center, t1_center, decimal=decimal_precision)
      np.testing.assert_almost_equal(c2_center, t2_center, decimal=decimal_precision)
      np.testing.assert_almost_equal(c1_area, t1_area, decimal=decimal_precision)
      np.testing.assert_almost_equal(c2_area, t2_area, decimal=decimal_precision)
      np.testing.assert_equal(c1_nf, t1_nf)
      np.testing.assert_equal(c2_nf, t2_nf)
      np.testing.assert_equal(c1_nv, t1_nv)
      np.testing.assert_equal(c2_nv, t2_nv)

      cmp += 2
  # Number of cells
  np.testing.assert_equal(Steps * Steps * 2, len(d_cells))

# Test 2
def test_halos():
  """
    # Test halo cells (by vertex and by face)
  """

  # create `nb_partitions` local domains
  domain_tables = DomainTables(nb_partitions=4, mesh_name=mesh_name, float_precision=float_precision, dim=dim)
  halo_test(domain_tables, get_neighboring_by_vertex, get_neighboring_by_face)
  domain_tables = DomainTables(nb_partitions=5, mesh_name=mesh_name, float_precision=float_precision, dim=dim)
  halo_test(domain_tables, get_neighboring_by_vertex, get_neighboring_by_face)
  domain_tables = DomainTables(nb_partitions=6, mesh_name=mesh_name, float_precision=float_precision, dim=dim)
  halo_test(domain_tables, get_neighboring_by_vertex, get_neighboring_by_face)
  domain_tables = DomainTables(nb_partitions=7, mesh_name=mesh_name, float_precision=float_precision, dim=dim)
  halo_test(domain_tables, get_neighboring_by_vertex, get_neighboring_by_face)

if __name__ == '__main__':
  test_halos()
  test_unified_domain()
