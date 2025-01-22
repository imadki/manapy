import subprocess
import os
from manapy.ddm import Domain
from manapy.base.base import Struct
import numpy as np

def create_partitions(nb_partitions, mesh_name, float_precision, dim):
  root_file = os.path.dirname(os.path.realpath(__file__))
  mesh_file_path = os.path.join(root_file, 'mesh', mesh_name)
  script_path = os.path.join(root_file, 'create_partitions_mpi_worker.py')
  cmd = ["mpirun", "--allow-run-as-root", "--use-hwthread-cpus", "-n", str(nb_partitions), "--oversubscribe", "python3", script_path, mesh_file_path, float_precision, str(dim)]

  result = subprocess.run(cmd, env=os.environ.copy(), stderr=subprocess.PIPE)
  if result.returncode != 0:
    print(result.__str__(), os.getcwd())
    raise SystemExit(result.returncode)

class DomainTest:
  def __init__(self, nb_partitions, mesh_name, float_precision, dim):
    create_partitions(1, mesh_name, float_precision=float_precision, dim=dim)
    create_partitions(nb_partitions, mesh_name, float_precision=float_precision, dim=dim)

    running_conf = Struct(backend="numba", signature=True, cache=True, float_precision=float_precision)
    domain = Domain(dim=dim, conf=running_conf)

    self.nb_partitions = nb_partitions

    self.d_cells = []
    self.d_faces = []
    self.d_nodes = []
    self.d_cell_faces = []
    self.d_cell_center = []
    self.d_cell_volume = []
    self.d_cell_cellfid = []
    self.d_cell_cellnid = []
    self.d_cell_halonid = []
    self.d_cell_loctoglob = []
    self.d_node_loctoglob = []
    self.d_halo_halosext = []
    self.d_face_halofid = []
    self.d_face_name = []

    for i in range(nb_partitions):
      domain.create_domain(nb_partitions, i)

      self.d_cells.append(domain.cells.nodeid)
      self.d_faces.append(domain.faces.nodeid)
      self.d_nodes.append(domain.nodes.vertex)

      self.d_cell_faces.append(domain.cells.faceid)
      self.d_cell_center.append(domain.cells.center)
      self.d_cell_volume.append(domain.cells.volume)
      self.d_cell_halonid.append(domain.cells.halonid)
      self.d_cell_loctoglob.append(domain.cells.loctoglob)
      self.d_cell_cellfid.append(domain.cells.cellfid)
      self.d_cell_cellnid.append(domain.cells.cellnid)
      self.d_node_loctoglob.append(domain.nodes.loctoglob)
      self.d_halo_halosext.append(domain.halos.halosext)
      self.d_face_halofid.append(domain.faces.halofid)
      self.d_face_name.append(domain.faces.name)

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

if __name__ == "__main__":
  mesh_name = 'triangles.msh'
  float_precision = 'float32'
  dim = 2
  create_partitions(2, mesh_name, float_precision=float_precision, dim=dim)

