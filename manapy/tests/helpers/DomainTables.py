import subprocess
import os
import h5py

def create_partitions(nb_partitions, mesh_name, float_precision, dim):
  root_file = os.path.dirname(os.path.realpath(__file__))
  mesh_file_path = os.path.join(root_file, '..', 'mesh', mesh_name)
  script_path = os.path.join(root_file, 'create_partitions_mpi_worker.py')
  cmd = ["mpirun", "--allow-run-as-root", "--use-hwthread-cpus", "-n", str(nb_partitions), "--oversubscribe", "python3", script_path, mesh_file_path, float_precision, str(dim)]

  result = subprocess.run(cmd, env=os.environ.copy(), stderr=subprocess.PIPE)
  if result.returncode != 0:
    print(result.__str__(), os.getcwd())
    raise SystemExit(result.returncode)


class DomainTables:
  __slots__ = [
    "d_cells",
    "d_faces",
    "d_nodes",
    "d_cell_nodeid",
    "d_cell_faces",
    "d_cell_center",
    "d_cell_volume",
    "d_cell_halonid",
    "d_cell_loctoglob",
    "d_cell_cellfid",
    "d_cell_cellnid",
    "d_cell_nf",
    "d_cell_ghostnid",
    "d_cell_haloghostnid",
    "d_cell_haloghostcenter",
    "d_cell_tc",
    "d_node_loctoglob",
    "d_node_cellid",
    "d_node_name",
    "d_node_oldname",
    "d_node_ghostid",
    "d_node_haloghostid",
    "d_node_ghostcenter",
    "d_node_haloghostcenter",
    "d_node_ghostfaceinfo",
    "d_node_haloghostfaceinfo",
    "d_node_halonid",
    "d_halo_halosext",
    "d_halo_halosint",
    "d_halo_neigh",
    "d_halo_centvol",
    "d_halo_sizehaloghost",
    "d_halo_indsend",
    "d_face_halofid",
    "d_face_name",
    "d_face_normal",
    "d_face_center",
    "d_face_measure",
    "d_face_ghostcenter",
    "d_face_oldname",
    "d_face_cellid",
    "nb_partitions",
    "float_precision"
  ]

  def __init__(self, nb_partitions, mesh_name, float_precision, dim, create_par_fun=create_partitions):
    if create_par_fun:
      create_par_fun(nb_partitions, mesh_name, float_precision=float_precision, dim=dim)

    self.nb_partitions = nb_partitions
    self.float_precision = float_precision

    for i in range(nb_partitions):
      mesh_dir = "domain_meshes" + str(nb_partitions) + "PROC"
      filename = os.path.join(mesh_dir, f"mesh{i}.hdf5")
      with h5py.File(filename, "r") as f:
        for key in f.keys():
          arr = self.add_attribute_if_not_exists(key, nb_partitions)
          arr[i] = f[key][...]

  def add_attribute_if_not_exists(self, attr_name, nb_partitions):
    if not hasattr(self, attr_name):
      setattr(self, attr_name, [i for i in range(nb_partitions)])
    return getattr(self, attr_name)




