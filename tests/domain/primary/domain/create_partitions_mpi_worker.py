import sys
from manapy.partitions import MeshPartition
from manapy.base.base import Struct
from mpi4py import MPI
import h5py
from create_domain import Domain
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#  mpirun -n 6 python3 create_partitions_mpi_worker.py /media/aben-ham/SSD/aben-ham/work/manapy/tests/domain/primary/2D/mesh/rectangles.msh float32 2

def save_tables(domain):
  mesh_dir = "domain_meshes" + str(size) + "PROC"
  if rank == 0 and not os.path.exists(mesh_dir):
    os.mkdir(mesh_dir)

  MPI.COMM_WORLD.barrier()

  filename = os.path.join(mesh_dir, f"mesh{rank}.hdf5")
  print(f"saving mesh {filename}")
  if os.path.exists(filename):
    os.remove(filename)

  with h5py.File(filename, "w") as f:
    f.create_dataset("d_cells", data=domain.cells.nodeid)
    f.create_dataset("d_faces", data=domain.faces.nodeid)
    f.create_dataset("d_nodes", data=domain.nodes.vertex)
    f.create_dataset("d_cell_nodeid", data=domain.cells.nodeid)
    f.create_dataset("d_cell_faces", data=domain.cells.faceid)
    f.create_dataset("d_cell_center", data=domain.cells.center)
    f.create_dataset("d_cell_volume", data=domain.cells.volume)
    f.create_dataset("d_cell_halonid", data=domain.cells.halonid)
    f.create_dataset("d_cell_loctoglob", data=domain.cells.loctoglob)
    f.create_dataset("d_cell_cellfid", data=domain.cells.cellfid)
    f.create_dataset("d_cell_cellnid", data=domain.cells.cellnid)
    f.create_dataset("d_cell_nf", data=domain.cells.nf)
    f.create_dataset("d_cell_ghostnid", data=domain.cells.ghostnid)
    f.create_dataset("d_cell_haloghostnid", data=domain.cells.haloghostnid)
    f.create_dataset("d_cell_haloghostcenter", data=domain.cells.haloghostcenter)
    # f.create_dataset("d_cell_tc", data=domain.cells.tc)
    f.create_dataset("d_node_loctoglob", data=domain.nodes.loctoglob)
    f.create_dataset("d_node_cellid", data=domain.nodes.cellid)
    f.create_dataset("d_node_name", data=domain.nodes.name)
    f.create_dataset("d_node_oldname", data=domain.nodes.oldname)
    f.create_dataset("d_node_ghostid", data=domain.nodes.ghostid)
    f.create_dataset("d_node_haloghostid", data=domain.nodes.haloghostid)
    f.create_dataset("d_node_ghostcenter", data=domain.nodes.ghostcenter)
    f.create_dataset("d_node_haloghostcenter", data=domain.nodes.haloghostcenter)
    f.create_dataset("d_node_ghostfaceinfo", data=domain.nodes.ghostfaceinfo)
    f.create_dataset("d_node_haloghostfaceinfo", data=domain.nodes.haloghostfaceinfo)
    f.create_dataset("d_node_halonid", data=domain.nodes.halonid)
    f.create_dataset("d_halo_halosext", data=domain.halos.halosext)
    f.create_dataset("d_halo_halosint", data=domain.halos.halosint)
    f.create_dataset("d_halo_neigh", data=domain.halos.neigh)
    f.create_dataset("d_halo_centvol", data=domain.halos.centvol)
    f.create_dataset("d_halo_sizehaloghost", data=domain.halos.sizehaloghost)
    # f.create_dataset("d_halo_indsend", data=domain.halos.indsend)
    f.create_dataset("d_face_halofid", data=domain.faces.halofid)
    f.create_dataset("d_face_name", data=domain.faces.name)
    f.create_dataset("d_face_normal", data=domain.faces.normal)
    f.create_dataset("d_face_center", data=domain.faces.center)
    f.create_dataset("d_face_measure", data=domain.faces.mesure)
    f.create_dataset("d_face_ghostcenter", data=domain.faces.ghostcenter)
    f.create_dataset("d_face_oldname", data=domain.faces.oldname)
    f.create_dataset("d_face_cellid", data=domain.faces.cellid)

def create_partitions(mesh_file_path, float_precision, dim):
  domain = Domain.create_domain(mesh_file_path, dim, float_precision, recreate=True)
  save_tables(domain)


if len(sys.argv) == 4:
  mesh_file_path = sys.argv[1]
  float_precision = sys.argv[2]
  dim = int(sys.argv[3])
  if (float_precision == 'float32' or float_precision == 'float64') and (dim == 2 or dim == 3):
    print("path", mesh_file_path, "precision", float_precision, "dim", dim, "rank", rank)
    create_partitions(mesh_file_path, float_precision, dim)
  else:
    raise Exception(f"Invalid float_precision argument or Invalid dim argument {dim} {float_precision}")
else:
  raise Exception("Invalid mesh_file_path argument")

