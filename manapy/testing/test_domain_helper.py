from manapy.domain import Domain
import subprocess
import sys
import os
import pickle
import shutil
import numpy as np
from manapy.testing.ReferenceTables import ReferenceTables
from manapy.helpers.mesh_files import test_meshes_folder

def sort_float_arr(dim, arr):
  # Lexicographic sort by rows.
  if dim == 3:
    arr = np.round(arr, decimals=3)  # np.round to limit sort precision sometime 10.5 is bigger than 10.5
    keys = [arr[:, 0], arr[:, 1], arr[:, 2]]
    indices = np.lexsort(keys)
    arr = arr[indices]
  else:
    arr = np.round(arr, decimals=2)  # np.round to limit sort precision sometime 10.5 is bigger than 10.5
    keys = [arr[:, 0], arr[:, 1]]
    indices = np.lexsort(keys)
    arr = arr[indices]
  return arr, indices


def get_local_domains(nb_parts: int, mesh_path: str, dim: int, partitioning_type: str):
  print(mesh_path)
  create_partitions_mpi_worker = f"""
import pickle
from manapy.domain import Domain, Partitioning, Mesh
from manapy.helpers import get_test_mesh
from mpi4py import MPI
import os
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
local_domain = Domain.create_domain('{mesh_path}', {dim}, {partitioning_type}, recreate=True)

mesh = Mesh('{mesh_path}', {dim})
partitioner = Partitioning(mesh)
if size > 1:
  part_vert, nb_parts = partitioner.set_part_vert(size, {partitioning_type})
else:
  part_vert = np.zeros(shape=(len(mesh.cells)), dtype=np.int32)

mesh_dir = "domain_meshes" + str(size) + "PROC"
if rank == 0 and not os.path.exists(mesh_dir):
  os.mkdir(mesh_dir)

MPI.COMM_WORLD.barrier()

filename = os.path.join(mesh_dir, f"domain_{{rank}}.pkl")
print(f"saving mesh {{filename}}")
if os.path.exists(filename):
  os.remove(filename)

def make_picklable(obj):
  for k, v in vars(obj).items():
    try:
      pickle.dumps(v)
    except:
      setattr(obj, k, None)
  return obj

with open(filename, "wb") as f:
  pickle.dump(make_picklable(local_domain), f)

if rank == 0:
  filename = os.path.join(mesh_dir, '..', f"part_vert.pkl")
  with open(filename, "wb") as f:
    pickle.dump(part_vert, f)
  """

  tmp_file_name = "tmp_7466237981mdfg.py"
  with open(tmp_file_name, "w") as f:
    f.write(create_partitions_mpi_worker)

  mpirun_path = os.path.join(os.path.dirname(sys.executable), "mpirun")
  python_path = os.path.join(os.path.dirname(sys.executable), "python3")

  # try:
  subprocess.run([
    mpirun_path,
    "--oversubscribe",
    "-n", str(nb_parts),
    python_path,
    tmp_file_name
  ], check=True)
  # except subprocess.CalledProcessError as e:
  #   print(f"Error: command failed with exit code {e.returncode}", file=sys.stderr)
  #   sys.exit(e.returncode)

  os.remove(tmp_file_name)

  local_domains : list[Domain] = []
  mesh_dir = "domain_meshes" + str(nb_parts) + "PROC"
  local_mesh_dir = f"local_domain_{nb_parts}"
  for rank in range(nb_parts):
    filename = os.path.join(mesh_dir, f"domain_{rank}.pkl")
    with open(filename, "rb") as f:
      local_domain: Domain = pickle.load(f)
      local_domains.append(local_domain)

  shutil.rmtree(mesh_dir)
  shutil.rmtree(local_mesh_dir)
  shutil.rmtree("results")
  return local_domains

def get_reference_domain(reference_domain: str, dim: int):
  if isinstance(reference_domain, str):
    reference_domain = os.path.join(test_meshes_folder, '..', reference_domain)
  filename = "part_vert.pkl"
  with open(filename, "rb") as f:
    part_vert = pickle.load(f)
  os.remove(filename)
  return ReferenceTables(reference_domain, part_vert, dim)

if __name__ == "__main__":
  from manapy.helpers import get_mesh

  dim, mesh_path, mesh_name = get_mesh(0)
  lds = get_local_domains(4, mesh_path, dim, "Partitioning.Par_Nodal")

  reference_domain = get_reference_domain("rectangles.hd5", dim)
  print(lds[0].cells.nodeid)