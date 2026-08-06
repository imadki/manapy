from manapy.domain import Domain
import subprocess
import sys
import os
import pickle
import shutil
import numpy as np
from manapy.backends.config import ManapyConfig
from manapy.testing.ReferenceTables import ReferenceTables
from manapy.helpers.mesh_files import test_meshes_folder


def make_test_config(**overrides) -> ManapyConfig:
  """The ManapyConfig every test builds its domain with.

  Defaults to the float64/int64 CPU pair the reference tables were generated
  with, and stays overridable from the environment so the same suite can be run
  against another precision pair or on the GPU without editing a test:
      MANAPY_DEVICE=cuda pytest tests/
  """
  kwargs = {
    "float_precision": os.environ.get("MANAPY_FLOAT", "float64"),
    "int_precision": os.environ.get("MANAPY_INT", "int64"),
    "device": os.environ.get("MANAPY_DEVICE", "cpu"),
  }
  kwargs.update(overrides)
  return ManapyConfig(**kwargs)


def _supports_oversubscribe(mpi_exec_path):
  try:
    result = subprocess.run(
      [mpi_exec_path, "--oversubscribe", "-n", "4", "hostname"],
      text=True,
      check=True, env=os.environ.copy(), capture_output=True
    )
    return result.returncode == 0
  except subprocess.CalledProcessError:
    return False

def duplicate_config(configs):
  import copy
  # duplicate config for nb_parts
  res = []
  for config in configs:
    if isinstance(config["nb_parts"], list):
      for nb_part in config["nb_parts"]:
        new_config = copy.copy(config)
        new_config["nb_parts"] = nb_part
        res.append(new_config)
    else:
      res.append(config)
  return res

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


def get_local_domains(nb_parts: int, mesh_path: str, dim: int, partitioning_type: str = "Partitioning.Par_Nodal",
                      config: ManapyConfig = None):
  print(mesh_path)
  # The partitions are built in a spawned mpirun, so the config cannot be handed
  # over as an object: the worker rebuilds an identical one from its fields.
  config = config if config is not None else make_test_config()
  create_partitions_mpi_worker = f"""
import pickle
from manapy.domain import Domain, Partitioning, Mesh
from manapy.backends.config import ManapyConfig
from manapy.helpers import get_test_mesh
from mpi4py import MPI
import os
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
config = ManapyConfig(float_precision='{config.float_precision}',
                      int_precision='{config.int_precision}',
                      device='{config.device.name.lower()}',
                      verbose={config.verbose})
local_domain = Domain.create_domain('{mesh_path}', {dim}, config, {partitioning_type}, recreate=True)

mesh = Mesh('{mesh_path}', {dim}, config)
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

  # Check MPI executable existance
  if not (os.path.isfile(mpirun_path) and os.access(mpirun_path, os.X_OK)):
    mpirun_path = shutil.which("mpirun")

    if mpirun_path is None:
      raise FileNotFoundError("mpirun not found in local path or system PATH")

  print("Using mpirun:", mpirun_path)

  is_supports_oversubscribe = _supports_oversubscribe(mpirun_path)
  try:
    if is_supports_oversubscribe:
      subprocess.run([
        mpirun_path,
        "--oversubscribe",
        "-n", str(nb_parts),
        python_path,
        tmp_file_name,
      ], check=True, env=os.environ.copy(), capture_output=True, text=True)
    else:
      subprocess.run([
        mpirun_path,
        "-n", str(nb_parts),
        python_path,
        tmp_file_name
      ], check=True, env=os.environ.copy(), capture_output=True, text=True)
  except subprocess.CalledProcessError as e:
    print("Command failed")
    print("Return code:", e.returncode)
    print("STDOUT:\n", e.stdout)
    print("STDERR:\n", e.stderr)
    exit(1)

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
  shutil.rmtree("vtk_results")
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

  dim, mesh_path, mesh_name = get_mesh("rectangles.msh")
  lds = get_local_domains(4, mesh_path, dim, "Partitioning.Par_Nodal", make_test_config())

  reference_domain = get_reference_domain("rectangles.hd5", dim)
  print(lds[0].cells.nodeid)