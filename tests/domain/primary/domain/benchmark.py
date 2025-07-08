import os
from create_domain import Domain as DomainAlt
from manapy.partitions import MeshPartition
from manapy.base.base import Struct
from manapy.ddm import Domain
import sys
import psutil
#  gmsh ../mesh/tetra_test_2.geo -3 -setnumber Nx 20 -setnumber Ny 20 -setnumber Nz 20  -o tetra_test.msh

import threading

peak_mem = 0

def monitor_memory():
    global peak_mem
    process = psutil.Process(os.getpid())
    while True:
        mem = process.memory_info().rss
        if mem > peak_mem:
            peak_mem = mem
        time.sleep(0.005)  # check every 5ms

# Start monitoring in a background thread
monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
monitor_thread.start()

def mem_usage():
  global peak_mem
  mem_bytes = peak_mem
  mem_mb = mem_bytes / (1024 ** 2)

  return f"{mem_mb:.2f}MB"


if len(sys.argv) != 4:
  print("Usage: python benchmark.py <size> <is_alt=0/1")
  sys.exit(1)

size = int(sys.argv[1])
is_alt = int(sys.argv[2])
mesh_name = sys.argv[3]

mesh_list = [
  (3, mesh_name),
]
float_precision = 'float32' # the test does not support float64 or int64 yet
root_file = os.getcwd()
dim, mesh_path = mesh_list[0] # also modify dim variable accordingly

def create_original_domain(recreate=True):
  running_conf = Struct(backend="numba", signature=True, cache=True, float_precision=float_precision)
  if recreate:
    MeshPartition(mesh_path, dim=dim, conf=running_conf, periodic=[0,0,0])
  return Domain(dim=dim, conf=running_conf)

def partitioning():
    running_conf = Struct(backend="numba", signature=True, cache=True, float_precision=float_precision)
    MeshPartition(mesh_path, dim=dim, conf=running_conf, periodic=[0, 0, 0], forced_size=size)

import time

start = time.time()

if is_alt:
  DomainAlt.partitioning(mesh_path, dim, float_precision, size)
else:
  partitioning()


print(f"END:: Execution time: {time.time() - start:.6f} seconds")
print(f"{time.time() - start:.6f} {mem_usage()}")

