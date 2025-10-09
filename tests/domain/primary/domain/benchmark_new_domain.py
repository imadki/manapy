import os
from create_domain import Partitioning, Mesh
import sys
import psutil
import time
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


if len(sys.argv) != 3:
  print("wrong Usage")
  sys.exit(1)

mesh_name = sys.argv[1]

mesh_list = [
  (3, mesh_name),
]
float_precision = 'float32' # the test does not support float64 or int64 yet
root_file = os.getcwd()
dim, mesh_path = mesh_list[0] # also modify dim variable accordingly



mesh = Mesh(mesh_path, dim)
partitioning = Partitioning(mesh, float_precision)


nb_cells = len(partitioning.cells)
nb_parts = 2**int(sys.argv[2])
if nb_cells / nb_parts < 10:
  exit(0)
start = time.time()
local_domains = partitioning.create_sub_domains(nb_parts=nb_parts)
end = time.time() - start
print(f">>> nbcells [{nb_cells}] nb_parts [{nb_parts}] [{end:.4f}]s")
print(f">>> [{mem_usage()}]")



