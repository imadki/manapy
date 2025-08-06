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


if len(sys.argv) != 2:
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
nb_parts = 2048
local_domains = partitioning.create_sub_domains(nb_parts=nb_parts)
print(mem_usage())


