import warnings

from mpi4py import MPI
import time

def re_import(module_name, class_name):
  import importlib

  # examples: module_name -> helpers.TablesTestHexa3D
  # class_name TablesTestHexa3D
  module = importlib.import_module(module_name)
  importlib.reload(module)
  return getattr(module, class_name)

class Logger:
  def __init__(self):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    self.rank = rank
    self.dic = {}
    self.start = time.time()
    self.last_entry = ""

  def reset_start(self):
    self.start = time.time()

  def print_sorted_results(self):
    dic = self.dic
    sorted_dic = dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))
    print("------------------------------------------------")
    print(">>>>>>>>>>>>>>>>>> Results <<<<<<<<<<<<<<<<<<<<<")
    print("------------------------------------------------")
    for item in sorted_dic:
      print(f'{item} => {sorted_dic[item]:.6f} seconds')

  def log(self, string):
    if string == "":
      raise RuntimeError("string cannot be empty")
    if self.last_entry == "":
      self.start = time.time()
    self.last_entry = string
    self.dic[string] = time.time()

  def out(self, string=""):
    if string == "":
      string = self.last_entry
    time_taken = time.time() - self.dic[string]
    time_from_start = time.time() - self.start
    print(f"[Rank: {self.rank}] {string} Acc {time_from_start:.6f} seconds (delta: {time_taken:.6f} seconds)", flush=True)

log_step = Logger()



