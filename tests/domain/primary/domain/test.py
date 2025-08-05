from numba import int32, float64, types, njit
from numba.experimental import jitclass
from numba.typed import List
import numpy as np
from numba.typed import Dict, List

# 1. Define jitclass spec
spec = [
    ('x', float64),
    ('id', int32),
    ('map_halo_neighsub', types.DictType(int32, types.ListType(int32))),
]

@jitclass(spec)
class LocalDomainStructData:
    def __init__(self):
        self.x  = 0.0
        self.id = 0
        # just create an *untyped* empty Dict placeholder
        self.map_halo_neighsub = Dict.empty(key_type=int32, value_type=List.empty_list(int32))

type = LocalDomainStructData.class_type.instance_type
@njit
def init_dict():
  l = List.empty_list(type)
  l.append(LocalDomainStructData())
  l[0].map_halo_neighsub[int32(0)] = List.empty_list(int32)
  l[0].map_halo_neighsub[int32(0)].append(int32(22))

  print(l[0].map_halo_neighsub[int32(0)])

init_dict()


def prepare_clean_folder(path):
  if not os.path.exists(path):
    os.makedirs(path)
  else:
    for item in os.listdir(path):
      item_path = os.path.join(path, item)
      if os.path.isfile(item_path) or os.path.islink(item_path):
        os.remove(item_path)
      elif os.path.isdir(item_path):
        shutil.rmtree(item_path)


bf_cellid
shared_bf_recv_size















