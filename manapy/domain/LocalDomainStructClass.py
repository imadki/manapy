import numpy as np
from numba.typed import Dict, List
import h5py
import manapy.backends.types as types

# Created inside PartitioningClass
class LocalDomainStruct:
  def __init__(self):
    # Arrays: use zeros(1) as placeholder
    # Returned tables and Scalars
    self.nodes = np.zeros((1, 1), dtype=types.np_float_type) # [[node x, y, z]]
    self.cells = np.zeros((1, 1), dtype=types.np_int_type) # [[cells nodes]]
    self.cells_type = np.zeros(1, dtype=np.int8) # [cell type]
    self.phy_faces = np.zeros((1, 1), dtype=types.np_int_type) # [[physical face nodes]]
    self.phy_faces_name = np.zeros(1, dtype=types.np_int_type) # [physical face name]

    self.cell_loctoglob = np.zeros(1, dtype=types.np_int_type) # [cell global index]
    self.node_loctoglob = np.zeros(1, dtype=types.np_int_type) # [node global index]
    self.node_oldname = np.zeros(1, dtype=types.np_int_type) # [node old name, ...]

    self.halo_neighsub = np.zeros((1, 1), dtype=types.np_int_type) # [[NeighborP1, NeighborP2, ...], [NbHalosIntConnectedToP1, ...]]
    self.node_halos = np.zeros(1, dtype=types.np_int_type) # int32[:] [NodiId, haloId, ...] shape=(2 * nb_halos) couple (NodeId, haloId) for each exthalo, HaloId is an index point to halo_halosext, nodeId is the local nodeId.
    self.halo_halosext = np.zeros((1, 1), dtype=types.np_int_type) # [[global index of halocell, global index of cell nodes, size]] shape=(nb_halos, max_cell_nodeid + 2) Halos of a partition P is the Concatenation of Interiors of the neighbor parts that are connected to P.
    self.halo_halosint = np.zeros(1, dtype=types.np_int_type) # [HalosIntConnectedToP1 halos ..., HalosIntConnectedToP2 halos ..., ...]
    self.halo_centvol = np.zeros((1, 1), dtype=types.np_float_type)  # [halocell_center_{x, y, z}, halocell_volume_{x, y, z}] # z axis only on 3D

    self.phyid_neighbor = np.zeros((1, 1), dtype=types.np_int_type) # [[index0 point to halo_halobf, index1 ..., size]] shape=(nb_nodes, max_node_halobf + 1)
    self.phyid_recv = np.zeros(1, dtype=types.np_int_type) # [boundary faces global index, ...] description="represent the global index of boundary faces that is needed from this partition either from itself or the other partitions, all other tables that will use boundary faces must point to this table"
    self.phyid_send = np.zeros(1, dtype=types.np_int_type) # [recv_part_index, size, size indices point to phyid_recv, ...] description="used when this part need to send its boundary faces to recv_part"
    self.node_halophyid = np.zeros(1, dtype=types.np_int_type)
    self.cell_halophyid = np.zeros(1, dtype=types.np_int_type)

    self.cell_tc = np.zeros(1, dtype=types.np_int_type) # Array stored only on rank0, its size = number of cells of global domain [rank0 loctoglob..., rank1 loctoglob..., rank2 loc.......]

    # Scalars
    self.max_cell_nodeid = 0
    self.max_cell_faceid = 0
    self.max_face_nodeid = 0
    self.max_node_haloid = 0
    self.max_cell_halonid = 0
    self.max_node_phyid = 0
    self.max_node_halophyid = 0
    self.max_cell_phyid = 0
    self.max_cell_halophyid = 0
    self.float_precision = 32 if types.FLOAT_TYPE == 'float32' else 64
    self.int_precision = 32 if types.INT_TYPE == 'int32' else 64
    self.dim = 0 # 2 or 3


  @staticmethod
  def new_local_domains(nb):
    list_local_domains = []
    for i in range(nb):
      obj = LocalDomainStruct()
      list_local_domains.append(obj)
    return list_local_domains

  @staticmethod
  def save_hdf5(ld: 'LocalDomainStruct', path):
    with h5py.File(path, 'w') as f:
      f.create_dataset('nodes', data=ld.nodes)
      f.create_dataset('cells', data=ld.cells)
      f.create_dataset('cells_type', data=ld.cells_type)
      f.create_dataset('phy_faces', data=ld.phy_faces)
      f.create_dataset('phy_faces_name', data=ld.phy_faces_name)
      f.create_dataset('cell_loctoglob', data=ld.cell_loctoglob)
      f.create_dataset('node_loctoglob', data=ld.node_loctoglob)
      f.create_dataset('node_oldname', data=ld.node_oldname)
      f.create_dataset('halo_neighsub', data=ld.halo_neighsub)
      f.create_dataset('halo_halosint', data=ld.halo_halosint)
      f.create_dataset('node_halos', data=ld.node_halos)
      f.create_dataset('node_halophyid', data=ld.node_halophyid)
      f.create_dataset('cell_halophyid', data=ld.cell_halophyid)
      f.create_dataset('phyid_recv', data=ld.phyid_recv)
      f.create_dataset('phyid_neighbor', data=ld.phyid_neighbor)
      f.create_dataset('phyid_send', data=ld.phyid_send)
      f.create_dataset('cell_tc', data=ld.cell_tc)
      f.create_dataset('halo_halosext', data=ld.halo_halosext)
      f.create_dataset('halo_centvol', data=ld.halo_centvol)
      f.create_dataset('dim', data=ld.dim)
      f.create_dataset('float_precision', data=ld.float_precision)
      f.create_dataset('int_precision', data=ld.int_precision)
      f.create_dataset('max_cell_nodeid', data=ld.max_cell_nodeid)
      f.create_dataset('max_cell_faceid', data=ld.max_cell_faceid)
      f.create_dataset('max_face_nodeid', data=ld.max_face_nodeid)
      f.create_dataset('max_node_haloid', data=ld.max_node_haloid)
      f.create_dataset('max_cell_halonid', data=ld.max_cell_halonid)
      f.create_dataset('max_node_phyid', data=ld.max_node_phyid)
      f.create_dataset('max_node_halophyid', data=ld.max_node_halophyid)
      f.create_dataset('max_cell_phyid', data=ld.max_cell_phyid)
      f.create_dataset('max_cell_halophyid', data=ld.max_cell_halophyid)

  @staticmethod
  def load_hd5(path: 'str'):
    local_domain = LocalDomainStruct()

    with h5py.File(path, 'r') as f:
      local_domain.nodes = f['nodes'][...]
      local_domain.cells = f['cells'][...]
      local_domain.cells_type = f['cells_type'][...]
      local_domain.phy_faces = f['phy_faces'][...]
      local_domain.phy_faces_name = f['phy_faces_name'][...]
      local_domain.cell_loctoglob = f['cell_loctoglob'][...]
      local_domain.node_loctoglob = f['node_loctoglob'][...]
      local_domain.node_oldname = f['node_oldname'][...]
      local_domain.halo_neighsub = f['halo_neighsub'][...]
      local_domain.halo_halosint = f['halo_halosint'][...]
      local_domain.node_halos = f['node_halos'][...]
      local_domain.node_halophyid = f['node_halophyid'][...]
      local_domain.cell_halophyid = f['cell_halophyid'][...]
      local_domain.phyid_recv = f['phyid_recv'][...]
      local_domain.phyid_send = f['phyid_send'][...]
      local_domain.phyid_neighbor = f['phyid_neighbor'][...]
      local_domain.cell_tc = f['cell_tc'][...]
      local_domain.halo_halosext = f['halo_halosext'][...]
      local_domain.halo_centvol = f['halo_centvol'][...]
      local_domain.dim = f['dim'][()]
      local_domain.float_precision = f['float_precision'][()]
      local_domain.int_precision = f['int_precision'][()]
      local_domain.max_cell_nodeid = f['max_cell_nodeid'][()]
      local_domain.max_cell_faceid = f['max_cell_faceid'][()]
      local_domain.max_face_nodeid = f['max_face_nodeid'][()]
      local_domain.max_node_haloid = f['max_node_haloid'][()]
      local_domain.max_cell_halonid = f['max_cell_halonid'][()]
      local_domain.max_node_phyid = f['max_node_phyid'][()]
      local_domain.max_node_halophyid = f['max_node_halophyid'][()]
      local_domain.max_cell_phyid = f['max_cell_phyid'][()]
      local_domain.max_cell_halophyid = f['max_cell_halophyid'][()]

    int_precision = 'int32' if local_domain.int_precision == 32 else 'int64'
    float_precision = 'float32' if local_domain.float_precision == 32 else 'float64'
    if int_precision != types.INT_TYPE or float_precision != types.FLOAT_TYPE:
      raise RuntimeError(f"Stored local domain has different (float/int) type precision from what types.py has. Consider changing types.py (float/int) types Or load different domain domain_types={float_precision}/{int_precision} types={types.FLOAT_TYPE}/{types.INT_TYPE}")
    return local_domain