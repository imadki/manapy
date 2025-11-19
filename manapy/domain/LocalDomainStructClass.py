import numpy as np
from numba.typed import Dict, List
import h5py

# Created inside PartitioningClass
class LocalDomainStruct:
  def __init__(self):
    # Arrays: use zeros(1) as placeholder
    # Returned tables and Scalars
    self.nodes = np.zeros((1, 1), dtype=np.float64) # [[node x, y, z]]
    self.cells = np.zeros((1, 1), dtype=np.int32) # [[cells nodes]]
    self.cells_type = np.zeros(1, dtype=np.int8) # [cell type]
    self.phy_faces = np.zeros((1, 1), dtype=np.int32) # [[physical face nodes]]
    self.phy_faces_name = np.zeros(1, dtype=np.int32) # [physical face name]
    self.cell_loctoglob = np.zeros(1, dtype=np.int32) # [cell global index]
    self.node_loctoglob = np.zeros(1, dtype=np.int32) # [node global index]
    self.node_oldname = np.zeros(1, dtype=np.int32) # [node old name, ...]
    self.halo_neighsub = np.zeros((1, 1), dtype=np.int32) # [[NeighborP1, NeighborP2, ...], [NbHalosIntConnectedToP1, ...]]
    self.node_halos = np.zeros(1, dtype=np.int32) # int32[:] [NodiId, haloId, ...] shape=(2 * nb_halos) couple (NodeId, haloId) for each exthalo, HaloId is an index point to halo_halosext, nodeId is the local nodeId.
    self.node_halophyid = np.zeros((1, 1), dtype=np.int32) # [[index0 point to halo_halobf, index1 ..., size]] shape=(nb_nodes, max_node_halobf + 1)
    self.halo_halosext = np.zeros((1, 1), dtype=np.int32) # [[global index of halocell, global index of cell nodes, size]] shape=(nb_halos, max_cell_nodeid + 2) Halos of a partition P is the Concatenation of Interiors of the neighbor parts that are connected to P.
    self.halo_halosint = np.zeros(1, dtype=np.int32) # [HalosIntConnectedToP1 halos ..., HalosIntConnectedToP2 halos ..., ...]
    self.halo_centvol = np.zeros((1, 1), dtype=np.float64)  # [halocell_center_{x, y, z}, halocell_volume_{x, y, z}] # z axis only on 3D
    self.phyid_recv = np.zeros(1, dtype=np.int32) # [boundary faces global index, ...] description="represent the global index of boundary faces that is needed from this partition either from itself or the other paritions, all other tables that will use boundary faces must point to this table"
    self.phyid_recv_part_size = np.zeros(1, dtype=np.int32) # [boundary faces part, size]
    self.phyid_send = np.zeros(1, dtype=np.int32) # [recv_part_index, size, size indices point to phyid_recv, ...] description="used when this part need to send its boundary faces to recv_part"

    # Scalars
    self.max_cell_nodeid = 0
    self.max_cell_faceid = 0
    self.max_face_nodeid = 0
    self.max_node_haloid = 0
    self.max_cell_halonid = 0
    self.float_precision = 0 # 32 or 64
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
      f.create_dataset('phyid_recv', data=ld.phyid_recv)
      f.create_dataset('phyid_recv_part_size', data=ld.phyid_recv_part_size)
      f.create_dataset('phyid_send', data=ld.phyid_send)
      f.create_dataset('halo_halosext', data=ld.halo_halosext)
      f.create_dataset('halo_centvol', data=ld.halo_centvol)
      f.create_dataset('dim', data=ld.dim)
      f.create_dataset('float_precision', data=ld.float_precision)
      f.create_dataset('max_cell_nodeid', data=ld.max_cell_nodeid)
      f.create_dataset('max_cell_faceid', data=ld.max_cell_faceid)
      f.create_dataset('max_face_nodeid', data=ld.max_face_nodeid)
      f.create_dataset('max_node_haloid', data=ld.max_node_haloid)
      f.create_dataset('max_cell_halonid', data=ld.max_cell_halonid)

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
      local_domain.phyid_recv = f['phyid_recv'][...]
      local_domain.phyid_recv_part_size = f['phyid_recv_part_size'][...]
      local_domain.phyid_send = f['phyid_send'][...]
      local_domain.halo_halosext = f['halo_halosext'][...]
      local_domain.halo_centvol = f['halo_centvol'][...]
      local_domain.dim = f['dim'][()]
      local_domain.float_precision = f['float_precision'][()]
      local_domain.max_cell_nodeid = f['max_cell_nodeid'][()]
      local_domain.max_cell_faceid = f['max_cell_faceid'][()]
      local_domain.max_face_nodeid = f['max_face_nodeid'][()]
      local_domain.max_node_haloid = f['max_node_haloid'][()]
      local_domain.max_cell_halonid = f['max_cell_halonid'][()]

    return local_domain