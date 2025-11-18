from manapy.domain import Mesh, Partitioning
from manapy.tests import LocalDomain1Cpu
import h5py
import numpy as np
from manapy.tests.meshes import get_mesh



# Create global domain
dim, mesh_path, mesh_name = get_mesh(3)
mesh = Mesh(mesh_path, dim)
partitioning = Partitioning(mesh)
local_domain_data = partitioning.create_sub_domains()
global_domain = LocalDomain1Cpu.create_local_domains(local_domain_data)

def _create_face_to_phyid(nb_faces, phyid_to_faceid: 'int32[:]'):
  face_to_phyid = np.ones(shape=nb_faces, dtype=np.int32) * -1
  face_to_phyid[phyid_to_faceid] = np.arange(phyid_to_faceid.shape[0])
  return face_to_phyid

def _remap_fid_to_phyid(cell_ghostnid, node_ghostid, face_to_phyid):
  for i in range(cell_ghostnid.shape[0]):
    cg = cell_ghostnid[i]
    for j in range(cg[-1]):
      fid = cg[j]
      cg[j] = face_to_phyid[fid]

  for i in range(node_ghostid.shape[0]):
    ng = node_ghostid[i]
    for j in range(ng[-1]):
      fid = ng[j]
      ng[j] = face_to_phyid[fid]


# Create the hd5 file
path = 'hybrid_test_tables.hd5'
with h5py.File(path, 'w') as f:
  face_to_phyid = _create_face_to_phyid(len(global_domain[0].faces), global_domain[0].phyid_to_faceid)
  _remap_fid_to_phyid(global_domain[0].cell_ghostnid, global_domain[0].node_ghostid, face_to_phyid)

  f.create_dataset('cells', data=global_domain[0].cells)
  f.create_dataset('cell_center', data=global_domain[0].cell_center)
  f.create_dataset('cell_volume', data=global_domain[0].cell_volume)
  f.create_dataset('cell_cellfid', data=global_domain[0].cell_cellfid)
  f.create_dataset('cell_cellnid', data=global_domain[0].cell_cellnid)
  f.create_dataset('cell_faceid', data=global_domain[0].cell_faceid)
  f.create_dataset('faces', data=global_domain[0].faces)
  f.create_dataset('face_cellid', data=global_domain[0].face_cellid)
  f.create_dataset('face_center', data=global_domain[0].face_center)
  f.create_dataset('face_oldname', data=global_domain[0].face_oldname)
  f.create_dataset('face_to_phyid', data=face_to_phyid)
  f.create_dataset('phy_faces', data=global_domain[0].phy_faces)
  f.create_dataset('face_normal', data=global_domain[0].face_normal)
  f.create_dataset('face_measure', data=global_domain[0].face_measure)
  f.create_dataset('face_tangent', data=global_domain[0].face_tangent)
  f.create_dataset('face_binormal', data=global_domain[0].face_binormal)
  f.create_dataset('nodes', data=global_domain[0].nodes)
  f.create_dataset('node_cellid', data=global_domain[0].node_cellid)
  f.create_dataset('node_oldname', data=global_domain[0].node_oldname)
  f.create_dataset('shared_ghost_info', data=global_domain[0].shared_ghost_info)
  f.create_dataset('cell_ghostnid', data=global_domain[0].cell_ghostnid)
  f.create_dataset('node_ghostid', data=global_domain[0].node_ghostid)