from manapy.domain import Mesh, Partitioning
from manapy.testing import LocalDomain1Cpu
from manapy.helpers import get_mesh

# To change float type go to manapy/backends/types.py
dim, mesh_path, mesh_name = get_mesh(1)
mesh = Mesh(mesh_path, dim)
partitioning = Partitioning(mesh)
nb_parts = 16
part_vert = partitioning.make_n_part_mesh_nodal(nb_parts)
local_domains = partitioning.create_sub_domains() # intermediate step

ld = LocalDomain1Cpu.create_local_domains(local_domains) # list of local domains


print(ld[0].node_haloghostid)
# print(ld[0].cell_center)
# print(ld[0].cell_volume)
# print("cell_cellfid\n", ld[0].cell_cellfid)
# print("cell_cellnid\n", ld[0].cell_cellnid)
# print("cell_faceid\n", ld[0].cell_faceid)
# print("faces\n", ld[0].faces)
# print("face_cellid\n", ld[0].face_cellid)
# # print(ld[0].face_center)
# print("face_oldname\n", ld[0].face_oldname)
# print("phy_faces\n", ld[0].phy_faces)
# # print(ld[0].face_normal)
# # print(ld[0].face_measure)
# # print(ld[0].face_tangent)
# # print(ld[0].face_binormal)
# print("nodes\n", ld[0].nodes)
# print("node_cellid\n", ld[0].node_cellid)
# print("node_oldname\n", ld[0].node_oldname)
# # print(ld[0].ghost_info_flt)
# # print(ld[0].ghost_info_int)
# print("cell_ghostnid\n", ld[0].cell_ghostnid)
# print("node_ghostid\n", ld[0].node_ghostid)
# print(ld[0].phyid_to_faceid)