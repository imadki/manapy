import numpy as np
from manapy.backends.compile_fun import compile

def _rhs_value_dirichlet_node(Pbordnode: 'float[:]', nodes: 'int32[:]', value: 'float[:]'):
  for i in nodes:
    Pbordnode[i] = value[i]


def _rhs_value_dirichlet_face(Pbordface: 'float[:]', faces: 'int32[:]', value: 'float[:]'):
  for i in faces:
    Pbordface[i] = value[i]


def _rhs_value_neumannNH_face(w_c: 'float[:]', Pbordface: 'float[:]', cellid: 'int32[:,:]', faces: 'int32[:]',
                             cst: 'float[:]', dist: 'float[:]'):
  for i in faces:
    val = w_c[cellid[i][0]] + cst[i] * dist[i]
    Pbordface[i] = (val + w_c[cellid[i][0]]) / 2.

#################################################################################
#################################################################################

def _ghost_value_nonslip(w_c: 'float[:]', w_ghost: 'float[:]', face_cellid: 'int32[:,:]', bc_faces: 'int32[:]',
                        cst: 'float[:]', face_dist_ortho: 'float[:]'):
  for i in bc_faces:
    w_ghost[i] = -1 * w_c[face_cellid[i][0]]


def _ghost_value_neumann(w_c: 'float[:]', w_ghost: 'float[:]', face_cellid: 'int32[:,:]', bc_faces: 'int32[:]',
                        cst: 'float[:]', face_dist_ortho: 'float[:]'):
  for i in bc_faces:
    w_ghost[i] = w_c[face_cellid[i][0]]


def _ghost_value_neumannNH(w_c: 'float[:]', w_ghost: 'float[:]', face_cellid: 'int32[:,:]', bc_faces: 'int32[:]',
                          cst: 'float[:]', face_dist_ortho: 'float[:]'):
  for i in bc_faces:
    w_ghost[i] = w_c[face_cellid[i][0]] + cst[i] * face_dist_ortho[i]


def _ghost_value_dirichlet(value: 'float[:]', w_ghost: 'float[:]', face_cellid: 'int32[:,:]', bc_faces: 'int32[:]',
                          cst: 'float[:]', face_dist_ortho: 'float[:]'):
  for i in bc_faces:
    w_ghost[i] = value[i]

#################################################################################
#################################################################################

def _haloghost_value_neumann(w_halo: 'float[:]', w_haloghost: 'float[:]', node_haloghostcenter: 'float[:,:,:]',
                            node_haloghostcenter_info: 'int32[:,:,:]', BCindex: 'int32', d_halonodes: 'int32[:]', cst: 'float[:]'):
  for i in d_halonodes:
    for j in range(node_haloghostcenter_info[i].shape[0]):
      if node_haloghostcenter_info[i, j, -1] != -1:
        if node_haloghostcenter_info[i, j, -2] == BCindex:
          cellhalo = node_haloghostcenter_info[i, j, -3]
          cellghost = node_haloghostcenter_info[i, j, -1]
          w_haloghost[cellghost] = w_halo[cellhalo]


def _haloghost_value_neumannNH(w_halo: 'float[:]', w_haloghost: 'float[:]', node_haloghostcenter: 'float[:,:,:]',
                              node_haloghostcenter_info: 'int32[:,:,:]', BCindex: 'int32', d_halonodes: 'int32[:]', cst: 'float[:]'):
  # TODO dist is not well computed (work only if NH is in the infaces)
  for i in d_halonodes:
    for j in range(node_haloghostcenter_info[i].shape[0]):
      if node_haloghostcenter_info[i, j, -1] != -1:
        if node_haloghostcenter_info[i, j, -2] == BCindex:
          cellhalo = node_haloghostcenter_info[i, j, -3]
          cellghost = node_haloghostcenter_info[i, j, -1]

          # distance function is removed because the call can be reduced to return np.abs(node_haloghostcenter[i, j, 0])
          dist = 2 * np.abs(node_haloghostcenter[i, j, 0])
          w_haloghost[cellghost] = w_halo[cellhalo] + cst[i] * dist


def _haloghost_value_dirichlet(value: 'float[:]', w_haloghost: 'float[:]', node_haloghostcenter: 'float[:,:,:]',
                              node_haloghostcenter_info: 'int32[:,:,:]', BCindex: 'int32', d_halonodes: 'int32[:]', cst: 'float[:]'):
  for i in d_halonodes:
    for j in range(node_haloghostcenter_info[i].shape[0]):
      if node_haloghostcenter_info[i, j, -1] != -1:
        if node_haloghostcenter_info[i, j, -2] == BCindex:
          cellghost = node_haloghostcenter_info[i, j, -1]
          w_haloghost[cellghost] = value[cellghost]


def _haloghost_value_nonslip(w_halo: 'float[:]', w_haloghost: 'float[:]', node_haloghostcenter: 'float[:,:,:]',
                            node_haloghostcenter_info: 'int32[:,:,:]', BCindex: 'int32', d_halonodes: 'int32[:]', cst: 'float[:]'):
  for i in d_halonodes:
    for j in range(node_haloghostcenter_info[i].shape[0]):
      if node_haloghostcenter_info[i, j, -1] != -1:
        if node_haloghostcenter_info[i, j, -2] == BCindex:
          cellghost = node_haloghostcenter_info[i, j, -1]
          w_haloghost[cellghost] = -1 * w_halo[cellghost]

#################################################################################
#################################################################################
# public
ghost_value_neumann = compile(_ghost_value_neumann)
ghost_value_dirichlet = compile(_ghost_value_dirichlet)
ghost_value_nonslip = compile(_ghost_value_nonslip)
haloghost_value_nonslip = compile(_haloghost_value_nonslip)
ghost_value_neumannNH = compile(_ghost_value_neumannNH)
haloghost_value_neumannNH = compile(_haloghost_value_neumannNH)
haloghost_value_dirichlet = compile(_haloghost_value_dirichlet)
haloghost_value_neumann = compile(_haloghost_value_neumann)