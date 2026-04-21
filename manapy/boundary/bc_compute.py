import numpy as np
from manapy.backends.compile_fun import compile

def _rhs_value_dirichlet_node(Pbordnode: 'float[:]', nodes: 'int[:]', value: 'float[:]'):
  for i in nodes:
    Pbordnode[i] = value[i]


def _rhs_value_dirichlet_face(Pbordface: 'float[:]', faces: 'int[:]', value: 'float[:]'):
  for i in faces:
    Pbordface[i] = value[i]


def _rhs_value_neumannNH_face(w_c: 'float[:]', Pbordface: 'float[:]', cellid: 'int[:,:]', faces: 'int[:]',
                             cst: 'float[:]', dist: 'float[:]'):
  for i in faces:
    val = w_c[cellid[i][0]] + cst[i] * dist[i]
    Pbordface[i] = (val + w_c[cellid[i][0]]) / 2.

#################################################################################
#################################################################################

def _ghost_value_nonslip(w_c: 'float[:]', w_ghost: 'float[:]', face_cellid: 'int[:,:]', bc_faces: 'int[:]',
                        cst: 'float[:]', face_dist_ortho: 'float[:]', face_ghost_id: 'int[:]'):
  for i in bc_faces:
    ghost_id = face_ghost_id[i]
    w_ghost[ghost_id] = -1 * w_c[face_cellid[i][0]]


def _ghost_value_neumann(w_c: 'float[:]', w_ghost: 'float[:]', face_cellid: 'int[:,:]', bc_faces: 'int[:]',
                        cst: 'float[:]', face_dist_ortho: 'float[:]', face_ghost_id: 'int[:]'):
  for i in bc_faces:
    ghost_id = face_ghost_id[i]
    w_ghost[ghost_id] = w_c[face_cellid[i][0]]


def _ghost_value_neumannNH(w_c: 'float[:]', w_ghost: 'float[:]', face_cellid: 'int[:,:]', bc_faces: 'int[:]',
                          cst: 'float[:]', face_dist_ortho: 'float[:]', face_ghost_id: 'int[:]'):
  for i in bc_faces:
    ghost_id = face_ghost_id[i]
    w_ghost[ghost_id] = w_c[face_cellid[i][0]] + cst[i] * face_dist_ortho[i]


def _ghost_value_dirichlet(value: 'float[:]', w_ghost: 'float[:]', face_cellid: 'int[:,:]', bc_faces: 'int[:]',
                          cst: 'float[:]', face_dist_ortho: 'float[:]', face_ghost_id: 'int[:]'):
  for i in bc_faces:
    ghost_id = face_ghost_id[i]
    w_ghost[ghost_id] = value[i]

#################################################################################
#################################################################################

def _haloghost_value_neumann(w_halo: 'float[:]', w_haloghost: 'float[:]', node_haloghostid: 'int[:, :]', ghost_ext_info_int: 'int[:,:]',
                             ghost_ext_info_flt: 'float[:, :]', BCindex: 'int', d_halonodes: 'int[:]', cst: 'float[:]'):
  for i in d_halonodes:
    for j in range(node_haloghostid[i, -1]):
      ghost_id = node_haloghostid[i, j]
      face_oldname = ghost_ext_info_int[ghost_id, 1]
      if face_oldname == BCindex:
        cellhalo = ghost_ext_info_int[ghost_id, 0]
        w_haloghost[ghost_id] = w_halo[cellhalo]


def _haloghost_value_neumannNH(w_halo: 'float[:]', w_haloghost: 'float[:]', node_haloghostid: 'int[:, :]', ghost_ext_info_int: 'int[:,:]',
                             ghost_ext_info_flt: 'float[:, :]', BCindex: 'int', d_halonodes: 'int[:]', cst: 'float[:]'):
  # TODO dist is not well computed (work only if NH is in the infaces)
  for i in d_halonodes:
    for j in range(node_haloghostid[i, -1]):
      ghost_id = node_haloghostid[i, j]
      face_oldname = ghost_ext_info_int[ghost_id, 1]
      if face_oldname == BCindex:
        cellhalo = ghost_ext_info_int[ghost_id, 0]

        # distance function is removed because the call can be reduced to return np.abs(ghost_ext_info_flt[ghost_id, 0])
        # dist(ghost_ext_info_flt[ghost_id][0:2], np.array([0., haloghostcenter[i][j][1]]))
        dist = 2 * np.abs(ghost_ext_info_flt[ghost_id, 0])
        w_haloghost[ghost_id] = w_halo[cellhalo] + cst[i] * dist


def _haloghost_value_dirichlet(w_halo: 'float[:]', w_haloghost: 'float[:]', node_haloghostid: 'int[:, :]', ghost_ext_info_int: 'int[:,:]',
                             ghost_ext_info_flt: 'float[:, :]', BCindex: 'int', d_halonodes: 'int[:]', cst: 'float[:]'):
  for i in d_halonodes:
    for j in range(node_haloghostid[i, -1]):
      ghost_id = node_haloghostid[i, j]
      face_oldname = ghost_ext_info_int[ghost_id, 1]
      if face_oldname == BCindex:
        w_haloghost[ghost_id] = w_halo[ghost_id]


def _haloghost_value_nonslip(w_halo: 'float[:]', w_haloghost: 'float[:]', node_haloghostid: 'int[:, :]', ghost_ext_info_int: 'int[:,:]',
                             ghost_ext_info_flt: 'float[:, :]', BCindex: 'int', d_halonodes: 'int[:]', cst: 'float[:]'):
  for i in d_halonodes:
    for j in range(node_haloghostid[i, -1]):
      ghost_id = node_haloghostid[i, j]
      face_oldname = ghost_ext_info_int[ghost_id, 1]
      if face_oldname == BCindex:
        w_haloghost[ghost_id] = -1 * w_halo[ghost_id]

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