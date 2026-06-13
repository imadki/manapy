# -*- coding: utf-8 -*-
import numpy as np


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


# Corps grid-stride: meme source pour CPU et GPU quand le kernel s'y prete.
def ghost_value_dirichlet(start: 'int', stride: 'int', value: 'float[:]', w_ghost: 'float[:]',
                          face_cellid: 'int[:,:]', bc_faces: 'int[:]', cst: 'float[:]',
                          face_dist_ortho: 'float[:]'):
  for idx in range(start, bc_faces.shape[0], stride):
    i = bc_faces[idx]
    w_ghost[i] = value[i]


def ghost_value_neumann(start: 'int', stride: 'int', w_c: 'float[:]', w_ghost: 'float[:]',
                        face_cellid: 'int[:,:]', bc_faces: 'int[:]', cst: 'float[:]',
                        face_dist_ortho: 'float[:]'):
  for idx in range(start, bc_faces.shape[0], stride):
    i = bc_faces[idx]
    w_ghost[i] = w_c[face_cellid[i][0]]


def ghost_value_neumannNH(start: 'int', stride: 'int', w_c: 'float[:]', w_ghost: 'float[:]',
                          face_cellid: 'int[:,:]', bc_faces: 'int[:]', cst: 'float[:]',
                          face_dist_ortho: 'float[:]'):
  for idx in range(start, bc_faces.shape[0], stride):
    i = bc_faces[idx]
    w_ghost[i] = w_c[face_cellid[i][0]] + cst[i] * face_dist_ortho[i]


def ghost_value_nonslip(start: 'int', stride: 'int', w_c: 'float[:]', w_ghost: 'float[:]',
                        face_cellid: 'int[:,:]', bc_faces: 'int[:]', cst: 'float[:]',
                        face_dist_ortho: 'float[:]'):
  for idx in range(start, bc_faces.shape[0], stride):
    i = bc_faces[idx]
    w_ghost[i] = -1.0 * w_c[face_cellid[i][0]]


def haloghost_value_neumann(start: 'int', stride: 'int', w_halo: 'float[:]', w_haloghost: 'float[:]',
                            node_haloghostid: 'int[:, :]', ghost_ext_info_int: 'int[:,:]',
                            ghost_ext_info_flt: 'float[:, :]', BCindex: 'int', d_halonodes: 'int[:]',
                            cst: 'float[:]'):
  for idx in range(start, d_halonodes.shape[0], stride):
    i = d_halonodes[idx]
    for j in range(node_haloghostid[i, -1]):
      ghost_id = node_haloghostid[i, j]
      if ghost_ext_info_int[ghost_id, 1] == BCindex:
        w_haloghost[ghost_id] = w_halo[ghost_ext_info_int[ghost_id, 0]]


def haloghost_value_dirichlet(start: 'int', stride: 'int', w_halo: 'float[:]', w_haloghost: 'float[:]',
                              node_haloghostid: 'int[:, :]', ghost_ext_info_int: 'int[:,:]',
                              ghost_ext_info_flt: 'float[:, :]', BCindex: 'int', d_halonodes: 'int[:]',
                              cst: 'float[:]'):
  for idx in range(start, d_halonodes.shape[0], stride):
    i = d_halonodes[idx]
    for j in range(node_haloghostid[i, -1]):
      ghost_id = node_haloghostid[i, j]
      if ghost_ext_info_int[ghost_id, 1] == BCindex:
        w_haloghost[ghost_id] = w_halo[ghost_id]


def haloghost_value_neumannNH(start: 'int', stride: 'int', w_halo: 'float[:]', w_haloghost: 'float[:]',
                              node_haloghostid: 'int[:, :]', ghost_ext_info_int: 'int[:,:]',
                              ghost_ext_info_flt: 'float[:, :]', BCindex: 'int', d_halonodes: 'int[:]',
                              cst: 'float[:]'):
  for idx in range(start, d_halonodes.shape[0], stride):
    i = d_halonodes[idx]
    for j in range(node_haloghostid[i, -1]):
      ghost_id = node_haloghostid[i, j]
      if ghost_ext_info_int[ghost_id, 1] == BCindex:
        dist = 2.0 * abs(ghost_ext_info_flt[ghost_id, 0])
        w_haloghost[ghost_id] = w_halo[ghost_ext_info_int[ghost_id, 0]] + cst[i] * dist


def haloghost_value_nonslip(start: 'int', stride: 'int', w_halo: 'float[:]', w_haloghost: 'float[:]',
                            node_haloghostid: 'int[:, :]', ghost_ext_info_int: 'int[:,:]',
                            ghost_ext_info_flt: 'float[:, :]', BCindex: 'int', d_halonodes: 'int[:]',
                            cst: 'float[:]'):
  for idx in range(start, d_halonodes.shape[0], stride):
    i = d_halonodes[idx]
    for j in range(node_haloghostid[i, -1]):
      ghost_id = node_haloghostid[i, j]
      if ghost_ext_info_int[ghost_id, 1] == BCindex:
        w_haloghost[ghost_id] = -1.0 * w_halo[ghost_id]


GHOST_BODIES = {
  'dirichlet': ghost_value_dirichlet,
  'neumann':   ghost_value_neumann,
  'periodic':  ghost_value_neumann,
  'neumannNH': ghost_value_neumannNH,
  'nonslip':   ghost_value_nonslip,
}

HALOGHOST_BODIES = {
  'dirichlet': haloghost_value_dirichlet,
  'neumann':   haloghost_value_neumann,
  'periodic':  haloghost_value_neumann,
  'neumannNH': haloghost_value_neumannNH,
  'nonslip':   haloghost_value_nonslip,
}
