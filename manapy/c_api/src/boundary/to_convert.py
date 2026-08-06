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


# NB: unlike the other haloghost bodies, the first array here is NOT the halo
# cell field -- it is the prescribed per-halo-ghost value array (Variable's
# `valuehalo`, sized halos.sizehaloghost and evaluated at each ghost's face
# centre), hence the ghost_id indexing. Named accordingly to avoid confusion.
def haloghost_value_dirichlet(start: 'int', stride: 'int', value_haloghost: 'float[:]', w_haloghost: 'float[:]',
                              node_haloghostid: 'int[:, :]', ghost_ext_info_int: 'int[:,:]',
                              ghost_ext_info_flt: 'float[:, :]', BCindex: 'int', d_halonodes: 'int[:]',
                              cst: 'float[:]'):
  for idx in range(start, d_halonodes.shape[0], stride):
    i = d_halonodes[idx]
    for j in range(node_haloghostid[i, -1]):
      ghost_id = node_haloghostid[i, j]
      if ghost_ext_info_int[ghost_id, 1] == BCindex:
        w_haloghost[ghost_id] = value_haloghost[ghost_id]


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
        # cst is per halo ghost (Variable's `valuehalo`), not per node: a ghost
        # is reachable from every node of its face, and those nodes carry
        # different values (a corner node even carries a *neighbouring*
        # boundary's tag), so indexing by the node made the result depend on
        # which node visited the ghost last.
        w_haloghost[ghost_id] = w_halo[ghost_ext_info_int[ghost_id, 0]] + cst[ghost_id] * dist


def haloghost_value_nonslip(start: 'int', stride: 'int', w_halo: 'float[:]', w_haloghost: 'float[:]',
                            node_haloghostid: 'int[:, :]', ghost_ext_info_int: 'int[:,:]',
                            ghost_ext_info_flt: 'float[:, :]', BCindex: 'int', d_halonodes: 'int[:]',
                            cst: 'float[:]'):
  for idx in range(start, d_halonodes.shape[0], stride):
    i = d_halonodes[idx]
    for j in range(node_haloghostid[i, -1]):
      ghost_id = node_haloghostid[i, j]
      if ghost_ext_info_int[ghost_id, 1] == BCindex:
        # w_halo is the halo *cell* field, so it must be indexed by the halo
        # cell behind the ghost (column 0), exactly as haloghost_value_neumann
        # does -- not by ghost_id, which lives in a different index space
        # (halos.sizehaloghost vs nbhalos).
        w_haloghost[ghost_id] = -1.0 * w_halo[ghost_ext_info_int[ghost_id, 0]]


# ---------------------------------------------------------------------------
# Free-slip (slip wall): vector boundary condition.
#
# The velocity vector is reflected across the boundary face so that the normal
# component vanishes at the face while the tangential component is preserved:
#       U_ghost = U_c - 2 (U_c . n) n
# This couples the velocity components, so (unlike the scalar BCs above) the
# kernels take all components together plus the face normal. The normal is
# normalised internally, so it works whether it is unit or area-scaled.
# ---------------------------------------------------------------------------
def ghost_value_slip_2d(start: 'int', stride: 'int', u_c: 'float[:]', v_c: 'float[:]',
                        u_ghost: 'float[:]', v_ghost: 'float[:]', face_cellid: 'int[:,:]',
                        bc_faces: 'int[:]', normal: 'float[:,:]'):
  for idx in range(start, bc_faces.shape[0], stride):
    i = bc_faces[idx]
    c = face_cellid[i][0]
    nx = normal[i][0]
    ny = normal[i][1]
    nrm = (nx * nx + ny * ny) ** 0.5
    nx = nx / nrm
    ny = ny / nrm
    udotn = u_c[c] * nx + v_c[c] * ny
    u_ghost[i] = u_c[c] - 2.0 * udotn * nx
    v_ghost[i] = v_c[c] - 2.0 * udotn * ny


def ghost_value_slip_3d(start: 'int', stride: 'int', u_c: 'float[:]', v_c: 'float[:]',
                        w_c: 'float[:]', u_ghost: 'float[:]', v_ghost: 'float[:]',
                        w_ghost: 'float[:]', face_cellid: 'int[:,:]', bc_faces: 'int[:]',
                        normal: 'float[:,:]'):
  for idx in range(start, bc_faces.shape[0], stride):
    i = bc_faces[idx]
    c = face_cellid[i][0]
    nx = normal[i][0]
    ny = normal[i][1]
    nz = normal[i][2]
    nrm = (nx * nx + ny * ny + nz * nz) ** 0.5
    nx = nx / nrm
    ny = ny / nrm
    nz = nz / nrm
    udotn = u_c[c] * nx + v_c[c] * ny + w_c[c] * nz
    u_ghost[i] = u_c[c] - 2.0 * udotn * nx
    v_ghost[i] = v_c[c] - 2.0 * udotn * ny
    w_ghost[i] = w_c[c] - 2.0 * udotn * nz


def haloghost_value_slip_2d(start: 'int', stride: 'int', u_halo: 'float[:]', v_halo: 'float[:]',
                            u_haloghost: 'float[:]', v_haloghost: 'float[:]',
                            node_haloghostid: 'int[:, :]', ghost_ext_info_int: 'int[:,:]',
                            ghost_ext_info_flt: 'float[:, :]', BCindex: 'int', d_halonodes: 'int[:]'):
  for idx in range(start, d_halonodes.shape[0], stride):
    i = d_halonodes[idx]
    for j in range(node_haloghostid[i, -1]):
      ghost_id = node_haloghostid[i, j]
      if ghost_ext_info_int[ghost_id, 1] == BCindex:
        c = ghost_ext_info_int[ghost_id, 0]
        nx = ghost_ext_info_flt[ghost_id, 7]
        ny = ghost_ext_info_flt[ghost_id, 8]
        nrm = (nx * nx + ny * ny) ** 0.5
        nx = nx / nrm
        ny = ny / nrm
        udotn = u_halo[c] * nx + v_halo[c] * ny
        u_haloghost[ghost_id] = u_halo[c] - 2.0 * udotn * nx
        v_haloghost[ghost_id] = v_halo[c] - 2.0 * udotn * ny


def haloghost_value_slip_3d(start: 'int', stride: 'int', u_halo: 'float[:]', v_halo: 'float[:]',
                            w_halo: 'float[:]', u_haloghost: 'float[:]', v_haloghost: 'float[:]',
                            w_haloghost: 'float[:]', node_haloghostid: 'int[:, :]',
                            ghost_ext_info_int: 'int[:,:]', ghost_ext_info_flt: 'float[:, :]',
                            BCindex: 'int', d_halonodes: 'int[:]'):
  for idx in range(start, d_halonodes.shape[0], stride):
    i = d_halonodes[idx]
    for j in range(node_haloghostid[i, -1]):
      ghost_id = node_haloghostid[i, j]
      if ghost_ext_info_int[ghost_id, 1] == BCindex:
        c = ghost_ext_info_int[ghost_id, 0]
        nx = ghost_ext_info_flt[ghost_id, 7]
        ny = ghost_ext_info_flt[ghost_id, 8]
        nz = ghost_ext_info_flt[ghost_id, 9]
        nrm = (nx * nx + ny * ny + nz * nz) ** 0.5
        nx = nx / nrm
        ny = ny / nrm
        nz = nz / nrm
        udotn = u_halo[c] * nx + v_halo[c] * ny + w_halo[c] * nz
        u_haloghost[ghost_id] = u_halo[c] - 2.0 * udotn * nx
        v_haloghost[ghost_id] = v_halo[c] - 2.0 * udotn * ny
        w_haloghost[ghost_id] = w_halo[c] - 2.0 * udotn * nz
