# -*- coding: utf-8 -*-
"""
GPU (numba.cuda) port of manapy/solvers/ls/ls_compute.py.

Same approach as advec/cuda_fvm_utils.py: each get_kernel_*() returns a wrapper
whose signature is IDENTICAL to the matching CPU kernel in ls_compute.py, so it
is a drop-in for LinearSolver's _compute_P_gradient / _get_triplet / _get_rhs.

Parallelisation:
  - per-face kernels that write their own slot (e.g. gradient -> Px_face[i]) map
    one thread per face index, no atomics;
  - assembly kernels that APPEND a data-dependent number of COO triplets use a
    global atomic counter to reserve a contiguous block per face (order becomes
    non-deterministic, which is harmless: Ginkgo read_distributed and scipy
    csr_matrix both SUM duplicate (row,col) entries and ignore triplet order).

This file is imported lazily by LinearSolver only when domain.backend.name=="gpu".
"""
from numba import cuda

from manapy.backends.gpu import get_active_backend, GPUArray


# ---------------------------------------------------------------------------
# Device helper: membership test (mirrors ls_diamond._search_element).
def device_search_element(a: 'int[:]', target_value: 'int') -> 'int':
  find = 0
  for k in range(a.shape[0]):
    if a[k] == target_value:
      find = 1
      break
  return find


# ---------------------------------------------------------------------------
def get_kernel_get_rhs_loc_2d():
  gpu = get_active_backend()
  search = gpu.compile_kernel(device_search_element, device=True)

  def kernel(face_cellid: 'int[:,:]', faces: 'int[:,:]', node_oldname: 'int[:]',
             cell_volume: 'float[:]', node_ghostid: 'int[:,:]', cell_loctoglob: 'int[:]',
             face_param1: 'float[:]', face_param2: 'float[:]', face_param3: 'float[:]',
             face_param4: 'float[:]', Pbordnode: 'float[:]', Pbordface: 'float[:]',
             rhs_loc: 'float[:]', BCdirichlet: 'int[:]', matrixinnerfaces: 'int[:]',
             d_halofaces: 'int[:]', dirichletfaces: 'int[:]'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for idx in range(start, matrixinnerfaces.shape[0], stride):
      i = matrixinnerfaces[idx]
      c_right = face_cellid[i][1]; c_left = face_cellid[i][0]
      i_1 = faces[i][0]; i_2 = faces[i][1]
      if search(BCdirichlet, node_oldname[i_1]) == 1:
        V = Pbordnode[i_1]
        cuda.atomic.add(rhs_loc, c_left, -1.0 * V * face_param4[i] / cell_volume[c_left])
        cuda.atomic.add(rhs_loc, c_right, V * face_param4[i] / cell_volume[c_right])
      if search(BCdirichlet, node_oldname[i_2]) == 1:
        V = Pbordnode[i_2]
        cuda.atomic.add(rhs_loc, c_left, -1.0 * V * face_param2[i] / cell_volume[c_left])
        cuda.atomic.add(rhs_loc, c_right, V * face_param2[i] / cell_volume[c_right])

    for idx in range(start, d_halofaces.shape[0], stride):
      i = d_halofaces[idx]
      c_left = face_cellid[i][0]
      i_1 = faces[i][0]; i_2 = faces[i][1]
      if search(BCdirichlet, node_oldname[i_1]) == 1:
        V = Pbordnode[i_1]
        cuda.atomic.add(rhs_loc, c_left, -1.0 * V * face_param4[i] / cell_volume[c_left])
      if search(BCdirichlet, node_oldname[i_2]) == 1:
        V = Pbordnode[i_2]
        cuda.atomic.add(rhs_loc, c_left, -1.0 * V * face_param2[i] / cell_volume[c_left])

    for idx in range(start, dirichletfaces.shape[0], stride):
      i = dirichletfaces[idx]
      c_left = face_cellid[i][0]
      i_1 = faces[i][0]; i_2 = faces[i][1]
      if node_ghostid[i_1, -1] > 0:
        V = Pbordnode[i_1]
        cuda.atomic.add(rhs_loc, c_left, -1.0 * V * face_param4[i] / cell_volume[c_left])
      if node_ghostid[i_2, -1] > 0:
        V = Pbordnode[i_2]
        cuda.atomic.add(rhs_loc, c_left, -1.0 * V * face_param2[i] / cell_volume[c_left])
      cuda.atomic.add(rhs_loc, c_left,
                      -2.0 * face_param3[i] / cell_volume[c_left] * Pbordface[i])

  kernel = gpu.compile_kernel(kernel)

  argcache = {}

  def result(*args):
    args = GPUArray.to_device_list(argcache, args)
    rhs_loc = args[12]
    gpu.assign(rhs_loc, 0.0)
    size = max(len(args[14]), len(args[15]), len(args[16]))
    grid, block = gpu.get_gpu_params(size)
    kernel[grid, block, gpu.stream](*args)
    if gpu.stream is not None:
      gpu.stream.synchronize()

  return result


# ---------------------------------------------------------------------------
def get_kernel_get_triplet_2d():
  """COO matrix-triplet assembly. Each face reserves slots in a_loc/irn_loc/
  jcn_loc via a global atomic counter (one atomic.add per emitted triplet);
  triplet ORDER is non-deterministic but the total count matches the CPU exactly
  (same branches) and Ginkgo/scipy sum duplicate (row,col) entries."""
  gpu = get_active_backend()
  search = gpu.compile_kernel(device_search_element, device=True)

  def kernel(face_cellid: 'int[:,:]', faces: 'int[:,:]', nodes: 'float[:,:]',
             face_haloid: 'int[:]', halo_halosext: 'int[:,:]', node_oldname: 'int[:]',
             cell_volume: 'float[:]', node_cellid: 'int[:,:]', cell_center: 'float[:,:]',
             halo_centvol: 'float[:,:]', node_haloid: 'int[:,:]', node_periodicid: 'int[:,:]',
             ghost_info_flt: 'float[:,:]', ghost_ext_info_flt: 'float[:,:]',
             ghost_info_int: 'int[:,:]', ghost_ext_info_int: 'int[:,:]',
             node_ghostid: 'int[:,:]', node_haloghostid: 'int[:,:]',
             face_air_diamond: 'float[:]', node_lambda_x: 'float[:]',
             node_lambda_y: 'float[:]', node_lambda_z: 'float[:]', node_number: 'int[:]',
             node_R_x: 'float[:]', node_R_y: 'float[:]', node_R_z: 'float[:]',
             face_param1: 'float[:]', face_param2: 'float[:]', face_param3: 'float[:]',
             face_param4: 'float[:]', cell_shift: 'float[:,:]', nbelements: 'int',
             cell_loctoglob: 'int[:]', BCdirichlet: 'int[:]', a_loc: 'float[:]',
             irn_loc: 'int[:]', jcn_loc: 'int[:]', matrixinnerfaces: 'int[:]',
             d_halofaces: 'int[:]', dirichletfaces: 'int[:]', cnt: 'int[:]'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    center = cuda.local.array(2, cell_center.dtype)
    parameters = cuda.local.array(2, cell_center.dtype)

    # ---- inner faces (left + right contributions) ----
    for idx in range(start, matrixinnerfaces.shape[0], stride):
      i = matrixinnerfaces[idx]
      c_left = face_cellid[i][0]; c_leftglob = cell_loctoglob[c_left]
      c_right = face_cellid[i][1]; c_rightglob = cell_loctoglob[c_right]
      parameters[0] = face_param4[i]; parameters[1] = face_param2[i]

      p = cuda.atomic.add(cnt, 0, 1)
      irn_loc[p] = c_leftglob; jcn_loc[p] = c_leftglob
      a_loc[p] = face_param1[i] / cell_volume[c_left]

      cmptparam = 0
      for kn in range(faces[i][-1]):
        nod = faces[i][kn]
        if search(BCdirichlet, node_oldname[nod]) == 0:
          for j in range(node_cellid[nod][-1]):
            center[0] = cell_center[node_cellid[nod][j]][0]
            center[1] = cell_center[node_cellid[nod][j]][1]
            xdiff = center[0] - nodes[nod][0]; ydiff = center[1] - nodes[nod][1]
            alpha = (1.0 + node_lambda_x[nod] * xdiff + node_lambda_y[nod] * ydiff) / (
                node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
            jg = cell_loctoglob[node_cellid[nod][j]]
            p = cuda.atomic.add(cnt, 0, 1)
            irn_loc[p] = c_leftglob; jcn_loc[p] = jg
            a_loc[p] = alpha / cell_volume[c_left] * parameters[cmptparam]
            p = cuda.atomic.add(cnt, 0, 1)
            irn_loc[p] = c_rightglob; jcn_loc[p] = jg
            a_loc[p] = -1.0 * alpha / cell_volume[c_right] * parameters[cmptparam]
          for j in range(node_ghostid[nod, -1]):
            gid = node_ghostid[nod, j]
            center[0] = ghost_info_flt[gid][0]; center[1] = ghost_info_flt[gid][1]
            xdiff = center[0] - nodes[nod][0]; ydiff = center[1] - nodes[nod][1]
            alpha = (1.0 + node_lambda_x[nod] * xdiff + node_lambda_y[nod] * ydiff) / (
                node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
            jg = cell_loctoglob[ghost_info_int[gid, 0]]
            p = cuda.atomic.add(cnt, 0, 1)
            irn_loc[p] = c_leftglob; jcn_loc[p] = jg
            a_loc[p] = alpha / cell_volume[c_left] * parameters[cmptparam]
            p = cuda.atomic.add(cnt, 0, 1)
            irn_loc[p] = c_rightglob; jcn_loc[p] = jg
            a_loc[p] = -1.0 * alpha / cell_volume[c_right] * parameters[cmptparam]
          for j in range(node_haloghostid[nod, -1]):
            gid = node_haloghostid[nod, j]
            center[0] = ghost_ext_info_flt[gid][0]; center[1] = ghost_ext_info_flt[gid][1]
            xdiff = center[0] - nodes[nod][0]; ydiff = center[1] - nodes[nod][1]
            alpha = (1.0 + node_lambda_x[nod] * xdiff + node_lambda_y[nod] * ydiff) / (
                node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
            jg = halo_halosext[ghost_ext_info_int[gid, 0]][0]
            p = cuda.atomic.add(cnt, 0, 1)
            irn_loc[p] = c_leftglob; jcn_loc[p] = jg
            a_loc[p] = alpha / cell_volume[c_left] * parameters[cmptparam]
            p = cuda.atomic.add(cnt, 0, 1)
            irn_loc[p] = c_rightglob; jcn_loc[p] = jg
            a_loc[p] = -1.0 * alpha / cell_volume[c_right] * parameters[cmptparam]
          for j in range(node_periodicid[nod][-1]):
            if nodes[nod][3] == 11 or nodes[nod][3] == 22:
              center[0] = cell_center[node_periodicid[nod][j]][0] + cell_shift[node_periodicid[nod][j]][0]
              center[1] = cell_center[node_periodicid[nod][j]][1]
            if nodes[nod][3] == 33 or nodes[nod][3] == 44:
              center[0] = cell_center[node_periodicid[nod][j]][0]
              center[1] = cell_center[node_periodicid[nod][j]][1] + cell_shift[node_periodicid[nod][j]][1]
            xdiff = center[0] - nodes[nod][0]; ydiff = center[1] - nodes[nod][1]
            alpha = (1.0 + node_lambda_x[nod] * xdiff + node_lambda_y[nod] * ydiff) / (
                node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
            jg = cell_loctoglob[node_periodicid[nod][j]]
            p = cuda.atomic.add(cnt, 0, 1)
            irn_loc[p] = c_leftglob; jcn_loc[p] = jg
            a_loc[p] = alpha / cell_volume[c_left] * parameters[cmptparam]
            p = cuda.atomic.add(cnt, 0, 1)
            irn_loc[p] = c_rightglob; jcn_loc[p] = jg
            a_loc[p] = -1.0 * alpha / cell_volume[c_right] * parameters[cmptparam]
          for j in range(node_haloid[nod][-1]):
            center[0] = halo_centvol[node_haloid[nod][j]][0]; center[1] = halo_centvol[node_haloid[nod][j]][1]
            xdiff = center[0] - nodes[nod][0]; ydiff = center[1] - nodes[nod][1]
            alpha = (1.0 + node_lambda_x[nod] * xdiff + node_lambda_y[nod] * ydiff) / (
                node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
            jg = halo_halosext[node_haloid[nod][j]][0]
            p = cuda.atomic.add(cnt, 0, 1)
            irn_loc[p] = c_leftglob; jcn_loc[p] = jg
            a_loc[p] = alpha / cell_volume[c_left] * parameters[cmptparam]
            p = cuda.atomic.add(cnt, 0, 1)
            irn_loc[p] = c_rightglob; jcn_loc[p] = jg
            a_loc[p] = -1.0 * alpha / cell_volume[c_right] * parameters[cmptparam]
        cmptparam += 1

      p = cuda.atomic.add(cnt, 0, 1)
      irn_loc[p] = c_leftglob; jcn_loc[p] = c_rightglob
      a_loc[p] = face_param3[i] / cell_volume[c_left]
      p = cuda.atomic.add(cnt, 0, 1)
      irn_loc[p] = c_rightglob; jcn_loc[p] = c_leftglob
      a_loc[p] = -1.0 * face_param1[i] / cell_volume[c_right]
      p = cuda.atomic.add(cnt, 0, 1)
      irn_loc[p] = c_rightglob; jcn_loc[p] = c_rightglob
      a_loc[p] = -1.0 * face_param3[i] / cell_volume[c_right]

    # ---- halo faces (left contributions only) ----
    for idx in range(start, d_halofaces.shape[0], stride):
      i = d_halofaces[idx]
      c_left = face_cellid[i][0]; c_leftglob = cell_loctoglob[c_left]
      parameters[0] = face_param4[i]; parameters[1] = face_param2[i]
      c_rightglob = halo_halosext[face_haloid[i]][0]

      p = cuda.atomic.add(cnt, 0, 1)
      irn_loc[p] = c_leftglob; jcn_loc[p] = c_leftglob
      a_loc[p] = face_param1[i] / cell_volume[c_left]
      p = cuda.atomic.add(cnt, 0, 1)
      irn_loc[p] = c_leftglob; jcn_loc[p] = c_rightglob
      a_loc[p] = face_param3[i] / cell_volume[c_left]

      cmptparam = 0
      for kn in range(faces[i][-1]):
        nod = faces[i][kn]
        if search(BCdirichlet, node_oldname[nod]) == 0:
          for j in range(node_cellid[nod][-1]):
            center[0] = cell_center[node_cellid[nod][j]][0]; center[1] = cell_center[node_cellid[nod][j]][1]
            xdiff = center[0] - nodes[nod][0]; ydiff = center[1] - nodes[nod][1]
            alpha = (1.0 + node_lambda_x[nod] * xdiff + node_lambda_y[nod] * ydiff) / (
                node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
            p = cuda.atomic.add(cnt, 0, 1)
            irn_loc[p] = c_leftglob; jcn_loc[p] = cell_loctoglob[node_cellid[nod][j]]
            a_loc[p] = alpha / cell_volume[c_left] * parameters[cmptparam]
          for j in range(node_ghostid[nod, -1]):
            gid = node_ghostid[nod][j]
            center[0] = ghost_info_flt[gid][0]; center[1] = ghost_info_flt[gid][1]
            xdiff = center[0] - nodes[nod][0]; ydiff = center[1] - nodes[nod][1]
            alpha = (1.0 + node_lambda_x[nod] * xdiff + node_lambda_y[nod] * ydiff) / (
                node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
            p = cuda.atomic.add(cnt, 0, 1)
            irn_loc[p] = c_leftglob; jcn_loc[p] = cell_loctoglob[ghost_info_int[gid, 0]]
            a_loc[p] = alpha / cell_volume[c_left] * parameters[cmptparam]
          for j in range(node_haloghostid[nod, -1]):
            gid = node_haloghostid[nod][j]
            center[0] = ghost_ext_info_flt[gid][0]; center[1] = ghost_ext_info_flt[gid][1]
            xdiff = center[0] - nodes[nod][0]; ydiff = center[1] - nodes[nod][1]
            alpha = (1.0 + node_lambda_x[nod] * xdiff + node_lambda_y[nod] * ydiff) / (
                node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
            p = cuda.atomic.add(cnt, 0, 1)
            irn_loc[p] = c_leftglob; jcn_loc[p] = halo_halosext[ghost_ext_info_int[gid, 0]][0]
            a_loc[p] = alpha / cell_volume[c_left] * parameters[cmptparam]
          for j in range(node_haloid[nod][-1]):
            center[0] = halo_centvol[node_haloid[nod][j]][0]; center[1] = halo_centvol[node_haloid[nod][j]][1]
            xdiff = center[0] - nodes[nod][0]; ydiff = center[1] - nodes[nod][1]
            alpha = (1.0 + node_lambda_x[nod] * xdiff + node_lambda_y[nod] * ydiff) / (
                node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
            p = cuda.atomic.add(cnt, 0, 1)
            irn_loc[p] = c_leftglob; jcn_loc[p] = halo_halosext[node_haloid[nod][j]][0]
            a_loc[p] = alpha / cell_volume[c_left] * parameters[cmptparam]
        cmptparam += 1

    # ---- dirichlet faces (diagonal only) ----
    for idx in range(start, dirichletfaces.shape[0], stride):
      i = dirichletfaces[idx]
      c_left = face_cellid[i][0]; c_leftglob = cell_loctoglob[c_left]
      p = cuda.atomic.add(cnt, 0, 1)
      irn_loc[p] = c_leftglob; jcn_loc[p] = c_leftglob
      a_loc[p] = face_param1[i] / cell_volume[c_left]
      p = cuda.atomic.add(cnt, 0, 1)
      irn_loc[p] = c_leftglob; jcn_loc[p] = c_leftglob
      a_loc[p] = -1.0 * face_param3[i] / cell_volume[c_left]

  kernel = gpu.compile_kernel(kernel)
  d_cnt = cuda.device_array(shape=(1,), dtype=gpu.int_precision)

  argcache = {}

  def result(*args):
    args = GPUArray.to_device_list(argcache, args)
    gpu.assign(d_cnt, 0)
    size = max(len(args[37]), len(args[38]), len(args[39]))
    grid, block = gpu.get_gpu_params(size)
    kernel[grid, block, gpu.stream](*args, d_cnt)
    if gpu.stream is not None:
      gpu.stream.synchronize()

  return result


# ---------------------------------------------------------------------------
def get_kernel_compute_2dmatrix_size():
  """Count COO triplets (sizes _row/_col/_data). Mirrors the CPU counter
  EXACTLY, including its halo-face over-count of 1 (harmless over-allocation)."""
  gpu = get_active_backend()
  search = gpu.compile_kernel(device_search_element, device=True)

  def kernel(faces: 'int[:,:]', node_cellid: 'int[:,:]', node_haloid: 'int[:,:]',
             node_periodicid: 'int[:,:]', node_ghostid: 'int[:,:]',
             node_haloghostid: 'int[:,:]', node_oldname: 'int[:]', BCdirichlet: 'int[:]',
             matrixinnerfaces: 'int[:]', d_halofaces: 'int[:]', dirichletfaces: 'int[:]',
             cnt: 'int[:]'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for idx in range(start, matrixinnerfaces.shape[0], stride):
      i = matrixinnerfaces[idx]
      c = 1
      for kn in range(faces[i][-1]):
        nod = faces[i][kn]
        if search(BCdirichlet, node_oldname[nod]) == 0:
          c += 2 * (node_cellid[nod][-1] + node_ghostid[nod, -1] +
                    node_haloghostid[nod, -1] + node_periodicid[nod][-1] +
                    node_haloid[nod][-1])
      c += 3
      cuda.atomic.add(cnt, 0, c)

    for idx in range(start, d_halofaces.shape[0], stride):
      i = d_halofaces[idx]
      c = 3
      for kn in range(faces[i][-1]):
        nod = faces[i][kn]
        if search(BCdirichlet, node_oldname[nod]) == 0:
          c += (node_cellid[nod][-1] + node_ghostid[nod, -1] +
                node_haloghostid[nod, -1] + node_haloid[nod][-1])
      cuda.atomic.add(cnt, 0, c)

    for idx in range(start, dirichletfaces.shape[0], stride):
      cuda.atomic.add(cnt, 0, 2)

  kernel = gpu.compile_kernel(kernel)
  d_cnt = cuda.device_array(shape=(1,), dtype=gpu.int_precision)

  argcache = {}

  def result(*args):
    args = GPUArray.to_device_list(argcache, args)
    gpu.assign(d_cnt, 0)
    size = max(len(args[8]), len(args[9]), len(args[10]))
    grid, block = gpu.get_gpu_params(size)
    kernel[grid, block, gpu.stream](*args, d_cnt)
    if gpu.stream is not None:
      gpu.stream.synchronize()
    return int(d_cnt.copy_to_host(stream=gpu.stream)[0])

  return result


# ---------------------------------------------------------------------------
def get_kernel_get_rhs_glob_2d():
  gpu = get_active_backend()
  search = gpu.compile_kernel(device_search_element, device=True)

  def kernel(face_cellid: 'int[:,:]', faces: 'int[:,:]', node_oldname: 'int[:]',
             cell_volume: 'float[:]', node_ghostid: 'int[:,:]', cell_loctoglob: 'int[:]',
             face_param1: 'float[:]', face_param2: 'float[:]', face_param3: 'float[:]',
             face_param4: 'float[:]', Pbordnode: 'float[:]', Pbordface: 'float[:]',
             rhs: 'float[:]', BCdirichlet: 'int[:]', matrixinnerfaces: 'int[:]',
             d_halofaces: 'int[:]', dirichletfaces: 'int[:]'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for idx in range(start, matrixinnerfaces.shape[0], stride):
      i = matrixinnerfaces[idx]
      c_left = face_cellid[i][0]; c_leftglob = cell_loctoglob[c_left]
      c_right = face_cellid[i][1]; c_rightglob = cell_loctoglob[c_right]
      i_1 = faces[i][0]; i_2 = faces[i][1]
      if search(BCdirichlet, node_oldname[i_1]) == 1:
        V = Pbordnode[i_1]
        cuda.atomic.add(rhs, c_leftglob, -1.0 * V * face_param4[i] / cell_volume[c_left])
        cuda.atomic.add(rhs, c_rightglob, V * face_param4[i] / cell_volume[c_right])
      if search(BCdirichlet, node_oldname[i_2]) == 1:
        V = Pbordnode[i_2]
        cuda.atomic.add(rhs, c_leftglob, -1.0 * V * face_param2[i] / cell_volume[c_left])
        cuda.atomic.add(rhs, c_rightglob, V * face_param2[i] / cell_volume[c_right])

    for idx in range(start, d_halofaces.shape[0], stride):
      i = d_halofaces[idx]
      c_left = face_cellid[i][0]; c_leftglob = cell_loctoglob[c_left]
      i_1 = faces[i][0]; i_2 = faces[i][1]
      if search(BCdirichlet, node_oldname[i_1]) == 1:
        cuda.atomic.add(rhs, c_leftglob, -1.0 * Pbordnode[i_1] * face_param4[i] / cell_volume[c_left])
      if search(BCdirichlet, node_oldname[i_2]) == 1:
        cuda.atomic.add(rhs, c_leftglob, -1.0 * Pbordnode[i_2] * face_param2[i] / cell_volume[c_left])

    for idx in range(start, dirichletfaces.shape[0], stride):
      i = dirichletfaces[idx]
      c_left = face_cellid[i][0]; c_leftglob = cell_loctoglob[c_left]
      i_1 = faces[i][0]; i_2 = faces[i][1]
      if node_ghostid[i_1, -1] > 0:
        cuda.atomic.add(rhs, c_leftglob, -1.0 * Pbordnode[i_1] * face_param4[i] / cell_volume[c_left])
      if node_ghostid[i_2, -1] > 0:
        cuda.atomic.add(rhs, c_leftglob, -1.0 * Pbordnode[i_2] * face_param2[i] / cell_volume[c_left])
      cuda.atomic.add(rhs, c_leftglob, -2.0 * face_param3[i] / cell_volume[c_left] * Pbordface[i])

  kernel = gpu.compile_kernel(kernel)

  argcache = {}

  def result(*args):
    args = GPUArray.to_device_list(argcache, args)
    gpu.assign(args[12], 0.0)  # rhs
    size = max(len(args[14]), len(args[15]), len(args[16]))
    grid, block = gpu.get_gpu_params(size)
    kernel[grid, block, gpu.stream](*args)
    if gpu.stream is not None:
      gpu.stream.synchronize()

  return result
