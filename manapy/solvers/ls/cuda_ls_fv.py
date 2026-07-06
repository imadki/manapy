# -*- coding: utf-8 -*-
"""
GPU (numba.cuda) port of manapy/solvers/ls/ls_fv.py (cell-centred finite-volume
Gauss-linear Laplacian, with optional non-orthogonal correction).

Same convention as cuda_ls_compute.py: each get_kernel_*() returns a wrapper
whose signature is IDENTICAL to the matching CPU kernel in ls_fv.py, so it is a
drop-in for LinearSolver's _get_triplet / _get_rhs / _get_rhs_correction /
_compute_P_gradient_fv.

Parallelisation:
  - the triplet (matrix) assembly writes to DETERMINISTIC slots: face k of a
    section writes at a fixed base offset (4 entries per inner face, 2 per halo
    face, 1 per dirichlet face). No atomic counter is needed, unlike diamond;
  - the face-gradient kernel writes its own face slot -> no atomics;
  - the RHS kernels accumulate per cell -> cuda.atomic.add.

Kernels are dimension-generic (they use the full face_normal[:, 0:3]), so they
work for both 2D and 3D. Imported lazily by LinearSolver only when
domain.backend.name == "gpu" and the scheme is fv / fv_corrected.
"""
import math

from numba import cuda

from manapy.backends.gpu import get_active_backend, GPUArray


# ---------------------------------------------------------------------------
# Device helper: corrected face gradient (mirrors ls_fv._set_fv_gradient).
# Stores -grad(P). The CPU version raises on a degenerate face; a CUDA device
# function cannot raise, so we guard the denominators with a tiny epsilon
# instead (a zero projected distance only happens on a broken mesh).
def _device_set_fv_gradient(P_left: 'float', P_right: 'float',
                            gfx: 'float', gfy: 'float', gfz: 'float',
                            nx: 'float', ny: 'float', nz: 'float',
                            dx: 'float', dy: 'float', dz: 'float',
                            i: 'int', Px_face: 'float[:]', Py_face: 'float[:]',
                            Pz_face: 'float[:]'):
  mag = math.sqrt(nx * nx + ny * ny + nz * nz)
  denom = nx * dx + ny * dy + nz * dz
  if denom < 0.0:
    denom = -denom
  if denom < 1e-300:
    denom = 1e-300
  if mag < 1e-300:
    mag = 1e-300
  sn = (P_right - P_left) * mag / denom            # grad(P).n_hat (two-point)
  gdotn = (gfx * nx + gfy * ny + gfz * nz) / mag    # interpolated grad . n_hat
  corr = sn - gdotn
  Px_face[i] = -(gfx + corr * nx / mag)
  Py_face[i] = -(gfy + corr * ny / mag)
  Pz_face[i] = -(gfz + corr * nz / mag)


# ---------------------------------------------------------------------------
def get_kernel_get_triplet_fv():
  """Assemble the COO triplets of the FV Laplacian. Deterministic offsets:
  section layout is [inner: 4/face][halo: 2/face][dirichlet: 1/face], exactly
  matching ls_fv._compute_fv_matrix_size, so every output slot is written."""
  gpu = get_active_backend()

  def kernel(face_cellid: 'int[:,:]', face_fv_coeff: 'float[:]',
             halo_halosext: 'int[:,:]', cell_volume: 'float[:]',
             cell_loctoglob: 'int[:]', face_haloid: 'int[:]', a_loc: 'float[:]',
             irn_loc: 'int[:]', jcn_loc: 'int[:]', matrixinnerfaces: 'int[:]',
             d_halofaces: 'int[:]', dirichletfaces: 'int[:]'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n_inner = matrixinnerfaces.shape[0]
    n_halo = d_halofaces.shape[0]
    base_halo = 4 * n_inner
    base_dir = 4 * n_inner + 2 * n_halo

    for idx in range(start, n_inner, stride):
      i = matrixinnerfaces[idx]
      c_left = face_cellid[i, 0]
      c_right = face_cellid[i, 1]
      c_leftglob = cell_loctoglob[c_left]
      c_rightglob = cell_loctoglob[c_right]
      coeff = face_fv_coeff[i]
      p = 4 * idx

      value = coeff / cell_volume[c_left]
      irn_loc[p] = c_leftglob
      jcn_loc[p] = c_leftglob
      a_loc[p] = -value
      irn_loc[p + 1] = c_leftglob
      jcn_loc[p + 1] = c_rightglob
      a_loc[p + 1] = value

      value = coeff / cell_volume[c_right]
      irn_loc[p + 2] = c_rightglob
      jcn_loc[p + 2] = c_leftglob
      a_loc[p + 2] = value
      irn_loc[p + 3] = c_rightglob
      jcn_loc[p + 3] = c_rightglob
      a_loc[p + 3] = -value

    for idx in range(start, n_halo, stride):
      i = d_halofaces[idx]
      c_left = face_cellid[i, 0]
      c_leftglob = cell_loctoglob[c_left]
      c_rightglob = halo_halosext[face_haloid[i], 0]
      coeff = face_fv_coeff[i]
      value = coeff / cell_volume[c_left]
      p = base_halo + 2 * idx

      irn_loc[p] = c_leftglob
      jcn_loc[p] = c_leftglob
      a_loc[p] = -value
      irn_loc[p + 1] = c_leftglob
      jcn_loc[p + 1] = c_rightglob
      a_loc[p + 1] = value

    for idx in range(start, dirichletfaces.shape[0], stride):
      i = dirichletfaces[idx]
      c_left = face_cellid[i, 0]
      c_leftglob = cell_loctoglob[c_left]
      coeff = face_fv_coeff[i]
      value = coeff / cell_volume[c_left]
      p = base_dir + idx

      irn_loc[p] = c_leftglob
      jcn_loc[p] = c_leftglob
      a_loc[p] = -value

  kernel = gpu.compile_kernel(kernel)
  argcache = {}

  def result(*args):
    args = GPUArray.to_device_list(argcache, args)
    # Every slot is written, so no zero-init is needed.
    size = max(len(args[9]), len(args[10]), len(args[11]), 1)
    grid, block = gpu.get_gpu_params(size)
    kernel[grid, block, gpu.stream](*args)
    if gpu.stream is not None:
      gpu.stream.synchronize()

  return result


# ---------------------------------------------------------------------------
def _get_kernel_get_rhs_fv(use_global):
  """Shared factory for the FV Dirichlet RHS. use_global selects the global row
  index (cell_loctoglob) for centralized solvers, or the local cell index."""
  gpu = get_active_backend()

  def kernel(face_cellid: 'int[:,:]', face_fv_coeff: 'float[:]',
             cell_volume: 'float[:]', cell_loctoglob: 'int[:]',
             Pbordface: 'float[:]', rhs: 'float[:]', dirichletfaces: 'int[:]'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for idx in range(start, dirichletfaces.shape[0], stride):
      i = dirichletfaces[idx]
      c_left = face_cellid[i, 0]
      if use_global:
        row = cell_loctoglob[c_left]
      else:
        row = c_left
      coeff = face_fv_coeff[i]
      cuda.atomic.add(rhs, row, -coeff * Pbordface[i] / cell_volume[c_left])

  kernel = gpu.compile_kernel(kernel)
  argcache = {}

  def result(*args):
    args = GPUArray.to_device_list(argcache, args)
    rhs = args[5]
    gpu.assign(rhs, 0.0)
    size = max(len(args[6]), 1)
    grid, block = gpu.get_gpu_params(size)
    kernel[grid, block, gpu.stream](*args)
    if gpu.stream is not None:
      gpu.stream.synchronize()

  return result


def get_kernel_get_rhs_fv_glob():
  return _get_kernel_get_rhs_fv(True)


def get_kernel_get_rhs_fv_loc():
  return _get_kernel_get_rhs_fv(False)


# ---------------------------------------------------------------------------
def _get_kernel_get_rhs_fv_correction(use_global):
  """Shared factory for the explicit non-orthogonal correction source. Per-cell
  accumulation -> cuda.atomic.add. use_global picks the global/local row."""
  gpu = get_active_backend()

  def kernel(face_cellid: 'int[:,:]', face_haloid: 'int[:]',
             cell_volume: 'float[:]', cell_loctoglob: 'int[:]',
             face_fv_corrx: 'float[:]', face_fv_corry: 'float[:]',
             face_fv_corrz: 'float[:]', face_fv_weight_left: 'float[:]',
             gradcellx: 'float[:]', gradcelly: 'float[:]', gradcellz: 'float[:]',
             gradhalocellx: 'float[:]', gradhalocelly: 'float[:]',
             gradhalocellz: 'float[:]', rhs: 'float[:]',
             matrixinnerfaces: 'int[:]', d_halofaces: 'int[:]',
             dirichletfaces: 'int[:]', d_periodicboundaryfaces: 'int[:]'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for idx in range(start, matrixinnerfaces.shape[0], stride):
      i = matrixinnerfaces[idx]
      c_left = face_cellid[i, 0]
      c_right = face_cellid[i, 1]
      wl = face_fv_weight_left[i]
      wr = 1.0 - wl
      gx = wl * gradcellx[c_left] + wr * gradcellx[c_right]
      gy = wl * gradcelly[c_left] + wr * gradcelly[c_right]
      gz = wl * gradcellz[c_left] + wr * gradcellz[c_right]
      corr = face_fv_corrx[i] * gx + face_fv_corry[i] * gy + face_fv_corrz[i] * gz
      if use_global:
        rl = cell_loctoglob[c_left]
        rr = cell_loctoglob[c_right]
      else:
        rl = c_left
        rr = c_right
      cuda.atomic.add(rhs, rl, -corr / cell_volume[c_left])
      cuda.atomic.add(rhs, rr, corr / cell_volume[c_right])

    for idx in range(start, d_periodicboundaryfaces.shape[0], stride):
      i = d_periodicboundaryfaces[idx]
      c_left = face_cellid[i, 0]
      c_right = face_cellid[i, 1]
      wl = face_fv_weight_left[i]
      wr = 1.0 - wl
      gx = wl * gradcellx[c_left] + wr * gradcellx[c_right]
      gy = wl * gradcelly[c_left] + wr * gradcelly[c_right]
      gz = wl * gradcellz[c_left] + wr * gradcellz[c_right]
      corr = face_fv_corrx[i] * gx + face_fv_corry[i] * gy + face_fv_corrz[i] * gz
      if use_global:
        rl = cell_loctoglob[c_left]
        rr = cell_loctoglob[c_right]
      else:
        rl = c_left
        rr = c_right
      cuda.atomic.add(rhs, rl, -corr / cell_volume[c_left])
      cuda.atomic.add(rhs, rr, corr / cell_volume[c_right])

    for idx in range(start, d_halofaces.shape[0], stride):
      i = d_halofaces[idx]
      c_left = face_cellid[i, 0]
      c_right = face_haloid[i]
      wl = face_fv_weight_left[i]
      wr = 1.0 - wl
      gx = wl * gradcellx[c_left] + wr * gradhalocellx[c_right]
      gy = wl * gradcelly[c_left] + wr * gradhalocelly[c_right]
      gz = wl * gradcellz[c_left] + wr * gradhalocellz[c_right]
      corr = face_fv_corrx[i] * gx + face_fv_corry[i] * gy + face_fv_corrz[i] * gz
      if use_global:
        rl = cell_loctoglob[c_left]
      else:
        rl = c_left
      cuda.atomic.add(rhs, rl, -corr / cell_volume[c_left])

    for idx in range(start, dirichletfaces.shape[0], stride):
      i = dirichletfaces[idx]
      c_left = face_cellid[i, 0]
      corr = (face_fv_corrx[i] * gradcellx[c_left]
              + face_fv_corry[i] * gradcelly[c_left]
              + face_fv_corrz[i] * gradcellz[c_left])
      if use_global:
        rl = cell_loctoglob[c_left]
      else:
        rl = c_left
      cuda.atomic.add(rhs, rl, -corr / cell_volume[c_left])

  kernel = gpu.compile_kernel(kernel)
  argcache = {}

  def result(*args):
    args = GPUArray.to_device_list(argcache, args)
    rhs = args[14]
    gpu.assign(rhs, 0.0)
    size = max(len(args[15]), len(args[16]), len(args[17]), len(args[18]), 1)
    grid, block = gpu.get_gpu_params(size)
    kernel[grid, block, gpu.stream](*args)
    if gpu.stream is not None:
      gpu.stream.synchronize()

  return result


def get_kernel_get_rhs_fv_correction_glob():
  return _get_kernel_get_rhs_fv_correction(True)


def get_kernel_get_rhs_fv_correction_loc():
  return _get_kernel_get_rhs_fv_correction(False)


# ---------------------------------------------------------------------------
def get_kernel_compute_P_gradient_fv():
  """Corrected face gradient (-grad P) for the non-orthogonal correction. Each
  thread writes its own face slot (Px/Py/Pz_face[i]) -> no atomics."""
  gpu = get_active_backend()
  set_grad = gpu.compile_kernel(_device_set_fv_gradient, device=True)

  def kernel(P_c: 'float[:]', P_halo: 'float[:]', face_cellid: 'int[:,:]',
             face_name: 'int[:]', face_normal: 'float[:,:]',
             face_center: 'float[:,:]', face_haloid: 'int[:]',
             cell_center: 'float[:,:]', halo_centvol: 'float[:,:]',
             cell_shift: 'float[:,:]', Pbordface: 'float[:]',
             gradcellx: 'float[:]', gradcelly: 'float[:]', gradcellz: 'float[:]',
             gradhalocellx: 'float[:]', gradhalocelly: 'float[:]',
             gradhalocellz: 'float[:]', weight_left: 'float[:]',
             Px_face: 'float[:]', Py_face: 'float[:]', Pz_face: 'float[:]',
             d_innerfaces: 'int[:]', d_halofaces: 'int[:]',
             neumannfaces: 'int[:]', dirichletfaces: 'int[:]',
             d_periodicboundaryfaces: 'int[:]'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for idx in range(start, d_innerfaces.shape[0], stride):
      i = d_innerfaces[idx]
      c_left = face_cellid[i, 0]
      c_right = face_cellid[i, 1]
      wl = weight_left[i]
      wr = 1.0 - wl
      gfx = wl * gradcellx[c_left] + wr * gradcellx[c_right]
      gfy = wl * gradcelly[c_left] + wr * gradcelly[c_right]
      gfz = wl * gradcellz[c_left] + wr * gradcellz[c_right]
      dx = cell_center[c_right, 0] - cell_center[c_left, 0]
      dy = cell_center[c_right, 1] - cell_center[c_left, 1]
      dz = cell_center[c_right, 2] - cell_center[c_left, 2]
      set_grad(P_c[c_left], P_c[c_right], gfx, gfy, gfz,
               face_normal[i, 0], face_normal[i, 1], face_normal[i, 2],
               dx, dy, dz, i, Px_face, Py_face, Pz_face)

    for idx in range(start, d_periodicboundaryfaces.shape[0], stride):
      i = d_periodicboundaryfaces[idx]
      c_left = face_cellid[i, 0]
      c_right = face_cellid[i, 1]
      wl = weight_left[i]
      wr = 1.0 - wl
      gfx = wl * gradcellx[c_left] + wr * gradcellx[c_right]
      gfy = wl * gradcelly[c_left] + wr * gradcelly[c_right]
      gfz = wl * gradcellz[c_left] + wr * gradcellz[c_right]
      dx = cell_center[c_right, 0] - cell_center[c_left, 0]
      dy = cell_center[c_right, 1] - cell_center[c_left, 1]
      dz = cell_center[c_right, 2] - cell_center[c_left, 2]
      if face_name[i] == 11 or face_name[i] == 22:
        dx += cell_shift[c_right, 0]
      elif face_name[i] == 33 or face_name[i] == 44:
        dy += cell_shift[c_right, 1]
      elif face_name[i] == 55 or face_name[i] == 66:
        dz += cell_shift[c_right, 2]
      set_grad(P_c[c_left], P_c[c_right], gfx, gfy, gfz,
               face_normal[i, 0], face_normal[i, 1], face_normal[i, 2],
               dx, dy, dz, i, Px_face, Py_face, Pz_face)

    for idx in range(start, d_halofaces.shape[0], stride):
      i = d_halofaces[idx]
      c_left = face_cellid[i, 0]
      c_right = face_haloid[i]
      wl = weight_left[i]
      wr = 1.0 - wl
      gfx = wl * gradcellx[c_left] + wr * gradhalocellx[c_right]
      gfy = wl * gradcelly[c_left] + wr * gradhalocelly[c_right]
      gfz = wl * gradcellz[c_left] + wr * gradhalocellz[c_right]
      dx = halo_centvol[c_right, 0] - cell_center[c_left, 0]
      dy = halo_centvol[c_right, 1] - cell_center[c_left, 1]
      dz = halo_centvol[c_right, 2] - cell_center[c_left, 2]
      set_grad(P_c[c_left], P_halo[c_right], gfx, gfy, gfz,
               face_normal[i, 0], face_normal[i, 1], face_normal[i, 2],
               dx, dy, dz, i, Px_face, Py_face, Pz_face)

    for idx in range(start, neumannfaces.shape[0], stride):
      i = neumannfaces[idx]
      c_left = face_cellid[i, 0]
      nx = face_normal[i, 0]
      ny = face_normal[i, 1]
      nz = face_normal[i, 2]
      mag = math.sqrt(nx * nx + ny * ny + nz * nz)
      if mag < 1e-300:
        mag = 1e-300
      gfx = gradcellx[c_left]
      gfy = gradcelly[c_left]
      gfz = gradcellz[c_left]
      gdotn = (gfx * nx + gfy * ny + gfz * nz) / mag
      Px_face[i] = -(gfx - gdotn * nx / mag)
      Py_face[i] = -(gfy - gdotn * ny / mag)
      Pz_face[i] = -(gfz - gdotn * nz / mag)

    for idx in range(start, dirichletfaces.shape[0], stride):
      i = dirichletfaces[idx]
      c_left = face_cellid[i, 0]
      gfx = gradcellx[c_left]
      gfy = gradcelly[c_left]
      gfz = gradcellz[c_left]
      dx = face_center[i, 0] - cell_center[c_left, 0]
      dy = face_center[i, 1] - cell_center[c_left, 1]
      dz = face_center[i, 2] - cell_center[c_left, 2]
      set_grad(P_c[c_left], Pbordface[i], gfx, gfy, gfz,
               face_normal[i, 0], face_normal[i, 1], face_normal[i, 2],
               dx, dy, dz, i, Px_face, Py_face, Pz_face)

  kernel = gpu.compile_kernel(kernel)
  argcache = {}

  def result(*args):
    args = GPUArray.to_device_list(argcache, args)
    # Per-face writes; faces not in any group keep their previous value, exactly
    # like the CPU kernel (no zero-init).
    size = max(len(args[21]), len(args[22]), len(args[23]),
               len(args[24]), len(args[25]), 1)
    grid, block = gpu.get_gpu_params(size)
    kernel[grid, block, gpu.stream](*args)
    if gpu.stream is not None:
      gpu.stream.synchronize()

  return result
