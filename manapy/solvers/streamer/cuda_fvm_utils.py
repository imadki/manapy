# -*- coding: utf-8 -*-
"""
GPU kernels of the streamer (plasma discharge) solver.

Signatures match the CPU kernels in streamer/fvm_utils_compute.py exactly, so each
get_kernel_*() wrapper is a drop-in for the StreamerSolver attributes. Same GPU
pattern as advecdiff/cuda_fvm_utils.py:
  - thread indexing via cuda.grid(1)/gridsize(1) instead of python loops;
  - concurrent face contributions to two neighbour cells via cuda.atomic.add;
  - device-argument handles memoised per call (GPUArray.to_device_list);
  - no per-launch synchronize (kernels chain on the same stream); only time_step
    drains the stream to read dt on the host.

The convective term reuses advecdiff's GPU convective kernel (the streamer drifts
the electron density ne exactly like an advection-diffusion scalar).
"""
import math

from numba import cuda

from manapy.backends.gpu import get_active_backend, GPUArray


# ---------------------------------------------------------------------------
def get_kernel_explicitscheme_dissipative_ST():
  gpu = get_active_backend()

  def kernel(u_face: 'float[:]', v_face: 'float[:]', w_face: 'float[:]',
             Ex_face: 'float[:]', Ey_face: 'float[:]', Ez_face: 'float[:]',
             nex_face: 'float[:]', ney_face: 'float[:]', nez_face: 'float[:]',
             face_cellid: 'int[:,:]', face_normal: 'float[:,:]', face_name: 'int[:]',
             dissip_ne: 'float[:]'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = 2.5e19
    for i in range(start, face_cellid.shape[0], stride):
      nx = face_normal[i][0]; ny = face_normal[i][1]; nz = face_normal[i][2]
      q = nex_face[i] * nx + ney_face[i] * ny + nez_face[i] * nz
      E = math.sqrt(Ex_face[i] ** 2 + Ey_face[i] ** 2 + Ez_face[i] ** 2)
      ve = math.sqrt(u_face[i] ** 2 + v_face[i] ** 2 + w_face[i] ** 2)
      if E == 0.:
        De = 0.
      else:
        De = (0.3341e9 * (E / n) ** 0.54069) * (ve / E)
      flux_ne = De * q
      if face_name[i] == 0:
        cuda.atomic.add(dissip_ne, face_cellid[i][0], flux_ne)
        cuda.atomic.add(dissip_ne, face_cellid[i][1], -flux_ne)
      else:
        cuda.atomic.add(dissip_ne, face_cellid[i][0], flux_ne)

  kernel = gpu.compile_kernel(kernel)
  argcache = {}

  def result(*args):
    args = GPUArray.to_device_list(argcache, args)
    gpu.assign(args[12], 0.0)  # dissip_ne
    grid, block = gpu.get_gpu_params(len(args[9]))  # face_cellid
    kernel[grid, block, gpu.stream](*args)

  return result


# ---------------------------------------------------------------------------
def get_kernel_explicitscheme_source_ST():
  gpu = get_active_backend()

  def kernel(ne: 'float[:]', u: 'float[:]', v: 'float[:]', w: 'float[:]',
             Ex: 'float[:]', Ey: 'float[:]', Ez: 'float[:]', src_ne: 'float[:]',
             src_ni: 'float[:]', center: 'float[:,:]', br: 'int'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = 2.5e19
    for i in range(start, ne.shape[0], stride):
      xcent = center[i][0]; ycent = center[i][1]; zcent = center[i][2]
      E = math.sqrt(Ex[i] ** 2 + Ey[i] ** 2 + Ez[i] ** 2)
      ve = math.sqrt(u[i] ** 2 + v[i] ** 2 + w[i] ** 2)
      if E == 0.:
        alpha_n = 0.
      elif (E / n) > 1.5e-15:
        alpha_n = 2e-16 * math.exp(-7.248e-15 / (E / n))
      else:
        alpha_n = 6.619e-17 * math.exp(-5.593e-15 / (E / n))
      S = alpha_n * ve * ne[i] * n
      if br == 1:
        S += 1e25 * math.exp(-1. * ((xcent - 0.3) ** 2. + (ycent - 0.25) ** 2. + (zcent - 0.28) ** 2.) / (0.005 ** 2.))
      if br == 2:
        S += 1e25 * math.exp(-1. * ((xcent - 0.31) ** 2. + (ycent - 0.25) ** 2. + (zcent - 0.22) ** 2.) / (0.005 ** 2.))
      src_ne[i] = S
      src_ni[i] = S

  kernel = gpu.compile_kernel(kernel)
  argcache = {}

  def result(*args):
    args = GPUArray.to_device_list(argcache, args)
    grid, block = gpu.get_gpu_params(len(args[0]))  # ne
    kernel[grid, block, gpu.stream](*args)

  return result


# ---------------------------------------------------------------------------
def get_kernel_compute_el_field():
  gpu = get_active_backend()

  def kernel(Pgradfacex: 'float[:]', Pgradfacey: 'float[:]', Pgradfacez: 'float[:]',
             Ex_face: 'float[:]', Ey_face: 'float[:]', Ez_face: 'float[:]'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, Ex_face.shape[0], stride):
      Ex_face[i] = Pgradfacex[i]
      Ey_face[i] = Pgradfacey[i]
      Ez_face[i] = Pgradfacez[i]

  kernel = gpu.compile_kernel(kernel)
  argcache = {}

  def result(*args):
    args = GPUArray.to_device_list(argcache, args)
    grid, block = gpu.get_gpu_params(len(args[3]))  # Ex_face
    kernel[grid, block, gpu.stream](*args)

  return result


# ---------------------------------------------------------------------------
def get_kernel_compute_velocity():
  """Two phases: (1) per-face drift velocity from the mobility tables, then
  (2) per-cell face->cell average of velocity and field. Phase 2 reads the
  face values written by phase 1; same stream => correctly ordered."""
  gpu = get_active_backend()

  def kernel_faces(Ex_face: 'float[:]', Ey_face: 'float[:]', Ez_face: 'float[:]',
                   u_face: 'float[:]', v_face: 'float[:]', w_face: 'float[:]'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = 2.5e19
    for i in range(start, u_face.shape[0], stride):
      E = math.sqrt(Ex_face[i] ** 2 + Ey_face[i] ** 2 + Ez_face[i] ** 2)
      if E == 0.:
        u_face[i] = 0.; v_face[i] = 0.; w_face[i] = 0.
      elif (E / n) > 2e-15:
        c = -1. / E * (7.4e21 * E / n + 7.1e6)
        u_face[i] = c * Ex_face[i]; v_face[i] = c * Ey_face[i]; w_face[i] = c * Ez_face[i]
      elif 1e-16 < (E / n) <= 2e-15:
        c = -1. / E * (1.03e22 * E / n + 1.3e6)
        u_face[i] = c * Ex_face[i]; v_face[i] = c * Ey_face[i]; w_face[i] = c * Ez_face[i]
      elif 2.6e-17 < (E / n) <= 1e-16:
        c = -1. / E * (7.2973e21 * E / n + 1.63e6)
        u_face[i] = c * Ex_face[i]; v_face[i] = c * Ey_face[i]; w_face[i] = c * Ez_face[i]
      else:
        c = -1. / E * (6.87e22 * E / n + 3.38e4)
        u_face[i] = c * Ex_face[i]; v_face[i] = c * Ey_face[i]; w_face[i] = c * Ez_face[i]

  def kernel_cells(Ex_face: 'float[:]', Ey_face: 'float[:]', Ez_face: 'float[:]',
                   u_face: 'float[:]', v_face: 'float[:]', w_face: 'float[:]',
                   Ex: 'float[:]', Ey: 'float[:]', Ez: 'float[:]',
                   u: 'float[:]', v: 'float[:]', w: 'float[:]',
                   cell_faceid: 'int[:,:]', dim: 'int'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, cell_faceid.shape[0], stride):
      su = 0.; sv = 0.; sw = 0.; sEx = 0.; sEy = 0.; sEz = 0.
      for j in range(dim + 1):
        face = cell_faceid[i][j]
        su += u_face[face]; sv += v_face[face]; sw += w_face[face]
        sEx += Ex_face[face]; sEy += Ey_face[face]; sEz += Ez_face[face]
      d = dim + 1
      u[i] = su / d; v[i] = sv / d; w[i] = sw / d
      Ex[i] = sEx / d; Ey[i] = sEy / d; Ez[i] = sEz / d

  kf = gpu.compile_kernel(kernel_faces)
  kc = gpu.compile_kernel(kernel_cells)
  argcache = {}

  def result(*args):
    # args: Ex_face,Ey_face,Ez_face,u_face,v_face,w_face,Ex,Ey,Ez,u,v,w,cell_faceid,dim
    args = GPUArray.to_device_list(argcache, args)
    gf, bf = gpu.get_gpu_params(len(args[3]))   # u_face
    kf[gf, bf, gpu.stream](*args[:6])
    gc, bc = gpu.get_gpu_params(len(args[12]))  # cell_faceid
    kc[gc, bc, gpu.stream](*args)

  return result


# ---------------------------------------------------------------------------
def get_kernel_time_step_ST():
  gpu = get_active_backend()
  d_shared_dt = cuda.device_array(shape=(1,), dtype=gpu.float_precision)

  def kernel(u: 'float[:]', v: 'float[:]', w: 'float[:]', Ex: 'float[:]', Ey: 'float[:]',
             Ez: 'float[:]', cfl: 'float', face_normal: 'float[:,:]',
             face_measure: 'float[:]', cell_volume: 'float[:]', cell_faceid: 'int[:,:]',
             dim: 'int', shared_dt: 'float[:]'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = 2.5e19
    for i in range(start, cell_faceid.shape[0], stride):
      ve = math.sqrt(u[i] ** 2 + v[i] ** 2 + w[i] ** 2)
      E = math.sqrt(Ex[i] ** 2 + Ey[i] ** 2 + Ez[i] ** 2)
      if E == 0.:
        De = 0.
      else:
        De = 0.3341e9 * (E / n) ** 0.54069 * (ve / E)
      lam = 0.
      for j in range(dim + 1):
        face = cell_faceid[i][j]
        nx = face_normal[face][0]; ny = face_normal[face][1]; nz = face_normal[face][2]
        lam_convect = math.fabs(u[i] * nx + v[i] * ny + w[i] * nz) / face_measure[face]
        lam += lam_convect * face_measure[face]
        mes = math.sqrt(nx * nx + ny * ny + nz * nz)
        lam += De * mes ** 2 / cell_volume[i]
      if lam != 0.:
        cuda.atomic.min(shared_dt, 0, cfl * cell_volume[i] / lam)

  kernel = gpu.compile_kernel(kernel)
  argcache = {}

  def result(*args):
    args = GPUArray.to_device_list(argcache, args)
    gpu.assign(d_shared_dt, 1e6)
    grid, block = gpu.get_gpu_params(len(args[10]))  # cell_faceid
    kernel[grid, block, gpu.stream](*args, d_shared_dt)
    # host read of dt: the time-step synchronization point (drains the stream).
    host = d_shared_dt.copy_to_host(stream=gpu.stream)
    gpu.synchronize()
    return host[0]

  return result


# ---------------------------------------------------------------------------
def get_kernel_update_ST():
  gpu = get_active_backend()

  def kernel(ne_c: 'float[:]', ni_c: 'float[:]', rez_ne: 'float[:]', rez_ni: 'float[:]',
             dissip_ne: 'float[:]', dissip_ni: 'float[:]', src_ne: 'float[:]',
             src_ni: 'float[:]', dtime: 'float', cell_volume: 'float[:]'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, ne_c.shape[0], stride):
      ne_c[i] += dtime * ((rez_ne[i] + dissip_ne[i]) / cell_volume[i] + src_ne[i])
      ni_c[i] += dtime * ((rez_ni[i] + dissip_ni[i]) / cell_volume[i] + src_ni[i])

  kernel = gpu.compile_kernel(kernel)
  argcache = {}

  def result(*args):
    args = GPUArray.to_device_list(argcache, args)
    grid, block = gpu.get_gpu_params(len(args[0]))  # ne_c
    kernel[grid, block, gpu.stream](*args)

  return result


# ---------------------------------------------------------------------------
def get_kernel_update_rhs_loc():
  gpu = get_active_backend()

  def kernel(ne: 'float[:]', ni: 'float[:]', loctoglob: 'int[:]', rhs_updated: 'float[:]'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, ne.shape[0], stride):
      rhs_updated[i] = 1.8096e-8 * (ne[i] - ni[i])

  kernel = gpu.compile_kernel(kernel)
  argcache = {}

  def result(*args):
    args = GPUArray.to_device_list(argcache, args)
    grid, block = gpu.get_gpu_params(len(args[0]))  # ne
    kernel[grid, block, gpu.stream](*args)

  return result
