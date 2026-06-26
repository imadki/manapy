# -*- coding: utf-8 -*-
"""
Kernels GPU du solveur advection-diffusion (meme approche que advec/cuda_fvm_utils.py).

Les signatures sont IDENTIQUES aux kernels CPU (advecdiff/fvm_utils_compute.py),
de sorte que chaque wrapper renvoye par get_kernel_*() est un drop-in pour les
attributs de AdvectionDiffusionSolver. Differences avec le CPU :
  - indexation par thread (cuda.grid(1)) au lieu des boucles `for i in range(...)` ;
  - accumulations concurrentes via cuda.atomic.add (les faces internes ecrivent
    sur deux cellules voisines depuis des threads differents).

Par rapport a advec/cuda_fvm_utils.py, on ajoute le kernel dissipatif (diffusion)
et le terme diffusif dans le pas de temps.
"""
from numba import cuda

from manapy.backends.gpu import get_active_backend, GPUArray


def device_compute_upwind_flux(w_l: 'float', w_r: 'float', u_face: 'float', v_face: 'float',
                               w_face: 'float', normal: 'float[:]', flux_w: 'float[:]'):
  sign = u_face * normal[0] + v_face * normal[1] + w_face * normal[2]
  if sign >= 0:
    sol = w_l
  else:
    sol = w_r
  flux_w[0] = sign * sol


# ---------------------------------------------------------------------------
def get_kernel_explicitscheme_convective_2d():
  gpu = get_active_backend()
  compute_upwind_flux = gpu.compile_kernel(device_compute_upwind_flux, device=True)

  def kernel(rez_w: 'float[:]', w_c: 'float[:]', w_ghost: 'float[:]', w_halo: 'float[:]',
             u_face: 'float[:]', v_face: 'float[:]', w_face: 'float[:]',
             w_x: 'float[:]', w_y: 'float[:]', w_z: 'float[:]', wx_halo: 'float[:]',
             wy_halo: 'float[:]', wz_halo: 'float[:]', psi: 'float[:]', psi_halo: 'float[:]',
             cell_center: 'float[:,:]', face_center: 'float[:,:]', halo_centvol: 'float[:,:]',
             face_cellid: 'int[:,:]', face_normal: 'float[:,:]', face_haloid: 'int[:]',
             face_name: 'int[:]', d_innerfaces: 'int[:]', d_halofaces: 'int[:]',
             d_boundaryfaces: 'int[:]', d_periodicboundaryfaces: 'int[:]',
             cell_shift: 'float[:,:]', order: 'int'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    r_l = cuda.local.array(2, cell_center.dtype)
    r_r = cuda.local.array(2, cell_center.dtype)
    flux_w = cuda.local.array(1, cell_center.dtype)

    # faces internes : ecrit sur les deux cellules voisines
    for idx in range(start, d_innerfaces.shape[0], stride):
      i = d_innerfaces[idx]
      w_l = w_c[face_cellid[i][0]]
      normal = face_normal[i]
      w_r = w_c[face_cellid[i][1]]
      center_left = cell_center[face_cellid[i][0]]
      center_right = cell_center[face_cellid[i][1]]
      w_x_left = w_x[face_cellid[i][0]]; w_x_right = w_x[face_cellid[i][1]]
      w_y_left = w_y[face_cellid[i][0]]; w_y_right = w_y[face_cellid[i][1]]
      psi_left = psi[face_cellid[i][0]]; psi_right = psi[face_cellid[i][1]]
      r_l[0] = face_center[i][0] - center_left[0]; r_r[0] = face_center[i][0] - center_right[0]
      r_l[1] = face_center[i][1] - center_left[1]; r_r[1] = face_center[i][1] - center_right[1]
      w_l = w_l + (order - 1) * psi_left * (w_x_left * r_l[0] + w_y_left * r_l[1])
      w_r = w_r + (order - 1) * psi_right * (w_x_right * r_r[0] + w_y_right * r_r[1])
      compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
      cuda.atomic.add(rez_w, face_cellid[i][0], -flux_w[0])
      cuda.atomic.add(rez_w, face_cellid[i][1], flux_w[0])

    for idx in range(start, d_periodicboundaryfaces.shape[0], stride):
      i = d_periodicboundaryfaces[idx]
      w_l = w_c[face_cellid[i][0]]
      normal = face_normal[i]
      w_r = w_c[face_cellid[i][1]]
      center_left = cell_center[face_cellid[i][0]]
      center_right = cell_center[face_cellid[i][1]]
      w_x_left = w_x[face_cellid[i][0]]; w_x_right = w_x[face_cellid[i][1]]
      w_y_left = w_y[face_cellid[i][0]]; w_y_right = w_y[face_cellid[i][1]]
      psi_left = psi[face_cellid[i][0]]; psi_right = psi[face_cellid[i][1]]
      if face_name[i] == 11 or face_name[i] == 22:
        r_l[0] = face_center[i][0] - center_left[0]
        r_r[0] = face_center[i][0] - center_right[0] - cell_shift[face_cellid[i][1]][0]
        r_l[1] = face_center[i][1] - center_left[1]
        r_r[1] = face_center[i][1] - center_right[1]
      if face_name[i] == 33 or face_name[i] == 44:
        r_l[0] = face_center[i][0] - center_left[0]
        r_r[0] = face_center[i][0] - center_right[0]
        r_l[1] = face_center[i][1] - center_left[1]
        r_r[1] = face_center[i][1] - center_right[1] - cell_shift[face_cellid[i][1]][1]
      w_l = w_l + (order - 1) * psi_left * (w_x_left * r_l[0] + w_y_left * r_l[1])
      w_r = w_r + (order - 1) * psi_right * (w_x_right * r_r[0] + w_y_right * r_r[1])
      compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
      cuda.atomic.add(rez_w, face_cellid[i][0], -flux_w[0])

    for idx in range(start, d_halofaces.shape[0], stride):
      i = d_halofaces[idx]
      w_l = w_c[face_cellid[i][0]]
      normal = face_normal[i]
      w_r = w_halo[face_haloid[i]]
      center_left = cell_center[face_cellid[i][0]]
      center_right = halo_centvol[face_haloid[i]]
      w_x_left = w_x[face_cellid[i][0]]; w_x_right = wx_halo[face_haloid[i]]
      w_y_left = w_y[face_cellid[i][0]]; w_y_right = wy_halo[face_haloid[i]]
      psi_left = psi[face_cellid[i][0]]; psi_right = psi_halo[face_haloid[i]]
      r_l[0] = face_center[i][0] - center_left[0]; r_r[0] = face_center[i][0] - center_right[0]
      r_l[1] = face_center[i][1] - center_left[1]; r_r[1] = face_center[i][1] - center_right[1]
      w_l = w_l + (order - 1) * psi_left * (w_x_left * r_l[0] + w_y_left * r_l[1])
      w_r = w_r + (order - 1) * psi_right * (w_x_right * r_r[0] + w_y_right * r_r[1])
      compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
      cuda.atomic.add(rez_w, face_cellid[i][0], -flux_w[0])

    for idx in range(start, d_boundaryfaces.shape[0], stride):
      i = d_boundaryfaces[idx]
      w_l = w_c[face_cellid[i][0]]
      normal = face_normal[i]
      w_r = w_ghost[i]
      center_left = cell_center[face_cellid[i][0]]
      w_x_left = w_x[face_cellid[i][0]]
      w_y_left = w_y[face_cellid[i][0]]
      psi_left = psi[face_cellid[i][0]]
      r_l[0] = face_center[i][0] - center_left[0]
      r_l[1] = face_center[i][1] - center_left[1]
      w_l = w_l + (order - 1) * psi_left * (w_x_left * r_l[0] + w_y_left * r_l[1])
      compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
      cuda.atomic.add(rez_w, face_cellid[i][0], -flux_w[0])

  kernel = gpu.compile_kernel(kernel)
  argcache = {}
  def result(*args):
    args = GPUArray.to_device_list(argcache, args)
    rez_w = args[0]
    gpu.assign(rez_w, 0.0)
    # une grille dimensionnee sur le plus grand groupe de faces
    size = max(len(args[22]), len(args[23]), len(args[24]), len(args[25]))
    grid, block = gpu.get_gpu_params(size)
    kernel[grid, block, gpu.stream](*args)
    # pas de synchronize : enchainement sur le meme stream (ordonne) ; la synchro
    # n'a lieu qu'a la lecture host (dt dans time_step, to_host a la sauvegarde).

  return result


# ---------------------------------------------------------------------------
def get_kernel_explicitscheme_dissipative():
  gpu = get_active_backend()

  def kernel(wx_face: 'float[:]', wy_face: 'float[:]', wz_face: 'float[:]',
             face_cellid: 'int[:,:]', face_normal: 'float[:,:]', face_name: 'int[:]',
             dissip_w: 'float[:]', Dxx: 'float', Dyy: 'float', Dzz: 'float'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, face_cellid.shape[0], stride):
      norm = face_normal[i]
      q = Dxx * wx_face[i] * norm[0] + Dyy * wy_face[i] * norm[1] + Dzz * wz_face[i] * norm[2]
      flux_w = q
      if face_name[i] == 0:
        # face interne : ecrit sur les deux cellules voisines
        cuda.atomic.add(dissip_w, face_cellid[i][0], flux_w)
        cuda.atomic.add(dissip_w, face_cellid[i][1], -flux_w)
      else:
        cuda.atomic.add(dissip_w, face_cellid[i][0], flux_w)

  kernel = gpu.compile_kernel(kernel)
  argcache = {}
  def result(*args):
    args = GPUArray.to_device_list(argcache, args)
    dissip_w = args[6]
    gpu.assign(dissip_w, 0.0)
    grid, block = gpu.get_gpu_params(len(args[3]))  # face_cellid
    kernel[grid, block, gpu.stream](*args)
    # pas de synchronize : enchainement sur le meme stream (ordonne) ; la synchro
    # n'a lieu qu'a la lecture host (dt dans time_step, to_host a la sauvegarde).

  return result


# ---------------------------------------------------------------------------
def get_kernel_time_step():
  gpu = get_active_backend()
  d_shared_dt = cuda.device_array(shape=(1,), dtype=gpu.float_precision)

  def kernel(u: 'float[:]', v: 'float[:]', w: 'float[:]', cfl: 'float',
             face_normal: 'float[:,:]', face_measure: 'float[:]', cell_volume: 'float[:]',
             cell_faceid: 'int[:,:]', dim: 'int', Dxx: 'float', Dyy: 'float', Dzz: 'float',
             shared_dt: 'float[:]'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, len(cell_faceid), stride):
      lam = 0.0
      for j in range(cell_faceid[i][-1]):
        norm = face_normal[cell_faceid[i][j]]
        lam += abs(u[i] * norm[0] + v[i] * norm[1] + w[i] * norm[2])
        mes2 = norm[0] * norm[0] + norm[1] * norm[1] + norm[2] * norm[2]
        lam += (Dxx * mes2 + Dyy * mes2 + Dzz * mes2) / cell_volume[i]
      if lam != 0.0:
        cuda.atomic.min(shared_dt, 0, cfl * cell_volume[i] / lam)

  kernel = gpu.compile_kernel(kernel)
  argcache = {}
  def result(*args):
    args = GPUArray.to_device_list(argcache, args)
    gpu.assign(d_shared_dt, 1e6)
    grid, block = gpu.get_gpu_params(len(args[7]))  # cell_faceid
    kernel[grid, block, gpu.stream](*args, d_shared_dt)
    # Lecture host de dt : LE point de synchro du pas de temps (draine le stream,
    # donc tous les kernels en file -- gradient, interpolate, transport -- sont finis).
    host = d_shared_dt.copy_to_host(stream=gpu.stream)
    gpu.synchronize()
    return host[0]

  return result


# ---------------------------------------------------------------------------
def get_kernel_update_new_value():
  gpu = get_active_backend()

  def kernel(ne_c: 'float[:]', rez_ne: 'float[:]', dissip_ne: 'float[:]', src_ne: 'float[:]',
             dtime: 'float', cell_volume: 'float[:]'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, ne_c.shape[0], stride):
      cuda.atomic.add(ne_c, i, dtime * ((rez_ne[i] + dissip_ne[i]) / cell_volume[i] + src_ne[i]))

  kernel = gpu.compile_kernel(kernel)
  argcache = {}

  def result(*args):
    args = GPUArray.to_device_list(argcache, args)
    grid, block = gpu.get_gpu_params(len(args[0]))
    kernel[grid, block, gpu.stream](*args)
    # pas de synchronize : enchainement sur le meme stream (ordonne) ; la synchro
    # n'a lieu qu'a la lecture host (dt dans time_step, to_host a la sauvegarde).

  return result
