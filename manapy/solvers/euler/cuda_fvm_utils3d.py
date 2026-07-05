# -*- coding: utf-8 -*-
"""
GPU (CUDA) kernels for the 3D Euler solver — order-1 Rusanov, the scheme used by
the euler3d entropy-wave benchmark. Same approach as advecdiff/cuda_fvm_utils.py:

  - signatures are IDENTICAL to the CPU kernels (euler/fvm_utils3d_compute.py), so
    each get_kernel_*() closure is a drop-in for the EulerSolver attributes;
  - thread indexing (cuda.grid) instead of the CPU `for i in range(...)`;
  - the residual is a face->cell scatter, so inner faces write both neighbours from
    different threads -> cuda.atomic.add.

Scope: rusanov order 1, Neumann BC, non-viscous, constant gamma, single-GPU
(mono-rank, no GPU halos). WENO / Roe / viscous / variable-gamma are NOT ported.
"""
import math
from numba import cuda

from manapy.backends.gpu import get_active_backend, GPUArray


# ---------------------------------------------------------------------------
# device: Rusanov flux (exact port of _compute_flux_euler_3d_rusanov)
# ---------------------------------------------------------------------------
def device_flux_rusanov(rhol: 'float', Pl: 'float', rhoul: 'float', rhovl: 'float', rhowl: 'float', rhoEl: 'float',
                        rhor: 'float', Pr: 'float', rhour: 'float', rhovr: 'float', rhowr: 'float', rhoEr: 'float',
                        normal: 'float[:]', mesure: 'float', gamma: 'float', flux_out: 'float[:]'):
  ql = rhoul * normal[0] + rhovl * normal[1] + rhowl * normal[2]
  qr = rhour * normal[0] + rhovl * normal[1] + rhowl * normal[2]   # NB: matches CPU as-is
  cl = math.sqrt(gamma * Pl / rhol)
  cr = math.sqrt(gamma * Pr / rhor)

  fl_rho = rhoul * normal[0] + rhovl * normal[1] + rhowl * normal[2]
  fl_rhou = (rhoul * rhoul / rhol + Pl) * normal[0] + (rhoul * rhovl / rhol) * normal[1] + (rhoul * rhowl / rhol) * normal[2]
  fl_rhov = (rhovl * rhoul / rhol) * normal[0] + (rhovl * rhovl / rhol + Pl) * normal[1] + (rhovl * rhowl / rhol) * normal[2]
  fl_rhow = (rhoul * rhowl / rhol) * normal[0] + (rhovl * rhowl / rhol) * normal[1] + (rhowl * rhowl / rhol + Pl) * normal[2]
  fl_rhoE = rhoul / rhol * (rhoEl + Pl) * normal[0] + rhovl / rhol * (rhoEl + Pl) * normal[1] + rhowl / rhol * (rhoEl + Pl) * normal[2]

  fr_rho = rhour * normal[0] + rhovr * normal[1] + rhowr * normal[2]
  fr_rhou = (rhour * rhour / rhor + Pr) * normal[0] + (rhour * rhovr / rhor) * normal[1] + (rhour * rhowr / rhor) * normal[2]
  fr_rhov = (rhovr * rhour / rhor) * normal[0] + (rhovr * rhovr / rhor + Pr) * normal[1] + (rhovr * rhowr / rhor) * normal[2]
  fr_rhow = (rhour * rhowr / rhor) * normal[0] + (rhovr * rhowr / rhor) * normal[1] + (rhowr * rhowr / rhor + Pr) * normal[2]
  fr_rhoE = rhour / rhor * (rhoEr + Pr) * normal[0] + rhovr / rhor * (rhoEr + Pr) * normal[1] + rhowr / rhor * (rhoEr + Pr) * normal[2]

  ll1 = abs(ql / rhol - cl); ll2 = abs(ql / rhol); ll3 = abs(ql / rhol + cl)
  lr1 = abs(qr / rhor - cr); lr2 = abs(qr / rhor); lr3 = abs(qr / rhor + cr)
  Ll = max(ll1, max(ll2, ll3))
  Lr = max(lr1, max(lr2, lr3))
  S = Ll if Ll > Lr else Lr

  flux_out[0] = (0.5 * (fl_rho + fr_rho) - 0.5 * S * (rhor - rhol)) * mesure
  flux_out[1] = (0.5 * (fl_rhou + fr_rhou) - 0.5 * S * (rhour - rhoul)) * mesure
  flux_out[2] = (0.5 * (fl_rhov + fr_rhov) - 0.5 * S * (rhovr - rhovl)) * mesure
  flux_out[3] = (0.5 * (fl_rhow + fr_rhow) - 0.5 * S * (rhowr - rhowl)) * mesure
  flux_out[4] = (0.5 * (fl_rhoE + fr_rhoE) - 0.5 * S * (rhoEr - rhoEl)) * mesure


# ---------------------------------------------------------------------------
def get_kernel_explicitscheme_euler_3d_rusanov():
  gpu = get_active_backend()
  flux = gpu.compile_kernel(device_flux_rusanov, device=True)

  def kernel(rez_rho: 'float[:]', rez_rhou: 'float[:]', rez_rhov: 'float[:]', rez_rhow: 'float[:]', rez_rhoE: 'float[:]',
             rho_c: 'float[:]', P_c: 'float[:]', rhou_c: 'float[:]', rhov_c: 'float[:]', rhow_c: 'float[:]', rhoE_c: 'float[:]',
             rho_g: 'float[:]', P_g: 'float[:]', rhou_g: 'float[:]', rhov_g: 'float[:]', rhow_g: 'float[:]', rhoE_g: 'float[:]',
             rho_h: 'float[:]', P_h: 'float[:]', rhou_h: 'float[:]', rhov_h: 'float[:]', rhow_h: 'float[:]', rhoE_h: 'float[:]',
             cellidf: 'int[:,:]', halofid: 'int[:]', normal: 'float[:,:]', mesurf: 'float[:]', name: 'int[:]', gamma: 'float',
             tangent: 'float[:,:]', binormal: 'float[:,:]'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    nrm = cuda.local.array(3, normal.dtype)
    fout = cuda.local.array(5, rez_rho.dtype)
    for i in range(start, cellidf.shape[0], stride):
      mesu = mesurf[i]
      nrm[0] = normal[i][0] / mesu; nrm[1] = normal[i][1] / mesu; nrm[2] = normal[i][2] / mesu
      c0 = cellidf[i][0]
      rhol = rho_c[c0]; Pl = P_c[c0]; rhoul = rhou_c[c0]; rhovl = rhov_c[c0]; rhowl = rhow_c[c0]; rhoEl = rhoE_c[c0]
      if name[i] == 0:
        c1 = cellidf[i][1]
        flux(rhol, Pl, rhoul, rhovl, rhowl, rhoEl,
             rho_c[c1], P_c[c1], rhou_c[c1], rhov_c[c1], rhow_c[c1], rhoE_c[c1], nrm, mesu, gamma, fout)
        cuda.atomic.add(rez_rho, c0, -fout[0]);  cuda.atomic.add(rez_rho, c1, fout[0])
        cuda.atomic.add(rez_rhou, c0, -fout[1]); cuda.atomic.add(rez_rhou, c1, fout[1])
        cuda.atomic.add(rez_rhov, c0, -fout[2]); cuda.atomic.add(rez_rhov, c1, fout[2])
        cuda.atomic.add(rez_rhow, c0, -fout[3]); cuda.atomic.add(rez_rhow, c1, fout[3])
        cuda.atomic.add(rez_rhoE, c0, -fout[4]); cuda.atomic.add(rez_rhoE, c1, fout[4])
      elif name[i] == 10:
        h = halofid[i]
        flux(rhol, Pl, rhoul, rhovl, rhowl, rhoEl,
             rho_h[h], P_h[h], rhou_h[h], rhov_h[h], rhow_h[h], rhoE_h[h], nrm, mesu, gamma, fout)
        cuda.atomic.add(rez_rho, c0, -fout[0]);  cuda.atomic.add(rez_rhou, c0, -fout[1])
        cuda.atomic.add(rez_rhov, c0, -fout[2]); cuda.atomic.add(rez_rhow, c0, -fout[3])
        cuda.atomic.add(rez_rhoE, c0, -fout[4])
      else:
        flux(rhol, Pl, rhoul, rhovl, rhowl, rhoEl,
             rho_g[i], P_g[i], rhou_g[i], rhov_g[i], rhow_g[i], rhoE_g[i], nrm, mesu, gamma, fout)
        cuda.atomic.add(rez_rho, c0, -fout[0]);  cuda.atomic.add(rez_rhou, c0, -fout[1])
        cuda.atomic.add(rez_rhov, c0, -fout[2]); cuda.atomic.add(rez_rhow, c0, -fout[3])
        cuda.atomic.add(rez_rhoE, c0, -fout[4])

  kernel = gpu.compile_kernel(kernel)
  argcache = {}

  def result(*args):
    args = GPUArray.to_device_list(argcache, args)
    for k in range(5):                 # zero the five residual accumulators
      gpu.assign(args[k], 0.0)
    grid, block = gpu.get_gpu_params(len(args[24]))   # cellidf = nbfaces
    kernel[grid, block, gpu.stream](*args)

  return result


# ---------------------------------------------------------------------------
def get_kernel_time_step_euler_3d():
  gpu = get_active_backend()

  def kernel(rho: 'float[:]', P: 'float[:]', rhou: 'float[:]', rhov: 'float[:]', rhow: 'float[:]',
             cfl: 'float', normal: 'float[:,:]', mesure: 'float[:]', volume: 'float[:]',
             faceid: 'int[:,:]', gamma: 'float', dt_c: 'float[:]'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, len(faceid), stride):
      lam = 0.0
      velson = math.sqrt(gamma * abs(P[i] / rho[i]))
      for j in range(4):
        f = faceid[i][j]
        nx = normal[f][0]; ny = normal[f][1]; nz = normal[f][2]
        nn = math.sqrt(nx * nx + ny * ny + nz * nz)
        u_n = abs((rhou[i] * nx + rhov[i] * ny + rhow[i] * nz) / rho[i]) / nn
        lam += (u_n + velson) * mesure[f]
      dt_c[i] = cfl * volume[i] / lam

  kernel = gpu.compile_kernel(kernel)
  argcache = {}

  def result(*args):
    host_dtc = args[11]                               # original host dt_c
    dev = GPUArray.to_device_list(argcache, args)
    grid, block = gpu.get_gpu_params(len(dev[9]))     # faceid = ncells
    kernel[grid, block, gpu.stream](*dev)
    # copy the device-filled dt_c back to the host array so stepper's .min() sees it
    d = dev[11]
    src = d.to_host() if hasattr(d, "to_host") else d.copy_to_host(stream=gpu.stream)
    gpu.synchronize()
    host_dtc[:] = src

  return result


# ---------------------------------------------------------------------------
def get_kernel_update_euler_3d_fvc():
  gpu = get_active_backend()

  def kernel(rho_c: 'float[:]', P_c: 'float[:]', rhou_c: 'float[:]', rhov_c: 'float[:]', rhow_c: 'float[:]', rhoE_c: 'float[:]',
             rez_rho: 'float[:]', rez_rhou: 'float[:]', rez_rhov: 'float[:]', rez_rhow: 'float[:]', rez_rhoE: 'float[:]',
             gamma: 'float', dtime: 'float', vol: 'float[:]'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, rho_c.shape[0], stride):
      rho_c[i] += dtime * (rez_rho[i] / vol[i])
      rhou_c[i] += dtime * (rez_rhou[i] / vol[i])
      rhov_c[i] += dtime * (rez_rhov[i] / vol[i])
      rhow_c[i] += dtime * (rez_rhow[i] / vol[i])
      rhoE_c[i] += dtime * (rez_rhoE[i] / vol[i])
      P_c[i] = (gamma - 1.0) * (rhoE_c[i] - 0.5 * (rhou_c[i] * rhou_c[i] + rhov_c[i] * rhov_c[i] + rhow_c[i] * rhow_c[i]) / rho_c[i])

  kernel = gpu.compile_kernel(kernel)
  argcache = {}

  def result(*args):
    args = GPUArray.to_device_list(argcache, args)
    grid, block = gpu.get_gpu_params(len(args[0]))
    kernel[grid, block, gpu.stream](*args)

  return result


# ---------------------------------------------------------------------------
def get_kernel_ghost_value_Neumann3D():
  gpu = get_active_backend()

  def kernel(rhog: 'float[:]', Pg: 'float[:]', rhoug: 'float[:]', rhovg: 'float[:]', rhowg: 'float[:]',
             ug: 'float[:]', vg: 'float[:]', wg: 'float[:]', rhoEg: 'float[:]',
             rhoc: 'float[:]', Pc: 'float[:]', rhouc: 'float[:]', rhovc: 'float[:]', rhowc: 'float[:]', rhoEc: 'float[:]',
             cellid: 'int[:,:]', name: 'int[:]', normal: 'float[:,:]', mesure: 'float[:]'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, cellid.shape[0], stride):
      nm = name[i]
      if nm == 1 or nm == 2 or nm == 3 or nm == 4 or nm == 5 or nm == 6:
        c0 = cellid[i][0]
        rhog[i] = rhoc[c0]; rhoug[i] = rhouc[c0]; rhovg[i] = rhovc[c0]
        rhowg[i] = rhowc[c0]; rhoEg[i] = rhoEc[c0]; Pg[i] = Pc[c0]
        ug[i] = rhoug[i] / rhog[i]; vg[i] = rhovg[i] / rhog[i]; wg[i] = rhowg[i] / rhog[i]

  kernel = gpu.compile_kernel(kernel)
  argcache = {}

  def result(*args):
    args = GPUArray.to_device_list(argcache, args)
    grid, block = gpu.get_gpu_params(len(args[15]))   # cellid = nbfaces
    kernel[grid, block, gpu.stream](*args)

  return result
