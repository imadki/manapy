import numpy as np
from numba import cuda
from manapy.cuda.utils import (
    VarClass,
    GPU_Backend
)
from manapy.cuda.manapy.util_kernels import (
  kernel_assign,
  device_compute_upwind_flux,
)


# ✅ ❌ 🔨
# compute_upwind_flux ✅
# explicitscheme_convective_2d ✅
# explicitscheme_convective_3d ✅
# time_step ✅
# update_new_value ✅



# need compute_upwind_flux ✅
# kernel_assign -> rez_w[:] = 0 ✅
def get_kernel_explicitscheme_convective_2d():
  
  compute_upwind_flux = GPU_Backend.compile_kernel(device_compute_upwind_flux, device = True)


  def kernel_explicitscheme_convective_2d(
    rez_w:'float[:]', 
    w_c:'float[:]', 
    w_ghost:'float[:]', 
    w_halo:'float[:]',
    u_face:'float[:]', 
    v_face:'float[:]', 
    w_face:'float[:]', 
    w_x:'float[:]', 
    w_y:'float[:]', 
    w_z:'float[:]', 
    wx_halo:'float[:]', 
    wy_halo:'float[:]', 
    wz_halo:'float[:]', 
    psi:'float[:]', 
    psi_halo:'float[:]', 
    centerc:'float[:,:]', 
    centerf:'float[:,:]', 
    centerh:'float[:,:]', 
    centerg:'float[:,:]',
    cellidf:'int32[:,:]', 
    normalf:'float[:,:]', 
    halofid:'int32[:]',
    name:'uint32[:]', 
    innerfaces:'uint32[:]', 
    halofaces:'uint32[:]', 
    boundaryfaces:'uint32[:]', 
    periodicboundaryfaces:'uint32[:]', 
    shift:'float[:,:]', 
    order:'int32'
    ):

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    r_l =  cuda.local.array(2, centerc.dtype)
    r_r = cuda.local.array(2, centerc.dtype)
    flux_w = cuda.local.array(1, centerc.dtype)  

    #? rez_w[:] = 0.

    for idx in range(start, innerfaces.shape[0], stride):
      i = innerfaces[idx]
      
      w_l = w_c[cellidf[i][0]]
      normal = normalf[i]
      
      w_r = w_c[cellidf[i][1]]
      
      center_left = centerc[cellidf[i][0]]
      center_right = centerc[cellidf[i][1]]
      
      w_x_left = w_x[cellidf[i][0]]; w_x_right = w_x[cellidf[i][1]]
      w_y_left = w_y[cellidf[i][0]]; w_y_right = w_y[cellidf[i][1]]
      
      psi_left  = psi[cellidf[i][0]];  psi_right  = psi[cellidf[i][1]]
      
      r_l[0] = centerf[i][0] - center_left[0]; r_r[0] = centerf[i][0] - center_right[0]; 
      r_l[1] = centerf[i][1] - center_left[1]; r_r[1] = centerf[i][1] - center_right[1]; 
      
      w_l  = w_l  + (order - 1) * psi_left  * (w_x_left * r_l[0]  + w_y_left * r_l[1] )
      w_r  = w_r  + (order - 1) * psi_right * (w_x_right* r_r[0]  + w_y_right* r_r[1] )
      
      compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
      
      cuda.atomic.add(rez_w, cellidf[i][0], -flux_w[0])
      cuda.atomic.add(rez_w, cellidf[i][1], flux_w[0])
      #? rez_w[cellidf[i][0]]  -= flux_w[0]
      #? rez_w[cellidf[i][1]]  += flux_w[0]
    
    for idx in range(start, periodicboundaryfaces.shape[0], stride):
      i = periodicboundaryfaces[idx]
      
      w_l = w_c[cellidf[i][0]]
      normal = normalf[i]
      
      w_r = w_c[cellidf[i][1]]
      
      center_left = centerc[cellidf[i][0]]
      center_right = centerc[cellidf[i][1]]

      w_x_left = w_x[cellidf[i][0]]; w_x_right = w_x[cellidf[i][1]]
      w_y_left = w_y[cellidf[i][0]]; w_y_right = w_y[cellidf[i][1]]
      
      psi_left  = psi[cellidf[i][0]];  psi_right  = psi[cellidf[i][1]]
      
      if name[i] == 11 or name[i] == 22:
        r_l[0] = centerf[i][0] - center_left[0]; r_r[0] = centerf[i][0] - center_right[0] - shift[cellidf[i][1]][0] 
        r_l[1] = centerf[i][1] - center_left[1]; r_r[1] = centerf[i][1] - center_right[1] 
          
      if name[i] == 33 or name[i] == 44:
        r_l[0] = centerf[i][0] - center_left[0]; r_r[0] = centerf[i][0] - center_right[0] 
        r_l[1] = centerf[i][1] - center_left[1]; r_r[1] = centerf[i][1] - center_right[1] - shift[cellidf[i][1]][1] 
      
      w_l  = w_l  + (order - 1) * psi_left  * (w_x_left * r_l[0]  + w_y_left * r_l[1] )
      w_r  = w_r  + (order - 1) * psi_right * (w_x_right* r_r[0]  + w_y_right* r_r[1] )
      
      compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)

      cuda.atomic.add(rez_w, cellidf[i][0], -flux_w[0])
      #? rez_w[cellidf[i][0]]  -= flux_w[0]
                
    for idx in range(start, halofaces.shape[0], stride):
      i = halofaces[idx]
      
      w_l = w_c[cellidf[i][0]]
      normal = normalf[i]
      
      w_r  = w_halo[halofid[i]]
      
      center_left = centerc[cellidf[i][0]]
      center_right = centerh[halofid[i]]

      w_x_left = w_x[cellidf[i][0]];  w_x_right = wx_halo[halofid[i]]
      w_y_left = w_y[cellidf[i][0]];  w_y_right = wy_halo[halofid[i]]
      
      psi_left  = psi[cellidf[i][0]];   psi_right  = psi_halo[halofid[i]]
      
      r_l[0] = centerf[i][0] - center_left[0]; r_r[0] = centerf[i][0] - center_right[0]; 
      r_l[1] = centerf[i][1] - center_left[1]; r_r[1] = centerf[i][1] - center_right[1]; 
      
      w_l  = w_l  + (order - 1) * psi_left  * (w_x_left   * r_l[0] + w_y_left   * r_l[1])
      w_r  = w_r  + (order - 1) * psi_right * (w_x_right  * r_r[0] + w_y_right  * r_r[1])
      
      compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)

      cuda.atomic.add(rez_w, cellidf[i][0], -flux_w[0])
      #? rez_w[cellidf[i][0]]  -= flux_w[0]

    for idx in range(start, boundaryfaces.shape[0], stride):
      i = boundaryfaces[idx]
  
      w_l = w_c[cellidf[i][0]]
      normal = normalf[i]
      
      w_r  = w_ghost[i]
      center_left = centerc[cellidf[i][0]]
      
      w_x_left = w_x[cellidf[i][0]]; 
      w_y_left = w_y[cellidf[i][0]]; 
      
      psi_left  = psi[cellidf[i][0]];  
      
      r_l[0] = centerf[i][0] - center_left[0]
      r_l[1] = centerf[i][1] - center_left[1]
      
      w_l  = w_l  + (order - 1) * psi_left  * (w_x_left * r_l[0]  + w_y_left * r_l[1] )
      w_r  = w_r  
          
      compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)

      cuda.atomic.add(rez_w, cellidf[i][0], -flux_w[0])
      #? rez_w[cellidf[i][0]]  -= flux_w[0]

  kernel_explicitscheme_convective_2d = GPU_Backend.compile_kernel(kernel_explicitscheme_convective_2d)
  kernel_assign_float = GPU_Backend.compile_kernel(kernel_assign)
  
  def result(*args):
    VarClass.debug(kernel_explicitscheme_convective_2d, args)
    args = [VarClass.to_device(arg) for arg in args]
    size = len(args[0]) #rez_w
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_assign_float[nb_blocks, nb_threads, GPU_Backend.stream](args[0], 0.0) #rhs
    GPU_Backend.stream.synchronize()
    
    
    # innerfaces periodicboundaryfaces halofaces boundaryfaces
    size = max(len(args[23]), len(args[24]), len(args[25]), len(args[26]))
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_explicitscheme_convective_2d[nb_blocks, nb_threads, GPU_Backend.stream](*args)
    GPU_Backend.stream.synchronize()

  return result  

#------------------------------------
#------------------------------------
#------------------------------------


# need compute_upwind_flux ✅
# kernel_assign -> rez_w[:] = 0 ✅
def get_kernel_explicitscheme_convective_3d():

  compute_upwind_flux = GPU_Backend.compile_kernel(device_compute_upwind_flux, device = True)

  def kernel_explicitscheme_convective_3d(
    rez_w:'float[:]', 
    w_c:'float[:]', 
    w_ghost:'float[:]', 
    w_halo:'float[:]',
    u_face:'float[:]', 
    v_face:'float[:]', 
    w_face:'float[:]', 
    w_x:'float[:]', 
    w_y:'float[:]', 
    w_z:'float[:]', 
    wx_halo:'float[:]', 
    wy_halo:'float[:]', 
    wz_halo:'float[:]', 
    psi:'float[:]', 
    psi_halo:'float[:]', 
    centerc:'float[:,:]', 
    centerf:'float[:,:]', 
    centerh:'float[:,:]', 
    centerg:'float[:,:]',
    cellidf:'int32[:,:]', 
    normalf:'float[:,:]', 
    halofid:'int32[:]', 
    name:'uint32[:]',
    innerfaces:'uint32[:]', 
    halofaces:'uint32[:]', 
    boundaryfaces:'uint32[:]', 
    periodicboundaryfaces:'uint32[:]', 
    shift:'float[:,:]', 
    order:'int32'
    ):
    
      
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    r_l =  cuda.local.array(3, centerc.dtype)
    r_r = cuda.local.array(3, centerc.dtype)
    flux_w = cuda.local.array(1, centerc.dtype)  
    
    
    #? rez_w[:] = 0.


    for idx in range(start, innerfaces.shape[0], stride):
      i = innerfaces[idx]
      
      w_l = w_c[cellidf[i][0]]
      normal = normalf[i]
      
      w_r = w_c[cellidf[i][1]]
      
      center_left = centerc[cellidf[i][0]]
      center_right = centerc[cellidf[i][1]]
      
      w_x_left = w_x[cellidf[i][0]]; w_x_right = w_x[cellidf[i][1]]
      w_y_left = w_y[cellidf[i][0]]; w_y_right = w_y[cellidf[i][1]]
      w_z_left = w_z[cellidf[i][0]]; w_z_right = w_z[cellidf[i][1]]
      
      psi_left  = psi[cellidf[i][0]];  psi_right  = psi[cellidf[i][1]]
      
      r_l[0] = centerf[i][0] - center_left[0]; r_r[0] = centerf[i][0] - center_right[0]; 
      r_l[1] = centerf[i][1] - center_left[1]; r_r[1] = centerf[i][1] - center_right[1]; 
      r_l[2] = centerf[i][2] - center_left[2]; r_r[2] = centerf[i][2] - center_right[2]; 
      
      w_l  = w_l  + (order - 1) * psi_left  * (w_x_left * r_l[0]  + w_y_left * r_l[1] + w_z_left * r_l[2] )
      w_r  = w_r  + (order - 1) * psi_right * (w_x_right* r_r[0]  + w_y_right* r_r[1] + w_z_right* r_r[2] )
      
      compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
      
      cuda.atomic.add(rez_w, cellidf[i][0], -flux_w[0])
      cuda.atomic.add(rez_w, cellidf[i][1], flux_w[0])
      #? rez_w[cellidf[i][0]]  -= flux_w[0]
      #? rez_w[cellidf[i][1]]  += flux_w[0]
    
    for idx in range(start, periodicboundaryfaces.shape[0], stride):
      i = periodicboundaryfaces[idx]
      
      w_l = w_c[cellidf[i][0]]
      normal = normalf[i]
      
      w_r = w_c[cellidf[i][1]]
      
      center_left = centerc[cellidf[i][0]]
      center_right = centerc[cellidf[i][1]]

      w_x_left = w_x[cellidf[i][0]]; w_x_right = w_x[cellidf[i][1]]
      w_y_left = w_y[cellidf[i][0]]; w_y_right = w_y[cellidf[i][1]]
      w_z_left = w_z[cellidf[i][0]]; w_z_right = w_z[cellidf[i][1]]
      
      psi_left  = psi[cellidf[i][0]];  psi_right  = psi[cellidf[i][1]]
          
      if name[i] == 11 or name[i] == 22:
          r_l[0] = centerf[i][0] - center_left[0]; r_r[0] = centerf[i][0] - center_right[0] - shift[cellidf[i][1]][0] 
          r_l[1] = centerf[i][1] - center_left[1]; r_r[1] = centerf[i][1] - center_right[1] 
          r_l[2] = centerf[i][2] - center_left[2]; r_r[2] = centerf[i][2] - center_right[2]
          
      if name[i] == 33 or name[i] == 44:
          r_l[0] = centerf[i][0] - center_left[0]; r_r[0] = centerf[i][0] - center_right[0] 
          r_l[1] = centerf[i][1] - center_left[1]; r_r[1] = centerf[i][1] - center_right[1] - shift[cellidf[i][1]][1] 
          r_l[2] = centerf[i][2] - center_left[2]; r_r[2] = centerf[i][2] - center_right[2]
      
      if name[i] == 55 or name[i] == 66:
          r_l[0] = centerf[i][0] - center_left[0]; r_r[0] = centerf[i][0] - center_right[0] 
          r_l[1] = centerf[i][1] - center_left[1]; r_r[1] = centerf[i][1] - center_right[1] 
          r_l[2] = centerf[i][2] - center_left[2]; r_r[2] = centerf[i][2] - center_right[2] - shift[cellidf[i][1]][2] 
      
      w_l  = w_l  + (order - 1) * psi_left  * (w_x_left * r_l[0]  + w_y_left * r_l[1] + w_z_left * r_l[2] )
      w_r  = w_r  + (order - 1) * psi_right * (w_x_right* r_r[0]  + w_y_right* r_r[1] + w_z_right* r_r[2] )
      
      compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
      
      cuda.atomic.add(rez_w, cellidf[i][0], -flux_w[0])
      #? rez_w[cellidf[i][0]]  -= flux_w[0]
                
    for idx in range(start, halofaces.shape[0], stride):
      i = halofaces[idx] 
      
      w_l = w_c[cellidf[i][0]]
      normal = normalf[i]
      
      w_r  = w_halo[halofid[i]]
      
      center_left = centerc[cellidf[i][0]]
      center_right = centerh[halofid[i]]

      w_x_left = w_x[cellidf[i][0]];  w_x_right = wx_halo[halofid[i]]
      w_y_left = w_y[cellidf[i][0]];  w_y_right = wy_halo[halofid[i]]
      w_z_left = w_z[cellidf[i][0]];  w_z_right = wz_halo[halofid[i]]
      
      psi_left  = psi[cellidf[i][0]];   psi_right  = psi_halo[halofid[i]]
      
      r_l[0] = centerf[i][0] - center_left[0]; r_r[0] = centerf[i][0] - center_right[0] 
      r_l[1] = centerf[i][1] - center_left[1]; r_r[1] = centerf[i][1] - center_right[1] 
      r_l[2] = centerf[i][2] - center_left[2]; r_r[2] = centerf[i][2] - center_right[2]
      
      w_l  = w_l  + (order - 1) * psi_left  * (w_x_left * r_l[0]  + w_y_left * r_l[1] + w_z_left * r_l[2] )
      w_r  = w_r  + (order - 1) * psi_right * (w_x_right* r_r[0]  + w_y_right* r_r[1] + w_z_right* r_r[2] )
      
      compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
      
      cuda.atomic.add(rez_w, cellidf[i][0], -flux_w[0])
      #? rez_w[cellidf[i][0]]  -= flux_w[0]
    
    for idx in range(start, boundaryfaces.shape[0], stride):
      i = boundaryfaces[idx]
    
      w_l = w_c[cellidf[i][0]]
      normal = normalf[i]
      
      w_r  = w_ghost[i]
      center_left = centerc[cellidf[i][0]]
      
      w_x_left = w_x[cellidf[i][0]] 
      w_y_left = w_y[cellidf[i][0]] 
      w_z_left = w_z[cellidf[i][0]]
      
      psi_left  = psi[cellidf[i][0]];  
      
      r_l[0] = centerf[i][0] - center_left[0] 
      r_l[1] = centerf[i][1] - center_left[1]
      r_l[2] = centerf[i][2] - center_left[2]
      
      w_l  = w_l  + (order - 1) * psi_left  * (w_x_left * r_l[0]  + w_y_left * r_l[1] + w_z_left * r_l[2] )
      w_r  = w_r  
              
      compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)

      cuda.atomic.add(rez_w, cellidf[i][0], -flux_w[0])
      #? rez_w[cellidf[i][0]]  -= flux_w[0]
    
  kernel_explicitscheme_convective_3d = GPU_Backend.compile_kernel(kernel_explicitscheme_convective_3d)
  kernel_assign_float = GPU_Backend.compile_kernel(kernel_assign)
  
  def result(*args):
    VarClass.debug(kernel_explicitscheme_convective_3d, args)
    args = [VarClass.to_device(arg) for arg in args]
    size = len(args[0]) #rez_w
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_assign_float[nb_blocks, nb_threads, GPU_Backend.stream](args[0], 0.0) #rhs
    GPU_Backend.stream.synchronize()
    
    
    # innerfaces periodicboundaryfaces halofaces boundaryfaces
    size = max(len(args[23]), len(args[24]), len(args[25]), len(args[26]))
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_explicitscheme_convective_3d[nb_blocks, nb_threads, GPU_Backend.stream](*args)
    GPU_Backend.stream.synchronize()

  return result  


#------------------------------------
#------------------------------------
#------------------------------------

# kernel_assign -> shared_dt = 1e6 ✅
# return  ✅
# extra_param ✅
def get_kernel_time_step():
  
  d_shared_dt = cuda.device_array(shape=(1), dtype=GPU_Backend.float_precision)

  def kernel_time_step(
    u:'float[:]', 
    v:'float[:]', 
    w:'float[:]', 
    cfl:'float', 
    normal:'float[:,:]',
    mesure:'float[:]', 
    volume:'float[:]', 
    faceid:'int32[:,:]', 
    dim:'int32', 
    shared_dt : 'float[:]'
    ):

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    #? dt = 1e6

    for i in range(start, len(faceid), stride):
      lam = 0.0
      
      for j in range(faceid[i][-1]):
        norm = normal[faceid[i][j]]
        lam_convect = abs(u[i]*norm[0] + v[i]*norm[1] + w[i]*norm[2])
        lam += lam_convect
      
      # if i < 10:
      #   print(lam, u[i],norm[0] , v[i],norm[1] , w[i],norm[2])
      cuda.atomic.min(shared_dt, 0, cfl * volume[i] / lam)
      #? dt  = min(dt, cfl * volume[i]/lam)

  kernel_time_step = GPU_Backend.compile_kernel(kernel_time_step)
  kernel_assign_float = GPU_Backend.compile_kernel(kernel_assign)

  def result(*args):
    VarClass.debug(kernel_time_step, args)
    args = [VarClass.to_device(arg) for arg in args]
    kernel_assign_float[1, 1, GPU_Backend.stream](d_shared_dt, 1e6) #rhs
    GPU_Backend.stream.synchronize()
    
    # shared_dt
    size = len(args[7])
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_time_step[nb_blocks, nb_threads, GPU_Backend.stream](*args, d_shared_dt)
    return d_shared_dt.copy_to_host(stream=GPU_Backend.stream)[0]
    #GPU_Backend.stream.synchronize()

  return result 

#------------------------------------
#------------------------------------
#------------------------------------

def get_kernel_update_new_value():

  def kernel_update_new_value(
    ne_c:'float[:]', 
    rez_ne:'float[:]', 
    dissip_ne:'float[:]', 
    src_ne:'float[:]',
    dtime:'float', 
    vol:'float[:]'
    ):

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, ne_c.shape[0], stride):
      v  = dtime  * ((rez_ne[i]  +  dissip_ne[i]) / vol[i] + src_ne[i] )
      cuda.atomic.add(ne_c, i, v)

  kernel_update_new_value = GPU_Backend.compile_kernel(kernel_update_new_value)

  def result(*args):
    # ne_c
    VarClass.debug(kernel_update_new_value, args)
    args = [VarClass.to_device(arg) for arg in args]
    size = len(args[0])
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_update_new_value[nb_blocks, nb_threads, GPU_Backend.stream](*args)
    GPU_Backend.stream.synchronize()

  return result 

