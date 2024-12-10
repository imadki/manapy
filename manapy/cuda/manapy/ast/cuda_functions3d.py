import numpy as np
import math
from numba import cuda
from manapy.cuda.utils import (
    VarClass,
    GPU_Backend
)
from manapy.cuda.manapy.util_kernels import (
  kernel_assign,
  device_search_element
)


#? manapy/ast/functions3d.py

# ✅ ❌ 🔨
# cell_gradient_3d ✅ 
# face_gradient_3d ✅
# centertovertex_3d ✅
# barthlimiter_3d ✅ => need a test because the result is [1., 1., ..., 1.]
# compute_3dmatrix_size ✅
# get_triplet_3d ✅
# get_rhs_loc_3d ✅ => need a test because the result is [0., 0., ..., 0.]
# get_rhs_glob_3d ✅ => in the original function there so no reset to rhs[:] = 0.0 ||| as in get_rhs_loc_3d [0., 0., ..., 0.]
# compute_P_gradient_3d_diamond ✅

#! In get_triplet_3d
#! a_loc jcn_loc irn_loc if cmpt does not cover all the element of these arrays it should be reset before start the kernel


def get_kernel_cell_gradient_3d():
  
  def kernel_cell_gradient_3d(
    w_c:'float[:]', 
    w_ghost:'float[:]', 
    w_halo:'float[:]', 
    w_haloghost:'float[:]',
    centerc:'float[:,:]', 
    cellnid:'int32[:,:]', 
    ghostnid:'int32[:,:]', 
    haloghostnid:'int32[:,:]', 
    halonid:'int32[:,:]',
    nodecid:'uint32[:,:]', 
    periodicn:'int32[:,:]', 
    periodic:'int32[:,:]', 
    centergf:'float[:,:]', 
    halocenterg:'float[:,:]', 
    vertexn:'float[:,:]',
    centerh:'float[:,:]',
    shift:'float[:,:]',
    w_x:'float[:]',
    w_y:'float[:]', 
    w_z:'float[:]'
    ):

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, len(w_c), stride):
      i_xx = 0.
      i_yy = 0.
      i_zz = 0.
      i_xy = 0.
      i_xz = 0.
      i_yz = 0.
      
      j_x = 0.
      j_y = 0.
      j_z = 0.

      for j in range(cellnid[i][-1]):
        cell = cellnid[i][j]
        jx = centerc[cell][0] - centerc[i][0]
        jy = centerc[cell][1] - centerc[i][1]
        jz = centerc[cell][2] - centerc[i][2]
        
        i_xx += jx*jx
        i_yy += jy*jy
        i_zz += jz*jz
        i_xy += (jx * jy)
        i_xz += (jx * jz)
        i_yz += (jy * jz)

        j_x += (jx * (w_c[cell] - w_c[i] ))
        j_y += (jy * (w_c[cell] - w_c[i] ))
        j_z += (jz * (w_c[cell] - w_c[i] ))
          
      for j in range(ghostnid[i][-1]):
        cell = ghostnid[i][j]
        jx = centergf[cell][0] - centerc[i][0]
        jy = centergf[cell][1] - centerc[i][1]
        jz = centergf[cell][2] - centerc[i][2]
        
        i_xx += jx*jx
        i_yy += jy*jy
        i_zz += jz*jz
        i_xy += (jx * jy)
        i_xz += (jx * jz)
        i_yz += (jy * jz)

        j_x += (jx * (w_ghost[cell] - w_c[i] ))
        j_y += (jy * (w_ghost[cell] - w_c[i] ))
        j_z += (jz * (w_ghost[cell] - w_c[i] ))
          
      for j in range(periodicn[i][-1]):
        cell = periodicn[i][j]
        center = centerc[cell][0:3]
        jx = center[0] + shift[cell][0] - centerc[i][0]
        jy = center[1] + shift[cell][1] - centerc[i][1]
        jz = center[2] + shift[cell][2] - centerc[i][2]
        
        i_xx += jx*jx
        i_yy += jy*jy
        i_zz += jz*jz
        i_xy += (jx * jy)
        i_xz += (jx * jz)
        i_yz += (jy * jz)

        j_x += (jx * (w_c[cell] - w_c[i] ))
        j_y += (jy * (w_c[cell] - w_c[i] ))
        j_z += (jz * (w_c[cell] - w_c[i] ))
    
      #if nbproc > 1:      
      for j in range(halonid[i][-1]):
        cell = halonid[i][j]
        
        jx = centerh[cell][0] - centerc[i][0]
        jy = centerh[cell][1] - centerc[i][1]
        jz = centerh[cell][2] - centerc[i][2]
        
        i_xx += jx*jx
        i_yy += jy*jy
        i_zz += jz*jz
        i_xy += (jx * jy)
        i_xz += (jx * jz)
        i_yz += (jy * jz)

        j_x += (jx * (w_halo[cell] - w_c[i] ))
        j_y += (jy * (w_halo[cell] - w_c[i] ))
        j_z += (jz * (w_halo[cell] - w_c[i] ))
          
      for j in range(haloghostnid[i][-1]):
          #-3 the index of global face
        cell = haloghostnid[i][j]
        center = halocenterg[cell]
        
        jx = center[0] - centerc[i][0]
        jy = center[1] - centerc[i][1]
        jz = center[2] - centerc[i][2]
        
        i_xx += jx*jx
        i_yy += jy*jy
        i_zz += jz*jz
        i_xy += (jx * jy)
        i_xz += (jx * jz)
        i_yz += (jy * jz)

        j_x += (jx * (w_haloghost[cell] - w_c[i] ))
        j_y += (jy * (w_haloghost[cell] - w_c[i] ))
        j_z += (jz * (w_haloghost[cell] - w_c[i] ))
      
      dia = i_xx*i_yy*i_zz + 2.*i_xy*i_xz*i_yz - i_xx*i_yz**2 - i_yy*i_xz**2 - i_zz*i_xy**2

      w_x[i] = ((i_yy*i_zz - i_yz**2)*j_x   + (i_xz*i_yz - i_xy*i_zz)*j_y + (i_xy*i_yz - i_xz*i_yy)*j_z) / dia
      w_y[i] = ((i_xz*i_yz - i_xy*i_zz)*j_x + (i_xx*i_zz - i_xz**2)*j_y   + (i_xy*i_xz - i_yz*i_xx)*j_z) / dia
      w_z[i] = ((i_xy*i_yz - i_xz*i_yy)*j_x + (i_xy*i_xz - i_yz*i_xx)*j_y + (i_xx*i_yy - i_xy**2)*j_z  ) / dia
     
  kernel_cell_gradient_3d = GPU_Backend.compile_kernel(kernel_cell_gradient_3d)
  
  def result(*args):
    VarClass.debug(kernel_cell_gradient_3d, args)
    args = [VarClass.to_device(arg) for arg in args]
    size = len(args[0]) #w_c
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_cell_gradient_3d[nb_blocks, nb_threads, GPU_Backend.stream](*args)
    GPU_Backend.stream.synchronize()

  return result


def get_kernel_face_gradient_3d():
  
  def kernel_face_gradient_3d(w_c:'float[:]',
    w_ghost:'float[:]',
    w_halo:'float[:]',
    w_node:'float[:]',
    cellidf:'int32[:,:]',
    nodeidf:'int32[:,:]',
    centergf:'float[:,:]',
    halofid:'int32[:]',
    centerc:'float[:,:]',
    centerh:'float[:,:]',
    vertexn:'float[:,:]',
    airDiamond:'float[:]',
    normalf:'float[:,:]',
    f_1:'float[:,:]',
    f_2:'float[:,:]',
    f_3:'float[:,:]',
    f_4:'float[:,:]',
    shift:'float[:,:]',
    wx_face:'float[:]',
    wy_face:'float[:]',
    wz_face:'float[:]',
    innerfaces:'uint32[:]',
    halofaces:'uint32[:]',
    dirichletfaces:'uint32[:]',
    neumann:'uint32[:]',
    periodicfaces:'uint32[:]'
    ):
    
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    for idx in range(start, len(innerfaces), stride):
      i = innerfaces[idx]

      c_left = cellidf[i][0]
      c_right = cellidf[i][1]
  
      i_1 = nodeidf[i][0]
      i_2 = nodeidf[i][1]   
      i_3 = nodeidf[i][2] 
      i_4 = i_3
      if nodeidf[i][-1] == 4:
        i_4 = nodeidf[i][3]
      
      V_A = w_node[i_1]
      V_B = w_node[i_2]
      V_C = w_node[i_3]
      V_D = w_node[i_4]
      
      V_L = w_c[c_left]
      V_R = w_c[c_right]

      wx_face[i] = (f_1[i][0]*(V_A - V_C) + f_2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
      wy_face[i] = (f_1[i][1]*(V_A - V_C) + f_2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
      wz_face[i] = (f_1[i][2]*(V_A - V_C) + f_2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]
  
    for idx in range(start, len(periodicfaces), stride):
      i = periodicfaces[idx]

      c_left = cellidf[i][0]
      c_right = cellidf[i][1]
  
      i_1 = nodeidf[i][0]
      i_2 = nodeidf[i][1]   
      i_3 = nodeidf[i][2] 
      i_4 = i_3
      if nodeidf[i][-1] == 4:
        i_4 = nodeidf[i][3]
      
      V_A = w_node[i_1]
      V_B = w_node[i_2]
      V_C = w_node[i_3]
      V_D = w_node[i_4]
      
      V_L = w_c[c_left]
      V_R = w_c[c_right]

      wx_face[i] = (f_1[i][0]*(V_A - V_C) + f_2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
      wy_face[i] = (f_1[i][1]*(V_A - V_C) + f_2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
      wz_face[i] = (f_1[i][2]*(V_A - V_C) + f_2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]
  
    for idx in range(start, len(halofaces), stride):
      i = halofaces[idx]

      c_left = cellidf[i][0]
      c_right = halofid[i]
      
      i_1 = nodeidf[i][0]
      i_2 = nodeidf[i][1]   
      i_3 = nodeidf[i][2] 
      i_4 = i_3
      if nodeidf[i][-1] == 4:
        i_4 = nodeidf[i][3]
      
      V_A = w_node[i_1]
      V_B = w_node[i_2]
      V_C = w_node[i_3]
      V_D = w_node[i_4]
      
      V_L = w_c[c_left]
      V_R = w_halo[c_right]
      
      wx_face[i] = (f_1[i][0]*(V_A - V_C) + f_2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
      wy_face[i] = (f_1[i][1]*(V_A - V_C) + f_2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
      wz_face[i] = (f_1[i][2]*(V_A - V_C) + f_2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]
          
    for idx in range(start, len(dirichletfaces), stride):
      i = dirichletfaces[idx]

      c_left = cellidf[i][0]
      c_right = i
  
      i_1 = nodeidf[i][0]
      i_2 = nodeidf[i][1]   
      i_3 = nodeidf[i][2] 
      i_4 = i_3
      if nodeidf[i][-1] == 4:
        i_4 = nodeidf[i][3]
      
      V_A = w_node[i_1]
      V_B = w_node[i_2]
      V_C = w_node[i_3]
      V_D = w_node[i_4]
      
      V_L = w_c[c_left]
      V_R = w_ghost[c_right]
      
      wx_face[i] = (f_1[i][0]*(V_A - V_C) + f_2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
      wy_face[i] = (f_1[i][1]*(V_A - V_C) + f_2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
      wz_face[i] = (f_1[i][2]*(V_A - V_C) + f_2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]
        
    for idx in range(start, len(neumann), stride):
      i = neumann[idx]

      c_left = cellidf[i][0]
      c_right = i
  
      i_1 = nodeidf[i][0]
      i_2 = nodeidf[i][1]   
      i_3 = nodeidf[i][2] 
      i_4 = i_3
      if nodeidf[i][-1] == 4:
          i_4 = nodeidf[i][3]
      
      V_A = w_node[i_1]
      V_B = w_node[i_2]
      V_C = w_node[i_3]
      V_D = w_node[i_4]
      
      V_L = w_c[c_left]
      V_R = w_ghost[c_right]
      
      wx_face[i] = (f_1[i][0]*(V_A - V_C) + f_2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
      wy_face[i] = (f_1[i][1]*(V_A - V_C) + f_2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
      wz_face[i] = (f_1[i][2]*(V_A - V_C) + f_2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]

  kernel_face_gradient_3d = GPU_Backend.compile_kernel(kernel_face_gradient_3d)
  
  def result(*args):
    VarClass.debug(kernel_face_gradient_3d, args)
    args = [VarClass.to_device(arg) for arg in args]
    # innerfaces halofaces dirichletfaces neumann periodicfaces
    size = max(len(args[20]), len(args[21]), len(args[22]), len(args[23]), len(args[24]))
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_face_gradient_3d[nb_blocks, nb_threads, GPU_Backend.stream](*args)
    GPU_Backend.stream.synchronize()

  return result


def get_kernel_centertovertex_3d():
  
  def kernel_centertovertex_3d(
    w_c:'float[:]',
    w_ghost:'float[:]',
    w_halo:'float[:]',
    w_haloghost:'float[:]',
    centerc:'float[:,:]',
    centerh:'float[:,:]',
    cellid:'int32[:,:]',
    ghostid:'int32[:,:]',
    haloghostid:'int32[:,:]',
    periodicid:'int32[:,:]',
    haloid:'int32[:,:]',
    vertexn:'float[:,:]',
    centergf:'float[:,:]',
    halocenterg:'float[:,:]',
    R_x:'float[:]',
    R_y:'float[:]',
    R_z:'float[:]',
    lambda_x:'float[:]',lambda_y:'float[:]',
    lambda_z:'float[:]',
    number:'uint32[:]',
    shift:'float[:,:]',
    w_n:'float[:]'
    ):

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    #w_n[:] = 0.
    # center = zeros(3)
    
    for i in range(start, len(vertexn), stride):
      w_n[i] = 0.

      for j in range(cellid[i][-1]):
        cell = cellid[i][j]
        center = centerc[cell][:]
        
        xdiff = center[0] - vertexn[i][0]
        ydiff = center[1] - vertexn[i][1]
        zdiff = center[2] - vertexn[i][2]
        
        alpha = (1. + lambda_x[i]*xdiff + \
                  lambda_y[i]*ydiff + lambda_z[i]*zdiff)/ (number[i] + lambda_x[i]*R_x[i] + \
                                                          lambda_y[i]*R_y[i] + lambda_z[i]*R_z[i])
        w_n[i]  += alpha * w_c[cell]
        
      for j in range(ghostid[i][-1]):
        cell = ghostid[i][j]
        center = centergf[cell][0:3]
        
        xdiff = center[0] - vertexn[i][0]
        ydiff = center[1] - vertexn[i][1]
        zdiff = center[2] - vertexn[i][2]
        
        alpha = (1. + lambda_x[i]*xdiff + \
                  lambda_y[i]*ydiff + lambda_z[i]*zdiff)/ (number[i] + lambda_x[i]*R_x[i] + \
                                                          lambda_y[i]*R_y[i] + lambda_z[i]*R_z[i])
        
        w_n[i]  += alpha * w_ghost[cell]
      
      for j in range(haloghostid[i][-1]):
        cell = haloghostid[i][j]
        center = halocenterg[cell]
              
        xdiff = center[0] - vertexn[i][0]
        ydiff = center[1] - vertexn[i][1]
        zdiff = center[2] - vertexn[i][2]
        
        alpha = (1. + lambda_x[i]*xdiff + \
                  lambda_y[i]*ydiff + lambda_z[i]*zdiff)/ (number[i] + lambda_x[i]*R_x[i] + \
                                                          lambda_y[i]*R_y[i] + lambda_z[i]*R_z[i])
                                                          
        w_n[i]  += alpha * w_haloghost[cell]   
        
      for j in range(haloid[i][-1]):
        cell = haloid[i][j]
        center = centerh[cell][0:3]
      
        xdiff = center[0] - vertexn[i][0]
        ydiff = center[1] - vertexn[i][1]
        zdiff = center[2] - vertexn[i][2]
        
        alpha = (1. + lambda_x[i]*xdiff + \
                  lambda_y[i]*ydiff + lambda_z[i]*zdiff)/ (number[i] + lambda_x[i]*R_x[i] + \
                                                          lambda_y[i]*R_y[i] + lambda_z[i]*R_z[i])
                                                          
        w_n[i]  += alpha * w_halo[cell]    
                
      if vertexn[i][3] == 11 or vertexn[i][3] == 22 :
        for j in range(periodicid[i][-1]):
          cell = periodicid[i][j]
          center = centerc[cell][0:3] 
          
          xdiff = center[0] + shift[cell][0] - vertexn[i][0]
          ydiff = center[1] - vertexn[i][1]
          zdiff = center[2] - vertexn[i][2]
          
          alpha = (1. + lambda_x[i]*xdiff + \
                        lambda_y[i]*ydiff + lambda_z[i]*zdiff)/ (number[i] + lambda_x[i]*R_x[i] + \
                                                                lambda_y[i]*R_y[i] + lambda_z[i]*R_z[i])
          
          w_n[i]  += alpha * w_c[cell]
                  
      elif vertexn[i][3] == 33 or vertexn[i][3] == 44:
        for j in range(periodicid[i][-1]):
          cell = periodicid[i][j]
          center = centerc[cell][0:3] 
          
          xdiff = center[0] - vertexn[i][0]
          ydiff = center[1] + shift[cell][1] - vertexn[i][1]
          zdiff = center[2] - vertexn[i][2]
          
          alpha = (1. + lambda_x[i]*xdiff + \
                        lambda_y[i]*ydiff + lambda_z[i]*zdiff)/ (number[i] + lambda_x[i]*R_x[i] + \
                                                                lambda_y[i]*R_y[i] + lambda_z[i]*R_z[i])
          w_n[i]  += alpha * w_c[cell]
              
      elif vertexn[i][3] == 55 or vertexn[i][3] == 66:
        for j in range(periodicid[i][-1]):
          cell = periodicid[i][j]
          center = centerc[cell][0:3] 
          
          xdiff = center[0] - vertexn[i][0]
          ydiff = center[1] - vertexn[i][1]
          zdiff = center[2] + shift[cell][2] - vertexn[i][2]
          
          alpha = (1. + lambda_x[i]*xdiff + \
                        lambda_y[i]*ydiff + lambda_z[i]*zdiff)/ (number[i] + lambda_x[i]*R_x[i] + \
                                                                lambda_y[i]*R_y[i] + lambda_z[i]*R_z[i])
          w_n[i]  += alpha * w_c[cell]
    
  kernel_centertovertex_3d = GPU_Backend.compile_kernel(kernel_centertovertex_3d)
  
  def result(*args):
    VarClass.debug(kernel_centertovertex_3d, args)
    args = [VarClass.to_device(arg) for arg in args]
    # vertexn
    size = len(args[11])
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_centertovertex_3d[nb_blocks, nb_threads, GPU_Backend.stream](*args)
    GPU_Backend.stream.synchronize()

  return result


def get_kernel_barthlimiter_3d():
  
  def kernel_barthlimiter_3d(
    w_c:'float[:]',
    w_ghost:'float[:]',
    w_halo:'float[:]',
    w_x:'float[:]',
    w_y:'float[:]',
    w_z:'float[:]',
    psi:'float[:]',
    cellid:'int32[:,:]',
    faceid:'int32[:,:]',
    namef:'uint32[:]',
    halofid:'int32[:]',
    centerc:'float[:,:]',
    centerf:'float[:,:]'
    ):
    
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    # psi[:] = 1.
    
    for i in range(start, len(w_c), stride):
      w_max = w_c[i]
      w_min = w_c[i]

      psi[i] = 1.
      
      for j in range(faceid[i][-1]):
        face = faceid[i][j]
        if namef[face] == 0 or namef[face] > 10:
          w_max = max(w_max, w_c[cellid[face][0]], w_c[cellid[face][1]])
          w_min = min(w_min, w_c[cellid[face][0]], w_c[cellid[face][1]])
        elif namef[face] == 10:
          w_max = max(w_max,  w_c[cellid[face][0]], w_halo[halofid[face]])
          w_min = min(w_min,  w_c[cellid[face][0]], w_halo[halofid[face]])
        else:
          w_max = max(w_max,  w_c[cellid[face][0]], w_ghost[face])
          w_min = min(w_min,  w_c[cellid[face][0]], w_ghost[face])
        
      for j in range(faceid[i][-1]):
        face = faceid[i][j]

        r_xyz1 = centerf[face][0] - centerc[i][0]
        r_xyz2 = centerf[face][1] - centerc[i][1]
        r_xyz3 = centerf[face][2] - centerc[i][2]
        
        delta2 = w_x[i] * r_xyz1 + w_y[i] * r_xyz2 + w_z[i] * r_xyz3
        
        #TODO choice of epsilon
        if math.fabs(delta2) < 1e-10:
          psi_ij = 1.
        else:
          if delta2 > 0.:
            value = (w_max - w_c[i]) / delta2
            psi_ij = min(1., value)
          if delta2 < 0.:
            value = (w_min - w_c[i]) / delta2
            psi_ij = min(1., value)
        psi[i] = min(psi[i], psi_ij)
   
  kernel_barthlimiter_3d = GPU_Backend.compile_kernel(kernel_barthlimiter_3d)
  
  def result(*args):
    VarClass.debug(kernel_barthlimiter_3d, args)
    args = [VarClass.to_device(arg) for arg in args]
    # w_c
    size = len(args[0])
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_barthlimiter_3d[nb_blocks, nb_threads, GPU_Backend.stream](*args)
    GPU_Backend.stream.synchronize()

  return result


def get_kernel_compute_3dmatrix_size():
  
  d_cmpt = cuda.device_array(shape=(1), dtype='uint64')
  search_element = GPU_Backend.compile_kernel(device_search_element, device = True)

  def kernel_compute_3dmatrix_size(
    nodeidf:'int32[:,:]',
    halofid:'int32[:]',
    cellnid:'int32[:,:]',
    halonid:'int32[:,:]',
    periodicnid:'int32[:,:]',
    centergn:'float[:,:,:]',
    halocentergn:'float[:,:,:]',
    oldnamen:'uint32[:]',
    BCdirichlet:'uint32[:]',
    matrixinnerfaces:'uint32[:]',
    halofaces:'uint32[:]',
    dirichletfaces:'uint32[:]',
    s_cmpt:'uint64[:]'
    ):                                                                                                                                                                       
    
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    # cmpt = 0
    # nodes = zeros(4, dtype=int32)
    nodes = cuda.local.array(4, nodeidf.dtype)

    for idx in range(start, len(matrixinnerfaces), stride):
      i = matrixinnerfaces[idx]

      nodes[0] = nodeidf[i][0]
      nodes[1] = nodeidf[i][1]
      nodes[2] = nodeidf[i][2]
      nodes[3] = nodeidf[i][2]
      if nodeidf[i][-1] == 4:
        nodes[3] = nodeidf[i][3]
      
      cuda.atomic.add(s_cmpt, 0, 1)
      #cmpt = cmpt + 1
          
      for nod in nodes:
        if search_element(BCdirichlet, oldnamen[nod]) == 0: 
          for j in range(cellnid[nod][-1]):
            cuda.atomic.add(s_cmpt, 0, 1)
            #right cell-----------------------------------                                                                                              
            cuda.atomic.add(s_cmpt, 0, 1)
          
          for j in range(len(centergn[nod])):
            if centergn[nod][j][-1] != -1:
              cuda.atomic.add(s_cmpt, 0, 1)
              #right cell-----------------------------------                                                                                              
              cuda.atomic.add(s_cmpt, 0, 1)
                  
          for j in range(len(halocentergn[nod])):
            if halocentergn[nod][j][-1] != -1:
              cuda.atomic.add(s_cmpt, 0, 1)
              #right cell-----------------------------------                                                                                              
              cuda.atomic.add(s_cmpt, 0, 1)
              
          for j in range(periodicnid[nod][-1]):
            cuda.atomic.add(s_cmpt, 0, 1)
            #right cell-----------------------------------                                                                                              
            cuda.atomic.add(s_cmpt, 0, 1)
          
          for j in range(halonid[nod][-1]):
            cuda.atomic.add(s_cmpt, 0, 1)
            #right cell-----------------------------------                                                                                              
            cuda.atomic.add(s_cmpt, 0, 1)
      
      cuda.atomic.add(s_cmpt, 0, 1)
      # right cell------------------------------------------------------
      cuda.atomic.add(s_cmpt, 0, 1)
      cuda.atomic.add(s_cmpt, 0, 1)
            
    for idx in range(start, len(halofaces), stride):
      i = halofaces[idx]

      nodes[0] = nodeidf[i][0]
      nodes[1] = nodeidf[i][1]
      nodes[2] = nodeidf[i][2]
      nodes[3] = nodeidf[i][2]
      if nodeidf[i][-1] == 4:
        nodes[3] = nodeidf[i][3]
      
      cuda.atomic.add(s_cmpt, 0, 1)
      cuda.atomic.add(s_cmpt, 0, 1)
      
      for nod in nodes:
        if search_element(BCdirichlet, oldnamen[nod]) == 0: 
          for j in range(cellnid[nod][-1]):
            cuda.atomic.add(s_cmpt, 0, 1)
              
          for j in range(len(centergn[nod])):
            if centergn[nod][j][-1] != -1:
              cuda.atomic.add(s_cmpt, 0, 1)
                  
          for j in range(len(halocentergn[nod])):
            if halocentergn[nod][j][-1] != -1:
              cuda.atomic.add(s_cmpt, 0, 1)

          for j in range(halonid[nod][-1]):
            cuda.atomic.add(s_cmpt, 0, 1)
                    
    for idx in range(start, len(dirichletfaces), stride):
      i = dirichletfaces[idx]

      cuda.atomic.add(s_cmpt, 0, 1)
      cuda.atomic.add(s_cmpt, 0, 1)
            



  kernel_compute_3dmatrix_size = GPU_Backend.compile_kernel(kernel_compute_3dmatrix_size)
  kernel_assign_int64 = cuda.jit('void(uint64[:], uint64)')(kernel_assign)
  
  def result(*args):
    VarClass.debug(kernel_compute_3dmatrix_size, args)
    args = [VarClass.to_device(arg) for arg in args]
    kernel_assign_int64[1, 1, GPU_Backend.stream](d_cmpt, 0) #cmpt
    GPU_Backend.stream.synchronize()
    # matrixinnerfaces halofaces dirichletfaces
    size = max(len(args[9]), len(args[10]), len(args[11]))
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_compute_3dmatrix_size[nb_blocks, nb_threads, GPU_Backend.stream](*args, d_cmpt)
    # GPU_Backend.stream.synchronize()
    return d_cmpt.copy_to_host(stream=GPU_Backend.stream)[0]

  return result


def get_kernel_get_triplet_3d():
  
  d_cmpt = cuda.device_array(shape=(1), dtype='uint64')
  search_element = GPU_Backend.compile_kernel(device_search_element, device = True)

  def kernel_get_triplet_3d(
    cellfid:'int32[:,:]',
    nodeidf:'int32[:,:]',
    vertexn:'float[:,:]',
    halofid:'int32[:]',
    haloext:'int32[:,:]',
    oldnamen:'uint32[:]',
    volume:'float[:]',
    cellnid:'int32[:,:]',
    centerc:'float[:,:]',
    centerh:'float[:,:]',
    halonid:'int32[:,:]',
    periodicnid:'int32[:,:]',
    centergn:'float[:,:,:]',
    halocentergn:'float[:,:,:]',
    airDiamond:'float[:]',
    lambda_x:'float[:]',
    lambda_y:'float[:]',
    lambda_z:'float[:]',
    number:'uint32[:]',
    R_x:'float[:]',
    R_y:'float[:]',
    R_z:'float[:]',
    param1:'float[:]',
    param2:'float[:]',
    param3:'float[:]',
    param4:'float[:]',
    shift:'float[:,:]',
    nbelements:'int32',
    loctoglob:'int32[:]',
    BCdirichlet:'uint32[:]',
    a_loc:'float[:]',
    irn_loc:'int32[:]',
    jcn_loc:'int32[:]',
    matrixinnerfaces:'uint32[:]',
    halofaces:'uint32[:]',
    dirichletfaces:'uint32[:]',
    s_cmpt: 'uint64[:]'
    ):

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    parameters = cuda.local.array(4, param1.dtype)
    nodes = cuda.local.array(4, nodeidf.dtype)

    # cmpt = 0
    
    for idx in range(start, len(matrixinnerfaces), stride):
      i = matrixinnerfaces[idx]

      c_left = cellfid[i][0]
      c_leftglob  = loctoglob[c_left]
      
      nodes[0] = nodeidf[i][0]
      nodes[1] = nodeidf[i][1]
      nodes[2] = nodeidf[i][2]
      nodes[3] = nodeidf[i][2]
      if nodeidf[i][-1] == 4:
        nodes[3] = nodeidf[i][3]

      parameters[0] = param1[i]
      parameters[1] = param2[i]
      parameters[2] = -1. * param1[i]
      parameters[3] = -1. * param2[i]
      
      c_right = cellfid[i][1]
      c_rightglob = loctoglob[c_right]
      
      cmpt = cuda.atomic.add(s_cmpt, 0, 1)
      irn_loc[cmpt] = c_leftglob
      jcn_loc[cmpt] = c_leftglob
      value = -1 * param3[i] / volume[c_left]
      a_loc[cmpt] = value
      
      cmptparam = 0
      for nod in nodes:
        if search_element(BCdirichlet, oldnamen[nod]) == 0: 
          for j in range(cellnid[nod][-1]):
            center = centerc[cellnid[nod][j]]
            xdiff = center[0] - vertexn[nod][0]
            ydiff = center[1] - vertexn[nod][1]
            zdiff = center[2] - vertexn[nod][2]
            alpha = (1. + lambda_x[nod]*xdiff + \
                      lambda_y[nod]*ydiff + lambda_z[nod]*zdiff)/ (number[nod] + lambda_x[nod]*R_x[nod] + \
                                                                  lambda_y[nod]*R_y[nod] + lambda_z[nod]*R_z[nod])
            value = alpha / volume[c_left] * parameters[cmptparam]
            cmpt = cuda.atomic.add(s_cmpt, 0, 1)
            irn_loc[cmpt] = c_leftglob
            jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
            a_loc[cmpt] = value
            
            #right cell-----------------------------------                                                                                              
            value = -1. * alpha / volume[c_right] * parameters[cmptparam]
            cmpt = cuda.atomic.add(s_cmpt, 0, 1)
            irn_loc[cmpt] = c_rightglob
            jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
            a_loc[cmpt] = value
              
          for j in range(halonid[nod][-1]):
            center = centerh[halonid[nod][j]]
            xdiff = center[0] - vertexn[nod][0]
            ydiff = center[1] - vertexn[nod][1]
            zdiff = center[2] - vertexn[nod][2]
            alpha = (1. + lambda_x[nod]*xdiff + \
                      lambda_y[nod]*ydiff + lambda_z[nod]*zdiff)/ (number[nod] + lambda_x[nod]*R_x[nod] + \
                                                                  lambda_y[nod]*R_y[nod] + lambda_z[nod]*R_z[nod])
            value = alpha / volume[c_left] * parameters[cmptparam]
            
            cmpt = cuda.atomic.add(s_cmpt, 0, 1)
            irn_loc[cmpt] = c_leftglob
            jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
            a_loc[cmpt] = value

            #right cell-----------------------------------                                                                                              
            value = -1. * alpha / volume[c_right] * parameters[cmptparam]
            cmpt = cuda.atomic.add(s_cmpt, 0, 1)
            irn_loc[cmpt] = c_rightglob
            jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
            a_loc[cmpt] = value
                
          for j in range(periodicnid[nod][-1]):
            if vertexn[nod][3] == 11 or vertexn[nod][3] == 22:
              center[0] = centerc[periodicnid[nod][j]][0]  + shift[periodicnid[nod][j]][0]
              center[1] = centerc[periodicnid[nod][j]][1]  
              center[2] = centerc[periodicnid[nod][j]][2]
            if vertexn[nod][3] == 33 or vertexn[nod][3] == 44:
              center[0] = centerc[periodicnid[nod][j]][0]  
              center[1] = centerc[periodicnid[nod][j]][1]  + shift[periodicnid[nod][j]][1]
              center[2] = centerc[periodicnid[nod][j]][2]
            if vertexn[nod][3] == 55 or vertexn[nod][3] == 66:
              center[0] = centerc[periodicnid[nod][j]][0]  
              center[1] = centerc[periodicnid[nod][j]][1]  
              center[2] = centerc[periodicnid[nod][j]][2] + shift[periodicnid[nod][j]][2]
            
            xdiff = center[0] - vertexn[nod][0]
            ydiff = center[1] - vertexn[nod][1]
            zdiff = center[2] - vertexn[nod][2]
            alpha = (1. + lambda_x[nod]*xdiff + \
                      lambda_y[nod]*ydiff + lambda_z[nod]*zdiff)/ (number[nod] + lambda_x[nod]*R_x[nod] + \
                                                                  lambda_y[nod]*R_y[nod] + lambda_z[nod]*R_z[nod])
            value =  alpha / volume[c_left] * parameters[cmptparam]
            cmpt = cuda.atomic.add(s_cmpt, 0, 1)
            irn_loc[cmpt] = c_leftglob
            jcn_loc[cmpt] = loctoglob[periodicnid[nod][j]]
            a_loc[cmpt] = value
            #right cell-----------------------------------                                                                                              
            value =  -1. * alpha / volume[c_right] * parameters[cmptparam]
            cmpt = cuda.atomic.add(s_cmpt, 0, 1)
            irn_loc[cmpt] = c_rightglob
            jcn_loc[cmpt] = loctoglob[periodicnid[nod][j]]
            a_loc[cmpt] = value
              
          for j in range(len(centergn[nod])):
            if centergn[nod][j][-1] != -1:
              center = centergn[nod][j][0:3]
              xdiff = center[0] - vertexn[nod][0]
              ydiff = center[1] - vertexn[nod][1]
              zdiff = center[2] - vertexn[nod][2]
              alpha = (1. + lambda_x[nod]*xdiff + \
                        lambda_y[nod]*ydiff + lambda_z[nod]*zdiff)/ (number[nod] + lambda_x[nod]*R_x[nod] + \
                                                                    lambda_y[nod]*R_y[nod] + lambda_z[nod]*R_z[nod])
              value = alpha / volume[c_left] * parameters[cmptparam]
              
              index = np.int32(centergn[nod][j][3])
              cmpt = cuda.atomic.add(s_cmpt, 0, 1)
              irn_loc[cmpt] = c_leftglob
              jcn_loc[cmpt] = loctoglob[index]
              a_loc[cmpt] = value

              #right cell-----------------------------------                                                                                              
              value = -1. * alpha / volume[c_right] * parameters[cmptparam]
              cmpt = cuda.atomic.add(s_cmpt, 0, 1)
              irn_loc[cmpt] = c_rightglob
              jcn_loc[cmpt] = loctoglob[index]
              a_loc[cmpt] = value
              
          for j in range(len(halocentergn[nod])):
            if halocentergn[nod][j][-1] != -1:
              center = halocentergn[nod][j][0:3]
              xdiff = center[0] - vertexn[nod][0]
              ydiff = center[1] - vertexn[nod][1]
              zdiff = center[2] - vertexn[nod][2]
              alpha = (1. + lambda_x[nod]*xdiff + \
                        lambda_y[nod]*ydiff + lambda_z[nod]*zdiff)/ (number[nod] + lambda_x[nod]*R_x[nod] + \
                                                                    lambda_y[nod]*R_y[nod] + lambda_z[nod]*R_z[nod])
              value = alpha / volume[c_left] * parameters[cmptparam]
              index = np.int32(halocentergn[nod][j][3])
              cmpt = cuda.atomic.add(s_cmpt, 0, 1)
              irn_loc[cmpt] = c_leftglob
              jcn_loc[cmpt] = haloext[index][0]
              a_loc[cmpt] = value

              #right cell-----------------------------------                                                                                              
              value = -1. * alpha / volume[c_right] * parameters[cmptparam]
              cmpt = cuda.atomic.add(s_cmpt, 0, 1)
              irn_loc[cmpt] = c_rightglob
              jcn_loc[cmpt] = haloext[index][0]
              a_loc[cmpt] = value

        
        cmptparam = cmptparam +1
        
      cmpt = cuda.atomic.add(s_cmpt, 0, 1)
      irn_loc[cmpt] = c_leftglob
      jcn_loc[cmpt] = c_rightglob
      value = param3[i] / volume[c_left]
      a_loc[cmpt] = value
      

      # # right cell------------------------------------------------------
      cmpt = cuda.atomic.add(s_cmpt, 0, 1)
      irn_loc[cmpt] = c_rightglob
      jcn_loc[cmpt] = c_leftglob
      value = param3[i] / volume[c_right]
      a_loc[cmpt] = value
      
      cmpt = cuda.atomic.add(s_cmpt, 0, 1)
      irn_loc[cmpt] = c_rightglob
      jcn_loc[cmpt] = c_rightglob
      value = -1. * param3[i] / volume[c_right]
      a_loc[cmpt] = value
       
    for idx in range(start, len(halofaces), stride):
      i = halofaces[idx]
      
      c_left = cellfid[i][0]
      c_leftglob  = loctoglob[c_left]
      
      nodes[0] = nodeidf[i][0]
      nodes[1] = nodeidf[i][1]
      nodes[2] = nodeidf[i][2]
      nodes[3]  = nodeidf[i][2]
      if nodeidf[i][-1] == 4:
        nodes[3] = nodeidf[i][3]

      parameters[0] = param1[i]
      parameters[1] = param2[i]
      parameters[2] = -1. * param1[i]
      parameters[3] = -1. * param2[i]
      
      c_rightglob = haloext[halofid[i]][0]
      c_right     = halofid[i]
      
      cmptparam = 0
      for nod in nodes:
        if search_element(BCdirichlet, oldnamen[nod]) == 0: 
          for j in range(cellnid[nod][-1]):
            center = centerc[cellnid[nod][j]]
            xdiff = center[0] - vertexn[nod][0]
            ydiff = center[1] - vertexn[nod][1]
            zdiff = center[2] - vertexn[nod][2]
            alpha = (1. + lambda_x[nod]*xdiff + \
                      lambda_y[nod]*ydiff + lambda_z[nod]*zdiff)/ (number[nod] + lambda_x[nod]*R_x[nod] + \
                                                                  lambda_y[nod]*R_y[nod] + lambda_z[nod]*R_z[nod])
            value = alpha / volume[c_left] * parameters[cmptparam]
            cmpt = cuda.atomic.add(s_cmpt, 0, 1)
            irn_loc[cmpt] = c_leftglob
            jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
            a_loc[cmpt] = value
          
          for j in range(halonid[nod][-1]):
            center = centerh[halonid[nod][j]]
            xdiff = center[0] - vertexn[nod][0]
            ydiff = center[1] - vertexn[nod][1]
            zdiff = center[2] - vertexn[nod][2]
            alpha = (1. + lambda_x[nod]*xdiff + \
                      lambda_y[nod]*ydiff + lambda_z[nod]*zdiff)/ (number[nod] + lambda_x[nod]*R_x[nod] + \
                                                                  lambda_y[nod]*R_y[nod] + lambda_z[nod]*R_z[nod])
            value = alpha / volume[c_left] * parameters[cmptparam]
            cmpt = cuda.atomic.add(s_cmpt, 0, 1)
            irn_loc[cmpt] = c_leftglob
            jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
            a_loc[cmpt] = value
              
          for j in range(len(centergn[nod])):
            if centergn[nod][j][-1] != -1:
              center = centergn[nod][j][0:3]
              xdiff = center[0] - vertexn[nod][0]
              ydiff = center[1] - vertexn[nod][1]
              zdiff = center[2] - vertexn[nod][2]
              alpha = (1. + lambda_x[nod]*xdiff + \
                        lambda_y[nod]*ydiff + lambda_z[nod]*zdiff)/ (number[nod] + lambda_x[nod]*R_x[nod] + \
                                                                    lambda_y[nod]*R_y[nod] + lambda_z[nod]*R_z[nod])
              value = alpha / volume[c_left] * parameters[cmptparam]
              
              index = np.int32(centergn[nod][j][3])
              cmpt = cuda.atomic.add(s_cmpt, 0, 1)
              irn_loc[cmpt] = c_leftglob
              jcn_loc[cmpt] = loctoglob[index]
              a_loc[cmpt] = value
              
          for j in range(len(halocentergn[nod])):
            if halocentergn[nod][j][-1] != -1:
              center = halocentergn[nod][j][0:3]
              xdiff = center[0] - vertexn[nod][0]
              ydiff = center[1] - vertexn[nod][1]
              zdiff = center[2] - vertexn[nod][2]
              alpha = (1. + lambda_x[nod]*xdiff + \
                        lambda_y[nod]*ydiff + lambda_z[nod]*zdiff)/ (number[nod] + lambda_x[nod]*R_x[nod] + \
                                                                    lambda_y[nod]*R_y[nod] + lambda_z[nod]*R_z[nod])
              value = alpha / volume[c_left] * parameters[cmptparam]
              index = np.int32(halocentergn[nod][j][3])
              cmpt = cuda.atomic.add(s_cmpt, 0, 1)
              irn_loc[cmpt] = c_leftglob
              jcn_loc[cmpt] = haloext[index][0]
              a_loc[cmpt] = value

        cmptparam = cmptparam +1
      
      cmpt = cuda.atomic.add(s_cmpt, 0, 1)
      irn_loc[cmpt] = c_leftglob
      jcn_loc[cmpt] = c_leftglob
      value = -1 * param3[i] / volume[c_left]
      a_loc[cmpt] = value
      
      cmpt = cuda.atomic.add(s_cmpt, 0, 1)
      irn_loc[cmpt] = c_leftglob
      jcn_loc[cmpt] = c_rightglob
      value = param3[i] / volume[c_left]
      a_loc[cmpt] = value
            
    for idx in range(start, len(dirichletfaces), stride):
      i = dirichletfaces[idx]

      c_left = cellfid[i][0]
      c_leftglob  = loctoglob[c_left]
      
      parameters[0] = param1[i]
      parameters[1] = param2[i]
      parameters[2] = -1. * param1[i]
      parameters[3] = -1. * param2[i]
      
      cmpt = cuda.atomic.add(s_cmpt, 0, 1)
      irn_loc[cmpt] = c_leftglob
      jcn_loc[cmpt] = c_leftglob
      value = -1 * param3[i] / volume[c_left]
      a_loc[cmpt] = value
      
      cmpt = cuda.atomic.add(s_cmpt, 0, 1)
      irn_loc[cmpt] = c_leftglob
      jcn_loc[cmpt] = c_leftglob
      value = -1. * param3[i] / volume[c_left]
      a_loc[cmpt] = value
            

  kernel_get_triplet_3d = GPU_Backend.compile_kernel(kernel_get_triplet_3d)
  kernel_assign_int64 = cuda.jit('void(uint64[:], uint64)')(kernel_assign)
  
  def result(*args):
    VarClass.debug(kernel_get_triplet_3d, args)
    args = [VarClass.to_device(arg) for arg in args]
    kernel_assign_int64[1, 1, GPU_Backend.stream](d_cmpt, 0) #cmpt
    GPU_Backend.stream.synchronize()
    # matrixinnerfaces halofaces dirichletfaces
    size = max(len(args[33]), len(args[34]), len(args[35]))
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_get_triplet_3d[nb_blocks, nb_threads, GPU_Backend.stream](*args, d_cmpt)
    GPU_Backend.stream.synchronize()

  return result


def get_kernel_get_rhs_loc_3d():
  
  search_element = GPU_Backend.compile_kernel(device_search_element, device = True)

  def kernel_get_rhs_loc_3d(
    cellfid:'int32[:,:]',
    nodeidf:'int32[:,:]',
    oldname:'uint32[:]',
    volume:'float[:]',
    centergn:'float[:,:,:]',
    loctoglob:'int32[:]',
    param1:'float[:]',
    param2:'float[:]',
    param3:'float[:]',
    param4:'float[:]',
    Pbordnode:'float[:]',
    Pbordface:'float[:]',
    rhs_loc:'float[:]',
    BCdirichlet:'uint32[:]',
    centergf:'float[:,:]',
    matrixinnerfaces:'uint32[:]',
    halofaces:'uint32[:]',
    dirichletfaces:'uint32[:]',
    ):
    
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    # rhs_loc[:] = 0.
    parameters = cuda.local.array(4, param1.dtype)
    nodes = cuda.local.array(4, nodeidf.dtype)

    for idx in range(start, len(matrixinnerfaces), stride):
      i = matrixinnerfaces[idx]
    
      c_left = cellfid[i][0]
      c_right = cellfid[i][1]
      
      nodes[0] = nodeidf[i][0]
      nodes[1] = nodeidf[i][1]
      nodes[2] = nodeidf[i][2]
      nodes[3] = nodeidf[i][2]
      if nodeidf[i][-1] == 4:
        nodes[3] = nodeidf[i][3]

      parameters[0] = param1[i]
      parameters[1] = param2[i]
      parameters[2] = -1. * param1[i]
      parameters[3] = -1. * param2[i]
      
      cmpt = 0
      for nod in nodes:
        if search_element(BCdirichlet, oldname[nod]) == 1: 
          V = Pbordnode[nod]
          value_left = -1. * V * parameters[cmpt] / volume[c_left]

          #? rhs_loc[c_left] += value_left
          cuda.atomic.add(rhs_loc, c_left, value_left)
          
          value_right = V * parameters[cmpt] / volume[c_right]
          #? rhs_loc[c_right] += value_right
          cuda.atomic.add(rhs_loc, c_right, value_right)

        cmpt = cmpt +1

    for idx in range(start, len(halofaces), stride):
      i = halofaces[idx]

      c_left = cellfid[i][0]
      
      nodes[0] = nodeidf[i][0]
      nodes[1] = nodeidf[i][1]
      nodes[2] = nodeidf[i][2]
      nodes[3]   = nodeidf[i][2]
      if nodeidf[i][-1] == 4:
        nodes[3] = nodeidf[i][3]
      
      parameters[0] = param1[i]
      parameters[1] = param2[i]
      parameters[2] = -1. * param1[i]
      parameters[3] = -1. * param2[i]
  
      cmpt = 0
      for nod in nodes:
        if search_element(BCdirichlet, oldname[nod]) == 1:
          V = Pbordnode[nod]
          value_left = -1. * V * parameters[cmpt] / volume[c_left]
          #? rhs_loc[c_left] += value_left
          cuda.atomic.add(rhs_loc, c_left, value_left)
        cmpt = cmpt +1
    
    for idx in range(start, len(dirichletfaces), stride):
      i = dirichletfaces[idx]
      c_left = cellfid[i][0]
      
      nodes[0] = nodeidf[i][0]
      nodes[1] = nodeidf[i][1]
      nodes[2] = nodeidf[i][2]
      nodes[3]   = nodeidf[i][2]
      if nodeidf[i][-1] == 4:
        nodes[3] = nodeidf[i][3]
      
      parameters[0] = param1[i]; parameters[1] = param2[i]
      parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
      
      cmpt = 0
      for nod in nodes:
        if centergn[nod][0][3] != -1:    
          V = Pbordnode[nod]
          value_left = -1. * V * parameters[cmpt] / volume[c_left]
          #? rhs_loc[c_left] += value_left
          cuda.atomic.add(rhs_loc, c_left, value_left)
          
        cmpt +=1
          
      V_K = Pbordface[i]
      value = -2. * param3[i] / volume[c_left] * V_K
      #? rhs_loc[c_left] += value
      cuda.atomic.add(rhs_loc, c_left, value)
          

  kernel_get_rhs_loc_3d = GPU_Backend.compile_kernel(kernel_get_rhs_loc_3d)
  kernel_assign_float = GPU_Backend.compile_kernel(kernel_assign)
  
  def result(*args):
    VarClass.debug(kernel_get_rhs_loc_3d, args)
    args = [VarClass.to_device(arg) for arg in args]
    kernel_assign_float[1, 1, GPU_Backend.stream](args[12], 0.0) #rhs_loc
    GPU_Backend.stream.synchronize()
    # matrixinnerfaces halofaces dirichletfaces
    size = max(len(args[15]), len(args[16]), len(args[17]))
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_get_rhs_loc_3d[nb_blocks, nb_threads, GPU_Backend.stream](*args)
    GPU_Backend.stream.synchronize()

  return result


def get_kernel_get_rhs_glob_3d():
  
  search_element = GPU_Backend.compile_kernel(device_search_element, device = True)

  def kernel_get_rhs_glob_3d(
    cellfid:'int32[:,:]',
    nodeidf:'int32[:,:]',
    oldname:'uint32[:]',
    volume:'float[:]',
    centergn:'float[:,:,:]',
    loctoglob:'int32[:]',
    param1:'float[:]',
    param2:'float[:]',
    param3:'float[:]',
    param4:'float[:]',
    Pbordnode:'float[:]',
    Pbordface:'float[:]',
    rhs:'float[:]',
    BCdirichlet:'uint32[:]',
    centergf:'float[:,:]',
    matrixinnerfaces:'uint32[:]',
    halofaces:'uint32[:]',
    dirichletfaces:'uint32[:]'
    ):                                                                                                                                                                       

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    parameters = cuda.local.array(4, param1.dtype)
    nodes = cuda.local.array(4, nodeidf.dtype)
    
    for idx in range(start, len(matrixinnerfaces), stride):
      i = matrixinnerfaces[idx]

      c_left = cellfid[i][0]
      c_leftglob  = loctoglob[c_left]
      
      nodes[0] = nodeidf[i][0]
      nodes[1] = nodeidf[i][1]
      nodes[2] = nodeidf[i][2]
      nodes[3] = nodeidf[i][2]
      if nodeidf[i][-1] == 4:
        nodes[3] = nodeidf[i][3]
      
      parameters[0] = param1[i]; parameters[1] = param2[i]
      parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
      
      c_right = cellfid[i][1]
      c_rightglob = loctoglob[c_right]
        
      cmpt = 0
      for nod in nodes:
        if search_element(BCdirichlet, oldname[nod]) == 1: 
          V = Pbordnode[nod]
          value_left = -1. * V * parameters[cmpt] / volume[c_left]
          cuda.atomic.add(rhs, c_leftglob, value_left)
          #? rhs[c_leftglob] += value_left
          
          value_right = V * parameters[cmpt] / volume[c_right]
          cuda.atomic.add(rhs, c_rightglob, value_right)
          #? rhs[c_rightglob] += value_right

        cmpt = cmpt +1
    
    for idx in range(start, len(halofaces), stride):
      i = halofaces[idx]

      c_left = cellfid[i][0]
      c_leftglob  = loctoglob[c_left]
      
      nodes[0] = nodeidf[i][0]
      nodes[1] = nodeidf[i][1]
      nodes[2] = nodeidf[i][2]
      nodes[3] = nodeidf[i][2]
      if nodeidf[i][-1] == 4:
        nodes[3] = nodeidf[i][3]
      
      parameters[0] = param1[i]
      parameters[1] = param2[i]
      parameters[2] = -1. * param1[i]
      parameters[3] = -1. * param2[i]
      
      cmpt = 0
      for nod in nodes:
        if search_element(BCdirichlet, oldname[nod]) == 1: 
          V = Pbordnode[nod]
          value_left = -1. * V * parameters[cmpt] / volume[c_left]
          cuda.atomic.add(rhs, c_leftglob, value_left)
          #? rhs[c_leftglob] += value_left
        cmpt = cmpt +1
    
    for idx in range(start, len(dirichletfaces), stride):
      i = dirichletfaces[idx]

      c_left = cellfid[i][0]
      c_leftglob  = loctoglob[c_left]
      
      nodes[0] = nodeidf[i][0]
      nodes[1] = nodeidf[i][1]
      nodes[2] = nodeidf[i][2]
      nodes[3] = nodeidf[i][2]
      if nodeidf[i][-1] == 4:
        nodes[3] = nodeidf[i][3]
      
      parameters[0] = param1[i]
      parameters[1] = param2[i]
      parameters[2] = -1. * param1[i]
      parameters[3] = -1. * param2[i]
      
      cmpt = 0
      for nod in nodes:
        if centergn[nod][0][3] != -1:    
          V = Pbordnode[nod]
          value_left = -1. * V * parameters[cmpt] / volume[c_left]
          cuda.atomic.add(rhs, c_leftglob, value_left)
          #? rhs[c_leftglob] += value_left
        cmpt = cmpt +1
          
      V_K = Pbordface[i]
      value = -2. * param3[i] / volume[c_left] * V_K
      #? rhs[c_leftglob] += value
      cuda.atomic.add(rhs, c_leftglob, value)
      

  kernel_get_rhs_glob_3d = GPU_Backend.compile_kernel(kernel_get_rhs_glob_3d)
  kernel_assign_float = GPU_Backend.compile_kernel(kernel_assign)
  
  def result(*args):
    VarClass.debug(kernel_get_rhs_glob_3d, args)
    args = [VarClass.to_device(arg) for arg in args]
    kernel_assign_float[1, 1, GPU_Backend.stream](args[12], 0.0) #rhs_loc
    GPU_Backend.stream.synchronize()
    # matrixinnerfaces halofaces dirichletfaces
    size = max(len(args[15]), len(args[16]), len(args[17]))
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_get_rhs_glob_3d[nb_blocks, nb_threads, GPU_Backend.stream](*args)
    GPU_Backend.stream.synchronize()

  return result


def get_kernel_compute_P_gradient_3d_diamond():
  
  search_element = GPU_Backend.compile_kernel(device_search_element, device = True)

  def kernel_compute_P_gradient_3d_diamond(
    val_c:'float[:]',
    v_ghost:'float[:]',
    v_halo:'float[:]',
    v_node:'float[:]',
    cellidf:'int32[:,:]',
    nodeidf:'int32[:,:]',
    centergf:'float[:,:]',
    halofid:'int32[:]',
    centerc:'float[:,:]',
    centerh:'float[:,:]',
    oldname:'uint32[:]',
    airDiamond:'float[:]',
    n1:'float[:,:]',
    n2:'float[:,:]',
    n3:'float[:,:]',
    n4:'float[:,:]',
    normalf:'float[:,:]',
    shift:'float[:,:]',
    Pbordnode:'float[:]',
    Pbordface:'float[:]',
    Px_face:'float[:]',
    Py_face:'float[:]',
    Pz_face:'float[:]',
    BCdirichlet:'uint32[:]',
    innerfaces:'uint32[:]',
    halofaces:'uint32[:]',
    neumannfaces:'uint32[:]',
    dirichletfaces:'uint32[:]',
    periodicfaces:'uint32[:]'
    ):
  
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for idx in range(start, len(innerfaces), stride):
      i = innerfaces[idx]

      c_left = cellidf[i][0]
      c_right = cellidf[i][1]
      
      i_1 = nodeidf[i][0]
      i_2 = nodeidf[i][1]   
      i_3 = nodeidf[i][2] 
      i_4 = i_3
      if nodeidf[i][-1] == 4:
        i_4 = nodeidf[i][3]
      
      V_A = v_node[i_1]
      if search_element(BCdirichlet, oldname[i_1]) == 1: 
        V_A = Pbordnode[i_1]
      V_B = v_node[i_2]
      if search_element(BCdirichlet, oldname[i_2]) == 1: 
        V_B = Pbordnode[i_2]
      V_C = v_node[i_3]
      if search_element(BCdirichlet, oldname[i_3]) == 1: 
        V_C = Pbordnode[i_3]
      V_D = v_node[i_4]
      if search_element(BCdirichlet, oldname[i_4]) == 1: 
        V_D = Pbordnode[i_4]
      
      V_L = val_c[c_left]
      V_R = val_c[c_right]

      Px_face[i] = -1. * (n1[i][0]*(V_A - V_C) + n2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
      Py_face[i] = -1. * (n1[i][1]*(V_A - V_C) + n2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
      Pz_face[i] = -1. * (n1[i][2]*(V_A - V_C) + n2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]
  
    for idx in range(start, len(periodicfaces), stride):
      i = periodicfaces[idx]

      c_left = cellidf[i][0]
      c_right = cellidf[i][1]
      
      i_1 = nodeidf[i][0]
      i_2 = nodeidf[i][1]   
      i_3 = nodeidf[i][2] 
      i_4 = i_3
      if nodeidf[i][-1] == 4:
        i_4 = nodeidf[i][3]
      
      V_A = v_node[i_1]
      if search_element(BCdirichlet, oldname[i_1]) == 1: 
        V_A = Pbordnode[i_1]
      V_B = v_node[i_2]
      if search_element(BCdirichlet, oldname[i_2]) == 1: 
        V_B = Pbordnode[i_2]
      V_C = v_node[i_3]
      if search_element(BCdirichlet, oldname[i_3]) == 1: 
        V_C = Pbordnode[i_3]
      V_D = v_node[i_4]
      if search_element(BCdirichlet, oldname[i_4]) == 1: 
        V_D = Pbordnode[i_4]
      
      V_L = val_c[c_left]
      V_R = val_c[c_right]

      Px_face[i] = -1. * (n1[i][0]*(V_A - V_C) + n2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
      Py_face[i] = -1. * (n1[i][1]*(V_A - V_C) + n2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
      Pz_face[i] = -1. * (n1[i][2]*(V_A - V_C) + n2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]
  
    for idx in range(start, len(neumannfaces), stride):
      i = neumannfaces[idx]

      c_left = cellidf[i][0]
      c_right = i
      
      i_1 = nodeidf[i][0]
      i_2 = nodeidf[i][1]   
      i_3 = nodeidf[i][2] 
      i_4 = i_3
      if nodeidf[i][-1] == 4:
        i_4 = nodeidf[i][3]
      
      V_A = v_node[i_1]
      if search_element(BCdirichlet, oldname[i_1]) == 1: 
        V_A = Pbordnode[i_1]
      V_B = v_node[i_2]
      if search_element(BCdirichlet, oldname[i_2]) == 1: 
        V_B = Pbordnode[i_2]
      V_C = v_node[i_3]
      if search_element(BCdirichlet, oldname[i_3]) == 1: 
        V_C = Pbordnode[i_3]
      V_D = v_node[i_4]
      if search_element(BCdirichlet, oldname[i_4]) == 1: 
        V_D = Pbordnode[i_4]
      
      V_L = val_c[c_left]
      V_R = v_ghost[c_right]

      Px_face[i] = -1. * (n1[i][0]*(V_A - V_C) + n2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
      Py_face[i] = -1. * (n1[i][1]*(V_A - V_C) + n2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
      Pz_face[i] = -1. * (n1[i][2]*(V_A - V_C) + n2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]
      
    for idx in range(start, len(halofaces), stride):
      i = halofaces[idx]

      c_left = cellidf[i][0]
      c_right = halofid[i]
      
      i_1 = nodeidf[i][0]
      i_2 = nodeidf[i][1]   
      i_3 = nodeidf[i][2] 
      i_4 = i_3
      if nodeidf[i][-1] == 4:
        i_4 = nodeidf[i][3]
      
      V_A = v_node[i_1]
      if search_element(BCdirichlet, oldname[i_1]) == 1: 
        V_A = Pbordnode[i_1]
      V_B = v_node[i_2]
      if search_element(BCdirichlet, oldname[i_2]) == 1: 
        V_B = Pbordnode[i_2]
      V_C = v_node[i_3]
      if search_element(BCdirichlet, oldname[i_3]) == 1: 
        V_C = Pbordnode[i_3]
      V_D = v_node[i_4]
      if search_element(BCdirichlet, oldname[i_4]) == 1: 
        V_D = Pbordnode[i_4]
          
      V_L = val_c[c_left]
      V_R = v_halo[c_right]
      
      Px_face[i] = -1. * (n1[i][0]*(V_A - V_C) + n2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
      Py_face[i] = -1. * (n1[i][1]*(V_A - V_C) + n2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
      Pz_face[i] = -1. * (n1[i][2]*(V_A - V_C) + n2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]
          
    for idx in range(start, len(dirichletfaces), stride):
      i = dirichletfaces[idx]

      c_left = cellidf[i][0]
      c_right = i
  
      i_1 = nodeidf[i][0]
      i_2 = nodeidf[i][1]   
      i_3 = nodeidf[i][2] 
      i_4 = i_3
      if nodeidf[i][-1] == 4:
        i_4 = nodeidf[i][3]
      
      V_A = Pbordnode[i_1]
      V_B = Pbordnode[i_2]
      V_C = Pbordnode[i_3]
      V_D = Pbordnode[i_4]
      
      V_L = val_c[c_left]
      V_K = Pbordface[i]
      V_R = 2. * V_K - V_L
      
      Px_face[i] = -1. * (n1[i][0]*(V_A - V_C) + n2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
      Py_face[i] = -1. * (n1[i][1]*(V_A - V_C) + n2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
      Pz_face[i] = -1. * (n1[i][2]*(V_A - V_C) + n2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]

  

  kernel_compute_P_gradient_3d_diamond = GPU_Backend.compile_kernel(kernel_compute_P_gradient_3d_diamond)
  
  def result(*args):
    VarClass.debug(kernel_compute_P_gradient_3d_diamond, args)
    args = [VarClass.to_device(arg) for arg in args]
    # innerfaces periodicfaces neumannfaces halofaces dirichletfaces
    size = max(len(args[24]), len(args[25]), len(args[26]), len(args[27]), len(args[28]))
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_compute_P_gradient_3d_diamond[nb_blocks, nb_threads, GPU_Backend.stream](*args)
    GPU_Backend.stream.synchronize()

  return result
