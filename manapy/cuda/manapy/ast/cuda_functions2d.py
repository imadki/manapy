import numpy as np
from numba import cuda
from manapy.cuda.utils import (
    VarClass,
    GPU_Backend
)
from manapy.cuda.manapy.util_kernels import (
  kernel_assign,
  device_search_element
)


#? manapy/ast/functions2d.py

# ✅ ❌ 🔨
# cell_gradient_2d ✅
# face_gradient_2d ✅
# centertovertex_2d ✅
# barthlimiter_2d ✅
# get_triplet_2d ✅ 
# compute_2dmatrix_size ✅
# get_rhs_loc_2d ✅
# get_rhs_glob_2d ✅
# compute_P_gradient_2d_diamond ✅
# get_triplet_2d_with_contrib ✅

# compute_P_gradient_2d_FV4 ❌ (not implemented)
# Mat_Assembly ❌ (not needed)
# Vec_Assembly ❌ (not needed)
# get_rhs_glob_2d_with_contrib ❌ (not needed)
# get_rhs_loc_2d_with_contrib ❌ (not needed)

#! In get_triplet_2
#! a_loc jcn_loc irn_loc if cmpt does not cover all the element of these arrays it should be reset before start the kernel


def get_kernel_cell_gradient_2d():
  
  def kernel_cell_gradient_2d(
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
    
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(idx, w_c.shape[0], stride):  
      i_xx  = 0.;  i_yy  = 0.; i_xy = 0.
      j_xw = 0.;  j_yw = 0.

      centerc_i_x = centerc[i][0]
      centerc_i_y = centerc[i][1]
      w_c_i = w_c[i]

      for j in range(cellnid[i][-1]):
        cell = cellnid[i][j]
        j_x = centerc[cell][0] - centerc_i_x
        j_y = centerc[cell][1] - centerc_i_y
        i_xx += j_x * j_x
        i_yy += j_y * j_y
        i_xy += (j_x * j_y)

        j_xw += (j_x * (w_c[cell] - w_c_i ))
        j_yw += (j_y * (w_c[cell] - w_c_i ))

      for j in range(ghostnid[i][-1]):
        cell = ghostnid[i][j]
        j_x = centergf[cell][0] - centerc[i][0]
        j_y = centergf[cell][1] - centerc[i][1]
        i_xx += j_x*j_x
        i_yy += j_y*j_y
        i_xy += (j_x * j_y)

        j_xw += (j_x * (w_ghost[cell] - w_c[i] ))
        j_yw += (j_y * (w_ghost[cell] - w_c[i] ))
      
      for k in range(nodecid[i][-1]):
        nod = nodecid[i][k]
        if vertexn[nod][3] == 11 or vertexn[nod][3] == 22:
          for j in range(periodic[nod][-1]):
            cell = np.int32(periodic[nod][j])
            center = centerc[cell][0:3]
            j_x = center[0] + shift[cell][0] - centerc[i][0]
            j_y = center[1] - centerc[i][1]
            
            i_xx += j_x*j_x
            i_yy += j_y*j_y
            i_xy += (j_x * j_y)
            
            j_xw += (j_x * (w_c[cell] - w_c[i] ))
            j_yw += (j_y * (w_c[cell] - w_c[i] ))
                
        if vertexn[nod][3] == 33 or vertexn[nod][3] == 44:
          for j in range(periodic[nod][-1]):
            cell = np.int32(periodic[nod][j])
            center = centerc[cell][0:3]
            j_x = center[0] - centerc[i][0]
            j_y = center[1] + shift[cell][1] - centerc[i][1]
            
            i_xx += j_x*j_x
            i_yy += j_y*j_y
            i_xy += (j_x * j_y)
            
            j_xw += (j_x * (w_c[cell] - w_c[i] ))
            j_yw += (j_y * (w_c[cell] - w_c[i] ))
                  
      for j in range(halonid[i][-1]):
        cell = halonid[i][j]
        j_x = centerh[cell][0] - centerc[i][0]
        j_y = centerh[cell][1] - centerc[i][1]
        
        i_xx += j_x*j_x
        i_yy += j_y*j_y
        i_xy += (j_x * j_y)
        
        j_xw += (j_x * (w_halo[cell]  - w_c[i] ))
        j_yw += (j_y * (w_halo[cell]  - w_c[i] ))
              
      for j in range(haloghostnid[i][-1]):
        cell = haloghostnid[i][j]
        center = halocenterg[cell] #!---

        j_x = center[0] - centerc[i][0]
        j_y = center[1] - centerc[i][1]
        
        i_xx += j_x*j_x
        i_yy += j_y*j_y
        i_xy += (j_x * j_y)

        j_xw += (j_x * (w_haloghost[cell] - w_c[i] ))
        j_yw += (j_y * (w_haloghost[cell] - w_c[i] ))
            

      dia = i_xx * i_yy - i_xy * i_xy
      w_x[i]  = (i_yy * j_xw - i_xy * j_yw) / dia
      w_y[i]  = (i_xx * j_yw - i_xy * j_xw) / dia
      w_z[i]  = 0.
  
  kernel_cell_gradient_2d = GPU_Backend.compile_kernel(kernel_cell_gradient_2d)
  
  def result(*args):
    VarClass.debug(kernel_cell_gradient_2d, args)
    args = [VarClass.to_device(arg) for arg in args]
    size = len(args[0]) #w_c
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_cell_gradient_2d[nb_blocks, nb_threads, GPU_Backend.stream](*args)
    GPU_Backend.stream.synchronize()

  return result
  

def get_kernel_face_gradient_2d():
  
  def kernel_face_gradient_2d(
    w_c:'float[:]', 
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

    for idx in range(start, innerfaces.shape[0], stride):
      i = innerfaces[idx]
      c_left = cellidf[i][0]
      c_right = cellidf[i][1]
          
      i_1 = nodeidf[i][0]
      i_2 = nodeidf[i][1]
      
      vi1 = w_node[i_1]
      vi2 = w_node[i_2]
      vv1 = w_c[c_left]
      vv2 = w_c[c_right]
      
      wx_face[i] = 1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
      wy_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])

    for idx in range(start, periodicfaces.shape[0], stride):
      i = periodicfaces[idx]
      c_left = cellidf[i][0]
      c_right = cellidf[i][1]
          
      i_1 = nodeidf[i][0]
      i_2 = nodeidf[i][1]
      
      vi1 = w_node[i_1]
      vi2 = w_node[i_2]
      vv1 = w_c[c_left]
      vv2 = w_c[c_right]
      
      wx_face[i] = 1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
      wy_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])

    for idx in range(start, halofaces.shape[0], stride):
      i = halofaces[idx]
      c_left = cellidf[i][0]
      c_right = halofid[i]
      
      i_1 = nodeidf[i][0]
      i_2 = nodeidf[i][1]
      
      vi1 = w_node[i_1]
      vi2 = w_node[i_2]
      vv1 = w_c[c_left]
      vv2 = w_halo[c_right]
      
      wx_face[i] = 1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
      wy_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])

    for idx in range(start, dirichletfaces.shape[0], stride):
      i = dirichletfaces[idx]
      c_left = cellidf[i][0]
      c_right = i
      
      i_1 = nodeidf[i][0]
      i_2 = nodeidf[i][1]
      
      vi1 = w_node[i_1]
      vi2 = w_node[i_2]
      vv1 = w_c[c_left]
      vv2 = w_ghost[c_right]

      wx_face[i] = 1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
      wy_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])

    for idx in range(start, neumann.shape[0], stride):
      i = neumann[idx]
      c_left = cellidf[i][0]
      c_right = i
      
      i_1 = nodeidf[i][0]
      i_2 = nodeidf[i][1]
      
      vi1 = w_node[i_1]
      vi2 = w_node[i_2]
      vv1 = w_c[c_left]
      vv2 = w_ghost[c_right]

      wx_face[i] = 1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
      wy_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])

  kernel_face_gradient_2d = GPU_Backend.compile_kernel(kernel_face_gradient_2d)
  
  def result(*args):
    VarClass.debug(kernel_face_gradient_2d, args)
    args = [VarClass.to_device(arg) for arg in args]
    # innerfaces periodicfaces halofaces dirichletfaces neumann
    size = max(len(args[21]), len(args[22]), len(args[23]), len(args[24]), len(args[25]))
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_face_gradient_2d[nb_blocks, nb_threads, GPU_Backend.stream](*args)
    GPU_Backend.stream.synchronize()

  return result


def get_kernel_centertovertex_2d():
  
  def kernel_centertovertex_2d(
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
    lambda_x:'float[:]',
    lambda_y:'float[:]', 
    lambda_z:'float[:]', 
    number:'uint32[:]', 
    shift:'float[:,:]', 
    w_n:'float[:]'
     ):
    

      #? w_n[:] = 0.

      start = cuda.grid(1)
      stride = cuda.gridsize(1)

      for i in range(start, vertexn.shape[0], stride):
        w_n[i] = 0.
        for j in range(cellid[i][-1]):
          cell = cellid[i][j]
          #? center[:] = centerc[cell][:]
          
          xdiff = centerc[cell][0] - vertexn[i][0]
          ydiff = centerc[cell][1] - vertexn[i][1]
          alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
          
          w_n[i]  += alpha * w_c[cell]
            
        for j in range(ghostid[i][-1]):
          cell = ghostid[i][j]
          #? center[:] = centergf[cell][0:3]
          
          xdiff = centergf[cell][0] - vertexn[i][0]
          ydiff = centergf[cell][1] - vertexn[i][1]
          alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
          
          w_n[i]  += alpha * w_ghost[cell]
            
            
        for j in range(haloghostid[i][-1]):
          cell = haloghostid[i][j]
          #? center[:] = halocenterg[cell][0:3]
        
          xdiff = halocenterg[cell][0] - vertexn[i][0]
          ydiff = halocenterg[cell][1] - vertexn[i][1]
          
          alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
          
          w_n[i]  += alpha * w_haloghost[cell]
          
        for j in range(haloid[i][-1]):
          cell = haloid[i][j]
          #? center[:] = centerh[cell][0:3]
        
          xdiff = centerh[cell][0] - vertexn[i][0]
          ydiff = centerh[cell][1] - vertexn[i][1]
          alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
        
          w_n[i]  += alpha * w_halo[cell]
                
        #TODO Must be keeped like that checked ok ;)
        if vertexn[i][3] == 11 or vertexn[i][3] == 22 :
          for j in range(periodicid[i][-1]):
            cell = periodicid[i][j]
            #? center[:] = centerc[cell][0:3] 
            
            xdiff = centerc[cell][0] + shift[cell][0] - vertexn[i][0]
            ydiff = centerc[cell][1] - vertexn[i][1]
            alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
            
            w_n[i]  += alpha * w_c[cell]
                    
        elif vertexn[i][3] == 33 or vertexn[i][3] ==44:
          for j in range(periodicid[i][-1]):
            cell = periodicid[i][j]
            #? center[:] = centerc[cell][0:3] 
            
            xdiff =  centerc[cell][0] - vertexn[i][0]
            ydiff =  centerc[cell][1] + shift[cell][1] - vertexn[i][1]
            alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
            
            w_n[i]  += alpha * w_c[cell]
  
  kernel_centertovertex_2d = GPU_Backend.compile_kernel(kernel_centertovertex_2d)
  
  def result(*args):
    VarClass.debug(kernel_centertovertex_2d, args)
    args = [VarClass.to_device(arg) for arg in args]
    # vertexn
    size = len(args[11])
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_centertovertex_2d[nb_blocks, nb_threads, GPU_Backend.stream](*args)
    GPU_Backend.stream.synchronize()

  return result


def get_kernel_barthlimiter_2d():
  
  def kernel_barthlimiter_2d(
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

      for i in range(start, w_c.shape[0], stride):
        val  = 1.
        psi[i] = val

        w_max = w_c[i]
        w_min = w_c[i]

        for j in range(faceid[i][-1]):
            face = faceid[i][j]
            if namef[face] == 0 or namef[face] > 10:
            #11 or namef[face] == 22 or namef[face] == 33 or namef[face] == 44:
                w_max = max(w_max, w_c[cellid[face][0]], w_c[cellid[face][1]])
                w_min = min(w_min, w_c[cellid[face][0]], w_c[cellid[face][1]])
            elif namef[face] == 1 or namef[face] == 2 or namef[face] == 3 or namef[face] == 4:
                w_max = max(w_max,  w_c[cellid[face][0]], w_ghost[face])
                w_min = min(w_min,  w_c[cellid[face][0]], w_ghost[face])
            else:
                w_max = max(w_max,  w_c[cellid[face][0]], w_halo[halofid[face]])
                w_min = min(w_min,  w_c[cellid[face][0]], w_halo[halofid[face]])
        
        for j in range(faceid[i][-1]):
            face = faceid[i][j]

            r_xyz1 = centerf[face][0] - centerc[i][0] 
            r_xyz2 = centerf[face][1] - centerc[i][1]
            
            delta2 = w_x[i] * r_xyz1 + w_y[i] * r_xyz2 
            
            #TODO choice of epsilon
            #!np.fabs
            #!if abs(delta2) < 1e-8:
            psi_ij = 1.

            if abs(delta2) >= 1e-8:
                if delta2 > 0.:
                    value = (w_max - w_c[i]) / delta2
                    psi_ij = min(val, value)
                if delta2 < 0.:
                    value = (w_min - w_c[i]) / delta2
                    psi_ij = min(val, value)

            psi[i] = min(psi[i], psi_ij)

  kernel_barthlimiter_2d = GPU_Backend.compile_kernel(kernel_barthlimiter_2d)
  
  def result(*args):
    VarClass.debug(kernel_barthlimiter_2d, args)
    args = [VarClass.to_device(arg) for arg in args]
    size = len(args[0]) #w_c
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_barthlimiter_2d[nb_blocks, nb_threads, GPU_Backend.stream](*args)
    GPU_Backend.stream.synchronize()

  return result


# kernel_assign -> s_cmpt = 0 ✅
# need search_element ✅
def get_kernel_get_triplet_2d():

  d_s_cmpt = cuda.device_array(shape=(1), dtype='uint64')
  search_element = GPU_Backend.compile_kernel(device_search_element, device = True)

  def kernel_get_triplet_2d(
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

      parameters = cuda.local.array(2, param4.dtype)

      for idx in range(start, matrixinnerfaces.shape[0], stride):
        i = matrixinnerfaces[idx]

        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        parameters[0] = param4[i]
        parameters[1] = param2[i]
    
        c_right = cellfid[i][1]
        c_rightglob = loctoglob[c_right]
        
        cmpt = cuda.atomic.add(s_cmpt, 0, 1)
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = param1[i] / volume[c_left]
        a_loc[cmpt] = value
        
        cmptparam = 0
        for nod in nodeidf[i][:nodeidf[i][-1]]:
            if search_element(BCdirichlet, oldnamen[nod]) == 0: 
                for j in range(cellnid[nod][-1]):
                    center = centerc[cellnid[nod][j]][0:2]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                    value =  alpha / volume[c_left] * parameters[cmptparam]
                    cmpt = cuda.atomic.add(s_cmpt, 0, 1)
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
                    a_loc[cmpt] = value
                    #right cell-----------------------------------                                                                                              
                    value =  -1. * alpha / volume[c_right] * parameters[cmptparam]
                    cmpt = cuda.atomic.add(s_cmpt, 0, 1)
                    irn_loc[cmpt] = c_rightglob
                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
                    a_loc[cmpt] = value
                
                for j in range(len(centergn[nod])):
                    if centergn[nod][j][-1] != -1:
                        center = centergn[nod][j][0:2]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                        index = np.int32(centergn[nod][j][2])
                        value = alpha / volume[c_left] * parameters[cmptparam]
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
                        center = halocentergn[nod][j][0:2]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                        index = np.int32(halocentergn[nod][j][2])
                        value = alpha / volume[c_left] * parameters[cmptparam]
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
                    
                for j in range(periodicnid[nod][-1]):
                    if vertexn[nod][3] == 11 or vertexn[nod][3] == 22:
                        center[0] = centerc[periodicnid[nod][j]][0]  + shift[periodicnid[nod][j]][0]
                        center[1] = centerc[periodicnid[nod][j]][1]  
                    if vertexn[nod][3] == 33 or vertexn[nod][3] == 44:
                        center[0] = centerc[periodicnid[nod][j]][0]  
                        center[1] = centerc[periodicnid[nod][j]][1]  + shift[periodicnid[nod][j]][1]
                    
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
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
                
                for j in range(halonid[nod][-1]):
                    center = centerh[halonid[nod][j]][0:2]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                    value =  alpha / volume[c_left] * parameters[cmptparam]
                    cmpt = cuda.atomic.add(s_cmpt, 0, 1)
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
                    a_loc[cmpt] = value
                    #right cell-----------------------------------                                                                                              
                    value =  -1. * alpha / volume[c_right] * parameters[cmptparam]
                    cmpt = cuda.atomic.add(s_cmpt, 0, 1)
                    irn_loc[cmpt] = c_rightglob
                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
                    a_loc[cmpt] = value
            cmptparam =+1
        
        cmpt = cuda.atomic.add(s_cmpt, 0, 1)
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_rightglob
        value =  param3[i] / volume[c_left]
        a_loc[cmpt] = value

        # right cell------------------------------------------------------
        cmpt = cuda.atomic.add(s_cmpt, 0, 1)
        irn_loc[cmpt] = c_rightglob
        jcn_loc[cmpt] = c_leftglob
        value =  -1. * param1[i] / volume[c_right]
        a_loc[cmpt] = value
    
        cmpt = cuda.atomic.add(s_cmpt, 0, 1)
        irn_loc[cmpt] = c_rightglob
        jcn_loc[cmpt] = c_rightglob
        value =  -1. * param3[i] / volume[c_right]
        a_loc[cmpt] = value
      
      for idx in range(start, halofaces.shape[0], stride):
        i = halofaces[idx]
        
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        parameters[0] = param4[i]; parameters[1] = param2[i]
        
        c_rightglob = haloext[halofid[i]][0]
        c_right     = halofid[i]
        
        cmpt = cuda.atomic.add(s_cmpt, 0, 1)
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value =  param1[i] / volume[c_left]
        a_loc[cmpt] = value

        cmpt = cuda.atomic.add(s_cmpt, 0, 1)
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_rightglob
        value =  param3[i] / volume[c_left]
        a_loc[cmpt] = value
        
        cmptparam = 0
        for nod in nodeidf[i][:nodeidf[i][-1]]:
            if search_element(BCdirichlet, oldnamen[nod]) == 0: 
                for j in range(cellnid[nod][-1]):
                    center = centerc[cellnid[nod][j]][0:2]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                    value =  alpha / volume[c_left] * parameters[cmptparam]
                    cmpt = cuda.atomic.add(s_cmpt, 0, 1)
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
                    a_loc[cmpt] = value
                    
                for j in range(len(centergn[nod])):
                    if centergn[nod][j][-1] != -1:
                        center = centergn[nod][j][0:2]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                        index = np.int32(centergn[nod][j][2])
                        value = alpha / volume[c_left] * parameters[cmptparam]
                        cmpt = cuda.atomic.add(s_cmpt, 0, 1)
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = loctoglob[index]
                        a_loc[cmpt] = value
                        
                for j in range(len(halocentergn[nod])):
                    if halocentergn[nod][j][-1] != -1:
                        center = halocentergn[nod][j][0:2]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                        index = np.int32(halocentergn[nod][j][2])
                        value = alpha / volume[c_left] * parameters[cmptparam]
                        cmpt = cuda.atomic.add(s_cmpt, 0, 1)
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = haloext[index][0]
                        a_loc[cmpt] = value

                for j in range(halonid[nod][-1]):
                    center = centerh[halonid[nod][j]][0:2]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                    value =  alpha / volume[c_left] * parameters[cmptparam]
                    cmpt = cuda.atomic.add(s_cmpt, 0, 1)
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
                    a_loc[cmpt] = value
            cmptparam +=1
      
      for idx in range(start, dirichletfaces.shape[0], stride):
        i = dirichletfaces[idx]
        
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        cmpt = cuda.atomic.add(s_cmpt, 0, 1)
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = param1[i] / volume[c_left]
        a_loc[cmpt] = value
        
        cmpt = cuda.atomic.add(s_cmpt, 0, 1)
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = -1. * param3[i] / volume[c_left]
        a_loc[cmpt] = value
  
  kernel_get_triplet_2d = GPU_Backend.compile_kernel(kernel_get_triplet_2d)
  kernel_assign_int64 = cuda.jit('void(uint64[:], uint64)')(kernel_assign)

  def result(*args):
    VarClass.debug(kernel_get_triplet_2d, args)
    args = [VarClass.to_device(arg) for arg in args]
    kernel_assign_int64[1, 1, GPU_Backend.stream](d_s_cmpt, 0) #s_cmpt
    GPU_Backend.stream.synchronize()
    size = max(len(args[35]), len(args[34]), len(args[33])) #matrixinnerfaces, halofaces, dirichletfaces
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_get_triplet_2d[nb_blocks, nb_threads, GPU_Backend.stream](*args, d_s_cmpt)
    GPU_Backend.stream.synchronize()

  return result

# kernel_assign -> cmpt = 0 ✅
# need search_element ✅
# return ✅
def get_kernel_compute_2dmatrix_size():
  
  d_cmpt = cuda.device_array(shape=(1), dtype='uint64')
  search_element = GPU_Backend.compile_kernel(device_search_element, device = True)

  def kernel_compute_2dmatrix_size(
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
    cmpt : 'uint64[:]'
    ):                                                                                                                                                                       
      
      start = cuda.grid(1)
      stride = cuda.gridsize(1)

      for idx in range(start, matrixinnerfaces.shape[0], stride):
        i = matrixinnerfaces[idx]
        
        cuda.atomic.add(cmpt, 0, 1)
        
        for nod in nodeidf[i][:nodeidf[i][-1]]:
            if search_element(BCdirichlet, oldnamen[nod]) == False:# and search_element(BCneumannNH, oldnamen[nod]) == False:
            # if vertexn[nod][3] not in BCdirichlet:
                for j in range(cellnid[nod][-1]):
                    
                    #cuda.atomic.add(cmpt, 0, 1)
                    #right cell-----------------------------------                                                                                              
                    cuda.atomic.add(cmpt, 0, 2)
                
                for j in range(len(centergn[nod])):
                    if centergn[nod][j][-1] != -1:
                      
                        #cuda.atomic.add(cmpt, 0, 1)
                        #right cell-----------------------------------                                                                                              
                        cuda.atomic.add(cmpt, 0, 2)
                        
                for j in range(len(halocentergn[nod])):
                    if halocentergn[nod][j][-1] != -1:
                        
                        #cuda.atomic.add(cmpt, 0, 1)
                        #right cell-----------------------------------                                                                                              
                        cuda.atomic.add(cmpt, 0, 2)
                    
                for j in range(periodicnid[nod][-1]):
                    #cuda.atomic.add(cmpt, 0, 1)
                    #right cell-----------------------------------                                                                                              
                    cuda.atomic.add(cmpt, 0, 2)
                
                for j in range(halonid[nod][-1]):
                  
                    #cuda.atomic.add(cmpt, 0, 1)
                    #right cell-----------------------------------                                                                                              
                    cuda.atomic.add(cmpt, 0, 2)
        
        #cuda.atomic.add(cmpt, 0, 1)
        # right cell------------------------------------------------------
        #cuda.atomic.add(cmpt, 0, 1)
        cuda.atomic.add(cmpt, 0, 3)
              
      # elif namef[i] == 10:
      for idx in range(start, halofaces.shape[0], stride):
        i = halofaces[idx]
        #cuda.atomic.add(cmpt, 0, 1)
        
        #cuda.atomic.add(cmpt, 0, 1)
        cuda.atomic.add(cmpt, 0, 3)
        
        for nod in nodeidf[i][:nodeidf[i][-1]]:
            if search_element(BCdirichlet, oldnamen[nod]) == False:  
                for j in range(cellnid[nod][-1]):
                    cuda.atomic.add(cmpt, 0, 1)
                    
                for j in range(len(centergn[nod])):
                    if centergn[nod][j][-1] != -1:
                        cuda.atomic.add(cmpt, 0, 1)
                        
                for j in range(len(halocentergn[nod])):
                    if halocentergn[nod][j][-1] != -1:
                        cuda.atomic.add(cmpt, 0, 1)

                for j in range(halonid[nod][-1]):
                    cuda.atomic.add(cmpt, 0, 1)

      for idx in range(start, dirichletfaces.shape[0], stride):      
        #i = dirichletfaces[idx]
        #cuda.atomic.add(cmpt, 0, 1)
        cuda.atomic.add(cmpt, 0, 2)
  

  kernel_compute_2dmatrix_size = GPU_Backend.compile_kernel(kernel_compute_2dmatrix_size)
  kernel_assign_int64 = cuda.jit('void(uint64[:], uint64)')(kernel_assign)
  
  def result(*args):
    VarClass.debug(kernel_compute_2dmatrix_size, args)
    args = [VarClass.to_device(arg) for arg in args]
    kernel_assign_int64[1, 1, GPU_Backend.stream](d_cmpt, 0) #cmpt
    GPU_Backend.stream.synchronize()
    size = max(len(args[9]), len(args[10]), len(args[11])) #matrixinnerfaces, halofaces, dirichletfaces
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_compute_2dmatrix_size[nb_blocks, nb_threads, GPU_Backend.stream](*args, d_cmpt)
    return d_cmpt.copy_to_host(stream=GPU_Backend.stream)[0]
    #GPU_Backend.stream.synchronize()


  return result
        

# kernel_assign -> rhs_loc[:] = 0 ✅
# need search_element ✅
def get_kernel_get_rhs_loc_2d():

  search_element = GPU_Backend.compile_kernel(device_search_element, device=True)
  
  def kernel_get_rhs_loc_2d(
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
      dirichletfaces:'uint32[:]'
      ):                                                                                                                                                                       
      
      start = cuda.grid(1)
      stride = cuda.gridsize(1)

      #? rhs_loc[:] = 0. => secondary kernel
      for idx in range(start, matrixinnerfaces.shape[0], stride):
        i = matrixinnerfaces[idx]

        c_right = cellfid[i][1]
        c_left = cellfid[i][0]
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]    
        
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            V = Pbordnode[i_1]
            value_left = (-1. * V * param4[i]) / volume[c_left]
            cuda.atomic.add(rhs_loc, c_left, value_left)
            #!rhs_loc[c_left] +=  value_left
            
            value_right = V * param4[i] / volume[c_right]
            cuda.atomic.add(rhs_loc, c_right, value_right)
            #!rhs_loc[c_right] += value_right
            
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            V = Pbordnode[i_2]
            value_left =  (-1. * V * param2[i]) / volume[c_left]
            cuda.atomic.add(rhs_loc, c_left, value_left)
            #!rhs_loc[c_left] += value_left
            
            value_right =  (V * param2[i]) / volume[c_right]
            cuda.atomic.add(rhs_loc, c_right, value_right)
            #!rhs_loc[c_right] += value_right


      for idx in range(start, halofaces.shape[0], stride):       
        i = halofaces[idx]

        c_left = cellfid[i][0]
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]   
        
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            V = Pbordnode[i_1]
            value_left =  -1. * V * param4[i] / volume[c_left]
            #!rhs_loc[c_left] += value_left
            cuda.atomic.add(rhs_loc, c_left, value_left)
            
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            V = Pbordnode[i_2]
            value_left =  -1. * V * param2[i] / volume[c_left]
            #!rhs_loc[c_left] += value_left
            cuda.atomic.add(rhs_loc, c_left, value_left)

      for idx in range(start, dirichletfaces.shape[0], stride):   
        i = dirichletfaces[idx]

        c_left = cellfid[i][0]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]   
        
        if centergn[i_1][0][2] != -1:     
            V = Pbordnode[i_1]
            value_left = -1. * V * param4[i] / volume[c_left]
            #!rhs_loc[c_left] += value_left
            cuda.atomic.add(rhs_loc, c_left, value_left)
            
        if centergn[i_2][0][2] != -1: 
            V = Pbordnode[i_2]
            value_left = -1. * V * param2[i] / volume[c_left]
            #!rhs_loc[c_left] += value_left
            cuda.atomic.add(rhs_loc, c_left, value_left)
        
        V_K = Pbordface[i]
        value = -2. * param3[i] / volume[c_left] * V_K
        #!rhs_loc[c_left] += value
        cuda.atomic.add(rhs_loc, c_left, value)

  kernel_get_rhs_loc_2d = GPU_Backend.compile_kernel(kernel_get_rhs_loc_2d)
  kernel_assign_float = GPU_Backend.compile_kernel(kernel_assign)
  
  def result(*args):
    VarClass.debug(kernel_get_rhs_loc_2d, args)
    args = [VarClass.to_device(arg) for arg in args]

    size = len(args[12])
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_assign_float[nb_blocks, nb_threads, GPU_Backend.stream](args[12], 0.0) #rhs_loc
    GPU_Backend.stream.synchronize()
    
    size = max(len(args[17]), len(args[16]), len(args[15])) #matrixinnerfaces, halofaces, dirichletfaces
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_get_rhs_loc_2d[nb_blocks, nb_threads, GPU_Backend.stream](*args)
    GPU_Backend.stream.synchronize()

  return result


# kernel_assign -> rhs[:0] ✅
# need search_element ✅
def get_kernel_get_rhs_glob_2d():

  search_element = GPU_Backend.compile_kernel(device_search_element, device = True)
  
  def kernel_get_rhs_glob_2d(
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

      
      #? rhs[:] = 0. => secondary kernel
      for idx in range(start, matrixinnerfaces.shape[0], stride):
        i = matrixinnerfaces[idx]

        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]    
        
        c_right = cellfid[i][1]
        c_rightglob = loctoglob[c_right]
        
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            V = Pbordnode[i_1]
            value_left = -1. * V * param4[i] / volume[c_left]
            #!rhs[c_leftglob] +=  value_left
            cuda.atomic.add(rhs, c_leftglob, value_left)
            
            value_right = V * param4[i] / volume[c_right]
            #!rhs[c_rightglob] += value_right
            cuda.atomic.add(rhs, c_rightglob, value_right)
            
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            V = Pbordnode[i_2]
            value_left =  -1. * V * param2[i] / volume[c_left]
            #!rhs[c_leftglob] += value_left
            cuda.atomic.add(rhs, c_leftglob, value_left)
            
            value_right =  V * param2[i] / volume[c_right]
            #!rhs[c_rightglob] += value_right
            cuda.atomic.add(rhs, c_rightglob, value_right)

      for idx in range(start, halofaces.shape[0], stride):             
        i = halofaces[idx]

        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]    
        
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            V = Pbordnode[i_1]
            value_left =  -1. * V * param4[i] / volume[c_left]
            #!rhs[c_leftglob] += value_left
            cuda.atomic.add(rhs, c_leftglob, value_left)
        
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            V = Pbordnode[i_2]
            value_left =  -1. * V * param2[i] / volume[c_left]
            #!rhs[c_leftglob] += value_left
            cuda.atomic.add(rhs, c_leftglob, value_left)
      
      for idx in range(start, dirichletfaces.shape[0], stride):
        i = dirichletfaces[idx]

        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]  
        
        if centergn[i_1][0][2] != -1:     
            V = Pbordnode[i_1]
            value_left = -1. * V * param4[i] / volume[c_left]
            #!rhs[c_leftglob] += value_left
            cuda.atomic.add(rhs, c_leftglob, value_left)
            
        if centergn[i_2][0][2] != -1: 
            V = Pbordnode[i_2]
            value_left = -1. * V * param2[i] / volume[c_left]
            #!rhs[c_leftglob] += value_left
            cuda.atomic.add(rhs, c_leftglob, value_left)
        
        V_K = Pbordface[i]
        value = -2. * param3[i] / volume[c_left] * V_K
        #!rhs[c_leftglob] += value
        cuda.atomic.add(rhs, c_leftglob, value)

  kernel_get_rhs_glob_2d = GPU_Backend.compile_kernel(kernel_get_rhs_glob_2d)
  kernel_assign_float = GPU_Backend.compile_kernel(kernel_assign)
  
  def result(*args):
    VarClass.debug(kernel_get_rhs_glob_2d, args)
    args = [VarClass.to_device(arg) for arg in args]

    size = len(args[12])
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_assign_float[nb_blocks, nb_threads, GPU_Backend.stream](args[12], 0.0) #rhs
    GPU_Backend.stream.synchronize()
    
    size = max(len(args[17]), len(args[16]), len(args[15])) #matrixinnerfaces, halofaces, dirichletfaces
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_get_rhs_glob_2d[nb_blocks, nb_threads, GPU_Backend.stream](*args)
    GPU_Backend.stream.synchronize()

  return result        


# need search_element ✅
def get_kernel_compute_P_gradient_2d_diamond():

  search_element = GPU_Backend.compile_kernel(device_search_element, device = True)
  
  def kernel_compute_P_gradient_2d_diamond(
    P_c:'float[:]', 
    P_ghost:'float[:]', 
    P_halo:'float[:]', 
    P_node:'float[:]', 
    cellidf:'int32[:,:]', 
    nodeidf:'int32[:,:]', 
    centergf:'float[:,:]', 
    halofid:'int32[:]', 
    centerc:'float[:,:]', 
    centerh:'float[:,:]', 
    oldname:'uint32[:]', 
    airDiamond:'float[:]', 
    f_1:'float[:,:]', 
    f_2:'float[:,:]',
    f_3:'float[:,:]', 
    f_4:'float[:,:]', 
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

      for idx in range(start, innerfaces.shape[0], stride):
        i = innerfaces[idx]

        c_left = cellidf[i][0]
        c_right = cellidf[i][1]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]
        
        vi1 = P_node[i_1]
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            vi1 = Pbordnode[i_1]
        vi2 = P_node[i_2]
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            vi2 = Pbordnode[i_2]

        vv1 = P_c[c_left]
        vv2 = P_c[c_right]
        
        Px_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
        Py_face[i] =  1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])
      
      for idx in range(start, periodicfaces.shape[0], stride):
        i = periodicfaces[idx]
        
        c_left = cellidf[i][0]
        c_right = cellidf[i][1]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]
        
        vi1 = P_node[i_1]
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            vi1 = Pbordnode[i_1]
        vi2 = P_node[i_2]
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            vi2 = Pbordnode[i_2]

        vv1 = P_c[c_left]
        vv2 = P_c[c_right]
        
        Px_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
        Py_face[i] =  1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])
      
      for idx in range(start, neumannfaces.shape[0], stride):
        i = neumannfaces[idx]
        
        c_left = cellidf[i][0]
        c_right = i
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]
            
        vi1 = P_node[i_1]
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            vi1 = Pbordnode[i_1]
        vi2 = P_node[i_2]
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            vi2 = Pbordnode[i_2]

        vv1 = P_c[c_left]
        vv2 = P_ghost[c_right]
            
        Px_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
        Py_face[i] =  1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])

      for idx in range(start, halofaces.shape[0], stride):
        i = halofaces[idx]

        c_left = cellidf[i][0]
        c_right = halofid[i]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]
        
        vi1 = P_node[i_1]
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            vi1 = Pbordnode[i_1]
        vi2 = P_node[i_2]
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            vi2 = Pbordnode[i_2]
        
        vv1 = P_c[c_left]
        vv2 = P_halo[c_right]
        
        Px_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
        Py_face[i] =  1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])
      
      for idx in range(start, dirichletfaces.shape[0], stride):
          i = dirichletfaces[idx]
          
          c_left = cellidf[i][0]
          c_right = i
          
          i_1 = nodeidf[i][0]
          i_2 = nodeidf[i][1]
          
          vi1 = Pbordnode[i_1]
          vi2 = Pbordnode[i_2]
          vv1 = P_c[c_left]
          
          VK = Pbordface[i]
          vv2 = 2. * VK - vv1

          Px_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
          Py_face[i] =  1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])

  kernel_compute_P_gradient_2d_diamond = GPU_Backend.compile_kernel(kernel_compute_P_gradient_2d_diamond)
  
  def result(*args):
    VarClass.debug(kernel_compute_P_gradient_2d_diamond, args)
    args = [VarClass.to_device(arg) for arg in args]

    size = max(len(args[28]), len(args[27]), len(args[26]), len(args[25]), len(args[24])) # innerfaces periodicfaces neumannfaces halofaces dirichletfaces
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_compute_P_gradient_2d_diamond[nb_blocks, nb_threads, GPU_Backend.stream](*args)
    GPU_Backend.stream.synchronize()

  return result 


# kernel_assign -> s_cmpt = 0 ✅
# need search_element ✅
def get_kernel_get_triplet_2d_with_contrib():

  d_s_cmpt = cuda.device_array(shape=(1), dtype='uint64')
  search_element = GPU_Backend.compile_kernel(device_search_element, device = True)
  
  def kernel_get_triplet_2d_with_contrib(
    cellfid:'int32[:,:]', 
    nodeidf:'int32[:,:]', 
    faceidc:'int32[:,:]', 
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
    number:'uint32[:]', 
    R_x:'float[:]', 
    R_y:'float[:]', 
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
    Icell:'float[:]',
    alpha_P:'float', 
    perm_vec:'float[:]', 
    visc_vec:'float[:]', 
    BCneumannNH:'uint32[:]', 
    dist:'float[:]', 
    s_cmpt: 'uint64[:]'
    ):
      

      start = cuda.grid(1)
      stride = cuda.gridsize(1)

      parameters = cuda.local.array(2, param4.dtype)

      for idx in range(start, matrixinnerfaces.shape[0], stride):
        i = matrixinnerfaces[idx]

        nbfL = faceidc[cellfid[i][0]][-1]
        nbfR = faceidc[cellfid[i][1]][-1]
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        parameters[0] = param4[i]
        parameters[1] = param2[i]
    
        c_right = cellfid[i][1]
        c_rightglob = loctoglob[c_right]
        

        perm = 0.5 * (perm_vec[c_rightglob] + perm_vec[c_leftglob])
        visc = 0.5 * (visc_vec[c_rightglob] + visc_vec[c_leftglob])

        perm_visc = perm / visc
        
        cmpt = cuda.atomic.add(s_cmpt, 0, 1)
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = param1[i] / volume[c_left]
        a_loc[cmpt] = value*Icell[c_left]*(perm_visc) + (1/nbfL)*volume[c_left]*alpha_P*(1 - Icell[c_left])
        
        
        cmptparam = 0
        for nod in nodeidf[i][:nodeidf[i][-1]]:
            if search_element(BCdirichlet, oldnamen[nod]) == 0:# and search_element(BCneumannNH, oldnamen[nod]) == 0:
                for j in range(cellnid[nod][-1]):
                    center = centerc[cellnid[nod][j]][0:2]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                    value =  alpha / volume[c_left] * parameters[cmptparam]
                    cmpt = cuda.atomic.add(s_cmpt, 0, 1)
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
                    a_loc[cmpt] = value*Icell[c_left]*(perm_visc) 
                    #right cell-----------------------------------                                                                                              
                    value =  -1. * alpha / volume[c_right] * parameters[cmptparam]
                    cmpt = cuda.atomic.add(s_cmpt, 0, 1)
                    irn_loc[cmpt] = c_rightglob
                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
                    a_loc[cmpt] = value*Icell[c_right]*(perm_visc) 
                
                for j in range(len(centergn[nod])):
                    if centergn[nod][j][-1] != -1:
                        center = centergn[nod][j][0:2]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                        index = int(centergn[nod][j][2])
                        value = alpha / volume[c_left] * parameters[cmptparam]
                        cmpt = cuda.atomic.add(s_cmpt, 0, 1)
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = loctoglob[index]
                        a_loc[cmpt] = value*Icell[c_left]*(perm_visc) 
                        #right cell-----------------------------------                                                                                              
                        value = -1. * alpha / volume[c_right] * parameters[cmptparam]
                        cmpt = cuda.atomic.add(s_cmpt, 0, 1)
                        irn_loc[cmpt] = c_rightglob
                        jcn_loc[cmpt] = loctoglob[index]
                        a_loc[cmpt] = value*Icell[c_right]*(perm_visc) 
                        
                for j in range(len(halocentergn[nod])):
                    if halocentergn[nod][j][-1] != -1:
                        center = halocentergn[nod][j][0:2]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                        index = int(halocentergn[nod][j][2])
  #                        cell  = int(halocentergn[nod][j][-1])
                        value = alpha / volume[c_left] * parameters[cmptparam]
                        cmpt = cuda.atomic.add(s_cmpt, 0, 1)
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = haloext[index][0]
                        a_loc[cmpt] = value*Icell[c_left]*(perm_visc) 
                        #right cell-----------------------------------                                                                                              
                        value = -1. * alpha / volume[c_right] * parameters[cmptparam]
                        cmpt = cuda.atomic.add(s_cmpt, 0, 1)
                        irn_loc[cmpt] = c_rightglob
                        jcn_loc[cmpt] = haloext[index][0]
                        #TODO 
                        a_loc[cmpt] = value*Icell[c_right]*(perm_visc)#value*Ihaloghost[int(halocentergn[nod][j][-1])]*(perm/visc)   
                              
                for j in range(halonid[nod][-1]):
                    center = centerh[halonid[nod][j]][0:2]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                    value =  alpha / volume[c_left] * parameters[cmptparam]
                    cmpt = cuda.atomic.add(s_cmpt, 0, 1)
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
                    a_loc[cmpt] = value*Icell[c_left]*(perm_visc) 
                    #right cell-----------------------------------                                                                                              
                    value =  -1. * alpha / volume[c_right] * parameters[cmptparam]
                    cmpt = cuda.atomic.add(s_cmpt, 0, 1)
                    irn_loc[cmpt] = c_rightglob
                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
                    a_loc[cmpt] = value*Icell[c_right]*(perm_visc)#value*Ihalo[halonid[nod][j]]*(perm/visc) 
            cmptparam =+1
        
        cmpt = cuda.atomic.add(s_cmpt, 0, 1)
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_rightglob
        value =  param3[i] / volume[c_left]
        a_loc[cmpt] = value*Icell[c_left]*(perm_visc) 

        # right cell------------------------------------------------------
        cmpt = cuda.atomic.add(s_cmpt, 0, 1)
        irn_loc[cmpt] = c_rightglob
        jcn_loc[cmpt] = c_leftglob
        value =  -1. * param1[i] / volume[c_right]
        a_loc[cmpt] = value*Icell[c_right]*(perm_visc)
    
        cmpt = cuda.atomic.add(s_cmpt, 0, 1)
        irn_loc[cmpt] = c_rightglob
        jcn_loc[cmpt] = c_rightglob
        value =  -1. * param3[i] / volume[c_right]
        a_loc[cmpt] = value*Icell[c_right]*(perm_visc) + (1/nbfR)*volume[c_right]*alpha_P*(1 - Icell[c_right])
      
      for idx in range(start, dirichletfaces.shape[0], stride):
        i = dirichletfaces[idx]

        nbfL = faceidc[cellfid[i][0]][-1]
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        perm = perm_vec[c_leftglob]
        visc = visc_vec[c_leftglob]

        cmpt = cuda.atomic.add(s_cmpt, 0, 1)
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = param1[i] / volume[c_left]
        a_loc[cmpt] = value*Icell[c_left]*(perm/visc) + (1/nbfL)*volume[c_left]*alpha_P*(1 - Icell[c_left])
        
        cmpt = cuda.atomic.add(s_cmpt, 0, 1)
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = -1. * param3[i] / volume[c_left]
        a_loc[cmpt] = value*Icell[c_left]*(perm/visc) + (1/nbfL)*volume[c_left]*alpha_P*(1 - Icell[c_left])
  
  kernel_get_triplet_2d_with_contrib = GPU_Backend.compile_kernel(kernel_get_triplet_2d_with_contrib)
  kernel_assign_int64 = cuda.jit('void(uint64[:], uint64)')(kernel_assign)
  
  def result(*args):
    VarClass.debug(kernel_get_triplet_2d_with_contrib, args)
    args = [VarClass.to_device(arg) for arg in args]

    kernel_assign_int64[1, 1, GPU_Backend.stream](d_s_cmpt, 0) #rhs
    GPU_Backend.stream.synchronize()
    
    size = max(len(args[32]), len(args[34])) #matrixinnerfaces, dirichletfaces
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_get_triplet_2d_with_contrib[nb_blocks, nb_threads, GPU_Backend.stream](*args, d_s_cmpt)
    GPU_Backend.stream.synchronize()

  return result    






                    
                  


