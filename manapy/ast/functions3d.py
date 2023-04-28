from numpy import zeros, fabs, int32, float, uint32
from manapy.ast.ast_utils import search_element

def cell_gradient_3d(w_c:'float[:]', w_ghost:'float[:]', w_halo:'float[:]', w_haloghost:'float[:]',
                     centerc:'float[:,:]', cellnid:'int32[:,:]', ghostnid:'int32[:,:]', haloghostnid:'int32[:,:]', halonid:'int32[:,:]',
                     nodecid:'uint32[:,:]', periodicn:'int32[:,:]', periodic:'int32[:,:]', centergf:'float[:,:]', 
                     halocenterg:'float[:,:]', vertexn:'float[:,:]', centerh:'float[:,:]', shift:'float[:,:]',
                     w_x:'float[:]', w_y:'float[:]', w_z:'float[:]'):

    nbelement = len(w_c)
    center = zeros(3)
    
    for i in range(nbelement):
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
            center[:] = centerc[cell][0:3]
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
            center[:] = halocenterg[cell]
            
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
        
def face_gradient_3d(w_c:'float[:]', w_ghost:'float[:]', w_halo:'float[:]', w_node:'float[:]', cellidf:'int32[:,:]', 
                     nodeidf:'int32[:,:]', centergf:'float[:,:]', halofid:'int32[:]', centerc:'float[:,:]', 
                     centerh:'float[:,:]', vertexn:'float[:,:]', airDiamond:'float[:]', normalf:'float[:,:]',
                     f_1:'float[:,:]', f_2:'float[:,:]', f_3:'float[:,:]', f_4:'float[:,:]', shift:'float[:,:]', 
                     wx_face:'float[:]', wy_face:'float[:]', wz_face:'float[:]', innerfaces:'uint32[:]', halofaces:'uint32[:]',
                     dirichletfaces:'uint32[:]', neumann:'uint32[:]', periodicfaces:'uint32[:]'):
        
    for i in innerfaces:
        
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
    
    for i in periodicfaces:
    
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
    
    for i in halofaces:
       
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
            
    for i in dirichletfaces:
       
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
        
        
    for i in neumann:
     
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


def centertovertex_3d(w_c:'float[:]', w_ghost:'float[:]', w_halo:'float[:]', w_haloghost:'float[:]',
                      centerc:'float[:,:]', centerh:'float[:,:]', cellid:'int32[:,:]', ghostid:'int32[:,:]', haloghostid:'int32[:,:]',
                      periodicid:'int32[:,:]',
                      haloid:'int32[:,:]', vertexn:'float[:,:]', centergf:'float[:,:]', halocenterg:'float[:,:]',
                      R_x:'float[:]', R_y:'float[:]', R_z:'float[:]', lambda_x:'float[:]',lambda_y:'float[:]', 
                      lambda_z:'float[:]', number:'uint32[:]', shift:'float[:,:]',  w_n:'float[:]'):

    w_n[:] = 0.
    nbnode = len(vertexn)
    center = zeros(3)
    
    for i in range(nbnode):
        
        for j in range(cellid[i][-1]):
            cell = cellid[i][j]
            center[:] = centerc[cell][:]
           
            xdiff = center[0] - vertexn[i][0]
            ydiff = center[1] - vertexn[i][1]
            zdiff = center[2] - vertexn[i][2]
            
            alpha = (1. + lambda_x[i]*xdiff + \
                     lambda_y[i]*ydiff + lambda_z[i]*zdiff)/ (number[i] + lambda_x[i]*R_x[i] + \
                                                              lambda_y[i]*R_y[i] + lambda_z[i]*R_z[i])
            w_n[i]  += alpha * w_c[cell]
        
        for j in range(ghostid[i][-1]):
            cell = ghostid[i][j]
            center[:] = centergf[cell][0:3]
           
            xdiff = center[0] - vertexn[i][0]
            ydiff = center[1] - vertexn[i][1]
            zdiff = center[2] - vertexn[i][2]
            
            alpha = (1. + lambda_x[i]*xdiff + \
                     lambda_y[i]*ydiff + lambda_z[i]*zdiff)/ (number[i] + lambda_x[i]*R_x[i] + \
                                                              lambda_y[i]*R_y[i] + lambda_z[i]*R_z[i])
            
            w_n[i]  += alpha * w_ghost[cell]
        

        for j in range(haloghostid[i][-1]):
            cell = haloghostid[i][j]
            center[:] = halocenterg[cell]
                  
            xdiff = center[0] - vertexn[i][0]
            ydiff = center[1] - vertexn[i][1]
            zdiff = center[2] - vertexn[i][2]
            
            alpha = (1. + lambda_x[i]*xdiff + \
                     lambda_y[i]*ydiff + lambda_z[i]*zdiff)/ (number[i] + lambda_x[i]*R_x[i] + \
                                                              lambda_y[i]*R_y[i] + lambda_z[i]*R_z[i])
                                                              
            w_n[i]  += alpha * w_haloghost[cell]   
        
        for j in range(haloid[i][-1]):
                cell = haloid[i][j]
                center[:] = centerh[cell][0:3]
              
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
                center[:] = centerc[cell][0:3] 
                
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
                center[:] = centerc[cell][0:3] 
                
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
                center[:] = centerc[cell][0:3] 
                
                xdiff = center[0] - vertexn[i][0]
                ydiff = center[1] - vertexn[i][1]
                zdiff = center[2] + shift[cell][2] - vertexn[i][2]
                
                alpha = (1. + lambda_x[i]*xdiff + \
                             lambda_y[i]*ydiff + lambda_z[i]*zdiff)/ (number[i] + lambda_x[i]*R_x[i] + \
                                                                      lambda_y[i]*R_y[i] + lambda_z[i]*R_z[i])
                w_n[i]  += alpha * w_c[cell]
                
def barthlimiter_3d(w_c:'float[:]', w_ghost:'float[:]', w_halo:'float[:]',
                    w_x:'float[:]', w_y:'float[:]',  w_z:'float[:]', psi:'float[:]', 
                    cellid:'int32[:,:]', faceid:'int32[:,:]', namef:'uint32[:]',
                    halofid:'int32[:]', centerc:'float[:,:]', centerf:'float[:,:]'):
   
    nbelement = len(w_c)
    psi[:] = 1.
    
    for i in range(nbelement):
        w_max = w_c[i]
        w_min = w_c[i]

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
            if fabs(delta2) < 1e-10:
                psi_ij = 1.
            else:
                if delta2 > 0.:
                    value = (w_max - w_c[i]) / delta2
                    psi_ij = min(1., value)
                if delta2 < 0.:
                    value = (w_min - w_c[i]) / delta2
                    psi_ij = min(1., value)

            psi[i] = min(psi[i], psi_ij)
            
def compute_3dmatrix_size(nodeidf:'int32[:,:]', halofid:'int32[:]', cellnid:'int32[:,:]',  halonid:'int32[:,:]', periodicnid:'int32[:,:]', 
                        centergn:'float[:,:,:]', halocentergn:'float[:,:,:]', oldnamen:'uint32[:]', BCdirichlet:'uint32[:]', 
                        matrixinnerfaces:'uint32[:]', halofaces:'uint32[:]', dirichletfaces:'uint32[:]'):                                                                                                                                                                       
  
    cmpt = 0
    nodes = zeros(4, dtype=int32)
    
    for i in matrixinnerfaces:
       
        nodes[0:3] = nodeidf[i][0:3]
        nodes[3]   = nodeidf[i][2]
        if nodeidf[i][-1] == 4:
            nodes[3] = nodeidf[i][3]
        
        cmpt = cmpt + 1
            
        for nod in nodes:
            if search_element(BCdirichlet, oldnamen[nod]) == 0: 
                for j in range(cellnid[nod][-1]):
                    
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    cmpt = cmpt + 1
                
                for j in range(len(centergn[nod])):
                    if centergn[nod][j][-1] != -1:
                       
                        cmpt = cmpt + 1
                        #right cell-----------------------------------                                                                                              
                        cmpt = cmpt + 1
                        
                for j in range(len(halocentergn[nod])):
                    if halocentergn[nod][j][-1] != -1:
                        
                        cmpt = cmpt + 1
                        #right cell-----------------------------------                                                                                              
                        cmpt = cmpt + 1
                    
                for j in range(periodicnid[nod][-1]):
                        
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    cmpt = cmpt + 1
                
                for j in range(halonid[nod][-1]):
                   
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    cmpt = cmpt + 1
        
        cmpt = cmpt + 1
        # right cell------------------------------------------------------
        cmpt = cmpt + 1
        cmpt = cmpt + 1
            
    for i in halofaces:
        
        nodes[0:3] = nodeidf[i][0:3]
        nodes[3]   = nodeidf[i][2]
        if nodeidf[i][-1] == 4:
            nodes[3] = nodeidf[i][3]
        
        cmpt = cmpt + 1
        cmpt = cmpt + 1
        
        for nod in nodes:
            if search_element(BCdirichlet, oldnamen[nod]) == 0: 
                for j in range(cellnid[nod][-1]):
                    cmpt = cmpt + 1
                    
                for j in range(len(centergn[nod])):
                    if centergn[nod][j][-1] != -1:
                        cmpt = cmpt + 1
                        
                for j in range(len(halocentergn[nod])):
                    if halocentergn[nod][j][-1] != -1:
                        cmpt = cmpt + 1

                for j in range(halonid[nod][-1]):
                    cmpt = cmpt + 1
                    
    for i in dirichletfaces:
        cmpt = cmpt + 1
        cmpt = cmpt + 1
            
    return cmpt


def get_triplet_3d(cellfid:'int32[:,:]', nodeidf:'int32[:,:]', vertexn:'float[:,:]', halofid:'int32[:]',
                   haloext:'int32[:,:]', oldnamen:'uint32[:]', volume:'float[:]', 
                   cellnid:'int32[:,:]', centerc:'float[:,:]', centerh:'float[:,:]', halonid:'int32[:,:]', periodicnid:'int32[:,:]', 
                   centergn:'float[:,:,:]', halocentergn:'float[:,:,:]', airDiamond:'float[:]', 
                   lambda_x:'float[:]', lambda_y:'float[:]', lambda_z:'float[:]', number:'uint32[:]', R_x:'float[:]', R_y:'float[:]', 
                   R_z:'float[:]', param1:'float[:]', param2:'float[:]', param3:'float[:]', param4:'float[:]', shift:'float[:,:]',
                   nbelements:'intc', loctoglob:'int32[:]', BCdirichlet:'uint32[:]', a_loc:'float[:]', irn_loc:'int32[:]', jcn_loc:'int32[:]',
                   matrixinnerfaces:'uint32[:]', halofaces:'uint32[:]', dirichletfaces:'uint32[:]'):

    parameters = zeros(4)
    nodes = zeros(4, dtype=int32)
    
    cmpt = 0
    
    for i in matrixinnerfaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        nodes[0:3] = nodeidf[i][0:3]
        nodes[3]   = nodeidf[i][2]
        if nodeidf[i][-1] == 4:
            nodes[3] = nodeidf[i][3]

        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
        
        c_right = cellfid[i][1]
        c_rightglob = loctoglob[c_right]
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = -1 * param3[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
        
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
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    value = -1. * alpha / volume[c_right] * parameters[cmptparam]
                    irn_loc[cmpt] = c_rightglob
                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                
                for j in range(halonid[nod][-1]):
                    center = centerh[halonid[nod][j]]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    zdiff = center[2] - vertexn[nod][2]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff + lambda_z[nod]*zdiff)/ (number[nod] + lambda_x[nod]*R_x[nod] + \
                                                                          lambda_y[nod]*R_y[nod] + lambda_z[nod]*R_z[nod])
                    value = alpha / volume[c_left] * parameters[cmptparam]
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    value = -1. * alpha / volume[c_right] * parameters[cmptparam]
                    irn_loc[cmpt] = c_rightglob
                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                    
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
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = loctoglob[periodicnid[nod][j]]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    value =  -1. * alpha / volume[c_right] * parameters[cmptparam]
                    irn_loc[cmpt] = c_rightglob
                    jcn_loc[cmpt] = loctoglob[periodicnid[nod][j]]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                    
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
                        
                        index = int32(centergn[nod][j][3])
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = loctoglob[index]
                        a_loc[cmpt] = value
                        cmpt = cmpt + 1
                        #right cell-----------------------------------                                                                                              
                        value = -1. * alpha / volume[c_right] * parameters[cmptparam]
                        irn_loc[cmpt] = c_rightglob
                        jcn_loc[cmpt] = loctoglob[index]
                        a_loc[cmpt] = value
                        cmpt = cmpt + 1
                    
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
                        index = int32(halocentergn[nod][j][3])
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = haloext[index][0]
                        a_loc[cmpt] = value
                        cmpt = cmpt + 1
                        #right cell-----------------------------------                                                                                              
                        value = -1. * alpha / volume[c_right] * parameters[cmptparam]
                        irn_loc[cmpt] = c_rightglob
                        jcn_loc[cmpt] = haloext[index][0]
                        a_loc[cmpt] = value
                        cmpt = cmpt + 1
            cmptparam = cmptparam +1
           
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_rightglob
        value = param3[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1

        # right cell------------------------------------------------------
        irn_loc[cmpt] = c_rightglob
        jcn_loc[cmpt] = c_leftglob
        value = param3[i] / volume[c_right]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
    
        irn_loc[cmpt] = c_rightglob
        jcn_loc[cmpt] = c_rightglob
        value = -1. * param3[i] / volume[c_right]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
        
    for i in halofaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        nodes[0:3] = nodeidf[i][0:3]
        nodes[3]   = nodeidf[i][2]
        if nodeidf[i][-1] == 4:
            nodes[3] = nodeidf[i][3]

        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
        
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
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                
                for j in range(halonid[nod][-1]):
                    center = centerh[halonid[nod][j]]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    zdiff = center[2] - vertexn[nod][2]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff + lambda_z[nod]*zdiff)/ (number[nod] + lambda_x[nod]*R_x[nod] + \
                                                                          lambda_y[nod]*R_y[nod] + lambda_z[nod]*R_z[nod])
                    value = alpha / volume[c_left] * parameters[cmptparam]
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                    
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
                        
                        index = int32(centergn[nod][j][3])
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = loctoglob[index]
                        a_loc[cmpt] = value
                        cmpt = cmpt + 1
                    
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
                        index = int32(halocentergn[nod][j][3])
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = haloext[index][0]
                        a_loc[cmpt] = value
                        cmpt = cmpt + 1
            cmptparam = cmptparam +1
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = -1 * param3[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_rightglob
        value = param3[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
            
    for i in dirichletfaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = -1 * param3[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = -1. * param3[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
  

def get_rhs_loc_3d(cellfid:'int32[:,:]', nodeidf:'int32[:,:]', oldname:'uint32[:]', 
                    volume:'float[:]', centergn:'float[:,:,:]', loctoglob:'int32[:]', param1:'float[:]', param2:'float[:]', 
                    param3:'float[:]', param4:'float[:]', Pbordnode:'float[:]', Pbordface:'float[:]', rhs_loc:'float[:]', 
                    BCdirichlet:'uint32[:]', centergf:'float[:,:]', matrixinnerfaces:'uint32[:]',
                    halofaces:'uint32[:]', dirichletfaces:'uint32[:]'):
      
    rhs_loc[:] = 0.
    parameters = zeros(4)
    nodes = zeros(4, dtype=int32)

    for i in matrixinnerfaces:
    
        c_left = cellfid[i][0]
        c_right = cellfid[i][1]
        
        nodes[0:3] = nodeidf[i][0:3]
        nodes[3]   = nodeidf[i][2]
        if nodeidf[i][-1] == 4:
            nodes[3] = nodeidf[i][3]

        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
        
        cmpt = 0
        for nod in nodes:
            if search_element(BCdirichlet, oldname[nod]) == 1: 
                V = Pbordnode[nod]
                value_left = -1. * V * parameters[cmpt] / volume[c_left]
                rhs_loc[c_left] += value_left
                
                value_right = V * parameters[cmpt] / volume[c_right]
                rhs_loc[c_right] += value_right

            cmpt = cmpt +1

    for i in halofaces:
        
        c_left = cellfid[i][0]
        
        nodes[0:3] = nodeidf[i][0:3]
        nodes[3]   = nodeidf[i][2]
        if nodeidf[i][-1] == 4:
            nodes[3] = nodeidf[i][3]
        
        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
    
        cmpt = 0
        for nod in nodes:
            if search_element(BCdirichlet, oldname[nod]) == 1:
                V = Pbordnode[nod]
                value_left = -1. * V * parameters[cmpt] / volume[c_left]
                rhs_loc[c_left] += value_left
            cmpt = cmpt +1
    
    for i in dirichletfaces:
        c_left = cellfid[i][0]
        
        nodes[0:3] = nodeidf[i][0:3]
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
                rhs_loc[c_left] += value_left
            
            cmpt +=1
            
        V_K = Pbordface[i]
        value = -2. * param3[i] / volume[c_left] * V_K
        rhs_loc[c_left] += value

def get_rhs_glob_3d(cellfid:'int32[:,:]', nodeidf:'int32[:,:]', oldname:'uint32[:]', 
                    volume:'float[:]', centergn:'float[:,:,:]', loctoglob:'int32[:]', param1:'float[:]', param2:'float[:]', 
                    param3:'float[:]', param4:'float[:]', Pbordnode:'float[:]', Pbordface:'float[:]',  rhs:'float[:]',
                    BCdirichlet:'uint32[:]', centergf:'float[:,:]', matrixinnerfaces:'uint32[:]',
                    halofaces:'uint32[:]', dirichletfaces:'uint32[:]'):                                                                                                                                                                       

    parameters = zeros(4)
    nodes = zeros(4, dtype=int32)
    
    for i in matrixinnerfaces:
        
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        nodes[0:3] = nodeidf[i][0:3]
        nodes[3]   = nodeidf[i][2]
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
                rhs[c_leftglob] += value_left
                
                value_right = V * parameters[cmpt] / volume[c_right]
                rhs[c_rightglob] += value_right

            cmpt = cmpt +1
    
    for i in halofaces:
        
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        nodes[0:3] = nodeidf[i][0:3]
        nodes[3]   = nodeidf[i][2]
        if nodeidf[i][-1] == 4:
            nodes[3] = nodeidf[i][3]
        
        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
        
        cmpt = 0
        for nod in nodes:
            if search_element(BCdirichlet, oldname[nod]) == 1: 
                V = Pbordnode[nod]
                value_left = -1. * V * parameters[cmpt] / volume[c_left]
                rhs[c_leftglob] += value_left
            cmpt = cmpt +1
    
    for i in dirichletfaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        nodes[0:3] = nodeidf[i][0:3]
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
                rhs[c_leftglob] += value_left
                
            cmpt = cmpt +1
            
        V_K = Pbordface[i]
        value = -2. * param3[i] / volume[c_left] * V_K
        rhs[c_leftglob] += value


def compute_P_gradient_3d_diamond(val_c:'float[:]', v_ghost:'float[:]', v_halo:'float[:]', v_node:'float[:]', cellidf:'int32[:,:]', 
                                  nodeidf:'int32[:,:]', centergf:'float[:,:]', halofid:'int32[:]', centerc:'float[:,:]', 
                                  centerh:'float[:,:]', oldname:'uint32[:]', airDiamond:'float[:]', n1:'float[:,:]', n2:'float[:,:]',
                                  n3:'float[:,:]', n4:'float[:,:]', normalf:'float[:,:]', shift:'float[:,:]', Pbordnode:'float[:]',
                                  Pbordface:'float[:]', 
                                  Px_face:'float[:]', Py_face:'float[:]', Pz_face:'float[:]', BCdirichlet:'uint32[:]', innerfaces:'uint32[:]',
                                  halofaces:'uint32[:]', neumannfaces:'uint32[:]', dirichletfaces:'uint32[:]', periodicfaces:'uint32[:]'):
 
    for i in innerfaces:
       
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
    
    for i in periodicfaces:
        
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
    
    for i in neumannfaces:
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
        
    for i in halofaces:
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
            
    for i in dirichletfaces:   
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
