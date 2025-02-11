from numpy import  int32, uint32
import numpy as np
#from manapy.ast.ast_utils import search_element

def cell_gradient_2d(w_c:'float[:]', w_ghost:'float[:]', w_halo:'float[:]', w_haloghost:'float[:]',
                     centerc:'float[:,:]', cellnid:'int32[:,:]', ghostnid:'int32[:,:]', haloghostnid:'int32[:,:]', halonid:'int32[:,:]',
                     nodecid:'uint32[:,:]', periodicn:'int32[:,:]', periodic:'int32[:,:]', centergf:'float[:,:]', 
                     halocenterg:'float[:,:]', vertexn:'float[:,:]', centerh:'float[:,:]', shift:'float[:,:]',
                     w_x:'float[:]', w_y:'float[:]', w_z:'float[:]'):
    
    center = np.zeros(3)
    nbelement = len(w_c)
    
    for i in range(nbelement):
        i_xx  = 0.;  i_yy  = 0.; i_xy = 0.
        j_xw = 0.;  j_yw = 0.
        
        for j in range(cellnid[i][-1]):
            cell = cellnid[i][j]
            j_x = centerc[cell][0] - centerc[i][0]
            j_y = centerc[cell][1] - centerc[i][1]
            i_xx += j_x*j_x
            i_yy += j_y*j_y
            i_xy += (j_x * j_y)

            j_xw += (j_x * (w_c[cell] - w_c[i] ))
            j_yw += (j_y * (w_c[cell] - w_c[i] ))
            
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
                    center[:] = centerc[cell][0:3]
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
                    center[:] = centerc[cell][0:3]
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
            center[:] = halocenterg[cell]

            j_x = center[0] - centerc[i][0]
            j_y = center[1] - centerc[i][1]
            
            i_xx += j_x*j_x
            i_yy += j_y*j_y
            i_xy += (j_x * j_y)

            j_xw += (j_x * (w_haloghost[cell] - w_c[i] ))
            j_yw += (j_y * (w_haloghost[cell] - w_c[i] ))
            
            
            
        dia = i_xx * i_yy - i_xy*i_xy

        w_x[i]  = (i_yy * j_xw - i_xy * j_yw) / dia
        w_y[i]  = (i_xx * j_yw - i_xy * j_xw) / dia
        w_z[i]  = 0.



def face_gradient_2d(w_c:'float[:]', w_ghost:'float[:]', w_halo:'float[:]', w_node:'float[:]', cellidf:'int32[:,:]', 
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
        
        vi1 = w_node[i_1]
        vi2 = w_node[i_2]
        vv1 = w_c[c_left]
        vv2 = w_c[c_right]
        
        wx_face[i] = 1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
        wy_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])
    
    for i in periodicfaces:
     
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
    
    for i in halofaces:
       
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
    
    
    for i in dirichletfaces:
       
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
    
    
    for i in neumann:
     
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


def centertovertex_2d(w_c:'float[:]', w_ghost:'float[:]', w_halo:'float[:]', w_haloghost:'float[:]',
                      centerc:'float[:,:]', centerh:'float[:,:]', cellid:'int32[:,:]', ghostid:'int32[:,:]', haloghostid:'int32[:,:]',
                      periodicid:'int32[:,:]',
                      haloid:'int32[:,:]', vertexn:'float[:,:]', centergf:'float[:,:]', halocenterg:'float[:,:]',
                      R_x:'float[:]', R_y:'float[:]', R_z:'float[:]', lambda_x:'float[:]',lambda_y:'float[:]', 
                      lambda_z:'float[:]', number:'uint32[:]', shift:'float[:,:]',  w_n:'float[:]'):
   

    w_n[:] = 0.
    
    nbnode = len(vertexn)
    center = np.zeros(3)
    
    for i in range(nbnode):
        for j in range(cellid[i][-1]):
            cell = cellid[i][j]
            center[:] = centerc[cell][:]
           
            xdiff = center[0] - vertexn[i][0]
            ydiff = center[1] - vertexn[i][1]
            alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
            
            w_n[i]  += alpha * w_c[cell]
            
        for j in range(ghostid[i][-1]):
            cell = ghostid[i][j]
            center[:] = centergf[cell][0:3]
           
            xdiff = center[0] - vertexn[i][0]
            ydiff = center[1] - vertexn[i][1]
            alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
            
            w_n[i]  += alpha * w_ghost[cell]
            
            
        for j in range(haloghostid[i][-1]):
            cell = haloghostid[i][j]
            center[:] = halocenterg[cell][0:3]
          
            xdiff = center[0] - vertexn[i][0]
            ydiff = center[1] - vertexn[i][1]
            
            alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
            
            w_n[i]  += alpha * w_haloghost[cell]
            
        for j in range(haloid[i][-1]):
                cell = haloid[i][j]
                center[:] = centerh[cell][0:3]
              
                xdiff = center[0] - vertexn[i][0]
                ydiff = center[1] - vertexn[i][1]
                alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
             
                w_n[i]  += alpha * w_halo[cell]
                
        #TODO Must be keeped like that checked ok ;)
        if vertexn[i][3] == 11 or vertexn[i][3] == 22 :
            for j in range(periodicid[i][-1]):
                cell = periodicid[i][j]
                center[:] = centerc[cell][0:3] 
                
                xdiff = center[0] + shift[cell][0] - vertexn[i][0]
                ydiff = center[1] - vertexn[i][1]
                alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
                
                w_n[i]  += alpha * w_c[cell]
                    
        elif vertexn[i][3] == 33 or vertexn[i][3] ==44:
            for j in range(periodicid[i][-1]):
                cell = periodicid[i][j]
                center[:] = centerc[cell][0:3] 
                
                xdiff = center[0] - vertexn[i][0]
                ydiff = center[1] + shift[cell][1] - vertexn[i][1]
                alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
                
                w_n[i]  += alpha * w_c[cell]
                    


def barthlimiter_2d(w_c:'float[:]', w_ghost:'float[:]', w_halo:'float[:]',
                    w_x:'float[:]', w_y:'float[:]',  w_z:'float[:]', psi:'float[:]', 
                    cellid:'int32[:,:]', faceid:'int32[:,:]', namef:'uint32[:]',
                    halofid:'int32[:]', centerc:'float[:,:]', centerf:'float[:,:]'):
    
    nbelement = len(w_c)
    val  = 1.
    psi[:] = val

    for i in range(nbelement):
        w_max = w_c[i]
        w_min = w_c[i]

        for j in range(faceid[i][-1]):
            face = faceid[i][j]
            if namef[face] == 0 or namef[face] > 10:#
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
            if np.fabs(delta2) < 1e-8:
                psi_ij = 1.
            else:
                if delta2 > 0.:
                    value = (w_max - w_c[i]) / delta2
                    psi_ij = min(val, value)
                if delta2 < 0.:
                    value = (w_min - w_c[i]) / delta2
                    psi_ij = min(val, value)

            psi[i] = min(psi[i], psi_ij)
          
            
def get_triplet_2d(cellfid:'int32[:,:]', nodeidf:'int32[:,:]', vertexn:'float[:,:]', halofid:'int32[:]',
                   haloext:'int32[:,:]', oldnamen:'uint32[:]', volume:'float[:]', 
                   cellnid:'int32[:,:]', centerc:'float[:,:]', centerh:'float[:,:]', halonid:'int32[:,:]', periodicnid:'int32[:,:]', 
                   centergn:'float[:,:,:]', halocentergn:'float[:,:,:]', airDiamond:'float[:]', 
                   lambda_x:'float[:]', lambda_y:'float[:]', lambda_z:'float[:]', number:'uint32[:]', R_x:'float[:]', R_y:'float[:]', 
                   R_z:'float[:]', param1:'float[:]', param2:'float[:]', param3:'float[:]', param4:'float[:]', shift:'float[:,:]',
                   nbelements:'int32', loctoglob:'int32[:]', BCdirichlet:'uint32[:]', a_loc:'float[:]', irn_loc:'int32[:]', jcn_loc:'int32[:]',
                   matrixinnerfaces:'uint32[:]', halofaces:'uint32[:]', dirichletfaces:'uint32[:]'):                                                                                                                                                                       
    
    
    def search_element(a:'int32[:]', target_value:'int32'):
        find = 0
        for val in a:
            if val == target_value:
                find = 1
                break
        return find
    center = np.zeros(2)
    parameters = np.zeros(2)
    cmpt = 0

    for i in matrixinnerfaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        parameters[0] = param4[i]; parameters[1] = param2[i]
    
        c_right = cellfid[i][1]
        c_rightglob = loctoglob[c_right]
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = param1[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
        
        cmptparam = 0
        for nod in nodeidf[i][:nodeidf[i][-1]]:
            if search_element(BCdirichlet, oldnamen[nod]) == 0: 
                for j in range(cellnid[nod][-1]):
                    center[:] = centerc[cellnid[nod][j]][0:2]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                    value =  alpha / volume[c_left] * parameters[cmptparam]
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    value =  -1. * alpha / volume[c_right] * parameters[cmptparam]
                    irn_loc[cmpt] = c_rightglob
                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                
                for j in range(len(centergn[nod])):
                    if centergn[nod][j][-1] != -1:
                        center[:] = centergn[nod][j][0:2]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                        index = np.int32(centergn[nod][j][2])
                        value = alpha / volume[c_left] * parameters[cmptparam]
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
                        center[:] = halocentergn[nod][j][0:2]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                        index = np.int32(halocentergn[nod][j][2])
                        value = alpha / volume[c_left] * parameters[cmptparam]
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
                
                for j in range(halonid[nod][-1]):
                    center[:] = centerh[halonid[nod][j]][0:2]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                    value =  alpha / volume[c_left] * parameters[cmptparam]
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    value =  -1. * alpha / volume[c_right] * parameters[cmptparam]
                    irn_loc[cmpt] = c_rightglob
                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
            cmptparam =+1
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_rightglob
        value =  param3[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1

        # right cell------------------------------------------------------
        irn_loc[cmpt] = c_rightglob
        jcn_loc[cmpt] = c_leftglob
        value =  -1. * param1[i] / volume[c_right]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
    
        irn_loc[cmpt] = c_rightglob
        jcn_loc[cmpt] = c_rightglob
        value =  -1. * param3[i] / volume[c_right]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
    
    for i in halofaces:
        
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        parameters[0] = param4[i]; parameters[1] = param2[i]
        
        c_rightglob = haloext[halofid[i]][0]
        c_right     = halofid[i]
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value =  param1[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1

        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_rightglob
        value =  param3[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
        
        cmptparam = 0
        for nod in nodeidf[i][:nodeidf[i][-1]]:
            if search_element(BCdirichlet, oldnamen[nod]) == 0: 
                for j in range(cellnid[nod][-1]):
                    center[:] = centerc[cellnid[nod][j]][0:2]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                    value =  alpha / volume[c_left] * parameters[cmptparam]
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                    
                for j in range(len(centergn[nod])):
                    if centergn[nod][j][-1] != -1:
                        center[:] = centergn[nod][j][0:2]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                        index = np.int32(centergn[nod][j][2])
                        value = alpha / volume[c_left] * parameters[cmptparam]
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = loctoglob[index]
                        a_loc[cmpt] = value
                        cmpt = cmpt + 1
                        
                for j in range(len(halocentergn[nod])):
                    if halocentergn[nod][j][-1] != -1:
                        center[:] = halocentergn[nod][j][0:2]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                        index = np.int32(halocentergn[nod][j][2])
                        value = alpha / volume[c_left] * parameters[cmptparam]
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = haloext[index][0]
                        a_loc[cmpt] = value
                        cmpt = cmpt + 1

                for j in range(halonid[nod][-1]):
                    center[:] = centerh[halonid[nod][j]][0:2]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                    value =  alpha / volume[c_left] * parameters[cmptparam]
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
            cmptparam +=1
            
    for i in dirichletfaces:
        
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = param1[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = -1. * param3[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1

def compute_2dmatrix_size(nodeidf:'int32[:,:]', halofid:'int32[:]', cellnid:'int32[:,:]',  halonid:'int32[:,:]', periodicnid:'int32[:,:]', 
                        centergn:'float[:,:,:]', halocentergn:'float[:,:,:]', oldnamen:'uint32[:]', BCdirichlet:'uint32[:]', 
                        matrixinnerfaces:'uint32[:]', halofaces:'uint32[:]', 
                        dirichletfaces:'uint32[:]'):                                                                                                                                                                       
    
    def search_element(a:'int32[:]', target_value:'int32'):
        find = 0
        for val in a:
            if val == target_value:
                find = 1
                break
        return find
    cmpt = 0
    for i in matrixinnerfaces:
        cmpt = cmpt + 1
        
        for nod in nodeidf[i][:nodeidf[i][-1]]:
            if search_element(BCdirichlet, oldnamen[nod]) == 0:# and search_element(BCneumannNH, oldnamen[nod]) == 0:
            # if vertexn[nod][3] not in BCdirichlet:
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
            
    # elif namef[i] == 10:
    for i in halofaces:
        cmpt = cmpt + 1
        
        cmpt = cmpt + 1
        cmpt = cmpt + 1
        
        for nod in nodeidf[i][:nodeidf[i][-1]]:
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

def get_rhs_loc_2d(cellfid:'int32[:,:]', nodeidf:'int32[:,:]', oldname:'uint32[:]', 
                    volume:'float[:]', centergn:'float[:,:,:]', loctoglob:'int32[:]', param1:'float[:]', param2:'float[:]', 
                    param3:'float[:]', param4:'float[:]', Pbordnode:'float[:]', Pbordface:'float[:]', rhs_loc:'float[:]', 
                    BCdirichlet:'uint32[:]', centergf:'float[:,:]', matrixinnerfaces:'uint32[:]',
                    halofaces:'uint32[:]', dirichletfaces:'uint32[:]'):                                                                                                                                                                       
    
    def search_element(a:'int32[:]', target_value:'int32'):
        find = 0
        for val in a:
            if val == target_value:
                find = 1
                break
        return find
    
    rhs_loc[:] = 0.
    for i in matrixinnerfaces:
        c_right = cellfid[i][1]
        c_left = cellfid[i][0]
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]    
        
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            V = Pbordnode[i_1]
            value_left = -1. * V * param4[i] / volume[c_left]
            rhs_loc[c_left] +=  value_left
            
            value_right = V * param4[i] / volume[c_right]
            rhs_loc[c_right] += value_right
            
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            V = Pbordnode[i_2]
            value_left =  -1. * V * param2[i] / volume[c_left]
            rhs_loc[c_left] += value_left
            
            value_right =  V * param2[i] / volume[c_right]
            rhs_loc[c_right] += value_right
                
    for i in halofaces:
        c_left = cellfid[i][0]
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]   
        
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            V = Pbordnode[i_1]
            value_left =  -1. * V * param4[i] / volume[c_left]
            rhs_loc[c_left] += value_left
            
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            V = Pbordnode[i_2]
            value_left =  -1. * V * param2[i] / volume[c_left]
            rhs_loc[c_left] += value_left
            
    # TODO verify
    for i in dirichletfaces:
        
        c_left = cellfid[i][0]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]   
        
        if centergn[i_1][0][2] != -1:     
            V = Pbordnode[i_1]
            value_left = -1. * V * param4[i] / volume[c_left]
            rhs_loc[c_left] += value_left
           
        if centergn[i_2][0][2] != -1: 
            V = Pbordnode[i_2]
            value_left = -1. * V * param2[i] / volume[c_left]
            rhs_loc[c_left] += value_left
        
        V_K = Pbordface[i]
        value = -2. * param3[i] / volume[c_left] * V_K
        rhs_loc[c_left] += value
       
def get_rhs_glob_2d(cellfid:'int32[:,:]', nodeidf:'int32[:,:]', oldname:'uint32[:]', 
                    volume:'float[:]', centergn:'float[:,:,:]', loctoglob:'int32[:]', param1:'float[:]', param2:'float[:]', 
                    param3:'float[:]', param4:'float[:]', Pbordnode:'float[:]', Pbordface:'float[:]',  rhs:'float[:]',
                    BCdirichlet:'uint32[:]', centergf:'float[:,:]', matrixinnerfaces:'uint32[:]',
                    halofaces:'uint32[:]', dirichletfaces:'uint32[:]'):                                                                                                                                                                       
    
    def search_element(a:'int32[:]', target_value:'int32'):
        find = 0
        for val in a:
            if val == target_value:
                find = 1
                break
        return find
    
    rhs[:] = 0.
    for i in matrixinnerfaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]    
        
        c_right = cellfid[i][1]
        c_rightglob = loctoglob[c_right]
        
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            V = Pbordnode[i_1]
            value_left = -1. * V * param4[i] / volume[c_left]
            rhs[c_leftglob] +=  value_left
            
            value_right = V * param4[i] / volume[c_right]
            rhs[c_rightglob] += value_right
            
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            V = Pbordnode[i_2]
            value_left =  -1. * V * param2[i] / volume[c_left]
            rhs[c_leftglob] += value_left
            
            value_right =  V * param2[i] / volume[c_right]
            rhs[c_rightglob] += value_right
                    
    for i in halofaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]    
        
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            V = Pbordnode[i_1]
            value_left =  -1. * V * param4[i] / volume[c_left]
            rhs[c_leftglob] += value_left
        
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            V = Pbordnode[i_2]
            value_left =  -1. * V * param2[i] / volume[c_left]
            rhs[c_leftglob] += value_left
            
    for i in dirichletfaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]  
        
        if centergn[i_1][0][2] != -1:     
            V = Pbordnode[i_1]
            value_left = -1. * V * param4[i] / volume[c_left]
            rhs[c_leftglob] += value_left
           
        if centergn[i_2][0][2] != -1: 
            V = Pbordnode[i_2]
            value_left = -1. * V * param2[i] / volume[c_left]
            rhs[c_leftglob] += value_left
        
        V_K = Pbordface[i]
        value = -2. * param3[i] / volume[c_left] * V_K
        rhs[c_leftglob] += value


def compute_P_gradient_2d_diamond(P_c:'float[:]', P_ghost:'float[:]', P_halo:'float[:]', P_node:'float[:]', cellidf:'int32[:,:]', 
                                  nodeidf:'int32[:,:]', centergf:'float[:,:]',  halofid:'int32[:]', centerc:'float[:,:]', 
                                  centerh:'float[:,:]', oldname:'uint32[:]', airDiamond:'float[:]', f_1:'float[:,:]', f_2:'float[:,:]',
                                  f_3:'float[:,:]', f_4:'float[:,:]', normalf:'float[:,:]', shift:'float[:,:]', Pbordnode:'float[:]', 
                                  Pbordface:'float[:]', 
                                  Px_face:'float[:]', Py_face:'float[:]', Pz_face:'float[:]', BCdirichlet:'uint32[:]', innerfaces:'uint32[:]',
                                  halofaces:'uint32[:]', neumannfaces:'uint32[:]', dirichletfaces:'uint32[:]', periodicfaces:'uint32[:]'):
    
    def search_element(a:'int32[:]', target_value:'int32'):
        find = 0
        for val in a:
            if val == target_value:
                find = 1
                break
        return find
    
    for i in innerfaces:
        
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
     
    for i in periodicfaces:
        
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
    
    for i in neumannfaces:
        
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
            
    for i in halofaces:

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
    
    for i in dirichletfaces:
        
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


def compute_P_gradient_2d_FV4():
    pass

        
#def get_triplet_2d_with_contrib(cellfid:'int32[:,:]', nodeidf:'int32[:,:]', faceidc:'int32[:,:]', vertexn:'float[:,:]', halofid:'int32[:]',
#                                haloext:'int32[:,:]', oldnamen:'uint32[:]', volume:'float[:]', 
#                                cellnid:'int32[:,:]', centerc:'float[:,:]', centerh:'float[:,:]', halonid:'int32[:,:]', 
#                                periodicnid:'int32[:,:]', 
#                                centergn:'float[:,:,:]', halocentergn:'float[:,:,:]', airDiamond:'float[:]', 
#                                lambda_x:'float[:]', lambda_y:'float[:]', number:'uint32[:]', R_x:'float[:]', 
#                                R_y:'float[:]', param1:'float[:]', 
#                                param2:'float[:]', param3:'float[:]', param4:'float[:]', shift:'float[:,:]', 
#                                nbelements:'int32', loctoglob:'int32[:]',
#                                BCdirichlet:'uint32[:]', a_loc:'float[:]', irn_loc:'int32[:]', jcn_loc:'int32[:]',
#                                matrixinnerfaces:'uint32[:]', halofaces:'uint32[:]', dirichletfaces:'uint32[:]',
#                                Icell:'float[:]', #Ihalo:'float[:]', Ihaloghost:'float[:]',
#                                alpha_P:'float', perm:'float', 
#                                visc:'float', BCneumannNH:'uint32[:]'):
#    
#    def search_element(a:'int32[:]', target_value:'int32'):
#        find = 0
#        for val in a:
#            if val == target_value:
#                find = 1
#                break
#        return find
#    
#    center = np.zeros(2)
#    parameters = np.zeros(2)
#    cmpt = 0
#
#    for i in matrixinnerfaces:
#        nbfL = faceidc[cellfid[i][0]][-1]
#        nbfR = faceidc[cellfid[i][1]][-1]
#        c_left = cellfid[i][0]
#        c_leftglob  = loctoglob[c_left]
#        
#        parameters[0] = param4[i]; parameters[1] = param2[i]
#    
#        c_right = cellfid[i][1]
#        c_rightglob = loctoglob[c_right]
#        
#        irn_loc[cmpt] = c_leftglob
#        jcn_loc[cmpt] = c_leftglob
#        value = param1[i] / volume[c_left]
#        a_loc[cmpt] = value*Icell[c_left]*(perm/visc) + (1/nbfL)*volume[c_left]*alpha_P*(1 - Icell[c_left])
#        cmpt = cmpt + 1
#        
#        cmptparam = 0
#        for nod in nodeidf[i][:nodeidf[i][-1]]:
#            if search_element(BCdirichlet, oldnamen[nod]) == 0:# and search_element(BCneumannNH, oldnamen[nod]) == 0:
#                for j in range(cellnid[nod][-1]):
#                    center[:] = centerc[cellnid[nod][j]][0:2]
#                    xdiff = center[0] - vertexn[nod][0]
#                    ydiff = center[1] - vertexn[nod][1]
#                    alpha = (1. + lambda_x[nod]*xdiff + \
#                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
#                    value =  alpha / volume[c_left] * parameters[cmptparam]
#                    irn_loc[cmpt] = c_leftglob
#                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
#                    a_loc[cmpt] = value*Icell[c_left]*(perm/visc) 
#                    cmpt = cmpt + 1
#                    #right cell-----------------------------------                                                                                              
#                    value =  -1. * alpha / volume[c_right] * parameters[cmptparam]
#                    irn_loc[cmpt] = c_rightglob
#                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
#                    a_loc[cmpt] = value*Icell[c_right]*(perm/visc) 
#                    cmpt = cmpt + 1
#                
#                for j in range(len(centergn[nod])):
#                    if centergn[nod][j][-1] != -1:
#                        center[:] = centergn[nod][j][0:2]
#                        xdiff = center[0] - vertexn[nod][0]
#                        ydiff = center[1] - vertexn[nod][1]
#                        alpha = (1. + lambda_x[nod]*xdiff + \
#                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
#                        index = np.int32(centergn[nod][j][2])
#                        value = alpha / volume[c_left] * parameters[cmptparam]
#                        irn_loc[cmpt] = c_leftglob
#                        jcn_loc[cmpt] = loctoglob[index]
#                        a_loc[cmpt] = value*Icell[c_left]*(perm/visc) 
#                        cmpt = cmpt + 1
#                        #right cell-----------------------------------                                                                                              
#                        value = -1. * alpha / volume[c_right] * parameters[cmptparam]
#                        irn_loc[cmpt] = c_rightglob
#                        jcn_loc[cmpt] = loctoglob[index]
#                        a_loc[cmpt] = value*Icell[c_right]*(perm/visc) 
#                        cmpt = cmpt + 1
#                        
#                for j in range(len(halocentergn[nod])):
#                    if halocentergn[nod][j][-1] != -1:
#                        center[:] = halocentergn[nod][j][0:2]
#                        xdiff = center[0] - vertexn[nod][0]
#                        ydiff = center[1] - vertexn[nod][1]
#                        alpha = (1. + lambda_x[nod]*xdiff + \
#                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
#                        index = np.int32(halocentergn[nod][j][2])
##                        cell  = np.int32(halocentergn[nod][j][-1])
#                        value = alpha / volume[c_left] * parameters[cmptparam]
#                        irn_loc[cmpt] = c_leftglob
#                        jcn_loc[cmpt] = haloext[index][0]
#                        a_loc[cmpt] = value*Icell[c_left]*(perm/visc) 
#                        cmpt = cmpt + 1
#                        #right cell-----------------------------------                                                                                              
#                        value = -1. * alpha / volume[c_right] * parameters[cmptparam]
#                        irn_loc[cmpt] = c_rightglob
#                        jcn_loc[cmpt] = haloext[index][0]
#                        #TODO 
#                        a_loc[cmpt] = value*Icell[c_right]*(perm/visc)#value*Ihaloghost[np.int32(halocentergn[nod][j][-1])]*(perm/visc)   
#                        cmpt = cmpt + 1
#                              
#                for j in range(halonid[nod][-1]):
#                    center[:] = centerh[halonid[nod][j]][0:2]
#                    xdiff = center[0] - vertexn[nod][0]
#                    ydiff = center[1] - vertexn[nod][1]
#                    alpha = (1. + lambda_x[nod]*xdiff + \
#                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
#                    value =  alpha / volume[c_left] * parameters[cmptparam]
#                    irn_loc[cmpt] = c_leftglob
#                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
#                    a_loc[cmpt] = value*Icell[c_left]*(perm/visc) 
#                    cmpt = cmpt + 1
#                    #right cell-----------------------------------                                                                                              
#                    value =  -1. * alpha / volume[c_right] * parameters[cmptparam]
#                    irn_loc[cmpt] = c_rightglob
#                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
#                    a_loc[cmpt] = value*Icell[c_right]*(perm/visc)#value*Ihalo[halonid[nod][j]]*(perm/visc) 
#                    cmpt = cmpt + 1
#            cmptparam =+1
#        
#        irn_loc[cmpt] = c_leftglob
#        jcn_loc[cmpt] = c_rightglob
#        value =  param3[i] / volume[c_left]
#        a_loc[cmpt] = value*Icell[c_left]*(perm/visc) 
#        cmpt = cmpt + 1
#
#        # right cell------------------------------------------------------
#        irn_loc[cmpt] = c_rightglob
#        jcn_loc[cmpt] = c_leftglob
#        value =  -1. * param1[i] / volume[c_right]
#        a_loc[cmpt] = value*Icell[c_right]*(perm/visc)
#        cmpt = cmpt + 1
#    
#        irn_loc[cmpt] = c_rightglob
#        jcn_loc[cmpt] = c_rightglob
#        value =  -1. * param3[i] / volume[c_right]
#        a_loc[cmpt] = value*Icell[c_right]*(perm/visc) + (1/nbfR)*volume[c_right]*alpha_P*(1 - Icell[c_right])
#        cmpt = cmpt + 1
#    
#    for i in halofaces:
#        nbfL = faceidc[cellfid[i][0]][-1]
#        
#        c_left = cellfid[i][0]
#        c_leftglob  = loctoglob[c_left]
#        
#        parameters[0] = param4[i]; parameters[1] = param2[i]
#        
#        c_rightglob = haloext[halofid[i]][0]
#        c_right     = halofid[i]
#        
#        irn_loc[cmpt] = c_leftglob
#        jcn_loc[cmpt] = c_leftglob
#        value =  param1[i] / volume[c_left]
#        a_loc[cmpt] = value*Icell[c_left]*(perm/visc) + (1/nbfL)*volume[c_left]*alpha_P*(1 - Icell[c_left])
#        cmpt = cmpt + 1
#
#        irn_loc[cmpt] = c_leftglob
#        jcn_loc[cmpt] = c_rightglob
#        value =  param3[i] / volume[c_left]
#        a_loc[cmpt] = value*Icell[c_left]*(perm/visc)
#        cmpt = cmpt + 1
#        
#        cmptparam = 0
#        for nod in nodeidf[i][:nodeidf[i][-1]]:
#            if search_element(BCdirichlet, oldnamen[nod]) == 0 and search_element(BCneumannNH, oldnamen[nod]) == 0: 
#                for j in range(cellnid[nod][-1]):
#                    center[:] = centerc[cellnid[nod][j]][0:2]
#                    xdiff = center[0] - vertexn[nod][0]
#                    ydiff = center[1] - vertexn[nod][1]
#                    alpha = (1. + lambda_x[nod]*xdiff + \
#                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
#                    value =  alpha / volume[c_left] * parameters[cmptparam]
#                    irn_loc[cmpt] = c_leftglob
#                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
#                    a_loc[cmpt] = value*Icell[c_left]*(perm/visc)
#                    cmpt = cmpt + 1
#                    
#                for j in range(len(centergn[nod])):
#                    if centergn[nod][j][-1] != -1:
#                        center[:] = centergn[nod][j][0:2]
#                        xdiff = center[0] - vertexn[nod][0]
#                        ydiff = center[1] - vertexn[nod][1]
#                        alpha = (1. + lambda_x[nod]*xdiff + \
#                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
#                        index = np.int32(centergn[nod][j][2])
#                        value = alpha / volume[c_left] * parameters[cmptparam]
#                        irn_loc[cmpt] = c_leftglob
#                        jcn_loc[cmpt] = loctoglob[index]
#                        a_loc[cmpt] = value*Icell[c_left]*(perm/visc)
#                        cmpt = cmpt + 1
#                        
#                for j in range(len(halocentergn[nod])):
#                    if halocentergn[nod][j][-1] != -1:
#                        center[:] = halocentergn[nod][j][0:2]
#                        xdiff = center[0] - vertexn[nod][0]
#                        ydiff = center[1] - vertexn[nod][1]
#                        alpha = (1. + lambda_x[nod]*xdiff + \
#                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
#                        index = np.int32(halocentergn[nod][j][2])
#                        value = alpha / volume[c_left] * parameters[cmptparam]
#                        irn_loc[cmpt] = c_leftglob
#                        jcn_loc[cmpt] = haloext[index][0]
#                        a_loc[cmpt] = value*Icell[c_left]*(perm/visc)
#                        cmpt = cmpt + 1
#
#                for j in range(halonid[nod][-1]):
#                    center[:] = centerh[halonid[nod][j]][0:2]
#                    xdiff = center[0] - vertexn[nod][0]
#                    ydiff = center[1] - vertexn[nod][1]
#                    alpha = (1. + lambda_x[nod]*xdiff + \
#                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
#                    value =  alpha / volume[c_left] * parameters[cmptparam]
#                    irn_loc[cmpt] = c_leftglob
#                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
#                    a_loc[cmpt] = value*Icell[c_left]*(perm/visc)
#                    cmpt = cmpt + 1
#            cmptparam +=1
#            
#    for i in dirichletfaces:
#        nbfL = faceidc[cellfid[i][0]][-1]
#        c_left = cellfid[i][0]
#        c_leftglob  = loctoglob[c_left]
#        
#        irn_loc[cmpt] = c_leftglob
#        jcn_loc[cmpt] = c_leftglob
#        value = param1[i] / volume[c_left]
#        a_loc[cmpt] = value*Icell[c_left]*(perm/visc) + (1/nbfL)*volume[c_left]*alpha_P*(1 - Icell[c_left])
#        cmpt = cmpt + 1
#        
#        irn_loc[cmpt] = c_leftglob
#        jcn_loc[cmpt] = c_leftglob
#        value = -1. * param3[i] / volume[c_left]
#        a_loc[cmpt] = value*Icell[c_left]*(perm/visc) + (1/nbfL)*volume[c_left]*alpha_P*(1 - Icell[c_left])
#        cmpt = cmpt + 1
def get_triplet_2d_with_contrib(cellfid:'int32[:,:]', nodeidf:'int32[:,:]', faceidc:'int32[:,:]', vertexn:'float[:,:]', halofid:'int32[:]',
                                haloext:'int32[:,:]', oldnamen:'uint32[:]', volume:'float[:]', 
                                cellnid:'int32[:,:]', centerc:'float[:,:]', centerh:'float[:,:]', halonid:'int32[:,:]', 
                                periodicnid:'int32[:,:]', 
                                centergn:'float[:,:,:]', halocentergn:'float[:,:,:]', airDiamond:'float[:]', 
                                lambda_x:'float[:]', lambda_y:'float[:]', number:'uint32[:]', R_x:'float[:]', 
                                R_y:'float[:]', param1:'float[:]', 
                                param2:'float[:]', param3:'float[:]', param4:'float[:]', shift:'float[:,:]', 
                                nbelements:'int32', loctoglob:'int32[:]',
                                BCdirichlet:'uint32[:]', a_loc:'float[:]', irn_loc:'int32[:]', jcn_loc:'int32[:]',
                                matrixinnerfaces:'uint32[:]', halofaces:'uint32[:]', dirichletfaces:'uint32[:]',
                                Icell:'float[:]', #Ihalo:'float[:]', Ihaloghost:'float[:]',
                                alpha_P:'float', perm_vec:'float[:]', 
                                visc_vec:'float[:]', BCneumannNH:'uint32[:]', dist:'float[:]'):
    
    def search_element(a:'int32[:]', target_value:'int32'):
        find = 0
        for val in a:
            if val == target_value:
                find = 1
                break
        return find
        
    center = np.zeros(2)
    parameters = np.zeros(2)
    cmpt = 0

    for i in matrixinnerfaces:
        nbfL = faceidc[cellfid[i][0]][-1]
        nbfR = faceidc[cellfid[i][1]][-1]
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        parameters[0] = param4[i]; parameters[1] = param2[i]
    
        c_right = cellfid[i][1]
        c_rightglob = loctoglob[c_right]
        
        #perm_visc = dist[i][0]/(dist[i][1]/(perm_vec[c_rightglob]/visc_vec[c_rightglob]) + dist[i][2]/(perm_vec[c_leftglob]/visc_vec[c_leftglob]))

        # perm = dist[i][0]/(dist[i][1]/perm_vec[c_rightglob] + dist[i][2]/perm_vec[c_leftglob])
        # visc = dist[i][0]/(dist[i][1]/visc_vec[c_rightglob] + dist[i][2]/visc_vec[c_leftglob])

        perm = 0.5 * (perm_vec[c_rightglob] + perm_vec[c_leftglob])
        visc = 0.5 * (visc_vec[c_rightglob] + visc_vec[c_leftglob])

        perm_visc = perm / visc
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = param1[i] / volume[c_left]
        a_loc[cmpt] = value*Icell[c_left]*(perm_visc) + (1/nbfL)*volume[c_left]*alpha_P*(1 - Icell[c_left])
        cmpt = cmpt + 1
        
        cmptparam = 0
        for nod in nodeidf[i][:nodeidf[i][-1]]:
            if search_element(BCdirichlet, oldnamen[nod]) == 0:# and search_element(BCneumannNH, oldnamen[nod]) == 0:
                for j in range(cellnid[nod][-1]):
                    center[:] = centerc[cellnid[nod][j]][0:2]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                    value =  alpha / volume[c_left] * parameters[cmptparam]
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
                    a_loc[cmpt] = value*Icell[c_left]*(perm_visc) 
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    value =  -1. * alpha / volume[c_right] * parameters[cmptparam]
                    irn_loc[cmpt] = c_rightglob
                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
                    a_loc[cmpt] = value*Icell[c_right]*(perm_visc) 
                    cmpt = cmpt + 1
                
                for j in range(len(centergn[nod])):
                    if centergn[nod][j][-1] != -1:
                        center[:] = centergn[nod][j][0:2]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                        index = int(centergn[nod][j][2])
                        value = alpha / volume[c_left] * parameters[cmptparam]
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = loctoglob[index]
                        a_loc[cmpt] = value*Icell[c_left]*(perm_visc) 
                        cmpt = cmpt + 1
                        #right cell-----------------------------------                                                                                              
                        value = -1. * alpha / volume[c_right] * parameters[cmptparam]
                        irn_loc[cmpt] = c_rightglob
                        jcn_loc[cmpt] = loctoglob[index]
                        a_loc[cmpt] = value*Icell[c_right]*(perm_visc) 
                        cmpt = cmpt + 1
                        
                for j in range(len(halocentergn[nod])):
                    if halocentergn[nod][j][-1] != -1:
                        center[:] = halocentergn[nod][j][0:2]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                        index = int(halocentergn[nod][j][2])
#                        cell  = int(halocentergn[nod][j][-1])
                        value = alpha / volume[c_left] * parameters[cmptparam]
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = haloext[index][0]
                        a_loc[cmpt] = value*Icell[c_left]*(perm_visc) 
                        cmpt = cmpt + 1
                        #right cell-----------------------------------                                                                                              
                        value = -1. * alpha / volume[c_right] * parameters[cmptparam]
                        irn_loc[cmpt] = c_rightglob
                        jcn_loc[cmpt] = haloext[index][0]
                        #TODO 
                        a_loc[cmpt] = value*Icell[c_right]*(perm_visc)#value*Ihaloghost[int(halocentergn[nod][j][-1])]*(perm/visc)   
                        cmpt = cmpt + 1
                              
                for j in range(halonid[nod][-1]):
                    center[:] = centerh[halonid[nod][j]][0:2]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                    value =  alpha / volume[c_left] * parameters[cmptparam]
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
                    a_loc[cmpt] = value*Icell[c_left]*(perm_visc) 
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    value =  -1. * alpha / volume[c_right] * parameters[cmptparam]
                    irn_loc[cmpt] = c_rightglob
                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
                    a_loc[cmpt] = value*Icell[c_right]*(perm_visc)#value*Ihalo[halonid[nod][j]]*(perm/visc) 
                    cmpt = cmpt + 1
            cmptparam =+1
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_rightglob
        value =  param3[i] / volume[c_left]
        a_loc[cmpt] = value*Icell[c_left]*(perm_visc) 
        cmpt = cmpt + 1

        # right cell------------------------------------------------------
        irn_loc[cmpt] = c_rightglob
        jcn_loc[cmpt] = c_leftglob
        value =  -1. * param1[i] / volume[c_right]
        a_loc[cmpt] = value*Icell[c_right]*(perm_visc)
        cmpt = cmpt + 1
    
        irn_loc[cmpt] = c_rightglob
        jcn_loc[cmpt] = c_rightglob
        value =  -1. * param3[i] / volume[c_right]
        a_loc[cmpt] = value*Icell[c_right]*(perm_visc) + (1/nbfR)*volume[c_right]*alpha_P*(1 - Icell[c_right])
        cmpt = cmpt + 1
    '''
    for i in halofaces:
        nbfL = faceidc[cellfid[i][0]][-1]
        
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        perm = perm_vec[c_leftglob] 
        visc = visc_vec[c_leftglob]

        parameters[0] = param4[i]; parameters[1] = param2[i]
        
        c_rightglob = haloext[halofid[i]][0]
        c_right     = halofid[i]
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value =  param1[i] / volume[c_left]
        a_loc[cmpt] = value*Icell[c_left]*(perm/visc) + (1/nbfL)*volume[c_left]*alpha_P*(1 - Icell[c_left])
        cmpt = cmpt + 1

        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_rightglob
        value =  param3[i] / volume[c_left]
        a_loc[cmpt] = value*Icell[c_left]*(perm/visc)
        cmpt = cmpt + 1
        
        cmptparam = 0
        for nod in nodeidf[i]:
            if search_element(BCdirichlet, oldnamen[nod]) == 0 and search_element(BCneumannNH, oldnamen[nod]) == 0: 
                for j in range(cellnid[nod][-1]):
                    center[:] = centerc[cellnid[nod][j]][0:2]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                    value =  alpha / volume[c_left] * parameters[cmptparam]
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
                    a_loc[cmpt] = value*Icell[c_left]*(perm/visc)
                    cmpt = cmpt + 1
                    
                for j in range(len(centergn[nod])):
                    if centergn[nod][j][-1] != -1:
                        center[:] = centergn[nod][j][0:2]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                        index = int(centergn[nod][j][2])
                        value = alpha / volume[c_left] * parameters[cmptparam]
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = loctoglob[index]
                        a_loc[cmpt] = value*Icell[c_left]*(perm/visc)
                        cmpt = cmpt + 1
                        
                for j in range(len(halocentergn[nod])):
                    if halocentergn[nod][j][-1] != -1:
                        center[:] = halocentergn[nod][j][0:2]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                        index = int(halocentergn[nod][j][2])
                        value = alpha / volume[c_left] * parameters[cmptparam]
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = haloext[index][0]
                        a_loc[cmpt] = value*Icell[c_left]*(perm/visc)
                        cmpt = cmpt + 1

                for j in range(halonid[nod][-1]):
                    center[:] = centerh[halonid[nod][j]][0:2]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                    value =  alpha / volume[c_left] * parameters[cmptparam]
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
                    a_loc[cmpt] = value*Icell[c_left]*(perm/visc)
                    cmpt = cmpt + 1
            cmptparam +=1
    '''
    for i in dirichletfaces:
        nbfL = faceidc[cellfid[i][0]][-1]
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        perm = perm_vec[c_leftglob]
        visc = visc_vec[c_leftglob]

        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = param1[i] / volume[c_left]
        a_loc[cmpt] = value*Icell[c_left]*(perm/visc) + (1/nbfL)*volume[c_left]*alpha_P*(1 - Icell[c_left])
        cmpt = cmpt + 1
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = -1. * param3[i] / volume[c_left]
        a_loc[cmpt] = value*Icell[c_left]*(perm/visc) + (1/nbfL)*volume[c_left]*alpha_P*(1 - Icell[c_left])
        cmpt = cmpt + 1
    
def Mat_Assembly(row, col, data,# b, 
                 P_ghost, Icell, 
                 perm, visc, alpha, 
                 cellfid, volume, faceidc,
                 mesuref, matrixinnerfaces, 
                 dirichletfaces, dist):
    
    row[:] = 0
    col[:] = 0
    data[:] = 0.

    cmpt = 0
    for face in matrixinnerfaces:
        nbfL = faceidc[cellfid[0][0]][-1]
        nbfR = faceidc[cellfid[0][1]][-1]
        K = cellfid[face][0] 
        L = cellfid[face][1]
        mesure = mesuref[face]
        volumeK = volume[K]
        volumeL = volume[L]
        
        row[cmpt] = K
        col[cmpt] = K
        data[cmpt] = - Icell[K]*(perm/visc)*(mesure/dist[face]) + (1/nbfL)*volumeK*alpha*(1 - Icell[K])
        cmpt = cmpt + 1
        
        row[cmpt] = L
        col[cmpt] = L
        data[cmpt] = - Icell[L]*(perm/visc)*(mesure/dist[face]) + (1/nbfR)*volumeL*alpha*(1 - Icell[L])
        cmpt = cmpt + 1
        
        row[cmpt] = K
        col[cmpt] = L
        data[cmpt] = Icell[K]*(perm/visc)*(mesure/dist[face]) 
        cmpt = cmpt + 1
        
        row[cmpt] = L
        col[cmpt] = K
        data[cmpt] = Icell[L]*(perm/visc)*(mesure/dist[face]) 
        cmpt = cmpt + 1
        
    for face in dirichletfaces: 
        nbfL = faceidc[cellfid[0][0]][-1]
        K = cellfid[face][0] 
        mesure = mesuref[face]
        volumeK = volume[K]
        
        row[cmpt] = K
        col[cmpt] = K
        data[cmpt] = - Icell[K]*(perm/visc)*(mesure/dist[face]) + (1/nbfL)*volumeK*alpha*(1 - Icell[K])
        cmpt = cmpt + 1
        
    
def Vec_Assembly(P_ghost, Icell, perm, visc, alpha, cellid,
                 volume, mesuref, dirichletfaces, neumannNHfaces,
                  distortho, b, cst, normalf):
    
    for face in dirichletfaces:
        K = cellid[face][0] 
        mesure = mesuref[face]
        dist1 = distortho[face]
        b[K] = b[K] - Icell[K]*P_ghost[face]*(perm/visc)*(mesure/dist1)
            
    for face in neumannNHfaces:
        K = cellid[face][0] 
        mesure = mesuref[face]
        b[K] -= 1 * Icell[K]*(perm/visc)*cst*mesure
        

#def get_rhs_glob_2d_with_contrib(cellfid:'int32[:,:]', nodeidf:'int32[:,:]', oldname:'uint32[:]', 
#                                 volume:'float[:]', centergn:'float[:,:,:]', loctoglob:'int32[:]', 
#                                 param1:'float[:]', param2:'float[:]', 
#                                 param3:'float[:]', param4:'float[:]', Pbordnode:'float[:]', 
#                                 Pbordface:'float[:]',  rhs:'float[:]',
#                                 BCdirichlet:'uint32[:]', centergf:'float[:,:]', matrixinnerfaces:'uint32[:]',
#                                 halofaces:'uint32[:]', dirichletfaces:'uint32[:]', neumannNHfaces:'uint32[:]', 
#                                 Icell:'float[:]', Inode:'float[:]', perm:'float', visc:'float',
#                                 cst:'float', mesuref:'float[:]', normalf:'float[:,:]'):    
#    
#    def search_element(a:'int32[:]', target_value:'int32'):
#        find = 0
#        for val in a:
#            if val == target_value:
#                find = 1
#                break
#        return find
#    
#    rhs[:] = 0.
#    for i in matrixinnerfaces:
#        c_left = cellfid[i][0]
#        c_leftglob  = loctoglob[c_left]
#        
#        i_1 = nodeidf[i][0]
#        i_2 = nodeidf[i][1]    
#        
#        c_right = cellfid[i][1]
#        c_rightglob = loctoglob[c_right]
#        
#        if search_element(BCdirichlet, oldname[i_1]) == 1: 
#            VL = Pbordnode[i_1]*Icell[c_left]*(perm/visc)
#            value_left = -1. * VL * param4[i] / volume[c_left]
#            rhs[c_leftglob] +=  value_left
#            
#            VR = Pbordnode[i_1]*Icell[c_right]*(perm/visc)
#            value_right = VR * param4[i] / volume[c_right]
#            rhs[c_rightglob] += value_right
#            
#        if search_element(BCdirichlet, oldname[i_2]) == 1: 
#            VL = Pbordnode[i_2]*Icell[c_left]*(perm/visc)
#            value_left =  -1. * VL * param2[i] / volume[c_left]
#            rhs[c_leftglob] += value_left
#            
#            VR = Pbordnode[i_2]*Icell[c_right]*(perm/visc)
#            value_right =  VR * param2[i] / volume[c_right]
#            rhs[c_rightglob] += value_right
#                    
#    for i in halofaces:
#        c_left = cellfid[i][0]
#        c_leftglob  = loctoglob[c_left]
#         
#        i_1 = nodeidf[i][0]
#        i_2 = nodeidf[i][1]    
#        
#        if search_element(BCdirichlet, oldname[i_1]) == 1: 
#            VL = Pbordnode[i_1]*Icell[c_left]*(perm/visc)
#            value_left =  -1. * VL * param4[i] / volume[c_left]
#            rhs[c_leftglob] += value_left
#        
#        if search_element(BCdirichlet, oldname[i_2]) == 1: 
#            VR = Pbordnode[i_2]*Icell[c_left]*(perm/visc)
#            value_left =  -1. * VR * param2[i] / volume[c_left]
#            rhs[c_leftglob] += value_left
#            
#    for i in dirichletfaces:
#        c_left = cellfid[i][0]
#        c_leftglob  = loctoglob[c_left]
#        
#        i_1 = nodeidf[i][0]
#        i_2 = nodeidf[i][1]  
#        
#        if centergn[i_1][0][2] != -1:     
#            VL = Pbordnode[i_1]*Icell[c_left]*(perm/visc)
#            value_left = -1. * VL * param4[i] / volume[c_left]
#            rhs[c_leftglob] += value_left
#           
#        if centergn[i_2][0][2] != -1: 
#            VL = Pbordnode[i_2]*Icell[c_left]*(perm/visc)
#            value_left = -1. * VL * param2[i] / volume[c_left]
#            rhs[c_leftglob] += value_left
#        
#        V_K = Pbordface[i]*Icell[c_left]*(perm/visc)
#        value = -2. * param3[i] / volume[c_left] * V_K
#        rhs[c_leftglob] += value
#        
#    for i in neumannNHfaces:
#        c_left = cellfid[i][0]
#        c_leftglob  = loctoglob[c_left]
#        
#        rhs[c_leftglob] += 1 * Icell[c_left]*(perm/visc)*cst*normalf[i][0]/ volume[c_left]

def get_rhs_glob_2d_with_contrib(cellfid:'int32[:,:]', nodeidf:'int32[:,:]', oldname:'uint32[:]', 
                                 volume:'float[:]', centergn:'float[:,:,:]', loctoglob:'int32[:]', 
                                 param1:'float[:]', param2:'float[:]', 
                                 param3:'float[:]', param4:'float[:]', Pbordnode:'float[:]', 
                                 Pbordface:'float[:]',  rhs:'float[:]',
                                 BCdirichlet:'uint32[:]', centergf:'float[:,:]', matrixinnerfaces:'uint32[:]',
                                 halofaces:'uint32[:]', dirichletfaces:'uint32[:]', neumannNHfaces:'uint32[:]', 
                                 Icell:'float[:]', Inode:'float[:]', perm_vec:'float[:]', visc_vec:'float[:]',
                                 cst:'float', mesuref:'float[:]', normalf:'float[:,:]', dist:'float[:]'):  
    
    def search_element(a:'int32[:]', target_value:'int32'):
        find = 0
        for val in a:
            if val == target_value:
                find = 1
                break
        return find
    
    rhs[:] = 0.
    for i in matrixinnerfaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]    
        
        c_right = cellfid[i][1]
        c_rightglob = loctoglob[c_right]
        

        perm = 0.5 * (perm_vec[c_rightglob] + perm_vec[c_leftglob])
        visc = 0.5 * (visc_vec[c_rightglob] + visc_vec[c_leftglob])

        perm_visc = perm / visc

        # perm = dist[i][0]/(dist[i][1]/perm_vec[c_rightglob] + dist[i][2]/perm_vec[c_leftglob])
        # visc = dist[i][0]/(dist[i][1]/visc_vec[c_rightglob] + dist[i][2]/visc_vec[c_leftglob])

        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            VL = Pbordnode[i_1]*Icell[c_left]*(perm_visc)
            value_left = -1. * VL * param4[i] / volume[c_left]
            rhs[c_leftglob] +=  value_left
            
            VR = Pbordnode[i_1]*Icell[c_right]*(perm_visc)
            value_right = VR * param4[i] / volume[c_right]
            rhs[c_rightglob] += value_right
            
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            VL = Pbordnode[i_2]*Icell[c_left]*(perm_visc)
            value_left =  -1. * VL * param2[i] / volume[c_left]
            rhs[c_leftglob] += value_left
            
            VR = Pbordnode[i_2]*Icell[c_right]*(perm_visc)
            value_right =  VR * param2[i] / volume[c_right]
            rhs[c_rightglob] += value_right
                    
    for i in halofaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]

        perm = perm_vec[c_leftglob] 
        visc = visc_vec[c_leftglob]

        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]    
        
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            VL = Pbordnode[i_1]*Icell[c_left]*(perm/visc)
            value_left =  -1. * VL * param4[i] / volume[c_left]
            rhs[c_leftglob] += value_left
        
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            VR = Pbordnode[i_2]*Icell[c_left]*(perm/visc)
            value_left =  -1. * VR * param2[i] / volume[c_left]
            rhs[c_leftglob] += value_left
            
    for i in dirichletfaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        perm = perm_vec[c_leftglob] 
        visc = visc_vec[c_leftglob]

        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]  
        
        if centergn[i_1][0][2] != -1:     
            VL = Pbordnode[i_1]*Icell[c_left]*(perm/visc)
            value_left = -1. * VL * param4[i] / volume[c_left]
            rhs[c_leftglob] += value_left
           
        if centergn[i_2][0][2] != -1: 
            VL = Pbordnode[i_2]*Icell[c_left]*(perm/visc)
            value_left = -1. * VL * param2[i] / volume[c_left]
            rhs[c_leftglob] += value_left
        
        V_K = Pbordface[i]*Icell[c_left]*(perm/visc)
        value = -2. * param3[i] / volume[c_left] * V_K
        rhs[c_leftglob] += value
        
    for i in neumannNHfaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]

        perm = perm_vec[c_left] 
        visc = visc_vec[c_left]
        rhs[c_leftglob] -= 1 * Icell[c_left]*(perm/visc)*cst*(np.sqrt(normalf[i][0]**2+normalf[i][1]**2))/volume[c_left]
        
def get_rhs_loc_2d_with_contrib(cellfid:'int32[:,:]', nodeidf:'int32[:,:]', oldname:'uint32[:]', 
                                 volume:'float[:]', centergn:'float[:,:,:]', loctoglob:'int32[:]', 
                                 param1:'float[:]', param2:'float[:]', 
                                 param3:'float[:]', param4:'float[:]', Pbordnode:'float[:]', 
                                 Pbordface:'float[:]',  rhs:'float[:]',
                                 BCdirichlet:'uint32[:]', centergf:'float[:,:]', matrixinnerfaces:'uint32[:]',
                                 halofaces:'uint32[:]', dirichletfaces:'uint32[:]', neumannNHfaces:'uint32[:]', 
                                 Icell:'float[:]', Inode:'float[:]', perm:'float', visc:'float',
                                 cst:'float', mesuref:'float[:]', normalf:'float[:,:]'):

    
    def search_element(a:'int32[:]', target_value:'int32'):
        find = 0
        for val in a:
            if val == target_value:
                find = 1
                break
        return find
    
    rhs[:] = 0.
    for i in matrixinnerfaces:
        c_left = cellfid[i][0]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]    
        
        c_right = cellfid[i][1]
        
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            VL = Pbordnode[i_1]*Icell[c_left]*(perm/visc)
            value_left = -1. * VL * param4[i] / volume[c_left]
            rhs[c_left] +=  value_left
            
            VR = Pbordnode[i_1]*Icell[c_right]*(perm/visc)
            value_right = VR * param4[i] / volume[c_right]
            rhs[c_right] += value_right
            
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            VL = Pbordnode[i_2]*Icell[c_left]*(perm/visc)
            value_left =  -1. * VL * param2[i] / volume[c_left]
            rhs[c_left] += value_left
            
            VR = Pbordnode[i_2]*Icell[c_right]*(perm/visc)
            value_right =  VR * param2[i] / volume[c_right]
            rhs[c_right] += value_right
                    
    for i in halofaces:
        c_left = cellfid[i][0]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]    
        
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            VL = Pbordnode[i_1]*Icell[c_left]*(perm/visc)
            value_left =  -1. * VL * param4[i] / volume[c_left]
            rhs[c_left] += value_left
        
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            VR = Pbordnode[i_2]*Icell[c_left]*(perm/visc)
            value_left =  -1. * VR * param2[i] / volume[c_left]
            rhs[c_left] += value_left
            
    for i in dirichletfaces:
        c_left = cellfid[i][0]
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]  
        
        if centergn[i_1][0][2] != -1:     
            VL = Pbordnode[i_1]*Icell[c_left]*(perm/visc)
            value_left = -1. * VL * param4[i] / volume[c_left]
            rhs[c_left] += value_left
           
        if centergn[i_2][0][2] != -1: 
            VL = Pbordnode[i_2]*Icell[c_left]*(perm/visc)
            value_left = -1. * VL * param2[i] / volume[c_left]
            rhs[c_left] += value_left
        
        V_K = Pbordface[i]*Icell[c_left]*(perm/visc)
        value = -2. * param3[i] / volume[c_left] * V_K
        rhs[c_left] += value
        
    for i in neumannNHfaces:
        c_left = cellfid[i][0]
        rhs[c_left] += 1 * Icell[c_left]*(perm/visc)*cst*normalf[i][0]/volume[c_left]

