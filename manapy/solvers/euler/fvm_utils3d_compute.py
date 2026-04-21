from manapy.backends.compile_fun import compile
import numpy as np

def _node_for_interpolation_3d(xCenterForInterp: 'float64[:,:]', yCenterForInterp: 'float64[:,:]', zCenterForInterp: 'float64[:,:]', nodefid: 'int32[:,:]', cellfid: 'int32[:,:]', halofid: 'int32[:]', cellcenter: 'float64[:,:]', vertexcenter: 'float64[:,:]', ghostcenter: 'float64[:,:]', halocenter: 'float64[:,:]', name: 'uint32[:]'):

    nbfaces = len(nodefid)
    for i in range(nbfaces):
        
        xCenterForInterp[i][0:3] = vertexcenter[nodefid[i][0:3], 0]
        xCenterForInterp[i][3]   = cellcenter[cellfid[i][0]][0]
        
        yCenterForInterp[i][0:3] = vertexcenter[nodefid[i][0:3],1]
        yCenterForInterp[i][3]   = cellcenter[cellfid[i][0]][1]
        
        zCenterForInterp[i][0:3] = vertexcenter[nodefid[i][0:3],2]
        zCenterForInterp[i][3]   = cellcenter[cellfid[i][0]][2]
     
        if name[i] == 0:

            xCenterForInterp[i][4] = cellcenter[cellfid[i][1]][0]
            yCenterForInterp[i][4] = cellcenter[cellfid[i][1]][1]
            zCenterForInterp[i][4] = cellcenter[cellfid[i][1]][2]
            
        elif name[i] == 10: 

            xCenterForInterp[i][4] = halocenter[halofid[i]][0]
            yCenterForInterp[i][4] = halocenter[halofid[i]][1]
            zCenterForInterp[i][4] = halocenter[halofid[i]][2]
            
        else:

            xCenterForInterp[i][4] = ghostcenter[i][0]
            yCenterForInterp[i][4] = ghostcenter[i][1]
            zCenterForInterp[i][4] = ghostcenter[i][2]
            
def _node_value_for_interpolation_3d(ValForInterp: 'float64[:,:]', w_cell: 'float64[:]', w_node: 'float64[:]', w_ghost: 'float64[:]', w_halo: 'float64[:]', nodefid: 'int32[:,:]', cellfid: 'int32[:,:]', halofid: 'int32[:]', name: 'uint32[:]', face_ghost_id: 'int[:]'):


    nbfaces = len(nodefid)
    for i in range(nbfaces):
        
        ValForInterp[i][0:3] = w_node[nodefid[i][0:3]]
        ValForInterp[i][3]   = w_cell[cellfid[i][0]]
        
        if name[i] == 0:
            ValForInterp[i][4] = w_cell[cellfid[i][1]]

        elif name[i] == 10: 
            ValForInterp[i][4] = w_halo[halofid[i]]
           
        else:
            ValForInterp[i][4] = w_ghost[face_ghost_id[i]]

def _weight_parameters_carac_3d(xCenterForInterp: 'float64[:]', yCenterForInterp: 'float64[:]', zCenterForInterp: 'float64[:]', X0: 'float64', Y0: 'float64', Z0: 'float64'):

    I_xx = 0.
    I_yy = 0.
    I_zz = 0.
    I_xy = 0.
    I_xz = 0.
    I_yz = 0.
    R_x  = 0.
    R_y  = 0.
    R_z  = 0.
    
    lambda_x = 0.
    lambda_y = 0.
    lambda_z = 0.

    #loop over the 5 points arround the face
    
    for i in range(0, 5):
        Rx = xCenterForInterp[i] - X0
        Ry = yCenterForInterp[i] - Y0
        Rz = zCenterForInterp[i] - Z0
       
        I_xx += (Rx * Rx)
        I_yy += (Ry * Ry)
        I_zz += (Rz * Rz)
        I_xy += (Rx * Ry)
        I_xz += (Rx * Rz)
        I_yz += (Ry * Rz)
       
        R_x += Rx
        R_y += Ry
        R_z += Rz
        
    D = I_xx*I_yy*I_zz + 2*I_xy*I_xz*I_yz - I_xx*I_yz*I_yz - I_yy*I_xz*I_xz - I_zz*I_xy*I_xy
    
      
    lambda_x = ((I_yz*I_yz - I_yy*I_zz)*R_x + (I_xy*I_zz - I_xz*I_yz)*R_y + (I_xz*I_yy - I_xy*I_yz)*R_z) / D
    lambda_y = ((I_xy*I_zz - I_xz*I_yz)*R_x + (I_xz*I_xz - I_xx*I_zz)*R_y + (I_yz*I_xx - I_xz*I_xy)*R_z) / D
    lambda_z = ((I_xz*I_yy - I_xy*I_yz)*R_x + (I_yz*I_xx - I_xz*I_xy)*R_y + (I_xy*I_xy - I_xx*I_yy)*R_z) / D
    
    return R_x, R_y, R_z, lambda_x, lambda_y, lambda_z

def _set_carac_field_3d(ValForInterp: 'float64[:]', xCenterForInterp: 'float64[:]', yCenterForInterp: 'float64[:]', zCenterForInterp: 'float64[:]', X0: 'float64', Y0: 'float64', Z0: 'float64'):

    w_carac = 0.
    R_x = 0.
    R_y = 0.
    R_z = 0.
    lambda_x = 0.
    lambda_y = 0.
    lambda_z = 0.
    R_x, R_y, R_z, lambda_x, lambda_y, lambda_z = _weight_parameters_carac_3d(xCenterForInterp, yCenterForInterp, zCenterForInterp, X0, Y0, Z0)
    
    for i in range(0, 5):
        
        xdiff = xCenterForInterp[i] - X0
        ydiff = yCenterForInterp[i] - Y0
        zdiff = zCenterForInterp[i] - Z0
        
        alpha_interp = (1. + lambda_x*xdiff + lambda_y*ydiff + lambda_z*zdiff)/ (5. + lambda_x*R_x + lambda_y*R_y + lambda_z*R_z)
        
        w_carac  += alpha_interp * ValForInterp[i]
   
    return w_carac

def _compute_normale_inverse(inormal: 'float64[:,:]', itangent: 'float64[:,:]', ibinormal: 'float64[:,:]', normal: 'float64[:,:]', tangent: 'float64[:,:]', binormal: 'float64[:,:]'):
    nbfaces = len(normal)
    for k in range(nbfaces): 
        a = normal[k][0]/np.sqrt(normal[k][0]*normal[k][0]+normal[k][1]*normal[k][1]+normal[k][2]*normal[k][2])
        b = normal[k][1]/np.sqrt(normal[k][0]*normal[k][0]+normal[k][1]*normal[k][1]+normal[k][2]*normal[k][2])
        c = normal[k][2]/np.sqrt(normal[k][0]*normal[k][0]+normal[k][1]*normal[k][1]+normal[k][2]*normal[k][2])
        
        d = tangent[k][0]/np.sqrt(tangent[k][0]*tangent[k][0]+tangent[k][1]*tangent[k][1]+tangent[k][2]*tangent[k][2])
        e = tangent[k][1]/np.sqrt(tangent[k][0]*tangent[k][0]+tangent[k][1]*tangent[k][1]+tangent[k][2]*tangent[k][2])
        f = tangent[k][2]/np.sqrt(tangent[k][0]*tangent[k][0]+tangent[k][1]*tangent[k][1]+tangent[k][2]*tangent[k][2])
        
        g = binormal[k][0]/np.sqrt(binormal[k][0]*binormal[k][0]+binormal[k][1]*binormal[k][1]+binormal[k][2]*binormal[k][2])
        h = binormal[k][1]/np.sqrt(binormal[k][0]*binormal[k][0]+binormal[k][1]*binormal[k][1]+binormal[k][2]*binormal[k][2])
        i = binormal[k][2]/np.sqrt(binormal[k][0]*binormal[k][0]+binormal[k][1]*binormal[k][1]+binormal[k][2]*binormal[k][2])
        
        det = a*e*i+b*f*g+c*d*h-c*e*g-f*h*a-i*b*d
        
        inormal[k][0]=(e*i-f*h)/det
        inormal[k][1]=(c*h-b*i)/det
        inormal[k][2]=(b*f-c*e)/det
        
        itangent[k][0]=(f*g-d*i)/det
        itangent[k][1]=(a*i-c*g)/det
        itangent[k][2]=(c*d-a*f)/det
        
        ibinormal[k][0]=(d*h-e*g)/det
        ibinormal[k][1]=(b*g-a*h)/det
        ibinormal[k][2]=(a*e-b*d)/det
        
def _departure_euler_3d(X0: 'float64[:]', Y0: 'float64[:]', Z0: 'float64[:]', rhof: 'float64[:]', rhouf: 'float64[:]', rhovf: 'float64[:]', rhowf: 'float64[:]', rhoValForInterp: 'float64[:,:]', uValForInterp: 'float64[:,:]', vValForInterp: 'float64[:,:]', wValForInterp: 'float64[:,:]', xCenterForInterp: 'float64[:,:]', yCenterForInterp: 'float64[:,:]', zCenterForInterp: 'float64[:,:]', nodeidf: 'int32[:,:]', normalf: 'float64[:,:]', mesuref: 'float64[:]', centerf: 'float64[:,:]', vertexn: 'float64[:,:]', centerc: 'float64[:,:]', centerg: 'float64[:,:]', centerh: 'float64[:,:]', name: 'uint32[:]', dt: 'float64', alphaf: 'float64'):

    nbfaces = len(nodeidf)
    
    X0[:] = centerf[:,0]
    Y0[:] = centerf[:,1]
    Z0[:] = centerf[:,2]
   
    u_ed = 0.
    v_ed = 0.
    w_ed = 0.
    
    for i in range(nbfaces):  
        
        rho_ed = _set_carac_field_3d(rhoValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], zCenterForInterp[i], X0[i], Y0[i], Z0[i])
        u_ed   = _set_carac_field_3d(uValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], zCenterForInterp[i], X0[i], Y0[i], Z0[i])/rho_ed
        v_ed   = _set_carac_field_3d(vValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], zCenterForInterp[i], X0[i], Y0[i], Z0[i])/rho_ed
        w_ed   = _set_carac_field_3d(wValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], zCenterForInterp[i], X0[i], Y0[i], Z0[i])/rho_ed
        
        rhof[i]  = rho_ed 
        rhouf[i] = rho_ed*u_ed
        rhovf[i] = rho_ed*v_ed
        rhowf[i] = rho_ed*w_ed

        
        u_n = (u_ed*normalf[i][0] + v_ed*normalf[i][1] + w_ed*normalf[i][2])/np.sqrt(normalf[i][0]*normalf[i][0]+normalf[i][1]*normalf[i][1]+normalf[i][2]*normalf[i][2])
            
        u_nx = u_n*normalf[i][0]/np.sqrt(normalf[i][0]*normalf[i][0]+normalf[i][1]*normalf[i][1]+normalf[i][2]*normalf[i][2])
        u_ny = u_n*normalf[i][1]/np.sqrt(normalf[i][0]*normalf[i][0]+normalf[i][1]*normalf[i][1]+normalf[i][2]*normalf[i][2])
        u_nz = u_n*normalf[i][2]/np.sqrt(normalf[i][0]*normalf[i][0]+normalf[i][1]*normalf[i][1]+normalf[i][2]*normalf[i][2])
        
        # ########### Euler ###########
        
        X0[i] =  X0[i] - alphaf*dt*u_nx
        Y0[i] =  Y0[i] - alphaf*dt*u_ny
        Z0[i] =  Z0[i] - alphaf*dt*u_nz
        
        ############ RK3 ##########""
        
        # xrk1 =  X0[i] - alphaf*dt*u_nx
        # yrk1 =  Y0[i] - alphaf*dt*u_ny
        # zrk1 =  Z0[i] - alphaf*dt*u_nz
                
        # u    =  _set_carac_field_3d(uValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], zCenterForInterp[i],xrk1,yrk1,zrk1)/rho_ed
        # v    =  _set_carac_field_3d(vValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], zCenterForInterp[i],xrk1,yrk1,zrk1)/rho_ed
        # w    =  _set_carac_field_3d(wValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], zCenterForInterp[i],xrk1,yrk1,zrk1)/rho_ed
      
                    
        # xrk2 = 0.75*X0[i]+0.25*xrk1-0.25*dt*alphaf*u
        # yrk2 = 0.75*Y0[i]+0.25*yrk1-0.25*dt*alphaf*v
        # zrk2 = 0.75*Z0[i]+0.25*zrk1-0.25*dt*alphaf*w
        
        # u    =  _set_carac_field_3d(uValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], zCenterForInterp[i],xrk2,yrk2,zrk2)/rho_ed
        # v    =  _set_carac_field_3d(vValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], zCenterForInterp[i],xrk2,yrk2,zrk2)/rho_ed
        # w    =  _set_carac_field_3d(wValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], zCenterForInterp[i],xrk2,yrk2,zrk2)/rho_ed
        
        # X0[i] = (X0[i] + 2.0*xrk2-2.0*alphaf*dt*u)/3.0
        # Y0[i] = (Y0[i] + 2.0*yrk2-2.0*alphaf*dt*v)/3.0
        # Z0[i] = (Z0[i] + 2.0*yrk2-2.0*alphaf*dt*w)/3.0
        
def _predictor_euler_3d(rho_p: 'float64[:]', P_p: 'float64[:]', rhou_p: 'float64[:]', rhov_p: 'float64[:]', rhow_p: 'float64[:]', rhoE_p: 'float64[:]', rhof: 'float64[:]', rhouf: 'float64[:]', rhovf: 'float64[:]', rhowf: 'float64[:]', rhoValForInterp: 'float64[:,:]', rhouValForInterp: 'float64[:,:]', rhovValForInterp: 'float64[:,:]', rhowValForInterp: 'float64[:,:]', rhoEValForInterp: 'float64[:,:]', xCenterForInterp: 'float64[:,:]', yCenterForInterp: 'float64[:,:]', zCenterForInterp: 'float64[:,:]', X0: 'float64[:]', Y0: 'float64[:]', Z0: 'float64[:]', ugradfacex: 'float64[:]', ugradfacey: 'float64[:]', ugradfacez: 'float64[:]', vgradfacex: 'float64[:]', vgradfacey: 'float64[:]', vgradfacez: 'float64[:]', wgradfacex: 'float64[:]', wgradfacey: 'float64[:]', wgradfacez: 'float64[:]', Pgradfacex: 'float64[:]', Pgradfacey: 'float64[:]', Pgradfacez: 'float64[:]', gamma: 'float64', d_t: 'float64', alphaf: 'float64', nodeidf: 'int32[:,:]', normal: 'float64[:,:]', tangent: 'float64[:,:]', binormal: 'float64[:,:]', inormal: 'float64[:,:]', itangent: 'float64[:,:]', ibinormal: 'float64[:,:]', mesuref: 'float64[:]'):
    
    
    nbfaces = len(nodeidf)

    for i in range(nbfaces):
        
        rho_ed  = _set_carac_field_3d(rhoValForInterp[i],  xCenterForInterp[i], yCenterForInterp[i], zCenterForInterp[i], X0[i], Y0[i], Z0[i])
        rhou_ed = _set_carac_field_3d(rhouValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], zCenterForInterp[i], X0[i], Y0[i], Z0[i])
        rhov_ed = _set_carac_field_3d(rhovValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], zCenterForInterp[i], X0[i], Y0[i], Z0[i])
        rhow_ed = _set_carac_field_3d(rhowValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], zCenterForInterp[i], X0[i], Y0[i], Z0[i])
        rhoE_ed = _set_carac_field_3d(rhoEValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], zCenterForInterp[i], X0[i], Y0[i], Z0[i])

        rhop  = rho_ed
        rhoup = rhou_ed
        rhovp = rhov_ed
        rhowp = rhow_ed
        rhoEp = rhoE_ed
        
        up = rhou_ed/rho_ed
        vp = rhov_ed/rho_ed
        wp = rhow_ed/rho_ed
             
        Pp = (gamma-1)*(rhoE_ed-0.5*rhop*(up**2+vp**2+wp**2))
        
        ###  Grad_normal (u)  !!! Si on le remplace par grade au centre ?? 
                
        ux = ugradfacex[i]
        uy = ugradfacey[i]
        uz = ugradfacez[i]
        
        vx = vgradfacex[i]
        vy = vgradfacey[i]
        vz = vgradfacez[i]
        
        wx = wgradfacex[i]
        wy = wgradfacey[i]
        wz = wgradfacez[i]

        ###### #  Grad_normal (u)  ####### 
        
        unx = (ux*normal[i,0] +  vx*normal[i,1] + wx*normal[i,2])/np.sqrt(normal[i,0]*normal[i,0]+normal[i,1]*normal[i,1]+normal[i,2]*normal[i,2]) 
        uny = (uy*normal[i,0] +  vy*normal[i,1] + wy*normal[i,2])/np.sqrt(normal[i,0]*normal[i,0]+normal[i,1]*normal[i,1]+normal[i,2]*normal[i,2]) 
        unz = (uz*normal[i,0] +  vz*normal[i,1] + wz*normal[i,2])/np.sqrt(normal[i,0]*normal[i,0]+normal[i,1]*normal[i,1]+normal[i,2]*normal[i,2]) 
        
        u_n = (up*normal[i,0] +  vp*normal[i,1] + wp*normal[i,2])/np.sqrt(normal[i,0]*normal[i,0]+normal[i,1]*normal[i,1]+normal[i,2]*normal[i,2]) 

        #######  Rhou_n, Rhou_t1 , Rhou_t2   ####### 
        
        rhou_n = (rhoup*normal[i,0] +  rhovp*normal[i,1] + rhowp*normal[i,2])/np.sqrt(normal[i,0]*normal[i,0]+normal[i,1]*normal[i,1]+normal[i,2]*normal[i,2]) 
        rhou_t = (rhoup*tangent[i,0] +  rhovp*tangent[i,1] + rhowp*tangent[i,2])/np.sqrt(tangent[i,0]*tangent[i,0]+tangent[i,1]*tangent[i,1]+tangent[i,2]*tangent[i,2]) 
        rhou_b = (rhoup*binormal[i,0] +  rhovp*binormal[i,1] + rhowp*binormal[i,2])/np.sqrt(binormal[i,0]*binormal[i,0]+binormal[i,1]*binormal[i,1]+binormal[i,2]*binormal[i,2]) 
        
        #######  Grad_normal (u_n)  ####### 
         
        Un_grad = (unx*normal[i,0] + uny*normal[i,1] + unz*normal[i,2])/np.sqrt(normal[i,0]*normal[i,0]+normal[i,1]*normal[i,1]+normal[i,2]*normal[i,2]) 
        
        Pn_grad = (Pgradfacex[i]*normal[i,0] + Pgradfacey[i]*normal[i,1] + Pgradfacez[i]*normal[i,2])/np.sqrt(normal[i,0]*normal[i,0]+normal[i,1]*normal[i,1]+normal[i,2]*normal[i,2]) 
        
        #######  Predictor   ####### 
        
        rhopre = rhop    - alphaf*d_t*rhop*Un_grad
        rhounp = rhou_n  - alphaf*d_t*(rhou_n*Un_grad + Pn_grad) 
        rhoutp = rhou_t  - alphaf*d_t*(rhou_t*Un_grad) 
        rhoubp = rhou_b  - alphaf*d_t*(rhou_b*Un_grad)
        rhoEpre= rhoEp   - alphaf*d_t*(rhoEp*Un_grad + u_n*Pn_grad + Pp*Un_grad)
        
        #######  Rhou, Rhov , Rhow ####### 
        
        rho_p[i]  = rhopre
        rhou_p[i] = (rhounp*inormal[i,0] + rhoutp*inormal[i,1] + rhoubp*inormal[i,2])/np.sqrt(inormal[i,0]*inormal[i,0]+inormal[i,1]*inormal[i,1]+inormal[i,2]*inormal[i,2]) 
        rhov_p[i] = (rhounp*itangent[i,0] + rhoutp*itangent[i,1] + rhoubp*itangent[i,2])/np.sqrt(itangent[i,0]*itangent[i,0]+itangent[i,1]*itangent[i,1]+itangent[i,2]*itangent[i,2]) 
        rhow_p[i] = (rhounp*ibinormal[i,0] + rhoutp*ibinormal[i,1] + rhoubp*ibinormal[i,2])/np.sqrt(ibinormal[i,0]*ibinormal[i,0]+ibinormal[i,1]*ibinormal[i,1]+ibinormal[i,2]*ibinormal[i,2]) 
        rhoE_p[i] = rhoEpre
        P_p[i]    = (gamma-1)*(rhoE_p[i] - 0.5*(rhou_p[i]*rhou_p[i] + rhov_p[i]*rhov_p[i] + rhow_p[i]*rhow_p[i])/rho_p[i])

def _initialisation_euler_3d(rho: 'float64[:]', P: 'float64[:]', rhou: 'float64[:]', rhov: 'float64[:]', rhow: 'float64[:]', rhoE: 'float64[:]', center: 'float64[:,:]', rho1: 'float64', rho2: 'float64', P1: 'float64', P2: 'float64', u1: 'float64', u2: 'float64', xm: 'float64', gamma: 'float64', choix: 'int32'):

    nbelements = len(center)
    
    if choix == 0:
        for i in range(nbelements):
            xcent = center[i][0]
            # ycent = center[i][1]
            # zcent = center[i][2]
            if xcent<xm:
                rho[i]   = rho1
                rhou[i]  = u1*rho1
                P[i]     = P1
                rhov[i]  = 0.
                rhow[i]  = 0.
            else:
                rho[i]   = rho2
                rhou[i]  = u2*rho2
                P[i]     = P2
                rhov[i]  = 0.
                rhow[i]  = 0.
                
            rhoE[i]  = 0.5*(rhou[i]**2 + rhov[i]**2 + rhow[i]**2)/rho[i] + P[i]/(gamma-1)
            
    elif choix == 1:
        
        for i in range(nbelements):
            rho[i]   = 1.0948
            P[i]     = 90808.0041
            rhou[i]  = 0.
            rhov[i]  = 0.
            rhow[i]  = 0.
            rhoE[i]  = 0.5*(rhou[i]**2 + rhov[i]**2+rhow[i]**2)/rho[i] + P[i]/(gamma-1)

    elif choix == 2:
        
        sigma=0.1
        for i in range(nbelements):
            xcent    = center[i][0]
            rho[i]   = np.exp(-1.*((xcent-0.5)**2) / sigma) 
            P[i]     = 1.
            rhou[i]  = 0.*rho[i]
            rhov[i]  = 0.
            rhow[i]  = 0.
            rhoE[i]  = 0.5*(rhou[i]**2+rhov[i]**2+rhow[i]**2)/rho[i]+P[i]/(gamma-1)                  

    elif choix == 3:
        
        for i in range(nbelements):
            
            xcent = center[i][0]
            ycent = center[i][1]
            zcent = center[i][2]
            
            xm =  np.sqrt(xcent**2 + ycent**2 + zcent**2)
            
            if xm < 0.5:
                rho[i]   = 1.0
                P[i]     = 1
                rhou[i]  = 0
                rhov[i]  = 0
                rhow[i]  = 0.
            else:
                rho[i]   = 0.125
                P[i]     = 0.1
                rhou[i]  = 0
                rhov[i]  = 0
                rhow[i]  = 0.
#            
            u = rhou[i]/rho[i]
            v = rhov[i]/rho[i]
            w = rhow[i]/rho[i] 
            
            rhoE[i]  = 0.5*rho[i]*(u**2 + v**2 + w**2)  + P[i]/(gamma-1)  

def _ghost_value_Neumann3D(rhog: 'float64[:]', Pg: 'float64[:]', rhoug: 'float64[:]', rhovg: 'float64[:]', rhowg: 'float64[:]', ug: 'float64[:]', vg: 'float64[:]', wg: 'float64[:]', rhoEg: 'float64[:]', rhoc: 'float64[:]', Pc: 'float64[:]', rhouc: 'float64[:]', rhovc: 'float64[:]', rhowc: 'float64[:]', rhoEc: 'float64[:]', cellid: 'int32[:,:]', name: 'uint32[:]', normal: 'float64[:,:]', mesure: 'float64[:]'):
    
    nbface = len(cellid)
    
    for i in range(nbface):
        
       
        if name[i] == 1:
            
            rhog[i]  = rhoc[cellid[i][0]]
            rhoug[i] = rhouc[cellid[i][0]] 
            rhovg[i] = rhovc[cellid[i][0]] 
            rhowg[i] = rhowc[cellid[i][0]] 
            rhoEg[i] = rhoEc[cellid[i][0]]
            Pg[i]    = Pc[cellid[i][0]]
            ug[i]    = rhoug[i]/rhog[i]
            vg[i]    = rhovg[i]/rhog[i]
            wg[i]    = rhowg[i]/rhog[i]  
            
        elif name[i] == 2:
            
            rhog[i]  = rhoc[cellid[i][0]]
            rhoug[i] = rhouc[cellid[i][0]] 
            rhovg[i] = rhovc[cellid[i][0]] 
            rhowg[i] = rhowc[cellid[i][0]] 
            rhoEg[i] = rhoEc[cellid[i][0]]
            Pg[i]    = Pc[cellid[i][0]]
            ug[i]    = rhoug[i]/rhog[i]
            vg[i]    = rhovg[i]/rhog[i]
            wg[i]    = rhowg[i]/rhog[i]    

        elif (name[i] == 3 or name[i] == 4 or name[i] == 5 or name[i] == 6):
            
            
            rhog[i]  = rhoc[cellid[i][0]]
            rhoug[i] = rhouc[cellid[i][0]] 
            rhovg[i] = rhovc[cellid[i][0]] 
            rhowg[i] = rhowc[cellid[i][0]] 
            rhoEg[i] = rhoEc[cellid[i][0]]
            Pg[i]    = Pc[cellid[i][0]]
            ug[i]    = rhoug[i]/rhog[i]
            vg[i]    = rhovg[i]/rhog[i]
            wg[i]    = rhowg[i]/rhog[i]  
            
def _ghost_value_TubeSchok3D(rhog: 'float64[:]', Pg: 'float64[:]', rhoug: 'float64[:]', rhovg: 'float64[:]', rhowg: 'float64[:]', ug: 'float64[:]', vg: 'float64[:]', wg: 'float64[:]', rhoEg: 'float64[:]', rhoc: 'float64[:]', Pc: 'float64[:]', rhouc: 'float64[:]', rhovc: 'float64[:]', rhowc: 'float64[:]', rhoEc: 'float64[:]', cellid: 'int32[:,:]', name: 'uint32[:]', normal: 'float64[:,:]', mesure: 'float64[:]'):
    
    nbface = len(cellid)
    s_n = np.zeros(3)
    
    for i in range(nbface):
        
       
        if name[i] == 1:
            
            rhog[i]  = rhoc[cellid[i][0]]
            rhoug[i] = rhouc[cellid[i][0]] 
            rhovg[i] = rhovc[cellid[i][0]] 
            rhowg[i] = rhowc[cellid[i][0]] 
            rhoEg[i] = rhoEc[cellid[i][0]]
            Pg[i]    = Pc[cellid[i][0]]
            ug[i]    = rhoug[i]/rhog[i]
            vg[i]    = rhovg[i]/rhog[i]
            wg[i]    = rhowg[i]/rhog[i]  
            
        
        elif name[i] == 2:
            
            rhog[i]  = rhoc[cellid[i][0]]
            rhoug[i] = rhouc[cellid[i][0]] 
            rhovg[i] = rhovc[cellid[i][0]] 
            rhowg[i] = rhowc[cellid[i][0]] 
            rhoEg[i] = rhoEc[cellid[i][0]]
            Pg[i]    = Pc[cellid[i][0]]
            ug[i]    = rhoug[i]/rhog[i]
            vg[i]    = rhovg[i]/rhog[i]
            wg[i]    = rhowg[i]/rhog[i]    

        elif (name[i] == 3 or name[i] == 4 or name[i] == 5 or name[i] == 6):
            
            u_i = rhouc[cellid[i][0]]/rhoc[cellid[i][0]]
            v_i = rhovc[cellid[i][0]]/rhoc[cellid[i][0]]
            w_i = rhowc[cellid[i][0]]/rhoc[cellid[i][0]]
       
            s_n[:] = normal[i][:]/np.sqrt(normal[i][0]*normal[i][0]+normal[i][1]*normal[i][1]+normal[i][2]*normal[i][2])
            
            u_g = u_i-2*(u_i*s_n[0]+v_i*s_n[1]+w_i*s_n[2])*s_n[0]
            v_g = v_i-2*(u_i*s_n[0]+v_i*s_n[1]+w_i*s_n[2])*s_n[1]
            w_g = w_i-2*(u_i*s_n[0]+v_i*s_n[1]+w_i*s_n[2])*s_n[2]
                      
            rhog[i]  = rhoc[cellid[i][0]]
            rhoug[i] = rhoc[cellid[i][0]] * u_g
            rhovg[i] = rhoc[cellid[i][0]] * v_g
            rhovg[i] = rhoc[cellid[i][0]] * w_g
            
            rhoEg[i] = rhoEc[cellid[i][0]]
            Pg[i]    = Pc[cellid[i][0]]
            ug[i]    = rhoug[i]/rhog[i]
            vg[i]    = rhovg[i]/rhog[i]
            wg[i]    = rhowg[i]/rhog[i]

def _ghost_value_gamm(rhog: 'float64[:]', Pg: 'float64[:]', rhoug: 'float64[:]', rhovg: 'float64[:]', rhowg: 'float64[:]', ug: 'float64[:]', vg: 'float64[:]', wg: 'float64[:]', rhoEg: 'float64[:]', rhoc: 'float64[:]', Pc: 'float64[:]', rhouc: 'float64[:]', rhovc: 'float64[:]', rhowc: 'float64[:]', rhoEc: 'float64[:]', cellid: 'int32[:,:]', name: 'uint32[:]', normal: 'float64[:,:]', mesure: 'float64[:]'):
    
    nbface = len(cellid)
    s_n = np.zeros(3)
    
    for i in range(nbface):
        
       
        if name[i] == 1:
                                    
            kappa = 1.4
            p0    = 101391.8555
            rho0  = 1.1845
            p = min(p0,Pc[cellid[i][0]])
            M2 = ((p / p0)**(-(kappa - 1.0) / kappa) - 1.0) * 2.0 / (kappa - 1.0)
            tmp = 1.0 + (kappa - 1.0) * 0.5 * M2
            rho = rho0 * tmp**(-1.0 /(kappa - 1.0))
            a2 = kappa * p / rho
            rhoVel = rho*np.sqrt(M2*a2)
            
            e = p / (kappa - 1) + 0.5 * (rhoVel**2)/rho 
            
            rhog[i]  = rho
            Pg[i]    = p
            rhoEg[i] = rho*e
            rhoug[i] = rhoVel
            rhovg[i] = 0.
            rhowg[i] = 0.
            ug[i] = rhoug[i]/rhog[i]
            vg[i] = rhovg[i]/rhog[i]
            wg[i] = rhowg[i]/rhog[i]            
        
        elif name[i] == 2:
            
             kappa = 1.4
             MaIs  = 0.675
             p0=101391.8555
             p = p0*((1. + (kappa - 1.) / 2. * MaIs*MaIs)**(kappa / (1. - kappa)))
             
             he = p / (kappa - 1.) + 0.5*(rhouc[cellid[i][0]]**2 + rhovc[cellid[i][0]]**2 + rhowc[cellid[i][0]]**2)/rhoc[cellid[i][0]] 
             
             rhog[i]  = rhoc[cellid[i][0]]            
             rhoug[i] = rhouc[cellid[i][0]]
             rhovg[i] = rhovc[cellid[i][0]]
             rhowg[i] = rhowc[cellid[i][0]]
             rhoEg[i] = he
             Pg[i] = p
             ug[i] = rhoug[i]/rhog[i]
             vg[i] = rhovg[i]/rhog[i]
             wg[i] = rhowg[i]/rhog[i]
    
        elif (name[i] == 3 or name[i] == 4):
            
            u_i = rhouc[cellid[i][0]]/rhoc[cellid[i][0]]
            v_i = rhovc[cellid[i][0]]/rhoc[cellid[i][0]]
            w_i = rhowc[cellid[i][0]]/rhoc[cellid[i][0]]
       
            s_n[:] = normal[i][:]/np.sqrt(normal[i][0]*normal[i][0]+normal[i][1]*normal[i][1]+normal[i][2]*normal[i][2])
            
            u_g = u_i-2*(u_i*s_n[0]+v_i*s_n[1]+w_i*s_n[2])*s_n[0]
            v_g = v_i-2*(u_i*s_n[0]+v_i*s_n[1]+w_i*s_n[2])*s_n[1]
            w_g = w_i-2*(u_i*s_n[0]+v_i*s_n[1]+w_i*s_n[2])*s_n[2]
                      
            rhog[i]  = rhoc[cellid[i][0]]
            rhoug[i] = rhoc[cellid[i][0]] * u_g
            rhovg[i] = rhoc[cellid[i][0]] * v_g
            rhovg[i] = rhoc[cellid[i][0]] * w_g
            
            rhoEg[i] = rhoEc[cellid[i][0]]
            Pg[i]    = Pc[cellid[i][0]]
            ug[i]    = rhoug[i]/rhog[i]
            vg[i]    = rhovg[i]/rhog[i]
            wg[i]    = rhowg[i]/rhog[i]
        elif (name[i] == 5 or name[i] == 6):
            
            rhog[i]  = rhoc[cellid[i][0]]
            rhoug[i] = rhouc[cellid[i][0]] 
            rhovg[i] = rhovc[cellid[i][0]] 
            rhowg[i] = rhowc[cellid[i][0]] 
            rhoEg[i] = rhoEc[cellid[i][0]]
            Pg[i]    = Pc[cellid[i][0]]
            ug[i]    = rhoug[i]/rhog[i]
            vg[i]    = rhovg[i]/rhog[i]
            wg[i]    = rhowg[i]/rhog[i]             

def _halghost_value_Neumann3D(rhog: 'float64[:]', Pg: 'float64[:]', rhoug: 'float64[:]', rhovg: 'float64[:]', rhowg: 'float64[:]', ug: 'float64[:]', vg: 'float64[:]', wg: 'float64[:]', rhoEg: 'float64[:]', rho_halo: 'float64[:]', P_halo: 'float64[:]', rhou_halo: 'float64[:]', rhov_halo: 'float64[:]', rhow_halo: 'float64[:]', rhoE_halo: 'float64[:]', cellid: 'int32[:,:]', name: 'uint32[:]', normal: 'float64[:,:]', halonodes: 'uint32[:]', haloghostcenter: 'float64[:,:,:]', haloghostfaceinfo: 'float64[:,:,:]'):
    
    for i in halonodes:
        for j in range(len(haloghostcenter[i])):
            if haloghostcenter[i][j][-1] != -1:
                
                cellhalo  = int(haloghostcenter[i][j][-3])
                cellghost = int(haloghostcenter[i][j][-1])
                
                if haloghostcenter[i][j][-2] == 1:
                    
                    rhog[cellghost]  = rho_halo[cellhalo]
                    rhoug[cellghost] = rhoug[cellghost] 
                    rhovg[cellghost] = rhovg[cellghost]
                    rhowg[cellghost] = rhowg[cellghost]
                    rhoEg[cellghost] = rhoE_halo[cellhalo]
                    ug[cellghost]    = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost]    = rhovg[cellghost]/rhog[cellghost]
                    wg[cellghost]    = rhowg[cellghost]/rhog[cellghost]
                                       
                    
                elif haloghostcenter[i][j][-2] == 2:
                    
                    rhog[cellghost]  = rho_halo[cellhalo]
                    rhoug[cellghost] = rhoug[cellghost] 
                    rhovg[cellghost] = rhovg[cellghost]
                    rhowg[cellghost] = rhowg[cellghost]
                    rhoEg[cellghost] = rhoE_halo[cellhalo]
                    ug[cellghost]    = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost]    = rhovg[cellghost]/rhog[cellghost]
                    wg[cellghost]    = rhowg[cellghost]/rhog[cellghost]
                    
                    
                    
                elif (haloghostcenter[i][j][-2] == 3 or haloghostcenter[i][j][-2] == 4 or haloghostcenter[i][j][-2] == 5 or haloghostcenter[i][j][-2] == 6):  
                    
  
                    rhog[cellghost]  = rho_halo[cellhalo]
                    rhoug[cellghost] = rhoug[cellghost] 
                    rhovg[cellghost] = rhovg[cellghost]
                    rhowg[cellghost] = rhowg[cellghost]
                    rhoEg[cellghost] = rhoE_halo[cellhalo]
                    ug[cellghost]    = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost]    = rhovg[cellghost]/rhog[cellghost]
                    wg[cellghost]    = rhowg[cellghost]/rhog[cellghost]
                    
def _halghost_value_TubeSchok3D(rhog: 'float64[:]', Pg: 'float64[:]', rhoug: 'float64[:]', rhovg: 'float64[:]', rhowg: 'float64[:]', ug: 'float64[:]', vg: 'float64[:]', wg: 'float64[:]', rhoEg: 'float64[:]', rho_halo: 'float64[:]', P_halo: 'float64[:]', rhou_halo: 'float64[:]', rhov_halo: 'float64[:]', rhow_halo: 'float64[:]', rhoE_halo: 'float64[:]', cellid: 'int32[:,:]', name: 'uint32[:]', normal: 'float64[:,:]', halonodes: 'uint32[:]', haloghostcenter: 'float64[:,:,:]', haloghostfaceinfo: 'float64[:,:,:]'):
    
    s_n = np.zeros(3) 
    for i in halonodes:
        for j in range(len(haloghostcenter[i])):
            if haloghostcenter[i][j][-1] != -1:
                
                cellhalo  = int(haloghostcenter[i][j][-3])
                cellghost = int(haloghostcenter[i][j][-1])
                
                if haloghostcenter[i][j][-2] == 1:
                    
                    rhog[cellghost]  = rho_halo[cellhalo]
                    rhoug[cellghost] = rhoug[cellghost] 
                    rhovg[cellghost] = rhovg[cellghost]
                    rhowg[cellghost] = rhowg[cellghost]
                    rhoEg[cellghost] = rhoE_halo[cellhalo]
                    ug[cellghost]    = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost]    = rhovg[cellghost]/rhog[cellghost]
                    wg[cellghost]    = rhowg[cellghost]/rhog[cellghost]
                                       
                    
                elif haloghostcenter[i][j][-2] == 2:
                    
                    rhog[cellghost]  = rho_halo[cellhalo]
                    rhoug[cellghost] = rhoug[cellghost] 
                    rhovg[cellghost] = rhovg[cellghost]
                    rhowg[cellghost] = rhowg[cellghost]
                    rhoEg[cellghost] = rhoE_halo[cellhalo]
                    ug[cellghost]    = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost]    = rhovg[cellghost]/rhog[cellghost]
                    wg[cellghost]    = rhowg[cellghost]/rhog[cellghost]
                    
                    
                    
                elif (haloghostcenter[i][j][-2] == 3 or haloghostcenter[i][j][-2] == 4 or haloghostcenter[i][j][-2] == 5 or haloghostcenter[i][j][-2] == 6):  
                    
  
                    u_i = rhou_halo[cellhalo]/rho_halo[cellhalo]
                    v_i = rhov_halo[cellhalo]/rho_halo[cellhalo]
                    w_i = rhow_halo[cellhalo]/rho_halo[cellhalo]
                    
#                   mesure = np.sqrt(haloghostfaceinfo[i][j][3]**2 + haloghostfaceinfo[i][j][4]**2 + haloghostfaceinfo[i][j][5]**2)
                    
                    s_n[0] = haloghostfaceinfo[i][j][3] #/ mesure
                    s_n[1] = haloghostfaceinfo[i][j][4] #/ mesure
                    s_n[2] = haloghostfaceinfo[i][j][5] #/ mesure
                    
                    
                    u_g = u_i-2*(u_i*s_n[0]+v_i*s_n[1]+w_i*s_n[2])*s_n[0]
                    v_g = v_i-2*(u_i*s_n[0]+v_i*s_n[1]+w_i*s_n[2])*s_n[1]
                    w_g = w_i-2*(u_i*s_n[0]+v_i*s_n[1]+w_i*s_n[2])*s_n[2]
                                      
                    rhog[cellghost] = rho_halo[cellhalo]
                    rhoug[cellghost] = rho_halo[cellhalo] * u_g
                    rhovg[cellghost] = rho_halo[cellhalo] * v_g
                    rhowg[cellghost] = rho_halo[cellhalo] * w_g
                    
                    rhoEg[cellghost] = rhoE_halo[cellhalo]
                    Pg[cellghost] = P_halo[cellhalo]
                    ug[cellghost] = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost] = rhovg[cellghost]/rhog[cellghost]
                    wg[cellghost] = rhowg[cellghost]/rhog[cellghost]
                    
            
def _halghost_value_gamm(rhog: 'float64[:]', Pg: 'float64[:]', rhoug: 'float64[:]', rhovg: 'float64[:]', rhowg: 'float64[:]', ug: 'float64[:]', vg: 'float64[:]', wg: 'float64[:]', rhoEg: 'float64[:]', rho_halo: 'float64[:]', P_halo: 'float64[:]', rhou_halo: 'float64[:]', rhov_halo: 'float64[:]', rhow_halo: 'float64[:]', rhoE_halo: 'float64[:]', cellid: 'int32[:,:]', name: 'uint32[:]', normal: 'float64[:,:]', halonodes: 'uint32[:]', haloghostcenter: 'float64[:,:,:]', haloghostfaceinfo: 'float64[:,:,:]'):
    s_n = np.zeros(3)
    for i in halonodes:
        for j in range(len(haloghostcenter[i])):
            if haloghostcenter[i][j][-1] != -1:
                
                cellhalo  = int(haloghostcenter[i][j][-3])
                cellghost = int(haloghostcenter[i][j][-1])
                
                if haloghostcenter[i][j][-2] == 1:
                    
                    kappa = 1.4
                    p0    = 101391.8555
                    rho0  = 1.1845
                    p     = min(p0,P_halo[cellhalo])
                    M2    = ((p / p0)**(-(kappa - 1.0) / kappa) - 1.0) * 2.0 / (kappa - 1.0)
                    tmp   = 1.0 + (kappa - 1.0) * 0.5 * M2
                    rho   = rho0 * tmp**(-1.0 /(kappa - 1.0))
                    a2    = kappa * p / rho
                    rhoVel = rho*np.sqrt(M2 * a2)
                    e = p / (kappa - 1) + 0.5 * (rhoVel**2)/rho 
    
                    rhog[cellghost]  = rho
                    Pg[cellghost]    = p
                    rhoEg[cellghost] = rho*e
                    rhoug[cellghost] = rhoVel
                    rhovg[cellghost] = 0.
                    rhowg[cellghost] = 0.
                    
                    ug[cellghost]    = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost]    = rhovg[cellghost]/rhog[cellghost]
                    wg[cellghost]    = rhowg[cellghost]/rhog[cellghost]
                    
                    
                elif haloghostcenter[i][j][-2] == 2:
                    
                    kappa = 1.4
                    MaIs  = 0.675
                    p0    = 101391.8555
                    p     = p0*((1. + (kappa - 1.) / 2. * MaIs*MaIs)**(kappa / (1. - kappa)))
                   
                    he = p / (kappa - 1.) + 0.5 * (rhou_halo[cellhalo]**2 + rhov_halo[cellhalo]**2 + rhow_halo[cellhalo]**2)/rho_halo[cellhalo] 
                   
                    rhog[cellghost]  = rho_halo[cellhalo]            
                    rhoug[cellghost] = rhou_halo[cellhalo] 
                    rhovg[cellghost] = rhov_halo[cellhalo]
                    rhowg[cellghost] = rhow_halo[cellhalo]
                    
                    rhoEg[cellghost] = he
                    Pg[cellghost]    = p
                    ug[cellghost]    = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost]    = rhovg[cellghost]/rhog[cellghost]
                    wg[cellghost]    = rhowg[cellghost]/rhog[cellghost]
                    
                    
                elif (haloghostcenter[i][j][-2] == 3 or haloghostcenter[i][j][-2] == 4):  
                    
                    u_i = rhou_halo[cellhalo]/rho_halo[cellhalo]
                    v_i = rhov_halo[cellhalo]/rho_halo[cellhalo]
                    w_i = rhow_halo[cellhalo]/rho_halo[cellhalo]
                    
                    mesure = np.sqrt(haloghostfaceinfo[i][j][3]**2 + haloghostfaceinfo[i][j][4]**2 + haloghostfaceinfo[i][j][5]**2)
                    
                    s_n[0] = haloghostfaceinfo[i][j][3] / mesure
                    s_n[1] = haloghostfaceinfo[i][j][4] / mesure
                    s_n[2] = haloghostfaceinfo[i][j][5] / mesure
                    
                    
                    u_g = u_i-2*(u_i*s_n[0]+v_i*s_n[1]+w_i*s_n[2])*s_n[0]
                    v_g = v_i-2*(u_i*s_n[0]+v_i*s_n[1]+w_i*s_n[2])*s_n[1]
                    w_g = w_i-2*(u_i*s_n[0]+v_i*s_n[1]+w_i*s_n[2])*s_n[2]
                                      
                    rhog[cellghost] = rho_halo[cellhalo]
                    rhoug[cellghost] = rho_halo[cellhalo] * u_g
                    rhovg[cellghost] = rho_halo[cellhalo] * v_g
                    rhowg[cellghost] = rho_halo[cellhalo] * w_g
                    
                    rhoEg[cellghost] = rhoE_halo[cellhalo]
                    Pg[cellghost] = P_halo[cellhalo]
                    ug[cellghost] = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost] = rhovg[cellghost]/rhog[cellghost]
                    wg[cellghost] = rhowg[cellghost]/rhog[cellghost]
                
                elif haloghostcenter[i][j][-2] == 5 or haloghostcenter[i][j][-2] == 6:  
                                   
                    rhog[cellghost]  = rho_halo[cellhalo]            
                    rhoug[cellghost] = rhou_halo[cellhalo] 
                    rhovg[cellghost] = rhov_halo[cellhalo]
                    rhowg[cellghost] = rhow_halo[cellhalo]
                    
                    rhoEg[cellghost] = rhoE_halo[cellhalo]
                    Pg[cellghost]    = P_halo[cellhalo]
                    ug[cellghost]    = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost]    = rhovg[cellghost]/rhog[cellghost]
                    wg[cellghost]    = rhowg[cellghost]/rhog[cellghost]  
                
def _time_step_euler_3d(rho: 'float64[:]', P: 'float64[:]', rhou: 'float64[:]', rhov: 'float64[:]', rhow: 'float64[:]', cfl: 'float64', normal: 'float64[:,:]', mesure: 'float64[:]', volume: 'float64[:]', faceid: 'int32[:,:]', gamma: 'float64', dt_c: 'float64[:]'):
    
    nbelement =  len(faceid)
    u_n = 0.
  
    for i in range(nbelement):
        lam = 0.
        velson = np.sqrt(gamma*np.fabs(P[i]/rho[i]))
        for j in range(4):
            u_n = np.fabs((rhou[i]*normal[faceid[i][j]][0] + rhov[i]*normal[faceid[i][j]][1] + rhow[i]*normal[faceid[i][j]][2])/rho[i])/np.sqrt(normal[faceid[i][j]][0]*normal[faceid[i][j]][0]+normal[faceid[i][j]][1]*normal[faceid[i][j]][1]+normal[faceid[i][j]][2]*normal[faceid[i][j]][2])
            lam_convect = u_n + velson
            lam += lam_convect * mesure[faceid[i][j]]
               
        dt_c[i]  = cfl * volume[i]/lam            

def _compute_flux_euler_3d_fvc(rhop: 'float64', Pp: 'float64', rhoup: 'float64', rhovp: 'float64', rhowp: 'float64', rhoEp: 'float64', normal: 'float64[:]', mesure: 'float64'):
          
        flux_rho   = (rhoup*normal[0] + rhovp*normal[1] + rhowp*normal[2])
        flux_rhou  = ((rhoup*rhoup/rhop+Pp)*normal[0]   + (rhoup*rhovp/rhop)*normal[1]    + (rhoup*rhowp/rhop)*normal[2])
        flux_rhov  = ((rhovp*rhoup/rhop)*normal[0]     + (rhovp*rhovp/rhop+Pp)*normal[1] + (rhovp*rhowp/rhop)*normal[2])
        flux_rhow  = ((rhowp*rhoup/rhop)*normal[0]     + (rhowp*rhovp/rhop)*normal[1]    + (rhowp*rhowp/rhop+Pp)*normal[2])
        flux_rhoE  = (rhoup/rhop*(rhoEp+Pp)*normal[0]   + rhovp/rhop*(rhoEp+Pp)*normal[1] + rhowp/rhop*(rhoEp+Pp)*normal[2])
        
        return flux_rho,flux_rhou,flux_rhov,flux_rhow,flux_rhoE     

######
def _compute_flux_euler_3d_rusanov(rhol: 'float64', Pl: 'float64', rhoul: 'float64', rhovl: 'float64', rhowl: 'float64', rhoEl: 'float64', rhor: 'float64', Pr: 'float64', rhour: 'float64', rhovr: 'float64', rhowr: 'float64', rhoEr: 'float64', normal: 'float64[:]', mesure: 'float64', gamma: 'float64'):
    
    ql = (rhoul*normal[0] + rhovl*normal[1] + rhowl*normal[2])
    qr = (rhour*normal[0] + rhovl*normal[1] + rhowl*normal[2])
    cl = np.sqrt(gamma*Pl/rhol)
    cr = np.sqrt(gamma*Pr/rhor)
    
    
    fl_rho  = (rhoul*normal[0] + rhovl*normal[1] + rhowl*normal[2])
    fl_rhou = ((rhoul*rhoul/rhol+Pl)*normal[0] + (rhoul*rhovl/rhol)*normal[1] + (rhoul*rhowl/rhol)*normal[2] )
    fl_rhov = ((rhovl*rhoul/rhol)*normal[0] + (rhovl*rhovl/rhol+Pl)*normal[1] + (rhovl*rhowl/rhol)*normal[2])
    fl_rhow = ((rhoul*rhowl/rhol)*normal[0] + (rhovl*rhowl/rhol)*normal[1] +  (rhowl*rhowl/rhol+Pl)*normal[2] )
    fl_rhoE = (rhoul/rhol*(rhoEl+Pl)*normal[0] + rhovl/rhol*(rhoEl+Pl)*normal[1] + rhowl/rhol*(rhoEl+Pl)*normal[2])
    
    
    fr_rho  = (rhour*normal[0] + rhovr*normal[1] + rhowr*normal[2])
    fr_rhou = ((rhour*rhour/rhor+Pr)*normal[0] + (rhour*rhovr/rhor)*normal[1] + (rhour*rhowr/rhor)*normal[2] )
    fr_rhov = ((rhovr*rhour/rhor)*normal[0] + (rhovr*rhovr/rhor+Pr)*normal[1] + (rhovr*rhowr/rhor)*normal[2])
    fr_rhow = ((rhour*rhowr/rhor)*normal[0] + (rhovr*rhowr/rhor)*normal[1] +  (rhowr*rhowr/rhor+Pr)*normal[2] )
    fr_rhoE = (rhour/rhor*(rhoEr+Pr)*normal[0] + rhovr/rhor*(rhoEr+Pr)*normal[1] + rhowr/rhor*(rhoEr+Pr)*normal[2])
    
    
    lambdal1 = np.fabs((ql)/rhol - cl)
    lambdal2 = np.fabs((ql)/rhol)
    lambdal3 = np.fabs((ql)/rhol + cl)

    lambdar1 = np.fabs((qr)/rhor - cr)
    lambdar2 = np.fabs((qr)/rhor)
    lambdar3 = np.fabs((qr)/rhor + cr) 
    
   
    Ll = max([lambdal1,lambdal2,lambdal3])
    Lr = max([lambdar1,lambdar2,lambdar3])
    S = 0.
    
    if (Ll > Lr):
        S = Ll
    else:
        S = Lr
   
    flux_rho  = (0.5 * (fl_rho + fr_rho) -  0.5 * S  * (rhor - rhol))*mesure
    flux_rhou = (0.5 * (fl_rhou + fr_rhou) - 0.5 * S * (rhour - rhoul))*mesure
    flux_rhov = (0.5 * (fl_rhov + fr_rhov) - 0.5 * S * (rhovr - rhovl))*mesure
    flux_rhow = (0.5 * (fl_rhow + fr_rhow) - 0.5 * S * (rhowr - rhowl))*mesure
    flux_rhoE = (0.5 * (fl_rhoE + fr_rhoE) - 0.5 * S * (rhoEr - rhoEl))*mesure
    

    return flux_rho,flux_rhou,flux_rhov,flux_rhow,flux_rhoE 
######################
def _compute_flux_euler_3d_Roe(rhol: 'float64', Pl: 'float64', rhoul: 'float64', rhovl: 'float64', rhowl: 'float64', rhoEl: 'float64', rhor: 'float64', Pr: 'float64', rhour: 'float64', rhovr: 'float64', rhowr: 'float64', rhoEr: 'float64', normal: 'float64[:]', mesure: 'float64', gamma: 'float64', tangent: 'float64[:]', binormal: 'float64[:]'):
 
    normal[:]=normal[:]/np.sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2])
    tangent[:]=tangent[:]/np.sqrt(tangent[0]*tangent[0]+tangent[1]*tangent[1]+tangent[2]*tangent[2])
    binormal[:]=binormal[:]/np.sqrt(binormal[0]*binormal[0]+binormal[1]*binormal[1]+binormal[2]*binormal[2])     
    ul =  rhoul/rhol
    ur =  rhour/rhor
        
    vl =  rhovl/rhol
    vr =  rhovr/rhor
    
    wl =  rhowl/rhol
    wr =  rhowr/rhor
    
    Hl  =  (rhoEl + Pl)/rhol
    Hr  =  (rhoEr + Pr)/rhor
    
    DD = np.sqrt(rhol) + np.sqrt(rhor)
    
    rhostar =  np.sqrt(rhol*rhor)
    ustar  =   (np.sqrt(rhol)*ul + np.sqrt(rhor)*ur)/DD
    vstar  =   (np.sqrt(rhol)*vl + np.sqrt(rhor)*vr)/DD
    wstar  =   (np.sqrt(rhol)*wl + np.sqrt(rhor)*wr)/DD
    Hstar  =   (np.sqrt(rhol)*Hl + np.sqrt(rhor)*Hr)/DD
    
    VV     =  ustar**2 + vstar**2 + wstar**2
    
    uueta  = ustar*normal[0] + vstar*normal[1] + wstar*normal[2]   
    uutang = ustar*tangent[0] + vstar*tangent[1] + wstar*tangent[2]
    uubino = ustar*binormal[0] + vstar*binormal[1] + wstar*binormal[2] 
    
    Estar  = (1/gamma)*Hstar + 0.5*((gamma-1)/gamma)*VV
    Pstar  =  (gamma -1 )*(rhostar*Estar - 0.5*rhostar*VV)
    cc     =  np.sqrt(gamma*Pstar/rhostar)
    
    GG = gamma - 1 
    
    Lam1 = uueta - cc
    Lam2 = uueta
    Lam3 = uueta + cc
    Lam4 = uueta
    Lam5 = uueta
    
 
    LL      = np.zeros((5,5))
    RR      = np.zeros((5,5))
    RR1     = np.zeros((5,5))
    
    LL[0,0] = np.fabs(Lam1)  
    LL[1,1] = np.fabs(Lam2)  
    LL[2,2] = np.fabs(Lam3)  
    LL[3,3] = np.fabs(Lam4) 
    LL[4,4] = np.fabs(Lam5) 
    
    RR[0,0] = 1 
    RR[0,1] = 1 
    RR[0,2] = 1
    RR[0,3] = 0
    RR[0,4] = 0
   
    RR[1,0] = ustar - cc*normal[0]
    RR[1,1] = ustar 
    RR[1,2] = ustar + cc*normal[0]
    RR[1,3] = binormal[0] 
    RR[1,4] = tangent[0]
    
    RR[2,0] = vstar - cc*normal[1]
    RR[2,1] = vstar 
    RR[2,2] = vstar + cc*normal[1]
    RR[2,3] = binormal[1]
    RR[2,4] = tangent[1]
    
    RR[3,0] = wstar - cc*normal[2]
    RR[3,1] = wstar
    RR[3,2] = wstar + cc*normal[2]
    RR[3,3] = binormal[2] 
    RR[3,4] = tangent[2]
 
    RR[4,0] = Hstar - cc*uueta
    RR[4,1] = 0.5*VV
    RR[4,2] = Hstar + cc*uueta
    RR[4,3] = uubino 
    RR[4,4] = uutang 
 
       
    RR1[0,0] =  0.5*((GG/(2*cc**2))*VV + uueta/cc)
    RR1[0,1] = -0.5*((GG/cc**2)*ustar + normal[0]/cc)
    RR1[0,2] = -0.5*((GG/cc**2)*vstar + normal[1]/cc)
    RR1[0,3] = -0.5*((GG/cc**2)*wstar + normal[2]/cc)
    RR1[0,4] =  GG/(2*cc**2)

    RR1[1,0] = 1 - (GG/(2*cc**2))*VV
    RR1[1,1] = (GG/cc**2)*ustar
    RR1[1,2] = (GG/(cc**2))*vstar
    RR1[1,3] = (GG/(cc**2))*wstar
    RR1[1,4] = -GG/cc**2
    
    RR1[2,0] =  0.5*((GG/(2*cc**2))*VV - uueta/cc)
    RR1[2,1] = -0.5*((GG/cc**2)*ustar - normal[0]/cc)
    RR1[2,2] = -0.5*((GG/cc**2)*vstar - normal[1]/cc)
    RR1[2,3] = -0.5*((GG/cc**2)*wstar - normal[2]/cc)
    RR1[2,4] =  GG/(2*cc**2)
    
    RR1[3,0] = -uubino
    RR1[3,1] = binormal[0]
    RR1[3,2] = binormal[1]
    RR1[3,3] = binormal[2]
    RR1[3,4] = 0
    
    RR1[4,0] = -uutang 
    RR1[4,1] = tangent[0]
    RR1[4,2] = tangent[1]
    RR1[4,3] = tangent[2]
    RR1[4,4] = 0
    
    mat1    = np.zeros((5,5))
    ammat   = np.zeros((5,5))
    
    lenmatrix1 = RR.shape[0]
    lenmatrix2 = LL.shape[0]
    
    for i in range(lenmatrix1):
        for j in range(len(LL[0])):
            for kk in range(lenmatrix2):
                mat1[i][j] += RR[i][kk] * LL[kk][j]
                
                
    lenmatrix1 = mat1.shape[0]
    lenmatrix2 = RR1.shape[0]
    
    for i in range(lenmatrix1):
        for j in range(len(RR1[0])):
            for k in range(lenmatrix2):
                
                ammat[i][j] += mat1[i][k] * RR1[k][j]
                
    
    fl_rho  = (rhoul*normal[0] + rhovl*normal[1] + rhowl*normal[2])
    fl_rhou = ((rhoul*rhoul/rhol+Pl)*normal[0] + (rhoul*rhovl/rhol)*normal[1] + (rhoul*rhowl/rhol)*normal[2] )
    fl_rhov = ((rhovl*rhoul/rhol)*normal[0] + (rhovl*rhovl/rhol+Pl)*normal[1] + (rhovl*rhowl/rhol)*normal[2])
    fl_rhow = ((rhoul*rhowl/rhol)*normal[0] + (rhovl*rhowl/rhol)*normal[1] +  (rhowl*rhowl/rhol+Pl)*normal[2] )
    fl_rhoE = (rhoul/rhol*(rhoEl+Pl)*normal[0] + rhovl/rhol*(rhoEl+Pl)*normal[1] + rhowl/rhol*(rhoEl+Pl)*normal[2])
    
    
    fr_rho  = (rhour*normal[0] + rhovr*normal[1] + rhowr*normal[2])
    fr_rhou = ((rhour*rhour/rhor+Pr)*normal[0] + (rhour*rhovr/rhor)*normal[1] + (rhour*rhowr/rhor)*normal[2] )
    fr_rhov = ((rhovr*rhour/rhor)*normal[0] + (rhovr*rhovr/rhor+Pr)*normal[1] + (rhovr*rhowr/rhor)*normal[2])
    fr_rhow = ((rhour*rhowr/rhor)*normal[0] + (rhovr*rhowr/rhor)*normal[1] +  (rhowr*rhowr/rhor+Pr)*normal[2] )
    fr_rhoE = (rhour/rhor*(rhoEr+Pr)*normal[0] + rhovr/rhor*(rhoEr+Pr)*normal[1] + rhowr/rhor*(rhoEr+Pr)*normal[2])
    
    
    w_dif    = np.zeros(5)
    
    w_dif[0] = rhor - rhol
    w_dif[1] = rhour - rhoul
    w_dif[2] = rhovr - rhovl
    w_dif[3] = rhowr - rhowl
    w_dif[4] = rhoEr - rhoEl
    
    rhonew = 0.
    unew = 0.
    vnew = 0.
    wnew = 0.
    Enew = 0.
    
    for i in range(5):
        
        rhonew += ammat[0][i] * w_dif[i]
        unew   += ammat[1][i] * w_dif[i]
        vnew   += ammat[2][i] * w_dif[i]
        wnew   += ammat[3][i] * w_dif[i]
        Enew   += ammat[4][i] * w_dif[i]
        
    u_rho  = rhonew
    u_rhou = unew
    u_rhov = vnew
    u_rhow = wnew
    u_rhoE = Enew  
    
    
    flux_rho  = (0.5 * (fl_rho + fr_rho)   - 0.5*u_rho)*mesure
    flux_rhou = (0.5 * (fl_rhou + fr_rhou) - 0.5*u_rhou)*mesure
    flux_rhov = (0.5 * (fl_rhov + fr_rhov) - 0.5*u_rhov)*mesure
    flux_rhow = (0.5 * (fl_rhow + fr_rhow) - 0.5*u_rhow)*mesure
    flux_rhoE = (0.5 * (fl_rhoE + fr_rhoE) - 0.5*u_rhoE)*mesure
 

    return flux_rho,flux_rhou,flux_rhov,flux_rhow,flux_rhoE         



############"
def _explicitscheme_euler_3d_rusanov(rez_rho: 'float64[:]', rez_rhou: 'float64[:]', rez_rhov: 'float64[:]', rez_rhow: 'float64[:]', rez_rhoE: 'float64[:]', rho_c: 'float64[:]', P_c: 'float64[:]', rhou_c: 'float64[:]', rhov_c: 'float64[:]', rhow_c: 'float64[:]', rhoE_c: 'float64[:]', rho_g: 'float64[:]', P_g: 'float64[:]', rhou_g: 'float64[:]', rhov_g: 'float64[:]', rhow_g: 'float64[:]', rhoE_g: 'float64[:]', rho_h: 'float64[:]', P_h: 'float64[:]', rhou_h: 'float64[:]', rhov_h: 'float64[:]', rhow_h: 'float64[:]', rhoE_h: 'float64[:]', cellidf: 'int32[:,:]', halofid: 'int32[:]', normal: 'float64[:,:]', mesurf: 'float64[:]', name: 'uint32[:]', gamma: 'float64', tangent: 'float64[:,:]', binormal: 'float64[:,:]'):

    nbface = len(cellidf)
       
    rez_rho[:]  = np.zeros(len(rez_rho))
    rez_rhou[:] = np.zeros(len(rez_rhou))
    rez_rhov[:] = np.zeros(len(rez_rhov))
    rez_rhow[:] = np.zeros(len(rez_rhov))
    rez_rhoE[:] = np.zeros(len(rez_rhoE))

    for i in range(nbface):
  
  
  
            norm = normal[i]/mesurf[i]
            mesu = mesurf[i]

                       
            rhol = rho_c[cellidf[i][0]]
            Pl   = P_c[cellidf[i][0]]
            rhoul=rhou_c[cellidf[i][0]]
            rhovl=rhov_c[cellidf[i][0]]
            rhowl=rhow_c[cellidf[i][0]]
            rhoEl=rhoE_c[cellidf[i][0]]
                        
            if name[i] == 0:
         
                rhor=rho_c[cellidf[i][1]]
                Pr=P_c[cellidf[i][1]]
                rhour=rhou_c[cellidf[i][1]]
                rhovr=rhov_c[cellidf[i][1]]
                rhowr=rhow_c[cellidf[i][1]]
                rhoEr=rhoE_c[cellidf[i][1]]
                
                flux_rho,flux_rhou,flux_rhov,flux_rhow,flux_rhoE = _compute_flux_euler_3d_rusanov(rhol,Pl,rhoul,rhovl,rhowl,rhoEl,rhor,Pr,rhour,rhovr,rhowr,rhoEr,norm, mesu,gamma)
                
                rez_rho[cellidf[i][0]] -= flux_rho
                rez_rho[cellidf[i][1]] += flux_rho
                rez_rhou[cellidf[i][0]] -= flux_rhou
                rez_rhou[cellidf[i][1]] += flux_rhou
                rez_rhov[cellidf[i][0]] -= flux_rhov
                rez_rhov[cellidf[i][1]] += flux_rhov
                rez_rhow[cellidf[i][0]] -= flux_rhow
                rez_rhow[cellidf[i][1]] += flux_rhow
                rez_rhoE[cellidf[i][0]] -= flux_rhoE
                rez_rhoE[cellidf[i][1]] += flux_rhoE
            
            elif name[i] == 10:
         
                rhor=rho_h[halofid[i]]
                Pr=P_h[halofid[i]]
                rhour=rhou_h[halofid[i]]
                rhovr=rhov_h[halofid[i]]
                rhowr=rhow_h[halofid[i]]
                rhoEr=rhoE_h[halofid[i]]
                
                flux_rho,flux_rhou,flux_rhov,flux_rhow,flux_rhoE = _compute_flux_euler_3d_rusanov(rhol,Pl,rhoul,rhovl,rhowl,rhoEl,rhor,Pr,rhour,rhovr,rhowr,rhoEr,norm, mesu,gamma)
                
                rez_rho[cellidf[i][0]]  -= flux_rho
                rez_rhou[cellidf[i][0]] -= flux_rhou
                rez_rhov[cellidf[i][0]] -= flux_rhov
                rez_rhow[cellidf[i][0]] -= flux_rhow
                rez_rhoE[cellidf[i][0]] -= flux_rhoE
    
            else:
         
                rhor=rho_g[i]
                Pr = P_g[i]
                rhour=rhou_g[i]
                rhovr=rhov_g[i]
                rhowr=rhow_g[i]
                rhoEr=rhoE_g[i]
                
                flux_rho,flux_rhou,flux_rhov,flux_rhow,flux_rhoE  = _compute_flux_euler_3d_rusanov(rhol,Pl,rhoul,rhovl,rhowl,rhoEl,rhor,Pr,rhour,rhovr,rhowr,rhoEr,norm, mesu,gamma)
                               
                rez_rho[cellidf[i][0]]  -= flux_rho
                rez_rhou[cellidf[i][0]] -= flux_rhou
                rez_rhov[cellidf[i][0]] -= flux_rhov
                rez_rhow[cellidf[i][0]] -= flux_rhow
                rez_rhoE[cellidf[i][0]] -= flux_rhoE             

#############""
def _explicitscheme_euler_3d_Roe(rez_rho: 'float64[:]', rez_rhou: 'float64[:]', rez_rhov: 'float64[:]', rez_rhow: 'float64[:]', rez_rhoE: 'float64[:]', rho_c: 'float64[:]', P_c: 'float64[:]', rhou_c: 'float64[:]', rhov_c: 'float64[:]', rhow_c: 'float64[:]', rhoE_c: 'float64[:]', rho_g: 'float64[:]', P_g: 'float64[:]', rhou_g: 'float64[:]', rhov_g: 'float64[:]', rhow_g: 'float64[:]', rhoE_g: 'float64[:]', rho_h: 'float64[:]', P_h: 'float64[:]', rhou_h: 'float64[:]', rhov_h: 'float64[:]', rhow_h: 'float64[:]', rhoE_h: 'float64[:]', cellidf: 'int32[:,:]', halofid: 'int32[:]', normal: 'float64[:,:]', mesurf: 'float64[:]', name: 'uint32[:]', gamma: 'float64', tangent: 'float64[:,:]', binormal: 'float64[:,:]'):

    nbface = len(cellidf)
       
    rez_rho[:]  = np.zeros(len(rez_rho))
    rez_rhou[:] = np.zeros(len(rez_rhou))
    rez_rhov[:] = np.zeros(len(rez_rhov))
    rez_rhow[:] = np.zeros(len(rez_rhov))
    rez_rhoE[:] = np.zeros(len(rez_rhoE))

    for i in range(nbface):
  

            norm = normal[i]
            mesu = mesurf[i]
            tang = tangent[i]
            binorm = binormal[i]
                     
            rhol = rho_c[cellidf[i][0]]
            Pl   = P_c[cellidf[i][0]]
            rhoul=rhou_c[cellidf[i][0]]
            rhovl=rhov_c[cellidf[i][0]]
            rhowl=rhow_c[cellidf[i][0]]
            rhoEl=rhoE_c[cellidf[i][0]]
                        
            if name[i] == 0:
         
                rhor=rho_c[cellidf[i][1]]
                Pr=P_c[cellidf[i][1]]
                rhour=rhou_c[cellidf[i][1]]
                rhovr=rhov_c[cellidf[i][1]]
                rhowr=rhow_c[cellidf[i][1]]
                rhoEr=rhoE_c[cellidf[i][1]]
                
                flux_rho,flux_rhou,flux_rhov,flux_rhow,flux_rhoE = _compute_flux_euler_3d_Roe(rhol,Pl,rhoul,rhovl,rhowl,rhoEl,rhor,Pr,rhour,rhovr,rhowr,rhoEr,norm, mesu,gamma,tang,binorm)
                
                rez_rho[cellidf[i][0]] -= flux_rho
                rez_rho[cellidf[i][1]] += flux_rho
                rez_rhou[cellidf[i][0]] -= flux_rhou
                rez_rhou[cellidf[i][1]] += flux_rhou
                rez_rhov[cellidf[i][0]] -= flux_rhov
                rez_rhov[cellidf[i][1]] += flux_rhov
                rez_rhow[cellidf[i][0]] -= flux_rhow
                rez_rhow[cellidf[i][1]] += flux_rhow
                rez_rhoE[cellidf[i][0]] -= flux_rhoE
                rez_rhoE[cellidf[i][1]] += flux_rhoE
            
            elif name[i] == 10:
         
                rhor=rho_h[halofid[i]]
                Pr=P_h[halofid[i]]
                rhour=rhou_h[halofid[i]]
                rhovr=rhov_h[halofid[i]]
                rhowr=rhow_h[halofid[i]]
                rhoEr=rhoE_h[halofid[i]]
                
                flux_rho,flux_rhou,flux_rhov,flux_rhow,flux_rhoE = _compute_flux_euler_3d_Roe(rhol,Pl,rhoul,rhovl,rhowl,rhoEl,rhor,Pr,rhour,rhovr,rhowr,rhoEr,norm, mesu,gamma,tang,binorm) 
                
                rez_rho[cellidf[i][0]]  -= flux_rho
                rez_rhou[cellidf[i][0]] -= flux_rhou
                rez_rhov[cellidf[i][0]] -= flux_rhov
                rez_rhow[cellidf[i][0]] -= flux_rhow
                rez_rhoE[cellidf[i][0]] -= flux_rhoE
    
            else:
         
                rhor=rho_g[i]
                Pr=P_g[i]
                rhour=rhou_g[i]
                rhovr=rhov_g[i]
                rhowr=rhow_g[i]
                rhoEr=rhoE_g[i]
                
                flux_rho,flux_rhou,flux_rhov,flux_rhow,flux_rhoE = _compute_flux_euler_3d_Roe(rhol,Pl,rhoul,rhovl,rhowl,rhoEl,rhor,Pr,rhour,rhovr,rhowr,rhoEr,norm, mesu,gamma,tang,binorm)
               
                rez_rho[cellidf[i][0]]  -= flux_rho
                rez_rhou[cellidf[i][0]] -= flux_rhou
                rez_rhov[cellidf[i][0]] -= flux_rhov
                rez_rhow[cellidf[i][0]] -= flux_rhow
                rez_rhoE[cellidf[i][0]] -= flux_rhoE     

def _explicitscheme_euler_3d_fvc(rez_rho: 'float64[:]', rez_rhou: 'float64[:]', rez_rhov: 'float64[:]', rez_rhow: 'float64[:]', rez_rhoE: 'float64[:]', rho_p: 'float64[:]', P_p: 'float64[:]', rhou_p: 'float64[:]', rhov_p: 'float64[:]', rhow_p: 'float64[:]', rhoE_p: 'float64[:]', cellidf: 'int32[:,:]', normal: 'float64[:,:]', mesurf: 'float64[:]', name: 'uint32[:]'):

    nbface = len(cellidf)
    
    #from numpy import zeros
    
    rez_rho[:]  = np.zeros(len(rez_rho))
    rez_rhou[:] = np.zeros(len(rez_rhou))
    rez_rhov[:] = np.zeros(len(rez_rhov))
    rez_rhow[:] = np.zeros(len(rez_rhow))
    rez_rhoE[:] = np.zeros(len(rez_rhoE))

    for i in range(nbface):
  
            norm = normal[i]
            mesu = mesurf[i]
                  
            flux_rho,flux_rhou,flux_rhov,flux_rhow,flux_rhoE = _compute_flux_euler_3d_fvc(rho_p[i],P_p[i],rhou_p[i],rhov_p[i],rhow_p[i],rhoE_p[i],norm, mesu)
            
            if name[i] == 0:
         
                rez_rho[cellidf[i][0]]  -= flux_rho
                rez_rho[cellidf[i][1]]  += flux_rho
                
                rez_rhou[cellidf[i][0]] -= flux_rhou
                rez_rhou[cellidf[i][1]] += flux_rhou
                
                rez_rhov[cellidf[i][0]] -= flux_rhov
                rez_rhov[cellidf[i][1]] += flux_rhov
                
                rez_rhow[cellidf[i][0]] -= flux_rhow
                rez_rhow[cellidf[i][1]] += flux_rhow
                
                rez_rhoE[cellidf[i][0]] -= flux_rhoE
                rez_rhoE[cellidf[i][1]] += flux_rhoE
            
            elif name[i] == 10:
                
                rez_rho[cellidf[i][0]]  -= flux_rho
                rez_rhou[cellidf[i][0]] -= flux_rhou
                rez_rhov[cellidf[i][0]] -= flux_rhov
                rez_rhow[cellidf[i][0]] -= flux_rhow
                rez_rhoE[cellidf[i][0]] -= flux_rhoE
    
            else:
               
                rez_rho[cellidf[i][0]]  -= flux_rho
                rez_rhou[cellidf[i][0]] -= flux_rhou
                rez_rhov[cellidf[i][0]] -= flux_rhov
                rez_rhow[cellidf[i][0]] -= flux_rhow
                rez_rhoE[cellidf[i][0]] -= flux_rhoE

def _update_euler_3d_fvc(rho_c: 'float64[:]', P_c: 'float64[:]', rhou_c: 'float64[:]', rhov_c: 'float64[:]', rhow_c: 'float64[:]', rhoE_c: 'float64[:]', rez_rho: 'float64[:]', rez_rhou: 'float64[:]', rez_rhov: 'float64[:]', rez_rhow: 'float64[:]', rez_rhoE: 'float64[:]', gamma: 'float64', dtime: 'float64', vol: 'float64[:]'):

    rho_c[:]    += dtime * ((rez_rho[:]) /vol[:])
    rhou_c[:]   += dtime * ((rez_rhou[:]) /vol[:])
    rhov_c[:]   += dtime * ((rez_rhov[:]) /vol[:])
    rhow_c[:]   += dtime * ((rez_rhow[:]) /vol[:])
    rhoE_c[:]   += dtime * ((rez_rhoE[:]) /vol[:])
    P_c[:]       = (gamma-1)*(rhoE_c[:]-0.5*(rhou_c[:]*rhou_c[:] + rhov_c[:]*rhov_c[:] + rhow_c[:]*rhow_c[:])/rho_c[:])

############################################################################
# Public
node_for_interpolation_3d = compile(_node_for_interpolation_3d)
node_value_for_interpolation_3d = compile(_node_value_for_interpolation_3d)
weight_parameters_carac_3d = compile(_weight_parameters_carac_3d)
set_carac_field_3d = compile(_set_carac_field_3d)
compute_normale_inverse = compile(_compute_normale_inverse)
departure_euler_3d = compile(_departure_euler_3d)
predictor_euler_3d = compile(_predictor_euler_3d)
initialisation_euler_3d = compile(_initialisation_euler_3d)
ghost_value_Neumann3D = compile(_ghost_value_Neumann3D)
ghost_value_TubeSchok3D = compile(_ghost_value_TubeSchok3D)
ghost_value_gamm = compile(_ghost_value_gamm)
halghost_value_Neumann3D = compile(_halghost_value_Neumann3D)
halghost_value_TubeSchok3D = compile(_halghost_value_TubeSchok3D)
halghost_value_gamm = compile(_halghost_value_gamm)
time_step_euler_3d = compile(_time_step_euler_3d)
compute_flux_euler_3d_fvc = compile(_compute_flux_euler_3d_fvc)
compute_flux_euler_3d_rusanov = compile(_compute_flux_euler_3d_rusanov)
compute_flux_euler_3d_Roe = compile(_compute_flux_euler_3d_Roe)
explicitscheme_euler_3d_rusanov = compile(_explicitscheme_euler_3d_rusanov)
explicitscheme_euler_3d_Roe = compile(_explicitscheme_euler_3d_Roe)
explicitscheme_euler_3d_fvc = compile(_explicitscheme_euler_3d_fvc)
update_euler_3d_fvc = compile(_update_euler_3d_fvc)
