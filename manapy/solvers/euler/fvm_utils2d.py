from numba import njit
import numpy as np

@njit("void(float64[:,:], float64[:,:], int32[:,:], int32[:,:], int32[:], float64[:,:],\
            float64[:,:], float64[:,:], float64[:,:], uint32[:])",  cache=True, fastmath=True)    
def node_for_interpolation_2d(xCenterForInterp, yCenterForInterp,nodefid, 
                              cellfid, halofid,cellcenter, vertexcenter,
                              ghostcenter, halocenter, name):

    nbfaces = len(nodefid)
    for i in range(nbfaces):
        
        xCenterForInterp[i][0:2] = vertexcenter[nodefid[i][0:2], 0]
        xCenterForInterp[i][2]   = cellcenter[cellfid[i][0]][0]
        
        yCenterForInterp[i][0:2] = vertexcenter[nodefid[i][0:2], 1]
        yCenterForInterp[i][2] = cellcenter[cellfid[i][0]][1]
 
        if name[i] == 0:

            xCenterForInterp[i][3] = cellcenter[cellfid[i][1]][0]
            yCenterForInterp[i][3] = cellcenter[cellfid[i][1]][1]

            
        elif name[i] == 10: 

            xCenterForInterp[i][3] = halocenter[halofid[i]][0]
            yCenterForInterp[i][3] = halocenter[halofid[i]][1]

            
        else:

            xCenterForInterp[i][3] = ghostcenter[i][0]
            yCenterForInterp[i][3] = ghostcenter[i][1]
            
@njit("void(float64[:,:], float64[:], float64[:], float64[:], float64[:], int32[:,:], int32[:,:], int32[:], uint32[:])",  cache=True, fastmath=True)         
def node_value_for_interpolation_2d(ValForInterp,w_cell, w_node,
                                    w_ghost, w_halo, nodefid,
                                    cellfid, halofid, name):


    nbfaces = len(nodefid)
    for i in range(nbfaces):
        
        ValForInterp[i][0:2] = w_node[nodefid[i][0:2]]
        ValForInterp[i][2]   = w_cell[cellfid[i][0]]
   
        if name[i] == 0:
            ValForInterp[i][3] = w_cell[cellfid[i][1]]
        elif name[i] == 10: 
            ValForInterp[i][3] = w_halo[halofid[i]]
        else:
            ValForInterp[i][3] = w_ghost[i]
@njit("Tuple((float64, float64, float64, float64))(float64[:], float64[:], float64, float64)",  cache=True, fastmath=True)                 
def weight_parameters_carac_2d(xCenterForInterp, yCenterForInterp,  X0, Y0):

    I_xx = 0.
    I_yy = 0.
    I_xy = 0.
    R_x  = 0.
    R_y  = 0.
    lambda_x = 0.
    lambda_y = 0.

  #loop over the 5 points arround the face
    for i in range(0, 4):
        Rx = xCenterForInterp[i] - X0
        Ry = yCenterForInterp[i] - Y0
        I_xx += (Rx * Rx)
        I_yy += (Ry * Ry)
        I_xy += (Rx * Ry)
        R_x += Rx
        R_y += Ry

    D = I_xx*I_yy - I_xy*I_xy
    lambda_x = (I_xy*R_y - I_yy*R_x) / D
    lambda_y = (I_xy*R_x - I_xx*R_y) / D

    return R_x, R_y, lambda_x, lambda_y
        
@njit("float64(float64[:], float64[:], float64[:], float64, float64)",  cache=True, fastmath=True)                 
def set_carac_field_2d(ValForInterp, xCenterForInterp, yCenterForInterp, X0, Y0):

    w_carac = 0.
    R_x = 0.
    R_y = 0.
    lambda_x = 0.
    lambda_y = 0.
    R_x, R_y, lambda_x, lambda_y=weight_parameters_carac_2d(xCenterForInterp, yCenterForInterp, X0, Y0)
    
    for i in range(0, 4):
        
        xdiff = xCenterForInterp[i] - X0
        ydiff = yCenterForInterp[i] - Y0
        
        alpha_interp = (1. + lambda_x*xdiff + lambda_y*ydiff)/ (4. + lambda_x*R_x + lambda_y*R_y)
        
        w_carac  += alpha_interp * ValForInterp[i]
   
    return w_carac
@njit("void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:], float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, int32)",  cache=True, fastmath=True)                 
def initialisation_euler_2d(rho,P,rhou,rhov,rhoE,
                            center,rho1,rho2,P1,P2,xm,
                            gamma,u1,u2,v1,v2, choix):
    
    nbelements = len(center)
    
    if choix == 0:
        
        for i in range(nbelements):
            
            xcent = center[i][0]
       
            if xcent<xm:
                rho[i]   = rho1
                rhou[i]  = u1*rho1
                P[i]     = P1
                rhov[i]  = v1*rho1
            else:
                rho[i]   = rho2
                P[i]     = P2
                rhou[i]  = u2*rho2
                rhov[i]  = v2*rho2
            
            u = rhou[i]/rho[i]
            v = rhov[i]/rho[i]
                
            rhoE[i]  = 0.5*rho[i]*(u**2 + v**2)  + P[i]/(gamma-1)
            
    elif choix == 1:
                    
        for i in range(nbelements):
            rho[i]   = 1.0948
            P[i]     = 90808.0041
            rhou[i]  = 0.
            rhov[i]  = 0.
            rhoE[i]  = 0.5*(rhou[i]**2+rhov[i]**2)/rho[i]+P[i]/(gamma-1)
            
    elif choix == 2:
        
        sigma=0.1
        
        for i in range(nbelements):
            xcent    = center[i][0]
            rho[i]   = np.exp(-1.*((xcent-0.5)**2) / sigma) 
            P[i]     = 1.
            rhou[i]  = 0.*rho[i]
            rhov[i]  = 0.
            rhoE[i]  = 0.5*(rhou[i]**2+rhov[i]**2)/rho[i]+P[i]/(gamma-1)   
            
    elif choix == 3:
        
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]
            
            if (xcent<0.5 and ycent>=0.5):
                P[i]     = 0.3
                rho[i]   = 0.5323
                rhou[i]  = 1.206*0.5323
                rhov[i]  = 0*0.5323
                
            elif (xcent<0.5 and ycent<=0.5):
                
                rho[i]   = 0.138
                P[i]     = 0.029
                rhou[i]  = 1.206*0.138
                rhov[i]  = 1.206*0.138
                
            elif (xcent>0.5 and ycent>=0.5):
                
                rho[i]   = 1.5
                P[i]     = 1.5
                rhou[i]  = 0*1.5
                rhov[i]  = 0*1.5
                
            elif (xcent>0.5 and ycent<=0.5):
                 rho[i]   = 0.5323
                 P[i]     = 0.3
                 rhou[i]  = 0
                 rhov[i]  = 1.206*0.5323
                
            rhoE[i]  = 0.5*(rhou[i]**2 + rhov[i]**2)/rho[i] + P[i]/(gamma-1)
            
    elif choix == 4:
        
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]
            
            if (xcent<0.5 and ycent>=0.5):
                P[i]     = 0.35
                rho[i]   = 0.5065
                rhou[i]  = 0.8939*0.5065
                rhov[i]  = 0
            elif (xcent<0.5 and ycent<=0.5):
                P[i]     = 1.1
                rho[i]   = 1.1
                rhou[i]  = 0.8939*1.1
                rhov[i]  = 0.8939*1.1
            elif (xcent>0.5 and ycent>=0.5):
                P[i]     = 1.1
                rho[i]   = 1.1
                rhou[i]  = 0
                rhov[i]  = 0
                
            elif (xcent>0.5 and ycent<=0.5):
                P[i]     = 0.35
                rho[i]   = 0.5065
                rhou[i]  = 0*0.5065
                rhov[i]  = 0.8939*0.5065
                
            rhoE[i]  = 0.5*(rhou[i]**2 + rhov[i]**2)/rho[i] + P[i]/(gamma-1) 
            
    elif choix == 5:
        
        for i in range(nbelements):
            
            xcent = center[i][0]
            ycent = center[i][1]
            
            xm =  np.sqrt(xcent**2 + ycent**2)
            
            if xm < 0.5:
                rho[i]   = 1.0
                P[i]     = 1
                rhou[i]  = 0
                rhov[i]  = 0
            else:
                rho[i]   = 0.125
                P[i]     = 0.1
                rhou[i]  = 0
                rhov[i]  = 0
#            
            u = rhou[i]/rho[i]
            v = rhov[i]/rho[i]
                
            rhoE[i]  = 0.5*rho[i]*(u**2 + v**2)  + P[i]/(gamma-1)
            
    elif choix == 7:
        
        for i in range(nbelements):
            
            xcent = center[i][0]
            ycent = center[i][1]
            
            rho[i]   = 1.4
            P[i]     = 1
            rhou[i]  = 3*rho[i] 
            rhov[i]  = 0
            u = rhou[i]/rho[i]
            v = rhov[i]/rho[i]
                
            rhoE[i]  = 0.5*rho[i]*(u**2 + v**2)  + P[i]/(gamma-1)      
                 
    elif choix == 6:
        
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]
            
            if (xcent<0.5 and ycent>=0.5):
                P[i]     = 1.0
                rho[i]   = 2.0
                rhou[i]  = 0.75*1.0
                rhov[i]  = 0.5*1.0
            elif (xcent<0.5 and ycent<=0.5):
                P[i]     = 1.
                rho[i]   = 1.
                rhou[i]  = -0.75*1.0
                rhov[i]  = 0.5*1.0
            elif (xcent>0.5 and ycent>=0.5):
                P[i]     = 1.
                rho[i]   = 1.
                rhou[i]  = 0.75
                rhov[i]  = -0.5
                
            elif (xcent>0.5 and ycent<=0.5):
                P[i]     = 1.0
                rho[i]   = 3.0
                rhou[i]  = -0.75*3.0
                rhov[i]  = -0.5*3.0
                
            rhoE[i]  = 0.5*(rhou[i]**2 + rhov[i]**2)/rho[i] + P[i]/(gamma-1)
            
            
    elif choix == 12:
        
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]
            
            if (xcent<0.5 and ycent>=0.5):
                P[i]     = 1.0
                rho[i]   = 1.0
                rhou[i]  = 0.7276*1.0
                rhov[i]  = 0*1.0
            elif (xcent<0.5 and ycent<=0.5):
                P[i]     = 1.
                rho[i]   = 0.8
                rhou[i]  = 0*0.8
                rhov[i]  = 0.0*0.8
            elif (xcent>0.5 and ycent>=0.5):
                P[i]     = .4
                rho[i]   = 0.5313
                rhou[i]  = 0.0
                rhov[i]  = 0.0
                
            elif (xcent>0.5 and ycent<=0.5):
                P[i]     = 1.0
                rho[i]   = 1.0
                rhou[i]  = 0
                rhov[i]  = 0.7276*1.0
                
            rhoE[i]  = 0.5*(rhou[i]**2 + rhov[i]**2)/rho[i] + P[i]/(gamma-1) 
            
    elif choix == 15:
        
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]
            
            if (xcent<0.5 and ycent>=0.5):
                P[i]     = 0.4
                rho[i]   = 0.5197
                rhou[i]  = -0.6259*0.5197
                rhov[i]  = -0.3*0.5197
            elif (xcent<0.5 and ycent<=0.5):
                P[i]     = 0.4
                rho[i]   = 0.8
                rhou[i]  = 0.1*0.8
                rhov[i]  = -0.3*0.8
            elif (xcent>0.5 and ycent>=0.5):
                P[i]     = 1.0
                rho[i]   = 1.0
                rhou[i]  = 0.1
                rhov[i]  =-0.3
                
            elif (xcent>0.5 and ycent<=0.5):
                P[i]     = 0.4
                rho[i]   = 0.5313
                rhou[i]  = 0.1*0.5313
                rhov[i]  = 0.4276*0.5313
                
            rhoE[i]  = 0.5*(rhou[i]**2 + rhov[i]**2)/rho[i] + P[i]/(gamma-1)   
            
    elif choix == 17:
        
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]
            
            if (xcent<0.5 and ycent>=0.5):
                P[i]     = 1
                rho[i]   = 2
                rhou[i]  = 0
                rhov[i]  = -0.3*2
            elif (xcent<0.5 and ycent<=0.5):
                P[i]     = 0.4
                rho[i]   = 1.0625
                rhou[i]  = 0
                rhov[i]  = 0.2145*1.0625
            elif (xcent>0.5 and ycent>=0.5):
                P[i]     = 1.0
                rho[i]   = 1.0
                rhou[i]  = 0.0
                rhov[i]  = -0.4*1
                
            elif (xcent>0.5 and ycent<=0.5):
                P[i]     = 0.4
                rho[i]   = 0.5197
                rhou[i]  = 0
                rhov[i]  = -1.1259*0.5197
                
            rhoE[i]  = 0.5*(rhou[i]**2 + rhov[i]**2)/rho[i] + P[i]/(gamma-1) 
            
    elif choix == 18:
        for i in range(nbelements):
            
            xcent = center[i][0]
            ycent = center[i][1]
                     
            if xcent < 1/6 + ycent/np.sqrt(3):
                rho[i]   = 8.0
                rhou[i]  = 8.25*8*np.cos(np.pi/6)
                rhov[i]  = -8.25*8*np.sin(np.pi/6)
                P[i]     = 116.5
                
            else:
                rho[i]   = 1.4
                P[i]     = 1
                rhou[i]  = 0.0
                rhov[i]  = 0.0
            rhoE[i]  = 0.5*(rhou[i]**2 + rhov[i]**2)/rho[i] + P[i]/(gamma-1)
    
    elif choix == 19:
        
        for i in range(nbelements):
            
            xcent = center[i][0]
            ycent = center[i][1]
            
            xm =  np.sqrt(xcent**2 + ycent**2)
            
            if xm < 0.13:
                rho[i]   = 2.0
                rhou[i]  = 0
                P[i]     = 15
                rhov[i]  = 0
            else:
                rho[i]   = 1.0
                P[i]     = 1.0
                rhou[i]  = 0
                rhov[i]  = 0
#            
            u = rhou[i]/rho[i]
            v = rhov[i]/rho[i]
                
            rhoE[i]  = 0.5*rho[i]*(u**2 + v**2)  + P[i]/(gamma-1)       

@njit("void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], int32[:,:], uint32[:], float64[:,:], float64[:], float64[:,:],float64)",  cache=True, fastmath=True)                        
def ghost_value_DoubleMach(rhog,Pg,rhoug,rhovg,ug,vg,
                     rhoEg, rhoc,Pc,rhouc,rhovc,rhoEc,
                     cellid, name, normal, mesure,center,t):
    
    nbface = len(cellid)
    s_n = np.zeros(2)
    
    for i in range(nbface):
        
       
        if name[i] == 1:
                                    
            ### Neumann
            
            rhog[i]  = rhoc[cellid[i][0]]
            rhoug[i] = rhouc[cellid[i][0]]
            rhovg[i] = rhovc[cellid[i][0]] 
            rhoEg[i] = rhoEc[cellid[i][0]]
            Pg[i]    = Pc[cellid[i][0]]
            ug[i] = rhoug[i]/rhog[i]
            vg[i] = rhovg[i]/rhog[i]
            
#            
        
        elif name[i] == 2:
            
            ### Neumann
          
            rhog[i]  = rhoc[cellid[i][0]]
            rhoug[i] = rhouc[cellid[i][0]]
            rhovg[i] = rhovc[cellid[i][0]] 
            rhoEg[i] = rhoEc[cellid[i][0]]
            Pg[i]    = Pc[cellid[i][0]]
            ug[i] = rhoug[i]/rhog[i]
            vg[i] = rhovg[i]/rhog[i]

            
        elif name[i] == 3: 
            
            #### Double Mach
            
            xs = center[i][0]
            
            if xs > 1/6:
                               
              u_i = rhouc[cellid[i][0]]/rhoc[cellid[i][0]]
              v_i = rhovc[cellid[i][0]]/rhoc[cellid[i][0]]
       
              s_n[:] = normal[i][0:2]/np.sqrt(normal[i][0]*normal[i][0]+normal[i][1]*normal[i][1])
            
              u_g = u_i-2*(u_i*s_n[0]+v_i*s_n[1])*s_n[0]
              v_g = v_i-2*(u_i*s_n[0]+v_i*s_n[1])*s_n[1]
                        
              rhog[i]  = rhoc[cellid[i][0]]
              rhoug[i] = rhoc[cellid[i][0]] * u_g
              rhovg[i] = rhoc[cellid[i][0]] * v_g
              rhoEg[i] = rhoEc[cellid[i][0]]
              Pg[i] = Pc[cellid[i][0]]
              ug[i] = rhoug[i]/rhog[i]
              vg[i] = rhovg[i]/rhog[i]
              
            else:
                
              rhog[i]  = 8
              rhoug[i] = 8.25*8*np.cos(np.pi/6)
              rhovg[i] = -8.25*8*np.sin(np.pi/6)

              Pg[i]    = 116.5
            
              rhoEg[i]  = 0.5*(rhoug[i]**2 + rhovg[i]**2)/rhog[i] + Pg[i]/(1.4-1)            
              ug[i]    = rhoug[i]/rhog[i]
              vg[i]    = rhovg[i]/rhog[i]


        elif name[i] == 4: 
            
            ## Double Mach
            xcent = center[i][0]
            
            if xcent < 1/6 +(1+20*t)/np.sqrt(3):
                
                rhog[i]   = 8.0
                rhoug[i]  = 8.25*8*np.cos(np.pi/6)
                rhovg[i]  =-8.25*8*np.sin(np.pi/6)
                Pg[i]     = 116.5
                
            else:
                rhog[i]   = 1.4
                Pg[i]     = 1
                rhoug[i]  = 0.0
                rhovg[i]  = 0.0
                
            rhoEg[i]  = 0.5*(rhoug[i]**2 + rhovg[i]**2)/rhog[i] + Pg[i]/(1.4-1)            
            ug[i]    = rhoug[i]/rhog[i]
            vg[i]    = rhovg[i]/rhog[i] 
@njit("void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], int32[:,:], uint32[:], float64[:,:], float64[:], float64[:,:],float64)",  cache=True, fastmath=True)                            
def ghost_value_TubeSchok(rhog,Pg,rhoug,rhovg,ug,vg,
                     rhoEg, rhoc,Pc,rhouc,rhovc,rhoEc,
                     cellid, name, normal, mesure,center,t):
    
    nbface = len(cellid)
    s_n = np.zeros(2)
    
    for i in range(nbface):
              
        if name[i] == 1:
                                   
            ### Neumann
            
            rhog[i]  = rhoc[cellid[i][0]]
            rhoug[i] = rhouc[cellid[i][0]]
            rhovg[i] = rhovc[cellid[i][0]] 
            rhoEg[i] = rhoEc[cellid[i][0]]
            Pg[i]    = Pc[cellid[i][0]]
            ug[i] = rhoug[i]/rhog[i]
            vg[i] = rhovg[i]/rhog[i]
 
        elif name[i] == 2:
            
            ### Neumann
          
            rhog[i]  = rhoc[cellid[i][0]]
            rhoug[i] = rhouc[cellid[i][0]]
            rhovg[i] = rhovc[cellid[i][0]] 
            rhoEg[i] = rhoEc[cellid[i][0]]
            Pg[i]    = Pc[cellid[i][0]]
            ug[i] = rhoug[i]/rhog[i]
            vg[i] = rhovg[i]/rhog[i]

            
        elif name[i] == 3: 
         
            ### slip conditions
            
            u_i = rhouc[cellid[i][0]]/rhoc[cellid[i][0]]
            v_i = rhovc[cellid[i][0]]/rhoc[cellid[i][0]]
       
            s_n[:] = normal[i][0:2]/np.sqrt(normal[i][0]*normal[i][0]+normal[i][1]*normal[i][1])
            
            u_g = u_i-2*(u_i*s_n[0]+v_i*s_n[1])*s_n[0]
            v_g = v_i-2*(u_i*s_n[0]+v_i*s_n[1])*s_n[1]
                        
            rhog[i]  = rhoc[cellid[i][0]]
            rhoug[i] = rhoc[cellid[i][0]] * u_g
            rhovg[i] = rhoc[cellid[i][0]] * v_g
            rhoEg[i] = rhoEc[cellid[i][0]]
            Pg[i] = Pc[cellid[i][0]]
            ug[i] = rhoug[i]/rhog[i]
            vg[i] = rhovg[i]/rhog[i]
          
        elif name[i] == 4: 
            
            ### slip conditions
            
            u_i = rhouc[cellid[i][0]]/rhoc[cellid[i][0]]
            v_i = rhovc[cellid[i][0]]/rhoc[cellid[i][0]]
       
            s_n[:] = normal[i][0:2]/np.sqrt(normal[i][0]*normal[i][0]+normal[i][1]*normal[i][1])
            
            u_g = u_i-2*(u_i*s_n[0]+v_i*s_n[1])*s_n[0]
            v_g = v_i-2*(u_i*s_n[0]+v_i*s_n[1])*s_n[1]
                        
            rhog[i]  = rhoc[cellid[i][0]]
            rhoug[i] = rhoc[cellid[i][0]] * u_g
            rhovg[i] = rhoc[cellid[i][0]] * v_g
            rhoEg[i] = rhoEc[cellid[i][0]]
            Pg[i] = Pc[cellid[i][0]]
            ug[i] = rhoug[i]/rhog[i]
            vg[i] = rhovg[i]/rhog[i]
@njit("void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], int32[:,:], uint32[:], float64[:,:], float64[:], float64[:,:],float64)",  cache=True, fastmath=True)                
def ghost_value_Neumann(rhog,Pg,rhoug,rhovg,ug,vg,
                     rhoEg, rhoc,Pc,rhouc,rhovc,rhoEc,
                     cellid, name, normal, mesure,center,t):
    
    nbface = len(cellid)
    
    for i in range(nbface):
              
        if name[i] == 1:
                                   
            ### Neumann
            
            rhog[i]  = rhoc[cellid[i][0]]
            rhoug[i] = rhouc[cellid[i][0]]
            rhovg[i] = rhovc[cellid[i][0]] 
            rhoEg[i] = rhoEc[cellid[i][0]]
            Pg[i]    = Pc[cellid[i][0]]
            ug[i] = rhoug[i]/rhog[i]
            vg[i] = rhovg[i]/rhog[i]
 
        elif name[i] == 2:
            
            ### Neumann
          
            rhog[i]  = rhoc[cellid[i][0]]
            rhoug[i] = rhouc[cellid[i][0]]
            rhovg[i] = rhovc[cellid[i][0]] 
            rhoEg[i] = rhoEc[cellid[i][0]]
            Pg[i]    = Pc[cellid[i][0]]
            ug[i] = rhoug[i]/rhog[i]
            vg[i] = rhovg[i]/rhog[i]

            
        elif name[i] == 3: 
         
            ### Neumann
                   
            rhog[i]  = rhoc[cellid[i][0]]
            rhoug[i] = rhouc[cellid[i][0]]
            rhovg[i] = rhovc[cellid[i][0]] 
            rhoEg[i] = rhoEc[cellid[i][0]]
            Pg[i]    = Pc[cellid[i][0]]
            ug[i] = rhoug[i]/rhog[i]
            vg[i] = rhovg[i]/rhog[i]
          
        elif name[i] == 4: 
            
            ### Neumann
            
            rhog[i]  = rhoc[cellid[i][0]]
            rhoug[i] = rhouc[cellid[i][0]]
            rhovg[i] = rhovc[cellid[i][0]] 
            rhoEg[i] = rhoEc[cellid[i][0]]
            Pg[i]    = Pc[cellid[i][0]]
            ug[i] = rhoug[i]/rhog[i]
            vg[i] = rhovg[i]/rhog[i]

@njit("void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], int32[:,:], uint32[:], float64[:,:], float64[:], float64[:,:],float64)",  cache=True, fastmath=True)                
def ghost_value_Gamm2D(rhog,Pg,rhoug,rhovg,ug,vg,
                     rhoEg, rhoc,Pc,rhouc,rhovc,rhoEc,
                     cellid, name, normal, mesure,center,t):
    
    nbface = len(cellid)
    s_n = np.zeros(2)

    for i in range(nbface):
        
       
        if name[i] == 1:
                        
            gamma = 1.4
            p0    = 101391.8555
            rho0  = 1.1845
            p     = min(p0,Pc[cellid[i][0]])
            M2    = ((p / p0)**(-(gamma - 1.0)/gamma) - 1.0) * 2.0/(gamma - 1.0)
            tmp   = 1.0 + (gamma - 1.0) * 0.5 * M2
            rho   = rho0 * tmp**(-1.0 /(gamma - 1.0))
            a2    = gamma * p/rho
            rhoVel = rho*np.sqrt(M2 * a2)
            
            e = p / (gamma - 1) + 0.5 * (rhoVel**2)/rho 
            
            rhog[i]  = rho
            Pg[i]    = p
            rhoEg[i] = rho*e
            rhoug[i] = rhoVel
            rhovg[i] = 0.
            ug[i]    = rhoug[i]/rhog[i]
            vg[i]    = rhovg[i]/rhog[i]  
            
        elif name[i] == 2:
            
             gamma = 1.4
             MaIs  = 0.675
             p0    = 101391.8555
             p     = p0*((1. + (gamma - 1.) / 2. * MaIs*MaIs)**(gamma / (1. - gamma)))
             
             he = p / (gamma - 1.) + 0.5 * (rhouc[cellid[i][0]]**2 + rhovc[cellid[i][0]]**2)/rhoc[cellid[i][0]] 
             
             rhog[i]  = rhoc[cellid[i][0]]            
             rhoug[i] = rhouc[cellid[i][0]]
             rhovg[i] = rhovc[cellid[i][0]]
             rhoEg[i] = he
             Pg[i]    = p
             ug[i]    = rhoug[i]/rhog[i]
             vg[i]    = rhovg[i]/rhog[i]
            
            
        elif name[i] == 3: 
         
            ### slip conditions
            
            u_i = rhouc[cellid[i][0]]/rhoc[cellid[i][0]]
            v_i = rhovc[cellid[i][0]]/rhoc[cellid[i][0]]
       
            s_n[:] = normal[i][0:2]/np.sqrt(normal[i][0]*normal[i][0]+normal[i][1]*normal[i][1])
            
            u_g = u_i-2*(u_i*s_n[0]+v_i*s_n[1])*s_n[0]
            v_g = v_i-2*(u_i*s_n[0]+v_i*s_n[1])*s_n[1]
                        
            rhog[i]  = rhoc[cellid[i][0]]
            rhoug[i] = rhoc[cellid[i][0]] * u_g
            rhovg[i] = rhoc[cellid[i][0]] * v_g
            rhoEg[i] = rhoEc[cellid[i][0]]
            Pg[i] = Pc[cellid[i][0]]
            ug[i] = rhoug[i]/rhog[i]
            vg[i] = rhovg[i]/rhog[i]
          
        elif name[i] == 4: 
            
            ### slip conditions
            
            u_i = rhouc[cellid[i][0]]/rhoc[cellid[i][0]]
            v_i = rhovc[cellid[i][0]]/rhoc[cellid[i][0]]
       
            s_n[:] = normal[i][0:2]/np.sqrt(normal[i][0]*normal[i][0]+normal[i][1]*normal[i][1])
            
            u_g = u_i-2*(u_i*s_n[0]+v_i*s_n[1])*s_n[0]
            v_g = v_i-2*(u_i*s_n[0]+v_i*s_n[1])*s_n[1]
                        
            rhog[i]  = rhoc[cellid[i][0]]
            rhoug[i] = rhoc[cellid[i][0]] * u_g
            rhovg[i] = rhoc[cellid[i][0]] * v_g
            rhoEg[i] = rhoEc[cellid[i][0]]
            Pg[i] = Pc[cellid[i][0]]
            ug[i] = rhoug[i]/rhog[i]
            vg[i] = rhovg[i]/rhog[i]
            
            ### Neumann           
@njit("void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], int32[:,:], uint32[:], float64[:,:], uint32[:], float64[:,:,:], float64[:,:,:],float64)",  cache=True, fastmath=True)                      
def halghost_value_TubeSchok(rhog,Pg,rhoug,rhovg,ug,
                          vg,rhoEg, rho_halo,P_halo,rhou_halo,
                          rhov_halo,rhoE_halo, cellid, name, normal,
                          halonodes, haloghostcenter,haloghostfaceinfo,t):
    s_n = np.zeros(2)
    
    for i in halonodes:
        for j in range(len(haloghostcenter[i])):
            if haloghostcenter[i][j][-1] != -1:
                
                cellhalo  = int(haloghostcenter[i][j][-3])
                cellghost = int(haloghostcenter[i][j][-1])
                
                if haloghostcenter[i][j][-2] == 1:
                    
                     ### Neumann
                     
                    rhog[cellghost]  = rho_halo[cellhalo]
                    rhoug[cellghost] = rhou_halo[cellhalo]
                    rhovg[cellghost] = rhov_halo[cellhalo]
                    rhoEg[cellghost] = rhoE_halo[cellhalo]
                    Pg[cellghost]    = P_halo[cellhalo]
                    ug[cellghost]    = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost]    = rhovg[cellghost]/rhog[cellghost] 
  
                    
                elif haloghostcenter[i][j][-2] == 2:
                                                          
                     ### Neumann
                     
                    rhog[cellghost]  = rho_halo[cellhalo]
                    rhoug[cellghost] = rhou_halo[cellhalo]
                    rhovg[cellghost] = rhov_halo[cellhalo]
                    rhoEg[cellghost] = rhoE_halo[cellhalo]
                    Pg[cellghost]    = P_halo[cellhalo]
                    ug[cellghost]    = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost]    = rhovg[cellghost]/rhog[cellghost] 
                    
                                            
                elif haloghostcenter[i][j][-2] == 3:   
                    
                     ### Slip conditions
                     
                    u_i = rhou_halo[cellhalo]/rho_halo[cellhalo]
                    v_i = rhov_halo[cellhalo]/rho_halo[cellhalo]
                    #                    
                    s_n[0] = haloghostfaceinfo[i][j][2]/np.sqrt(haloghostfaceinfo[i][j][2]*haloghostfaceinfo[i][j][2]+haloghostfaceinfo[i][j][3]*haloghostfaceinfo[i][j][3])
                    s_n[1] = haloghostfaceinfo[i][j][3]/np.sqrt(haloghostfaceinfo[i][j][2]*haloghostfaceinfo[i][j][2]+haloghostfaceinfo[i][j][3]*haloghostfaceinfo[i][j][3])
                    
                    u_g = u_i-2*(u_i*s_n[0]+v_i*s_n[1])*s_n[0]
                    v_g = v_i-2*(u_i*s_n[0] + v_i*s_n[1])*s_n[1]
                                      
                    rhog[cellghost] = rho_halo[cellhalo]
                    rhoug[cellghost] = rho_halo[cellhalo] * u_g
                    rhovg[cellghost] = rho_halo[cellhalo] * v_g
                    rhoEg[cellghost] = rhoE_halo[cellhalo]
                    Pg[cellghost]    = P_halo[cellhalo]
                    ug[cellghost] = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost] = rhovg[cellghost]/rhog[cellghost]
 
                elif haloghostcenter[i][j][-2] == 4 :
                    
                     ## Slip conditions
                     
                    u_i = rhou_halo[cellhalo]/rho_halo[cellhalo]
                    v_i = rhov_halo[cellhalo]/rho_halo[cellhalo]
                    #                    
                    s_n[0] = haloghostfaceinfo[i][j][2]
                    s_n[1] = haloghostfaceinfo[i][j][3]
                    
                    u_g = u_i-2*(u_i*s_n[0]+v_i*s_n[1])*s_n[0]
                    v_g = v_i-2*(u_i*s_n[0] + v_i*s_n[1])*s_n[1]
                                      
                    rhog[cellghost] = rho_halo[cellhalo]
                    rhoug[cellghost] = rho_halo[cellhalo] * u_g
                    rhovg[cellghost] = rho_halo[cellhalo] * v_g
                    rhoEg[cellghost] = rhoE_halo[cellhalo]
                    Pg[cellghost]    = P_halo[cellhalo]
                    ug[cellghost] = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost] = rhovg[cellghost]/rhog[cellghost]
@njit("void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], int32[:,:], uint32[:], float64[:,:], uint32[:], float64[:,:,:], float64[:,:,:],float64)",  cache=True, fastmath=True)                               
def halghost_value_Neumann(rhog,Pg,rhoug,rhovg,ug,
                          vg,rhoEg, rho_halo,P_halo,rhou_halo,
                          rhov_halo,rhoE_halo, cellid, name, normal,
                          halonodes, haloghostcenter,haloghostfaceinfo,t):
    
    for i in halonodes:
        for j in range(len(haloghostcenter[i])):
            if haloghostcenter[i][j][-1] != -1:
                
                cellhalo  = int(haloghostcenter[i][j][-3])
                cellghost = int(haloghostcenter[i][j][-1])
                
                if haloghostcenter[i][j][-2] == 1:
                    
                     ### Neumann
                     
                    rhog[cellghost]  = rho_halo[cellhalo]
                    rhoug[cellghost] = rhou_halo[cellhalo]
                    rhovg[cellghost] = rhov_halo[cellhalo]
                    rhoEg[cellghost] = rhoE_halo[cellhalo]
                    Pg[cellghost]    = P_halo[cellhalo]
                    ug[cellghost]    = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost]    = rhovg[cellghost]/rhog[cellghost] 
  
                    
                elif haloghostcenter[i][j][-2] == 2:
                                                          
                     ### Neumann
                     
                    rhog[cellghost]  = rho_halo[cellhalo]
                    rhoug[cellghost] = rhou_halo[cellhalo]
                    rhovg[cellghost] = rhov_halo[cellhalo]
                    rhoEg[cellghost] = rhoE_halo[cellhalo]
                    Pg[cellghost]    = P_halo[cellhalo]
                    ug[cellghost]    = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost]    = rhovg[cellghost]/rhog[cellghost] 
                    
                                            
                elif haloghostcenter[i][j][-2] == 3:   
                    
                     ### Neumann
                     
                    rhog[cellghost]  = rho_halo[cellhalo]
                    rhoug[cellghost] = rhou_halo[cellhalo]
                    rhovg[cellghost] = rhov_halo[cellhalo]
                    rhoEg[cellghost] = rhoE_halo[cellhalo]
                    Pg[cellghost]    = P_halo[cellhalo]
                    ug[cellghost]    = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost]    = rhovg[cellghost]/rhog[cellghost] 
 
                elif haloghostcenter[i][j][-2] == 4 :
                    
                     ### Neumann
                     
                    rhog[cellghost]  = rho_halo[cellhalo]
                    rhoug[cellghost] = rhou_halo[cellhalo]
                    rhovg[cellghost] = rhov_halo[cellhalo]
                    rhoEg[cellghost] = rhoE_halo[cellhalo]
                    Pg[cellghost]    = P_halo[cellhalo]
                    ug[cellghost]    = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost]    = rhovg[cellghost]/rhog[cellghost] 
@njit("void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], int32[:,:], uint32[:], float64[:,:], uint32[:], float64[:,:,:], float64[:,:,:],float64)",  cache=True, fastmath=True)                               
def halghost_value_Gamm2D(rhog,Pg,rhoug,rhovg,ug,
                          vg,rhoEg, rho_halo,P_halo,rhou_halo,
                          rhov_halo,rhoE_halo, cellid, name, normal,
                          halonodes, haloghostcenter,haloghostfaceinfo,t):
    s_n = np.zeros(2)
    
    for i in halonodes:
        for j in range(len(haloghostcenter[i])):
            if haloghostcenter[i][j][-1] != -1:
                
                cellhalo  = int(haloghostcenter[i][j][-3])
                cellghost = int(haloghostcenter[i][j][-1])
                
                if haloghostcenter[i][j][-2] == 1:
                    
                    gamma = 1.4
                    p0    = 101391.8555
                    rho0  = 1.1845
                    p     = min(p0,P_halo[cellhalo])
                    M2    = ((p / p0)**(-(gamma - 1.0) / gamma) - 1.0) * 2.0 / (gamma - 1.0)
                    tmp   = 1.0 + (gamma - 1.0) * 0.5 * M2
                    rho   = rho0 * tmp**(-1.0 /(gamma - 1.0))
                    a2    = gamma * p / rho
                    rhoVel = rho*np.sqrt(M2 * a2)
                    e = p / (gamma - 1) + 0.5 * (rhoVel**2)/rho 
                    
                    rhog[cellghost]  = rho
                    Pg[cellghost]    = p
                    rhoEg[cellghost] = rho*e
                    rhoug[cellghost] = rhoVel
                    rhovg[cellghost] = 0.
                    ug[cellghost]    = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost]    = rhovg[cellghost]/rhog[cellghost] 
                                       
                    
                elif haloghostcenter[i][j][-2] == 2:
                    
                    gamma = 1.4
                    MaIs  = 0.675
                    p0    = 101391.8555
                    p     = p0*((1. + (gamma - 1.) / 2. * MaIs*MaIs)**(gamma / (1. - gamma)))
                   
                    he = p / (gamma - 1.) + 0.5 * (rhou_halo[cellhalo]**2 + rhov_halo[cellhalo]**2)/rho_halo[cellhalo] 
                   
                    rhog[cellghost]  = rho_halo[cellhalo]            
                    rhoug[cellghost] = rhou_halo[cellhalo] 
                    rhovg[cellghost] = rhov_halo[cellhalo] 
                    rhoEg[cellghost] = he
                    Pg[cellghost]    = p
                    ug[cellghost]    = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost]    = rhovg[cellghost]/rhog[cellghost]
                    
                              
                  
                elif haloghostcenter[i][j][-2] == 3:   
                    
                     ### Slip conditions
                     
                    u_i = rhou_halo[cellhalo]/rho_halo[cellhalo]
                    v_i = rhov_halo[cellhalo]/rho_halo[cellhalo]
                    #                    
                    s_n[0] = haloghostfaceinfo[i][j][2]/np.sqrt(haloghostfaceinfo[i][j][2]*haloghostfaceinfo[i][j][2]+haloghostfaceinfo[i][j][3]*haloghostfaceinfo[i][j][3])
                    s_n[1] = haloghostfaceinfo[i][j][3]/np.sqrt(haloghostfaceinfo[i][j][2]*haloghostfaceinfo[i][j][2]+haloghostfaceinfo[i][j][3]*haloghostfaceinfo[i][j][3])
                    
                    u_g = u_i-2*(u_i*s_n[0]+v_i*s_n[1])*s_n[0]
                    v_g = v_i-2*(u_i*s_n[0] + v_i*s_n[1])*s_n[1]
                                      
                    rhog[cellghost]  = rho_halo[cellhalo]
                    rhoug[cellghost] = rho_halo[cellhalo] * u_g
                    rhovg[cellghost] = rho_halo[cellhalo] * v_g
                    rhoEg[cellghost] = rhoE_halo[cellhalo]
                    Pg[cellghost]    = P_halo[cellhalo]
                    ug[cellghost]    = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost]    = rhovg[cellghost]/rhog[cellghost]
                                
                    
                elif haloghostcenter[i][j][-2] == 4 :
                                     
                    u_i = rhou_halo[cellhalo]/rho_halo[cellhalo]
                    v_i = rhov_halo[cellhalo]/rho_halo[cellhalo]
                    #                    
                    s_n[0] = haloghostfaceinfo[i][j][2]/np.sqrt(haloghostfaceinfo[i][j][2]*haloghostfaceinfo[i][j][2]+haloghostfaceinfo[i][j][3]*haloghostfaceinfo[i][j][3])
                    s_n[1] = haloghostfaceinfo[i][j][3]/np.sqrt(haloghostfaceinfo[i][j][2]*haloghostfaceinfo[i][j][2]+haloghostfaceinfo[i][j][3]*haloghostfaceinfo[i][j][3])
                    
                    u_g = u_i-2*(u_i*s_n[0]+v_i*s_n[1])*s_n[0]
                    v_g = v_i-2*(u_i*s_n[0] + v_i*s_n[1])*s_n[1]
                                      
                    rhog[cellghost] = rho_halo[cellhalo]
                    rhoug[cellghost] = rho_halo[cellhalo] * u_g
                    rhovg[cellghost] = rho_halo[cellhalo] * v_g
                    rhoEg[cellghost] = rhoE_halo[cellhalo]
                    Pg[cellghost]    = P_halo[cellhalo]
                    ug[cellghost] = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost] = rhovg[cellghost]/rhog[cellghost]
                     
                                                   
@njit("void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], int32[:,:], uint32[:], float64[:,:], uint32[:], float64[:,:,:], float64[:,:,:],float64)",  cache=True, fastmath=True)                              
def halghost_value_DoubleMach(rhog,Pg,rhoug,rhovg,ug,
                          vg,rhoEg, rho_halo,P_halo,rhou_halo,
                          rhov_halo,rhoE_halo, cellid, name, normal,
                          halonodes, haloghostcenter,haloghostfaceinfo,t):
    s_n = np.zeros(2)
    for i in halonodes:
        for j in range(len(haloghostcenter[i])):
            if haloghostcenter[i][j][-1] != -1:
                
                cellhalo  = int(haloghostcenter[i][j][-3])
                cellghost = int(haloghostcenter[i][j][-1])
                
                if haloghostcenter[i][j][-2] == 1:
                    
                     ### Neumann
                     
                    rhog[cellghost]  = rho_halo[cellhalo]
                    rhoug[cellghost] = rhou_halo[cellhalo]
                    rhovg[cellghost] = rhov_halo[cellhalo]
                    rhoEg[cellghost] = rhoE_halo[cellhalo]
                    Pg[cellghost]    = P_halo[cellhalo]
                    ug[cellghost]    = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost]    = rhovg[cellghost]/rhog[cellghost] 
                                        
                    
                elif haloghostcenter[i][j][-2] == 2:
                    
                                      
                     ### Neumann
                     
                    rhog[cellghost]  = rho_halo[cellhalo]
                    rhoug[cellghost] = rhou_halo[cellhalo]
                    rhovg[cellghost] = rhov_halo[cellhalo]
                    rhoEg[cellghost] = rhoE_halo[cellhalo]
                    Pg[cellghost]    = P_halo[cellhalo]
                    ug[cellghost]    = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost]    = rhovg[cellghost]/rhog[cellghost] 
                    
                                     
                elif haloghostcenter[i][j][-2] == 3:   

                     
                    xs = haloghostcenter[i][j][0]
                     
                    if xs > 1/6:
                                       
                        u_i = rhou_halo[cellhalo]/rho_halo[cellhalo]
                        v_i = rhov_halo[cellhalo]/rho_halo[cellhalo]
                        #                    
                        s_n[0] = haloghostfaceinfo[i][j][2]/np.sqrt(haloghostfaceinfo[i][j][2]*haloghostfaceinfo[i][j][2]+haloghostfaceinfo[i][j][3]*haloghostfaceinfo[i][j][3])
                        s_n[1] = haloghostfaceinfo[i][j][3]/np.sqrt(haloghostfaceinfo[i][j][2]*haloghostfaceinfo[i][j][2]+haloghostfaceinfo[i][j][3]*haloghostfaceinfo[i][j][3])
                        
                        u_g = u_i-2*(u_i*s_n[0]+v_i*s_n[1])*s_n[0]
                        v_g = v_i-2*(u_i*s_n[0] + v_i*s_n[1])*s_n[1]
                                          
                        rhog[cellghost] = rho_halo[cellhalo]
                        rhoug[cellghost] = rho_halo[cellhalo] * u_g
                        rhovg[cellghost] = rho_halo[cellhalo] * v_g
                        rhoEg[cellghost] = rhoE_halo[cellhalo]
                    
                        Pg[cellghost]    = P_halo[cellhalo]
                        ug[cellghost] = rhoug[cellghost]/rhog[cellghost]
                        vg[cellghost] = rhovg[cellghost]/rhog[cellghost]
                        
                    else:
                        rhog[cellghost]  = 8
                        rhoug[cellghost] = 8.25*8*np.cos(np.pi/6)
                        rhovg[cellghost] = -8.25*8*np.sin(np.pi/6)
            
                        Pg[cellghost]    = 116.5
                        ug[cellghost]    = rhoug[cellghost]/rhog[cellghost]
                        vg[cellghost]    = rhovg[cellghost]/rhog[cellghost] 
                        rhoEg[cellghost]  = 0.5*(rhoug[cellghost]**2 + rhovg[cellghost]**2)/rhog[cellghost] + Pg[cellghost]/(1.4-1)
                                
                    
                elif haloghostcenter[i][j][-2] == 4 :
                    
                     
                    xs = haloghostcenter[i][j][0]
                    
                    if xs < 1/6 +(1+20*t)/np.sqrt(3):
                        
                        rhog[cellghost]   = 8.0
                        rhoug[cellghost]  = 8.25*8*np.cos(np.pi/6)
                        rhovg[cellghost]  = -8.25*8*np.sin(np.pi/6)
                        Pg[cellghost]     = 116.5
                        
                    else:
                        rhog[cellghost]   = 1.4
                        Pg[cellghost]     = 1
                        rhoug[cellghost]  = 0.0
                        rhovg[cellghost]  = 0.0
                        
                    rhoEg[cellghost]  = 0.5*(rhoug[cellghost]**2 + rhovg[cellghost]**2)/rhog[cellghost] + Pg[cellghost]/(1.4-1)            
                    ug[cellghost]    = rhoug[cellghost]/rhog[cellghost]
                    vg[cellghost]    = rhovg[cellghost]/rhog[cellghost] 
                            
#            
# 
# 
@njit("void(float64[:], float64[:], float64[:], float64[:], float64, float64[:,:], float64[:], float64[:], int32[:,:], float64, float64[:])",  cache=True, fastmath=True)                 
def time_step_euler_2d(rho,P,rhou, rhov,  cfl, normal, 
                        mesure, volume, faceid,gamma, dt_c):
    
    nbelement =  len(faceid)
    u_n = 0.
  
    for i in range(nbelement):
        lam = 0.
        
        velson = np.sqrt(gamma*np.fabs(P[i]/rho[i]))
        
        for j in range(3):
            
            u_n = np.fabs((rhou[i]*normal[faceid[i][j]][0] + rhov[i]*normal[faceid[i][j]][1])/rho[i])/np.sqrt(normal[faceid[i][j]][0]*normal[faceid[i][j]][0]+normal[faceid[i][j]][1]*normal[faceid[i][j]][1])
            
            lam_convect = u_n + velson
           
            lam += lam_convect * mesure[faceid[i][j]]   
            
        dt_c[i]  = cfl*volume[i]/lam 
         
#        if scheme == "FVC":
#            dt_c[i]  = cfl*volume[i]/(lam*np.sqrt(2*alphaf) 
#        else:
#            dt_c[i]  = cfl*volume[i]/lam  
@njit("void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], int32[:,:], float64[:,:], float64[:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], uint32[:], float64, float64)", cache= True)                 
def departure_euler_2d(X0,Y0,rhof,rhouf,rhovf,
                       rhoValForInterp,uValForInterp,vValForInterp, 
                       xCenterForInterp, yCenterForInterp, nodeidf, 
                       normalf, mesuref,centerf,vertexn,
                       centerc,centerg,centerh,name,dt, alphaf):

    nbfaces = len(nodeidf)

    
    X0[:] = centerf[:,0]
    Y0[:] = centerf[:,1]
    

  
#    u_ed = 0.
#    v_ed = 0.

    for i in range(nbfaces):
           
            rho_ed = set_carac_field_2d(rhoValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], X0[i], Y0[i])
            u_ed   = set_carac_field_2d(uValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], X0[i], Y0[i])/rho_ed
            v_ed   = set_carac_field_2d(vValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], X0[i], Y0[i])/rho_ed
    
            
    #        rhof[i]  = rho_ed 
    #        rhouf[i] = rho_ed * u_ed
    #        rhovf[i] = rho_ed * v_ed
            
            u_n = (u_ed*normalf[i][0] + v_ed*normalf[i][1])/mesuref[i] 
                
            u_nx = u_n*normalf[i][0]/mesuref[i]
            u_ny = u_n*normalf[i][1]/mesuref[i]
    
            
            # ########### Euler ###########
                  
            X0[i] =  X0[i] - alphaf*dt*u_nx
            Y0[i] =  Y0[i] - alphaf*dt*u_ny
        
        ############ RK3 ##########""
        
#        xrk1 =  X0[i] - alphaf*dt*u_nx
#        yrk1 =  Y0[i] - alphaf*dt*u_ny
#        
#                
#        u    =  set_carac_field_2d(uValForInterp[i], xCenterForInterp[i], yCenterForInterp[i],xrk1,yrk1)/rho_ed
#        v    =  set_carac_field_2d(vValForInterp[i], xCenterForInterp[i], yCenterForInterp[i],xrk1,yrk1)/rho_ed
#       
#      
#                    
#        xrk2 = 0.75*X0[i]+0.25*xrk1-0.25*dt*alphaf*u
#        yrk2 = 0.75*Y0[i]+0.25*yrk1-0.25*dt*alphaf*v
#        
#        
#        u    =  set_carac_field_2d(uValForInterp[i], xCenterForInterp[i], yCenterForInterp[i],xrk2,yrk2)/rho_ed
#        v    =  set_carac_field_2d(vValForInterp[i], xCenterForInterp[i], yCenterForInterp[i],xrk2,yrk2)/rho_ed
#        
#        
#        X0[i] = (X0[i] +2.0*xrk2-2.0*alphaf*dt*u)/3.0
#        Y0[i] = (Y0[i] +2.0*yrk2-2.0*alphaf*dt*v)/3.0
      

@njit("void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64, float64, float64, int32[:,:], float64[:,:], float64[:])", cache= True)           
def predictor_euler_2d(rho_p,P_p,rhou_p,rhov_p,rhoE_p,
                       rhoValForInterp,rhouValForInterp,rhovValForInterp,
                       rhoEValForInterp, xCenterForInterp, yCenterForInterp,
                       X0,Y0,
                       ugradfacex,
                       ugradfacey,vgradfacex,vgradfacey,
                       Pgradfacex,Pgradfacey, 
                       gamma, d_t, alphaf,nodeidf, normal, mesuref):    
 
    nbfaces = len(nodeidf)

   
    for i in range(nbfaces):
        
        rho_ed  = set_carac_field_2d(rhoValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], X0[i], Y0[i])
        rhou_ed = set_carac_field_2d(rhouValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], X0[i], Y0[i])
        rhov_ed = set_carac_field_2d(rhovValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], X0[i], Y0[i])
        rhoE_ed = set_carac_field_2d(rhoEValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], X0[i], Y0[i],)

        rhop = rho_ed
        rhoup=rhou_ed
        rhovp=rhov_ed
        rhoEp=rhoE_ed
        
        up=rhou_ed/rho_ed
        vp=rhov_ed/rho_ed

        
        Pp=(gamma-1)*(rhoE_ed-0.5*rhop*(up*up +vp*vp))
               
        ###  Grad_normal (u) 
         
        ux = ugradfacex[i]
        uy = ugradfacey[i]
        
        vx = vgradfacex[i]
        vy = vgradfacey[i]
        
        
        
        ###### #  Grad_normal (u)  #######

        unx = (ux*normal[i,0] + vx*normal[i,1])/mesuref[i]
        uny = (uy*normal[i,0] + vy*normal[i,1])/mesuref[i]
                
        u_n = (up*normal[i,0] +  vp*normal[i,1])/mesuref[i]

        rhou_n = (rhoup*normal[i,0] +  rhovp*normal[i,1])/mesuref[i]
        rhou_t = (rhovp*normal[i,0] -  rhoup*normal[i,1])/mesuref[i]
     
        Un_grad = (unx*normal[i,0] + uny*normal[i,1])/mesuref[i]
        
        Pn_grad = (Pgradfacex[i]*normal[i,0] + Pgradfacey[i]*normal[i,1])/mesuref[i]
                
        rho_p[i] = rhop*(1 - alphaf*d_t*Un_grad)
        rhounp   = rhou_n - alphaf*d_t*(rhou_n*Un_grad + Pn_grad) 
        rhoutp   = rhou_t - alphaf*d_t*(rhou_t*Un_grad) 
        rhoE_p[i]= rhoEp - alphaf*d_t*(rhoEp*Un_grad + u_n*Pn_grad + Pp*Un_grad)
        
        rhou_p[i] = (rhounp*normal[i,0] - rhoutp*normal[i,1])/mesuref[i]
        rhov_p[i] = (rhounp*normal[i,1] + rhoutp*normal[i,0])/mesuref[i]
        
        P_p[i]    = (gamma-1)*(rhoE_p[i] - 0.5*(rhou_p[i]*rhou_p[i] + rhov_p[i]*rhov_p[i])/rho_p[i])
        
        
@njit("Tuple((float64,float64,float64,float64))(float64, float64, float64, float64, float64, float64[:], float64)", cache= True)             
def compute_flux_euler_2d_fvc(rhop,Pp,rhoup,rhovp,rhoEp,normal, mesure):
          
        flux_rho   = (rhoup*normal[0] + rhovp*normal[1])*mesure
        flux_rhou  = ((rhoup*rhoup/rhop + Pp)*normal[0] + (rhoup*rhovp/rhop)*normal[1])*mesure
        flux_rhov  = ((rhovp*rhoup/rhop)*normal[0] + (rhovp*rhovp/rhop+Pp)*normal[1])*mesure
        flux_rhoE  = (rhoup/rhop*(rhoEp + Pp)*normal[0] + rhovp/rhop*(rhoEp+Pp)*normal[1])*mesure
        
        
        return flux_rho,flux_rhou,flux_rhov,flux_rhoE     
@njit("Tuple((float64,float64,float64,float64))(float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64[:], float64, float64)", cache= True)             
def compute_flux_euler_2d_rusanov(rhol,Pl,rhoul,rhovl,rhoEl,rhor,Pr,rhour,rhovr,rhoEr, normal, mesure, gamma):
    
    ql = (rhoul*normal[0] + rhovl*normal[1])
    qr = (rhour*normal[0] + rhovl*normal[1])
    
    cl = np.sqrt(gamma*Pl/rhol)
    cr = np.sqrt(gamma*Pr/rhor)
    
    
    fl_rho  = (rhoul*normal[0] + rhovl*normal[1])
    fl_rhou = ((rhoul*rhoul/rhol+Pl)*normal[0] + (rhoul*rhovl/rhol)*normal[1])
    fl_rhov = ((rhovl*rhoul/rhol)*normal[0] + (rhovl*rhovl/rhol+Pl)*normal[1])
    fl_rhoE = (rhoul/rhol*(rhoEl+Pl)*normal[0] + rhovl/rhol*(rhoEl+Pl)*normal[1])

    fr_rho =  (rhour*normal[0] + rhovr*normal[1])
    fr_rhou = ((rhour*rhour/rhor+Pr)*normal[0] + (rhour*rhovr/rhor)*normal[1])
    fr_rhov = ((rhovr*rhour/rhor)*normal[0] + (rhovr*rhovr/rhor+Pr)*normal[1])
    fr_rhoE = (rhour/rhor*(rhoEr+Pr)*normal[0] + rhovr/rhor*(rhoEr+Pr)*normal[1])
    
    
    lambdal1 = np.fabs((ql)/rhol - cl)
    lambdal2 = np.fabs((ql)/rhol)
    lambdal3 = np.fabs((ql)/rhol + cl)

    lambdar1 = np.fabs((qr)/rhor - cr)
    lambdar2 = np.fabs((qr)/rhor)
    lambdar3 = np.fabs((qr)/rhor + cr) 
    
   
    Ll = max(lambdal1,lambdal2,lambdal3)
    Lr = max(lambdar1,lambdar2,lambdar3)
    S = 0.
    
    if (Ll > Lr):
        S = Ll
    else:
        S = Lr
   
    flux_rho  = (0.5 * (fl_rho + fr_rho)   -  0.5 * S  * (rhor - rhol))*mesure
    flux_rhou = (0.5 * (fl_rhou + fr_rhou) - 0.5 * S * (rhour - rhoul))*mesure
    flux_rhov = (0.5 * (fl_rhov + fr_rhov) - 0.5 * S * (rhovr - rhovl))*mesure
    flux_rhoE = (0.5 * (fl_rhoE + fr_rhoE) - 0.5 * S * (rhoEr - rhoEl))*mesure
    

    return flux_rho,flux_rhou,flux_rhov,flux_rhoE     
#######################
@njit("Tuple((float64,float64,float64,float64))(float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64[:], float64, float64)", cache= True)                         
def compute_flux_euler_2d_Roe(rhol,Pl,rhoul,rhovl,
                              rhoEl,rhor,Pr,rhour,
                              rhovr,rhoEr, normal, 
                              mesure, gamma):
    
    uul =  rhoul/rhol
    uur =  rhour/rhor
    
    vvl =  rhovl/rhol
    vvr =  rhovr/rhor
    
    Hl  =  (rhoEl + Pl)/rhol
    Hr  =  (rhoEr + Pr)/rhor
    
    DD = np.sqrt(rhol) + np.sqrt(rhor)
    
    #rhostar =  np.sqrt(rhol*rhor)
    ustar   =  (np.sqrt(rhol)*uul + np.sqrt(rhor)*uur)/DD
    vstar   =  (np.sqrt(rhol)*vvl + np.sqrt(rhor)*vvr)/DD
    Hstar   =  (np.sqrt(rhol)*Hl + np.sqrt(rhor)*Hr)/DD
    VV      =  ustar**2 + vstar**2  
    #Estar  = (1/gamma)*Hstar + 0.5*((gamma-1)/gamma)*VV
    #Pstar  =  (gamma -1 )*(rhostar*Estar - 0.5*rhostar*VV)
    #cc     =  np.sqrt(gamma*Pstar/rhostar)
    cc     =  np.sqrt((gamma -1)*(Hstar -0.5*VV))
    uueta   =  ustar*normal[0] + vstar*normal[1]  
    GG = (gamma - 1)/cc**2
    
    Lam1 = uueta - cc
    Lam2 = uueta
    Lam3 = uueta + cc
    Lam4 = Lam2
    
    #print(cc)
    LL      = np.zeros((4,4))
    RR      = np.zeros((4,4))
    RR1     = np.zeros((4,4))
    mat1    = np.zeros((4,4))
    ammat   = np.zeros((4,4))
    
    LL[0,0] = np.fabs(Lam1)  
    LL[1,1] = np.fabs(Lam2)  
    LL[2,2] = np.fabs(Lam3)  
    LL[3,3] = np.fabs(Lam4)   
    
    RR[0,0] = 1 
    RR[0,1] = 1 
    RR[0,2] = 1
    RR[0,3] = 0
       
    RR[1,0] = ustar - cc*normal[0]
    RR[1,1] = ustar 
    RR[1,2] = ustar + cc*normal[0]
    RR[1,3] = -1*normal[1]
    
    RR[2,0] = vstar - cc*normal[1]
    RR[2,1] = vstar 
    RR[2,2] = vstar + cc*normal[1]
    RR[2,3] = normal[0]
    
    RR[3,0] = Hstar - cc*uueta
    RR[3,1] = 0.5*VV
    RR[3,2] = Hstar + cc*uueta
    RR[3,3] = vstar*normal[0] - ustar*normal[1]
    
    RR1[0,0] = 0.5*(0.5*GG*VV + uueta/cc)
    RR1[0,1] = -0.5*(GG*ustar + normal[0]/cc)
    RR1[0,2] = -0.5*(GG*vstar + normal[1]/cc)
    RR1[0,3] = 0.5*GG

    RR1[1,0] = 1 - 0.5*GG*VV
    RR1[1,1] = GG*ustar
    RR1[1,2] = GG*vstar
    RR1[1,3] = -GG
    
    RR1[2,0] = 0.5*(0.5*GG*VV - uueta/cc)
    RR1[2,1] = -0.5*(GG*ustar - normal[0]/cc)
    RR1[2,2] = -0.5*(GG*vstar - normal[1]/cc)
    RR1[2,3] = 0.5*GG
    
    RR1[3,0] = ustar*normal[1] - vstar*normal[0] 
    RR1[3,1] = -normal[1]
    RR1[3,2] = normal[0]
    RR1[3,3] = 0
    
    
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
    
    fl_rho  = (rhoul*normal[0] + rhovl*normal[1])
    fl_rhou = ((rhoul*rhoul/rhol+Pl)*normal[0] + (rhoul*rhovl/rhol)*normal[1])
    fl_rhov = ((rhovl*rhoul/rhol)*normal[0] + (rhovl*rhovl/rhol+Pl)*normal[1])
    fl_rhoE = (rhoul/rhol*(rhoEl+Pl)*normal[0] + rhovl/rhol*(rhoEl+Pl)*normal[1])

    fr_rho =  (rhour*normal[0] + rhovr*normal[1])
    fr_rhou = ((rhour*rhour/rhor+Pr)*normal[0] + (rhour*rhovr/rhor)*normal[1])
    fr_rhov = ((rhovr*rhour/rhor)*normal[0] + (rhovr*rhovr/rhor+Pr)*normal[1])
    fr_rhoE = (rhour/rhor*(rhoEr+Pr)*normal[0] + rhovr/rhor*(rhoEr+Pr)*normal[1])
    
    
    
    w_dif    = np.zeros(4)
    
    w_dif[0] = rhor - rhol
    w_dif[1] = rhour - rhoul
    w_dif[2] = rhovr - rhovl
    w_dif[3] = rhoEr - rhoEl
    
    rhonew = 0.
    unew = 0.
    vnew = 0.
    Enew = 0.
    
    for i in range(4):
        
        rhonew += ammat[0][i] * w_dif[i]
        unew   += ammat[1][i] * w_dif[i]
        vnew   += ammat[2][i] * w_dif[i]
        Enew   += ammat[3][i] * w_dif[i]
        
    u_rho = rhonew
    u_rhou = unew
    u_rhov = vnew
    u_rhoE = Enew  
            

   
    flux_rho  = (0.5 * (fl_rho + fr_rho)   - 0.5*u_rho)*mesure
    flux_rhou = (0.5 * (fl_rhou + fr_rhou) - 0.5*u_rhou)*mesure
    flux_rhov = (0.5 * (fl_rhov + fr_rhov) - 0.5*u_rhov)*mesure
    flux_rhoE = (0.5 * (fl_rhoE + fr_rhoE) - 0.5*u_rhoE)*mesure
    

    return flux_rho,flux_rhou,flux_rhov,flux_rhoE     
@njit("void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], int32[:,:], float64[:,:], float64[:], uint32[:])", cache= True)                                  
def explicitscheme_euler_2d_fvc(rez_rho,rez_rhou,rez_rhov,
                                rez_rhoE,rho_p,P_p,
                                rhou_p,rhov_p,rhoE_p,
                                cellidf,normal,mesurf,name):

    nbface = len(cellidf)
   
    rez_rho[:]  = np.zeros(len(rez_rho))
    rez_rhou[:] = np.zeros(len(rez_rhou))
    rez_rhov[:] = np.zeros(len(rez_rhov))
    rez_rhoE[:] = np.zeros(len(rez_rhoE))

    for i in range(nbface):
  
            norm = normal[i]/mesurf[i]
            mesu = mesurf[i]

                  
            flux_rho,flux_rhou,flux_rhov,flux_rhoE = compute_flux_euler_2d_fvc(rho_p[i],P_p[i],rhou_p[i],rhov_p[i],rhoE_p[i], norm, mesu)
            
            if name[i] == 0:
         
                rez_rho[cellidf[i][0]]  -= flux_rho
                rez_rho[cellidf[i][1]]  += flux_rho
                rez_rhou[cellidf[i][0]] -= flux_rhou
                rez_rhou[cellidf[i][1]] += flux_rhou
                rez_rhov[cellidf[i][0]] -= flux_rhov
                rez_rhov[cellidf[i][1]] += flux_rhov
                rez_rhoE[cellidf[i][0]] -= flux_rhoE
                rez_rhoE[cellidf[i][1]] += flux_rhoE
            
            elif name[i] == 10:
                
                rez_rho[cellidf[i][0]]  -= flux_rho
                rez_rhou[cellidf[i][0]] -= flux_rhou
                rez_rhov[cellidf[i][0]] -= flux_rhov
                rez_rhoE[cellidf[i][0]] -= flux_rhoE
    
            else:
               
                rez_rho[cellidf[i][0]]  -= flux_rho
                rez_rhou[cellidf[i][0]] -= flux_rhou
                rez_rhov[cellidf[i][0]] -= flux_rhov
                rez_rhoE[cellidf[i][0]] -= flux_rhoE
                
@njit("void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],  float64[:], float64[:], float64[:], float64[:], int32[:,:], int32[:], float64[:,:], float64[:], uint32[:], float64)", cache= True)                     
def explicitscheme_euler_2d_rusanov(rez_rho,rez_rhou,rez_rhov,
                                    rez_rhoE,rho_c,P_c,
                                    rhou_c,rhov_c,rhoE_c,
                                    rho_g,P_g,rhou_g,
                                    rhov_g,rhoE_g,rho_h,
                                    P_h,rhou_h,rhov_h,
                                    rhoE_h, cellidf,halofid,
                                    normal,mesurf,name,gamma):
    
    

    
    center_left = np.zeros(2)
    center_right = np.zeros(2)
    r_r = np.zeros(2)

    nbface = len(cellidf)
       
    rez_rho[:]  = np.zeros(len(rez_rho))
    rez_rhou[:] = np.zeros(len(rez_rhou))
    rez_rhov[:] = np.zeros(len(rez_rhov))
    rez_rhoE[:] = np.zeros(len(rez_rhoE))

    for i in range(nbface):
  
            norm = normal[i]/mesurf[i]
            mesu = mesurf[i]
            
            rhol = rho_c[cellidf[i][0]]
            Pl   = P_c[cellidf[i][0]]
            rhoul=rhou_c[cellidf[i][0]]
            rhovl=rhov_c[cellidf[i][0]]
            rhoEl=rhoE_c[cellidf[i][0]]
            
            """
            if name[i] == 11 or name[i] == 22:
                rhor=rho_c[cellidf[i][1]]
                Pr=P_c[cellidf[i][1]]
                rhour=rhou_c[cellidf[i][1]]
                rhovr=rhov_c[cellidf[i][1]]
                rhoEr=rhoE_c[cellidf[i][1]]
                #### Ordre 2 ####
                center_left[:] = centerc[cellidf[i][0]][0:2]
                center_right[:] = centerc[cellidf[i][1]][0:2]
                
                w_x_left = w_x[cellidf[i][0]]; w_x_right = w_x[cellidf[i][1]]
                w_y_left = w_y[cellidf[i][0]]; w_y_right = w_y[cellidf[i][1]]
                
                psi_left  = psi[cellidf[i][0]];  psi_right  = psi[cellidf[i][1]]

                r_l[0] = centerf[i][0] - center_left[0]; r_r[0] = centerf[i][0] - center_right[0] - shift[cellidf[i][1]][0] 
                r_l[1] = centerf[i][1] - center_left[1]; r_r[1] = centerf[i][1] - center_right[1] 
            
            if name[i] == 33 or name[i] == 44:
                
                rhor=rho_c[cellidf[i][1]]
                Pr=P_c[cellidf[i][1]]
                rhour=rhou_c[cellidf[i][1]]
                rhovr=rhov_c[cellidf[i][1]]
                rhoEr=rhoE_c[cellidf[i][1]]
                #### Ordre 2 ####
                center_left[:] = centerc[cellidf[i][0]][0:2]
                center_right[:] = centerc[cellidf[i][1]][0:2]
                
                w_x_left = w_x[cellidf[i][0]]; w_x_right = w_x[cellidf[i][1]]
                w_y_left = w_y[cellidf[i][0]]; w_y_right = w_y[cellidf[i][1]]
                
                psi_left  = psi[cellidf[i][0]];  psi_right  = psi[cellidf[i][1]]
                
                r_l[0] = centerf[i][0] - center_left[0]; r_r[0] = centerf[i][0] - center_right[0] 
                r_l[1] = centerf[i][1] - center_left[1]; r_r[1] = centerf[i][1] - center_right[1] - shift[cellidf[i][1]][1] 
               """ 
            if name[i] == 0:
         
                rhor=rho_c[cellidf[i][1]]
                Pr=P_c[cellidf[i][1]]
                rhour=rhou_c[cellidf[i][1]]
                rhovr=rhov_c[cellidf[i][1]]
                rhoEr=rhoE_c[cellidf[i][1]]

                flux_rho,flux_rhou,flux_rhov,flux_rhoE = compute_flux_euler_2d_rusanov(rhol,Pl,rhoul,rhovl,rhoEl,rhor,Pr,rhour,rhovr,rhoEr,norm, mesu,gamma)
                
   
                rez_rho[cellidf[i][0]] -= flux_rho
                rez_rho[cellidf[i][1]] += flux_rho
                rez_rhou[cellidf[i][0]] -= flux_rhou
                rez_rhou[cellidf[i][1]] += flux_rhou
                rez_rhov[cellidf[i][0]] -= flux_rhov
                rez_rhov[cellidf[i][1]] += flux_rhov
                rez_rhoE[cellidf[i][0]] -= flux_rhoE
                rez_rhoE[cellidf[i][1]] += flux_rhoE
            
            elif name[i] == 10:
         
                rhor=rho_h[halofid[i]]
                Pr=P_h[halofid[i]]
                rhour=rhou_h[halofid[i]]
                rhovr=rhov_h[halofid[i]]
                rhoEr=rhoE_h[halofid[i]]

                flux_rho,flux_rhou,flux_rhov,flux_rhoE = compute_flux_euler_2d_rusanov(rhol,Pl,rhoul,rhovl,rhoEl,rhor,Pr,rhour,rhovr,rhoEr,norm, mesu,gamma)
               
                rez_rho[cellidf[i][0]] -= flux_rho
                rez_rhou[cellidf[i][0]] -= flux_rhou
                rez_rhov[cellidf[i][0]] -= flux_rhov
                rez_rhoE[cellidf[i][0]] -= flux_rhoE
    
            else:
         
                rhor=rho_g[i]
                Pr=P_g[i]
                rhour=rhou_g[i]
                rhovr=rhov_g[i]
                rhoEr=rhoE_g[i]

                flux_rho,flux_rhou,flux_rhov,flux_rhoE = compute_flux_euler_2d_rusanov(rhol,Pl,rhoul,rhovl,rhoEl,rhor,Pr,rhour,rhovr,rhoEr,norm, mesu,gamma)
               
                rez_rho[cellidf[i][0]] -= flux_rho
                rez_rhou[cellidf[i][0]] -= flux_rhou
                rez_rhov[cellidf[i][0]] -= flux_rhov
                rez_rhoE[cellidf[i][0]] -= flux_rhoE             
####################
@njit("void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], int32[:,:], int32[:], float64[:,:], float64[:], uint32[:], float64)", cache= True)                   
def explicitscheme_euler_2d_Roe(rez_rho,rez_rhou,rez_rhov,
                                    rez_rhoE,rho_c,P_c,
                                    rhou_c,rhov_c,rhoE_c,
                                    rho_g,P_g,rhou_g,
                                    rhov_g,rhoE_g,rho_h,
                                    P_h,rhou_h,rhov_h,
                                    rhoE_h, cellidf,halofid,
                                    normal,mesurf,name,gamma):

    nbface = len(cellidf)
       
    rez_rho[:]  = np.zeros(len(rez_rho))
    rez_rhou[:] = np.zeros(len(rez_rhou))
    rez_rhov[:] = np.zeros(len(rez_rhov))
    rez_rhoE[:] = np.zeros(len(rez_rhoE))

    for i in range(nbface):
  
            norm = normal[i]/mesurf[i]
            mesu = mesurf[i]
            
            rhol = rho_c[cellidf[i][0]]
            Pl   = P_c[cellidf[i][0]]
            rhoul= rhou_c[cellidf[i][0]]
            rhovl= rhov_c[cellidf[i][0]]
            rhoEl= rhoE_c[cellidf[i][0]]
                        
            if name[i] == 0:
         
                rhor=rho_c[cellidf[i][1]]
                Pr=P_c[cellidf[i][1]]
                rhour=rhou_c[cellidf[i][1]]
                rhovr=rhov_c[cellidf[i][1]]
                rhoEr=rhoE_c[cellidf[i][1]]
                             
                flux_rho,flux_rhou,flux_rhov,flux_rhoE = compute_flux_euler_2d_Roe(rhol,Pl,rhoul,rhovl,rhoEl,rhor,Pr,rhour,rhovr,rhoEr,norm, mesu,gamma)
   
                rez_rho[cellidf[i][0]] -= flux_rho
                rez_rho[cellidf[i][1]] += flux_rho
                rez_rhou[cellidf[i][0]] -= flux_rhou
                rez_rhou[cellidf[i][1]] += flux_rhou
                rez_rhov[cellidf[i][0]] -= flux_rhov
                rez_rhov[cellidf[i][1]] += flux_rhov
                rez_rhoE[cellidf[i][0]] -= flux_rhoE
                rez_rhoE[cellidf[i][1]] += flux_rhoE
            
            elif name[i] == 10:
         
                rhor=rho_h[halofid[i]]
                Pr=P_h[halofid[i]]
                rhour=rhou_h[halofid[i]]
                rhovr=rhov_h[halofid[i]]
                rhoEr=rhoE_h[halofid[i]]
             
                flux_rho,flux_rhou,flux_rhov,flux_rhoE = compute_flux_euler_2d_Roe(rhol,Pl,rhoul,rhovl,rhoEl,rhor,Pr,rhour,rhovr,rhoEr,norm, mesu,gamma)
                rez_rho[cellidf[i][0]] -= flux_rho
                rez_rhou[cellidf[i][0]] -= flux_rhou
                rez_rhov[cellidf[i][0]] -= flux_rhov
                rez_rhoE[cellidf[i][0]] -= flux_rhoE
    
            else:
         
                rhor=rho_g[i]
                Pr=P_g[i]
                rhour=rhou_g[i]
                rhovr=rhov_g[i]
                rhoEr=rhoE_g[i]
                
                flux_rho,flux_rhou,flux_rhov,flux_rhoE = compute_flux_euler_2d_Roe(rhol,Pl,rhoul,rhovl,rhoEl,rhor,Pr,rhour,rhovr,rhoEr,norm, mesu,gamma)
                rez_rho[cellidf[i][0]]  -= flux_rho
                rez_rhou[cellidf[i][0]] -= flux_rhou
                rez_rhov[cellidf[i][0]] -= flux_rhov
                rez_rhoE[cellidf[i][0]] -= flux_rhoE       
@njit("void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64, float64, float64[:])", cache= True)                                    
def update_euler_2d_fvc(rho_c, P_c,rhou_c, rhov_c,rhoE_c,
                        rez_rho,rez_rhou,rez_rhov,rez_rhoE,
                        gamma,dtime,vol):

    rho_c[:]   += dtime*((rez_rho[:])/vol[:])
    rhou_c[:]  += dtime*((rez_rhou[:])/vol[:])
    rhov_c[:]  += dtime*((rez_rhov[:])/vol[:])
    rhoE_c[:]  += dtime*((rez_rhoE[:])/vol[:])
    P_c[:]      = (gamma-1)*(rhoE_c[:]-0.5*(rhou_c[:]*rhou_c[:] + rhov_c[:]*rhov_c[:])/rho_c[:])
    
