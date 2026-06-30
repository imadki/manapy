from manapy.backends.compile_fun import compile
import numpy as np

def _node_for_interpolation_2d(xCenterForInterp: 'float64[:,:]', yCenterForInterp: 'float64[:,:]', nodefid: 'int32[:,:]', cellfid: 'int32[:,:]', halofid: 'int32[:]', cellcenter: 'float64[:,:]', vertexcenter: 'float64[:,:]', ghostcenter: 'float64[:,:]', halocenter: 'float64[:,:]', name: 'uint32[:]'):

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
            
def _node_value_for_interpolation_2d(ValForInterp: 'float64[:,:]', w_cell: 'float64[:]', w_node: 'float64[:]', w_ghost: 'float64[:]', w_halo: 'float64[:]', nodefid: 'int32[:,:]', cellfid: 'int32[:,:]', halofid: 'int32[:]', name: 'uint32[:]'):


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

def _weight_parameters_carac_2d(xCenterForInterp: 'float64[:]', yCenterForInterp: 'float64[:]', X0: 'float64', Y0: 'float64'):

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
        
def _set_carac_field_2d(ValForInterp: 'float64[:]', xCenterForInterp: 'float64[:]', yCenterForInterp: 'float64[:]', X0: 'float64', Y0: 'float64'):

    w_carac = 0.
    R_x = 0.
    R_y = 0.
    lambda_x = 0.
    lambda_y = 0.
    R_x, R_y, lambda_x, lambda_y=_weight_parameters_carac_2d(xCenterForInterp, yCenterForInterp, X0, Y0)
    
    for i in range(0, 4):
        
        xdiff = xCenterForInterp[i] - X0
        ydiff = yCenterForInterp[i] - Y0
        
        alpha_interp = (1. + lambda_x*xdiff + lambda_y*ydiff)/ (4. + lambda_x*R_x + lambda_y*R_y)
        
        w_carac  += alpha_interp * ValForInterp[i]
   
    return w_carac

def _initialisation_euler_2d(rho: 'float64[:]', P: 'float64[:]', rhou: 'float64[:]', rhov: 'float64[:]', rhoE: 'float64[:]', center: 'float64[:,:]', rho1: 'float64', rho2: 'float64', P1: 'float64', P2: 'float64', xm: 'float64', gamma: 'float64', u1: 'float64', u2: 'float64', v1: 'float64', v2: 'float64', choix: 'int32'):
    
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

def _ghost_value_DoubleMach(rhog: 'float64[:]', Pg: 'float64[:]', rhoug: 'float64[:]', rhovg: 'float64[:]', ug: 'float64[:]', vg: 'float64[:]', rhoEg: 'float64[:]', rhoc: 'float64[:]', Pc: 'float64[:]', rhouc: 'float64[:]', rhovc: 'float64[:]', rhoEc: 'float64[:]', cellid: 'int32[:,:]', name: 'uint32[:]', normal: 'float64[:,:]', mesure: 'float64[:]', center: 'float64[:,:]', t: 'float64'):
    
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

def _ghost_value_TubeSchok(rhog: 'float64[:]', Pg: 'float64[:]', rhoug: 'float64[:]', rhovg: 'float64[:]', ug: 'float64[:]', vg: 'float64[:]', rhoEg: 'float64[:]', rhoc: 'float64[:]', Pc: 'float64[:]', rhouc: 'float64[:]', rhovc: 'float64[:]', rhoEc: 'float64[:]', cellid: 'int32[:,:]', name: 'uint32[:]', normal: 'float64[:,:]', mesure: 'float64[:]', center: 'float64[:,:]', t: 'float64'):
    
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

def _ghost_value_Neumann(rhog: 'float64[:]', Pg: 'float64[:]', rhoug: 'float64[:]', rhovg: 'float64[:]', ug: 'float64[:]', vg: 'float64[:]', rhoEg: 'float64[:]', rhoc: 'float64[:]', Pc: 'float64[:]', rhouc: 'float64[:]', rhovc: 'float64[:]', rhoEc: 'float64[:]', cellid: 'int32[:,:]', name: 'uint32[:]', normal: 'float64[:,:]', mesure: 'float64[:]', center: 'float64[:,:]', t: 'float64'):
    
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

def _ghost_value_Gamm2D(rhog: 'float64[:]', Pg: 'float64[:]', rhoug: 'float64[:]', rhovg: 'float64[:]', ug: 'float64[:]', vg: 'float64[:]', rhoEg: 'float64[:]', rhoc: 'float64[:]', Pc: 'float64[:]', rhouc: 'float64[:]', rhovc: 'float64[:]', rhoEc: 'float64[:]', cellid: 'int32[:,:]', name: 'uint32[:]', normal: 'float64[:,:]', mesure: 'float64[:]', center: 'float64[:,:]', t: 'float64'):
    
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

def _halghost_value_TubeSchok(rhog: 'float64[:]', Pg: 'float64[:]', rhoug: 'float64[:]', rhovg: 'float64[:]', ug: 'float64[:]', vg: 'float64[:]', rhoEg: 'float64[:]', rho_halo: 'float64[:]', P_halo: 'float64[:]', rhou_halo: 'float64[:]', rhov_halo: 'float64[:]', rhoE_halo: 'float64[:]', cellid: 'int32[:,:]', name: 'uint32[:]', normal: 'float64[:,:]', halonodes: 'uint32[:]', haloghostcenter: 'float64[:,:,:]', haloghostfaceinfo: 'float64[:,:,:]', t: 'float64'):
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

def _halghost_value_Neumann(rhog: 'float64[:]', Pg: 'float64[:]', rhoug: 'float64[:]', rhovg: 'float64[:]', ug: 'float64[:]', vg: 'float64[:]', rhoEg: 'float64[:]', rho_halo: 'float64[:]', P_halo: 'float64[:]', rhou_halo: 'float64[:]', rhov_halo: 'float64[:]', rhoE_halo: 'float64[:]', cellid: 'int32[:,:]', name: 'uint32[:]', normal: 'float64[:,:]', halonodes: 'uint32[:]', haloghostcenter: 'float64[:,:,:]', haloghostfaceinfo: 'float64[:,:,:]', t: 'float64'):
    
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

def _halghost_value_Gamm2D(rhog: 'float64[:]', Pg: 'float64[:]', rhoug: 'float64[:]', rhovg: 'float64[:]', ug: 'float64[:]', vg: 'float64[:]', rhoEg: 'float64[:]', rho_halo: 'float64[:]', P_halo: 'float64[:]', rhou_halo: 'float64[:]', rhov_halo: 'float64[:]', rhoE_halo: 'float64[:]', cellid: 'int32[:,:]', name: 'uint32[:]', normal: 'float64[:,:]', halonodes: 'uint32[:]', haloghostcenter: 'float64[:,:,:]', haloghostfaceinfo: 'float64[:,:,:]', t: 'float64'):
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
                     
                                                   
def _halghost_value_DoubleMach(rhog: 'float64[:]', Pg: 'float64[:]', rhoug: 'float64[:]', rhovg: 'float64[:]', ug: 'float64[:]', vg: 'float64[:]', rhoEg: 'float64[:]', rho_halo: 'float64[:]', P_halo: 'float64[:]', rhou_halo: 'float64[:]', rhov_halo: 'float64[:]', rhoE_halo: 'float64[:]', cellid: 'int32[:,:]', name: 'uint32[:]', normal: 'float64[:,:]', halonodes: 'uint32[:]', haloghostcenter: 'float64[:,:,:]', haloghostfaceinfo: 'float64[:,:,:]', t: 'float64'):
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

def _time_step_euler_2d(rho: 'float64[:]', P: 'float64[:]', rhou: 'float64[:]', rhov: 'float64[:]', cfl: 'float64', normal: 'float64[:,:]', mesure: 'float64[:]', volume: 'float64[:]', faceid: 'int32[:,:]', gamma: 'float64', dt_c: 'float64[:]'):
    
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

def _departure_euler_2d(X0: 'float64[:]', Y0: 'float64[:]', rhof: 'float64[:]', rhouf: 'float64[:]', rhovf: 'float64[:]', rhoValForInterp: 'float64[:,:]', uValForInterp: 'float64[:,:]', vValForInterp: 'float64[:,:]', xCenterForInterp: 'float64[:,:]', yCenterForInterp: 'float64[:,:]', nodeidf: 'int32[:,:]', normalf: 'float64[:,:]', mesuref: 'float64[:]', centerf: 'float64[:,:]', vertexn: 'float64[:,:]', centerc: 'float64[:,:]', centerg: 'float64[:,:]', centerh: 'float64[:,:]', name: 'uint32[:]', dt: 'float64', alphaf: 'float64'):

    nbfaces = len(nodeidf)

    
    X0[:] = centerf[:,0]
    Y0[:] = centerf[:,1]
    

  
#    u_ed = 0.
#    v_ed = 0.

    for i in range(nbfaces):
           
            rho_ed = _set_carac_field_2d(rhoValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], X0[i], Y0[i])
            u_ed   = _set_carac_field_2d(uValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], X0[i], Y0[i])/rho_ed
            v_ed   = _set_carac_field_2d(vValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], X0[i], Y0[i])/rho_ed
    
            
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
#        u    =  _set_carac_field_2d(uValForInterp[i], xCenterForInterp[i], yCenterForInterp[i],xrk1,yrk1)/rho_ed
#        v    =  _set_carac_field_2d(vValForInterp[i], xCenterForInterp[i], yCenterForInterp[i],xrk1,yrk1)/rho_ed
#       
#      
#                    
#        xrk2 = 0.75*X0[i]+0.25*xrk1-0.25*dt*alphaf*u
#        yrk2 = 0.75*Y0[i]+0.25*yrk1-0.25*dt*alphaf*v
#        
#        
#        u    =  _set_carac_field_2d(uValForInterp[i], xCenterForInterp[i], yCenterForInterp[i],xrk2,yrk2)/rho_ed
#        v    =  _set_carac_field_2d(vValForInterp[i], xCenterForInterp[i], yCenterForInterp[i],xrk2,yrk2)/rho_ed
#        
#        
#        X0[i] = (X0[i] +2.0*xrk2-2.0*alphaf*dt*u)/3.0
#        Y0[i] = (Y0[i] +2.0*yrk2-2.0*alphaf*dt*v)/3.0
      

def _predictor_euler_2d(rho_p: 'float64[:]', P_p: 'float64[:]', rhou_p: 'float64[:]', rhov_p: 'float64[:]', rhoE_p: 'float64[:]', rhoValForInterp: 'float64[:,:]', rhouValForInterp: 'float64[:,:]', rhovValForInterp: 'float64[:,:]', rhoEValForInterp: 'float64[:,:]', xCenterForInterp: 'float64[:,:]', yCenterForInterp: 'float64[:,:]', X0: 'float64[:]', Y0: 'float64[:]', ugradfacex: 'float64[:]', ugradfacey: 'float64[:]', vgradfacex: 'float64[:]', vgradfacey: 'float64[:]', Pgradfacex: 'float64[:]', Pgradfacey: 'float64[:]', gamma: 'float64', d_t: 'float64', alphaf: 'float64', nodeidf: 'int32[:,:]', normal: 'float64[:,:]', mesuref: 'float64[:]'):
 
    nbfaces = len(nodeidf)

   
    for i in range(nbfaces):
        
        rho_ed  = _set_carac_field_2d(rhoValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], X0[i], Y0[i])
        rhou_ed = _set_carac_field_2d(rhouValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], X0[i], Y0[i])
        rhov_ed = _set_carac_field_2d(rhovValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], X0[i], Y0[i])
        rhoE_ed = _set_carac_field_2d(rhoEValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], X0[i], Y0[i],)

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
        
        
def _compute_flux_euler_2d_fvc(rhop: 'float64', Pp: 'float64', rhoup: 'float64', rhovp: 'float64', rhoEp: 'float64', normal: 'float64[:]', mesure: 'float64'):
          
        flux_rho   = (rhoup*normal[0] + rhovp*normal[1])*mesure
        flux_rhou  = ((rhoup*rhoup/rhop + Pp)*normal[0] + (rhoup*rhovp/rhop)*normal[1])*mesure
        flux_rhov  = ((rhovp*rhoup/rhop)*normal[0] + (rhovp*rhovp/rhop+Pp)*normal[1])*mesure
        flux_rhoE  = (rhoup/rhop*(rhoEp + Pp)*normal[0] + rhovp/rhop*(rhoEp+Pp)*normal[1])*mesure
        
        
        return flux_rho,flux_rhou,flux_rhov,flux_rhoE     
def _compute_flux_euler_2d_rusanov(rhol: 'float64', Pl: 'float64', rhoul: 'float64', rhovl: 'float64', rhoEl: 'float64', rhor: 'float64', Pr: 'float64', rhour: 'float64', rhovr: 'float64', rhoEr: 'float64', normal: 'float64[:]', mesure: 'float64', gamma: 'float64'):
    
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

def _compute_flux_euler_2d_Roe(rhol: 'float64', Pl: 'float64', rhoul: 'float64', rhovl: 'float64', rhoEl: 'float64', rhor: 'float64', Pr: 'float64', rhour: 'float64', rhovr: 'float64', rhoEr: 'float64', normal: 'float64[:]', mesure: 'float64', gamma: 'float64'):
    
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

def _explicitscheme_euler_2d_fvc(rez_rho: 'float64[:]', rez_rhou: 'float64[:]', rez_rhov: 'float64[:]', rez_rhoE: 'float64[:]', rho_p: 'float64[:]', P_p: 'float64[:]', rhou_p: 'float64[:]', rhov_p: 'float64[:]', rhoE_p: 'float64[:]', cellidf: 'int32[:,:]', normal: 'float64[:,:]', mesurf: 'float64[:]', name: 'uint32[:]'):

    nbface = len(cellidf)
   
    rez_rho[:]  = np.zeros(len(rez_rho))
    rez_rhou[:] = np.zeros(len(rez_rhou))
    rez_rhov[:] = np.zeros(len(rez_rhov))
    rez_rhoE[:] = np.zeros(len(rez_rhoE))

    for i in range(nbface):
  
            norm = normal[i]/mesurf[i]
            mesu = mesurf[i]

                  
            flux_rho,flux_rhou,flux_rhov,flux_rhoE = _compute_flux_euler_2d_fvc(rho_p[i],P_p[i],rhou_p[i],rhov_p[i],rhoE_p[i], norm, mesu)
            
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
                
def _explicitscheme_euler_2d_rusanov(rez_rho: 'float64[:]', rez_rhou: 'float64[:]', rez_rhov: 'float64[:]', rez_rhoE: 'float64[:]', rho_c: 'float64[:]', P_c: 'float64[:]', rhou_c: 'float64[:]', rhov_c: 'float64[:]', rhoE_c: 'float64[:]', rho_g: 'float64[:]', P_g: 'float64[:]', rhou_g: 'float64[:]', rhov_g: 'float64[:]', rhoE_g: 'float64[:]', rho_h: 'float64[:]', P_h: 'float64[:]', rhou_h: 'float64[:]', rhov_h: 'float64[:]', rhoE_h: 'float64[:]', cellidf: 'int32[:,:]', halofid: 'int32[:]', normal: 'float64[:,:]', mesurf: 'float64[:]', name: 'uint32[:]', gamma: 'float64'):
    
    

    
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

                flux_rho,flux_rhou,flux_rhov,flux_rhoE = _compute_flux_euler_2d_rusanov(rhol,Pl,rhoul,rhovl,rhoEl,rhor,Pr,rhour,rhovr,rhoEr,norm, mesu,gamma)
                
   
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

                flux_rho,flux_rhou,flux_rhov,flux_rhoE = _compute_flux_euler_2d_rusanov(rhol,Pl,rhoul,rhovl,rhoEl,rhor,Pr,rhour,rhovr,rhoEr,norm, mesu,gamma)
               
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

                flux_rho,flux_rhou,flux_rhov,flux_rhoE = _compute_flux_euler_2d_rusanov(rhol,Pl,rhoul,rhovl,rhoEl,rhor,Pr,rhour,rhovr,rhoEr,norm, mesu,gamma)
               
                rez_rho[cellidf[i][0]] -= flux_rho
                rez_rhou[cellidf[i][0]] -= flux_rhou
                rez_rhov[cellidf[i][0]] -= flux_rhov
                rez_rhoE[cellidf[i][0]] -= flux_rhoE
####################

# ---- variable-gamma (multispecies) Rusanov path --------------------------
# Same fluxes as the scalar version, but the sound speed (hence the Rusanov
# wave speed) uses the per-cell ratio of specific heats, and the pressure
# update uses the per-cell gamma. gamma is supplied as cell/ghost/halo arrays
# built from the composition (e.g. MixtureThermo.mixture_gamma).
def _compute_flux_euler_2d_rusanov_vg(rhol: 'float64', Pl: 'float64', rhoul: 'float64', rhovl: 'float64', rhoEl: 'float64', rhor: 'float64', Pr: 'float64', rhour: 'float64', rhovr: 'float64', rhoEr: 'float64', normal: 'float64[:]', mesure: 'float64', gammal: 'float64', gammar: 'float64'):
    ql = (rhoul*normal[0] + rhovl*normal[1])
    qr = (rhour*normal[0] + rhovl*normal[1])
    cl = np.sqrt(gammal*Pl/rhol)
    cr = np.sqrt(gammar*Pr/rhor)

    fl_rho  = (rhoul*normal[0] + rhovl*normal[1])
    fl_rhou = ((rhoul*rhoul/rhol+Pl)*normal[0] + (rhoul*rhovl/rhol)*normal[1])
    fl_rhov = ((rhovl*rhoul/rhol)*normal[0] + (rhovl*rhovl/rhol+Pl)*normal[1])
    fl_rhoE = (rhoul/rhol*(rhoEl+Pl)*normal[0] + rhovl/rhol*(rhoEl+Pl)*normal[1])

    fr_rho =  (rhour*normal[0] + rhovr*normal[1])
    fr_rhou = ((rhour*rhour/rhor+Pr)*normal[0] + (rhour*rhovr/rhor)*normal[1])
    fr_rhov = ((rhovr*rhour/rhor)*normal[0] + (rhovr*rhovr/rhor+Pr)*normal[1])
    fr_rhoE = (rhour/rhor*(rhoEr+Pr)*normal[0] + rhovr/rhor*(rhoEr+Pr)*normal[1])

    Ll = max(np.fabs(ql/rhol - cl), np.fabs(ql/rhol), np.fabs(ql/rhol + cl))
    Lr = max(np.fabs(qr/rhor - cr), np.fabs(qr/rhor), np.fabs(qr/rhor + cr))
    S = Ll if Ll > Lr else Lr

    flux_rho  = (0.5*(fl_rho + fr_rho)   - 0.5*S*(rhor - rhol))*mesure
    flux_rhou = (0.5*(fl_rhou + fr_rhou) - 0.5*S*(rhour - rhoul))*mesure
    flux_rhov = (0.5*(fl_rhov + fr_rhov) - 0.5*S*(rhovr - rhovl))*mesure
    flux_rhoE = (0.5*(fl_rhoE + fr_rhoE) - 0.5*S*(rhoEr - rhoEl))*mesure
    return flux_rho, flux_rhou, flux_rhov, flux_rhoE


def _explicitscheme_euler_2d_rusanov_vg(rez_rho: 'float64[:]', rez_rhou: 'float64[:]', rez_rhov: 'float64[:]', rez_rhoE: 'float64[:]', rho_c: 'float64[:]', P_c: 'float64[:]', rhou_c: 'float64[:]', rhov_c: 'float64[:]', rhoE_c: 'float64[:]', rho_g: 'float64[:]', P_g: 'float64[:]', rhou_g: 'float64[:]', rhov_g: 'float64[:]', rhoE_g: 'float64[:]', rho_h: 'float64[:]', P_h: 'float64[:]', rhou_h: 'float64[:]', rhov_h: 'float64[:]', rhoE_h: 'float64[:]', cellidf: 'int32[:,:]', halofid: 'int32[:]', normal: 'float64[:,:]', mesurf: 'float64[:]', name: 'uint32[:]', gamma_c: 'float64[:]', gamma_g: 'float64[:]', gamma_h: 'float64[:]'):
    nbface = len(cellidf)
    rez_rho[:]  = np.zeros(len(rez_rho))
    rez_rhou[:] = np.zeros(len(rez_rhou))
    rez_rhov[:] = np.zeros(len(rez_rhov))
    rez_rhoE[:] = np.zeros(len(rez_rhoE))
    for i in range(nbface):
        norm = normal[i]/mesurf[i]
        mesu = mesurf[i]
        il = cellidf[i][0]
        rhol = rho_c[il]; Pl = P_c[il]; rhoul = rhou_c[il]; rhovl = rhov_c[il]; rhoEl = rhoE_c[il]
        gl = gamma_c[il]
        if name[i] == 0:
            ir = cellidf[i][1]
            rhor = rho_c[ir]; Pr = P_c[ir]; rhour = rhou_c[ir]; rhovr = rhov_c[ir]; rhoEr = rhoE_c[ir]; gr = gamma_c[ir]
            flux_rho, flux_rhou, flux_rhov, flux_rhoE = _compute_flux_euler_2d_rusanov_vg(rhol,Pl,rhoul,rhovl,rhoEl,rhor,Pr,rhour,rhovr,rhoEr,norm,mesu,gl,gr)
            rez_rho[il] -= flux_rho; rez_rho[ir] += flux_rho
            rez_rhou[il] -= flux_rhou; rez_rhou[ir] += flux_rhou
            rez_rhov[il] -= flux_rhov; rez_rhov[ir] += flux_rhov
            rez_rhoE[il] -= flux_rhoE; rez_rhoE[ir] += flux_rhoE
        elif name[i] == 10:
            h = halofid[i]
            rhor = rho_h[h]; Pr = P_h[h]; rhour = rhou_h[h]; rhovr = rhov_h[h]; rhoEr = rhoE_h[h]; gr = gamma_h[h]
            flux_rho, flux_rhou, flux_rhov, flux_rhoE = _compute_flux_euler_2d_rusanov_vg(rhol,Pl,rhoul,rhovl,rhoEl,rhor,Pr,rhour,rhovr,rhoEr,norm,mesu,gl,gr)
            rez_rho[il] -= flux_rho; rez_rhou[il] -= flux_rhou; rez_rhov[il] -= flux_rhov; rez_rhoE[il] -= flux_rhoE
        else:
            rhor = rho_g[i]; Pr = P_g[i]; rhour = rhou_g[i]; rhovr = rhov_g[i]; rhoEr = rhoE_g[i]; gr = gamma_g[i]
            flux_rho, flux_rhou, flux_rhov, flux_rhoE = _compute_flux_euler_2d_rusanov_vg(rhol,Pl,rhoul,rhovl,rhoEl,rhor,Pr,rhour,rhovr,rhoEr,norm,mesu,gl,gr)
            rez_rho[il] -= flux_rho; rez_rhou[il] -= flux_rhou; rez_rhov[il] -= flux_rhov; rez_rhoE[il] -= flux_rhoE


def _doubleflux_residual_euler_2d(rez_rho: 'float64[:]', rez_rhou: 'float64[:]', rez_rhov: 'float64[:]', rez_rhoE: 'float64[:]', rho_c: 'float64[:]', P_c: 'float64[:]', rhou_c: 'float64[:]', rhov_c: 'float64[:]', rho_g: 'float64[:]', P_g: 'float64[:]', rhou_g: 'float64[:]', rhov_g: 'float64[:]', rho_h: 'float64[:]', P_h: 'float64[:]', rhou_h: 'float64[:]', rhov_h: 'float64[:]', gamma_c: 'float64[:]', cellfaceid: 'int32[:,:]', facecellid: 'int32[:,:]', facehalofid: 'int32[:]', normal: 'float64[:,:]', mesurf: 'float64[:]', name: 'uint32[:]'):
    # Double-flux method (Abgrall-Billet): cell i is updated with its OWN frozen
    # gamma_i. The neighbour's energy is reinterpreted in cell i's gamma frame
    # (rhoE_j = p_j/(gamma_i-1) + KE_j) so a uniform-pressure contact stays in exact
    # pressure equilibrium (no spurious oscillation). Per-cell loop over its faces;
    # the flux is computed twice per face (once per adjacent cell's gamma) -> not
    # strictly energy-conservative, but oscillation-free at multi-gamma contacts.
    nbcell = len(cellfaceid)
    rez_rho[:] = np.zeros(len(rez_rho))
    rez_rhou[:] = np.zeros(len(rez_rhou))
    rez_rhov[:] = np.zeros(len(rez_rhov))
    rez_rhoE[:] = np.zeros(len(rez_rhoE))
    norm = np.zeros(2)
    for i in range(nbcell):
        gi = gamma_c[i]
        rli = rho_c[i]; uli = rhou_c[i] / rli; vli = rhov_c[i] / rli; pli = P_c[i]
        KEi = 0.5 * rli * (uli * uli + vli * vli)
        rhoEi = pli / (gi - 1.0) + KEi
        nf = cellfaceid[i][-1]
        for jf in range(nf):
            f = cellfaceid[i][jf]
            mesu = mesurf[f]
            nx = normal[f][0] / mesu; ny = normal[f][1] / mesu
            if facecellid[f][0] == i:
                sgn = 1.0
                if name[f] == 0:
                    j = facecellid[f][1]
                    rj = rho_c[j]; uj = rhou_c[j] / rj; vj = rhov_c[j] / rj; pj = P_c[j]
                elif name[f] == 10:
                    hh = facehalofid[f]
                    rj = rho_h[hh]; uj = rhou_h[hh] / rj; vj = rhov_h[hh] / rj; pj = P_h[hh]
                else:
                    rj = rho_g[f]; uj = rhou_g[f] / rj; vj = rhov_g[f] / rj; pj = P_g[f]
            else:
                sgn = -1.0
                j = facecellid[f][0]
                rj = rho_c[j]; uj = rhou_c[j] / rj; vj = rhov_c[j] / rj; pj = P_c[j]
            norm[0] = sgn * nx; norm[1] = sgn * ny
            rhoEj = pj / (gi - 1.0) + 0.5 * rj * (uj * uj + vj * vj)
            fr, fru, frv, frE = _compute_flux_euler_2d_rusanov_vg(
                rli, pli, rli * uli, rli * vli, rhoEi,
                rj, pj, rj * uj, rj * vj, rhoEj, norm, mesu, gi, gi)
            rez_rho[i] -= fr; rez_rhou[i] -= fru; rez_rhov[i] -= frv; rez_rhoE[i] -= frE


def _update_euler_2d_vg(rho_c: 'float64[:]', P_c: 'float64[:]', rhou_c: 'float64[:]', rhov_c: 'float64[:]', rhoE_c: 'float64[:]', rez_rho: 'float64[:]', rez_rhou: 'float64[:]', rez_rhov: 'float64[:]', rez_rhoE: 'float64[:]', gamma_c: 'float64[:]', dtime: 'float64', vol: 'float64[:]'):
    rho_c[:]  += dtime*(rez_rho[:]/vol[:])
    rhou_c[:] += dtime*(rez_rhou[:]/vol[:])
    rhov_c[:] += dtime*(rez_rhov[:]/vol[:])
    rhoE_c[:] += dtime*(rez_rhoE[:]/vol[:])
    P_c[:]     = (gamma_c[:]-1.0)*(rhoE_c[:]-0.5*(rhou_c[:]*rhou_c[:] + rhov_c[:]*rhov_c[:])/rho_c[:])


def _update_euler_2d_df(rho_c: 'float64[:]', P_c: 'float64[:]', rhou_c: 'float64[:]', rhov_c: 'float64[:]', rhoE_c: 'float64[:]', rez_rho: 'float64[:]', rez_rhou: 'float64[:]', rez_rhov: 'float64[:]', rez_rhoE: 'float64[:]', gamma_c: 'float64[:]', dtime: 'float64', vol: 'float64[:]'):
    # Double-flux update: the conserved energy is re-derived from the current
    # pressure and the frozen gamma (rhoE = p/(gamma-1) + KE) before adding the
    # residual -- this resync is what removes the multi-gamma contact oscillation.
    ke_old = 0.5*(rhou_c[:]*rhou_c[:] + rhov_c[:]*rhov_c[:])/rho_c[:]
    rhoE_c[:] = P_c[:]/(gamma_c[:]-1.0) + ke_old
    rho_c[:]  += dtime*(rez_rho[:]/vol[:])
    rhou_c[:] += dtime*(rez_rhou[:]/vol[:])
    rhov_c[:] += dtime*(rez_rhov[:]/vol[:])
    rhoE_c[:] += dtime*(rez_rhoE[:]/vol[:])
    P_c[:]     = (gamma_c[:]-1.0)*(rhoE_c[:]-0.5*(rhou_c[:]*rhou_c[:] + rhov_c[:]*rhov_c[:])/rho_c[:])


def _explicitscheme_euler_2d_Roe(rez_rho: 'float64[:]', rez_rhou: 'float64[:]', rez_rhov: 'float64[:]', rez_rhoE: 'float64[:]', rho_c: 'float64[:]', P_c: 'float64[:]', rhou_c: 'float64[:]', rhov_c: 'float64[:]', rhoE_c: 'float64[:]', rho_g: 'float64[:]', P_g: 'float64[:]', rhou_g: 'float64[:]', rhov_g: 'float64[:]', rhoE_g: 'float64[:]', rho_h: 'float64[:]', P_h: 'float64[:]', rhou_h: 'float64[:]', rhov_h: 'float64[:]', rhoE_h: 'float64[:]', cellidf: 'int32[:,:]', halofid: 'int32[:]', normal: 'float64[:,:]', mesurf: 'float64[:]', name: 'uint32[:]', gamma: 'float64'):

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
                             
                flux_rho,flux_rhou,flux_rhov,flux_rhoE = _compute_flux_euler_2d_Roe(rhol,Pl,rhoul,rhovl,rhoEl,rhor,Pr,rhour,rhovr,rhoEr,norm, mesu,gamma)
   
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
             
                flux_rho,flux_rhou,flux_rhov,flux_rhoE = _compute_flux_euler_2d_Roe(rhol,Pl,rhoul,rhovl,rhoEl,rhor,Pr,rhour,rhovr,rhoEr,norm, mesu,gamma)
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
                
                flux_rho,flux_rhou,flux_rhov,flux_rhoE = _compute_flux_euler_2d_Roe(rhol,Pl,rhoul,rhovl,rhoEl,rhor,Pr,rhour,rhovr,rhoEr,norm, mesu,gamma)
                rez_rho[cellidf[i][0]]  -= flux_rho
                rez_rhou[cellidf[i][0]] -= flux_rhou
                rez_rhov[cellidf[i][0]] -= flux_rhov
                rez_rhoE[cellidf[i][0]] -= flux_rhoE       

def _explicitscheme_euler_2d_rusanov_o2(rez_rho: 'float64[:]', rez_rhou: 'float64[:]', rez_rhov: 'float64[:]', rez_rhoE: 'float64[:]', rho_c: 'float64[:]', rhou_c: 'float64[:]', rhov_c: 'float64[:]', rhoE_c: 'float64[:]', rho_g: 'float64[:]', rhou_g: 'float64[:]', rhov_g: 'float64[:]', rhoE_g: 'float64[:]', rho_h: 'float64[:]', rhou_h: 'float64[:]', rhov_h: 'float64[:]', rhoE_h: 'float64[:]', rhox: 'float64[:]', rhoy: 'float64[:]', rhoux: 'float64[:]', rhouy: 'float64[:]', rhovx: 'float64[:]', rhovy: 'float64[:]', rhoEx: 'float64[:]', rhoEy: 'float64[:]', rhox_h: 'float64[:]', rhoy_h: 'float64[:]', rhoux_h: 'float64[:]', rhouy_h: 'float64[:]', rhovx_h: 'float64[:]', rhovy_h: 'float64[:]', rhoEx_h: 'float64[:]', rhoEy_h: 'float64[:]', psi_rho: 'float64[:]', psi_rhou: 'float64[:]', psi_rhov: 'float64[:]', psi_rhoE: 'float64[:]', psi_rho_h: 'float64[:]', psi_rhou_h: 'float64[:]', psi_rhov_h: 'float64[:]', psi_rhoE_h: 'float64[:]', cellidf: 'int32[:,:]', halofid: 'int32[:]', normal: 'float64[:,:]', mesurf: 'float64[:]', name: 'uint32[:]', centerc: 'float64[:,:]', centerf: 'float64[:,:]', halo_centvol: 'float64[:,:]', gamma: 'float64', order: 'int32'):
    # Second-order MUSCL: linear reconstruction of the conservative state at the
    # face with a Barth limiter (psi), then Rusanov flux. P is recomputed from
    # the reconstructed conservative state via the ideal-gas EOS.
    nbface = len(cellidf)
    o = order - 1.0

    rez_rho[:]  = np.zeros(len(rez_rho))
    rez_rhou[:] = np.zeros(len(rez_rhou))
    rez_rhov[:] = np.zeros(len(rez_rhov))
    rez_rhoE[:] = np.zeros(len(rez_rhoE))

    for i in range(nbface):
        norm = normal[i]/mesurf[i]
        mesu = mesurf[i]

        L = cellidf[i][0]
        rlx = centerf[i][0] - centerc[L][0]
        rly = centerf[i][1] - centerc[L][1]
        rhol  = rho_c[L]  + o*psi_rho[L] *(rhox[L]*rlx  + rhoy[L]*rly)
        rhoul = rhou_c[L] + o*psi_rhou[L]*(rhoux[L]*rlx + rhouy[L]*rly)
        rhovl = rhov_c[L] + o*psi_rhov[L]*(rhovx[L]*rlx + rhovy[L]*rly)
        rhoEl = rhoE_c[L] + o*psi_rhoE[L]*(rhoEx[L]*rlx + rhoEy[L]*rly)
        Pl = (gamma-1.0)*(rhoEl - 0.5*(rhoul*rhoul + rhovl*rhovl)/rhol)

        if name[i] == 0:
            R = cellidf[i][1]
            rrx = centerf[i][0] - centerc[R][0]
            rry = centerf[i][1] - centerc[R][1]
            rhor  = rho_c[R]  + o*psi_rho[R] *(rhox[R]*rrx  + rhoy[R]*rry)
            rhour = rhou_c[R] + o*psi_rhou[R]*(rhoux[R]*rrx + rhouy[R]*rry)
            rhovr = rhov_c[R] + o*psi_rhov[R]*(rhovx[R]*rrx + rhovy[R]*rry)
            rhoEr = rhoE_c[R] + o*psi_rhoE[R]*(rhoEx[R]*rrx + rhoEy[R]*rry)
            Pr = (gamma-1.0)*(rhoEr - 0.5*(rhour*rhour + rhovr*rhovr)/rhor)

            flux_rho,flux_rhou,flux_rhov,flux_rhoE = _compute_flux_euler_2d_rusanov(rhol,Pl,rhoul,rhovl,rhoEl,rhor,Pr,rhour,rhovr,rhoEr,norm, mesu,gamma)

            rez_rho[L]  -= flux_rho;  rez_rho[R]  += flux_rho
            rez_rhou[L] -= flux_rhou; rez_rhou[R] += flux_rhou
            rez_rhov[L] -= flux_rhov; rez_rhov[R] += flux_rhov
            rez_rhoE[L] -= flux_rhoE; rez_rhoE[R] += flux_rhoE

        elif name[i] == 10:
            h = halofid[i]
            rrx = centerf[i][0] - halo_centvol[h][0]
            rry = centerf[i][1] - halo_centvol[h][1]
            rhor  = rho_h[h]  + o*psi_rho_h[h] *(rhox_h[h]*rrx  + rhoy_h[h]*rry)
            rhour = rhou_h[h] + o*psi_rhou_h[h]*(rhoux_h[h]*rrx + rhouy_h[h]*rry)
            rhovr = rhov_h[h] + o*psi_rhov_h[h]*(rhovx_h[h]*rrx + rhovy_h[h]*rry)
            rhoEr = rhoE_h[h] + o*psi_rhoE_h[h]*(rhoEx_h[h]*rrx + rhoEy_h[h]*rry)
            Pr = (gamma-1.0)*(rhoEr - 0.5*(rhour*rhour + rhovr*rhovr)/rhor)

            flux_rho,flux_rhou,flux_rhov,flux_rhoE = _compute_flux_euler_2d_rusanov(rhol,Pl,rhoul,rhovl,rhoEl,rhor,Pr,rhour,rhovr,rhoEr,norm, mesu,gamma)

            rez_rho[L]  -= flux_rho
            rez_rhou[L] -= flux_rhou
            rez_rhov[L] -= flux_rhov
            rez_rhoE[L] -= flux_rhoE

        else:
            # physical boundary: ghost state (first order on the ghost side)
            rhor  = rho_g[i]
            rhour = rhou_g[i]
            rhovr = rhov_g[i]
            rhoEr = rhoE_g[i]
            Pr = (gamma-1.0)*(rhoEr - 0.5*(rhour*rhour + rhovr*rhovr)/rhor)

            flux_rho,flux_rhou,flux_rhov,flux_rhoE = _compute_flux_euler_2d_rusanov(rhol,Pl,rhoul,rhovl,rhoEl,rhor,Pr,rhour,rhovr,rhoEr,norm, mesu,gamma)

            rez_rho[L]  -= flux_rho
            rez_rhou[L] -= flux_rhou
            rez_rhov[L] -= flux_rhov
            rez_rhoE[L] -= flux_rhoE


def _update_euler_2d_fvc(rho_c: 'float64[:]', P_c: 'float64[:]', rhou_c: 'float64[:]', rhov_c: 'float64[:]', rhoE_c: 'float64[:]', rez_rho: 'float64[:]', rez_rhou: 'float64[:]', rez_rhov: 'float64[:]', rez_rhoE: 'float64[:]', gamma: 'float64', dtime: 'float64', vol: 'float64[:]'):

    rho_c[:]   += dtime*((rez_rho[:])/vol[:])
    rhou_c[:]  += dtime*((rez_rhou[:])/vol[:])
    rhov_c[:]  += dtime*((rez_rhov[:])/vol[:])
    rhoE_c[:]  += dtime*((rez_rhoE[:])/vol[:])
    P_c[:]      = (gamma-1)*(rhoE_c[:]-0.5*(rhou_c[:]*rhou_c[:] + rhov_c[:]*rhov_c[:])/rho_c[:])

####################
# Viscous (Navier-Stokes) fluxes -- mono-species Newtonian model,
# written in dimensional variables.
#
#   tau_xx = 2*mu*u_x - (2/3)*mu*(u_x + v_y)
#   tau_yy = 2*mu*v_y - (2/3)*mu*(u_x + v_y)
#   tau_xy = mu*(u_y + v_x)
#   energy flux  = (u*tau_xx + v*tau_xy + kappa*T_x ,  u*tau_xy + v*tau_yy + kappa*T_y)
#   kappa = mu*cp/Pr,   cp = gamma*R/(gamma-1),   T = P/(rho*R)
#
# The viscous term enters the NS system as +div(F_visc), so it accumulates with
# the *diffusion* sign convention (rez[left] += G, rez[right] -= G), i.e. the
# opposite of the inviscid Euler flux. It MUST run AFTER the inviscid scheme,
# which zeroes the rez_* accumulators; the viscous kernel only adds to them and
# leaves rez_rho untouched (no viscous mass flux).

def _face_transport_props_2d(mu_face: 'float64[:]', kappa_face: 'float64[:]',
                             T_c: 'float64[:]', T_g: 'float64[:]', T_h: 'float64[:]',
                             cellidf: 'int32[:,:]', halofid: 'int32[:]', name: 'uint32[:]',
                             law: 'int32', mu_const: 'float64', mu_ref: 'float64',
                             T_ref: 'float64', S_suth: 'float64', cp: 'float64', Pr: 'float64'):
    # Fill per-face dynamic viscosity mu and conductivity kappa. law==0: constant
    # mu; law==1: Sutherland mu(T) evaluated at the face temperature.
    nbface = len(cellidf)
    for i in range(nbface):
        if law == 0:
            muf = mu_const
        else:
            if name[i] == 0:
                Tf = 0.5 * (T_c[cellidf[i][0]] + T_c[cellidf[i][1]])
            elif name[i] == 10:
                Tf = 0.5 * (T_c[cellidf[i][0]] + T_h[halofid[i]])
            else:
                Tf = 0.5 * (T_c[cellidf[i][0]] + T_g[i])
            muf = mu_ref * (Tf / T_ref) ** 1.5 * (T_ref + S_suth) / (Tf + S_suth)
        mu_face[i] = muf
        kappa_face[i] = muf * cp / Pr


def _explicitscheme_euler_2d_viscous(rez_rhou: 'float64[:]', rez_rhov: 'float64[:]', rez_rhoE: 'float64[:]',
                                     u_gx: 'float64[:]', u_gy: 'float64[:]', v_gx: 'float64[:]', v_gy: 'float64[:]',
                                     T_gx: 'float64[:]', T_gy: 'float64[:]',
                                     u_c: 'float64[:]', u_g: 'float64[:]', u_h: 'float64[:]',
                                     v_c: 'float64[:]', v_g: 'float64[:]', v_h: 'float64[:]',
                                     mu_face: 'float64[:]', kappa_face: 'float64[:]',
                                     cellidf: 'int32[:,:]', halofid: 'int32[:]',
                                     normal: 'float64[:,:]', name: 'uint32[:]'):
    twothirds = 2.0 / 3.0
    nbface = len(cellidf)
    for i in range(nbface):
        nx = normal[i][0]
        ny = normal[i][1]
        mu = mu_face[i]
        kap = kappa_face[i]

        # face-centred gradients from the diamond reconstruction (Variable.compute_face_gradient)
        ux = u_gx[i]; uy = u_gy[i]
        vx = v_gx[i]; vy = v_gy[i]
        Tx = T_gx[i]; Ty = T_gy[i]

        div = ux + vy
        tau_xx = 2.0 * mu * ux - twothirds * mu * div
        tau_yy = 2.0 * mu * vy - twothirds * mu * div
        tau_xy = mu * (uy + vx)

        # face velocity for the viscous work term tau.u (= wall value at boundaries)
        il = cellidf[i][0]
        if name[i] == 0:
            ir = cellidf[i][1]
            uf = 0.5 * (u_c[il] + u_c[ir])
            vf = 0.5 * (v_c[il] + v_c[ir])
        elif name[i] == 10:
            uf = 0.5 * (u_c[il] + u_h[halofid[i]])
            vf = 0.5 * (v_c[il] + v_h[halofid[i]])
        else:
            uf = 0.5 * (u_c[il] + u_g[i])
            vf = 0.5 * (v_c[il] + v_g[i])

        # F_visc . n  (normal is area-scaled: |normal| = face measure)
        G_rhou = tau_xx * nx + tau_xy * ny
        G_rhov = tau_xy * nx + tau_yy * ny
        qx = uf * tau_xx + vf * tau_xy + kap * Tx
        qy = uf * tau_xy + vf * tau_yy + kap * Ty
        G_rhoE = qx * nx + qy * ny

        rez_rhou[il] += G_rhou
        rez_rhov[il] += G_rhov
        rez_rhoE[il] += G_rhoE
        if name[i] == 0:
            ir = cellidf[i][1]
            rez_rhou[ir] -= G_rhou
            rez_rhov[ir] -= G_rhov
            rez_rhoE[ir] -= G_rhoE


def _viscous_time_step_2d(rho: 'float64[:]', mu_face: 'float64[:]', kappa_face: 'float64[:]',
                          cp: 'float64', cfl: 'float64', mesure: 'float64[:]',
                          volume: 'float64[:]', faceid: 'int32[:,:]', dt_c: 'float64[:]'):
    # Diffusive stability limit, mirroring diffusion/_time_step. The effective
    # diffusivity is the max of momentum (4/3 mu/rho) and thermal (kappa/(rho cp)).
    nbelement = len(faceid)
    for i in range(nbelement):
        lam = 0.0
        for j in range(faceid[i][-1]):
            fid = faceid[i][j]
            mes = mesure[fid]
            num = (4.0 / 3.0) * mu_face[fid]
            kth = kappa_face[fid] / cp
            if kth > num:
                num = kth
            nu = num / rho[i]
            lam += nu * mes * mes / volume[i]
        if lam != 0.0:
            dt_c[i] = cfl * volume[i] / lam
        else:
            dt_c[i] = 1.e6

def _mu_sgs_smagorinsky_2d(mut: 'float64[:]', rho: 'float64[:]',
                           ux: 'float64[:]', uy: 'float64[:]', vx: 'float64[:]', vy: 'float64[:]',
                           delta: 'float64[:]', Cs: 'float64'):
    # Smagorinsky eddy viscosity per cell: mu_t = rho (Cs delta)^2 |S|,
    # |S| = sqrt(2 S_ij S_ij), S_ij the resolved strain-rate tensor.
    n = len(rho)
    for c in range(n):
        s11 = ux[c]
        s22 = vy[c]
        s12 = 0.5 * (uy[c] + vx[c])
        ss = s11 * s11 + s22 * s22 + 2.0 * s12 * s12
        smag = np.sqrt(2.0 * ss)
        cd = Cs * delta[c]
        mut[c] = rho[c] * cd * cd * smag


def _mu_sgs_wale_2d(mut: 'float64[:]', rho: 'float64[:]',
                    ux: 'float64[:]', uy: 'float64[:]', vx: 'float64[:]', vy: 'float64[:]',
                    delta: 'float64[:]', Cw: 'float64'):
    # WALE eddy viscosity (Nicoud & Ducros 1999):
    #   mu_t = rho (Cw delta)^2 (Sd:Sd)^{3/2} / [ (S:S)^{5/2} + (Sd:Sd)^{5/4} ]
    # Sd = traceless symmetric part of the square of the velocity-gradient tensor.
    onethird = 1.0 / 3.0
    n = len(rho)
    for c in range(n):
        g11 = ux[c]; g12 = uy[c]
        g21 = vx[c]; g22 = vy[c]
        # g^2 = g.g
        h11 = g11 * g11 + g12 * g21
        h12 = g11 * g12 + g12 * g22
        h21 = g21 * g11 + g22 * g21
        h22 = g21 * g12 + g22 * g22
        tr = onethird * (h11 + h22)
        sd11 = h11 - tr
        sd22 = h22 - tr
        sd12 = 0.5 * (h12 + h21)
        sdsd = sd11 * sd11 + sd22 * sd22 + 2.0 * sd12 * sd12
        s11 = g11; s22 = g22; s12 = 0.5 * (g12 + g21)
        ss = s11 * s11 + s22 * s22 + 2.0 * s12 * s12
        denom = ss ** 2.5 + sdsd ** 1.25
        if denom > 0.0:
            sra = sdsd ** 1.5 / denom
        else:
            sra = 0.0
        cd = Cw * delta[c]
        mut[c] = rho[c] * cd * cd * sra


def _add_sgs_face_props_2d(mu_face: 'float64[:]', kappa_face: 'float64[:]',
                           mut_c: 'float64[:]', mut_h: 'float64[:]',
                           cellidf: 'int32[:,:]', halofid: 'int32[:]', name: 'uint32[:]',
                           cp: 'float64', Prt: 'float64'):
    # Add the face-averaged turbulent viscosity to the (already laminar) face
    # transport props. Turbulent conductivity uses the turbulent Prandtl number.
    cp_Prt = cp / Prt
    nbface = len(cellidf)
    for i in range(nbface):
        il = cellidf[i][0]
        if name[i] == 0:
            mutf = 0.5 * (mut_c[il] + mut_c[cellidf[i][1]])
        elif name[i] == 10:
            mutf = 0.5 * (mut_c[il] + mut_h[halofid[i]])
        else:
            mutf = mut_c[il]
        mu_face[i] += mutf
        kappa_face[i] += mutf * cp_Prt


def _ghost_value_NonReflecting(rhog: 'float64[:]', Pg: 'float64[:]', rhoug: 'float64[:]', rhovg: 'float64[:]',
                               ug: 'float64[:]', vg: 'float64[:]', rhoEg: 'float64[:]',
                               rhoc: 'float64[:]', Pc: 'float64[:]', rhouc: 'float64[:]', rhovc: 'float64[:]', rhoEc: 'float64[:]',
                               cellid: 'int32[:,:]', name: 'uint32[:]', normal: 'float64[:,:]', mesure: 'float64[:]',
                               center: 'float64[:,:]', t: 'float64',
                               gamma: 'float64', rho_inf: 'float64', u_inf: 'float64', v_inf: 'float64', p_inf: 'float64'):
    # Characteristic far-field (non-reflecting) ghost state via Riemann invariants.
    # Characteristic far-field (NSCBC) non-reflecting BC: the outgoing
    # acoustic invariant is taken from the interior, the incoming one from the
    # free-stream reference (rho_inf, u_inf, v_inf, p_inf); entropy and the
    # tangential velocity follow the flow direction. Applied to every physical
    # boundary face (name >= 1). The downstream Riemann flux then sees a state
    # that lets outgoing waves leave with minimal reflection.
    gm1 = gamma - 1.0
    c_inf = np.sqrt(gamma * p_inf / rho_inf)
    nbface = len(cellid)
    for i in range(nbface):
        if name[i] == 0 or name[i] == 10:
            continue
        il = cellid[i][0]
        mes = mesure[i]
        nx = normal[i][0] / mes
        ny = normal[i][1] / mes

        rhoL = rhoc[il]
        uL = rhouc[il] / rhoL
        vL = rhovc[il] / rhoL
        pL = Pc[il]
        cL = np.sqrt(gamma * pL / rhoL)
        unL = uL * nx + vL * ny

        un_inf = u_inf * nx + v_inf * ny

        if unL <= -cL:
            # supersonic inflow: state fully imposed from the free-stream
            rho_b = rho_inf; u_b = u_inf; v_b = v_inf; p_b = p_inf
        elif unL >= cL:
            # supersonic outflow: state fully extrapolated from the interior
            rho_b = rhoL; u_b = uL; v_b = vL; p_b = pL
        else:
            # subsonic: blend the two acoustic Riemann invariants
            Rp = unL + 2.0 * cL / gm1          # outgoing (un+c branch), interior
            Rm = un_inf - 2.0 * c_inf / gm1     # incoming (un-c branch), free-stream
            un_b = 0.5 * (Rp + Rm)
            c_b = 0.25 * gm1 * (Rp - Rm)
            if un_b <= 0.0:
                # inflow: entropy and tangential velocity from the free-stream
                s_b = p_inf / rho_inf ** gamma
                utx = u_inf - un_inf * nx
                uty = v_inf - un_inf * ny
            else:
                # outflow: entropy and tangential velocity from the interior
                s_b = pL / rhoL ** gamma
                utx = uL - unL * nx
                uty = vL - unL * ny
            rho_b = (c_b * c_b / (gamma * s_b)) ** (1.0 / gm1)
            p_b = rho_b * c_b * c_b / gamma
            u_b = un_b * nx + utx
            v_b = un_b * ny + uty

        rhog[i] = rho_b
        Pg[i] = p_b
        rhoug[i] = rho_b * u_b
        rhovg[i] = rho_b * v_b
        ug[i] = u_b
        vg[i] = v_b
        rhoEg[i] = p_b / gm1 + 0.5 * rho_b * (u_b * u_b + v_b * v_b)


############################################################################
# NOTHING is compiled at import. Call setup() once (uniformly on all MPI ranks)
# before using any kernel below. This module is dimension-specific (2D), so a
# single guard suffices. Nested helpers (weight_parameters_carac -> set_carac_field,
# and the compute_flux_* used by the explicit schemes) are compiled and rebound to
# their module globals before their callers, so numba can resolve njit->njit calls.
_done = False

def setup(dim=2):
  global _done
  if dim != 2:
    raise ValueError(f"euler.fvm_utils2d expects dim=2, got {dim}")
  if _done:
    return

  # Nested helpers first (rebind the underscore globals used inside the kernels).
  global _weight_parameters_carac_2d, _set_carac_field_2d
  global _compute_flux_euler_2d_fvc, _compute_flux_euler_2d_rusanov, _compute_flux_euler_2d_Roe
  _weight_parameters_carac_2d = compile(_weight_parameters_carac_2d)
  _set_carac_field_2d = compile(_set_carac_field_2d)
  _compute_flux_euler_2d_fvc = compile(_compute_flux_euler_2d_fvc)
  _compute_flux_euler_2d_rusanov = compile(_compute_flux_euler_2d_rusanov)
  _compute_flux_euler_2d_Roe = compile(_compute_flux_euler_2d_Roe)
  global _compute_flux_euler_2d_rusanov_vg
  _compute_flux_euler_2d_rusanov_vg = compile(_compute_flux_euler_2d_rusanov_vg)

  # Public entry points (helpers are re-exported under their public names).
  global node_for_interpolation_2d, node_value_for_interpolation_2d
  global weight_parameters_carac_2d, set_carac_field_2d
  global initialisation_euler_2d, ghost_value_DoubleMach, ghost_value_TubeSchok
  global ghost_value_Neumann, ghost_value_Gamm2D
  global halghost_value_TubeSchok, halghost_value_Neumann, halghost_value_Gamm2D, halghost_value_DoubleMach
  global time_step_euler_2d, departure_euler_2d, predictor_euler_2d
  global compute_flux_euler_2d_fvc, compute_flux_euler_2d_rusanov, compute_flux_euler_2d_Roe
  global explicitscheme_euler_2d_fvc, explicitscheme_euler_2d_rusanov, explicitscheme_euler_2d_Roe
  global explicitscheme_euler_2d_rusanov_o2
  global update_euler_2d_fvc
  global explicitscheme_euler_2d_viscous, face_transport_props_2d, viscous_time_step_2d
  global ghost_value_NonReflecting
  global mu_sgs_smagorinsky_2d, mu_sgs_wale_2d, add_sgs_face_props_2d
  global compute_flux_euler_2d_rusanov_vg, explicitscheme_euler_2d_rusanov_vg, update_euler_2d_vg
  global doubleflux_residual_euler_2d, update_euler_2d_df

  node_for_interpolation_2d = compile(_node_for_interpolation_2d)
  node_value_for_interpolation_2d = compile(_node_value_for_interpolation_2d)
  weight_parameters_carac_2d = _weight_parameters_carac_2d
  set_carac_field_2d = _set_carac_field_2d
  initialisation_euler_2d = compile(_initialisation_euler_2d)
  ghost_value_DoubleMach = compile(_ghost_value_DoubleMach)
  ghost_value_TubeSchok = compile(_ghost_value_TubeSchok)
  ghost_value_Neumann = compile(_ghost_value_Neumann)
  ghost_value_Gamm2D = compile(_ghost_value_Gamm2D)
  halghost_value_TubeSchok = compile(_halghost_value_TubeSchok)
  halghost_value_Neumann = compile(_halghost_value_Neumann)
  halghost_value_Gamm2D = compile(_halghost_value_Gamm2D)
  halghost_value_DoubleMach = compile(_halghost_value_DoubleMach)
  time_step_euler_2d = compile(_time_step_euler_2d)
  departure_euler_2d = compile(_departure_euler_2d)
  predictor_euler_2d = compile(_predictor_euler_2d)
  compute_flux_euler_2d_fvc = _compute_flux_euler_2d_fvc
  compute_flux_euler_2d_rusanov = _compute_flux_euler_2d_rusanov
  compute_flux_euler_2d_Roe = _compute_flux_euler_2d_Roe
  explicitscheme_euler_2d_fvc = compile(_explicitscheme_euler_2d_fvc)
  explicitscheme_euler_2d_rusanov = compile(_explicitscheme_euler_2d_rusanov)
  explicitscheme_euler_2d_Roe = compile(_explicitscheme_euler_2d_Roe)
  explicitscheme_euler_2d_rusanov_o2 = compile(_explicitscheme_euler_2d_rusanov_o2)
  update_euler_2d_fvc = compile(_update_euler_2d_fvc)
  explicitscheme_euler_2d_viscous = compile(_explicitscheme_euler_2d_viscous)
  face_transport_props_2d = compile(_face_transport_props_2d)
  viscous_time_step_2d = compile(_viscous_time_step_2d)
  ghost_value_NonReflecting = compile(_ghost_value_NonReflecting)
  mu_sgs_smagorinsky_2d = compile(_mu_sgs_smagorinsky_2d)
  mu_sgs_wale_2d = compile(_mu_sgs_wale_2d)
  add_sgs_face_props_2d = compile(_add_sgs_face_props_2d)
  compute_flux_euler_2d_rusanov_vg = _compute_flux_euler_2d_rusanov_vg
  explicitscheme_euler_2d_rusanov_vg = compile(_explicitscheme_euler_2d_rusanov_vg)
  update_euler_2d_vg = compile(_update_euler_2d_vg)
  doubleflux_residual_euler_2d = compile(_doubleflux_residual_euler_2d)
  update_euler_2d_df = compile(_update_euler_2d_df)

  _done = True
