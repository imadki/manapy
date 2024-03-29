from numpy import int32, float, uint32
from numba import njit, prange
import numpy as np

@njit('void(float64, float64, float64, float64, float64, float64[:], float64[:])')
def compute_upwind_flux(w_l:'float', w_r:'float', u_face:'float', v_face:'float', w_face:'float', 
                        normal:'float[:]', flux_w:'float[:]'):
     
    sol = 0.
    sign = u_face * normal[0] + v_face * normal[1] + w_face * normal[2]

    if sign >= 0:
        sol = w_l
    else:
        sol = w_r
    
    flux_w[0] = sign * sol

def explicitscheme_convective_2d(rez_w:'float[:]', w_c:'float[:]', w_ghost:'float[:]', w_halo:'float[:]',
                                 u_face:'float[:]', v_face:'float[:]', w_face:'float[:]', 
                                 w_x:'float[:]', w_y:'float[:]', w_z:'float[:]', wx_halo:'float[:]', wy_halo:'float[:]', 
                                 wz_halo:'float[:]', psi:'float[:]', psi_halo:'float[:]', 
                                 centerc:'float[:,:]', centerf:'float[:,:]', centerh:'float[:,:]', centerg:'float[:,:]',
                                 cellidf:'int32[:,:]',  normalf:'float[:,:]', halofid:'int32[:]',
                                 name:'uint32[:]', innerfaces:'uint32[:]', halofaces:'uint32[:]', boundaryfaces:'uint32[:]', 
                                 periodicboundaryfaces:'uint32[:]', shift:'float[:,:]',  order:'int32'):

    
    def compute_upwind_flux(w_l:'float', w_r:'float', u_face:'float', v_face:'float', w_face:'float', 
                        normal:'float[:]', flux_w:'float[:]'):
     
        sol = 0.
        sign = u_face * normal[0] + v_face * normal[1] + w_face * normal[2]
    
        if sign >= 0:
            sol = w_l
        else:
            sol = w_r
    
        flux_w[0] = sign * sol
        
    center_left = np.zeros(2)
    center_right = np.zeros(2)
    r_l = np.zeros(2)
    r_r = np.zeros(2)
   
    normal = np.zeros(3)
    flux_w = np.zeros(1)    
   
    rez_w[:] = 0.

    for i in innerfaces:
        
        w_l = w_c[cellidf[i][0]]
        normal[:] = normalf[i][:]
        
        w_r = w_c[cellidf[i][1]]
        
        center_left[:] = centerc[cellidf[i][0]][0:2]
        center_right[:] = centerc[cellidf[i][1]][0:2]
        
        w_x_left = w_x[cellidf[i][0]]; w_x_right = w_x[cellidf[i][1]]
        w_y_left = w_y[cellidf[i][0]]; w_y_right = w_y[cellidf[i][1]]
        
        psi_left  = psi[cellidf[i][0]];  psi_right  = psi[cellidf[i][1]]
        
        r_l[0] = centerf[i][0] - center_left[0]; r_r[0] = centerf[i][0] - center_right[0]; 
        r_l[1] = centerf[i][1] - center_left[1]; r_r[1] = centerf[i][1] - center_right[1]; 
        
        w_l  = w_l  + (order - 1) * psi_left  * (w_x_left * r_l[0]  + w_y_left * r_l[1] )
        w_r  = w_r  + (order - 1) * psi_right * (w_x_right* r_r[0]  + w_y_right* r_r[1] )
        
        compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
        
        rez_w[cellidf[i][0]]  -= flux_w[0]
        rez_w[cellidf[i][1]]  += flux_w[0]
    
    for i in periodicboundaryfaces:
        
        w_l = w_c[cellidf[i][0]]
        normal[:] = normalf[i][:]
        
        w_r = w_c[cellidf[i][1]]
        
        center_left[:] = centerc[cellidf[i][0]][0:2]
        center_right[:] = centerc[cellidf[i][1]][0:2] 

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
        rez_w[cellidf[i][0]]  -= flux_w[0]
                
    
    for i in halofaces:
        
        w_l = w_c[cellidf[i][0]]
        normal[:] = normalf[i][:]
        
        w_r  = w_halo[halofid[i]]
        
        center_left[:] = centerc[cellidf[i][0]][0:2]
        center_right[:] = centerh[halofid[i]][0:2]

        w_x_left = w_x[cellidf[i][0]];  w_x_right = wx_halo[halofid[i]]
        w_y_left = w_y[cellidf[i][0]];  w_y_right = wy_halo[halofid[i]]
        
        psi_left  = psi[cellidf[i][0]];   psi_right  = psi_halo[halofid[i]]
        
        r_l[0] = centerf[i][0] - center_left[0]; r_r[0] = centerf[i][0] - center_right[0]; 
        r_l[1] = centerf[i][1] - center_left[1]; r_r[1] = centerf[i][1] - center_right[1]; 
        
        w_l  = w_l  + (order - 1) * psi_left  * (w_x_left   * r_l[0] + w_y_left   * r_l[1])
        w_r  = w_r  + (order - 1) * psi_right * (w_x_right  * r_r[0] + w_y_right  * r_r[1])
        
        compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
        rez_w[cellidf[i][0]]  -= flux_w[0]
   
    for i in boundaryfaces:
      
        w_l = w_c[cellidf[i][0]]
        normal[:] = normalf[i][:]
        
        w_r  = w_ghost[i]
        center_left[:] = centerc[cellidf[i][0]][0:2]
        
        w_x_left = w_x[cellidf[i][0]]; 
        w_y_left = w_y[cellidf[i][0]]; 
        
        psi_left  = psi[cellidf[i][0]];  
        
        r_l[0] = centerf[i][0] - center_left[0]; 
        r_l[1] = centerf[i][1] - center_left[1];
        
        w_l  = w_l  + (order - 1) * psi_left  * (w_x_left * r_l[0]  + w_y_left * r_l[1] )
        w_r  = w_r  
               
        compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
        rez_w[cellidf[i][0]]  -= flux_w[0]



def explicitscheme_convective_3d(rez_w:'float[:]', w_c:'float[:]', w_ghost:'float[:]', w_halo:'float[:]',
                                 u_face:'float[:]', v_face:'float[:]', w_face:'float[:]', 
                                 w_x:'float[:]', w_y:'float[:]', w_z:'float[:]', wx_halo:'float[:]', wy_halo:'float[:]', 
                                 wz_halo:'float[:]', psi:'float[:]', psi_halo:'float[:]', 
                                 centerc:'float[:,:]', centerf:'float[:,:]', centerh:'float[:,:]', centerg:'float[:,:]',
                                 cellidf:'int32[:,:]',  normalf:'float[:,:]', halofid:'int32[:]', name:'uint32[:]',
                                 innerfaces:'uint32[:]', halofaces:'uint32[:]', boundaryfaces:'uint32[:]', 
                                 periodicboundaryfaces:'uint32[:]', shift:'float[:,:]',  order:'int32'):
  
    
    center_left = np.zeros(3)
    center_right = np.zeros(3)
    r_l = np.zeros(3)
    r_r = np.zeros(3)
   
    normal = np.zeros(3)
    flux_w = np.zeros(1)    
   
    rez_w[:] = 0.

    for i in innerfaces:
        
        w_l = w_c[cellidf[i][0]]
        normal[:] = normalf[i][:]
        
        w_r = w_c[cellidf[i][1]]
        
        center_left[:] = centerc[cellidf[i][0]][:]
        center_right[:] = centerc[cellidf[i][1]][:]
        
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
        
        rez_w[cellidf[i][0]]  -= flux_w[0]
        rez_w[cellidf[i][1]]  += flux_w[0]
    
    for i in periodicboundaryfaces:
        
        w_l = w_c[cellidf[i][0]]
        normal[:] = normalf[i][:]
        
        w_r = w_c[cellidf[i][1]]
        
        center_left[:] = centerc[cellidf[i][0]][:]
        center_right[:] = centerc[cellidf[i][1]][:]

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
        rez_w[cellidf[i][0]]  -= flux_w[0]
                
    
    for i in halofaces:
        
        w_l = w_c[cellidf[i][0]]
        normal[:] = normalf[i][:]
        
        w_r  = w_halo[halofid[i]]
        
        center_left[:] = centerc[cellidf[i][0]][:]
        center_right[:] = centerh[halofid[i]][0:3]

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
        rez_w[cellidf[i][0]]  -= flux_w[0]
   
    for i in boundaryfaces:
      
        w_l = w_c[cellidf[i][0]]
        normal[:] = normalf[i][:]
        
        w_r  = w_ghost[i]
        center_left[:] = centerc[cellidf[i][0]][:]
        
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
        rez_w[cellidf[i][0]]  -= flux_w[0]
        
  
def time_step(u:'float[:]', v:'float[:]', w:'float[:]',  cfl:'float', normal:'float[:,:]', 
              mesure:'float[:]', volume:'float[:]', faceid:'int32[:,:]', dim:'int32'):
   
    nbelement =  len(faceid)
    norm = np.zeros(3)
    dt = 1e6
    for i in range(nbelement):
        lam = 0.
       
        for j in range(faceid[i][-1]):
            norm[:] = normal[faceid[i][j]][:]
            lam_convect = np.fabs(u[i]*norm[0] + v[i]*norm[1] + w[i]*norm[2])
            lam += lam_convect
        
        dt  = min(dt, cfl * volume[i]/lam)
     
    return dt

def update_new_value(ne_c:'float[:]', rez_ne:'float[:]', dissip_ne:'float[:]',  src_ne:'float[:]',
                     dtime:'float', vol:'float[:]'):
    nbelements = len(ne_c)
    for i in range(nbelements):
        ne_c[i]  += dtime  * ((rez_ne[i]  +  dissip_ne[i]) /vol[i] + src_ne[i] )
