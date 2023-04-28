from numpy import zeros, fabs, sqrt, int32, uint32, array, arccos, cos, pi

def update_SW(h_c:'float[:]', hu_c:'float[:]', hv_c:'float[:]', hc_c:'float[:]', Z_c:'float[:]', 
              rez_h:'float[:]', rez_hu:'float[:]', rez_hv:'float[:]', rez_hc:'float[:]', rez_Z:'float[:]',
              src_h:'float[:]', src_hu:'float[:]', src_hv:'float[:]', src_hc:'float[:]', src_Z:'float[:]',
              dissip_hc:'float[:]',  corio_hu:'float[:]', corio_hv:'float[:]', wind_hu:'float', wind_hv:'float',
              dtime:'float', vol:'float[:]'):

    for i in range(len(h_c)):
        h_c[i]  += dtime  * (rez_h[i]    + src_h[i] )/vol[i]
        hu_c[i] += dtime  * ((rez_hu[i]  + src_hu[i])/vol[i] + corio_hu[i] + wind_hu)
        hv_c[i] += dtime  * ((rez_hv[i]  + src_hv[i])/vol[i] + corio_hv[i] + wind_hv)
        hc_c[i] += dtime  * (rez_hc[i]   + src_hc[i] - dissip_hc[i])/vol[i]
        Z_c[i]  += dtime  * (rez_Z[i]    + src_Z[i] )/vol[i]
    
def time_step_SW(h_c:'float[:]', hu_c:'float[:]', hv_c:'float[:]', cfl:'float', normal:'float[:,:]', mesure:'float[:]', volume:'float[:]', 
                 faceid:'int32[:,:]', grav:'float', Dxx:'float', Dyy:'float'):
   
    #from numpy import sqrt, fabs, zeros
    nbelement =  len(faceid)
    u_n = 0.
    norm = zeros(3)
    dt = 1e6
  
    for i in range(nbelement):
        velson = sqrt(grav*h_c[i])
        lam = 0.
        for j in range(faceid[i][-1]):
            norm[:] = normal[faceid[i][j]][:]
            
            #convective part
            u_n = fabs(hu_c[i]/h_c[i]*norm[0] + hv_c[i]/h_c[i]*norm[1])
            lam_convect = u_n/mesure[faceid[i][j]] + velson
            lam += lam_convect * mesure[faceid[i][j]]
      
            #diffusion part
            mes = sqrt(norm[0]*norm[0] + norm[1]*norm[1])
            lam_diff = Dxx * mes**2 + Dyy * mes**2
            lam += lam_diff/volume[i]
        dt  = min(dt, cfl * volume[i]/lam)
        
    return dt

def term_source_srnh_SW(src_h:'float[:]', src_hu:'float[:]', src_hv:'float[:]', src_hc:'float[:]', src_Z:'float[:]', 
                        h_c:'float[:]', hu_c:'float[:]', hv_c:'float[:]', hc_c:'float[:]', Z_c:'float[:]', 
                        h_ghost:'float[:]', hu_ghost:'float[:]', hv_ghost:'float[:]', hc_ghost:'float[:]', Z_ghost:'float[:]',
                        h_halo:'float[:]', hu_halo:'float[:]', hv_halo:'float[:]', hc_halo:'float[:]', Z_halo:'float[:]',
                        h_x:'float[:]', h_y:'float[:]', psi:'float[:]',
                        hx_halo:'float[:]', hy_halo:'float[:]', psi_halo:'float[:]', 
                        nodeidc:'int32[:,:]', faceidc:'int32[:,:]', cellidc:'int32[:,:]',  cellidf:'int32[:,:]',
                        centerc:'float[:,:]', normalc:'float[:,:,:]', 
                        namef:'int32[:]', centerf:'float[:,:]', centerh:'float[:,:]',
                        vertexn:'float[:,:]', halofid:'int32[:]', grav:'float', order:'int'):
    
    nbelement = len(h_c)
    hi_p =  zeros(3)
    zi_p =  zeros(3)

    zv =  zeros(3)
    
    mata =  zeros(3)
    matb =  zeros(3)
    
    ns  = zeros((3, 3))
    ss  = zeros((3, 3))
    s_1 = zeros(3)
    s_2 = zeros(3)
    s_3 = zeros(3)
    b   = zeros(3)
    G   = zeros(3)

    for i in range(nbelement):
        
        G[:] = centerc[i]
        c_1 = 0.
        c_2 = 0.

        for j in range(3):
            f = faceidc[i][j]
            ss[j] = normalc[i][j]
            
            if namef[f] == 10 :
                
                h_1p = h_c[i]
                z_1p = Z_c[i]
                
                h_p1 = h_halo[halofid[f]]
                z_p1 = Z_halo[halofid[f]]
                
            elif namef[f] == 0:
                
                h_1p = h_c[i]
                z_1p = Z_c[i]
                
                h_p1 = h_c[cellidc[i][j]]
                z_p1 = Z_c[cellidc[i][j]]
                
            else:
                h_1p = h_c[i]
                z_1p = Z_c[i]
                
                h_p1 = h_ghost[f]
                z_p1 = Z_ghost[f]
                
            zv[j] = z_p1
            mata[j] = h_p1*ss[j][0]
            matb[j] = h_p1*ss[j][1]

            
            c_1 = c_1 + (0.5*(h_1p + h_p1)*0.5*(h_1p + h_p1))  *ss[j][0]
            c_2 = c_2 + (0.5*(h_1p + h_p1)*0.5*(h_1p + h_p1))  *ss[j][1]
            
            hi_p[j] = h_1p
            zi_p[j] = z_1p
            

        c_3 = 3.0 * h_1p
            
        delta = (mata[1]*matb[2]-mata[2]*matb[1]) - (mata[0]*matb[2]-matb[0]*mata[2]) + (mata[0]*matb[1]-matb[0]*mata[1])

        deltax = c_3*(mata[1]*matb[2]-mata[2]*matb[1]) - (c_1*matb[2]-c_2*mata[2]) + (c_1*matb[1]-c_2*mata[1])

        deltay = (c_1*matb[2]-c_2*mata[2]) - c_3*(mata[0]*matb[2]-matb[0]*mata[2]) + (mata[0]*c_2-matb[0]*c_1)

        deltaz = (mata[1]*c_2-matb[1]*c_1) - (mata[0]*c_2-matb[0]*c_1) + c_3*(mata[0]*matb[1]-matb[0]*mata[1])
        
        h_1 = deltax/delta
        h_2 = deltay/delta
        h_3 = deltaz/delta
            
        z_1 = zi_p[0] + hi_p[0] - h_1
        z_2 = zi_p[1] + hi_p[1] - h_2
        z_3 = zi_p[2] + hi_p[2] - h_3

        b[:] =  vertexn[nodeidc[i][1]][0:3]

        ns[0] = array([(G[1]-b[1]), -(G[0]-b[0]), 0.])
        ns[1] = ns[0] - ss[1]  #  N23                                                                                                                                                                       
        ns[2] = ns[0] + ss[0]  #  N31    
        
        s_1 = 0.5*h_1*(zv[0]*ss[0] + z_2*ns[0] + z_3*(-1)*ns[2])
        s_2 = 0.5*h_2*(zv[1]*ss[1] + z_1*(-1)*ns[0] + z_3*ns[1])
        s_3 = 0.5*h_3*(zv[2]*ss[2] + z_1*ns[2] + z_2*(-1)*ns[1])
        
        #TODO
        src_h[i]   = 0
        src_hu[i]  = -grav * (s_1[0] + s_2[0] + s_3[0])
        src_hv[i]  = -grav * (s_1[1] + s_2[1] + s_3[1]) 
        src_hc[i]  = 0.
        src_Z[i]   = 0.
 
from numba import njit

@njit
def srnh_scheme(hu_l:'float', hu_r:'float', hv_l:'float', hv_r:'float', h_l:'float', h_r:'float', hc_l:'float', 
                hc_r:'float', Z_l:'float', Z_r:'float', normal:'float[:]', mesure:'float', grav:'float', flux:'float[:]'):
    
    #from numpy import zeros, sqrt, arccos, cos, fabs, pi
    
    ninv =  zeros(2)
    w_dif =  zeros(5)
    rmat =  zeros((5, 5))
    
    As = 0
    p  = 0.4
    xi = 1/(1-p)
    ninv = zeros(2)

    ninv[0] = -1*normal[1]
    ninv[1] = normal[0]

    u_h = (hu_l / h_l * sqrt(h_l)
       + hu_r / h_r * sqrt(h_r)) /(sqrt(h_l) + sqrt(h_r))

    v_h = (hv_l / h_l * sqrt(h_l)
           + hv_r / h_r * sqrt(h_r)) /(sqrt(h_l) + sqrt(h_r))

    c_h = (hc_l / h_l * sqrt(h_l)
           + hc_r / h_r * sqrt(h_r)) /(sqrt(h_l) + sqrt(h_r))
    
    #uvh =  array([uh, vh])
    un_h = u_h*normal[0] + v_h*normal[1]
    un_h = un_h / mesure
    vn_h = u_h*ninv[0] + v_h*ninv[1]
    vn_h = vn_h / mesure

    hroe = (h_l+h_r)/2
    uroe = un_h
    vroe = vn_h
    croe = c_h
    
    uleft = hu_l*normal[0] + hv_l*normal[1]
    uleft = uleft / mesure
    vleft = hu_l*ninv[0] + hv_l*ninv[1]
    vleft = vleft / mesure

    uright = hu_r*normal[0] + hv_r*normal[1]
    uright = uright / mesure
    vright = hu_r*ninv[0] + hv_r*ninv[1]
    vright = vright / mesure

    w_lrh = (h_l  + h_r)/2
    w_lrhu = (uleft + uright)/2
    w_lrhv = (vleft + vright)/2
    w_lrhc = (hc_l  + hc_r)/2
    w_lrz = (Z_l + Z_r)/2
    
    w_dif[0] = h_r - h_l
    w_dif[1] = uright - uleft
    w_dif[2] = vright - vleft
    w_dif[3] = hc_r - hc_l
    w_dif[4] = Z_r - Z_l
    
    d=As*xi*(3*uroe**2 + vroe**2)
    sound =  sqrt(grav * hroe)
    Q=-(uroe**2 + 3*grav*(hroe+d))/9
    R=uroe*(9*grav*(2*hroe-d) - 2*uroe**2)/54
    theta=arccos(R/( sqrt(-Q**3)))
    
     # Les valeurs propres
    lambda1 = 2* sqrt(-Q)* cos(theta/3) + (2/3)*uroe
    lambda2 = 2* sqrt(-Q)* cos((theta   + 2* pi)/3) + (2/3)*uroe
    lambda3 = 2* sqrt(-Q)* cos((theta   + 4* pi)/3) + (2/3)*uroe
    lambda4 = uroe
    lambda5 = uroe
         
    # définition de alpha
    alpha1 = lambda1 - uroe
    alpha2 = lambda2 - uroe
    alpha3 = lambda3 - uroe
     
    # définition de beta
    beta= 2*As*xi*vroe/hroe
    
    # définition de gamma
    gamma1 = sound**2 - uroe**2 + lambda2*lambda3 - beta*alpha2*alpha3*vroe
    gamma2 = sound**2 - uroe**2 + lambda1*lambda3 - beta*alpha1*alpha3*vroe
    gamma3 = sound**2 - uroe**2 + lambda1*lambda2 - beta*alpha1*alpha2*vroe
 
    # définition de sigma
    sigma1 = -alpha1*alpha2 + alpha2*alpha3 - alpha1*alpha3 + alpha1**2          
    sigma2 =  alpha1*alpha2 + alpha2*alpha3 - alpha1*alpha3 - alpha2**2 
    sigma3 =  alpha1*alpha2 - alpha2*alpha3 - alpha1*alpha3 + alpha3**2 #ici 


    epsilon = 1e-10

    if fabs(lambda1) < epsilon:
        sign1 = 0.
    else:
        sign1 = lambda1 /  fabs(lambda1)
        
    if  fabs(lambda2) < epsilon:
        sign2 = 0.
    else:
        sign2 = lambda2 /  fabs(lambda2)
    
    if   fabs(lambda3) < epsilon:
        sign3 = 0.
    else:
        sign3 = lambda3 /  fabs(lambda3)
    
    if  fabs(lambda4) < epsilon:
        sign4 = 0.
    else:
        sign4 = lambda4 /  fabs(lambda4)
    
    if  fabs(lambda5) < epsilon:
        sign5 = 0.
    else:
        sign5 = lambda5 /  fabs(lambda5)

    # 1ère colonne
    rmat[0][0] = sign1*(gamma1/sigma1) - sign2*(gamma2/sigma2) + sign3*(gamma3/sigma3) + sign5*(beta*vroe)
    rmat[1][0] = lambda1*sign1*(gamma1/sigma1) - lambda2*sign2*(gamma2/sigma2) + lambda3*sign3*(gamma3/sigma3) + sign5*(beta*uroe*vroe)
    rmat[2][0] = vroe*sign1*(gamma1/sigma1) - vroe*sign2*(gamma2/sigma2) + vroe*sign3*(gamma3/sigma3) - vroe*sign5*(1-beta*vroe)
    rmat[3][0] = croe*sign1*(gamma1/sigma1) - croe*sign2*(gamma2/sigma2) + croe*sign3*(gamma3/sigma3) - croe*sign4 + croe*sign5*beta*vroe
    rmat[4][0] = (alpha1**2/sound**2 - 1)*sign1*(gamma1/sigma1) - (alpha2**2/sound**2 - 1)*sign2*(gamma2/sigma2) + (alpha3**2/sound**2 - 1)*sign3*(gamma3/sigma3) - sign5*(beta*vroe)
    
    # 2ème colonne
    rmat[0][1] = - sign1*(alpha2 + alpha3)/sigma1 + sign2*(alpha1 + alpha3)/sigma2 - sign3*(alpha1 + alpha2)/sigma3
    rmat[1][1] = - lambda1*sign1*(alpha2 + alpha3)/sigma1 + lambda2*sign2*(alpha1 + alpha3)/sigma2 - lambda3*sign3*(alpha1 + alpha2)/sigma3
    rmat[2][1] = - vroe*sign1*(alpha2 + alpha3)/sigma1 + vroe*sign2*(alpha1 + alpha3)/sigma2 - vroe*sign3*(alpha1 + alpha2)/sigma3
    rmat[3][1] = - croe*sign1*(alpha2 + alpha3)/sigma1 + croe*sign2*(alpha1 + alpha3)/sigma2 - croe*sign3*(alpha1 + alpha2)/sigma3
    rmat[4][1] = - (alpha1**2/sound**2 - 1)*sign1*(alpha2 + alpha3)/sigma1 + (alpha2**2/sound**2 - 1)*sign2*(alpha1 + alpha3)/sigma2 - (alpha3**2/sound**2 - 1)*sign3*(alpha1 + alpha2)/sigma3

    # 3ème colonne 
    rmat[0][2] = sign1*beta*alpha2*alpha3/sigma1 - sign2*beta*alpha1*alpha3/sigma2 + sign3*beta*alpha1*alpha2/sigma3 - sign5*beta
    rmat[1][2] = lambda1*sign1*beta*alpha2*alpha3/sigma1 - lambda2*sign2*beta*alpha1*alpha3/sigma2 + lambda3*sign3*beta*alpha1*alpha2/sigma3 - sign5*beta*uroe
    rmat[2][2] = vroe*sign1*beta*alpha2*alpha3/sigma1 - vroe*sign2*beta*alpha1*alpha3/sigma2 + vroe*sign3*beta*alpha1*alpha2/sigma3 + sign5*(1-beta*vroe)
    rmat[3][2] = croe*sign1*beta*alpha2*alpha3/sigma1 - croe*sign2*beta*alpha1*alpha3/sigma2 + croe*sign3*beta*alpha1*alpha2/sigma3 - croe*sign5*beta
    rmat[4][2] = (alpha1**2/sound**2 - 1)*sign1*beta*alpha2*alpha3/sigma1 - (alpha2**2/sound**2 - 1)*sign2*beta*alpha1*alpha3/sigma2 + (alpha3**2/sound**2 - 1)*sign3*beta*alpha1*alpha2/sigma3 + sign5*beta

    # 4ème colonne  
    rmat[0][3] = 0.
    rmat[1][3] = 0.
    rmat[2][3] = 0.
    rmat[3][3] = sign4
    rmat[4][3] = 0.

    # 5ème colone
    rmat[0][4] = sign1*sound**2/sigma1 - sign2*sound**2/sigma2 + sign3*sound**2/sigma3
    rmat[1][4] = lambda1*sign1*sound**2/sigma1 - lambda2*sign2*sound**2/sigma2 + lambda3*sign3*sound**2/sigma3
    rmat[2][4] = vroe*sign1*sound**2/sigma1 - vroe*sign2*sound**2/sigma2 + vroe*sign3*sound**2/sigma3
    rmat[3][4] = croe*sign1*sound**2/sigma1 - croe*sign2*sound**2/sigma2 + croe*sign3*sound**2/sigma3
    rmat[4][4] = (alpha1**2/sound**2 - 1)*sign1*sound**2/sigma1 - (alpha2**2/sound**2 - 1)*sign2*sound**2/sigma2 + (alpha3**2/sound**2 - 1)*sign3*sound**2/sigma3

    
    hnew = sum(rmat[0][:]*w_dif[:])
    unew = sum(rmat[1][:]*w_dif[:])
    vnew = sum(rmat[2][:]*w_dif[:])
    cnew = sum(rmat[3][:]*w_dif[:])
    znew = sum(rmat[4][:]*w_dif[:])
    
    u_h  = hnew/2
    u_hu = unew/2
    u_hv = vnew/2
    u_hc = cnew/2
    u_z  = znew/2

    w_lrh  = w_lrh  - u_h
    w_lrhu = w_lrhu - u_hu
    w_lrhv = w_lrhv - u_hv
    w_lrhc = w_lrhc - u_hc
    w_lrz  = w_lrz  - u_z
    
    unew = 0.
    vnew = 0.

    unew = w_lrhu * normal[0] + w_lrhv * -1*normal[1]
    unew = unew / mesure
    vnew = w_lrhu * -1*ninv[0] + w_lrhv * ninv[1]
    vnew = vnew / mesure

    w_lrhu = unew
    w_lrhv = vnew

    q_s = normal[0] * unew + normal[1] * vnew

    flux[0]  = q_s
    flux[1]  = q_s * w_lrhu/w_lrh + 0.5 * grav * w_lrh * w_lrh * normal[0]
    flux[2]  = q_s * w_lrhv/w_lrh + 0.5 * grav * w_lrh * w_lrh * normal[1]
    flux[3]  = q_s * w_lrhc/w_lrh
    flux[4]  = As * xi * normal[0] * unew * (unew**2 +vnew**2) / w_lrh**3 + As * xi * normal[1] * vnew * (unew**2 +vnew**2) / w_lrh**3
        
def explicitscheme_convective_SW(rez_h:'float[:]', rez_hu:'float[:]', rez_hv:'float[:]', rez_hc:'float[:]', rez_Z:'float[:]', 
                                 h_c:'float[:]', hu_c:'float[:]', hv_c:'float[:]', hc_c:'float[:]', Z_c:'float[:]', 
                                 h_ghost:'float[:]', hu_ghost:'float[:]', hv_ghost:'float[:]', hc_ghost:'float[:]', Z_ghost:'float[:]',
                                 h_halo:'float[:]', hu_halo:'float[:]', hv_halo:'float[:]', hc_halo:'float[:]', Z_halo:'float[:]',
                                 h_x:'float[:]', h_y:'float[:]', hx_halo:'float[:]', hy_halo:'float[:]',
                                 hc_x:'float[:]', hc_y:'float[:]', hcx_halo:'float[:]', hcy_halo:'float[:]',
                                 psi:'float[:]', psi_halo:'float[:]', 
                                 centerc:'float[:,:]', centerf:'float[:,:]', centerh:'float[:,:]', centerg:'float[:,:]',
                                 cellidf:'int32[:,:]', mesuref:'float[:]', normalf:'float[:,:]', halofid:'int32[:]',
                                 innerfaces:'int32[:]', halofaces:'int32[:]', boundaryfaces:'int32[:]', grav:'float', order:'int'):
    
    rez_h[:] = 0.; rez_hu[:] = 0.; rez_hv[:] = 0.; rez_hc[:] = 0.; rez_Z[:] = 0.
    
    #from numpy import zeros
    
    flux = zeros(5)
    r_l = zeros(2)
    r_r = zeros(2)
    
    for i in innerfaces:
       
        h_l  = h_c[cellidf[i][0]]
        hu_l = hu_c[cellidf[i][0]]
        hv_l = hv_c[cellidf[i][0]]
        hc_l = hc_c[cellidf[i][0]]
        Z_l  = Z_c[cellidf[i][0]]
        
        normal = normalf[i]
        mesure = mesuref[i]
        
        h_r  = h_c[cellidf[i][1]]
        hu_r = hu_c[cellidf[i][1]]
        hv_r = hv_c[cellidf[i][1]]
        hc_r = hc_c[cellidf[i][1]]
        Z_r  = Z_c[cellidf[i][1]]
        
        center_left = centerc[cellidf[i][0]]
        center_right = centerc[cellidf[i][1]]

        h_x_left = h_x[cellidf[i][0]];   h_x_right = h_x[cellidf[i][1]]
        h_y_left = h_y[cellidf[i][0]];   h_y_right = h_y[cellidf[i][1]]
        hc_x_left = hc_x[cellidf[i][0]]; hc_x_right = hc_x[cellidf[i][1]]
        hc_y_left = hc_y[cellidf[i][0]]; hc_y_right = hc_y[cellidf[i][1]]
        
        psi_left = psi[cellidf[i][0]]; psi_right = psi[cellidf[i][1]]
        
        r_l[0] = centerf[i][0] - center_left[0]; r_r[0] = centerf[i][0] - center_right[0]; 
        r_l[1] = centerf[i][1] - center_left[1]; r_r[1] = centerf[i][1] - center_right[1]; 
        
        h_l  = h_l  + (order - 1) * psi_left  * (h_x_left * r_l[0]  + h_y_left * r_l[1] )
        h_r  = h_r  + (order - 1) * psi_right * (h_x_right* r_r[0]  + h_y_right* r_r[1] )
        
        hc_l  = hc_l  + (order - 1) * psi_left  * (hc_x_left * r_l[0]  + hc_y_left * r_l[1] )
        hc_r  = hc_r  + (order - 1) * psi_right * (hc_x_right* r_r[0]  + hc_y_right* r_r[1] )
  
        srnh_scheme(hu_l, hu_r, hv_l, hv_r, h_l, h_r, hc_l, hc_r, Z_l, Z_r, normal, mesure, grav, flux)
            
        rez_h[cellidf[i][0]]  -= flux[0]
        rez_hu[cellidf[i][0]] -= flux[1]
        rez_hv[cellidf[i][0]] -= flux[2]
        rez_hc[cellidf[i][0]] -= flux[3]
        rez_Z[cellidf[i][0]]  -= flux[4]
          
        rez_h[cellidf[i][1]]  += flux[0]
        rez_hu[cellidf[i][1]] += flux[1]
        rez_hv[cellidf[i][1]] += flux[2]
        rez_hc[cellidf[i][1]] += flux[3]
        rez_Z[cellidf[i][1]]  += flux[4]

    for i in halofaces:
        
        h_l  = h_c[cellidf[i][0]]
        hu_l = hu_c[cellidf[i][0]]
        hv_l = hv_c[cellidf[i][0]]
        hc_l = hc_c[cellidf[i][0]]
        Z_l  = Z_c[cellidf[i][0]]
        
        normal = normalf[i]
        mesure = mesuref[i]
        h_r  = h_halo[halofid[i]]
        hu_r = hu_halo[halofid[i]]
        hv_r = hv_halo[halofid[i]]
        hc_r = hc_halo[halofid[i]]
        Z_r  = Z_halo[halofid[i]]
        
        
        center_left = centerc[cellidf[i][0]]
        center_right = centerh[halofid[i]]

        h_x_left = h_x[cellidf[i][0]];   h_x_right = hx_halo[halofid[i]]
        h_y_left = h_y[cellidf[i][0]];   h_y_right = hy_halo[halofid[i]]
        hc_x_left = hc_x[cellidf[i][0]]; hc_x_right = hcx_halo[halofid[i]]
        hc_y_left = hc_y[cellidf[i][0]]; hc_y_right = hcy_halo[halofid[i]]
       
        psi_left = psi[cellidf[i][0]]; psi_right = psi_halo[halofid[i]]
        
        r_l[0] = centerf[i][0] - center_left[0]; r_r[0] = centerf[i][0] - center_right[0]; 
        r_l[1] = centerf[i][1] - center_left[1]; r_r[1] = centerf[i][1] - center_right[1]; 
        
        h_l  = h_l  + (order - 1) * psi_left  * (h_x_left   * r_l[0] + h_y_left   * r_l[1] )
        h_r  = h_r  + (order - 1) * psi_right * (h_x_right  * r_r[0] + h_y_right  * r_r[1] )
        
        hc_l  = hc_l  + (order - 1) * psi_left  * (hc_x_left * r_l[0]  + hc_y_left * r_l[1] )
        hc_r  = hc_r  + (order - 1) * psi_right * (hc_x_right* r_r[0]  + hc_y_right* r_r[1] )
        
        srnh_scheme(hu_l, hu_r, hv_l, hv_r, h_l, h_r, hc_l, hc_r, Z_l, Z_r, normal, mesure, grav, flux)
            
        rez_h[cellidf[i][0]]  -= flux[0]
        rez_hu[cellidf[i][0]] -= flux[1]
        rez_hv[cellidf[i][0]] -= flux[2]
        rez_hc[cellidf[i][0]] -= flux[3]
        rez_Z[cellidf[i][0]]  -= flux[4]
    
    for i in boundaryfaces:
       
        h_l  = h_c[cellidf[i][0]]
        hu_l = hu_c[cellidf[i][0]]
        hv_l = hv_c[cellidf[i][0]]
        hc_l = hc_c[cellidf[i][0]]
        Z_l  = Z_c[cellidf[i][0]]
        
        normal = normalf[i]
        mesure = mesuref[i]
        h_r  = h_ghost[i]
        hu_r = hu_ghost[i]
        hv_r = hv_ghost[i]
        hc_r = hc_ghost[i]
        Z_r  = Z_ghost[i]
        
        center_left = centerc[cellidf[i][0]]
        center_right = centerg[i]

        h_x_left = h_x[cellidf[i][0]];   h_y_left = h_y[cellidf[i][0]]; 
        hc_x_left = hc_x[cellidf[i][0]]; hc_y_left = hc_y[cellidf[i][0]]; 
       
        psi_left = psi[cellidf[i][0]]; 
        
        r_l[0] = centerf[i][0] - center_left[0]; r_r[0] = centerf[i][0] - center_right[0]; 
        r_l[1] = centerf[i][1] - center_left[1]; r_r[1] = centerf[i][1] - center_right[1]; 
        
        h_l  = h_l  + (order - 1) * psi_left  * (h_x_left * r_l[0] + h_y_left * r_l[1] )
        h_r  = h_r 
            
        hc_l  = hc_l  + (order - 1) * psi_left  * (hc_x_left * r_l[0] + hc_y_left * r_l[1] )
        hc_r  = hc_r 
        
        srnh_scheme(hu_l, hu_r, hv_l, hv_r, h_l, h_r, hc_l, hc_r, Z_l, Z_r, normal, mesure, grav, flux)
            
        rez_h[cellidf[i][0]]  -= flux[0]
        rez_hu[cellidf[i][0]] -= flux[1]
        rez_hv[cellidf[i][0]] -= flux[2]
        rez_hc[cellidf[i][0]] -= flux[3]
        rez_Z[cellidf[i][0]]  -= flux[4]
          
def term_coriolis_SW(hu_c:'float[:]', hv_c:'float[:]', corio_hu:'float[:]', corio_hv:'float[:]', f_c:'float'):
    
    for i in range(len(hu_c)):
        corio_hu[i] =  f_c * hu_c[i]
        corio_hv[i] = -f_c * hv_c[i]
    
def term_friction_SW(h_c:'float[:]', hu_c:'float[:]', hv_c:'float[:]', grav:'float', eta:'float', time:'float'):
    
    nbelement = len(h_c)

    for i in range(nbelement):
        ufric = hu_c[i]/h_c[i]
        vfric = hv_c[i]/h_c[i]
        hfric = h_c[i]
        
        A = 1 + grav * time * eta**2 * sqrt(ufric**2 + vfric**2)/(hfric**(4/3))
        
        hutild = hu_c[i]/A
        hvtild = hv_c[i]/A
        
        hu_c[i] = hutild
        hv_c[i] = hvtild

def term_wind_SW(uwind:'float', vwind:'float'):

    RHO_air = 1.28
    WNORME = sqrt(uwind**2 + vwind**2)
    CW = RHO_air*(0.75 + 0.067*WNORME)*1e-3    

    TAUXWX = CW * uwind * WNORME
    TAUXWY = CW * vwind * WNORME

    return TAUXWX, TAUXWY

