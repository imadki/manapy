from numba import njit
from numpy import fabs, pi, cos, sin, exp, sqrt

@njit
def initialisation_SW(h:'float[:]', hu:'float[:]', hv:'float[:]', hc:'float[:]', 
                      Z:'float[:]', center:'float[:,:]', choix:'intc'):
   
    nbelements = len(center)
    
    if choix == 0:
        for i in range(nbelements):
            xcent = center[i][0]
            h[i] = 2
            Z[i]  = 0.
            
            if xcent < .5:
                h[i] = 5.
                
            hu[i] = 0.
            hv[i] = 0.
            hc[i] = 0.

    if choix == 1: 
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]
            
            Z[i] = exp(-((xcent - 0.5)**2 + (ycent - 0.5)**2)/0.01)
            h[i] = 1. - Z[i] 

            hu[i] = 0.
            hv[i] = 0.
            hc[i] = 0.#exp(-((xcent - 0.5)**2 + (ycent - 0.25)**2)/0.01)
        
        
        
@njit        
def initialisation_of_C_in_block(h:'float[:]', h_c:'float[:]', center:'float[:,:]', block:'float'):

    nbelements = len(h)    
    h_c[:] = 0.
    for i in range(nbelements):
        x1,y1=738438,508880
        x2,y2=726685,520052
        x11,y11=720569.5,539155.8
        x22,y22=718971,501842
        a=(y2-y1)/(x2-x1)
        b=y1-a*x1
        a11=(y22-y11)/(x22-x11)
        b11=y11-a11*x11
        xcent = center[i][0]
        ycent = center[i][1]
        #-----------------------block 1------------------------------------
        if  block==1:
            yf=2.1602*xcent-1056579.6436
            if ycent>=yf and ycent>=515979 and (ycent <= (xcent*a+b)) and (ycent <= (xcent*a11+b11)):
                h_c[i] = 1*h[i]
        #-----------------------block 2-----------------------------------    
        elif block==2:
            yf=0.14590609235545207*xcent+408725.147846333
            yf1=2.1602*xcent-1056579.6436
            if (ycent >= yf and ycent <= yf1) and xcent>=727969 and (ycent <= (xcent*a+b)) and (ycent <= (xcent*a11+b11)):
                h_c[i] = 1*h[i]
        #-----------------------block 3----------------------------------
        elif block==3:
            yf1=0.14590609235545207*xcent+408725.147846333
            yf=0.7909399773499434*xcent-64250.77463193657
            yf2=1.760233918128655 *xcent -773269.9064327485
            yf3=-0.651219512195122 *xcent+991484.7658536586

            if ((ycent >= yf and ycent <=yf1) and ( ycent>=yf2 and ycent <= yf3)and (ycent <= (xcent*a+b))) and (ycent <= (xcent*a11+b11)) \
               or ((ycent<=515289 and ycent >=514816) and (xcent<=727726 and ycent >= 727116)):
                h_c[i] = 1*h[i]
        #-----------------------block 4----------------------------------
        elif block==4:
            yf1=0.7909399773499434*xcent-64250.77463193657
            yf=1.782109398609852*xcent-789907.4239951647
            yff=-0.6090225563909775 *xcent + 959793.8195488722
            if ycent >= yf and ycent <= yf1 and ycent <= yff and (ycent <= (xcent*a+b)) and (ycent <= (xcent*a11+b11)):
                h_c[i] = 1*h[i]     
        #-----------------------block 5----------------------------------
        elif block==5:
            yf1=1.782109398609852*xcent-789907.4239951647
            yf=0.7346938775510204*xcent -32127.448979591834
            if ycent >= yf and ycent <= yf1 and (ycent <= (xcent*a+b)) and (ycent <= (xcent*a11+b11)):
                h_c[i] = 1*h[i]     
        #-----------------------block 6----------------------------------
        elif block==6:
            yf1=0.7346938775510204*xcent-32127.448979591834
            yf=2.9539422326307574*xcent-1678071.345042935
            if ycent >= yf and ycent <= yf1 and (ycent <= (xcent*a+b)) and (ycent <= (xcent*a11+b11)):
                h_c[i] = 1*h[i]       
        #-----------------------block 7----------------------------------
        elif block==7:
            yf=2.9539422326307574*xcent-1678071.345042935
            if ycent<=yf and (ycent <= (xcent*a+b)):
                h_c[i] = 1*h[i]
        #-----------------------block 8----------------------------------
        elif block==8:
            yf1= -0.6090225563909775 *xcent + 959793.8195488722
            yf2 = 1.8058727569331159 *xcent -807296.1663947799
            yf3 = 5.424242424242424 *xcent -3457241.7575757573
            yf4 = -0.21140939597315436 *xcent +670264.5503355705
            yf5 = 6.518518518518518 *xcent -4256675.296296296
            yf6 = 1.7223168654173764 *xcent -745534.3407155024
            yf7 = -0.22602739726027396 *xcent +680784.198630137
            #and (ycent <= (xcent*a+b))) and (ycent <= (xcent*a11+b11)
            if ((ycent<=yf6 and ycent >= yf1) and (ycent<=yf7 and ycent >=yf2)) \
               or ((ycent<=yf4 and ycent >= yf7) and (ycent<=yf5 and ycent >=yf3)):
                h_c[i] = 1*h[i]
        elif block==10:
            if ycent <= (xcent*a+b) and (ycent <= (xcent*a11+b11)):
               h_c[i] =1*h[i]
            

#        h_c[i] = h_c[i] * h[i]
############################################################################################

@njit
def initial_Tr(hc:'float[:]', volume_Ti:'float[:]'):

    n = len(hc)
    Q0 = 0.
    
    for i in range(n):
        Q0 += hc[i]*volume_Ti[i]
    return Q0

@njit
def concentration_overtime(hc:'float[:]',center:'float[:,:]', volume_Ti:'float[:]', block:'float'):

    #from numpy import zeros, max
    n = len(hc)
    Q = 0.
    
    for i in range(n):
        x1,y1=738438,508880
        x2,y2=726685,520052
        a=(y2-y1)/(x2-x1)
        b=y1-a*x1
        x11,y11=720569.5,539155.8
        x22,y22=718971,501842
        a11=(y22-y11)/(x22-x11)
        b11=y11-a11*x11
        xcent = center[i][0]
        ycent = center[i][1]
        #-----------------------block 1------------------------------------
        if  block==1:
            yf=2.1602*xcent-1056579.6436
            if ycent>=yf and ycent>=515979 and (ycent <= (xcent*a+b)) and (ycent <= (xcent*a11+b11)):
                Q += volume_Ti[i]*hc[i]
        #-----------------------block 2-----------------------------------    
        elif block==2:
            yf=0.14590609235545207*xcent+408725.147846333
            yf1=2.1602*xcent-1056579.6436
            if (ycent >= yf and ycent <= yf1) and xcent>=727969 and (ycent <= (xcent*a+b)) and (ycent <= (xcent*a11+b11)):
                Q += volume_Ti[i]*hc[i]
        #-----------------------block 3----------------------------------
        elif block==3:
            yf1=0.14590609235545207*xcent+408725.147846333
            yf=0.7909399773499434*xcent-64250.77463193657
            yf2=1.760233918128655 *xcent -773269.9064327485
            yf3=-0.651219512195122 *xcent+991484.7658536586

            if ((ycent >= yf and ycent <=yf1) and ( ycent>=yf2 and ycent <= yf3)and (ycent <= (xcent*a+b))) and (ycent <= (xcent*a11+b11)) \
               or ((ycent<=515289 and ycent >=514816) and (xcent<=727726 and ycent >= 727116)) :
                Q += volume_Ti[i]*hc[i]
        #-----------------------block 4----------------------------------
        elif block==4:
            yf1=0.7909399773499434*xcent-64250.77463193657
            yf=1.782109398609852*xcent-789907.4239951647
            yff=-0.6090225563909775 *xcent + 959793.8195488722
            if ycent >= yf and ycent <= yf1 and ycent <= yff and (ycent <= (xcent*a+b)) and (ycent <= (xcent*a11+b11)):
                Q += volume_Ti[i]*hc[i]     
        #-----------------------block 5----------------------------------
        elif block==5:
            yf1=1.782109398609852*xcent-789907.4239951647
            yf=0.7346938775510204*xcent -32127.448979591834
            if ycent >= yf and ycent <= yf1 and (ycent <= (xcent*a+b)) and (ycent <= (xcent*a11+b11)):
                Q += volume_Ti[i]*hc[i]     
        #-----------------------block 6----------------------------------
        elif block==6:
            yf1=0.7346938775510204*xcent-32127.448979591834
            yf=2.9539422326307574*xcent-1678071.345042935
            if ycent >= yf and ycent <= yf1 and (ycent <= (xcent*a+b)) and (ycent <= (xcent*a11+b11)):
                Q += volume_Ti[i]*hc[i]       
        #-----------------------block 7----------------------------------
        elif block==7:
            yf=2.9539422326307574*xcent-1678071.345042935
            if ycent<=yf and (ycent <= (xcent*a+b)) and (ycent <= (xcent*a11+b11)):
                Q += volume_Ti[i]*hc[i]
        #-----------------------block 8----------------------------------
        elif block==8:
            yf1= -0.6090225563909775 *xcent + 959793.8195488722
            yf2 = 1.8058727569331159 *xcent -807296.1663947799
            yf3 = 5.424242424242424 *xcent -3457241.7575757573
            yf4 = -0.21140939597315436 *xcent +670264.5503355705
            yf5 = 6.518518518518518 *xcent -4256675.296296296
            yf6 = 1.7223168654173764 *xcent -745534.3407155024
            yf7 = -0.22602739726027396 *xcent +680784.198630137
            
            if ((ycent<=yf6 and ycent >= yf1) and (ycent<=yf7 and ycent >=yf2)) \
               or ((ycent<=yf4 and ycent >= yf7) and (ycent<=yf5 and ycent >=yf3)):
                Q += volume_Ti[i]*hc[i]
        elif block==10:
             if ycent <= (xcent*a+b) and (ycent <= (xcent*a11+b11)):
                Q += volume_Ti[i]*hc[i]
               
        
    
    return Q


#@pure
def print_time(d_t:'float', niter:'int', time:'float', rapport:'float', resid:'float', rank:'int'):
    
    from numpy import float32
    
    if rank == 0:    
        print(" **************************** Computing ****************************")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Saving Results $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("Iteration = ", niter, "time = ", float32(time), "time step = ", d_t)
        print("Q/Q0 = ", rapport, "temps de residence", resid)

@njit
def initialisation_SW_bis(h:'float[:]', hu:'float[:]', hv:'float[:]', hc:'float[:]', Z:'float[:]',
                          center:'float[:,:]', choix:'int', Hbath:'float[:]', hmaxGlob:'float'):
   
    
    nbelements = len(center)
    if choix == 200:
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]
            h[i]  = 100#+1*exp(-(xcent**2+ycent**2)/(2**2))#+10*exp(-((xcent-4500)**2+(ycent-4500)**2)/(264**2))
            Z[i]  = 0.
            hu[i] = 0*h[i]
            hv[i] = 0*h[i]
            hc[i] = 1+exp(-(xcent**2+ycent**2)/(2**2))
        
    if choix == 0:
        for i in range(nbelements):
           
            xcent = center[i][0]
            ycent = center[i][1]
            if xcent<=90:
               h[i]=5
            else:
                h[i]=0
            Z[i]  = 0.
            hu[i] = 0.*h[i]
            hv[i] = 0.*h[i]
            hc[i] = 1.*h[i]#0.*h[i]#10*exp(-((xcent-2400)**2+(ycent-2400)**2)/(5**2))#+6.5*exp(-((xcent-2400)**2+(ycent-2400)**2)/(264**2))
            
    elif choix == 1:
        for i in range(nbelements):
            xcent = center[i][0]
            Z[i] = 0
    
            if fabs(xcent - 1500/2) <= 1500/8:
                Z[i] = 8
    
            h[i] = 16 - Z[i]
            hu[i] = 0.
            hv[i] = 0.
            hc[i] = 0.
    
    elif choix == 2:
        c1 = 0.04
        c2 = 0.02
        alpha = pi/6
        x0 = -20
        y0 = -10
        M = .5
        grav = 1.

        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]

            f =  -c2*((xcent - x0)**2 + (ycent - y0)**2)

            h[i]  = 1 - ((c1**2)/(4*grav*c2))*exp(2*f)
            hu[i] = (M * cos(alpha) + c1*(ycent - y0)*exp(f)) * h[i]
            hv[i] = (M * sin(alpha) - c1*(xcent - x0)*exp(f)) * h[i]

            hc[i] = 0
            Z[i] = 0  
            
    elif choix == 3:
        sigma1 = 264
        sigma2 = 264
        c_1 = 10
        c_2 = 6.5
        x_1 = 1400
        y_1 = 1400
        x_2 = 2400
        y_2 = 2400
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]

            h[i] = 1.
            hc[i] = c_1 * exp(-1* ((xcent - x_1)**2 
                + (ycent -y_1)**2)/ sigma1**2) + c_2 * exp(-1* ((xcent - x_2)**2
                + (ycent -y_2)**2)/ sigma2**2)
            hu[i] = .5
            hv[i] = .5
            Z[i] = 0.
            
    elif choix == 4:
        for i in range(nbelements):
            xcent = center[i][0]
            h[i]  = 0.042
            Z[i]  = 0.
            hu[i] = 0.
            hv[i] = 0.
            hc[i] = 0.
            
    elif choix == 5:
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]
            h[i]  = 0.33
            Z[i] = 0.

            if xcent <= 2.39:
                h[i] = 0.53
                Z[i]  = 0.

            hu[i] = 0.
            hv[i] = 0.
            hc[i] = 0.
    elif choix == 6:
        sigma = 5
        
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]
            zcent = center[i][2]    
            
            hc[i]  = 5 * exp(-1.*((xcent-6.)**2 + (ycent-0.)**2 + (zcent-0.)**2 ) / (sigma)**2) + 1.
            h[i] = 1.
            hu[i] = 2.
            hv[i] = 0.
            Z[i] = 0.
            
    elif choix == 7: #h gaussienne (c-propriété)
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]

            Z[i] = 0.8 * exp(-5*(xcent - 1.)**2 - 50* (ycent - 0.5)**2) #
            h[i] = 1 - Z[i]
            hu[i] = 0.
            hv[i] = 0.
            hc[i] = 0.


    elif choix == 8: #Lagune Nador

        h0 = 3.
        hmax  = hmaxGlob
        hpZ   = hmax + 10.
        SLIB2 = hpZ + h0
  
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]
            
            Z[i] = hpZ - Hbath[i]
            
            h[i] = SLIB2 - Z[i] #- 0.22408118

            
            hu[i] = 0.
            hv[i] = 0.
            #hc[i] = 100*exp(-((xcent-732296)**2+(ycent-509801)**2)/(600**2))   # for Nador lagoon
            #hc[i] = 100*exp(-((xcent-500443)**2+(ycent-598855)**2)/(2000**2))    # for geblartar
            hc[i] = 100*exp(-((xcent-460397)**2+(ycent-591263)**2)/(2000**2))    # for geblartar
            #hc[i] = 100*exp(-((xcent-458210)**2+(ycent-603856)**2)/(1000**2))    # for geblartar

    elif choix == 9: #Strait of Gibraltar

        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]
            
            Z[i] =0
            h[i] =10 
            hu[i] = 0.
            hv[i] = 0.
            #hc[i] = 100*exp(-((xcent-458210)**2+(ycent-603856)**2)/(1000**2))    # for geblartar
            hc[i] = 100*exp(-((xcent-500443)**2+(ycent-598855)**2)/(2000**2))    # for geblartar
        

            

    elif choix == 98: #Lagune Nador
        
        h0 = 3.
        hmax  = hmaxGlob
        hpZ   = hmax + 10.
        SLIB2 = hpZ + h0
        
        x0 = 731409
        y0 = 514001
        R = 200.

        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]
            
            dR = sqrt((xcent - x0)**2 + (ycent - y0)**2)
            #if (i == 10589):
            #if dR < 1000:
            #    print(dR, i)
                
            Z[i] = hpZ - Hbath[i]

            h[i] = SLIB2 - Z[i]

            hu[i] = 0.
            hv[i] = 0.
            
            if dR < R:
                hc[i] = 1*h[i]
            else:
                hc[i] = 0.
        

