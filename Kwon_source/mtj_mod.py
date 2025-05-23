import numpy as np

def mtj_mod(init,phi_init,t_step,v_pulse,t_pulse,t_relax,Happl,Hshe,J_stt,J_she,vhold,xi):
    bitstr = []

    uB = 9.274e-24
    h_bar = 1.054e-34          
    u0 = np.pi*4e-7               
    e = 1.6e-19                
    kb = 1.38e-23;              
    gamma = 2*u0*uB/h_bar
    print("gamma = ", gamma)
    gamma_b = gamma/u0

    # MTJ Parameters
    a = 50e-9
    b = 50e-9
    tf = 1.1e-9                          
    tox = 1.5e-9
    P = 0.6
    alpha = 0.03
    alphar = 10e-30 # Rashba spin coupling 
    # alphar = 0 # Rashba spin coupling 
    T = 300
    Ki = 1.0056364e-3
    Ms = 1.2e6
    Bsat = Ms*u0
    ksi = 75e-15
    gammap = gamma/(1+alpha*alpha)
    v = tf*np.pi*b*a/4
    Vh = 0.5
    delta = 40
    TMR = 1.2
    Rp = 5e3
    A = a*b*np.pi/4
    eta = 0.3
    w = 100e-9                    
    l = 100e-9
    d = 3e-9
    A2 = d*w
    rho = 200e-8
    R2 = rho*l/(w*d)
    
    Hx = 0;        
    Hy = Hshe;                     
    Hz = Happl   
    Htherm = np.sqrt((2*u0*alpha*kb*T)/(Bsat*gamma_b*t_step*v))/u0;  

    # DL / FL ratio
    # xi = (2*e*tf*P*alphar) / (h_bar*uB*abs(eta))
    # xi = -0.001 # joos
    
    # Spin torque factor;
    F = (gamma*h_bar)/(2*u0*e*tf*Ms)     
    
    # Demagnetization field; This is experimental value
    Nx = 0.010613177892974
    Ny = 0.010613177892974
    Nz = 0.978773644214052

    #------------------------Initialization----------------------------#

    phi = []
    theta = []
    energy = []
    power = []
    R = []
    phi.append(phi_init)                              
    theta.append(init)  
    energy.append(0)
    power.append(0)
    R.append(Rp)             # TMR from Parallel state (Start from Rp=60)
    for i in range(int(t_pulse/t_step)):
        V = v_pulse
        J_SHE = 0
        J_STT = J_stt
        Hk = (2*Ki)/(tf*Ms*u0)-(2*ksi*V)/(u0*Ms*tox*tf);
        Ax = Hx-Nx*Ms*np.sin(theta[-1])*np.cos(phi[-1])+np.random.normal()*Htherm
        Ay = Hy-Ny*Ms*np.sin(theta[-1])*np.sin(phi[-1])+np.random.normal()*Htherm
        Az = Hz-Nz*Ms*np.cos(theta[-1])+Hk*np.cos(theta[-1])+np.random.normal()*Htherm

        # LLG
        dtheta = gammap*(
            Ax*(alpha*np.cos(theta[-1])*np.cos(phi[-1])-np.sin(phi[-1]))
            + Ay*(alpha*np.cos(theta[-1])*np.sin(phi[-1])+np.cos(phi[-1]))
            - Az*alpha*np.sin(theta[-1]))                                                                                \
            - J_SHE*F*eta*(np.cos(phi[-1])*np.cos(theta[-1])+(alpha*np.sin(phi[-1]))/(1+alpha*alpha))                    \
            + xi * J_SHE*F*eta*(alpha*np.cos(phi[-1])*np.cos(theta[-1])-(np.sin(phi[-1]))/(1+alpha*alpha))                    \
            + ((F*P*J_STT)*np.sin(theta[-1])/(1+alpha*alpha))  
        
        dphi = gammap*(
            Ax*(-np.cos(theta[-1])*np.cos(phi[-1])-alpha*np.sin(phi[-1]))
            + Ay*(-np.cos(theta[-1])*np.sin(phi[-1])+alpha*np.cos(phi[-1]))
            + Az*np.sin(theta[-1]))/(np.sin(theta[-1]))                                                                         \
            + J_SHE*F*eta*(np.sin(phi[-1])-alpha*np.cos(phi[-1])*np.cos(theta[-1]))/(np.sin(theta[-1])*(1+alpha*alpha))         \
            - xi * J_SHE*F*eta*(alpha*np.sin(phi[-1])-np.cos(phi[-1])*np.cos(theta[-1]))/(np.sin(theta[-1])*(1+alpha*alpha))    \
            - ((alpha*F*P*J_STT)/(1+alpha*alpha))
           
        R1 = Rp*(1+(V/Vh)**2+TMR)/(1+(V/Vh)**2+TMR*(1+(np.sin(theta[-1])*np.cos(phi[-1]) ))/2)
        power.append(V**2/R1+R2*(np.abs(J_SHE*A2))**2+R1*(J_STT*A)**2)
        phi.append(phi[-1]+t_step*dphi)                                     
        theta.append(theta[-1]+t_step*dtheta)
        energy.append(energy[-1]+t_step*power[-1])
        R.append(R1)     # MTJ Resistance 
    for i in range(int(t_relax/t_step)):
        V = vhold
        J_SHE = J_she
        J_STT = -J_stt
        Hk = (2*Ki)/(tf*Ms*u0)-(2*ksi*V)/(u0*Ms*tox*tf);  # effective anisotropy field with VCMA effect
        Ax = Hx-Nx*Ms*np.sin(theta[-1])*np.cos(phi[-1])+np.random.normal()*Htherm
        Ay = Hy-Ny*Ms*np.sin(theta[-1])*np.sin(phi[-1])+np.random.normal()*Htherm
        Az = Hz-Nz*Ms*np.cos(theta[-1])+Hk*np.cos(theta[-1])+np.random.normal()*Htherm

        # LLG
        dtheta = gammap*(
            Ax*(alpha*np.cos(theta[-1])*np.cos(phi[-1])-np.sin(phi[-1]))
            + Ay*(alpha*np.cos(theta[-1])*np.sin(phi[-1])+np.cos(phi[-1]))
            - Az*alpha*np.sin(theta[-1]))                                                                                \
            - J_SHE*F*eta*(np.cos(phi[-1])*np.cos(theta[-1])+(alpha*np.sin(phi[-1]))/(1+alpha*alpha))                    \
            + xi * J_SHE*F*eta*(alpha*np.cos(phi[-1])*np.cos(theta[-1])-(np.sin(phi[-1]))/(1+alpha*alpha))                    \
            + ((F*P*J_STT)*np.sin(theta[-1])/(1+alpha*alpha))  
        
        dphi = gammap*(
            Ax*(-np.cos(theta[-1])*np.cos(phi[-1])-alpha*np.sin(phi[-1]))
            + Ay*(-np.cos(theta[-1])*np.sin(phi[-1])+alpha*np.cos(phi[-1]))
            + Az*np.sin(theta[-1]))/(np.sin(theta[-1]))                                                                         \
            + J_SHE*F*eta*(np.sin(phi[-1])-alpha*np.cos(phi[-1])*np.cos(theta[-1]))/(np.sin(theta[-1])*(1+alpha*alpha))         \
            - xi * J_SHE*F*eta*(alpha*np.sin(phi[-1])-np.cos(phi[-1])*np.cos(theta[-1]))/(np.sin(theta[-1])*(1+alpha*alpha))    \
            - ((alpha*F*P*J_STT)/(1+alpha*alpha))
          
        R1 = Rp*(1+(V/Vh)**2+TMR)/(1+(V/Vh)**2+TMR*(1+(np.sin(theta[-1])*np.cos(phi[-1]) ))/2)
        power.append(V**2/R1+R2*(np.abs(J_SHE*A2))**2+R1*(J_STT*A)**2)
        phi.append(phi[-1]+t_step*dphi)                                     
        theta.append(theta[-1]+t_step*dtheta)
        energy.append(energy[-1]+t_step*power[-1])
        R.append(R1)
    bitstr.append(1 if np.cos(theta[-1]) > 0 else -1)
    
    mx = np.sin(theta)*np.cos(phi)
    my = np.sin(theta)*np.sin(phi)
    mz = np.cos(theta)
    G = 1/np.array(R)
    t = np.arange(0,len(mz)*t_step,t_step)
    return theta[-1],phi[-1],np.array(t),np.array(R),G,np.array(mx),np.array(my),np.array(mz),bitstr[0],np.array(power),energy[-1]