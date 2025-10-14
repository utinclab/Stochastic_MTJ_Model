import numpy as np
from numba import njit
import os
import json

@njit
def Ms_T_dependence(T, beta, Tc, M0):
    """
    Calculate the saturation magnetization (Ms) as a function of temperature (T).
    
    Parameters:
    T (float): Temperature in Kelvin.
    beta (float): Critical exponent for the saturation magnetization.
    Tc (float): Curie temperature in Kelvin.
    
    Returns:
    float: Saturation magnetization at temperature T.
    """
    return M0 * (1 - T/Tc) ** beta

@njit
def mtj_mod(init,phi_init,t_step,v_pulse,t_pulse,t_relax,Happl,Hshe,J_stt,J_she,vhold, temperature, figure_path, flips=1, gamma_a=5.35e6):

    # Physical constants
    uB    = 9.274e-24         # Bohr magneton (J/T)
    h_bar = 1.054e-34         # Reduced Planck constant (J*s)
    u0    = np.pi * 4e-7      # Permeability of free space (T*m/A)
    e     = 1.6e-19           # Elementary charge (C)
    kb    = 1.38e-23          # Boltzmann constant (J/K)

    # MTJ Parameters
    a      = 50e-9            # MTJ width (m)
    b      = 50e-9            # MTJ length (m)
    tf     = 1e-9           # Free layer thickness (m)
    tox    = 1.5e-9           # Oxide thickness (m)
    alphar = 10e-30           # Rashba spin coupling
    T      = temperature      # Temperature (K)
    M_a    = Ms_T_dependence(T, 0.8, 424, 9.3e5)
    M_b    = Ms_T_dependence(T, 0.21, 424, 4.5e5)
    P      = 0.6              # Spin polarization factor
    if M_a > M_b:
        P = +0.6   # TM sublattice dominates, usual sign
    else:
        P = -0.6   # RE sublattice dominates, flipped sign
    # this shows a nice compentation of Ms values at 420K, which is the Curie temperature of the ferrimagnetic material used in the MTJ in the above paper
    ######################### Newer values #########################
    # Gyromagnetic ratios

    # Layer A (rad/(s*T)) # value just guess tho it doesnt seem to have a huge impact on the result
    gamma_a2 = 0.864e6           # Layer B (rad/(s*T))
    gamma   = (M_a - M_b) / (M_a/gamma_a - M_b/gamma_a2)
    gamma_b = gamma / u0

    # Anisotropy and magnetization
    # Ki   = 1.0056364e-5       # Interfacial anisotropy (J/m^2) # This value technically should depend on Ms and T, but the results get extremely sensitive to this value so I have set it to a constant value (again not very physical)
    Ki = 3.8e4 * tf
    Ki   = Ki - tf * u0 * (M_a - M_b) ** 2 / 2  # Average interfacial anisotropy
    # Ki /= 2
    Ms   = M_a - M_b    # Saturation magnetization (A/m)
    # I have noticed that the Ms value has a huge impact on the result and the device is basically useless if 10000 > abs(Ms - 1.2e6) A/m 
    alpha_a = 0.072             # Damping factor layer A
    alpha_b = 0.078             # Damping factor layer B
    alpha = (alpha_a * M_a / gamma_a + alpha_b * M_b / gamma_a2) / (M_a/gamma_a - M_b/gamma_a2)
    # alpha = 0.03
    Bsat = Ms * u0

    # VCMA and geometry
    ksi    = 75e-15           # VCMA coefficient (J/Vm)
    gammap = gamma / (1 + alpha * alpha)
    v      = tf * np.pi * b * a / 4
    Vh     = 0.5              # Voltage (V)
    delta  = 40
    TMR    = 1.2
    Rp     = 5e3              # Parallel resistance (Ohm)
    A      = a * b * np.pi / 4
    eta    = 0.3

    # SHE geometry and resistance
    w   = 100e-9              # Width (m)
    l   = 100e-9              # Length (m)
    d   = 3e-9                # Thickness (m)
    A2  = d * w
    rho = 200e-8              # Resistivity (Ohm*m)
    R2  = rho * l / (w * d)

    # Applied fields
    Hx = 0
    Hy = Hshe
    Hz = Happl

    # Thermal field
    Htherm = np.sqrt((2 * u0 * alpha * kb * T) / (Bsat * gamma_b * t_step * v)) / u0
    # print(f"alpha: {float(alpha)}, gamma: {float(gamma)}, Ms: {float(Ms)}, Ki: {float(Ki)}")
    
    # Spin torque factor;
    F = (gamma*h_bar)/(2*u0*e*tf*Ms)     
    
    # Demagnetization field; This is experimental value
    Nx = 0.010613177892974
    Ny = 0.010613177892974
    Nz = 0.978773644214052

    # save all initial parameters into a file
    # Collect parameters
    params = {
        "init": init,
        "phi_init": phi_init,
        "t_step": t_step,
        "v_pulse": v_pulse,
        "t_pulse": t_pulse,
        "t_relax": t_relax,
        "Happl": Happl,
        "Hshe": Hshe,
        "J_stt": J_stt,
        "J_she": J_she,
        "vhold": vhold,
        "temperature": temperature,
        "M_a": float(M_a),
        "M_b": float(M_b),
        "Ms": float(Ms),
        "gamma_a": float(gamma_a),
        "gamma_b": float(gamma_b),
        "gamma": float(gamma),
        "alpha_a": float(alpha_a),
        "alpha_b": float(alpha_b),
        "alpha": float(alpha),
        "Ki": float(Ki),
        "Bsat": float(Bsat),
        "SHE width": float(w),
        "SHE length": float(l),
        "SHE thickness": float(d),
        "SHE rho": float(rho),
        "VCMA ksi": float(ksi),
        "VCMA Vh": float(Vh),
        "TMR": float(TMR),
        "VCMA parallel resistance (Rp)": float(Rp),
        "VCMA eta": float(eta),
        "MTJ width (a)": float(a),
        "MTJ length (b)": float(b),
        "MTJ thickness (tf)": float(tf),
        "MTJ Rashba spin coupling (alphar)": float(alphar),
        "MTJ spin polarization (P)": float(P),
        "MTJ oxide thickness (tox)": float(tox),
        "Nx": Nx,
        "Ny": Ny,
        "Nz": Nz,
        "Htherm": float(Htherm),
        "F": float(F),
        "flips": int(flips)
    }

    #------------------------Initialization----------------------------#

    phi = []
    theta = []
    energy = []
    power = []
    R = []
    j_she = []
    bitstr = []
    phi.append(phi_init)                              
    theta.append(init)  
    energy.append(0)
    power.append(0)
    R.append(Rp)             # TMR from Parallel state (Start from Rp=60)
    j_she.append(0)
    for flip in range(flips):
        for i in range(int(t_pulse/t_step)):
            V = v_pulse
            J_SHE = 0
            J_STT = J_stt 
            Hk = (2*Ki)/(tf*Ms*u0)-(2*ksi*V)/(u0*Ms*tox*tf) # H_PMA - H-VCMA
            Ax = Hx-Nx*Ms*np.sin(theta[-1])*np.cos(phi[-1])+np.random.normal()*Htherm
            Ay = Hy-Ny*Ms*np.sin(theta[-1])*np.sin(phi[-1])+np.random.normal()*Htherm
            Az = Hz-Nz*Ms*np.cos(theta[-1])+Hk*np.cos(theta[-1])+np.random.normal()*Htherm

            # LLG
            dtheta = gammap*(
                Ax*(alpha*np.cos(theta[-1])*np.cos(phi[-1])-np.sin(phi[-1]))
                + Ay*(alpha*np.cos(theta[-1])*np.sin(phi[-1])+np.cos(phi[-1]))
                - Az*alpha*np.sin(theta[-1]))                                                                                \
                - J_SHE*F*eta*(np.cos(phi[-1])*np.cos(theta[-1])+(alpha*np.sin(phi[-1]))/(1+alpha*alpha))                    \
                + ((F*P*J_STT)*np.sin(theta[-1])/(1+alpha*alpha))  
            
            dphi = gammap*(
                Ax*(-np.cos(theta[-1])*np.cos(phi[-1])-alpha*np.sin(phi[-1]))
                + Ay*(-np.cos(theta[-1])*np.sin(phi[-1])+alpha*np.cos(phi[-1]))
                + Az*np.sin(theta[-1]))/(np.sin(theta[-1]))                                                                         \
                + J_SHE*F*eta*(np.sin(phi[-1])-alpha*np.cos(phi[-1])*np.cos(theta[-1]))/(np.sin(theta[-1])*(1+alpha*alpha))         \
                - ((alpha*F*P*J_STT)/(1+alpha*alpha))
            
            R1 = Rp*(1+(V/Vh)**2+TMR)/(1+(V/Vh)**2+TMR*(1+(np.sin(theta[-1])*np.cos(phi[-1]) ))/2)
            power.append(V**2/R1+R2*(np.abs(J_SHE*A2))**2+R1*(J_STT*A)**2)
            phi.append(phi[-1]+t_step*dphi)                                     
            theta.append(theta[-1]+t_step*dtheta)
            energy.append(energy[-1]+t_step*power[-1])
            R.append(R1)     # MTJ Resistance 
            j_she.append(J_SHE)
        for i in range(int(t_relax/t_step)):
            V = vhold
            J_SHE = J_she
            # J_SHE = 0
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
                + ((F*P*J_STT)*np.sin(theta[-1])/(1+alpha*alpha))  
            
            dphi = gammap*(
                Ax*(-np.cos(theta[-1])*np.cos(phi[-1])-alpha*np.sin(phi[-1]))
                + Ay*(-np.cos(theta[-1])*np.sin(phi[-1])+alpha*np.cos(phi[-1]))
                + Az*np.sin(theta[-1]))/(np.sin(theta[-1]))                                                                         \
                + J_SHE*F*eta*(np.sin(phi[-1])-alpha*np.cos(phi[-1])*np.cos(theta[-1]))/(np.sin(theta[-1])*(1+alpha*alpha))         \
                - ((alpha*F*P*J_STT)/(1+alpha*alpha))
            
            R1 = Rp*(1+(V/Vh)**2+TMR)/(1+(V/Vh)**2+TMR*(1+(np.sin(theta[-1])*np.cos(phi[-1]) ))/2)
            power.append(V**2/R1+R2*(np.abs(J_SHE*A2))**2+R1*(J_STT*A)**2)
            phi.append(phi[-1]+t_step*dphi)                                     
            theta.append(theta[-1]+t_step*dtheta)
            energy.append(energy[-1]+t_step*power[-1])
            R.append(R1)
            j_she.append(J_SHE)
        bitstr.append(1 if np.cos(theta[-1]) > 0 else -1)
    
    theta_arr = np.array(theta)
    phi_arr = np.array(phi)

    mx = np.sin(theta_arr) * np.cos(phi_arr)
    my = np.sin(theta_arr) * np.sin(phi_arr)
    mz = np.cos(theta_arr)

    G = 1/np.array(R)
    t = np.arange(0,len(mz)*t_step,t_step)
    # print(M_a, M_b)
    return theta_arr[-1], phi_arr[-1], t, np.array(R), G, mx, my, mz, bitstr[0], np.array(power), energy[-1], Ms, gamma, alpha, np.array(j_she), M_a, M_b, params