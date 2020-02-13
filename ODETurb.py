# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:18:27 2019

@author: sanja
"""
import numpy as np


def Cp(lamda, beta):
    
    c1 = 0.22; c2=116; c3 = 0.4;
    c4 = c5 = 0; c6= 5; c7=12.5; c8=0.08; c9=0.035; c10=0;
    
    cp = c1*( (c2/(lamda + c8*beta)) - c2*c9/(beta**3 + 1) - c3 * beta - 
             c4 * beta**c5  - c6 )\
    * np.exp(-c7/(lamda + c8*beta)) + c10*lamda;
    
    return cp 



def Turbine(t,x, Control):
    
    
    F = np.zeros(len(x))
    
    # Rename variables
    wg = x[0];
    theta_tw = x[1];
    wt = x[2];
    
    # Parameters
    Ht = 4; 
    Hg = 0.1*Ht; 
    ksh = 0.3; 
    csh = 0.01;
    wel = 2*np.pi*60;
    Kopt = .5787
    
    # add a little bit of uncertainty in the air density
    rho = 1.225 + 1e-2 * np.random.randn();
    R_turb = 58.6;
    V = 12;
    Prated = 5e6;
    GB = 145.5;
    #wtB = wel/(2*GB);
    lamda = wt*R_turb/V;
    beta = Control;
    
    Tmbase =  GB * Prated * 1/(wel/2);
    
    # Intermediate variables
    Tm = 0.5 * rho * np.pi* R_turb**2 * Cp(lamda,beta) * V**3 /(wt*Tmbase);
    
    # Equations
    # Control is the generator torque itself
    
    F[0] = 1/(2*Hg) * (ksh*theta_tw + \
                           csh*wel* (wt - wg) - Kopt*x[2]**2 );
    
    F[1] = wel*(wt - wg);
    
    F[2] = 1/(2*Ht) * (Tm - ksh * theta_tw \
                             - csh*wel*(wt - wg));
    
     
    return F



# Let us check if indeed the maximum value of Cpmax is optimal. 


lamda = np.linspace(0.5, 10.0, num=10)
beta = np.linspace(-5,10 , 20)
Cptry = []

for i in range (len(lamda)):
    for j  in range (len(beta)):
        Cptry.append(Cp(lamda[i],0))
        
print('Optimal value of Cp = ',np.max(Cptry))       
        


