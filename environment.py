# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:32:30 2019

@author: sanja
"""

import ODETurb
import numpy as np
# Environment for the wind turbine system. The goal is to maximize power
# Action: The control inputs are pitch angle and Generator Power
# Reward: The power itself
# x_next: Dictated by the ODE. (have to discretize and use it)
from numpy import linspace, zeros, exp
import matplotlib.pyplot as plt

def reset():
    x = np.random.uniform(0.2,1,4)
    return x

def step(a,x):
    
    # simulate to get next step using euler
    dt = 1e-2
    x_next = x + dt*ODETurb.Turbine(0,x,a)
    
    # Rewards for the system
    Kopt = 0.28;
    rewards = Kopt*x[2]**3;
    done =  x[0] > 1
    done = bool(done)
    
    return x_next, rewards, done 

