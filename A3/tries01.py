"""
01.01.2021

First version of solving Assignment 3 which is to be handed in
"""

from pathlib import Path
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# Initializing values
T = 100000
init_vals = np.random.uniform(low=-0.9, high=0.9, size=(2,3))
shocks = np.random.normal(size=T)
init_R = np.ones((2,2))

res = np.empty(shape=(3, T))

params = {"sigma": 2,
          "kappa": 0.3,
          "beta": 0.99,
          "phi1": 1.5,
          "phi2": 0.2,
          "gamma": 0.05}

def get_i(E_Y, E_pi, params):
    
    i = params["phi1"] * E_pi + params["phi2"] * E_Y
    return i

def get_Y(E_Y, E_pi, i, epsilon, params):
    
    Y = E_Y - (1/params["sigma"]) * (i - E_pi) + epsilon
    return Y

def get_pi(Y, E_pi, params):
    
    pi = params["beta"] * E_pi + params["kappa"] * Y
    return pi

def update_R(R_old, eta, params):
    
    R_new = R_old + params["gamma"] * (eta * np.transpose(eta) - R_old)
    return R_new

def update_C(C_old, R_new, z, eta):
    
    firstpart = params["gamma"] * np.dot(np.linalg.inv(R_new),eta)
    secondpart = z - np.dot(np.transpose(eta), C_old)

    C_new = C_old - firstpart * secondpart
    return C_new

def calc_current_vals(E_z, epsilon, params):
    E_Y, E_pi, E_i = E_z
    
    i = get_i(E_Y, E_pi, params)
    Y = get_Y(E_Y, E_pi, E_i, epsilon, params)
    pi = get_pi(Y, E_pi, params)
    
    return np.array([Y, pi, i])

def loopinglouie(C_hat, R, shocks, result_mat, params):

    periods = len(shocks) # Inferring time periods from shock array
    
    for t in np.arange(periods):
        # Expectations are in first line since E[z_t] = C0hat
        expectations = C_hat[0,:]
        
        # Calculate values in period t
        z_t = calc_current_vals(expectations, shocks[t], params)
        
        # Define eta as vector 
        eta = np.array([[1], 
                         [shocks[t]]])
        # update R and C
        R = update_R(R, eta, params)
        C_hat = update_C(C_hat, R, z_t, eta)
        
        # Saving values to result matrix
        result_mat[:,t] = z_t
    
    return result_mat

abc = loopinglouie(C_hat = init_vals, 
                   R = init_R, 
                   shocks = shocks, 
                   result_mat = res, 
                   params = params)






