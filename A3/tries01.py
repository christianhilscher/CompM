"""
01.01.2021

First version of solving Assignment 3 which is to be handed in
"""

from pathlib import Path
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# Initializing values
T = 1000
shocks = np.random.normal(size=T)
shocks = np.zeros(T)

C_hat_mat = np.empty(shape=(2, 3, T))
R_mat = np.empty(shape=(2, 2, T))

# Making first guesses and setting values
C_hat_mat[:, :, 0] = np.random.uniform(low=-0.9, high=0.9, size=(2,3))
R_mat[:, :, 0] = np.ones((2,2))

params = {"sigma": 2,
          "kappa": 0.3,
          "beta": 0.99,
          "phi1": 1.5,
          "phi2": 0.2,
          "gamma": 0.05}

def get_A_inv(params):
    """
    Calculating the inverse of the matrix A where A * z_t = B
    Inputs: 
        - params: All the parameters
    """
    A = np.matrix([[1, 0, 1/params["sigma"]],
                   [- params["kappa"], 1, 0],
                   [0, 0, 1]])
    
    A_inv = np.linalg.inv(A)
    
    return A_inv

def get_B(vals, params):
    
    # Unpacking 
    E_Y, E_pi, E_i = vals
    
    first_line = E_Y + (1/params["sigma"]) * E_pi
    second_line = params["beta"] * E_pi
    third_line = params["phi1"] * E_pi + params["phi2"] * E_Y
    
    B = np.matrix([[first_line],
                   [second_line],
                   [third_line]])
    
    return B

def get_C0C1(A_inv, B):
    
    # In case epsilon==0 -> z_t = C0 which is just A^-1 * B
    C0 = np.dot(A_inv, B)
    
    # In case epsilon != 0, z_t = C0 + C1 which is equal to B = (1, 0, 0) 
    # Is this true??
    output_shock = B + np.matrix([[1], 
                                  [0], 
                                  [0]])
    
    C0C1 = np.dot(A_inv, output_shock)
    C1 = C0C1 - C0
    
    return [C0, C1]

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
    
    R_new = R_old + params["gamma"] * (eta @ np.transpose(eta) - R_old)
    return R_new

def update_C(C_old, R_new, z, eta, params):
    
    firstpart = params["gamma"] * np.linalg.inv(R_new) @eta
    secondpart = z - np.transpose(eta) @ C_old

    C_new = C_old - firstpart @ secondpart
    return C_new

def calc_current_vals(E_z, epsilon, params):
    E_Y, E_pi, E_i = E_z
    
    i = get_i(E_Y, E_pi, params)
    Y = get_Y(E_Y, E_pi, i, epsilon, params)
    pi = get_pi(Y, E_pi, params)
    
    return np.array([Y, pi, i])

def loopinglouie(C_hat, R, shocks, params):

    periods = len(shocks) # Inferring time periods from shock array
    z_t = np.empty(shape=(3, periods)) # Saving actual outcome of economy
    
    for t in range(1, periods):
        # Expectations are in first line since E[z_t] = C0hat
        expectations = C_hat[0, :, t]

        # Calculate values in period t
        z_t[:, t] = calc_current_vals(expectations, shocks[t], params)
        
        # Define eta as vector 
        eta = np.array([[1], 
                        [shocks[t]]])
        # update R and C
        R[:, :, t] = update_R(R[:, :, t-1], 
                              eta, 
                              params)
        
        C_hat[:, :, t] = update_C(C_hat[:, :, t-1], 
                                  R[:, :, t], 
                                  z_t[:, t], 
                                  eta,
                                  params)
        
        # Saving values to result matrix

    return C_hat, R, z_t

a, b, c = loopinglouie(C_hat = C_hat_mat,
                       R = R_mat,
                       shocks = shocks,
                       params = params)

###

C_hat = init_vals
R = init_R
shocks = shocks
result_mat = res
params = params


