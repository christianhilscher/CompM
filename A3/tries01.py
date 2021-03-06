"""
01.01.2021

First version of solving Assignment 3 which is to be handed in
"""

from pathlib import Path
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# Initializing values
T = 10000
shocks = np.random.normal(size=T)
#shocks = np.zeros(T)

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

    C_new = C_old + firstpart @ secondpart
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
    z_t[:, 0] = C_hat[0, :, 0] # Values of first period are same as initial guesses
    
    for t in range(1, periods):
        # Expectations are in first line since E[z_t] = C0hat
        expectations = C_hat[0, :, t-1]

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

def plot(res_mat):
    t = np.arange(res_mat.shape[1]) # Time periods will be the x-axis
    
    fig, axs = plt.subplots(3, figsize=(15, 6))
    
    titles = ["Y", "pi", "i"]
    
    # Looping through the variables
    for p in np.arange(res_mat.shape[0]):
        axs[p].plot(t, res_mat[p, :])
        axs[p].set_title(titles[p])

    plt.show()
    # figname = output / "plot01"
    # plt.savefig(figname)
    # print("Saved plot in output folder")

###############################################################################
a, b, c = loopinglouie(C_hat = C_hat_mat,
                       R = R_mat,
                       shocks = shocks,
                       params = params)

plot(c)