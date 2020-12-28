"""
28.12.2020

First version of solving Assignment 2 which is to be handed in
"""

from pathlib import Path
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

###############################################################################
# Specifying paths
dir = Path.cwd()
current_folder = dir / "A2"

output = current_folder / "output"
output.mkdir(parents=True, exist_ok=True)

###############################################################################
# Setting up hardcoded values
def setup(T=1000):

    init_mat = np.empty((T, 3)) # rows are time periods

    Y_0 = 100
    pi_0 = 1
    i_0 = 1
    init_mat[0,:] = [Y_0, pi_0, i_0]

    # Setting constants
    params = {"sigma": 2,
              "kappa": 0.3,
              "beta": 0.99,
              "phi1": 1.5,
              "phi2": 0.2}
    
    return init_mat, params

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
    adjusted_mat = B + np.matrix([[1], 
                                  [0], 
                                  [0]])
    
    C0C1 = np.dot(A_inv, adjusted_mat)
    C1 = C0C1 - C0
    
    return [C0, C1]
###############################################################################
# Functions for calculating 
def calc(e_of_z, params):
    
    B = get_B(e_of_z[0:3], params)
    A_inv = get_A_inv(params)
    
    C0, C1 = get_C0C1(A_inv, B)
    C_calculated = np.array([C0, C1]).flatten() #Putting it together into a 1D array
    
    return C_calculated - e_of_z

def task2_1(params):
    print("Task 2.1:")
    
    initial_vals = [0, 0, 0, 0, 0,0]
    res = optimize.fsolve(calc, initial_vals, params)
    print(f"Result for C0: {res[0:3]} \n Result for C1: {res[3:6]}")

###############################################################################
if __name__ == "__main__":
    mat, params = setup()
    task2_1(params)