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
def make_paths():
    dir = Path.cwd()
    current_folder = dir / "A2"

    output = current_folder / "output"
    output.mkdir(parents=True, exist_ok=True)

###############################################################################
# Setting up hardcoded values
def setup(T=1000):

    init_mat = np.empty((T, 3)) # rows are time periods

    # Initial values are drawn from a uniform distribution
    init_mat[0,:] = np.random.uniform(low=-0.99, high=0.99, size=3)

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
    
    # Calculating elements of B matrix individually
    first_line = E_Y + (1/params["sigma"]) * E_pi
    second_line = params["beta"] * E_pi
    third_line = params["phi1"] * E_pi + params["phi2"] * E_Y
    
    # Putting parts together
    B = np.matrix([[first_line],
                   [second_line],
                   [third_line]])
    
    return B

def get_C0C1(A_inv, B):
    
    # In case epsilon==0 -> z_t = C0 which is just A^-1 * B
    C0 = np.dot(A_inv, B)
    
    # In case epsilon != 0, z_t = C0 + C1 which is equal to B = (1, 0, 0) 
    output_shock = B + np.matrix([[1], 
                                  [0], 
                                  [0]])
    
    # C0C1 is the value of C0 + C1
    C0C1 = np.dot(A_inv, output_shock)
    
    # Getting C1 by substracting C0
    C1 = C0C1 - C0
    
    return [C0, C1]

def get_C0C1C2(A_inv, B):
    # If epsilon and eta == 0, z_t = C0 which is A^-1 * B
    C0 = np.dot(A_inv, B)
    
    # For getting C1, assume eta==0, epsilon==1, where epsilon is the shock to output
    output_shock = B + np.matrix([[1],
                                  [0],
                                  [0]])
    C0C1 = np.dot(A_inv, output_shock)
    C1 = C0C1 - C0
    
    # For getting C2, assume that epsilon==0 and eta==1
    inflation_shock = B + np.matrix([[0],
                                     [1],
                                     [0]])
    
    # Same procedure as above
    C0C2 = np.dot(A_inv, inflation_shock)
    C2 = C0C2 - C0
    
    return [C0, C1, C2]
###############################################################################
# Functions for calculating 

def calc_oneshock(e_of_z, params):
    
    # First getting the A and B matrices.
    B = get_B(e_of_z[0:3], params)
    A_inv = get_A_inv(params)
    
    # With those matrices now calculating C0 and C1
    C0, C1 = get_C0C1(A_inv, B)
    C_calculated = np.array([C0, C1]).flatten() #Putting it together into a 1D array
    
    return C_calculated - e_of_z

def calc_twoshocks(e_of_z, params):
    # Same function as above just using two shocks.
    
    B = get_B(e_of_z[0:3], params)
    A_inv = get_A_inv(params)
    
    C0, C1, C2 = get_C0C1C2(A_inv, B)
    C_calculated = np.array([C0, C1, C2]).flatten() #Putting it together into a 1D array
    
    return C_calculated - e_of_z

def task2_2(params):
    print("Task 2.2:")
    
    initial_vals = np.random.uniform(low=-1, high=1, size=6)
    res = optimize.fsolve(func = calc_oneshock, 
                          x0 = initial_vals,
                          args = params)
    print(f"Result for C0: {res[0:3]} \n Result for C1: {res[3:6]}")
    
def task2_3(params):
    print("Task 2.3:")
    
    initial_vals = np.random.uniform(low=-1, high=1, size=9)
    res = optimize.fsolve(func = calc_twoshocks, 
                          x0 = initial_vals, 
                          args = params)
    
    print(f"Result for C0: {res[0:3]} \n Result for C1: {res[3:6]} \n Result for C2: {res[6:9]}")
    
def run_A2():
    
    # Task 2.1 
    # Functions from previous week are adapted to fit the new setup
    
    # Setting the parameters of the model
    params = {"sigma": 2,
            "kappa": 0.3,
            "beta": 0.99,
            "phi1": 1.5,
            "phi2": 0.2}
    
    ###########################################################################
    # Task 2.2
    print("Task 2.2:")
    
    # Intital values are random draws
    initial_vals = np.random.uniform(low=-1, high=1, size=6)
    
    # Running fsolve on the function with one shock
    res = optimize.fsolve(func = calc_oneshock, 
                          x0 = initial_vals,
                          args = params)
    # Printing results
    
    print(f"Result for C0: {res[0:3]} \n Result for C1: {res[3:6]}")
    
    """
    The values for C_0 are the same as from Assingment 1. All of them are 0
    which makes sense insofar sa that C_0 caputres those parameters of the
    model which are not influenced by any shocks. Thus if we were to assume
    all shocks to be 0, the outcome would be the same as last week where we
    did not have any shocks. 
    """
    
    ###########################################################################
    # Task 2.3 
    # Same as above but assuming one output shock and one inflation shock
    print("Task 2.3:")
    
    initial_vals = np.random.uniform(low=-1, high=1, size=9)
    res = optimize.fsolve(func = calc_twoshocks, 
                          x0 = initial_vals, 
                          args = params)
    
    print(f"Result for C0: {res[0:3]} \n Result for C1: {res[3:6]} \n Result for C2: {res[6:9]}")
    
###############################################################################
if __name__ == "__main__":
    
    # No need for paths since not saving any plots
    # make_paths()
    run_A2()