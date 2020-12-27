"""
First try of solving assignment 1 
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

###############################################################################
# Specifying paths
dir = Path.cwd()
current_folder = dir / "A1"

output = current_folder / "output"
output.mkdir(parents=True, exist_ok=True)

###############################################################################
# Setting up hardcoded values
def setup():

    init_mat = np.empty((1000, 3)) # rows are time periods

    Y_0 = 100
    pi_0 = 1
    i_0 = 1

    init_mat[0,:] = [Y_0, pi_0, i_0]

    # Setting constants
    params = {"sigma": 2,
                "kappa": 0.3,
                "beta": 0.99,
                "phi_1": 1.5,
                "phi_2": 0.2}
    
    return init_mat, params

###############################################################################
###############################################################################
# Task 2.2

# Functions for calculating
def get_i(pi_tplus1, Y_tplus1, params):
    i_t = params["phi_1"] * pi_tplus1 + params["phi_2"] * Y_tplus1
    
    return i_t

def get_Y(pi_tplus1, Y_tplus1, i_t, params):
    Y_t = Y_tplus1 - (1/params["sigma"]) * (i_t - pi_tplus1)

    return Y_t

def get_pi(pi_tplus1, Y_t, params):
    pi_t = params["beta"] * pi_tplus1 + params["kappa"] * Y_t

    return pi_t

def loopinglouie(mat, params):

    Y = mat[:, 0]
    pi = mat[:, 1]
    i = mat[:, 2]

    for t in np.arange(1, len(Y)):

        i[t] = get_i(pi[t-1], Y[t-1], params)
        Y[t] = get_Y(pi[t-1], Y[t-1], i[t], params)
        pi[t] = get_pi(pi[t-1], Y[t], params)
    
    print("Ran successfully")
    return np.matrix([Y, pi, i]).transpose()

# Functions for plotting
def plot(res_mat):
    t = np.arange(res_mat.shape[0]) # Time periods will be the x-axis
    
    fig, axs = plt.subplots(3, figsize=(15, 6))
    

    axs[0].plot(t, res_mat[:, 0])
    axs[0].set_title("Y")

    axs[1].plot(t, res_mat[:, 1])
    axs[1].set_title("pi")

    axs[2].plot(t, res_mat[:, 2])
    axs[2].set_title("i")

    figname = output / "plot01"
    plt.savefig(figname)
    print("Saved plot in output folder")
###############################################################################
# Task 2.3 and following

def get_A_inv(params):
    A = np.matrix([[1, 0, 1/params["sigma"]],
                    [params["kappa"], 1, 0],
                    [0, 0, 1]])
    A_inv = np.linalg.inv(A)

    return A_inv

def calc(e_of_z, *addons):
    """
    Input values are expectations of future values
    diff is for whether to take the actual values or the differenced ones
    """

    #Unpacking addons. It's in one for the fsolve later
    params, diff = addons

    # Unpacking values from matrix
    EY = e_of_z[0]
    Epi = e_of_z[1]
    Ei = e_of_z[2]
    
    first_line = EY - (1/params["sigma"]) * (- Epi)
    second_line = params["beta"] * Epi
    third_line = params["phi_1"] * Epi + params["phi_2"] * EY

    B = np.matrix([[first_line],
                    [second_line],
                    [third_line]])
    
    A_inv = get_A_inv(params)

    z = np.dot(A_inv, B)

    # ravel to flatten matrix to array
    if diff:
        return e_of_z - np.ravel(z)
    else:
        return np.ravel(z)
###############################################################################
###############################################################################







# Defining task 2.2 which saves the plot in the end
def task2_2(init_mat, params):
    print("Starting with Task 2.2")
    
    res = loopinglouie(init_mat, params)
    plot(res)
    print("Completed Task 2.2")

def task2_3(params):
    print("Starting with tasks 2.3 - 2.5")
    
    values = np.array([0, 0, 0]) # Taking the values the previous thing converged to
    spec = (params, False) # False here stands for not taking the difference

    res = calc(values, *spec) 
    print(res)

def task2_7(params):
    print("Starting with task 2.7")
    
    values = np.array([0, 0, 0]) # Taking the values the previous thing converged to
    spec = (params, True) # True here stands for taking the difference

    res = calc(values, *spec) 
    print(res)

def task2_8(params):
    print("Starting with task 2.8")

    values = np.array([0, 0, 0]) # Taking the values the previous thing converged to
    spec = (params, True) # True here stands for taking the difference
    
    res = optimize.fsolve(calc, x0=values, args=spec)
    print(res)

if __name__ == "__main__":
    init_mat, params = setup()
    
    print("------")
    task2_2(init_mat, params)
    print("------")
    task2_3(params)
    print("------")
    task2_7(params)
    print("------")
    task2_8(params)