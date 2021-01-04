"""
03.01.2021

Second version of solving Assignment 3 which is to be handed in
"""

from pathlib import Path
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

###############################################################################
# Specifying paths
def make_paths():
    dir = Path.cwd()
    current_folder = dir / "A3"

    output = current_folder / "output"
    output.mkdir(parents=True, exist_ok=True)

###############################################################################
# Setting up hardcoded values

def setup(T=100000):
    shocks = np.random.normal(size=T)
    #shocks = np.zeros(T)

    # 3rd dimension is the time dimension
    C_hat_mat = np.empty(shape=(2, 3, T))
    R_mat = np.empty(shape=(2, 2, T))

    # Making first guesses and setting values
    C_hat_mat[:, :, 0] = np.random.uniform(low=-0.99, high=0.99, size=(2,3))
    R_mat[:, :, 0] = np.ones((2,2))

    # Specifying parameters
    params = {"sigma": 2,
              "kappa": 0.3,
              "beta": 0.99,
              "phi1": 1.5,
              "phi2": 0.2,
              "gamma": 0.05}
    
    return C_hat_mat, R_mat, shocks, params
###############################################################################
# Functions for calculating

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

###############################################################################
# Functions for looping and plotting

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

def repackage(C_hat, z_t, shocks):
    
    E_Y = C_hat[0, 0, :]
    E_pi = C_hat[0, 1, :]
    plot_matrix = np.concatenate([z_t, [E_Y], [E_pi], [shocks]])
    
    fig_titles = ["Output", 
                  "Inflation", 
                  "Nominal Interest Rate",
                  "Expected output",
                  "Expected Inlfation",
                  "Shocks to output"]
    
    return plot_matrix, fig_titles

def plot(res_mat, titles):
    t = np.arange(res_mat.shape[1]) # Time periods will be the x-axis
    n_plots = res_mat.shape[0]
    
    fig, axs = plt.subplots(n_plots, figsize=(30, 30))

    
    # Looping through the variables
    for p in np.arange(n_plots):
        axs[p].plot(t, res_mat[p, :])
        axs[p].set_title(titles[p])

    figname = output / "plot01"
    plt.savefig(figname)
    print("Saved plot in output folder")

###############################################################################
# Defining tasks

def run_A3():
    
    # Task 3.1 and 3.2
    # Using random numbers as intial values
    C_hat, R, shocks, params = setup()
    
    # Task 3.3 
    C_results, R_results, z_results = loopinglouie(C_hat,
                                                   R,
                                                   shocks,
                                                   params)
    
    # Task 3.4
    """
    The agents do learn the MSV. The final values of c_hat are the same 
    as last week where we used fslove to numerically compute the MSV.
    The numbers are not exactly equal but treating anything smaller than 
    1e-10 as a computational 0, they are identical.
    """
    
    # Task 3.5
    plot_mat, fig_titles = repackage(C_results,
                                     z_results,
                                     shocks)
    plot(plot_mat, fig_titles)
    
    # Task 3.6
    """
    The output shock has an influence on both output and inlfation. That's why
    actual output and inflation vary over time. Nominal interest rate however
    is not influenced and therefore quickly converges to 0.
    Epxected output and inflation also converge quite quickly to their steady 
    state which is 0.
    The shocks to output vary over time since they are drawn ~ N(0, 1). 
    """



###############################################################################
if __name__ == "__main__":
    
    make_paths()
    run_A3()