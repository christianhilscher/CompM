from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.optimize as fsolve 

#* Paths
dir = Path.cwd() 
out = dir / 'output'

#*##############
#! FUNCTIONS 
#*##############

def setup(T):
    #set up estimates for C0 and C1 (randomly drawn between -1 and 1)
    C0_hat = np.random.uniform(low = -1, high = 1, size = 3)
    C1_hat = np.random.uniform(low = -1, high = 1, size = 3)
    #get first fuess in 2x3 matrix
    first_guess = np.array([C0_hat, C1_hat])
    #generate array of shocks for T periods; distribution is N(0, 1)
    shocks = np.random.randn(T)
    #set up parameters 
    params = {'sigma': 2, 
            'kappa': 0.3, 
            'beta': 0.99, 
            'phi1': 1.5, 
            'phi2': 0.2}

    return first_guess, shocks, params

def A(params): 
    '''
    Get matrix A as derived before such that A z_t = B from model equations, where 
    z_t = [Y_t, pi_t, i_t].
    
    *params = dictionary of params obtained from setup()
    '''
    A = np.array([[1, 0, 1/params['sigma']], [-params['kappa'], 1, 0], 
                    [0, 0, 1]])
    A_inv = np.linalg.inv(A)
    return A, A_inv

def get_zt(C0C1, shock):
    '''
    Find z_t = C_0 + C_1 e_t, where e_t is the shock of interest.
    
    *C0C1 = 2x3 matrix containing C0 in first, C1 in second row
    *shock = value of e_t 
    '''
    #split up input array
    C0 = C0C1[0, :]
    C1 = C0C1[1, :]
    #calculate z_t 
    z_t = C0 + C1 * shock
    
    return z_t

def updating_R(R_old, shock, gamma = 0.05):
    '''
    Learning rule for moment matrix R. 
    
    *R_old = previous R
    *shock = value of e_t 
    *gamma = gain parameter of learning mechanism, set to 0.05 as default
    '''
    #first, get v
    shock = 1
    v = np.array((1, 1))
    
    
def updating_C()
def simul(T): 
    first_guess, shocks, params = setup(T)
    A_inv = A(params)[1]
    