import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

#* Paths
dir = Path.cwd() 
out = dir / 'output'

#*##############
#! FUNCTIONS 
#*##############
def setup(T = 1000): 
    '''
    Set up everything we need.
    '''
    #order of columns: i, pi, Y
    mat = np.empty((T, 3))
    #set initial values 
    i0 = 1
    pi0 = 1
    Y0 = 100
    #set inital values in matrix 
    for j, val in zip(range(3), [i0, pi0, Y0]):
        mat[0, j] = val
    params = {'sigma': 2, 
                'kappa': 0.3, 
                'beta': 0.99, 
                'phi1': 1.5, 
                'phi2': 0.2}
    return(mat, params)

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

def A_alt(params): 
    '''
    Alternative specification of A in which IS plugged into PC to remove 
    e_t from LHS of A z_t = B. 
    *params = dictionary of params obtained from setup()
    '''
    params = setup()[1]
    A = np.array([[1, 0, 1/params['sigma']], [0, 1, params['kappa']/params['sigma']], 
                    [0, 0, 1]])
    A_inv = np.linalg.inv(A)
    return A, A_inv

def get_zt_w_error(input_arr): 
    '''
    
    *input_arr = vector containing [E_Yt1, E_pit1, E_it1, e_Y, e_pi, e_i]
    '''
    params = setup()[1]
    A_inv = A(params)[1]
    A_inv_alt = A_alt(params)[1]
    B = np.array([input_arr[0] + 1/params['sigma'] * input_arr[1] + input_arr[3], 
                    params['beta'] * input_arr[1], 
                    params['phi1'] * input_arr[1] + params['phi2'] * input_arr[0]])
    B_alt = np.array([input_arr[0] + 1/params['sigma'] * input_arr[1] + input_arr[3], 
                    params['beta'] * input_arr[1] + 
                    params['kappa'] * (input_arr[0] + 1/params['sigma'] * input_arr[1] + input_arr[3]), 
                    params['phi1'] * input_arr[1] + params['phi2'] * input_arr[0]])
    z_t = np.dot(A_inv, B)
    z_t_alt = np.dot(A_inv_alt, B_alt)
    return z_t, z_t_alt

def get_Cs(input_arr): 
    '''
    '''
    input_arr[3] = 0
    z_t_noerror = get_zt_w_error(input_arr)[0]
    input_arr[3] = 1
    z_t_error = get_zt_w_error(input_arr)[0]
    C1 = z_t_error - z_t_noerror
    C0 = z_t_noerror
    return C0, C1

inputs = np.array([100, 1, 1, 1, 0, 0])
outcome_vecs = get_Cs(inputs)

