import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt
from numpy.lib.npyio import zipfile_factory
from scipy.optimize import fsolve

#* Paths
dir = Path.cwd() 
out = dir / 'output'

#*##############
#! FUNCTIONS 
#*##############

#* from PS1
def setup(T = 1000): 
    '''
    Set up everything we need.
    
    *T = periods simulated.
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

#* potential alternative definition (leads to same result)
def A_alt(params): 
    '''
    Alternative specification of A in which IS plugged into PC to remove 
    e_t from LHS of A z_t = B. 
    
    *params = dictionary of params obtained from setup()
    '''
    A = np.array([[1, 0, 1/params['sigma']], [0, 1, params['kappa']/params['sigma']], 
                    [0, 0, 1]])
    A_inv = np.linalg.inv(A)
    return A, A_inv

#* new functions: Task 2.1
def get_zt_w_shock(input_arr): 
    '''
    Get z_t = inv(A)*B with shock e_t affecting Y_t in IS curve.
    
    *input_arr = vector containing [E_Yt1, E_pit1, E_it1, C1_1, C1_2, C1_3]
    '''
    params = setup()[1]
    A_inv = A(params)[1]
    B = np.array([input_arr[0] + 1/params['sigma'] * input_arr[1] + input_arr[3], 
                    params['beta'] * input_arr[1], 
                    params['phi1'] * input_arr[1] + params['phi2'] * input_arr[0]])
    z_t = np.dot(A_inv, B)
    '''
    A_inv_alt = A_alt(params)[1]
    B_alt = np.array([input_arr[0] + 1/params['sigma'] * input_arr[1] + input_arr[3], 
                    params['beta'] * input_arr[1] + 
                    params['kappa'] * (input_arr[0] + 1/params['sigma'] * input_arr[1] + input_arr[3]), 
                    params['phi1'] * input_arr[1] + params['phi2'] * input_arr[0]])
    z_t_alt = np.dot(A_inv_alt, B_alt)'''
    return z_t#, z_t_alt

def get_C0C1(input_arr): 
    '''
    C0 is z_t w/ e_t = 0.
    Calculate z_t once with error e_t = 0 and once with error e_t = 1
    and take difference to find C1. 
    
    *input_arr = np.array that contains [E_Yt1, E_pit1, E_it1, C1_1, C1_2, C1_3]
    '''
    #first: get z_t with e_t = 0, set e_t = 0 in input
    input_arr[3] = 0
    #and calculate z_t
    zt_noshock = get_zt_w_shock(input_arr)
    #now z_t with e_t = 1
    input_arr[3] = 1
    zt_shock = get_zt_w_shock(input_arr)
    
    #C0 is z_t with e_t = 0
    C0 = zt_noshock
    #C1 is difference between z_t with shock and z_t without shock
    C1 = zt_shock - zt_noshock
    
    return C0, C1

#* functions for Task 2.3: z_t = inv(A)*B has to be adjusted and then need function to get C0, C1 and C2
def get_zt_2shocks(input_arr): 
    '''
    Get z_t = inv(A)*B with shock to output AND inflation. 
    
    *input_arr = np.array that contains [E_Yt1, E_pit1, E_it1, C1_1, C1_2, C1_3, C2_1, C2_2, C2_3]
    '''
    params = setup()[1]
    #A is the same as with one shock
    A_inv = A(params)[1]
    #only second element of B adjusts compared to one shock
    B = np.array([input_arr[0] + 1/params['sigma'] * input_arr[1] + input_arr[3], 
                #HERE NEW: shock to pi (u_t) = input_arr[4]
                params['beta'] * input_arr[1] + input_arr[4], 
                params['phi1'] * input_arr[1] + params['phi2'] * input_arr[0]])
    #calc z_t
    z_t = np.dot(A_inv, B)
    
    return z_t

def get_C0C1C2(input_arr): 
    '''
    C0 is z_t w/ e_t = 0.
    Calculate z_t once with error e_t = u_t = 0 and once with error e_t = 1, u_t = 0
    and take difference to find C1. 
    Calculate z_t with error e_t = u_t = 0 and once with error 
    *input_arr = np.array that contains [E_Yt1, E_pit1, E_it1, C1_1, C1_2, C1_3, C2_1, C2_2, C2_3], 
    where e_t = shock to output, u_t = shock to inflation, eps_t = shock to nominal interest;
    note that eps_t not considered (as assumed to be zero)    
    '''
    

#* function the same for all
def get_diff(expec, input_arr, one_shock = True): 
    '''
    Get difference between implied C0 and C1 obtained from get_C0C1() and
    values used for expecations. 
    
    *expec = np.array containing first guesses for C0 and C1 
    *input_arr = input array for get_C0C1()
    *one_shock = Bool, define whether consider only shock to Y (Task 2.2) or also shock to pi; 
                True by default
    '''
    if one_shock:
        #first get the implied values
        implied = np.array(get_C0C1(input_arr))
        #flatten C0 and C1 into a 1D array with C0, C1 
        implied = implied.flatten()
    else: 
        #first get the implied values
        implied = np.array(get_C0C1C2(input_arr))
        #flatten C0 and C1 into a 1D array with C0, C1, C2
        implied = implied.flatten()
    
    #get difference 
    diff = expec - implied 
    
    return diff

def calcs(epec, input_arr, one_shock = True): 
    '''
    Do calculations and print results in a nice manner.
    
    *expec = np.array containing first guesses for C0 and C1 
    *input_arr = input array for get_C0C1()
    '''
    

#*##############
#! TASKS
#*##############

#* 2.2
#use steady state from before without shock
inputs = np.array([0, 0, 0, 0, 0, 0])
solved = fsolve(get_diff, np.array([100, 1, 1, 10, 50, 0]), inputs)

#* 2.3
