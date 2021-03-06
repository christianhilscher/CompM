'''
Lucas Cruz Fernandez - University of Mannheim 

University of Heidelberg Student Number: 6000502
'''
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
    #set paramater to given values
    params = {'sigma': 2, 
                'kappa': 0.3, 
                'beta': 0.99, 
                'phi1': 1.5, 
                'phi2': 0.2}
    return(mat, params)

def A(params): 
    '''
    Get matrix A as derived such that A z_t = B from model equations, where 
    z_t = [Y_t, pi_t, i_t]. 
    
    *params = dictionary of params obtained from setup()
    '''
    A = np.array([[1, 0, 1/params['sigma']], [-params['kappa'], 1, 0], 
                    [0, 0, 1]])
    A_inv = np.linalg.inv(A)
    return A, A_inv

#* new functions: Task 1
def get_zt(input_arr, shock = True): 
    '''
    Calculate z_t using guesses for C0 and C1. 
    *input_arr = vector of length 6 containing first guesses
    '''
    #get parameters for calculating A
    params = setup()[1]
    #get inverse of A
    A_inv = A(params)[1]
    #get B
    B = np.array([input_arr[0] + 1/params['sigma'] * input_arr[1], 
                    params['beta'] * input_arr[1], 
                    params['phi1'] * input_arr[1] + params['phi2'] * input_arr[0]])
    #if shock e_t exists, vector (1, 0, 0) must be added to B
    #int(Bool) becomes 1 if Bool == True
    shock_array = np.array([int(shock), 0, 0])
    B_shock = B + shock_array
    #now z_t using z_t + inv(A) * B
    z_t = np.dot(A_inv, B_shock)
    return z_t

def get_C0C1(input_arr): 
    '''
    Find C0 and C1 using first guesses for C0 and C1. Note that C0 = inv(A)*B 
    and C1 = inv(A)*(1,0,0)'. Use steps as described in PS. 
    *input_arr = vector of length 6 containing first guesses
    '''
    #first get z_t when e_t = 0, this will be C0
    z_t_noshock = get_zt(input_arr, shock = False)
    C0 = z_t_noshock
    #then get z_t with shock e_t = 1, which will be C0 + C1
    z_t_shock = get_zt(input_arr, shock = True)
    #next, get C1 by substracting z_t_noshock from z_t_shock 
    C1 = z_t_shock - z_t_noshock
    C0C1 = np.array([C0, C1]).flatten()
    return C0C1

#* new functions: Task 2.3
def get_zt_2(input_arr, shock = True): 
    '''
    Calculate z_t using guesses for C0, C1 and C2. 
    *input_arr = vector of length 6 containing first guesses
    '''
    #get parameters for calculating A
    params = setup()[1]
    #get inverse of A
    A_inv = A(params)[1]
    #get B
    B = np.array([input_arr[0] + 1/params['sigma'] * input_arr[1], 
                    params['beta'] * input_arr[1], 
                    params['phi1'] * input_arr[1] + params['phi2'] * input_arr[0]])
    #if shock u_t exists, vector (0, 1, 0) must be added to B
    shock_array = np.array([0, int(shock), 0])
    B_shock = B + shock_array
    #now z_t using z_t + inv(A) * B
    z_t = np.dot(A_inv, B_shock)
    return z_t

def get_C0C1C2(input_arr): 
    '''
    Find C0, C1 and C2 using first guesses for them. Note that C0 = inv(A)*B 
    and C1 = inv(A)*(1,0,0)' and C2 = inv(A)*(0, 1, 0)'. Use steps as described in PS. 
    *input_arr = vector of length 9 containing first guesses
    '''
    #first, get C0 by getting z_t with no shocks 
    C0 = get_zt_2(input_arr, shock = False) 
    #now, get C1 by getting z_t with shock e_t = 1 and u_t = 0
    zt_w_et = get_zt(input_arr, shock = True)
    C1 = zt_w_et - C0
    #next up, set u_t = 1 and e_t = 0 
    zt_w_ut = get_zt_2(input_arr, shock = True)
    C2 = zt_w_ut - C0
    #now, bind everything together 
    C0C1C2 = np.array([C0, C1, C2]).flatten()
    
    return C0C1C2

def get_diff(input_arr, one_shock = True): 
    '''
    Function to calculate difference between implied values and 
    values used for expecation.
    
    *input_arr = vector of length 6 containing first guesses if one_shock == True
    OR length 9 necessary if two shocks used (one_shock == False).
    *one_shock = True if model with only one shock, i.e. only shock to output, if False use model 
    with shock to output and inflation.
    '''
    if one_shock: 
        implied = get_C0C1(input_arr)
        diff = input_arr - implied 
    else: 
        implied = get_C0C1C2(input_arr)
        diff = input_arr - implied 
    
    return diff

def solve(start, one_shock = True): 
    '''
    Use starting values supplied and apply fsolve() to get_diff() and
    print results for C0 and C1.
    
    *start = starting values/first guess array for C0 and C1 
    *one_shock = True if model with only one shock, i.e. only shock to output, if False use model 
    with shock to output and inflation.
    '''
    if one_shock:
        result = fsolve(get_diff, start) 
        C0 = result[:3]
        C1 = result[3:]
        print(f'C0: {C0} \n'
                f'C1: {C1}')
    else: 
        result = fsolve(get_diff, start, False) 
        C0 = result[:3]
        C1 = result[3:6]
        C2 = result[6:]
        print(f'C0: {C0} \n'
                f'C1: {C1} \n'
                f'C2: {C2}')

#*##############
#! CALCULATIONS
#*##############

#* Task 2: model with shock to output
#use some random values as starting guess - drawn from a uniform(-1, 1) distribution
start_oneshock = np.random.uniform(low = -1, high = 1, size = 6)
#run solve function, prints results for C0 and C1 
solve(start_oneshock)
'''
C0 is the same as in PS1, i.e. [0, 0, 0]. This should be expected as C0 is the steady 
part of z_t, which we calculated before. Meanwhile, C1 shows the reaction of the variables in z_t 
to the output shock - if there is no reaction to the shock, then the same steady state
should be reached as in the last assignment.
'''

#* Task 3: model with two shocks
#use some random values as starting guess - drawn from a uniform(-1, 1) distribution
start_twoshock = np.random.uniform(low = -1, high = 1, size = 9)
solve(start_twoshock, one_shock = False)