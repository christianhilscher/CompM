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

#* Model and simulations
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

def taylor(E_Yt1, E_pit1, params):
    '''
    Calculate nominal interest rate following Taylor rule.
    *E_Yt1 = expectation of output in t+1 
    *E_pit1 = expectation of inflation in t+1 
    *params = dict of parameters obtained from setup()
    '''
    i = params['phi1'] * E_pit1 + params['phi2'] * E_Yt1
    return i

def IS_curve(E_Yt1, E_pit1, i_t, params):
    '''
    Output in period t following log-linear IS curve.
    *E_Yt1 = expectation of output in t+1 
    *E_pit1 = expectation of inflation in t+1 
    *i_t = nominal interest rate obtained from taylor()
    *params = dict of parameters obtained from setup()
    '''
    Y_t = E_Yt1 - 1/params['sigma'] * (i_t - E_pit1)
    return Y_t

def phillips(E_pit1, Y_t, params): 
    '''
    Inflation according to NK Phillips-Curve. 
    *E_pit1 = expectation of inflation in t + 1
    *E_Yt1 = expectation of output in t + 1
    *params = dict of parameters obtained from setup()
    '''
    pi_t = params['beta'] * E_pit1 + params['kappa'] * Y_t
    return pi_t

def run_simul(T = 1000):
    '''
    Run simluation of T periods.
    '''
    mat, params = setup(T = T)
    i = mat[:, 0]
    pi = mat[:, 1]
    Y = mat[:, 2]
    #start loop at 1 because first value set exogenously in setup
    for t in range(1, T): 
        #static expecations as given
        E_Yt1 = Y[t-1]
        E_pit1 = pi[t-1]
        #now get values for period t applying functions defined before
        i[t] = taylor(E_Yt1, E_pit1, params)
        Y[t] = IS_curve(E_Yt1, E_pit1, i[t], params)
        pi[t] = phillips(E_pit1, Y[t], params)
    mat_final = np.array([i, pi, Y])
    return(mat_final)

#* Matrix functions
def A(params): 
    '''
    Get matrix A as derived in Task 3 and its inverse. 
    *params = dictionary of params obtained from setup()
    '''
    A = np.array([[1, 0, 1/params['sigma']], [-params['kappa'], 1, 0], 
                    [0, 0, 1]])
    A_inv = np.linalg.inv(A)
    return A, A_inv

def get_zt(input_arr): 
    '''
    Task 5.
    Calculate z_t = [Y_t, pi_t, i_t] using only the expectations of output, 
    inflation and nominal interest (last one not needed for calculating z_t but later on).
    A z_t = B <=> z_t = inv(A) * B
    *input_arr = array containing E_Yt1, E_pit1, and E_it1 (can be anything here), 
                note that order of elements must be this one!
    '''
    #get parameters for calculating A
    params = setup()[1]
    #get inverse of A
    A_inv = A(params)[1]
    #get B
    B = np.array([input_arr[0] + 1/params['sigma'] * input_arr[1], 
                    params['beta'] * input_arr[1], 
                    params['phi1'] * input_arr[1] + params['phi2'] * input_arr[0]])
    #now z_t using z_t + inv(A) * B
    z_t = np.dot(A_inv, B)
    return z_t

def get_zt_diff(input_arr): 
    '''
    Task 7. 
    Modified get_zt() such that difference between z_t and inputs is returned. 
    *E_Yt1 = expectation of output in t + 1
    *E_pit1 = expectation of inflation in t + 1
    *E_it1 = expectation of nominal interest in t+1
    '''
    z_t = get_zt(input_arr) 
    diff = input_arr - z_t 
    return diff

#* Plotting functions 
def convergence_plot(T, i, pi, Y): 
    '''
    Plot paths of i, pi and Y in one figure. 
    *T = number of periods simulated
    *i = np.array containing nominal interest rate path 
    *pi = np.array containing inflation rate path 
    *Y = np.array containing output path 
    '''
    x = list(range(0, T))
    fig, axs = plt.subplots(3)
    fig.suptitle('Convergence paths')
    for ax, vals in zip(axs, [i, pi, Y]): 
        ax.plot(x, vals)
    plt.subplots_adjust(hspace = 0.8)
    plt.show()
    plt.savefig(out / 'convergence_paths.png')

#*##############
#! Task 1: SIMULATION OF 1000 PERIODS
#*##############
#now run the sim 
results = run_simul()

#*##############
#! Task 2: PLOT IN MATPLOTLIB
#*##############
i = results[0]
pi = results[1]
Y = results[2]

convergence_plot(1000, i, pi, Y)

#*##############
#! Task 6: Matrix 
#*##############
#what happens if we calculate z_t using convergence values as inputs? 
z_t_converged = get_zt(Y[-1], pi[-1])
#basically nothing happens, we get the "same" value (0 for all here) as 
#we are in the steady state and hence next period is still steady state 

#*##############
#! Task 8: solve for steady state 
#*##############
init = np.array([100, 1, 1])
result = fsolve(get_zt_diff, init)

#*##############
#! Task 9: what solutions do we find?
#*##############
'''
Solutions are again zero, however, more precise you might say as in our simulation the 
model has not fully converged to python displaying the value as 0.0 but rather as very, 
very close to zero.
'''