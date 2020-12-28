import numpy as np 
import os 
import matplotlib.pyplot as plt

#*##############
#! FUNCTIONS
#*##############

#* Simulations
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
    
    returns: nominal interest rate i 
    '''
    i = params['phi1'] * E_Yt1 + params['phi2'] * E_pit1
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
def convergence_plot(T, i, pi, Y): 
    '''
    Plot paths of i, pi and Y in one figure. 
    *T = number of periods simulated
    *i = np.array containing nominal interest rate path 
    *pi = np.array containing inflation rate path 
    *Y = np.array containing output path 
    '''
    T = 1000
    x = list(range(0, T))
    fig, axs = plt.subplots(3)
    fig.suptitle('Convergence paths')
    for ax, vals in zip(axs, [i, pi, Y]): 
        ax.plot(x, vals)
    plt.subplots_adjust(hspace = 0.8)
    plt.show()