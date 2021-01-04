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

#* simulation functions 

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

def taylor(E_zt1, params):
    '''
    Calculate nominal interest rate following Taylor rule.
    *E_zt1 = E[z_t+1] where z_t = [Y_t, pi_t, i_t]
    *params = dict of parameters obtained from setup()
    '''
    i = params['phi1'] * E_zt1[1] + params['phi2'] * E_zt1[0]

    return i

def IS_curve(E_zt1, i_t, shock, params):
    '''
    Output in period t following log-linear IS curve.
    *E_zt1 = E[z_t+1] where z_t = [Y_t, pi_t, i_t]
    *i_t = nominal interest rate obtained from taylor()
    *params = dict of parameters obtained from setup()
    '''
    Y_t = E_zt1[0] - 1/params['sigma'] * (i_t - E_zt1[1]) + shock

    return Y_t

def phillips(E_zt1, Y_t, params): 
    '''
    Inflation according to NK Phillips-Curve. 
    *E_zt1 = E[z_t+1] where z_t = [Y_t, pi_t, i_t]
    *params = dict of parameters obtained from setup()
    '''
    pi_t = params['beta'] * E_zt1[1] + params['kappa'] * Y_t

    return pi_t

def get_zt(params, C0C1, shock): 
    '''
    Calculate z_t = [Y_t, pi_t, i_t]. 
    
    *params = dict of parameters obtained from setup() 
    *C0C1 = 2x3 matrix containing C_0^hat and C_1^hat in first and second row, respectively
    *shocks = shock in period t, necessary for Y_t
    '''
    #first, need E[z_t+1] which is simply C_0^hat 
    E_zt1 = C0C1[0, :]
    i_t = taylor(E_zt1, params)
    Y_t = IS_curve(E_zt1, i_t, shock, params)
    pi_t = phillips(E_zt1, Y_t, params)
    #now, bind everything together in one vector 
    z_t = np.array([Y_t, pi_t, i_t])
    
    return z_t

def updating_R(R_old, shock, gamma = 0.05):
    '''
    Learning rule for moment matrix R. 
    
    *R_old = previous R
    *shock = value of e_t 
    *gamma = gain parameter of learning mechanism, set to 0.05 as default
    '''
    #first, get v (2x1 array with (1, shock))
    v = np.array([[1], [shock]])
    #now update R
    R_new = R_old + gamma * (v @ np.transpose(v) - R_old)
    
    return R_new

def updating_C(C_old, R_new, shock, z, gamma = 0.05):
    '''
    Learning rule for C_hat. 
    
    *C_old = previos C_hat
    *shock = value of e_t
    *z = previous realization of z in current period
    *gamma = gain parameter of learning mechanism, set to 0.05 as default
    '''
    #first, get v (2x1 array with (1, shock))
    v = np.array([[1], [shock]])
    #now get inv(R_new)
    R_new_inv = np.linalg.inv(R_new)
    #get C_new 
    C_new = C_old + gamma * R_new_inv @ v @ (z - np.transpose(v) @ C_old)
    
    return C_new

def simul(T = 100000): 
    
    #get everything for setup: first guess for C^hat, shock array and paramaters
    first_guess, shocks, params = setup(T)
    #set up a 3d array to save C^hat each period
    C_hats = np.empty((T, 2, 3))
    #set fierst matrix in C_hat to be first guess 
    C_hats[0, :, :] = first_guess 
    #now set up the same for the moment matrix R 
    Rs = np.empty((T, 2, 2))
    #and again set first element to initial values (here every element is 1)
    Rs[0, :, :] = np.array(([1, 1], [1, 1]))
    #finally, set up a 3d array to save the z_t each period 
    zs = np.empty((T, 3))
    #now the loop
    for t in range(T): 
        #first, calculate z_t using C^hat's t^th element and shock in t
        z_t = get_zt(params, C_hats[t, :, :], shocks[t])
        zs[t, :] = z_t
        #if in last period, do not update C^hat and R anymore, i.e. stop loop
        if t+1 == T: 
            break
        #now, update R and save it in array
        Rs[t+1, :, :] = updating_R(Rs[t], shock = shocks[t])
        #and updating C^hat and saving it in array 
        C_hats[t+1, :, :] = updating_C(C_hats[t], Rs[t+1], shocks[t], z_t)
    #save the series as single arrays 
    #output 
    Y = zs[:, 0]
    #inflation
    pi = zs[:, 1]
    #interest rate
    i = zs[:, 2]
    
    #return shocks as well as we need them for plotting later on
    return Y, pi, i, C_hats, shocks

#* Plotting functions

def plot(series, T, subtitles = None, title = 'Convergence Paths'):
    '''
    Plot all series in series over T periods in subplots in one figure, optionally 
    giving every subplot a title.
    
    *series = list of series to plot in subplots
    *T = number of periods to plot
    *subtitles = list of titles to give subplots (in same order as series)
    *title = title for the figure
    '''
    #if no subtitles provided, make list with None to be able to loop over them
    if subtitles == None: 
        subtitles = [None] * len(series)
    #generate x-axis (same for each subplot)
    x = list(range(0, T))
    #set up subplots needed
    fig, axs = plt.subplots(len(series), figsize = (30, 30))
    #set title for overall figure 
    fig.suptitle(title, fontsize = 30)
    #now generate subplots 
    for ax, vals, sub in zip(axs, series, subtitles):
        #for each element in series add an axis to the figure and give it the title supplied in subtitles
        ax.plot(x, vals)
        ax.set_title(sub, fontsize = 30)
    plt.subplots_adjust(hspace = 0.8)
    plt.savefig(out / 'convergence_paths.png')
    plt.show()



#*##############
#! CALCULATIONS
#*##############

T = 100000
#* Task 3: run simulation 
Y, pi, i, C_hats, shocks = simul(T = T)

#* Task 4: does solution differ from MSV solution? 
finalC0C1 = C_hats[T-1]
'''
Indeed, the matrix C^hat is the same as the MSV solution in Assignment 2 for C0 and C1. 
The steady state is C0 = [0, 0, 0] in each element and C1 = [1, 0.3, 0].
'''

#* Task 5: Plots
#we are supposed to plot expected (!) output and inflation, which are simply C_0^hat from period t
#then first element in each E[z_t+1] is E[Y_t+1] and second one is E[pi_t+1]
E_Yt1 = C_hats[:, 0, 0]
E_pit1 = C_hats[:, 0, 1]

series_list = [Y, pi, i, E_Yt1, E_pit1, shocks]
subtitles_list = ['Output', 'Inflation', 'Nominal Interest Rate', 'Expected Output', 'Expected Inflation', 
                    'Shocks to Output']
plot = plot(series_list, T, subtitles_list)