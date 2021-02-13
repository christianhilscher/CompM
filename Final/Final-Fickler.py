# -*- coding: utf-8 -*-
"""
Created on Sat Feb  13 2021
            Final Assignment
    @author: Florian Fickler - 1545713
"""

#Importing packages
import numpy as np
import matplotlib.pyplot as plt
from  scipy.optimize import fsolve

# =============================================================================
# Define parameters in a dictionary
para = {"sigma" : 1,
        "kappa" : 0.3,
        "beta" : 0.995,
        "phi_pi" : 1.5,
        "phi_y" : 0.1,
        "rho_mu" : 0.7}

# =============================================================================
# 1. Define Matrices A,M,D
# Each line defines one matrix by entering the respective values that are either fixed or taken from the parameters dictionary above.
# The matrices are defined globaly, therefore, when they are needed in later functions, they do not need to be called as inputs.
A = np.array([[1,0,0,0],
              [0,1,0,1/para["sigma"]],
              [-1,-para["kappa"], 1, 0],
              [0,0,0,1]])

M = np.array([[0,0,0,0],
              [0,1,1/para["sigma"],0],
              [0,0,para["beta"],0],
              [0,0,0,0]])

D = np.array([[para["rho_mu"],0,0,0],
              [0,0,0,0],
              [0,0,0,0],
              [0, para["phi_y"],para["phi_pi"],0]])

# =============================================================================
# 2. Function for Calculating z_t
def Q2(guess):    
    # Setting the expectations and the previous levels to the guessed values
    Et_zt = guess
    ztminus1 =  guess
    
    # Setting the Error term to zero, e.g. assuming, that there is no shock
    u_t = 0
    
    # Calculating the RHS of Eq6
    RHS6 = M@Et_zt + D@ztminus1 + u_t
    
    # Calculating z_t as implied by Eq.6
    z_t = np.linalg.inv(A) @ RHS6
    
    #Calculating the difference from the guessed value and the actual value
    diff = z_t - guess
    
    # returning the difference
    return diff

# Executing the Function with guesses as inputs
diff = Q2([1,2,3,4])
print("The difference between the guessed and implied values are: ", diff)

# =============================================================================
# # 3.Using Fsolve to find the SS when there is no shock
SteadyState_nocons = fsolve(Q2, [1,2,3,4])
print("The SS without a Shock is: ", SteadyState_nocons)

# =============================================================================
# 4.
def Q4(F_guess):
    # A,M,D are defined globally 
    
    # Calculate the new F once
    F_new = np.zeros((4,4))
      
    # Run a While loop, comparing the guessed value to the new value
    # If they are "sufficiently close to each other, then exit the loop 
    
    while np.max(np.abs(F_new - F_guess)) > 1e-8:
        # Assign the new value as the new guess
        F_guess = F_new        
        # Calculate the new F again based on the altered guess
        F_new = np.linalg.inv(A - M@F_guess) @ D
        # Then reevaluate the difference between the newly estimated F and the previously estimated F
        # If the difference becomes small enough, the while loop is exited
           
    # If F has converged, calculate Q    
    Q = np.linalg.inv(A - M @ F_new)
    
    # return the converged F and Q
    return F_new, Q 

# =============================================================================
# 5.
# Set up some guess values for F


# Call function Q4
F, Q = Q4(np.array([[5],[6],[7],[8]]))

# General Solution
# z_t = F*z_tminus1 + Q * u_t

# =============================================================================
# 6.
# Saving the corrosponding vectors for the MSV solution
# z_t = C_mu*mu_neg1 + C_y*y_tneg1 + C_pi*pi_tneg1 + C_eps*eps_t^mu + C_eps_i*eps_t^i
# Vectors for the lagged variables are stored in each column of F
C_mu = F[:, 0]
C_y = F[:, 1]
C_pi = F[:, 2]
C_i = F[:, 3]

# Those for the white noise terms are stored in Q
C_eps_mu = Q[:, 0]
C_eps_i = Q[:, 3]

# =============================================================================
# 7. Calculating Impulse Responses for epsilon_mu
# First set up number of periods and empty arrays
N = 30
# For the Noise
epsilon_mu1 = np.zeros(N)
epsilon_i1 = np.zeros(N)
                     
# For the variables
mu1 = np.zeros(N)
y1 = np.zeros(N)
pi1 = np.zeros(N)
i1 = np.zeros(N)

# Insert Impulse
epsilon_mu1[0] =  0.01                 

# for loop to go through 30 periods
for j in range(N):
    mu1[j],y1[j],pi1[j],i1[j] = [C_mu[hh]*mu1[j-1] + 
                             C_y[hh]*y1[j-1] + 
                             C_pi[hh]*pi1[j-1] + 
                             C_i[hh]*i1[j-1] + 
                             C_eps_mu[hh]*epsilon_mu1[j] + 
                             C_eps_i[hh]*epsilon_i1[j]                               
                               for hh in range(4)]

# =============================================================================
# 8. Calculating Impulse Responses for epsilon_i
# First set up number of periods and empty arrays
N = 30
# For the Noise
epsilon_mu2 = np.zeros(N)
epsilon_i2 = np.zeros(N)
                     
# For the variables
mu2 = np.zeros(N)
y2 = np.zeros(N)
pi2 = np.zeros(N)
i2 = np.zeros(N)

# Insert Impulse
epsilon_i2[0] =  0.01                 

# for loop to go through 30 periods
for j in range(N):
    mu2[j],y2[j],pi2[j],i2[j] = [C_mu[hh]*mu2[j-1] + 
                             C_y[hh]*y2[j-1] + 
                             C_pi[hh]*pi2[j-1] + 
                             C_i[hh]*i2[j-1] + 
                             C_eps_mu[hh]*epsilon_mu2[j] + 
                             C_eps_i[hh]*epsilon_i2[j]                               
                               for hh in range(4)]

# =============================================================================
# 9. Creating Figures

#Figure 1 for the shock from Q7
plt.figure("Impulse Response Epsilon_Mu")

plt.subplot(4,1,1)
plt.title("Inflation Shock")
plt.xlabel("t")
plt.ylabel(r"$\mu$")
plt.plot([mu1[t] for t in range(N)], linewidth=4, color='red')
plt.axis('auto')

plt.subplot(4,1,2)
plt.title("Output")
plt.xlabel("t")
plt.ylabel(r"$y$")
plt.plot([y1[t] for t in range(N)], linewidth=4, color='blue')

plt.subplot(4,1,3)
plt.title("Inflation")
plt.xlabel("t")
plt.ylabel(r"$\pi$")
plt.plot([pi1[t] for t in range(N)], linewidth=4, color='brown')

plt.subplot(4,1,4)
plt.title("Nominal Interest Rate")
plt.xlabel("t")
plt.ylabel(r"$i$")
plt.plot([i1[t] for t in range(N)], linewidth=4, color='black')

plt.tight_layout()

#Figure 1 for the shock from Q8
plt.figure("Impulse Response Epsilon_I")

plt.subplot(4,1,1)
plt.title("Inflation Shock")
plt.xlabel("t")
plt.ylabel(r"$\mu$")
plt.plot([mu2[t] for t in range(N)], linewidth=4, color='red')
plt.axis('auto')

plt.subplot(4,1,2)
plt.title("Output")
plt.xlabel("t")
plt.ylabel(r"$y$")
plt.plot([y2[t] for t in range(N)], linewidth=4, color='blue')

plt.subplot(4,1,3)
plt.title("Inflation")
plt.xlabel("t")
plt.ylabel(r"$\pi$")
plt.plot([pi2[t] for t in range(N)], linewidth=4, color='brown')

plt.subplot(4,1,4)
plt.title("Nominal Interest Rate")
plt.xlabel("t")
plt.ylabel(r"$i$")
plt.plot([i2[t] for t in range(N)], linewidth=4, color='black')

plt.tight_layout()