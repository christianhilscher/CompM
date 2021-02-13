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

# Setting up the guessed values
guess = [1,2,3,4]

# Executing the Function with guesses as inputs
diff = Q2(guess)
print("The difference between the guessed and implied values are: ", diff)

# =============================================================================
# # 3.Using Fsolve to find the SS when there is no shock
SteadyState_nocons = fsolve(Q2, guess)
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
guess_F  =np.array([[5],[6],[7],[8]])

# Call function Q4
F, Q = Q4(guess_F)


    
# =============================================================================
# Q5 
