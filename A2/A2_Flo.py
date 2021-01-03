# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 18:47:03 2021

@author: Florian Fickler - 1545713
"""

## Assignment 2 ##

#Importing
import numpy as np
from  scipy.optimize import fsolve

# set up the parameters
sigma = 2
kappa = 0.3
beta = 0.99
phi_1 = 1.5
phi_2 = 0.2

# Set up time periods
T = 1000

# Setting up Arrays
y = np.zeros(T)
pi = np.zeros(T)
i = np.zeros(T)
e = np.zeros(T)

########
# Task #
########

# 1.1

A = np.array([[1,0,1/sigma],
              [-kappa, 1, 0],
              [0, 0, 1]])

A_inv = np.linalg.inv(A)

def output_shock(inputs):
    # unpack inputs
    EY, Epi, Ei = np.array(inputs[:3])
    rho1, rho2, rho3 = np.array(inputs[3:])
           
    # Def B
    B = np.array([EY + 1/sigma *Epi, beta * Epi, phi_1* Epi + phi_2 * EY])
    
    # A * z = B + e
    # z = C_0 + C_1 * e
    
    # For e = 0
    # z = C_0 = A_inv * B
    C_0 = np.dot(A_inv, B)
    
    #For e = 1
    # z = C_0 + C_1
    # z = A_inv(B+[e,0,0])
    B_prime = B + np.array([1,0,0])
    C_1 = np.dot(A_inv, B_prime) - C_0
    
    # Calculate the differences under rational expectations
    Diff = np.hstack((C_0, C_1)) - inputs
    
    return Diff

    
# 1.2

# Setting up inital values
Intials_6 = [0.8, 0.1, 0.7, 0.3, 0.2, 0.5]
# Alternative with random numbers
#Intial_ = np.random.uniform(low=-1, high=1, size=6)


Y_Schock_Converge = fsolve(output_shock, Intials_6)
print("These are the valuies for C_0:", Y_Schock_Converge[:3])
print("These are the valuies for C_1:", Y_Schock_Converge[3:])

# Explain why they differ? Do they differ? All basically zero, others are closer however


# 1.3

def output_inflation_shock(inputs):
    # unpack inputs
    EY, Epi, Ei = np.array(inputs[:3])
    rho1, rho2, rho3 = np.array(inputs[3:6])
    eta1, eta2, eta3 = np.array(inputs[6:])
           
    # Def B
    B = np.array([EY + 1/sigma *Epi, beta * Epi, phi_1* Epi + phi_2 * EY])
    
    # A * z = B + e + n
    # z = C_0 + C_1 * e + C_2 * n
    
    # For e = 0 and n = 0
    # z = C_0 = A_inv * B
    C_0 = np.dot(A_inv, B)
    
    #For e = 1 and n = 0
    # z = C_0 + C_1
    # z = A_inv(B+[1,0,0])
    B_prime_1 = B + np.array([1,0,0])
    C_1 = np.dot(A_inv, B_prime_1) - C_0
    
    #For n = 1 and e = 0
    # z = C_0 + C_2
    # z = A_inv(B+[0,1,0])
    B_prime_2 = B + np.array([0,1,0])
    C_2 = np.dot(A_inv, B_prime_2) - C_0
    
    # Calculate the differences under rational expectations
    Diff = np.hstack((C_0, C_1, C_2)) - inputs
    
    return Diff

# Setting up inital values
Intials_9 = [0.8, 0.1, 0.7, 0.3, 0.2, 0.5, 0.3, 0.2, 0.5]
# Alternative with random numbers
#Intial = np.random.uniform(low=-1, high=1, size=6)


Y_pi_Schock_Converge = fsolve(output_inflation_shock, Intials_9)
print("These are the valuies for C_0:", Y_pi_Schock_Converge[:3])
print("These are the valuies for C_1:", Y_pi_Schock_Converge[3:6])
print("These are the valuies for C_2:", Y_pi_Schock_Converge[6:])






























