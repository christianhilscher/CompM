# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 16:01:19 2021
            Assignment 3
    @author: Florian Fickler - 1545713
"""

#Importing packages
import numpy as np
import matplotlib.pyplot as plt

# Setup

# Set Time Dimension
T = 100000

# Define parameters
para = {"sigma" : 2,
        "kappa" : 0.3,
        "beta" : 0.99,
        "phi_1" : 1.5,
        "phi_2" : 0.2,
        "gamma" : 0.05}

# Set up emtpy arrays
y = np.zeros(T)
pi = np.zeros(T)
i = np.zeros(T)
Ey = np.zeros(T)
Epi = np.zeros(T)
Ei = np.zeros(T)

# Set up initial R matrix
R = np.ones((2,2))

#################
####** 1.1 **####
#################

# Initalize 2x3 array with random values from uniform [0,1) distribution
np.random.seed(1594)
C_hat = np.random.rand(2,3)

#################
####** 1.2 **####
#################

# 1D Array with shocks
epsilon = np.random.normal(size = T)

#################
####** 1.3 **####
#################

for t in range(0,T-1):
    #First, define expectations from C_hat
    Ey[t+1], Epi[t+1], Ei[t+1] = C_hat[0, :]
    
    #Then calculate economic outcomes based on expectations
    i[t] = para["phi_1"] * Epi[t+1] + para["phi_2"]* Ey[t+1]
    y[t] = Ey[t+1] - (1 / para["sigma"]) * (i[t]-Epi[t+1]) + epsilon[t]
    pi[t] = para["beta"] * Epi[t+1] + para["kappa"] * y[t]
    
    # Update R Matrix
    v = np.array([[1],[epsilon[t]]])
    R += para["gamma"] * (np.dot(v, np.transpose(v)) - R)
    
    # Update C_hat
    z = np.array([y[t], pi[t], i[t]])
    
    C_hat += para["gamma"] * np.dot(np.linalg.inv(R), np.dot(v, z -np.dot(np.transpose(v), C_hat)))
    

#################
####** 1.4 **####
#################

print("Values contained in C_0_hat: ", C_hat[0, :])
print("Values contained in C_1_hat: ", C_hat[1, :])

"""
Comparing the final C_hat matrix, we can see, that all values included in C_0_hat are sufficiently close to zero, and that the values in C_1_hat are 1, 0.3 and 0 respectively. This is identical to the MSV solution we calculated in the last assignment. Therefore, we can conclude, that using this learning algorithm, agents are able to learn the MSV solution.
"""

#################
####** 1.5 **####
#################

plt.figure("Assignment 3 - Question 1.5")

plt.subplot(6,1,1)
plt.title("Output")
plt.xlabel("t")
plt.ylabel(r"$y$")
plt.plot([y[t] for t in range(T)])

plt.subplot(6,1,2)
plt.title("Expected Output")
plt.xlabel("t")
plt.ylabel(r"$E(y)$")
plt.plot([Ey[t] for t in range(T)])

plt.subplot(6,1,3)
plt.title("Inflation")
plt.xlabel("t")
plt.ylabel(r"$\pi$")
plt.plot([pi[t] for t in range(T)])

plt.subplot(6,1,4)
plt.title("Expected Inflation")
plt.xlabel("t")
plt.ylabel(r"$E(\pi)$")
plt.plot([Epi[t] for t in range(T)])

plt.subplot(6,1,5)
plt.title("Nominal Interest Rate")
plt.xlabel("t")
plt.ylabel(r"$i$")
plt.plot([i[t] for t in range(T)])

plt.subplot(6,1,6)
plt.title("Output Shock")
plt.xlabel("t")
plt.ylabel(r"$\epsilon$")
plt.plot([epsilon[t] for t in range(T)])

plt.tight_layout()

#################
####** 1.6 **####
#################

"""
The shock to output which we introduced to the model impacts both the output and the inflation rate of the economy directly. The latter less strongly (psi_epsilon,pi = 0.3) compared to the first (psi_epsilon,y= 1). Therefore, neither of them converges to any value, since the shocks do not converge.
Expectiations, however, are based on the realized previous values, which depends on the shock, as well as the learning algorithm. Due to the learning algorithm, the percieved law of motion converges to the actual law of motion additional information is accumulated by the agent in each each period. We can see that in the agents expectations for output and inflation which converge to the MSV of these values without a shock, since the shock is drawn from a standard normal distribution with mean zero.
As Expectations converge, so does the nominal interest rate, as it is only a function of parameters and expectations of future realizations and therefore unaffected by the shock.
Further, we can observe, that the shocks to our economy do not have a visible impact on our expectations after a short time period. Therefore, one could say, that agents learn the ALM relatively quickly and then stick to it. 
"""