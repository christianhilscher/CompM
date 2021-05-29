# -*- coding: utf-8 -*-
"""
Created on Sat Feb  13 2021
            Final Assignment
    @author: Florian Fickler - 1545713
"""

# Importing packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd
from statsmodels.formula.api import ols

# =============================================================================
# Define parameters in a dictionary
# The dictionary allows to access this values later via their keys
para = {"sigma": 1,
        "kappa": 0.3,
        "beta": 0.995,
        "phi_pi": 1.5,
        "phi_y": 0.1,
        "rho_mu": 0.7}


# =============================================================================
# 1. Define Matrices A,M,D
# Each line defines one matrix by entering the respective values
# Values are either fixed or taken from the parameters dictionary above.
# Matrices are defined as derived  by the model
# The matrices are defined globaly
# Therefore, when they are needed in later functions, they can always be called

A = np.array([[1, 0, 0, 0],
              [0, 1, 0, 1/para["sigma"]],
              [-1, -para["kappa"], 1, 0],
              [0, 0, 0, 1]])

M = np.array([[0, 0, 0, 0],
              [0, 1, 1/para["sigma"], 0],
              [0, 0, para["beta"], 0],
              [0, 0, 0, 0]])

D = np.array([[para["rho_mu"], 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, para["phi_y"], para["phi_pi"], 0]])


# =============================================================================
# 2. Function for Calculating z_t
def Q2(guess):
    # the input guess, is the first (and exogenous) guess about the realizations in the first period
    # Setting the expectations and the previous levels to the guessed values
    Et_zt = guess
    ztminus1 = guess

    # Setting the Error term to zero, e.g. assuming, that there is no shock
    u_t = 0

    # Calculating the RHS of Eq6
    RHS6 = M@Et_zt + D@ztminus1 + u_t

    # Calculating z_t as implied by Eq.6
    # For this the matrix calculations in numpy are used
    z_t = np.linalg.inv(A) @ RHS6

    # Calculating the difference from the guessed value and the actual value
    diff = z_t - guess

    # returning the difference
    return diff


# Executing the Function with guesses as inputs
# My guesses are simply the integers 1-4. As in the model without a shock, there is only one SS (minimum state variable solution), the values of the guesses should not matter for the result.
diff = Q2([1, 2, 3, 4])
print("The difference between the guessed and implied values are: ", diff)


# =============================================================================
# # 3.Using Fsolve to find the SS when there is no shock
# F-Solve finds the root of a function.
# In our case, Q2 returns the difference between a periods guess and its realizations.
# If the guess equals the realization, Q2 returns 0, i.e., there is no difference.
# F-Solve will no look for the values of inital guesses, for which this is the case.
# If the difference is zero, than our guess and our realization are the same, and our model is in a SS.
SteadyState_nocons = fsolve(Q2, [1, 2, 3, 4])
print("The SS without any shock is: ", SteadyState_nocons)


# =============================================================================
# 4.
def Q4(F_guess):
    # A,M,D are defined globally

    # Intialize the new F once
    F_new = np.zeros((4, 4))

    # Run a While loop, comparing the guessed value to the new value
    # If they are "sufficiently close to each other, then exit the loop
    while np.max(np.abs(F_new - F_guess)) > 1e-8:
        # the absolute function from numpy is used to evalute the while loop
        # Assign the previous(initial) value as the current guess
        F_guess = F_new
        # Calculate the new F based on the current guess for F
        # The formula is equal to the hint provided in the question
        F_new = np.linalg.inv(A - M@F_guess) @ D
        # Then reevaluate the difference between the newly estimated F(F_new) and the previous F(F_guess)
        # If the difference becomes small enough, the while loop is exited.
        # If the difference is still larger then 1e-8, then the realization (F_new) is again assigned as the guess and the procedure is exectued again.
        # This itterative process is used, to find the SS values of our Model.
        # By evaluating if the difference between the last realization is sufficiently close to the following realization, we can claim, that those values indeed are SS values.
        # Thats why we first calculate the outcome of our model based on the previous value (or guess) and then determine the difference between what our inputs and the output they produce.
        # Alternatively, one could again use the F-Solve function to find the

    # If F has converged, calculate Q
    # Again, this formula comes from the hint.
    Q = np.linalg.inv(A - M @ F_new)

    # return the converged F and Q
    return F_new, Q


# =============================================================================
# 5.

# Call function Q4
# Function needs some intial guesses, I simply keep using the next integers 5-8.
F, Q = Q4(np.array([[5], [6], [7], [8]]))

# General Solution
# z_t = F*z_tminus1 + Q * u_t


# =============================================================================
# 6.
# Saving the corrosponding vectors of the MSV solution
# z_t = C_mu*mu_neg1 + C_y*y_tneg1 + C_pi*pi_tneg1 + C_eps*eps_t^mu + C_eps_i*eps_t^i
# Vectors for the lagged variables are stored in each column of F
C_mu = F[:, 0]  # whole columns are accessed by using : to access all rows for a given column
C_y = F[:, 1]
C_pi = F[:, 2]

# Those for the white noise terms are stored in Q
C_eps_mu = Q[:, 0]
C_eps_i = Q[:, 3]


# =============================================================================
# 7. Calculating Impulse Responses for epsilon_mu
# First set up number of periods
N = 30
# And arrays for the noise
# The arrays are filled with zeros and shocks will be added later
epsilon_mu1 = np.zeros(N)
epsilon_i1 = np.zeros(N)

# Also set up arrays for our state variables, again all values are set to zero, the SS of the model.
mu1 = np.zeros(N)
y1 = np.zeros(N)
pi1 = np.zeros(N)
i1 = np.zeros(N)

# Insert Impulse into the noise array
epsilon_mu1[0] = 0.01

# for loop to go through all 30 periods
# The loop uses the above defined 'C-vectors' of the MSV to calculate the relaizations of each variable for each period.
# Going through each period seperately, first realizations are calculated
# Then those realizations are used to calculate the following periods values.
for j in range(N):
    mu1[j], y1[j], pi1[j], i1[j] = [C_mu[hh]*mu1[j-1] +
                                    C_y[hh]*y1[j-1] +
                                    C_pi[hh]*pi1[j-1] +
                                    C_eps_mu[hh]*epsilon_mu1[j] +
                                    C_eps_i[hh]*epsilon_i1[j]
                                    for hh in range(4)]


# =============================================================================
# 8. Calculating Impulse Responses for epsilon_i
# The periods N from above are used

# Arrays for the noise are again set to zero
epsilon_mu2 = np.zeros(N)
epsilon_i2 = np.zeros(N)

# As well as the arrays for our state variables
mu2 = np.zeros(N)
y2 = np.zeros(N)
pi2 = np.zeros(N)
i2 = np.zeros(N)

# Insert Impulse into noise array
epsilon_i2[0] = 0.01

# For loop to go through 30 periods and calculate the realizations for each
# This loop works the same way, as the loop in Q7 did.
for j in range(N):
    mu2[j], y2[j], pi2[j], i2[j] = [C_mu[hh]*mu2[j-1] +
                                    C_y[hh]*y2[j-1] +
                                    C_pi[hh]*pi2[j-1] +
                                    C_eps_mu[hh]*epsilon_mu2[j] +
                                    C_eps_i[hh]*epsilon_i2[j]
                                    for hh in range(4)]


# =============================================================================
# 9. Creating Figures
# Figure 1 for the shock from Q7
# First open up a figure
plt.figure("Impulse Response Epsilon_Mu")  # Figure Title

# Then fill it with the corrosponding subplots
plt.subplot(4, 1, 1)  # Position of Subplot within Figure
plt.title("Inflation Shock")  # Subplot Title
plt.xlabel("t")  # x label
plt.ylabel(r"$\mu$", rotation=0, fontsize='large')  # rotated y label
plt.plot([mu1[t] for t in range(N)], linewidth=4, color='red')  # plot data and line properties
plt.grid(which='major', axis='both')  # adding gride lines

plt.subplot(4, 1, 2)
plt.title("Output")
plt.xlabel("t")
plt.ylabel(r"$y$", rotation=0, fontsize='large')
plt.plot([y1[t] for t in range(N)], linewidth=4, color='blue')
plt.grid(which='major', axis='both')

plt.subplot(4, 1, 3)
plt.title("Inflation")
plt.xlabel("t")
plt.ylabel(r"$\pi$", rotation=0, fontsize='large')
plt.plot([pi1[t] for t in range(N)], linewidth=4, color='brown')
plt.grid(which='major', axis='both')

plt.subplot(4, 1, 4)
plt.title("Nominal Interest Rate")
plt.xlabel("t")
plt.ylabel(r"$i$", rotation=0, fontsize='large')
plt.plot([i1[t] for t in range(N)], linewidth=4, color='black')
plt.grid(which='major', axis='both')

plt.tight_layout()  # Alter the layout for better fit

# Figure 2 for the shock from Q8
plt.figure("Impulse Response Epsilon_I")

plt.subplot(4, 1, 1)
plt.title("Inflation Shock")
plt.xlabel("t")
plt.ylabel(r"$\mu$", rotation=0, fontsize='large')
plt.plot([mu2[t] for t in range(N)], linewidth=4, color='red')
plt.grid(which='major', axis='both')

plt.subplot(4, 1, 2)
plt.title("Output")
plt.xlabel("t")
plt.ylabel(r"$y$", rotation=0, fontsize='large')
plt.plot([y2[t] for t in range(N)], linewidth=4, color='blue')
plt.grid(which='major', axis='both')

plt.subplot(4, 1, 3)
plt.title("Inflation")
plt.xlabel("t")
plt.ylabel(r"$\pi$", rotation=0, fontsize='large')
plt.plot([pi2[t] for t in range(N)], linewidth=4, color='brown')
plt.grid(which='major', axis='both')

plt.subplot(4, 1, 4)
plt.title("Nominal Interest Rate")
plt.xlabel("t")
plt.ylabel(r"$i$", rotation=0, fontsize='large')
plt.plot([i2[t] for t in range(N)], linewidth=4, color='black')
plt.grid(which='major', axis='both')

plt.tight_layout()


# =============================================================================
# 10. Interpretation of the Impulse Response

# Please see PDF for the answers to Q10.

# =============================================================================
# 11. Creating Figures
# Change the number of periods to 500
N = 500

# Set seed for radnom draws
np.random.seed(1594)

# Draw random realizations for both shocks
epsilon_mu3 = np.random.rand(N)
epsilon_i3 = np.random.rand(N)

# Initialize empty arrays for the state variables
mu3 = np.zeros(N)
y3 = np.zeros(N)
pi3 = np.zeros(N)
i3 = np.zeros(N)

# for loop to go through 30 periods
# This loop works as described in Q7
for j in range(N):
    mu3[j], y3[j], pi3[j], i3[j] = [C_mu[hh]*mu3[j-1] +
                                    C_y[hh]*y3[j-1] +
                                    C_pi[hh]*pi3[j-1] +
                                    C_eps_mu[hh]*epsilon_mu3[j] +
                                    C_eps_i[hh]*epsilon_i3[j]
                                    for hh in range(4)]

# After all values are calculated, transfer the Output and Inflation realizations into a pandas dataframe
# Using list and zip allows for easy transfer of the two in arrays saved variables to a df
# Column names are added as an argument as well
df = pd.DataFrame(list(zip(y3, pi3)), columns=['Output', 'Inflation'])


# =============================================================================
# 12. Calculating Inflation Expectations
# Advancing z_t to z_t+1 and taking expectations yields:
# EZ_t+1 =C_mu*mu_t C_y*y_t + C_pi*pi_t

# Inizialize empty array for inflation expectations
E_pi = np.zeros(N)

# Calculate expectations via for loop
# For this the expectations of z_t+1 are used and the corresponding columns of the C matrix are used to only calculate Expectations for Inflation.
for j in range(N):
    E_pi[j] = C_mu[2]*mu3[j] + C_y[2]*y3[j] + C_pi[2]*pi3[j]

# Add expectations to the dataframe
df['Inflation_Expectations'] = pd.DataFrame(E_pi)


# =============================================================================
# 13. Calculating the Forecast Error
# Inflation has to be shifted, so that we use pi_t+1
# The forecast error is then saved as a new collumn in the dataframe
df['fc_error'] = df['Inflation'].shift(-1) - df['Inflation_Expectations']


# =============================================================================
# 14. Regression
# Define OLS function
# Using the statsmodel package, one can write R-Style formuals.
# Here an interecept is included even though invisible
# ols simply sets up an OLS regression for our defined model/formula
# Data is drawn from the above defined dataframe df
reg = ols(formula='fc_error ~ Inflation', data=df)

# The model set up above is no fitted with robust SE
fit = reg.fit(cov_type='HAC', cov_kwds={'maxlags': 1})

# Print summary statistics for our fitted model
print(fit.summary())

# Interpretation

# Please see PDF for this part of Q14.
