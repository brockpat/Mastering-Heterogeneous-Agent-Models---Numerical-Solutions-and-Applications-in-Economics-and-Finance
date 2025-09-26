# -*- coding: utf-8 -*-
"""
Solves the SIMM using the EGM.

The Rouwenhort Method is taken from 
    Rognlie (2022): https://github.com/shade-econ/nber-workshop-2022/blob/main/Lectures/Lecture%201%20Standard%20Incomplete%20Markets%20Steady%20State.ipynb
"""

#%% Set Path
directory = ".../Code/"

import os
os.chdir(directory)

#%% Libraries
import numpy as np
import matplotlib.pyplot as plt
# Set global font size
plt.rcParams.update({'font.size': 12})

import scipy

#Functions that implement the Rouwenhorst Method
import Rouwenhorst

#%% Set Parameters

#risk-free rate
r = 0.01
#Discount Factor
beta = 0.95
#Risk Aversion
gamma = 2

#%% Utility Function 
def U(c):
    """
    Computes the utility of consumption 'c'.
    """
    #Log Utility if gamma == 1
    if gamma == 1:
        utility = np.log(c)
    else:
        #Standard CRRA Utility
        utility = c**(1-gamma)/(1-gamma)
    return utility

def Uprime(c):
    """
    Marginal utility
    """
    marginal_utility = c**(-gamma)
    return marginal_utility

def Uprime_Inv(c):
    """
    Inverse of marginal utility
    """
    value = c**(-1/gamma)
    return value

#%% Grid: Assets a'

def discretize_assets(amin, amax, n_a):
    """
    Computes the asset grid in a non-uniform manner. Gridpoints are denser
    closer to a_min.
    
    Args
    ----------
    amin : (float)
        the minimum value of the asset grid.
    amax : (float)
        the maximum value of the asset grid.
    N : (int)
        the number of points in the discretized asset grid.

    Returns
    -------
    a_grid : 1D Array with dimension (N,1)
        Asset grid.

    """
    # find maximum ubar of uniform grid corresponding to desired maximum amax of asset grid
    ubar = np.log(1 + np.log(1 + amax - amin))
    
    # make uniform grid
    u_grid = np.linspace(0, ubar, n_a)
    
    # double-exponentiate uniform grid and add amin to get grid from amin to amax
    return amin + np.exp(np.exp(u_grid) - 1) - 1

#Grid for a'
a_grid = discretize_assets(0,80,300)

#%% Grid: Income

#Rouwenhorst Method to get grid 'z_grid', 
#   stationary distribution 'pi' and 
#   Markov Transition Matrix 'Pi'.
z_grid, pi, Pi = Rouwenhorst.discretize_income(0.6, #rho 
                                               0.5, #Variance of AR(1)
                                               6)   #N_z

#%% Initialise the Guess on the Value Function Derivative

DaV_n = Uprime(0.5*(a_grid + z_grid[:,np.newaxis])) * (1+r)

#%% Numerical Algorithm

def backward_step(DaV_n):
    """
    One EGM backward step updating the derivative of the value function
    and policies.
    """
    #================================================
    # Step 1: Compute c(z,a') for (z,a') in G_z x G_a
    #================================================
    c = Uprime_Inv(beta * Pi @ DaV_n)

    #================================================
    # Step 2: Compute a(z,a') for (z,a') in G_z x G_a
    #================================================  
    a = (a_grid - z_grid[:,np.newaxis] + c) / (1+r)

    #==================================
    # Step 3: Invert a(z,a') to a'(z,a)
    #================================== 
    #Initialise object a'(z,a)
    a_prime = np.zeros((len(z_grid),len(a_grid)))
    
    #Fix rows (z gridpoint) and interpolate across the asset dimension
    for z_index in range(len(z_grid)):
        a_prime[z_index,:] = np.interp(a_grid,       #evaluate f(a) for a on G_a
                                       a[z_index,:], #x-coordinate: a not on G_a
                                       a_grid)       #y-coordinate: a' on G_a
        
    #======================================
    # Step 4: Enforce Inequality Constraint
    #====================================== 
    a_prime = np.maximum(a_prime, a_grid[0])

    #==================================================
    # Step 5: Compute c(z,a) for (z,a) in G_z times G_a
    #==================================================
    c = (1+r)*a_grid + z_grid[:,np.newaxis] - a_prime
    
    #=========================================
    # Step 6: Update Value Function Derivative
    #========================================= 
    DaV_n_plus_1 = Uprime(c) * (1+r)
    
    #Outputs
    return DaV_n_plus_1, c, a_prime 

#%% Value Function Iteration

#Initialise counter to track the number of iterations
iteration = 0

#Set tolerance and initialise distance to run the while loop
dist = 1.0
tolerance = 1e-8 *(1-beta) #epsilon_star

while dist > tolerance:
    
    #Get next iteration's Value Function
    DaV_n_plus_1, c, a_prime  = backward_step(DaV_n)
    
    #Compute the difference between iterations (supremum norm)
    dist = np.max(np.abs(DaV_n_plus_1 - DaV_n))
    
    #Update Value Function
    DaV_n = DaV_n_plus_1.copy()
    
    #Update Counter
    iteration += 1
    
    #Print status updates
    if iteration % 5 == 0:
        print(f"  iteration = {iteration}, Distance =  {dist:.2e}")
    
print(f"Converged after {iteration} iterations")

#%% Plots

#---- Consumption Policy ----
plt.figure(figsize=(10, 6))
for index_z, z in enumerate(z_grid):
    plt.plot(a_grid, c[index_z, :], label=f"$z_{{{index_z}}}$ = {z:.2f}")

plt.title("Consumption Policy by Income State $z_i$")
plt.xlabel("Assets $a$")
plt.ylabel("Consumption Policy")
plt.legend(title="Income States $z_i$")
plt.grid(True)
plt.tight_layout()
plt.show()

#---- Savings ----
savings = a_prime - a_grid

plt.figure(figsize=(10, 6))
for index_z, z in enumerate(z_grid):
    plt.plot(a_grid, savings[index_z, :], label=f"$z_{{{index_z}}}$ = {z:.2f}")
plt.title("Savings by Income State $z_i$")
plt.xlabel("Assets $a$")
plt.ylabel("Savings")
plt.legend(title="Income States $z_i$")
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Solve for the Value Function

def Bellman_Operator(V_next):
    """
    Under the previously found optimal policies, applies the Bellman Operator 
    on V_{t+1} and shifts
    the Value Function one period back to V_t
    
    Parameters
    ----------
    V_next : numpy.ndarray, 2D
        Value Function Matrix V_{t+1}(z, a)
    
    Returns
    -------
    Value Function V_t
    """

    #Avoid modifying the input (else-wise trouble when computing dist)
    V_next = V_next.copy()
        
    # Create interpolator for each e_index to evaluate V_{t+1}(z,a'(z,a)) for (z,a) in G_e x G_a
    for z_index, _ in enumerate(z_grid):
        # Create interpolation function for this e_index
        interp_func = scipy.interpolate.interp1d(a_grid, V_next[z_index, :], 
                             kind='linear', 
                             bounds_error=False, 
                             fill_value="extrapolate")
        
        # Get interpolated values at policy points a'(z,a)
        V_next[z_index, :] = interp_func(a_prime[z_index, :])
        
    #Compute V_t by applying the Bellman Operator
    return U(c) + beta*Pi@V_next

#Iterate on the Bellman Operator using V_T(z,a) = 0
V = np.zeros((len(z_grid),len(a_grid)))
for _ in range(500):
    
    #Apply Bellmann Operator
    V_prev = Bellman_Operator(V)
    
    #Compute Distance between Iterations
    dist = np.max(np.abs(V_prev - V))
    
    #Update Iteration
    V = V_prev
    
#Plot Value Function
plt.figure(figsize=(10, 6))
for index_z, z in enumerate(z_grid):
    plt.plot(a_grid, V[index_z, :], label=f"$z_{{{index_z}}}$ = {z:.2f}")
plt.title("Value Function by Income State $z_i$")
plt.xlabel("Assets $a$")
plt.ylabel("Value Function")
plt.legend(title="Income States $z_i$")
plt.grid(True)
plt.tight_layout()
plt.show()
