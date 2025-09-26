# -*- coding: utf-8 -*-
"""
Solves the SIMM with restricted labour choice using the first-order conditions
and the EGM.
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

#%% Set Parameters

#risk-free rate
r = 0.01
#Discount Factor
beta = 0.95
#Risk Aversion
gamma = 2
#Dis-Utility Working
phi = 2
#Wage
w = 2
#Benefit
b = 0.05

#%% Utility Function 
def U(c):
    if gamma == 1:
        utility = np.log(c)
    else:
        utility = c**(1-gamma)/(1-gamma)
    return utility

def Uprime(c):
    marginal_utility = c**(-gamma)
    return marginal_utility

def Uprime_Inv(c):
    value = c**(-1/gamma)
    return value

#%% Grid: Assets a'

def discretize_assets(amin, amax, n_a):
    """
    Computes the asset grid in a non-uniform manner. Gridpoints are denser
    closer to a_min.
    
    Parameters
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
a_grid = discretize_assets(0,40,100)

#%% Grid: Exogenous State Variable

#Grid
e_grid = np.array([0,1])

#Markov Transition Matrix
"""
Pi[0,0] = Prob(e'=0 | e=0)
Pi[0,1] = Prob(e'=1 | e=0)
Pi[1,0] = Prob(e'=0 | e=1)
Pi[1,1] = Prob(e'=1 | e=1)
"""
Pi = np.array([[0.2,0.8], [0.1,0.9]])

#%% Numerical Algorithm

def backward_step(DaV_n):
    """
    One EGM backward step updating the derivative of the value function
    and policies.
    """
    #================================================
    # Step 1: Compute c(e,a') for (e,a') in G_e x G_a
    #================================================
    c = Uprime_Inv(beta * Pi @ DaV_n)

    #================================================
    # Step 2: Compute n(e,a') for (e,a') in G_e x G_a
    #================================================  
    n = np.maximum(0, (1 - 1/( Uprime(c) * w )) ) * e_grid[:,np.newaxis]
    
    #================================================
    # Step 3: Compute a(e,a') for (e,a') in G_e x G_a
    #================================================ 
    a = (a_grid[np.newaxis,:] - n*w*e_grid[:,np.newaxis] - b*(1-e_grid[:,np.newaxis]) + c) / (1+r)

    #==================================
    # Step 4: Invert a(e,a') to a'(e,a)
    #================================== 
    #Initialise object a'(e,a)
    a_prime = np.zeros((len(e_grid),len(a_grid)))
    
    #Fix rows (e gridpoint) and interpolate across the asset dimension
    for e_index in range(len(e_grid)):
        a_prime[e_index,:] = np.interp(a_grid,       #evaluation points on G_a
                                       a[e_index,:], #x-coordinate: a
                                       a_grid)       #y-coordinate: a'
        
    #======================================
    # Step 5: Enforce Inequality Constraint
    #====================================== 
    a_prime = np.maximum(a_prime, a_grid[0])

    #==================================================
    # Step 6: Compute c(e,a) for (e,a) in G_e times G_a
    #==================================================
    root = scipy.optimize.root(solve_consumption, c.ravel(), 
                               args = (a_grid, a_prime, w, e_grid))
    c = root.x.reshape((len(e_grid),len(a_grid)))   
    
    #==================================================
    # Step 7: Compute n(e,a) for (e,a) in G_e times G_a
    #==================================================
    n = np.maximum(0, (1 - 1/( Uprime(c) * w )) ) * e_grid[:,np.newaxis]
    
    #=========================================
    # Step 8: Update Value Function Derivative
    #========================================= 
    DaV_n_plus_1 = Uprime(c) * (1+r)
    
    #Outputs
    return DaV_n_plus_1, c, a_prime, n 

def solve_consumption(c_flat, a_grid, a_prime, w, e_grid):
    """
    Computes the residual for a given consumption policy of the FOC
    
    Parameters
    ----------
    c_flat: ndarray, one-dimensional
    a_grid: ndarray (N_a)
            Asset Grid G_a
    a_prime: ndarray (N_e, N_a)
                Policy a'(e,a) 
    w: float
        wage 
    e_grid: ndarray (N_e)
            Employment Grid G_e
    """
    
    # reshape back to dimension (N_e,N_a)
    c = c_flat.reshape((len(e_grid),len(a_grid)))
    
    #Compute the Residual
    residual = (1+r)*a_grid - a_prime + np.maximum(0, (1 - 1/( Uprime(c) * w )) )* w * e_grid[:,np.newaxis] + b*(1-e_grid[:,np.newaxis]) - c


    # flatten residual to 1D
    return residual.ravel()

#%% Initialise the Guess on the Value Function Derivative

c_n = 0.5*((1+r)*a_grid + 0.5*w*e_grid[:,np.newaxis] + b*(1-e_grid[:,np.newaxis]))
DaV_n = Uprime(c_n)*(1+r)

#%% Value Function Iteration

#Initialise counter to track the number of iterations
iteration = 0

#Set tolerance and initialise distance to run the while loop
dist = 1
tolerance = 1e-8 *(1-beta) #epsilon_star

while dist > tolerance:
    
    #Get next iteration's Value Function
    DaV_n_plus_1, c, a_prime, n  = backward_step(DaV_n)
    
    #Compute the difference between iterations (supremum norm)
    dist = np.max(np.abs(DaV_n_plus_1 - DaV_n))
    
    #Update Value Function
    DaV_n = DaV_n_plus_1.copy()
    
    #Update Counter
    iteration += 1
    
    #Print status updates
    if iteration % 5 == 0:
        print(f"iteration = {iteration}, Distance =  {dist:.2e}")
    
#%% Bonus: Check if KKT conditions are satisfied

#Extract the Solution
DaV = DaV_n.copy()

#Prepare W_ip
def W_fun(DaV):
    """
    Computes W(e,a').
    """
    W = beta * Pi @ DaV
    
    return W

#Interpolation of W
def W_interpolation_fun(W):
    """
    Linearly Interpolates the function W(e,a') in its second argument.      
    """
    W_interpolation = []
    #Linearly Interpolate W(e_i,a) for every adjacent gridpoints a_j and a_{j+1}
    for e in e_grid:
        W_interpolation.append(scipy.interpolate.interp1d(a_grid,W[e,:], 
                                                          kind='linear', 
                                                          fill_value='extrapolate'))
    
    return W_interpolation 

def W_ip_fun(W_interpolation, e, a_prime):
    """
    Linearly interpolates and extrapolate W(e,a') on [a_min,a_max].
    """
    return W_interpolation[e](a_prime)


W               = W_fun(DaV)
W_interpolation = W_interpolation_fun(W)


#---- Compute Lagrange Multiplier mu_a ----

#Construct interpolated function W
W = np.zeros((len(e_grid),len(a_grid)))
for e_index, e in enumerate(e_grid):
    W[e_index,:] = W_ip_fun(W_interpolation, e, a_prime[e,:])

#Compute the Lagrange Multiplier
mu_a = Uprime(c) - W
"""
mu_a should be significantly positive where 'a_prime' maps to zero. Else-where,
mu_a should be approximately zero. Due to the interpolation error, the margins of
error can be somewhat larger for mu_a
"""

# ---- Compute Lagrange Multiplier mu_n ----
mu_n = np.zeros((len(e_grid),len(a_grid)))
mu_n[1,:] = -1/(n[1,:]-1) - Uprime(c[1,:])*w
"""
For e = 0, labour choice doesn't exist, so mu_n needn't be computed. 

For e = 1, mu_n should be significantly positive for large assets since
n(1,a_large) = 0. Else-wise, mu_n should be close to zero.
"""

#%% Plots

#---- Consumption Policy ----
plt.figure(figsize=(10, 6))
for index_e, e in enumerate(e_grid):
    plt.plot(a_grid, c[index_e, :], label=f"$e_{{{index_e}}}$ = {e}")

plt.title("Consumption Policy by Employment State $e_i$")
plt.xlabel("Assets $a$")
plt.ylabel("Consumption Policy")
plt.legend(title="Employment States $e_i$")
plt.grid(True)
plt.tight_layout()
plt.show()

#---- Savings ----
savings = a_prime - a_grid

plt.figure(figsize=(10, 6))
for index_e, e in enumerate(e_grid):
    plt.plot(a_grid, savings[index_e, :], label=f"$e_{{{index_e}}}$ = {e}")
plt.title("Savings by Employment State $e_i$")
plt.xlabel("Assets $a$")
plt.ylabel("Savings")
plt.legend(title="Employment States $e_i$")
plt.grid(True)
plt.tight_layout()
plt.show()
