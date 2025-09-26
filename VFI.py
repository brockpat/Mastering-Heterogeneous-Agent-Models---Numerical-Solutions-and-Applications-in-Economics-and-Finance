# -*- coding: utf-8 -*-
"""
Implements the baseline Value Function Iteration (VFI) algorithm with grid search.

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

#%% Grid: Endogenous State Variable

#Linearly spaced grids for assets
a_grid = np.linspace(0,     #a_min
                     10,    #a_max
                     30)    #N_a

#%% Grid: Exogenous State Variable

#Rouwenhorst Method to get grid 'z_grid', 
#   stationary distribution 'pi' and 
#   Markov Transition Matrix 'Pi'.
z_grid, pi, Pi = Rouwenhorst.discretize_income(0.6, #rho 
                                               0.5, #Variance of AR(1)
                                               8)   #N_z

#%% Control Set

#Correspondence Gamma-hat: 
def Gamma(z,a):
    """
    Finds all permissable values of consumption 'c' such that 
    next period's assets a' lie on 'a_grid' and 'c' is non-negative
    for the current realisation of the state space (z,a).
    """
    #Initialise Correspondence as empty
    c_range = []
    
    #Loop over all potential a'(z,a) in G_a
    for a_target in a_grid:
        c = (1+r)*a + z - a_target #compute consumption
        if c>0: #if 'c' admissable --> add it to the correspondance
            c_range.append(c)
    return c_range

#Initialise the Control_Set Gamma-hat(z,a)
Control_Set  = np.empty((len(z_grid),len(a_grid)),dtype=list)

#Fill the Control Set for every gridpoint (z,a) in the state space
#The Control Set is an object of dimension (len(z_grid),len(a_grid)) and each
#element is a list (feasible consumption set at that gridpoint of the state space)
for i in range(len(z_grid)):
    for j in range(len(a_grid)):
        Control_Set[i,j] = np.sort(Gamma(z_grid[i],a_grid[j]))
#%% Functions

def Bellman_Operator(V, z, a):
    """
    Computes the Bellman Operator at a gridpoint (z,a)

    Parameters
    ----------
    V : ndarray, shape (N_z, N_a)         
            Discretised Value Function V_{t+1}.
    z: float          
            Realisation of the exogenous state (z_t \in G_z)
    a: float
            Realisation of the endogenous state (a_t \in G_a)

    Returns
    -------
    V_t(z,a): 'max_value'
    c(z,a)  : 'c_star' --> Optimal consumption policy
    """

    #Get Index of z in the grid G_z. 
    index_z = np.searchsorted(z_grid,z)
    
    #Get the Index of a in the grid G_a.
    index_a = np.searchsorted(a_grid,a)
    
    #Obtain the set of admissible consumption choices
    Gamma = Control_Set[index_z, index_a]
    
    #Initialise Object to store the right hand side of the Bellmann Equation
    values = [] 

    #Iterate over feasible consumption choices
    for c in Gamma:
        #Compute next period's assets a_prime under consumption 'c'
        a_prime = (1+r) * a + z - c
        
        #Index of a_prime on G_a (a' lies on G_a exactly)
        index_a_prime = np.searchsorted(a_grid,a_prime)
        
        #Compute the right-hand-side of the Bellman Equation for consumption 'c'
        RHS_Bellman = U(c) + beta * Pi[index_z,:] @ V[:,index_a_prime]
        
        #Store Results
        values.append(RHS_Bellman)
        
    #Find the Maximum of RHS_Bellman which finalises the Bellman Operator
    max_value = np.max(values)
    
    #Extract the optimal consumption choice
    c_star = Gamma[np.argmax(values)] 
    
    return max_value, c_star 

def backward_step(V_n):
    """
    One backward step, i.e. one application of the Bellman Operator,
    updating the Value Function
    
    Parameters
    ----------
    V_n : ndarray, shape (N_z, N_a)         
            Discretised Value Function at iteration n.

    Returns
    -------
    V_n+1(z,a): Value Function shifted one period back in time
    c(z,a)  :   'c_pol' --> Optimal consumption policy
    a'(z,a) :   'a_prime_pol' --> Optimal asset policy 
    """
    
    #Initialise next Value Function
    V_n_plus_1 = np.zeros((len(z_grid),len(a_grid)))
    
    #Initialise consumption policy function
    c_pol = np.zeros((len(z_grid),len(a_grid)))
    
    #Loop over state space
    for index_z, z in enumerate(z_grid):
        for index_a, a in enumerate(a_grid):
            #Compute Bellman Operator
            value, c_star = Bellman_Operator(V_n, z, a)
            
            #Store results
            V_n_plus_1[index_z,index_a] = value
            c_pol[index_z,index_a]      = c_star
    
    #Compute policy a'(z,a)
    a_prime_pol = (1+r)*a_grid + z_grid[:,np.newaxis] - c_pol
    
    return V_n_plus_1, c_pol, a_prime_pol

#%% Initialise Value Function

#Initial guess on the stationary Value Function. Use ''staying put''
V_n = U(0.5*((1+r)*a_grid + z_grid[:,np.newaxis])/(1-beta))

#%% Value Function Iteration

#Initialise counter to track the number of iterations
n = 0

#Set tolerance and initialise distance to run the while loop
dist = 1
tolerance = 1e-8 *(1-beta) #epsilon_star

#Initialise Object to store all iterations
Value_Functions = []
Value_Functions.append(V_n)

#---- Value Function Iteration ----
while dist > tolerance: #while not converged

    #Get next iteration's Value Function
    V_n_plus_1, c_pol, a_prime_pol = backward_step(Value_Functions[n])
    
    #Store next iteration
    Value_Functions.append(V_n_plus_1)
    
    #Compute the difference between iterations (supremum norm)
    dist = np.max(np.abs(Value_Functions[n+1] - Value_Functions[n]))
    
    #Update Counter
    n += 1
    
    #Print status updates
    if n % 5 == 0:
        print(f"n = {n}, Distance = {dist:.2e}")

#%% Extract Results

#Stationary Value Function
V = Value_Functions[-1]

#Policy Functions
_, c_pol, a_prime_pol = backward_step(V)
savings = a_prime_pol - a_grid

#%% Plot Results

#---- Value Function ----
plt.figure(figsize=(10, 6))
for index_z, z in enumerate(z_grid):
    plt.plot(a_grid, V[index_z, :], label=f"$z_{{{index_z}}}$ = {z:.2f}")

plt.title("Value Function by Income State $z_i$")
plt.xlabel("Assets $a$")
plt.ylabel("Value Function $V(z_i,a)$")
plt.legend(title="Income states $z_i$")
plt.grid(True)
plt.tight_layout()
plt.show()

#---- Consumption Policy ----
plt.figure(figsize=(10, 6))
for index_z, z in enumerate(z_grid):
    plt.plot(a_grid, c_pol[index_z, :], label=f"$z_{{{index_z}}}$ = {z:.2f}")

plt.title("Consumption Policy by Income State $z_i$")
plt.xlabel("Assets $a$")
plt.ylabel("Consumption Policy $c(z_i,a)$")
plt.legend(title="Income states $z_i$")
plt.grid(True)
plt.tight_layout()
plt.show()

#---- Savings ----
plt.figure(figsize=(10, 6))
for index_z, z in enumerate(z_grid):
    plt.plot(a_grid, savings[index_z, :], label=f"$z_{{{index_z}}}$ = {z:.2f}")

plt.title("Savings by Income State $z_i$")
plt.xlabel("Assets $a$")
plt.ylabel("Savings $s(z_i,a)$")
plt.legend(title="Income states $z_i$")
plt.grid(True)
plt.tight_layout()
plt.show()