# -*- coding: utf-8 -*-
"""
Solves the SIMM with unrestricted labour choice using the first-order conditions
and root-finding.
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
w = 1
#Benefit
b = 1e-4

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
    Computes Marginal Utility U'.
    """
    marginal_utility = c**(-gamma)
    return marginal_utility

def Uprime_Inv(c):
    """
    Computes the Inverse of Marginal Utility [U']^{-1}.
    """
    value = c**(-1/gamma)
    return value

#%% Further Functions
def W_fun(D_aV):
    """
    Computes W(e,a').
    """
    W = beta * Pi @ D_aV
    
    return W

#Interpolation of W
def W_interpolation_fun(W):
    """
    Linearly Interpolates the function W(e,a') in its second argument. 
    No interpolation is performed across employment states.
      
    """
    W_interpolation = []
    #Linearly Interpolate W(e_i,a) for every adjacent gridpoints a_j and a_{j+1}
    for e in e_grid:
        W_interpolation.append(scipy.interpolate.interp1d(a_grid,W[e,:], 
                                                          kind='linear', 
                                                          fill_value='extrapolate'))
    
    return W_interpolation 

#Inter- AND Extrapolation of W
def W_ip_fun(W_interpolation, e, a_prime):
    """
    Linearly interpolates W(e,a') on [a_min,a_max]. Extrapolates W(e,a') outside 
    of 'a_grid' using the envelope condition
    """
    a_max = a_grid[-1] 
    
    if  a_prime <= a_max: #Linear Interpolation
        return W_interpolation[e](a_prime)
    
    if a_prime > a_max: #Extrapolation
    
        #threshold to ensure the extrapolation is continuous
        threshold = Uprime(Uprime_Inv(W[e,len(a_grid)-1])-np.max(a_grid))
        val = Uprime( threshold + a_prime)  
        return val
    
#---- Root Finding ----
def solve_consumption_unrestricted(c, e, a, e_index, a_index):
    """
    At a gridpoint (e,a) in G_e x G_a, this function computes the residual
    of the FOC for consumption in the unrestricted case (mu=0)
    
    Parameters
    ----------
    c: float
        value for consumption at gridpoint
    e: float
        value of exogenous shock e_t
    a: float
        value of endogenous shock a_t
    e_index: int
                position on grid G_e of 'e'
    a_index: int
                position on grid G_a of 'a'
                
    Returns
    -------
    residual of FOC
    """

    #Compute a'(e,a)
    a_prime = (1+r)*a + (Uprime(c)*w*e)**(1/phi) + b*(1-e) - c
    
    #Compute residual
    residual = c - Uprime_Inv( W_ip_fun(W_interpolation, e, a_prime) )
    
    return residual

def solve_consumption_constrained(c, e, a, e_index, a_index):
    """
    At a gridpoint (e,a) in G_e x G_a, this function computes the residual
    of the FOC for consumption in the restricted case (mu>0 <--> a'(e,a) = a_min)
    
    Parameters
    ----------
    c: float
        value for consumption at gridpoint
    e: float
        value of exogenous shock e_t
    a: float
        value of endogenous shock a_t
    e_index: int
                position on grid G_e of 'e'
    a_index: int
                position on grid G_a of 'a'
                
    Returns
    -------
    residual of FOC
    """
    
    #Compute Residual
    residual = a_grid[0] - (1+r) * a  - (Uprime(c)*w*e)**(1/phi)*w*e - b*(1-e) + c
    
    return residual

def nearest_neighbour_index(a_prime_pol, a_grid):
    """
    Uses nearest neighbour matching to map the policy function a'(e,a) onto
    the grid G_a.
    
    Parameters
    ----------
    a_prime_pol : ndarray (N_e, N_a)
        Policy function a' which doesn't map onto G_a.
    a_grid : ndarray (N_a)
        Asset Grid G_a.

    Returns
    -------
    a_prime_nn_index : ndarray (N_e, N_a)
        Policy function a' which maps onto G_a.

    """
    
    # Compute absolute differences between each element of a'(e,a) and a_grid
    differences = np.abs(a_prime_pol[..., np.newaxis] - a_grid)

    # Find the index of the closest value in a_grid for each element
    closest_indices = np.argmin(differences, axis=-1)

    # Get the closest values from a_grid
    a_prime_nn = a_grid[closest_indices]
    
    #Get the integer position
    a_prime_nn_index = np.searchsorted(a_grid, a_prime_nn) 
    
    return a_prime_nn_index

#%% Grid: Endogenous State Variable

#Linearly spaced grids for assets
a_grid = np.linspace(0,     #a_min
                     10,    #a_max
                     500)   #N_a

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

#%% Initialise Value Function Derivative

#Guess on consumption policy
c_n = 0.5*((1+r)*a_grid + 0.5*w*e_grid[:,np.newaxis] + b*(1-e_grid[:,np.newaxis]))
#Envelope Condition for Value Function derivative
DaV_n = Uprime(c_n)*(1+r)

#Plot Initial Guess
plt.figure(figsize=(10, 6))
for index_e, e in enumerate(e_grid):
    plt.plot(a_grid, DaV_n[index_e, :], label=f"$e_{{{index_e}}}$ = {e}")

plt.title("Value Function Derivative $\partial_a V(e,a)$ by Employment State $e_i$")
plt.xlabel("Assets $a$")
plt.ylabel("Value Function Derivative $\partial_a V(e_i,a)$")
plt.legend(title="Employment states $e_i$")
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Numerical Algorithm

def backward_step(DaV_n, c_n):
    """
    One backward step updating the derivative of the Value Function
    and policies.
    
    Parameters
    ----------
    DaV_n : ndarray (N_e, N_a)
        Current iteration of the Value Function derivative.
    c_n : ndarray (N_e, N_a)
        Current iteration of the consumption policy function.

    Returns
    -------
    DaV_n_plus_1 : ndarray (N_e, N_a)
        Next iteration of the Value Function derivative.
    c: ndarray (N_e, N_a)
        Next iteration of the consumption policy function.
    n: ndarray (N_e, N_a)
        Next iteration of the labour policy function.
    a_prime: ndarray (N_e, N_a)
        Next iteration of the asset policy function.
    mu: ndarray (N_e, N_a)
        Next iteration of the Lagrange Multiplier.
    """
    #Initialise Policy Functions
    c       = c_n
    n       = np.zeros((len(e_grid),len(a_grid)))
    a_prime = np.zeros((len(e_grid),len(a_grid)))
    
    #============================
    # Unrestricted Case: mu = 0
    #============================
    
    #---- Consumption ----
    #Root-finding at each gridpoint
    for e_index, e in enumerate(e_grid):
        for a_index, a in enumerate(a_grid):
            root = scipy.optimize.root(solve_consumption_unrestricted,
                                                     c[e_index,a_index],
                                                     args=(e, a, e_index, a_index)
                                                     )
            c[e_index,a_index] = float(root.x)
    
    #---- Labour ----
    for e_index, e in enumerate(e_grid):
        for a_index, a in enumerate(a_grid):
            n[e_index, a_index] = (Uprime(c[e_index,a_index])*w*e)**(1/phi)
    
    #---- Next Period's Assets ----
    for e_index, e in enumerate(e_grid):
        for a_index, a in enumerate(a_grid):       
            a_prime[e_index, a_index] = (1+r)*a + n[e_index, a_index]*w*e + b*(1-e) - c[e_index, a_index]
    
    #============================
    # Restricted Case: mu > 0
    #============================
            
    #Identify Gridpoints that violate the inequality constraint
    G_tilde = []  
    for e_index, _ in enumerate(e_grid):
        for a_index, _ in enumerate(a_grid):
            if a_prime[e_index,a_index] < a_grid[0]:
                G_tilde.append((e_index,a_index))
                
    #Enforce inequality constraint
    a_prime[a_prime<a_grid[0]] = a_grid[0]
    
    #Resolve for consumption & labour
    for gridpoint in G_tilde:
        
        #Unpack Gridpoints
        e_index = gridpoint[0]
        e       = e_grid[e_index]
        
        a_index = gridpoint[1]
        a       = a_grid[a_index]
        
        #Re-solve for consumption
        root = scipy.optimize.root(solve_consumption_constrained,
                                                 c[e_index,a_index],
                                                 args=(e, a, e_index, a_index)
                                                 )
        c[e_index,a_index] = float(root.x)
        
        #Re-solve for Labour
        n[e_index, a_index] = (Uprime(c[e_index, a_index])*w*e)**(1/phi)
        
    
    #Compute the Lagrangian Multiplier (if desired)
    mu = np.zeros((len(e_grid),len(a_grid)))
    for gridpoint in G_tilde:
        #Unpack Gridpoints
        e_index = gridpoint[0]
        e       = e_grid[e_index]
        
        a_index = gridpoint[1]
        a       = a_grid[a_index]
        
        mu[e_index, a_index] = Uprime(c[e_index, a_index]) - W[e_index,0]
    
    #Update Value Function Derivative
    DaV_n_plus_1 = Uprime(c)*(1+r)
    
    return DaV_n_plus_1, c, n, a_prime, mu

#%% Value Function Iteration

print(f"Solving the problem using {len(a_grid)} gridpoints for the endogenous state variable")

#Initialise counter to track the number of iterations
iteration = 0

#Set tolerance and initialise distance to run the while loop
dist = 1
tolerance = 1e-8 *(1-beta) #epsilon_star

#Initialise Object to store all iterations
Value_Functions = []
Value_Functions.append(DaV_n)

#---- Value Function Iteration ----
while dist > tolerance: #while not converged

    #Prepare W_ip (passed as globals to all other functions)
    W               = W_fun(Value_Functions[iteration])
    W_interpolation = W_interpolation_fun(W)
        
    #Get next iteration's Value Function
    DaV_n_plus_1, c_n_plus_1, n, a_prime, mu = backward_step(Value_Functions[iteration], c_n)
    
    #Store next iteration
    Value_Functions.append(DaV_n_plus_1)
    c_n = c_n_plus_1.copy()
    
    #Compute the difference between iterations (supremum norm)
    dist = np.max(np.abs(Value_Functions[iteration+1] - Value_Functions[iteration]))
    
    #Update Counter
    iteration += 1
    
    #Print status updates
    if iteration % 5 == 0:
        print(f"iteration = {iteration}, Distance =  {dist:.2e}")
        
#%% Extract Results

#Value Function Derivative
DaV = Value_Functions[-1]

#Policy Functions
_, c_pol, n_pol, a_prime_pol, _ = backward_step(DaV, c_n)
savings = a_prime_pol - a_grid

#%% Plot Policies

#---- Consumption Policy ----
plt.figure(figsize=(10, 6))
for index_e, e in enumerate(e_grid):
    plt.plot(a_grid, c_pol[index_e, :], label=f"$e_{{{index_e}}}$ = {e}")

plt.title("Consumption Policy by Employment State $e_i$")
plt.xlabel("Assets $a$")
plt.ylabel("Consumption Policy")
plt.legend(title="Employment states $e_i$")
plt.grid(True)
plt.tight_layout()
plt.show()

#---- Savings ----
plt.figure(figsize=(10, 6))
for index_e, e in enumerate(e_grid):
    plt.plot(a_grid, savings[index_e, :], label=f"$e_{{{index_e}}}$ = {e}")

plt.title("Savings by Employment State $e_i$")
plt.xlabel("Assets $a$")
plt.ylabel("Savings")
plt.legend(title="Employment states $e_i$")
plt.grid(True)
plt.tight_layout()
plt.show()

#---- Labour ----
plt.figure(figsize=(10, 6))
for index_e, e in enumerate(e_grid):
    plt.plot(a_grid, n[index_e, :], label=f"$e_{{{index_e}}}$ = {e}")

plt.title("Labour Supply by Employment State $e_i$")
plt.xlabel("Assets $a$")
plt.ylabel("Labour Supply")
plt.legend(title="Employment states $e_i$")
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Recover the Value Function

#=============================================
# Option 1: Solving System of Linear Equations
#=============================================
def find_VF(V_flat):
    """
    Solve for the Value Function V(e,a) from the Bellman Equation by using
    the previously found policy functions. 
    
    We build a system of linear equations by mapping a'(e,a) back onto G_a
    using nearest-neighbour matching
    
    Parameters
    ----------
    V_flat : numpy.ndarray, 1D
        Flattened array representing the current guess of the Value Function V(e, a)
    
    Computes the residual of the Bellman Equation under the optimal policies
    """
    
    #Get a'(e,a) that maps into the index position of 'a_grid'
    a_prime_nn_index = nearest_neighbour_index(a_prime_pol, a_grid)

    # Restore 2D shape
    V = V_flat.reshape((len(e_grid), len(a_grid)))  
    
    #Get V_{t+1} based on policy function
    V_next = np.zeros((len(e_grid), len(a_grid)))
    for e_index, _ in enumerate(e_grid):
        #a_prime_pol_nn_index to switch the column position based on policy a'(e,a)
        V_next[e_index,:] = V[e_index, a_prime_nn_index[e_index,:]] 
        
    #Compute the residual of the Bellman Equation
    residual = V - U(c_pol) + n_pol**(1+phi)/(1+phi) - beta*Pi@V_next
    
    return residual.flatten()

#Solve the system of linear equations from the Bellman Equation to get the Value Function
root = scipy.optimize.root(find_VF, np.zeros(len(e_grid)*len(a_grid)))
V = root.x.reshape((len(e_grid), len(a_grid)))


#========================================
# Option 2: Iterating on Bellman Operator
#========================================
def Bellman_Operator(V_next):
    """
    Under the previously found optimal policies, applies the Bellman Operator 
    on V_{t+1} and shifts
    the Value Function one period back to V_t
    
    Parameters
    ----------
    V_next : numpy.ndarray, 2D
        Value Function Matrix V_{t+1}(e, a)
        
    Returns
    -------
    Value Function V_t
    """

    #Avoid modifying the input (else-wise trouble when computing dist)
    V_next = V_next.copy()
        
    # Create interpolator for each e_index to evaluate V_{t+1}(e,a'(e,a)) for (e,a) in G_e x G_a
    for e_index, _ in enumerate(e_grid):
        # Create interpolation function for this e_index
        interp_func = scipy.interpolate.interp1d(a_grid, V_next[e_index, :], 
                             kind='linear', 
                             bounds_error=False, 
                             fill_value="extrapolate")
        
        # Get interpolated values at policy points a'(e,a)
        V_next[e_index, :] = interp_func(a_prime_pol[e_index, :])
        
    #Compute V_t by applying the Bellman Operator
    return U(c_pol) - n_pol**(1+phi)/(1+phi) + beta*Pi@V_next


#Iterate on the Bellman Operator using V_T(e,a) = 0 until the Value Functions
# have converged and the stationary Value Function is achieved
V = np.zeros((len(e_grid),len(a_grid)))
for _ in range(500):
    
    #Apply Bellmann Operator
    V_prev = Bellman_Operator(V)
    
    #Compute Distance between Iterations
    dist = np.max(np.abs(V_prev - V))
    
    #Update Iteration
    V = V_prev.copy()

print(f"Distance = {dist:.2e}")

#%% Plot Value Function

plt.figure(figsize=(10, 6))
for index_e, e in enumerate(e_grid):
    plt.plot(a_grid[8:], V[index_e, 8:], label=f"$e_{{{index_e}}}$ = {e}")

plt.title("Value Function by Employment State $e_i$")
plt.xlabel("Assets $a$")
plt.ylabel("Value Function $V(e_i,a)$")
plt.legend(title="Employment states $e_i$")
plt.grid(True)
plt.tight_layout()
plt.show()
