# -*- coding: utf-8 -*-
"""
Solves the Aiyagari Model with endogenous labour supply.

EGM is used to obtain the Policy Functions. 
Forward Iteration with nearest-neighbour matching is used to compute the stationary distribution.

The identical problem with discontinuities as in 'Hugget.py' is present.
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

#Discount Factor
beta = 0.95
#Risk Aversion
gamma = 2
#Dis-Utility Working
phi = 2
#Benefit
b = 0.05
# Elasticity of capital
alpha = 0.3

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
a_grid = discretize_assets(0,80,50)
"""
Due to benefits, assets can actually be zero and c>0 will always be in the control set.
"""

#%% Grid: Exogenous State Variable

#Grid
e_grid = np.array([0,1])

#Markov Transition Matrix
Pi = np.array([[0.2,0.8], [0.1,0.9]])
"""
Pi_e[0,0] = Prob(e'=0 | e=0)
Pi_e[0,1] = Prob(e'=1 | e=0)
Pi_e[1,0] = Prob(e'=0 | e=1) 
Pi_e[1,1] = Prob(e'=1 | e=1)
"""

#Stationary Distribution of Markov Chain
eigenvalues, eigenvectors = np.linalg.eig(Pi.T)

stationary_index = np.argmax(np.isclose(eigenvalues, 1.0))
pi = eigenvectors[:, stationary_index] 

pi = pi / np.sum(pi) #1st entry is unemployed, 2nd entry is employed

#Sanity Check: Output should be the zero vector
#print(pi - pi @ Pi)
#%% Households: Policy Functions

def solve_policies(r, w, DaV_n, epsilon = 1e-8):
    """
    Solve household policies by EGM given an interest rate r and wage w.

    Parameters
    ----------
    r : float
        Risk-free interest rate.
    w : float
        wage
    DaV_n : ndarray, shape (N_e, N_a)
        Initial guess for derivative of the Value Function.
    epsilon : float, optional
        Convergence tolerance for DaV (implemented as epsilon*(1-beta)).
        
    Returns
    -------
    DaV_n_plus_1 : ndarray, shape (N_e, N_a)
        Converged derivative of the Value Function.
    c : ndarray, shape (N_e, N_a)
        Consumption policy c(e,a) defined on e in G_e and a in G_a.
    n : ndarray, shape (N_e, N_a)
        Labour supply policy n(e,a) defined on e in G_e and a in G_a.
    a_prime : ndarray, shape (N_e, N_a)
        Policy a'(e,a) defined on e in G_e and a in G_a.
        a' will be off-grid, i.e. not in G_a, prior to nearest-neighbor matching.
    """
    
    print(" Solving for the Households' Policy Functions.")
    #Initialise counter to track the number of iterations
    iteration = 0
    
    #Set tolerance and initialise distance to run the while loop
    dist = 1.0
    tolerance = epsilon *(1-beta) #epsilon_star
    
    while dist > tolerance:
    
        #Get next iteration's Value Function
        DaV_n_plus_1, c, a_prime, n   = backward_step(r, w, DaV_n)
        
        #Compute the difference between iterations (supremum norm)
        dist = np.max(np.abs(DaV_n_plus_1 - DaV_n))
        
        #Update Value Function
        DaV_n = DaV_n_plus_1.copy()
        
        #Update Counter
        iteration += 1
        
        #Print status updates
        if iteration % 500 == 0:
            print("      iteration = ", iteration, " Distance = ", f"{dist:.2e}")
    
    print(f"   Converged after {iteration} iterations")
    
    #Return Value Function Derivative and Policy Functions
    return DaV_n_plus_1, c, a_prime, n


def backward_step(r, w, DaV_n):
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
    n = (Uprime(c)*w)**(1/phi)  # employed
    n[0,:] = 0                  # unemployed
    
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
        a_prime[e_index,:] = np.interp(a_grid, #evaluation points on G_a
                                       a[e_index,:], #x-coordinate: a
                                       a_grid) #y-coordinate: a'
    
    #======================================
    # Step 5: Enforce Inequality Constraint
    #====================================== 
    a_prime = np.maximum(a_prime, a_grid[0])
        
    #=========================================================
    # Step 6: Compute c(e,a) for all gridpoints simultaneously
    #=========================================================
    root = scipy.optimize.root(solve_consumption, c.ravel(), 
                                      args = (r, w, a_prime)
                                      )
    c = root.x.reshape((len(e_grid),len(a_grid)))
    
    #==================================================
    # Step 7: Compute n(e,a)
    #==================================================
    n = (Uprime(c)*w)**(1/phi)  # employed
    n[0,:] = 0                  # unemployed
    
    #=========================================
    # Step 8: Update Value Function Derivative
    #========================================= 
    DaV_n_plus_1 = Uprime(c) * (1+r)
    
    #Outputs
    return DaV_n_plus_1, c, a_prime, n 

def solve_consumption(c_flat, r, w, a_prime):
    """
    Vectorised residual function for all gridpoints in order to
    solve for consumption once a'(e,a) has been obtained.
    """
    c = c_flat.reshape((len(e_grid), len(a_grid)))  # reshape back to 2D

    e_vals = e_grid[:, np.newaxis]  # shape (len(e_grid), 1)
    a_vals = a_grid[np.newaxis, :]  # shape (1, len(a_grid))

    # Residual
    residual = (a_prime - (1 + r) * a_vals
                - (Uprime(c) * w * e_vals) ** (1 / phi) * w * e_vals
                - b * (1 - e_vals) + c)

    return residual.ravel()  # flatten to 1D for root solver


def nn_matching(a_grid, a_prime):
    """
    Project off-grid savings a'(e,a) back to the discrete asset grid G_a using
    nearest-neighbor matching.

    Parameters
    ----------
    a_grid : ndarray, shape (N_a,)
        Asset grid G_a.
    a_prime : ndarray, shape (N_e, N_a)
        Off-grid Policy.

    Returns
    -------
    a_prime_nn : ndarray, shape (N_e, N_a)
        Policy shifted to the nearest gridpoint on a_grid.
    closest_idx : ndarray, shape (N_e, N_a)
        Respective index of a_grid for each a'(e,a).
    """
    
    #Compute the difference of each entry in a'(e,a) to G_a
    diff = np.abs(a_prime[..., np.newaxis] - a_grid)
    
    #find the index of the closest element in a_grid.
    closest_idx = np.argmin(diff, axis=-1)
    
    #a'(e,a) will map exclusively into G_a
    a_prime_nn = a_grid[closest_idx]
    
    return a_prime_nn, closest_idx

#%% Household: Stationary Distribution
def forward_step(D, a_prime_nn_index):
    """
    One step of forward iteration on the distribution, transporting mass
    from (e,a) to (e',a').
    
    Parameters
    ----------
    D : ndarray, shape (N_e, N_a)
        Current cross-sectional distribution over states (e,a).
    a_prime_nn_index : ndarray, shape (N_e, N_a)
        Indices of next-period asset choices on a_grid for each (e,a).

    Returns
    -------
    D_next : ndarray, shape (N_e, N_a)
        Next-period distribution.
    """
    D_next = np.zeros_like(D)
    for a_index, a in enumerate(a_grid):
        for e_index, e in enumerate(e_grid):
            # Move the mass D[e,a] to (e', a'(e,a)) weighted by Pi[e, e']
            D_next[:, a_prime_nn_index[e_index, a_index]] += D[e_index, a_index] * Pi[e_index, :]

    return D_next

def stationary_Distribution(D, a_prime_nn_index, tolerance = 1e-8):
    """
    Compute the stationary distribution for a given Policy Function via forward
    iteration until convergence.

    Parameters
    ----------
    D : ndarray, shape (N_e, N_a)
        Initial guess for the distribution (must sum to 1).
    a_prime_nn_index : ndarray, shape (N_e, N_a)
        Indices of next-period asset choices on a_grid for each (e,a).
    tolerance : float, optional
        Sup‐norm stopping tolerance.

    Returns
    -------
    D_SE : ndarray, shape (N_e, N_a)
        Stationary distribution consistent with the policy a'(e,a).
    """
    #Initialise counter to track the number of iterations
    iteration = 0
    
    #Initialise distance
    dist = 1.0
    print(" Solving for the stationary Distribution.")

    while dist > tolerance:
        
        #Compute next period's distribution
        D_next = forward_step(D, a_prime_nn_index)
        
        #Compute Distance between iterations
        dist = np.max(np.abs(D-D_next))
        
        #Update iteration
        D = D_next
        
        #Update Counter
        iteration += 1
        
        #Print status updates
        if iteration % 500 == 0:
            print("     iteration = ", iteration, " Distance = ", f"{dist:.2e}")
    
    print(f"   Converged after {iteration} iterations")
    
    return D

#%% Household: Aggregates

def compute_aggregates(c, a_prime_nn, n, D):
    """
    Compute aggregates given policies and a stationary distribution.
    
    Parameters
    ----------
    c : ndarray, shape (N_e, N_a): Consumption Policy c(e,a)
    a_prime : ndarray, shape (N_e, N_a): Asset Policy a'(e,a)
    c : ndarray, shape (N_e, N_a): Consumption Policy c(e,a)
    D : ndarray, shape (N_e, N_a)
        Stationary distribution.
    
    Returns
    -------
    C : float
        Aggregate consumption. Freely floating as no goods market exists in the model.
    N : float
        Aggregate Labour.
    A : float
        Aggregate next-period assets.
        In stationary equilibrium with zero net supply, A(r*) = 0.
    """
    A = 0.0
    C = 0.0
    N = 0.0
    for e_index, e in enumerate(e_grid):
        for a_index, a in enumerate(a_grid):
            A += a_prime_nn[e_index, a_index]*D[e_index, a_index]
            C += c[e_index, a_index]*D[e_index, a_index]
            N += n[e_index, a_index]*D[e_index, a_index]
    return C, A, N

#%% Firm

def firm(r, w, Y):
    K = Y * ( alpha/(1-alpha) * w/r )**(1-alpha)
    N = Y * ( alpha/(1-alpha) * w/r )**(-alpha)
    
    return K, N

#%% Solve the Model

def solve_model(r, w, DaV_init=None, D_init=None):
    """
    Given factor prices r and w, obtain 
    - Household Policy Functions,
    - Firm Policy Functions, 
    - The stationary distribution,
    - model aggregates.
    """
    if DaV_init is None:
        #Heuristic initial DaV
        c_n = 0.5*((1+r)*a_grid + 0.5*w*e_grid[:,np.newaxis] + b*(1-e_grid[:,np.newaxis]))
        DaV_init = Uprime(c_n)*(1+r)
            
    if D_init is None:
        # Heuristic initial D_SE
        D_init = pi[:, np.newaxis] * np.ones_like(a_grid) / len(a_grid)
    
    #======================================
    #            Household
    #====================================== 
    
    #Solve for the households' Policy Functions
    DaV_n_plus_1, c, a_prime, n  = solve_policies(r, w, DaV_init)
    #Nearest-Neighbour matching of a'(e,a)
    a_prime_nn, a_prime_nn_index = nn_matching(a_grid, a_prime)
    
    #Stationary Distribution
    D_SE = stationary_Distribution(D_init, a_prime_nn_index)
    
    #Aggregates
    C, A, N_supply = compute_aggregates(c, a_prime_nn, n, D_SE)
    
    #======================================
    #            Firm
    #======================================
    
    K, N_demand = firm(r, w, Y=C)
    
    return DaV_n_plus_1, D_SE, c, a_prime, n, C, A, N_supply, K, N_demand

#%% Market Clearing

"""
IMPORTANT: The root-finding routine lacks precision and fails to converge to the desired 
           tolerance level for the same reasons as laid out in Huggett.py.
"""

def solve_equilibrium(r_init=0.05, w_init=2.0, tol=1e-8, max_iter=20, step_size = 0.1):
    """
    Newton-Raphson root-finder for (r, w) using forward-difference Jacobian in
    order to solve the market clearing conditions.
    """
    # Current guess vector
    x = np.array([r_init, w_init], dtype=float)

    for it in range(max_iter):
        # 1. Evaluate residuals at current guess
        print("=======================================================\n",
              f"Iteration {it}: Trying  r={x[0]:.6f} and w = {x[1]:.6f} \n",
              "=======================================================")
        F = market_clearing_conditions(*x)
        # 2. Compute maximum absolute residual (supremum-norm)
        normF = np.linalg.norm(F, ord=np.inf)

        print(f"Iteration {it} complete: r={x[0]:.6f}, w={x[1]:.6f}, residuals=[{F[0]:.2e}, {F[1]:.2e}], norm={normF:.2e}")
        
        # 3. Check convergence
        if normF < tol: #If converged, return prices
            print("===== Converged: Market Clearing Prices Found =====")
            return x[0], x[1]
        
        print("\n Computing 2 Forward Differences:")
        # 4. Approximate Jacobian at current guess
        J = forward_difference_jacobian(market_clearing_conditions, x, f0=F)

        # 5. Solve linear system J * dx = -F  → Newton step dx
        dx = np.linalg.solve(J, -F)
        
        # 6. Update guess
        x += step_size* dx
        print("Update Complete.")

    raise RuntimeError("Newton method did not converge")

def market_clearing_conditions(r, w):
    """
    Compute the residuals of the market clearing conditions:
    1) Labor market clearing: N_supply - N_demand = 0
    2) Capital market clearing: A - K = 0
    """
    _,_, _, _, _, _, A, N_supply, K, N_demand = solve_model(r, w)
    return np.array([
        A - K,              #Asset Market
        N_supply - N_demand #Labour Market
    ])

def forward_difference_jacobian(f, x, f0=None, h=1e-6):
    """
    Forward difference approximation of the Jacobian matrix.

    Parameters
    ----------
    f : callable
        Function returning a vector of residuals.
    x : ndarray
        Current guess for the variables (e.g. [r, w]).
    f0 : ndarray or None
          Precomputed residuals f(x). If None, will evaluate f(x).
    h : float
        Step size for forward difference.

    Returns
    -------
    J : ndarray
        Approximated Jacobian matrix.
    """
    # Evaluate f at the current point if not already done (base function values)
    if f0 is None:
        f0 = f(*x)
    # Problem dimensions: m equations, n variables
    n = len(x)
    m = len(f0)
    # Allocate Jacobian matrix
    J = np.zeros((m, n))

    # Loop over r and w and compute the partial derivatives
    for j in range(n):
        # Create a perturbed copy of x
        x_step = x.copy()
        x_step[j] += h # Forward step in variable j
        
        # Evaluate f at the perturbed point
        f1 = f(*x_step)
        # Approximate the derivative wrt variable j for all equations by the forward difference
        J[:, j] = (f1 - f0) / h

    return J

# Find market clearing prices
r_star, w_star = solve_equilibrium()
print(f" Equilibrium: r = {r_star}, w = {w_star}")

"""
Looking through the printed iterations, we can
find somewhat suitable candidates for r* and w*.
""" 
#%% Stationary Equilibrium
print("============================= \n",
      "Computing Equilibrium Objects \n",
      "=============================")
DaV_n_plus_1, D_SE, c, a_prime, n, C, A, N_supply, K, N_demand  = solve_model(0.047109, 1.627765)

