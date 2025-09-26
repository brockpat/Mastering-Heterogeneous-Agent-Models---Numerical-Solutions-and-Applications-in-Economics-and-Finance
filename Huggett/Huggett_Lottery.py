# -*- coding: utf-8 -*-
"""
Solves the Huggett Model by using the lottery approach to compute the stationary
distribution. This removes discontinuities and a market clearing interest rate
up to the desired tolerance level can be found.

The Rouwenhort Method is taken from 
    Rognlie (2022): https://github.com/shade-econ/nber-workshop-2022/blob/main/Lectures/Lecture%201%20Standard%20Incomplete%20Markets%20Steady%20State.ipynb

The lottery approach is taken from:
    Rognlie (2022): https://github.com/shade-econ/nber-workshop-2022/blob/main/Lectures/Lecture%201%20Standard%20Incomplete%20Markets%20Steady%20State.ipynb
    See also Young (2010)

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

#Discount Factor
beta = 0.95
#Risk Aversion
gamma = 2

#%% Utility Function 
def U(c):
    """
    CRRA utility.
    """
    if gamma == 1:
        utility = np.log(c)
    else:
        utility = c**(1-gamma)/(1-gamma)
    return utility

def Uprime(c):
    """
    Marginal utility.
    """
    marginal_utility = c**(-gamma)
    return marginal_utility

def Uprime_Inv(c):
    """
    Inverse of marginal utility.
    """
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
a_grid = discretize_assets(-2,40,150)
"""
Make sure that a_min is not too low. In other words, c>0 must always remain in the feasible
consumption set which is guaranteed if the Law of Motion of assets

        a_min = (1+r)*a_min + z_min - c 
        
has a solution for c>0.
"""

#%% Grid: Income

z_grid, pi, Pi = Rouwenhorst.discretize_income(0.6, #rho 
                                               0.5, #Variance of AR(1)
                                               6)   #N_z

#%% Solve for the Policy Functions

def solve_policies(r, DaV_n, epsilon = 1e-8):
    """
    Solve household policies by EGM given an interest rate r.

    Parameters
    ----------
    r : float
        Risk-free interest rate.
    DaV_n : ndarray, shape (N_z, N_a)
        Initial guess for derivative of the Value Function.
    epsilon : float, optional
        Convergence tolerance for DaV (implemented as epsilon*(1-beta)).
        
    Returns
    -------
    DaV_n_plus_1 : ndarray, shape (N_z, N_a)
        Converged derivative of the Value Function.
    c : ndarray, shape (N_z, N_a)
        Consumption policy c(z,a) defined on z in G_z and a in G_a.
    a_prime : ndarray, shape (N_z, N_a)
        Policy a'(z,a) defined on z in G_z and a in G_a.
    """
    
    print(" Solving for the Policy Functions.")
    #Initialise counter to track the number of iterations
    iteration = 0
    
    #Set tolerance and initialise distance to run the while loop
    dist = 1.0
    tolerance = epsilon *(1-beta) #epsilon_star
    
    while dist > tolerance:
    
        #Get next iteration's Value Function
        DaV_n_plus_1, c, a_prime  = backward_step(r, DaV_n)
        
        #Compute the difference between iterations (supremum norm)
        dist = np.max(np.abs(DaV_n_plus_1 - DaV_n))
        
        #Update Value Function
        DaV_n = DaV_n_plus_1.copy()
        
        #Update Counter
        iteration += 1
        
        #Print status updates
        if iteration % 500 == 0:
            print("     iteration = ", iteration, " Distance = ", f"{dist:.2e}")
    
    print(f"   Converged after {iteration} iterations")
    
    #Return Value Function Derivative and Policy Functions
    return DaV_n_plus_1, c, a_prime

#Function to iterate on the Bellman Equation
def backward_step(r, DaV_n):
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

#%% Compute the Stationary Distribution

def get_lottery(a, a_grid):
    """
    Compute lower index and lottery probabilities for off-grid asset choices.

    Parameters
    ----------
    a : ndarray (N_z, N_a)
        Off-grid policy function a'(z,a).
    a_grid : ndarray (N_a,)
        Asset grid G_a.

    Returns
    -------
    a_i : ndarray (N_z, N_a)
        Lower index of the bracketing asset grid point.
    a_pi : ndarray (N_z, N_a)
        Probability of going to the lower grid point.
    """
    # Clip to stay within grid bounds
    a = np.clip(a, a_grid[0], a_grid[-1])

    # Lower index
    a_i = np.searchsorted(a_grid, a) - 1
    a_i = np.clip(a_i, 0, len(a_grid) - 2)

    # Probability weight for lower grid point
    a_pi = (a_grid[a_i+1] - a) / (a_grid[a_i+1] - a_grid[a_i])

    return a_i, a_pi


def forward_policy_lottery(D, a_i, a_pi, Pi):
    """
    Forward iteration on distribution using lottery allocation.

    Parameters
    ----------
    D : ndarray (N_z, N_a)
        Current distribution over states (z,a).
    a_i : ndarray (N_z, N_a)
        Lower asset grid index.
    a_pi : ndarray (N_z, N_a)
        Probability for lower grid point.
    Pi : ndarray (N_z, N_z)
        Transition matrix for exogenous income states.

    Returns
    -------
    D_next : ndarray (N_z, N_a)
        Updated distribution.
    """
    N_z, N_a = D.shape
    D_next = np.zeros_like(D)

    for z in range(N_z):
        for a in range(N_a):
            mass = D[z, a]
            low = a_i[z, a]
            high = low + 1
            p_low = a_pi[z, a]
            p_high = 1.0 - p_low

            for zp in range(N_z):
                D_next[zp, low] += mass * p_low * Pi[z, zp]
                D_next[zp, high] += mass * p_high * Pi[z, zp]

    return D_next


def stationary_Distribution_lottery(D_init, a_prime, a_grid, Pi, tolerance=1e-8):
    """
    Compute stationary distribution using lottery interpolation.
    """
    a_i, a_pi = get_lottery(a_prime, a_grid)

    dist = 1.0
    iteration = 0
    D = D_init.copy()
    print(" Solving for the stationary distribution (lottery).")

    while dist > tolerance:
        D_next = forward_policy_lottery(D, a_i, a_pi, Pi)
        dist = np.max(np.abs(D_next - D))
        D = D_next
        iteration += 1
        if iteration % 500 == 0:
            print("     iteration =", iteration, " Distance =", f"{dist:.2e}")

    print(f"   Converged after {iteration} iterations")
    return D

#%% Compute Aggregates

#Aggregate Savings
def compute_aggregates(c, a_prime, D):
    """
    Compute aggregates given policies and a stationary distribution.
    
    Parameters
    ----------
    c : ndarray, shape (N_z, N_a)
    a_prime : ndarray, shape (N_z, N_a)
    D : ndarray, shape (N_z, N_a)
        Stationary distribution.
    
    Returns
    -------
    C : float
        Aggregate consumption. Freely floating as no goods market exists in the model.
    A : float
        Aggregate next-period assets.
        In stationary equilibrium with zero net supply, A(r*) = 0.
    """
    A = 0.0
    C = 0.0
    for z_index, z in enumerate(z_grid):
        for a_index, a in enumerate(a_grid):
            A += a_prime[z_index, a_index]*D[z_index, a_index]
            C += c[z_index, a_index]*D[z_index, a_index]
    return C, A

#%% Solve Model
def solve_model(r, DaV_init=None, D_init=None):
    """
    Given interest rate r, obtain Policy Functions, the stationary distribution
    and model aggregates.

    Parameters
    ----------
    r : float
        risk-free rate
    DaV_init : 
        Initial DaV guess; if None, uses a simple heuristic.
    D_init : n
        Initial stationary distribution guess; if None, start from product measure pi x Uniform(a).

    Returns
    -------
    result : dict
        {
            "r": r,
            "DaV": DaV,                # derivative of Value Function
            "c": c,                    # consumption policy on (z,a)
            "a_prime": a_prime,        # Policy a'(z,a)
            "D_SE": D_SE,              # stationary distribution
            "C": C,                    # aggregate consumption
            "A": A                     # aggregate assets A(r)
        }
    """
    if DaV_init is None:
        # Heuristic initial DaV
        DaV_init = Uprime(0.5*(a_grid + z_grid[:, np.newaxis])) * (1+r)

    if D_init is None:
        # Heuristic initial D_SE
        D_init = pi[:, np.newaxis] * np.ones_like(a_grid) / len(a_grid)

    #Obtain the Policy Functions
    DaV, c, a_prime = solve_policies(r, DaV_init)
    
    #Stationary Distribution with lottery method
    D_SE = stationary_Distribution_lottery(D_init, a_prime, a_grid, Pi)
    
    #Aggregates (using off-grid a_prime directly)
    C, A = compute_aggregates(c, a_prime, D_SE)

    #Return a dictionary storing all results
    return {
        "r": r,
        "DaV": DaV,
        "c": c,
        "a_prime": a_prime,
        "D_SE": D_SE,
        "C": C,
        "A": A
    }

#%% Find r* 

# Initial guess on market clearing rate r*
r = 0.03

history = []

#Set max iterations
max_iter = 40

#Set tolerance
tol=1e-6

#Set increment for finite differences
epsilon = 1e-4

for it in range(max_iter):
    # Evaluate A(r)
    print("=====================================\n",
          f"Iteration {it}: Trying  r={r:.9f} \n",
          "=====================================")
    out_r = solve_model(r)
    A_r = out_r["A"]
    print(f"Iteration {it} complete: r={r:.6f}, residual={A_r:.2e}")


    history.append((r, A_r))

    # Check convergence
    if abs(A_r) < tol:
        print(f"Converged in {it+1} iterations.")
        break 

    # Evaluate A(r+epsilon) for finite diff
    print("\n Computing Forward Difference:")
    out_r_eps = solve_model(r + epsilon)
    A_r_eps = out_r_eps["A"]

    # Finite difference derivative
    dA_dr = (A_r_eps - A_r) / epsilon

    # Newton update
    if dA_dr == 0:
        raise ZeroDivisionError("Derivative is zero — Newton step undefined.")
    
    #Update with damping
    r_new = r - 0.8 * A_r / dA_dr
    
    r = r_new

    print("Update Complete.")

print(f"The market clearing interest rate r* is {r}.")

#%% Equilibrium Objects
print("============================= \n",
      "Computing Equilibrium Objects \n",
      "=============================")
Stationary_Equilibrium = solve_model(r)