# -*- coding: utf-8 -*-
"""
Solves the Huggett Model. EGM is used to obtain the Policy Functions. 
Forward Iteration with nearest-neighbour matching is used to compute the stationary distribution.

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
a_grid = discretize_assets(-2,40,150)
"""
Make sure that a_min is not too low. In other words, c>0 must always remain in the 
control set which is guaranteed if the Law of Motion of assets

        a_min = (1+r)*a_min + z_min - c 
        
has a solution for c>0.
"""

#%% Grid: Income

#Rouwenhorst Method to get grid 'z_grid', 
#   stationary distribution 'pi' and 
#   Markov Transition Matrix 'Pi'.
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
        a' will be off-grid, i.e. not in G_a, prior to nearest-neighbour matching.
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

def nn_matching(a_grid, a_prime):
    """
    Project off-grid savings a'(z,a) back to the discrete asset grid G_a using
    nearest-neighbour matching.

    Parameters
    ----------
    a_grid : ndarray, shape (N_a,)
        Asset grid G_a.
    a_prime : ndarray, shape (N_z, N_a)
        Off-grid Policy.

    Returns
    -------
    a_prime_nn : ndarray, shape (N_z, N_a)
        Policy shifted to the nearest gridpoint on a_grid.
    closest_idx : ndarray, shape (N_z, N_a)
        Respective index of a_grid for each a'(z,a).
    """
    
    #Compute the difference of each entry in a'(z,a) to G_a
    diff = np.abs(a_prime[..., np.newaxis] - a_grid)
    
    #find the index of the closest element in a_grid.
    closest_idx = np.argmin(diff, axis=-1)
    
    #a'(z,a) will map exclusively into G_a
    a_prime_nn = a_grid[closest_idx]
    
    return a_prime_nn, closest_idx

#%% Compute the Stationary Distribution
def forward_step(D, a_prime_nn_index):
    """
    One step of forward iteration on the distribution, transporting mass
    from (z,a) to (z',a').

    Parameters
    ----------
    D : ndarray, shape (N_z, N_a)
        Current cross-sectional distribution over states (z,a).
    a_prime_nn_index : ndarray, shape (N_z, N_a)
        Indices of next-period asset choices on a_grid for each (z,a).

    Returns
    -------
    D_next : ndarray, shape (N_z, N_a)
        Next-period distribution.
    """
    D_next = np.zeros_like(D)
    for a_index, a in enumerate(a_grid):
        for z_index, z in enumerate(z_grid):
            # Move the mass D[z,a] to (z', a'(z,a))
            D_next[:, a_prime_nn_index[z_index, a_index]] += D[z_index, a_index] * Pi[z_index, :]
            """
            More simply, but also more slowly, D_next can be computed using the following loop:
            for zp_index, zp in enumerate(z_grid):
                D_next[zp_index, a_prime_nn_index[z_index, a_index]] = D_next[zp_index, a_prime_nn_index[z_index, a_index]] + D[z_index, a_index] * Pi[z_index, zp_index]
            """

    return D_next


def stationary_Distribution(D, a_prime_nn_index, tolerance = 1e-10):
    """
    Compute the stationary distribution for a given Policy Function via forward
    iteration until convergence.

    Parameters
    ----------
    D : ndarray, shape (N_z, N_a)
        Initial guess for the distribution (must sum to 1).
    a_prime_nn_index : ndarray, shape (N_z, N_a)
        Indices of next-period asset choices on a_grid for each (z,a).
    tolerance : float, optional
        Sup‐norm stopping tolerance.

    Returns
    -------
    D_SE : ndarray, shape (N_z, N_a)
        Stationary distribution consistent with the policy a'(z,a).
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

#%% Compute Aggregates

#Aggregate Savings
def compute_aggregates(c, a_prime_nn, D):
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
        Aggregate consumption. Freely floating as no goods market exists.
    A : float
        Aggregate next-period assets.
        In stationary equilibrium with zero net supply, A(r*) = 0.
    """
    A = 0.0
    C = 0.0
    for z_index, z in enumerate(z_grid):
        for a_index, a in enumerate(a_grid):
            A += a_prime_nn[z_index, a_index]*D[z_index, a_index]
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
        Initial stationary distribution guess; if None, start from 
        product measure pi x Uniform(a).

    Returns
    -------
    result : dict
        {
            "r": r,
            "DaV": DaV,                # derivative of Value Function
            "c": c,                    # consumption policy on (z,a)
            "a_prime": a_prime,        # Policy a'(z,a)
            "a_prime_nn": a_prime_nn,  # on-grid Policy a'(z,a)
            "a_prime_idx": idx,        # gridpoint indices of on-grid Policy
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
    #Nearest-Neighbour matching of a'(z,a)
    a_prime_nn, a_prime_nn_index = nn_matching(a_grid, a_prime)
    
    #Stationary Distribution
    D_SE = stationary_Distribution(D_init, a_prime_nn_index)
    
    #Aggregates
    C, A = compute_aggregates(c, a_prime_nn, D_SE)

    #Return a dictionary storing all results
    return {
        "r": r,
        "DaV": DaV,
        "c": c,
        "a_prime": a_prime,
        "a_prime_nn": a_prime_nn,
        "a_prime_idx": a_prime_nn_index,
        "D_SE": D_SE,
        "C": C,
        "A": A
    }
#%% Solve the model for a grid of r
"""
Plotting A(r) for various r is good for intuition and also to seek for boundaries
if a Bisection method to compute r* is desired.
"""

#Choose a grid of interest rates
r_grid = np.linspace(0, 0.04, 10)

#Initialise objects
A_vals = np.empty_like(r_grid)
C_vals = np.empty_like(r_grid)

#Solve at each r, collect A(r)
for i, r_val in enumerate(r_grid):
    print(f"\n===== Solving at r = {r_val:.6f} =====")
    out = solve_model(r_val)
    A_vals[i] = out["A"]
    C_vals[i] = out["C"]

#Plot the aggregate asset supply curve A(r)
plt.figure()
plt.plot(r_grid, A_vals, marker='o')
plt.axhline(0.0, linestyle='--')
plt.xlabel("Interest rate r")
plt.ylabel("Aggregate assets A(r)")
plt.title("Huggett: Asset Supply vs. Interest Rate")
plt.tight_layout()
plt.show()

#%% Find r* with a simple Newton Method using forward differences
"""
It is much more efficient to use automatic differentiation in order
to compute A'(r), i.e. the derivative of aggregate asset supply with respect to the
interest rate. However, this is non-trivial - see Boehl (2023).

Instead, we opt for simple forward differences. Even though slower,
it illustrates nicely in a simple fashion how to solve for r*.

Alternatively, in this univariate case one could also resort to a Bisection
method which would also be much more accurate and efficient.

One could reuse the previous solutions as new initial guesses to
improve the runtime when solving for the Value Function, Policy Functions and 
stationary distribution. To keep the code as simple as possible, we do not 
implement this.


IMPORTANT: The root-finding routine lacks precision and fails to converge to the desired 
           tolerance level. This is due to the nearest-neighbour matching 
           approach for a_prime, as it introduces discontinuities when updating r. Hence,
           aggregate assets A(r) are neither differentiable with respect to r, nor are they
           continuous. The file Huggett_Lottery will resolve this issue.
"""

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
    r_new = r - 0.1 * A_r / dA_dr
    
    r = r_new

    print("Update Complete.")

print(f"The market clearing interest rate r* is approximately {r}.")

#%% Equilibrium Objects
print(" ============================ \n",
      "Computing Equilibrium Objects \n",
      "=============================")
Stationary_Equilibrium = solve_model(r)
