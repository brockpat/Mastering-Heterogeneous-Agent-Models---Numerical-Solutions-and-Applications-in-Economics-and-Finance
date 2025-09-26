# -*- coding: utf-8 -*-
"""
Auxiliary File for the Krusell-Smith model.

Computes a stationary equilibrium of the Krusell-Smith model by fixing the
TFP shock to a constant level.

This is equivalent to the Aiyagari model with exogenous labour supply. The 
files 'Aiyagari.py' and 'Aiyagari_Lottery.py' feature endogenous labour choice.

We solve the (asset) market clearing condition by guessing the interest rate 'r' which
enables us to back out 'K' and 'w'.
"""

#%% Set Path
directory = ".../Code/"

import os
os.chdir(directory)

#%% Libraries
import numpy as np

#Markov Chains for exogenous shocks
from KS_MC import build_krusell_smith_markov, cond_em_Trans

#%% Set Parameters as in Krusell & Smith (1998)

#Discount Factor
beta = 0.99
#Risk Aversion (log utility)
gamma = 1
# Elasticity of capital
alpha = 0.36
#Depreciation rate
delta = 0.025

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
"""
We set the lower bound of 'a_grid' arbitrarily. Too small lower bounds
might cause numerical issues,
since the Value Function derivative is very steep for small asset endowments.

We calibrated a_max and the number of gridpoints N_a
such that the househald mass in the distribution function 'D_SE' was evenly spread 
out. In other words, every gridpoint (e,a) only contains little probability mass
and the largest probability masses are located far away from the boundaries
of 'a_grid'.
If this isn't the case, then the grid is not satisfactory and too coarse. 
If e.g. 80% of households are located at one single gridpoint, then the households 
are not spread out enough. 

We found the upper bound and the number of gridpoints by Trial & Error such
that the number of required gridpoints were kept as small as possible.
"""
a_grid = discretize_assets(0.1,1_000,150)

#%% Grid: Exogenous State Variable
"""
Adjustments to the exogenous state variables in order to permit a stationary
equilibrium.
"""
#========= TFP 'Shock' =========
z_g, z_b = 1.01, 0.99 # From Krusell & Smith (1998)
Z_grid = np.array([z_g,z_b])

#Compute the Markov Transition Matrices
Pi_z, P_joint, cond_em_Pi = build_krusell_smith_markov()

#Fix Z_bar taken as the average of z_b and z_g (since Z is 50% in 'b' and 50% in 'g')
Z_bar = 0.5*z_g + 0.5*z_b

#========= Employment Shock =========
e_grid = np.array([0,1])

#For Transition Matrix of employment, take the average
Pi_e = np.array([[0,0], [0,0]])
for element in cond_em_Pi.values():
    Pi_e = Pi_e +  1/4*element
"""
Pi_e[0,0] = Prob(e'=0 | e=0)
Pi_e[0,1] = Prob(e'=1 | e=0)
Pi_e[1,0] = Prob(e'=0 | e=1) 
Pi_e[1,1] = Prob(e'=1 | e=1)
"""

#Stationary Distribution of employment shock
eigenvalues, eigenvectors = np.linalg.eig(Pi_e.T)

stationary_index = np.argmax(np.isclose(eigenvalues, 1.0))
pi_e = eigenvectors[:, stationary_index] 

pi_e = pi_e / np.sum(pi_e) #1st entry is unemployed, 2nd entry is employed

#Fix aggregate labour supply L_bar
L_bar = pi_e[1]

#%% Capital 'K' and wage 'w' from interest rate 'r'

def firm(r):
    """
    Given r and a fix 'L_bar', we can back out the capital demand 'K'.
    Given 'K' and 'L_bar', we can compute w
    """
    K = ( r/(alpha*Z_bar)*L_bar**(alpha-1) )**(1/(alpha-1))
    w = (1-alpha)*Z_bar*(K/L_bar)**(alpha)
    return K, w

#%% Households: Policy Functions

def solve_policies(r, DaV_n, epsilon = 1e-8):
    """
    Solve household policies by EGM given an interest rate r.

    Parameters
    ----------
    r : float
        interest rate.
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
    a_prime : ndarray, shape (N_e, N_a)
        Policy a'(e,a) defined on e in G_e and a in G_a.
    """
    
    print(" Solving for the Households' Policy Functions.")
    #Initialise counter to track the number of iterations
    iteration = 0
    
    #Set tolerance and initialise distance to run the while loop
    dist = 1.0
    tolerance = epsilon *(1-beta) #epsilon_star
    
    #backward iteration
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

def backward_step(r, DaV_n):
    """
    One EGM backward step updating the derivative of the value function
    and policies.
    """
    
    #Step 0: Get wage
    _, w = firm(r)
    
    #================================================
    # Step 1: Compute c(e,a') for (e,a') in G_e x G_a
    #================================================
    c = Uprime_Inv(beta * Pi_e @ DaV_n)
    
    #================================================
    # Step 2: Compute a(e,a') for (e,a') in G_e x G_a
    #================================================ 
    a = (a_grid[np.newaxis,:] - e_grid[:,np.newaxis]*w + c) / (1+r-delta)
    
    #==================================
    # Step 3: Invert a(e,a') to a'(e,a)
    #================================== 
    #Initialise object a'(e,a)
    a_prime = np.zeros((len(e_grid),len(a_grid)))
    
    #Fix rows (e gridpoint) and interpolate across the asset dimension
    for e_index in range(len(e_grid)):
        a_prime[e_index,:] = np.interp(a_grid, #evaluation points on G_a
                                       a[e_index,:], #x-coordinate: a
                                       a_grid) #y-coordinate: a'
    
    #======================================
    # Step 4: Enforce Constraint
    #====================================== 
    a_prime = np.maximum(a_prime, a_grid[0])
        
    #=========================================================
    # Step 5: Compute c(e,a)
    #=========================================================
    c = -a_prime + (1+r-delta)*a_grid[np.newaxis,:] +  w*e_grid[:,np.newaxis]
    
    #=========================================
    # Step 6: Update Value Function Derivative
    #========================================= 
    DaV_n_plus_1 = Uprime(c) * (1+r-delta)
    
    #Outputs
    return DaV_n_plus_1, c, a_prime

#%% Household: Stationary Distribution

def get_lottery(a, a_grid):
    """
    Find lower gridpoint index and lottery probability for each off-grid asset value a'.
    """
    # Ensure values within bounds (avoid index overflow)
    a = np.clip(a, a_grid[0], a_grid[-1])

    # Lower index of bracket
    a_i = np.searchsorted(a_grid, a) - 1
    a_i = np.clip(a_i, 0, len(a_grid) - 2)  # ensure room for +1

    # Probability of going to lower gridpoint
    a_pi = (a_grid[a_i+1] - a) / (a_grid[a_i+1] - a_grid[a_i])

    return a_i, a_pi

def forward_policy_lottery(D, a_i, a_pi, Pi):
    """
    Forward step for stationary distribution using lottery interpolation.
    
    Parameters
    ----------
    D : ndarray (N_e, N_a)
        Current distribution over states (employment, assets)
    a_i : ndarray (N_e, N_a)
        Lower asset grid index for lottery
    a_pi : ndarray (N_e, N_a)
        Probability of going to lower gridpoint
    Pi : ndarray (N_e, N_e)
        Transition matrix for exogenous states
    """
    N_e, N_a = D.shape
    D_end = np.zeros_like(D)

    for e in range(N_e):
        for a in range(N_a):
            mass = D[e, a]
            low = a_i[e, a]
            high = low + 1
            p_low = a_pi[e, a]
            p_high = 1.0 - p_low

            # allocate mass across asset grid points for next period
            for e_next in range(N_e):
                D_end[e_next, low] += mass * p_low * Pi[e, e_next]
                D_end[e_next, high] += mass * p_high * Pi[e, e_next]

    return D_end

def stationary_Distribution_lottery(D_init, a_prime, a_grid, Pi, tolerance=1e-6):
    """
    Stationary distribution computation using lottery interpolation.
    """
    # Compute lottery indices and probabilities
    a_i, a_pi = get_lottery(a_prime, a_grid)

    iteration = 0
    dist = 1.0
    D = D_init.copy()
    print(" Solving for stationary distribution (lottery).")

    while dist > tolerance:
        D_next = forward_policy_lottery(D, a_i, a_pi, Pi)
        dist = np.max(np.abs(D_next - D))
        D = D_next
        iteration += 1
        if iteration % 500 == 0:
            print("     iteration =", iteration, " Distance =", f"{dist:.2e}")

    print(f"   Converged after {iteration} iterations")
    return D

#%% Household: Aggregates

def compute_aggregates(c, a_prime, D):
    """
    Compute aggregates given policies and a stationary distribution.
    
    Parameters
    ----------
    c : ndarray, shape (N_e, N_a)
    a_prime : ndarray, shape (N_e, N_a)
    D : ndarray, shape (N_e, N_a)
        Stationary distribution.
    
    Returns
    -------
    C : float
        Aggregate consumption. Freely floating, will be verified against the aggregate resource constraint.
    A : float
        Aggregate assets.
        In stationary equilibrium with zero net supply, A(r*) = 0.
    """
    A = 0.0
    C = 0.0
    for e_index, e in enumerate(e_grid):
        for a_index, a in enumerate(a_grid):
            A += a_prime[e_index, a_index]*D[e_index, a_index]
            C += c[e_index, a_index]*D[e_index, a_index]
    return C, A

#%% Solve the Model

def solve_model(r, DaV_init=None, D_init=None):
    """
    Solve the model for a given interest rate r.
    """
    
    #Wage given interest rate r (recall that L = L_bar is fix)
    _, w = firm(r)
    
    if DaV_init is None:
        #Heuristic initial DaV
        c_n = 0.5*((1+r-delta)*a_grid + 0.5*w*e_grid[:,np.newaxis])
        DaV_init = Uprime(c_n)*(1+r-delta)
            
    if D_init is None:
        # Heuristic initial D_SE
        D_init = pi_e[:, np.newaxis] * np.ones_like(a_grid) / len(a_grid)
    
    #======================================
    #            Household
    #====================================== 
    
    #Solve for the households' Policy Functions
    DaV_n_plus_1, c, a_prime  = solve_policies(r, DaV_init)
    
    #Stationary Distribution
    D_SE = stationary_Distribution_lottery(D_init, a_prime, a_grid, Pi_e) 
    
    #Aggregates
    C, A = compute_aggregates(c, a_prime, D_SE)
    
    #======================================
    #            Firm
    #======================================
    
    K, _ = firm(r)
    
    return DaV_n_plus_1, D_SE, c, a_prime, C, A, K

#%% Market Clearing

def solve_equilibrium(r_init=0.0347, tol=1e-8, max_iter=50, step_size = 1):
    """
    Newton-Raphson root-finder for (r) using forward-difference Jacobian in 
    order to solve the market clearing condition.
    
    We began with a warm-up routine to generate a good initial guess 'r_init' to 
    speed up convergence. Specifically, we computed the market clearing residual for
    r=[1%,2%,3%,4%] and found the smallest residual at r=3%. Starting from there, 
    we used a small step size (e.g. 10%) until the algorithm reached a region where 
    the loss was on the order of 10^1. This occurred at r=0.0347. Using this 
    value as the new starting point, we then switched to the maximum step size, 
    and the algorithm converged. This procedure is inspired from learning 
    rate warm-ups used in Deep Learning.
    
    It is important that beta*(1+r-delta), where r-delta is the effective
    interest rate for the households, remains below one so that the precautionary
    savings motive subsides for households with large wealth levels. This applies 
    to the initialisation 'r_init' as well as to all iterations. Else-wise, 
    'A' can diverge and the algorithm fails to converge.
    """
    # Current guess vector
    x = np.array([r_init], dtype=float)
    
    print(" ================================\n",
          "==== STATIONARY EQUILIBRIUM ====\n",
          "================================")
    for it in range(max_iter):
        
        # 1. Evaluate residuals at current guess
        print(" ==================================\n",
              f"Iteration {it}: Trying  r={x[0]:.9f} \n",
              "==================================")
        F = market_clearing_conditions(*x)
        # 2. Compute residual norm
        normF = np.linalg.norm(F, ord=np.inf)
        print(f"Iteration {it} complete: r={x[0]:.9f}, residual={F[0]:.2e} \n")
        
        # 3. Check convergence
        if normF < tol:
            print("===== Converged: Market Clearing Price Found =====")
            return x[0]
        
        #Search for better values if not converged.        
        print("Computing Forward Difference:")
        # 4. Approximate Jacobian at current guess
        J = forward_difference_jacobian(market_clearing_conditions, x, f0=F)

        # 5. Solve linear system J * dx = -F  → Newton step dx
        dx = np.linalg.solve(J, -F)
        
        # 6. Update guess
        x += step_size* dx
        print("Update Complete.")

    raise RuntimeError("Newton method did not converge")

def market_clearing_conditions(r):
    """
    Compute the residual of the market clearing condition:
    1) Capital market clearing: A - K = 0
    """
    _, _, _, _, _, A, K = solve_model(r)
    return np.array([
        A - K
    ])


def forward_difference_jacobian(f, x, f0=None, h=1e-10):
    """
    Forward difference approximation of the Jacobian matrix.
    
    h is small because the model can be very sensitive to interest
    rate changes.

    Parameters
    ----------
    f : callable
        Function returning a vector of residuals.
    x : ndarray
        Current guess for the variables (e.g. [r]).
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

    # Loop over r and approximate the derivative    
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
r_star = solve_equilibrium()
print(f" Equilibrium: r* = {r_star}")
print(f"Effective rate: r-delta = {r_star-delta}")

#%% Stationary Equilibrium
print(" =============\n",
      "Final Outputs\n",
      "=============")
DaV_n_plus_1, D_SE, c, a_prime, C, A, K  = solve_model(r_star)

print(f"  Aggregate Resource Residual: {C + K - (1-delta)*K - Z_bar*K**alpha * L_bar**(1-alpha)} \n",
      "which ought to be close to 0 due to Walras' Law ")
