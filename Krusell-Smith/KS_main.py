# -*- coding: utf-8 -*-
"""
Run this file to solve the Krusell-Smith model. 

The main output are the coefficients a_0, a_1, b_0 and b_1
of the updating rule M(K,Z) for the given Markov chains of the TFP shock and
employment.

Firstly, this file will run 'KS_MC.py' 
to compute the Markov Transition Matrices of the Markov chains.
-   In build_krusell_smith_markov() you can change the respective
    parameters to change the Markov chains. 
    However, in this case in
    'K_SE.py' you will have to adjust Z_bar and in 
    'KS_main.py' you will
    have to adjust factor_prices() in order to account for a potentially
    different labour supply.   

Secondly, it will run
'KS_SE.py' to compute a stationary equilibrium.
-   You can change the values of z_g and z_b freely. 
    However, K_grid was initialised for the given z_g and z_b. 
    You may have to expand the grid to not hit the bounds of it during the
    simulation of the Krusell-Smith model (see more below).
    
    Moreover, in the computation of the stationary equilibrium we provide detailed
    explanations of how we initialised grids and solvers. For different z_g
    and z_b, these initialisations might have to be changed.
-   KS_SE builds on 'Aiyagari_Lottery.py'

Next,
based on the stationary equilibrium, 
the grid 'K_grid' for capital 'K' is constructed. 
We chose a 30% deviation around the stationary equilibrium value 'K_SE' in order to 
be far away enough of the upper and lower bound of 'K_grid' during the simulations.
We found these values by simple Trial & Error of running the simulation.
We also chose 50 gridpoints, i.e. 1/3 of the number of gridpoints in the asset
grid G_a. 

Next,
We initialise the Value Function derivative such that it is equal to the
one of the stationary equilibrium for all (K,Z) in G_K x G_Z.

With these preliminaries it is possible to solve for the Krusell-Smith model.
"""

#%% Set Path
directory = ".../Code/"

import os
os.chdir(directory)

#%% Libraries
import numpy as np

import pandas as pd
import statsmodels.api as sm

#%% Exogenous Shocks 

#run the file
import KS_MC as MC

# ========================
# Extract relevant results
# ========================

#Markov Transition Matrices
Pi_z, P_joint, cond_em_Pi = MC.Pi_z, MC.P_joint, MC.cond_em_Pi

#History of the TFP shock
Z_states = MC.Z_states

#History of employment shares
employment_share = MC.employment_share #1st element is u, 2nd is 1-u


#%% Compute the stationary equilibrium

#run the file
import KS_SE as SE

# ========================
# Extract relevant results
# ========================

#Asset Grid
a_grid = SE.a_grid 

#Employment Grid
e_grid = SE.e_grid #[0,1]

#TFP Shock Grid
Z_grid = SE.Z_grid #['g','b']

#Stationary Equilibrium objects
DaV_SE, D_SE, c_SE, a_prime_SE, C_SE, A_SE, K_SE = SE.DaV_n_plus_1, SE.D_SE, SE.c, SE.a_prime, SE.C, SE.A, SE.K

#Parameters
beta, gamma, alpha, delta = SE.beta, SE.gamma, SE.alpha, SE.delta

#Utility Functions
U, Uprime, Uprime_Inv = SE.U, SE.Uprime, SE.Uprime_Inv

#Distribution Functions
get_lottery = SE.get_lottery
forward_policy_lottery = SE.forward_policy_lottery


#%% Aggregate Capital K

#Grid
# Set deviations from K_SE and verify in the simulation whether K_min and K_max are too restrictive
K_grid = np.linspace(K_SE-0.3*K_SE,K_SE+0.3*K_SE, 50)

#Law of Motion of K
def M(K, Z, params):
    log_K_prime = np.where(Z == Z_grid[0],  # good state
                           params["a_0"] + params["a_1"] * np.log(K), #good state
                           params["b_0"] + params["b_1"] * np.log(K)  #bad state
                           ) 
    return log_K_prime

#%% Initialise Value Function Derivative.
#     Use the stationary one such that it is identical for all (K,Z) in G_K x G_Z
DaV_SE = np.tile(DaV_SE[:, :, np.newaxis, np.newaxis], (1, 1, len(K_grid), len(Z_grid)))

#%% Factor Prices
def factor_prices(K, Z):
    """
    Compute factor prices w_t and r_t given K_t and Z_t
    
    Parameters:
    K: array of capital values 
    Z: array of TFP values
    
    Returns:
    w: wage matrix (N_K, N_Z)
    r: interest rate matrix (N_K, N_Z)
    """
    
    # Create L array with shape (2,) that matches Z
    L = np.where(Z == Z_grid[0],  # good state
                 0.96,            # good state
                 0.9)             # bad state
    
    # Reshape for broadcasting: K becomes (25, 1), Z and L become (1, 2)
    K_2d = K[:, np.newaxis]  # shape (25, 1)
    Z_2d = Z[np.newaxis, :]  # shape (1, 2)
    L_2d = L[np.newaxis, :]  # shape (1, 2)
    
    # Compute r and w using broadcasting
    r = alpha * Z_2d * (L_2d / K_2d) ** (1 - alpha)
    w = (1 - alpha) * Z_2d * (K_2d / L_2d) ** alpha
    
    return w, r

#%% Conditional Expectation
def W_all(DaV, Pi_z, cond_em_Pi):
    """
    Computes W[e,a',K',Z] = E[ V(e',a',K',Z') | e,Z ] for
    (e,a',K',Z) in G_e x G_a x G_K x G_Z. 
    
    We use the marginal transition probabilities for the computation.
    One can alternatively use the joint probability.

    Parameters
    ----------
    DaV : ndarray, shape (2, N_a, N_K, 2)
        Value Function Derivative over (e, a, K, Z).
    Pi_z : (2,2) ndarray
        TFP transition matrix [[p_gg,p_gb],
                              [p_bg,p_bb]].
    cond_em_Pi : dict
        Conditional employment matrices keyed by 'gg','gb','bb','bg'.
            Pi_e[0,0] = Prob(e'=0 | e=0, z→z')
            Pi_e[0,1] = Prob(e'=1 | e=0, z→z')
            Pi_e[1,0] = Prob(e'=0 | e=1, z→z')
            Pi_e[1,1] = Prob(e'=1 | e=1, z→z')

    Returns
    -------
    W : ndarray, shape (2, n_a, n_K, 2)
        Conditional expectation of DaV.
    """
    ne, na, nK, nZ = DaV.shape
    W = np.zeros_like(DaV)

    Z_list = ['g','b']  # ordering consistent with Pi_z
    for z_idx, Z in enumerate(Z_list):
        for e in range(ne):
            total = np.zeros((na,nK))
            # iterate over future aggregate states
            for zp_idx, Zp in enumerate(Z_list):
                p_z = Pi_z[z_idx, zp_idx]
                Pi_e = cond_em_Pi[Z + Zp]
                #Iterate over future idiosyncratic states
                for eprime in (0,1):
                    p_e = Pi_e[e,eprime]
                    total += DaV[eprime,:,:,zp_idx] * p_z * p_e
            W[e,:,:,z_idx] = total
            
    return W

def precompute_W_interpolator(W, K_grid):
    """
    Precompute slopes and intercepts for linear interpolation of W in K-dimension
    for all (e,a',Z) in G_e x G_a x G_Z.
    
    This code is vectorised. For understanding, for some gridpoint (e,a',Z),
    this code is the same as
    for e in range(N_e):
        for ap in range(N_a):
            for z in range(N_z):
                for i in range(n_K-1):
                    slope[e,ap,i,z] = (W[e,ap,i+1,z] - W[e,ap,i,z]) / dK[i]
                    intercept[e,ap,i,z] = W[e,ap,i,z] - slope[e,ap,i,z]*K_grid[i]
    to compute the piece-wise linear function between two values K_i and K_{i+1}

    Parameters
    ----------
    W : ndarray, shape (ne, na, nK, nZ)
        Value function array.
    K_grid : ndarray, shape (nK,)
        Grid for K where W is defined on (support points)

    Returns
    -------
    interp : dict
        Dictionary containing:
        - 'slope' : array (ne,na,nK-1,nZ)
        - 'intercept' : array (ne,na,nK-1,nZ)
        - 'K_grid' : the original K_grid
    """
    # differences in K
    dK = np.diff(K_grid)  # shape (nK-1,)

    # compute slopes and intercepts, broadcasting across (ne,na,nZ)
    slope = (W[:,:,1:,:] - W[:,:,:-1,:]) / dK[np.newaxis, np.newaxis, :, np.newaxis]
    intercept = W[:,:,:-1,:] - slope * K_grid[:-1][np.newaxis, np.newaxis, :, np.newaxis]

    return {'slope': slope, 'intercept': intercept, 'K_grid': K_grid}

#%% Policy Functions

def solve_policies(DaV_n, params, epsilon = 1e-8):
    """
    Solve household policies by EGM.

    Parameters
    ----------
    DaV_n : ndarray, shape (N_e, N_a, N_K, N_Z)
        Initial guess for derivative of the Value Function.
    epsilon : float, optional
        Convergence tolerance for DaV.
    params: dict
            Parameters of the updating rule M(K,Z)
        
    Returns
    -------
    DaV_n_plus_1 : ndarray, shape (N_e, N_a, N_K, N_Z)
        Converged derivative of the Value Function.
    c : ndarray, shape (N_e, N_a, N_K, N_Z) 
            Consumption policy defined on (e,a,K,Z) in G_e x G_a x G_K x G_Z
    a_prime : ndarray, shape (N_e, N_a, N_K, N_Z) 
                a' policy defined on (e,a,K,Z) in G_e x G_a x G_K x G_Z
    """
    
    print("  Solving for the Households' Policy Functions.")
    #Initialise counter to track the number of iterations
    iteration = 0
    
    #Set tolerance and initialise distance to run the while loop
    dist = 1.0
    tolerance = epsilon
    
    #Compute factor prices (identical in every iteration for all 'params'),
    #   so can be computed outside of the while loop
    w,r = factor_prices(K_grid,Z_grid)

    #backwards iteration
    while dist > tolerance:
    
        #Get next iteration's Value Function
        DaV_n_plus_1, c, a_prime  = backward_step(DaV_n, w, r, params)
        
        #Compute the difference between iterations (supremum norm)
        dist = np.max(np.abs(DaV_n_plus_1 - DaV_n))
        
        #Update Value Function
        DaV_n = DaV_n_plus_1.copy()
        
        #Update Counter
        iteration += 1
        
        #Print status updates
        if iteration % 100 == 0:
            print("     iteration = ", iteration, " Distance = ", f"{dist:.2e}")
    
    print(f"   Converged after {iteration} iterations")
    
    #Return Value Function Derivative and Policy Functions
    return DaV_n_plus_1, c, a_prime


def backward_step(DaV_n, w, r, params):
    """
    One EGM backward step updating the derivative of the Value Function
    and policies.
    """
    
    #================================
    # Step 0: Conditional Expectation
    #================================
        
    #Compute conditional expectation on (e,a',K',Z) in G_e x G_a x G_K x G_Z
    W = W_all(DaV_n, Pi_z, cond_em_Pi)
    
    #Compute the Linear Interpolation in K' argument
    W_interpolator = precompute_W_interpolator(W, K_grid)
    
    # --- Vectorised EVALUATION of W_ip on (e, a', K'(K,Z), Z) ---
    """
    Here is pseudo-code of a loop which achieves the exact same as our
    vectorised interpolation evaluation:
        
    W_ip = np.zeros_like(W)
    for e_idx in range(len(e_grid)):
        for a_idx in range(len(a_grid)):
            for K_idx, K_val in enumerate(K_grid):
                for Z_idx, Z_val in enumerate(Z_grid):
                    K_prime = np.exp(M(K_val, Z_val, params))
                    W_ip[e_idx, a_idx, K_idx, Z_idx] = W_ip_evaluate(
                        e_idx, a_idx, K_prime, Z_idx, W_interpolator
                    )
                    
    Looping severly slows down runtime, so we used vectorisation (numpy broadcasting).
    """
    
    # 1) K' for every (K,Z): shape (nK, nZ)
    Kp = np.exp(M(K_grid[:, None], np.array(Z_grid)[None, :], params))
    
    # 2) For each (K',Z), find index i with K_grid[i] <= K' < K_grid[i+1]
    i = np.searchsorted(K_grid, Kp, side='right') - 1
    i = np.clip(i, 0, len(K_grid) - 2)  # shape (nK, nZ)
    
    # 3) Gather slopes/intercepts at those i for every (e,a)
    #    slope/intercept: (ne, na, nK-1, nZ)
    slope = W_interpolator['slope']
    intercept = W_interpolator['intercept']
    
    # expand i to (1,1,nK,nZ) so we can gather along axis=2
    i_exp = i[None, None, :, :]
    
    # take_along_axis returns (ne, na, nK, nZ)
    slope_sel = np.take_along_axis(slope, i_exp, axis=2)
    intercept_sel = np.take_along_axis(intercept, i_exp, axis=2)
    
    # 4) Final interpolated values: broadcast Kp to (1,1,nK,nZ)
    W_ip = slope_sel * Kp[None, None, :, :] + intercept_sel

    #====================================================================
    # Step 1: Compute c(e,a',K,Z) for (e,a',K,Z) in G_e x G_a x G_K x G_Z
    #====================================================================
    c = Uprime_Inv(beta * W_ip)
    
    #====================================================================
    # Step 2: Compute a(e,a',K,Z) for (e,a',K,Z) in G_e x G_a x G_K x G_Z
    #==================================================================== 
    # Reshape arrays for broadcasting
    e_4d = e_grid[:, np.newaxis, np.newaxis, np.newaxis]  # shape (n_e, 1, 1, 1)
    ap_4d = a_grid[np.newaxis, :, np.newaxis, np.newaxis] # shape (1, n_a, 1, 1)
    w_4d = w[np.newaxis, np.newaxis, :, :]                # shape (1, 1, n_K, n_Z)
    r_4d = r[np.newaxis, np.newaxis, :, :]                # shape (1, 1, n_K, n_Z)
    #Compute a(e,a',K,Z)
    a = (ap_4d - w_4d * e_4d + c)/(1 + r_4d - delta)
    
    #=================================================================================
    # Step 3: Invert a(e,a',K,Z) to a'(e,a,K,Z) for (e,a,K,Z) in G_e x G_a x G_K x G_Z
    #================================================================================= 
    #Initialise object a'(e,a,K,Z)
    a_prime = np.zeros_like(a)
    
    #Fix (e,K,Z) and interpolate across the asset dimension
    for e_idx in range(len(e_grid)):
        for K_idx in range(len(K_grid)):
            for Z_idx in range(len(Z_grid)):
                a_prime[e_idx,:, K_idx,Z_idx] = np.interp(a_grid, #evaluation points on G_a
                                               a[e_idx,:, K_idx,Z_idx], #x-coordinate: a
                                               a_grid) #y-coordinate: a'
    
    #======================================
    # Step 4: Enforce Constraint
    #====================================== 
    a_prime = np.maximum(a_prime, a_grid[0])
        
    #==================================================================
    # Step 5: Compute c(e,a,K,Z) for (e,a,K,Z) in G_e x G_a x G_K x G_Z
    #==================================================================
    c = -a_prime + (1+r_4d-delta)*a_grid[np.newaxis, :, np.newaxis, np.newaxis] +  w_4d*e_4d
    
    #=========================================
    # Step 6: Update Value Function Derivative
    #========================================= 
    DaV_n_plus_1 = Uprime(c) * (1+r_4d-delta)
    
    #Outputs
    return DaV_n_plus_1, c, a_prime   

#%% Get a'(e,a,K_t,Z_t) with linear interpolation

def linear_interp_K(aprime, K_val):
    """
    Linearly interpolate a'(e, a, K, Z) along the K dimension. Find the
    two neighbouring K_i and K_{i+1} of K_val and choose the appropriate
    convex combination to obtain the linear interpolation.

    Parameters
    ----------
    aprime : ndarray
        4D array of shape (len(e_grid), len(a_grid), len(K_grid), len(Z_grid)).
    K_val : float
        Value of K to interpolate at.

    Returns
    -------
    float
        Interpolated value of a'(e, a, K_val, Z).
    """

    # find bracketing indices
    j = np.searchsorted(K_grid, K_val) - 1
    j = np.clip(j, 0, len(K_grid)-2)  # ensure within valid range

    # weight in [0,1]
    lam = (K_val - K_grid[j]) / (K_grid[j+1] - K_grid[j])

    # two neighbours
    f0 = aprime[:, :, j,:]
    f1 = aprime[:, :, j+1, :]

    # linear interpolation
    return (1-lam)*f0 + lam*f1

#%% Simulation
def next_period(K, D, a_prime, t, params):
    """
    Compute the model's 
    next period actual K_{t+1}, 
    perceived K^M_{t+1} from the updating rule M(K,Z) and 
    distribution D_{t+1} from its Law of Motion using Young (2010).
    
    Inputs are K_t, D_t, the Policy Function a'(e,a,K,Z) and the updating rule M(K,Z).

    Parameters
    ----------
    K : float
        K_t.
    D : matrix (N_e, N_a)
        D_t.
    a_prime : (N_e, N_a, N_K, N_Z)
        a' Policy Function.
    t : int
        current point of time.
    params: dict
            Parameters of the updating rule M(K,Z).

    Returns
    -------
    K_next : float
        K_{t+1} from the underlying household distribution (actual K_{t+1}).
    K_M_next : float
        K_{t+1} from the updating rule M(K,Z) (perceived K_{t+1}).
    D_next : (N_e, N_a)
        D_{t+1}.
    """
    
    #Draw Z_{t+1}
    if Z_states[t+1] == 'g':
        Z_idx = 0
    else:
        Z_idx = 1
    
    #Get corresponding Markov Transition Matrix for employment:
    Pi_e = cond_em_Pi[Z_states[t] + Z_states[t+1]]
        
    #Get a'(e,a,K_t,Z_t) for K_t off-grid
    a_prime_ip = linear_interp_K(a_prime, K)
    a_prime_ip = a_prime_ip[:,:,Z_idx]
    
    #Compute actual K_{t+1} from underlying distribution D_t
    K_next = 0.0 
    for e_index, e in enumerate(e_grid):
        for a_index, a in enumerate(a_grid):
            K_next += a_prime_ip[e_index, a_index]*D[e_index, a_index]
            
    #Compute perceived K^M_{t+1} from updating rule
    K_M_next = np.exp(M(K, Z_states[t], params))
            
    #Compute D_{t+1} using Young (2010)
    a_i, a_pi = get_lottery(a_prime_ip,a_grid) #Lottery
    D_next = forward_policy_lottery(D, a_i, a_pi, Pi_e) #One forward Iteration 
    
    return K_next, K_M_next, D_next


def simulate(params, K0 = K_SE, D0 = D_SE, T=11_000):
    """
    Simulate the model for T periods under a given M(K,Z).
    
    Time period t=0 is initialised with the stationary equilibrium.
    
    Parameters
    ----------
    params: dict
            Parameters of the updating rule M(K,Z)
    K0 : float
        Initial capital level. CAUTION: This cannot be freely chosen (see below)
    D0 : array (N_e, N_a)
        Initial Household distribution
        CAUTION: K0 should be compatible with D0, i.e.
        K0 = 0.0
        for e in e_grid
            for a in a_grid
                K0 += a*D(e,a).
        In the stationary equilibrium D0 and K0 are compatible.
        
    T : int
        Time horizon.

    Returns
    -------
    Times Series {K_{t+1},K^M_{t+1}, K_t, Z_t}
    Value Function Derivative of Bellman Equation
    Consumption Policy Function c(e,a,K,Z)
    a' Policy Function a'(e,a,K,Z)

    """
    #Time Series
    TS = [] #Data
    
    #Policy Functions given updating rule 'params'
    DaV_n_plus_1, c, a_prime = solve_policies(DaV_SE, params)
    
    #Initialise K_t and D_t
    K_t = K0.copy()
    D_t = D0.copy()
    
    #Simulate time series
    for t in range(0,T-1):
        
        if t % 1_000 == 0:
            print(f"  Time Period: {t}")
            
        #Next Period t+1 given current period t
        K_next, K_M_next, D_next = next_period(K_t, D_t, a_prime, t, params)
        
        #Save Data
        TS.append([K_next, K_M_next, K_t, Z_states[t]]) #K_{t+1}, K^M_{t+1}, K_t, Z_t
        
        #Update Next iteration
        K_t = K_next.copy() 
        D_t = D_next.copy()
        
    
    return TS, DaV_n_plus_1, c, a_prime

#%% Regression

def regression(df, verbose = False):
    """
    Computes the coefficients of the regression log(K') = beta_0 + beta_1*log(K)
    """
    #Drop first 1_000 elements.
    df = df.iloc[1000:].copy()
    
    #Log Values
    df['log_y'] = np.log(df['K_t+1'])
    df['log_x'] = np.log(df['K_t'])
    
    # Add constant term for intercept
    X = sm.add_constant(df['log_x'])  # Adds a column of 1s for the intercept
    y = df['log_y']
    
    # Fit the regression model
    model = sm.OLS(y, X)
    results = model.fit()
    
    # Print the regression results
    if verbose:
        print(results.summary())
    
    # Get coefficients
    intercept = results.params['const']
    slope = results.params['log_x']
    if verbose:
        print(f"\nRegression equation: log(y) = {intercept:.4f} + {slope:.4f} * log(x)")
    
    return results


#%% Iterations over Updating Rule

iterations = 0 

#Initialise the parameters for updating rule M
params = {
"a_0": np.log(K_SE),
"a_1": 0.0,
"b_0": np.log(K_SE),
"b_1": 0.0,
}

print(" ========================\n",
      "==== AGGREGATE RISK ====\n",
      "========================")


for _ in range(60):
    print("-------------------------")
    print(f"Iteration = {iterations}")
    print("-------------------------")
    
    #Print current parameters
    print(f" a_0: {params['a_0']:.8f} \n a_1: {params['a_1']:.8f} \n b_0: {params['b_0']:.8f} \n b_1: {params['b_1']:.8f}")

    
    #Solve and Simulate the Krusell-Smith Model
    TS, DaV_n_plus_1, c, a_prime = simulate(params)

    #Pandas Dataframe of Time Series
    df = pd.DataFrame(TS, columns=['K_t+1', 'KM_t+1', 'K_t', 'Z_t'])
    
    #Extract Separate Time Series depending on realisation of Z_t
    df_g = df.loc[df['Z_t'] == 'g'][['K_t+1', 'K_t']]
    df_b = df.loc[df['Z_t'] == 'b'][['K_t+1', 'K_t']]
    
    #Run the two indivudal Regressions
    reg_g = regression(df_g)
    reg_b = regression(df_b)
    
    #Compute the supremum distance of the coefficients
    dist_coefs = np.max(
                    np.abs(np.array([
                             params['a_0'] - reg_g.params['const'],
                             params['a_1'] - reg_g.params['log_x'],
                             params['b_0'] - reg_b.params['const'],
                             params['b_1'] - reg_b.params['log_x']
                             ])
                        )
                    )
    print(f"   Distance Coefficients: {dist_coefs:.2e}")
    
    #If these values are close to the edges of K_grid, then K_grid has
    #   to be extended
    print(f"   K_max = {df['K_t'].max():.3f} , K_SE = {K_SE:.3f},  K_min = {df['K_t'].min():.3f}")
    
    #Update Parameters of Decision Rule
    """
    We need relatively strong damping. If an updating rule in some iteration
    will yield K' that are either too low or too large, this will have consequences
    for the interest rate 'r' such that the Policy Functioncs can no longer be obtained, i.e.
    the numerical algorithm solve_policies() fails.
    """
    params['a_0'] = 0.8*params['a_0'] + 0.2*reg_g.params['const']
    params['a_1'] = 0.8*params['a_1'] + 0.2*reg_g.params['log_x']

    params['b_0'] = 0.8*params['b_0'] + 0.2*reg_b.params['const']
    params['b_1'] = 0.8*params['b_1'] + 0.2*reg_b.params['log_x']
    
    #Update Counter
    iterations += 1
    
#%% Equilibrium
print(" -------------------\n",
      "Final updating rule:\n",
      "--------------------\n",
      f"a_0: {params['a_0']:.12f} \n a_1: {params['a_1']:.12f} \n b_0: {params['b_0']:.12f} \n b_1: {params['b_1']:.12f}")
print(f"  Convergence Level: {dist_coefs:.2e}")

#Regression Results. The R^2 indicate whether the converged updating rule
# is actually able to reliably forecast K' given K and Z
reg_g = regression(df_g, True)
reg_b = regression(df_b, True)


#Equilibrium Values
print("----------------------------")
print("Solution Krusell-Smith Model")
print("----------------------------")

a0 = reg_g.params['const']
a1 = reg_g.params['log_x']
b0 = reg_b.params['const']
b1 = reg_b.params['log_x']
print(f" a_0: {a0} \n a_1: {a1} \n b_0: {b0} \n b_1: {b1}")
