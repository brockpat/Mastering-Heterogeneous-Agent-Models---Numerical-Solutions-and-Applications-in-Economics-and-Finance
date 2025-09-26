# -*- coding: utf-8 -*-
"""
Implements the Rouwenhorst Method to discretise an AR(1) process into a finite-state
Markov Chain. This method assumes that the log of the underlying random variable
follows an AR(1) process. 

Taken from Rognlie (2022): https://github.com/shade-econ/nber-workshop-2022/blob/main/Lectures/Lecture%201%20Standard%20Incomplete%20Markets%20Steady%20State.ipynb
"""

#%% Libraries
import numpy as np

#%% Rouwenhorst Method
def compute_Pi(N, p):
    """
    Constructs a discrete-state Markov Transition Matrix for an AR(1) process using the Rouwenhorst method.
    
    Parameters
    ----------
    N : int
        Number of discrete states in the Markov chain. Must be ≥ 2.
    p : float
        Persistence probability (probability of remaining in the current state). Must satisfy 0 < p < 1.

    Returns
    -------
    Pi: An N × N column-stochastic transition matrix, where element (i, j) represents the probability 
        of transitioning from state i to state j.
    """
    # base case Pi_2
    Pi = np.array([[p, 1 - p],
                   [1 - p, p]])
    
    # recursion to build up from Pi_2 to Pi_N
    for n in range(3, N + 1):
        Pi_old = Pi
        Pi = np.zeros((n, n))
        
        Pi[:-1, :-1] += p * Pi_old
        Pi[:-1, 1:] += (1 - p) * Pi_old
        Pi[1:, :-1] += (1 - p) * Pi_old
        Pi[1:, 1:] += p * Pi_old
        Pi[1:-1, :] /= 2
        
    return Pi

def stationary_markov(Pi, tol=1e-14):
    """
    Computes the stationary distribution of a discrete-state Markov chain via 
    iterative matrix multiplication.

    The stationary distribution π satisfies π = πP, where P is the transition 
    matrix. This implementation uses power iteration, which is robust for 
    ergodic Markov chains and guarantees convergence to the unique
    stationary distribution when it exists.

    Parameters
    ----------
    Pi : numpy.ndarray
        A square column Markov Transition Matrix of shape (n, n), where
        Pi[i,j] represents the probability of transitioning from state i to state j.
        Must satisfy: (1) All elements ∈ [0,1], (2) Each row sums to 1.
    tol : float, optional
        Convergence tolerance threshold. Iteration stops when the supremum norm between
        successive iterations is < tol. Default: 1e-14.

    Returns
    -------
    numpy.ndarray
        The stationary distribution vector of shape (n,), normalised to sum to 1.
    """
    # Initialise a uniform distribution over all states
    n = Pi.shape[0]
    pi = np.full(n, 1/n)
    
    # update distribution using Pi until successive iterations differ by less than tol
    for _ in range(10_000):
        pi_new = Pi.T @ pi
        #Check convergence
        if np.max(np.abs(pi_new - pi)) < tol:
            return pi_new
        pi = pi_new
    
    print("Stationary Distribution did not converge.")
    return pi

def discretize_income(rho, sigma, n_s):
    """
    Discretises a log-normal AR(1) income process into a finite-state Markov chain using the Rouwenhorst method.

    The income process follows:
        log(z_t) = ρ·log(z_{t-1}) + ε_t
    and is discretised into 'n_s' states with:
        1. State values (z) representing income levels (not logs)
        2. A transition matrix (Pi) preserving the persistence ρ
        3. A stationary distribution (π)

    Parameters
    ----------
    rho : float
        Persistence parameter (autocorrelation) of the log-income process. Must satisfy |ρ| < 1.
    sigma : float
        Standard deviation of the log-income innovations. Must be positive.
    n_s : int
        Number of discrete states in the Markov chain. Must be ≥ 2.

    Returns
    -------
    z : numpy.ndarray
        Array of shape (n_s,) containing income levels (not logs) in ascending order,
        normalised to have mean 1 under the stationary distribution.
    pi : numpy.ndarray
        Stationary distribution vector of shape (n_s,), summing to 1.
    Pi : numpy.ndarray
        Markov Transition Matrix
    
    """
    # choose inner-switching probability p to match persistence rho
    p = (1+rho)/2
    
    # start with states from 0 to n_s-1, scale by standard deviation
    s = np.arange(n_s)
    alpha = 2*sigma/np.sqrt(n_s-1)
    s = alpha*s
    
    # obtain Markov transition matrix Pi and its stationary distribution
    Pi = compute_Pi(n_s, p)
    pi = stationary_markov(Pi)
    
    # get income level z and scale so that mean is 1
    z = np.exp(s) #s is log(z)
    z /= np.vdot(pi, z)
    
    return z, pi, Pi
