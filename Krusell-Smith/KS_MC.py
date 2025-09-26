# -*- coding: utf-8 -*-
"""
Auxiliary File for the Krusell-Smith model.

Constructs the Markov chains and Transition Matrices of the exogenous shocks.

Draws a TFP shock series and the corresponding employment shock series
"""

#%% Libraries

import numpy as np
from collections import Counter

#%% Markov Transition Matrices

def build_krusell_smith_markov(
    dur_good=8, dur_bad=8,
    u_g=0.04, u_b=0.10,
    dur_u_g=1.5, dur_u_b=2.5,
    ratio_gb_vs_bb=1.25, ratio_bg_vs_gg=0.75
):
    """
    Construct the Markov transition matrices of the exogenous shocks.

    Parameters
    ----------
    dur_good : float
        Average duration (in quarters) of a 'good' aggregate state.
    dur_bad : float
        Average duration (in quarters) of a 'bad' aggregate state.
    u_g : float
        Unemployment rate in the good state.
    u_b : float
        Unemployment rate in the bad state.
    dur_u_g : float
        Average unemployment length (quarters) in the good state.
    dur_u_b : float
        Average unemployment length (quarters) in the bad state.
    ratio_gb_vs_bb : float
        Calibration restriction: 
            probabiilty of remaining unemployed when z changes from good to bad 
            vs when z remains b
        π_gb^00 / π_gb = ratio_gb_vs_bb * (π_bb^00 / π_bb).
    ratio_bg_vs_gg : float
        Calibration restriction:
            probabiilty of remaining unemployed when z changes from bad to good
            vs. when z remains good
        π_bg^00 / π_bg = ratio_bg_vs_gg * (π_gg^00 / π_gg).

    Returns
    -------
    Pi_z : (2,2) ndarray
        TFP transition matrix: [(g→g), (g→b)
                                (b→g), (b→b)]
    P_joint : (4,4) ndarray
        Joint transition matrix over (z,e).
        State order: [(g,0),(g,1),(b,0),(b,1)]. Hence,
        P_joint = [(g,0)→(g,0), (g,0)→(g,1), (g,0)→(b,0), (g,0)→(b,1)
                   (g,1)→(g,0), (g,1)→(g,1), (g,1)→(b,0), (g,1)→(b,1)
                   (b,0)→(g,0), (b,0)→(g,1), (b,0)→(b,0), (b,0)→(b,1)
                   (b,1)→(g,0), (b,1)→(g,1), (b,1)→(b,0), (b,1)→(b,1) ]
            
    cond_em_Pi : dict
        Dictionary of conditional employment Transition Matrix { 'gg','gb','bb','bg' }
        depending on the evolution z→z'.
    """

    # ==========================================
    # TFP shock transition matrix
    # ==========================================
    # probability of remaining in good state
    pi_gg = 1 - 1/dur_good
    # probability of remaining in bad state
    pi_bb = 1 - 1/dur_bad
    # probability of changing from g to b
    pi_gb = 1 - pi_gg
    # probability of changing from b to g
    pi_bg = 1 - pi_bb

    # Assemble aggregate shock transition matrix
    Pi_z = np.array([[pi_gg, pi_gb], #g→g, g→b
                     [pi_bg, pi_bb]])#b→g, b→b

    # ==========================================
    # Unemployment persistence
    # ==========================================
    # Average unemployment rate in good TFP state
    pU_g = 1 - 1/dur_u_g
    # Average unemployment rate in bad TFP state
    pU_b = 1 - 1/dur_u_b

    # ==========================================
    # System of equations for joint transition
    # ==========================================
    # Unknowns: π_zz'^00, π_zz'^10 for each aggregate transition (8 unknowns)
    # Order of unknowns: [gg00, gg10, gb00, gb10, bb00, bb10, bg00, bg10]
    A = []
    b = []

    # Flow consistency conditions
    A.append([u_g,(1-u_g),0,0,0,0,0,0]); b.append(u_g*pi_gg)  # g→g
    A.append([0,0,u_g,(1-u_g),0,0,0,0]); b.append(u_b*pi_gb)  # g→b
    A.append([0,0,0,0,u_b,(1-u_b),0,0]); b.append(u_b*pi_bb)  # b→b
    A.append([0,0,0,0,0,0,u_b,(1-u_b)]); b.append(u_g*pi_bg)  # b→g

    # Unemployment duration constraints
    A.append([1,0,1,0,0,0,0,0]); b.append(pU_g)
    A.append([0,0,0,0,1,0,1,0]); b.append(pU_b)

    # Ratio restrictions
    row = [0,0,1/pi_gb,0, -ratio_gb_vs_bb/pi_bb,0,0,0]
    A.append(row); b.append(0.0)
    row = [-(ratio_bg_vs_gg/pi_gg),0,0,0,0,0,1/pi_bg,0]
    A.append(row); b.append(0.0)

    # ==========================================
    # Solve linear system
    # ==========================================
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x = np.linalg.solve(A,b)
    gg00, gg10, gb00, gb10, bb00, bb10, bg00, bg10 = x

    # Recover complementary probabilities
    gg01, gg11 = pi_gg-gg00, pi_gg-gg10
    gb01, gb11 = pi_gb-gb00, pi_gb-gb10
    bb01, bb11 = pi_bb-bb00, pi_bb-bb10
    bg01, bg11 = pi_bg-bg00, pi_bg-bg10

    # ==========================================
    # Assemble joint Markov transition matrix
    # ==========================================
    # State order: (g,0),(g,1),(b,0),(b,1): 
        # E.g. gb01: z=g, z'=b, e=0, e'=1
    P_joint = np.array([
        [gg00, gg01, gb00, gb01],  # from (g,0)
        [gg10, gg11, gb10, gb11],  # from (g,1)
        [bg00, bg01, bb00, bb01],  # from (b,0)
        [bg10, bg11, bb10, bb11],  # from (b,1)
    ])

    # ==========================================
    # Conditional employment transition matrices
    # ==========================================
    cond_em_Pi = {
        'gg': cond_em_Trans(pi_gg, gg00, gg10),
        'gb': cond_em_Trans(pi_gb, gb00, gb10),
        'bb': cond_em_Trans(pi_bb, bb00, bb10),
        'bg': cond_em_Trans(pi_bg, bg00, bg10),
    }
    
    # ==========================================
    # Sanity checks
    # ==========================================
    assert np.allclose(Pi_z.sum(axis=1), 1), "Rows of Pi_z must sum to 1."
    assert np.allclose(P_joint.sum(axis=1), 1), "Rows of joint matrix must sum to 1."

    return Pi_z, P_joint, cond_em_Pi

# Conditional employment Markov Transition Matrix
def cond_em_Trans(pi_zz, pi00, pi10):
    """
    Build the conditional employment transition matrix P(e'|e, z→z').

    Parameters
    ----------
    pi_zz : float
        Probability of aggregate transition z→z'.
    pi00 : float
        Joint probability of (e=0 → e'=0) given z→z'.
    pi10 : float
        Joint probability of (e=1 → e'=0) given z→z'.

    Returns
    -------
    P_e : (2,2) ndarray
        Conditional transition matrix of employment:
        Rows = current e in {0,1}, Cols = next e' in {0,1}.
        For example:
            P_e[0,0] = Prob(e'=0 | e=0, z→z')
            P_e[0,1] = Prob(e'=1 | e=0, z→z')
            P_e[1,0] = Prob(e'=0 | e=1, z→z')
            P_e[1,1] = Prob(e'=1 | e=1, z→z')
    """
    return np.array([
        [pi00/pi_zz, (pi_zz - pi00)/pi_zz],
        [pi10/pi_zz, (pi_zz - pi10)/pi_zz],
    ])

#%% Run
Pi_z, P_joint, cond_em_Pi = build_krusell_smith_markov()

print(" =======================\n",
      "==== MARKOV CHAINS ====\n",
      "=======================")

#Print Transition Matrices
np.set_printoptions(precision=6, suppress=True)
print("Pi_z (TFP):\n", Pi_z)
print("\nJoint P(z,e)->(z',e') in order [(g,0),(g,1),(b,0),(b,1)]:")
print(P_joint)
print("\nP(e'|e, g->g):")
print(cond_em_Pi['gg'])
print("\nP(e'|e, g->b):")
print(cond_em_Pi['gb'])
print("\nP(e'|e, b->b):")
print(cond_em_Pi['bb'])
print("\nP(e'|e, b->g):")
print(cond_em_Pi['bg'])

#%% Simulate TFP Shock Z

def sample_TFP_shock(Pi_z, n_steps, initial_state = 'g', seed=2718281828):
    """
    Simulate a sequence of aggregate TFP states ('good' or 'bad') 
    from a two-state Markov chain.

    Parameters
    ----------
    Pi_z : ndarray (2x2)
        Transition matrix for aggregate TFP states:
            [[p_gg, p_gb],
             [p_bg, p_bb]]
        where p_gg = Prob(z'='g' | z='g'), etc.
    n_steps : int
        Number of periods (steps) to simulate.
    initial_state : str, optional
        Starting state of the chain ('g' for good or 'b' for bad).
        Default is 'g'.
    seed : int or None, optional
        Random seed for reproducibility. If None, randomness is uncontrolled.

    Returns
    -------
    sequence : list of str
        Simulated sequence of TFP states of length `n_steps`,
        each element is either 'g' (good) or 'b' (bad).
    """
    
    # Set the seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Convert Markov Transition Matrix Pi_z to dictionary format for row accessing
    # 'g': row for good state transitions [p_gg, p_gb]
    # 'b': row for bad state transitions [p_bg, p_bb]
    Pi_z = Pi_z.copy()
    Pi_z = {'g':Pi_z[0,:], 'b': Pi_z[1,:]}
        
    # Track current state and simulated sequence
    current_state = initial_state
    sequence = [current_state]
    
    # Iterate through time steps and sample next state
    for _ in range(n_steps - 1):
        # Get transition probabilities for current state
        probs = Pi_z[current_state]
        
        # Sample next state (0 = stay, 1 = switch)
        # For state 'g': [p_gg, p_gb] = [stay in g, switch to b]
        # For state 'b': [p_bg, p_bb] = [switch to g, stay in b]
        next_state_idx = np.random.choice([0, 1], p=probs)
        
        if current_state == 'g':
            current_state = 'b' if next_state_idx == 1 else 'g'
        else:  # current_state == 'b'
            current_state = 'g' if next_state_idx == 0 else 'b'
        
        sequence.append(current_state)
    
    return sequence

def calculate_average_run_lengths(sequence):
    """
    Compute average run lengths of consecutive 'good' (g) and 'bad' (b) states 
    of the simulated TFP shock

    Parameters
    ----------
    sequence : list of str
        Simulated sequence of TFP states, e.g., ['g','g','b','b','b','g',...]

    Returns
    -------
    results : dict
        Dictionary with statistics for each state:
        {
          'g': {
              'average': float,  # average run length of 'g'
              'runs': list[int], # list of individual run lengths of 'g'
              'count': int       # number of runs of 'g'
          },
          'b': {
              'average': float,  # average run length of 'b'
              'runs': list[int], # list of individual run lengths of 'b'
              'count': int       # number of runs of 'b'
          }
        }
    """
    
    # track first observed state
    current_state = sequence[0]
    # initialise run length
    current_run_length = 1
    # containers for run lengths
    g_runs = []
    b_runs = []
    
    # Iterate through the sequence and count consecutive runs
    for i in range(1, len(sequence)):
        if sequence[i] == current_state:
            current_run_length += 1
        else:
            # Record the completed run
            if current_state == 'g':
                g_runs.append(current_run_length)
            else:
                b_runs.append(current_run_length)
            
            # Reset for new state
            current_state = sequence[i]
            current_run_length = 1
    
    # Don't forget the last run
    if current_state == 'g':
        g_runs.append(current_run_length)
    else:
        b_runs.append(current_run_length)
    
    # Calculate averages
    avg_g = np.mean(g_runs) if g_runs else 0
    avg_b = np.mean(b_runs) if b_runs else 0
    
    return {
        'g': {'average': avg_g, 'runs': g_runs, 'count': len(g_runs)},
        'b': {'average': avg_b, 'runs': b_runs, 'count': len(b_runs)}
    }

#Simulate TFP Shock Z
Z_states = sample_TFP_shock(Pi_z, 11_000)

print(" ==================\n",
      "Sampling TFP Shock\n",
      "==================")

#====== Sanity Checks ======
# Count occurrences using Counter
counts = Counter(Z_states)
print(f"Count of 'g': {counts['g']}")
print(f"Count of 'b': {counts['b']}")

# Average length of the sequence
avg_length = calculate_average_run_lengths(Z_states)
print(f"Average Length of g: {avg_length['g']['average']:.6f}")
print(f"Average Length of b: {avg_length['b']['average']:.6f}")

#%% Simulate employment share

def sample_employment_share(Z_states):
    """
    Simulate the evolution of the employment shares over time, given a 
    sequence of aggregate TFP states.
    
    Initialise the employment share such that its distribution only depends
    on Z_t.
    
    Parameters
    ----------
    Z_states : list of str
        Sequence of aggregate TFP states, e.g. ['g','g','b','b',...].
        Each element must be 'g' (good state) or 'b' (bad state).

    Returns
    -------
    employment_share : list of ndarray
        List of arrays of length 2 representing the population distribution:
            [u_t, 1-u_t] where
                u_t = unemployment share at time t
        Length of the list equals len(Z_states).
    """
    
    # Initialise employment share at t=0 depending on initial TFP stat
    employment_share = []
    
    if Z_states[0] == 'g':
        employment_share.append(np.array([0.04, 0.96])) #u, 1-u
    else:
        employment_share.append(np.array([0.1, 0.9])) #u, 1-u
    
    # Iterate through TFP states and update employment distribution
    for t in range(1,len(Z_states)):
        # Identify aggregate transition z→z'
        z_zp = Z_states[t-1] + Z_states[t]
        
        # Update employment distribution:
        # π_{t+1} = π_{t} × P_e(z→z')
        # where π = [u, e], and P_e is the conditional transition matrix
        employment_share.append(employment_share[t-1] @ cond_em_Pi[z_zp])
        
    return employment_share

employment_share = sample_employment_share(Z_states)
