"""
POMDP Core Algorithms
Implements Algorithms 19.1, 19.2, 21.1 from "Algorithms for Decision Making"
"""

import numpy as np
from typing import List, Callable, Tuple, Any
import copy


class POMDP:
    """
    Algorithm 19.1: POMDP Structure
    Represents a Partially Observable Markov Decision Process.
    """
    
    def __init__(self, 
                 gamma: float,
                 S: List[Any],
                 A: List[Any],
                 O: List[Any],
                 T: Callable,
                 R: Callable,
                 O_func: Callable,
                 TRO: Callable = None):
        """
        Args:
            gamma: Discount factor
            S: State space (list of states)
            A: Action space (list of actions)
            O: Observation space (list of observations)
            T: Transition function T(s, a, s') -> probability
            R: Reward function R(s, a) -> reward
            O_func: Observation function O(a, s', o) -> probability
            TRO: Optional sampler (s, a) -> (s', r, o)
        """
        self.gamma = gamma
        self.S = S
        self.A = A
        self.O = O
        self.T = T
        self.R = R
        self.O_func = O_func
        self.TRO = TRO
        
        # Useful dimensions
        self.n_states = len(S)
        self.n_actions = len(A)
        self.n_observations = len(O)
    
    def state_index(self, s) -> int:
        """Get index of state s"""
        return self.S.index(s)
    
    def action_index(self, a) -> int:
        """Get index of action a"""
        return self.A.index(a)
    
    def obs_index(self, o) -> int:
        """Get index of observation o"""
        return self.O.index(o)


def update(b: np.ndarray, pomdp: POMDP, a: Any, o: Any) -> np.ndarray:
    """
    Algorithm 21.1: Bayesian Belief Update
    Updates belief state after taking action a and observing o.
    
    Args:
        b: Current belief (probability distribution over states)
        pomdp: POMDP instance
        a: Action taken
        o: Observation received
    
    Returns:
        Updated belief b'
    """
    S = pomdp.S
    T = pomdp.T
    O_func = pomdp.O_func
    
    b_prime = np.zeros_like(b)
    
    # Update belief for each next state
    for i_prime, s_prime in enumerate(S):
        po = O_func(a, s_prime, o)
        
        # Sum over current states
        prob_sum = sum(T(s, a, s_prime) * b[i] for i, s in enumerate(S))
        
        b_prime[i_prime] = po * prob_sum
    
    # Normalize
    total = np.sum(b_prime)
    
    if total > 0:
        b_prime = b_prime / total
    else:
        # If all zero (observation impossible), use uniform
        b_prime = np.ones_like(b_prime) / len(b_prime)
    
    return b_prime


def simulate_step(pomdp: POMDP, s: Any, a: Any) -> Tuple[Any, float, Any]:
    """
    Simulate one step of the POMDP.
    
    Args:
        pomdp: POMDP instance
        s: Current state
        a: Action to take
    
    Returns:
        Tuple of (next_state, reward, observation)
    """
    if pomdp.TRO is not None:
        return pomdp.TRO(s, a)
    
    # Sample next state
    T_probs = np.array([pomdp.T(s, a, s_prime) for s_prime in pomdp.S])
    T_probs = T_probs / T_probs.sum()
    
    # Ensure T_probs is valid by checking if the sum is zero
    if T_probs.sum() == 0:
        T_probs = np.ones(len(pomdp.S)) / len(pomdp.S)  # Assign uniform probabilities as fallback
    
    # Debugging: Check for NaN values in T_probs
    if np.isnan(T_probs).any():
        print(f"NaN detected in T_probs for state {s} and action {a}")
        print(f"T_probs: {T_probs}")
    
    s_prime_idx = np.random.choice(len(pomdp.S), p=T_probs)
    s_prime = pomdp.S[s_prime_idx]
    
    # Get reward
    r = pomdp.R(s, a)
    
    # Sample observation
    O_probs = np.array([pomdp.O_func(a, s_prime, o) for o in pomdp.O])
    O_probs = O_probs / O_probs.sum()
    
    # Ensure O_probs is valid by checking if the sum is zero
    if O_probs.sum() == 0:
        O_probs = np.ones(len(pomdp.O)) / len(pomdp.O)  # Assign uniform probabilities as fallback
    
    # Debugging: Check for NaN values in O_probs
    if np.isnan(O_probs).any():
        print(f"NaN detected in O_probs for action {a} and next state {s_prime}")
        print(f"O_probs: {O_probs}")
    
    o_idx = np.random.choice(len(pomdp.O), p=O_probs)
    o = pomdp.O[o_idx]
    
    return s_prime, r, o


def lookahead(pomdp: POMDP, b: np.ndarray, a: Any) -> float:
    """
    Simple lookahead utility for an action from a belief state.
    
    Args:
        pomdp: POMDP instance
        b: Belief state
        a: Action
    
    Returns:
        Expected immediate reward
    """
    return sum(b[i] * pomdp.R(s, a) for i, s in enumerate(pomdp.S))


def create_uniform_belief(pomdp: POMDP) -> np.ndarray:
    """
    Create a uniform belief over all states.
    
    Args:
        pomdp: POMDP instance
    
    Returns:
        Uniform belief array
    """
    return np.ones(pomdp.n_states) / pomdp.n_states


def create_belief_from_state(pomdp: POMDP, s: Any) -> np.ndarray:
    """
    Create a deterministic belief (all mass on one state).
    
    Args:
        pomdp: POMDP instance
        s: State
    
    Returns:
        Belief array with all mass on s
    """
    b = np.zeros(pomdp.n_states)
    b[pomdp.state_index(s)] = 1.0
    return b


class LookaheadAlphaVectorPolicy:
    """
    Policy represented by a set of alpha vectors.
    """
    
    def __init__(self, pomdp: POMDP, Gamma: List[np.ndarray]):
        """
        Args:
            pomdp: POMDP instance
            Gamma: List of alpha vectors (each is a tuple of (alpha_array, action))
        """
        self.pomdp = pomdp
        self.Gamma = Gamma  # List of (alpha, action) tuples
    
    def action(self, b: np.ndarray) -> Any:
        """
        Select best action for belief b.
        
        Args:
            b: Belief state
        
        Returns:
            Best action
        """
        if len(self.Gamma) == 0:
            # Fallback to first action
            return self.pomdp.A[0]
        
        # Find alpha vector with highest dot product with b
        best_value = -np.inf
        best_action = None
        
        for alpha_vec in self.Gamma:
            if isinstance(alpha_vec, tuple):
                alpha, action = alpha_vec
                value = np.dot(alpha, b)
                if value > best_value:
                    best_value = value
                    best_action = action
            else:
                # Just alpha vector, no associated action (shouldn't happen)
                value = np.dot(alpha_vec, b)
                if value > best_value:
                    best_value = value
                    best_action = self.pomdp.A[0]
        
        return best_action if best_action is not None else self.pomdp.A[0]
    
    def value(self, b: np.ndarray) -> float:
        """
        Compute value of belief b under this policy.
        
        Args:
            b: Belief state
        
        Returns:
            Value
        """
        if len(self.Gamma) == 0:
            return 0.0
        
        values = []
        for alpha_vec in self.Gamma:
            if isinstance(alpha_vec, tuple):
                alpha, _ = alpha_vec
                values.append(np.dot(alpha, b))
            else:
                values.append(np.dot(alpha_vec, b))
        
        return max(values) if values else 0.0

