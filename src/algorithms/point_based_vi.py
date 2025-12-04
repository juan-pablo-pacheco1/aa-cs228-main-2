"""
Point-Based Value Iteration for POMDPs
Implements Algorithms 21.6, 21.7, 21.8 from "Algorithms for Decision Making"
"""

import numpy as np
from typing import List, Tuple, Any
from .pomdp import POMDP, update, LookaheadAlphaVectorPolicy
import copy


def baws_lowerbound(pomdp: POMDP) -> np.ndarray:
    """
    Compute a simple lower bound for initialization.
    Uses the worst-case immediate reward for each state.
    
    Args:
        pomdp: POMDP instance
    
    Returns:
        Lower bound alpha vector
    """
    alpha = np.zeros(pomdp.n_states)
    
    for i, s in enumerate(pomdp.S):
        # Worst reward across all actions
        worst_reward = min(pomdp.R(s, a) for a in pomdp.A)
        # Bound: worst immediate reward / (1 - gamma)
        alpha[i] = worst_reward / (1.0 - pomdp.gamma) if pomdp.gamma < 1.0 else worst_reward
    
    return alpha


def backup(pomdp: POMDP, Gamma: List[Tuple[np.ndarray, Any]], b: np.ndarray) -> Tuple[np.ndarray, Any]:
    """
    Algorithm 21.7: Backup operation for belief point.
    Computes the best alpha vector for belief b given current value function Gamma.
    
    Args:
        pomdp: POMDP instance
        Gamma: Current set of alpha vectors [(alpha, action), ...]
        b: Belief point
    
    Returns:
        New alpha vector and associated action as tuple (alpha, action)
    """
    S, A, O = pomdp.S, pomdp.A, pomdp.O
    gamma = pomdp.gamma
    R, T, O_func = pomdp.R, pomdp.T, pomdp.O_func
    
    best_alpha = None
    best_action = None
    best_value = -np.inf
    
    # Try each action
    for a in A:
        # For each observation, find best alpha vector for updated belief
        Gamma_ao = []
        
        for o in O:
            # Update belief with this action and observation
            b_prime = update(b, pomdp, a, o)
            
            # Find alpha vector in Gamma with highest dot product with b'
            if len(Gamma) == 0:
                # No alpha vectors yet, use zero
                best_alpha_for_o = np.zeros(len(S))
            else:
                best_value_for_o = -np.inf
                best_alpha_for_o = None
                
                for alpha_vec in Gamma:
                    if isinstance(alpha_vec, tuple):
                        alpha, _ = alpha_vec
                    else:
                        alpha = alpha_vec
                    
                    value = np.dot(alpha, b_prime)
                    if value > best_value_for_o:
                        best_value_for_o = value
                        best_alpha_for_o = alpha
                
                if best_alpha_for_o is None:
                    best_alpha_for_o = np.zeros(len(S))
            
            Gamma_ao.append(best_alpha_for_o)
        
        # Construct alpha vector for this action
        alpha_a = np.zeros(len(S))
        
        for i, s in enumerate(S):
            # Immediate reward
            alpha_a[i] = R(s, a)
            
            # Add discounted future value
            future_value = 0.0
            for j, s_prime in enumerate(S):
                transition_prob = T(s, a, s_prime)
                
                for k, o in enumerate(O):
                    obs_prob = O_func(a, s_prime, o)
                    future_value += transition_prob * obs_prob * Gamma_ao[k][j]
            
            alpha_a[i] += gamma * future_value
        
        # Evaluate this alpha vector at belief b
        value_at_b = np.dot(alpha_a, b)
        
        if value_at_b > best_value:
            best_value = value_at_b
            best_alpha = alpha_a
            best_action = a
    
    return (best_alpha, best_action)


def alphavector_iteration(pomdp: POMDP, method: Any, Gamma: List[Tuple[np.ndarray, Any]]) -> List[Tuple[np.ndarray, Any]]:
    """
    Iterate the update operation multiple times.
    
    Args:
        pomdp: POMDP instance
        method: Method object with k_max attribute
        Gamma: Initial set of alpha vectors
    
    Returns:
        Updated set of alpha vectors
    """
    for k in range(method.k_max):
        Gamma = method.update(pomdp, Gamma)
    
    return Gamma


class PointBasedValueIteration:
    """
    Algorithm 21.6: Point-Based Value Iteration
    Performs backup operations on a fixed set of belief points.
    """
    
    def __init__(self, B: List[np.ndarray], k_max: int):
        """
        Args:
            B: Set of belief points
            k_max: Maximum number of iterations
        """
        self.B = B
        self.k_max = k_max
    
    def update(self, pomdp: POMDP, Gamma: List[Tuple[np.ndarray, Any]]) -> List[Tuple[np.ndarray, Any]]:
        """
        Update operation: backup each belief point.
        
        Args:
            pomdp: POMDP instance
            Gamma: Current set of alpha vectors
        
        Returns:
            New set of alpha vectors
        """
        return [backup(pomdp, Gamma, b) for b in self.B]
    
    def solve(self, pomdp: POMDP) -> LookaheadAlphaVectorPolicy:
        """
        Solve the POMDP using point-based value iteration.
        
        Args:
            pomdp: POMDP instance
        
        Returns:
            Policy based on learned alpha vectors
        """
        # Initialize with lower bound for each action
        Gamma = [(baws_lowerbound(pomdp), a) for a in pomdp.A]
        
        # Iterate
        Gamma = alphavector_iteration(pomdp, self, Gamma)
        
        return LookaheadAlphaVectorPolicy(pomdp, Gamma)


class RandomizedPointBasedValueIteration:
    """
    Algorithm 21.8: Randomized Point-Based Value Iteration
    More efficient variant that only adds alpha vectors when they improve coverage.
    """
    
    def __init__(self, B: List[np.ndarray], k_max: int):
        """
        Args:
            B: Set of belief points
            k_max: Maximum number of iterations
        """
        self.B = B
        self.k_max = k_max
    
    def update(self, pomdp: POMDP, Gamma: List[Tuple[np.ndarray, Any]]) -> List[Tuple[np.ndarray, Any]]:
        """
        Randomized update operation.
        
        Args:
            pomdp: POMDP instance
            Gamma: Current set of alpha vectors
        
        Returns:
            New set of alpha vectors
        """
        Gamma_prime = []
        B_prime = self.B.copy()
        
        while len(B_prime) > 0:
            # Randomly select a belief point
            idx = np.random.randint(len(B_prime))
            b = B_prime[idx]
            
            # Find current best alpha vector for b
            if len(Gamma) > 0:
                best_current_value = -np.inf
                for alpha_vec in Gamma:
                    if isinstance(alpha_vec, tuple):
                        alpha, _ = alpha_vec
                    else:
                        alpha = alpha_vec
                    value = np.dot(alpha, b)
                    if value > best_current_value:
                        best_current_value = value
                        best_current_alpha = alpha_vec
            else:
                best_current_value = -np.inf
                best_current_alpha = (np.zeros(pomdp.n_states), pomdp.A[0])
            
            # Perform backup
            alpha_prime_tuple = backup(pomdp, Gamma, b)
            alpha_prime, _ = alpha_prime_tuple
            
            # Check if new alpha improves value at b
            new_value = np.dot(alpha_prime, b)
            
            if new_value >= best_current_value:
                Gamma_prime.append(alpha_prime_tuple)
            else:
                Gamma_prime.append(best_current_alpha)
            
            # Remove beliefs that are well-covered by Gamma_prime
            B_prime_new = []
            for b_check in B_prime:
                # Value under Gamma_prime
                max_value_new = -np.inf
                for alpha_vec in Gamma_prime:
                    if isinstance(alpha_vec, tuple):
                        alpha, _ = alpha_vec
                    else:
                        alpha = alpha_vec
                    value = np.dot(alpha, b_check)
                    if value > max_value_new:
                        max_value_new = value
                
                # Value under Gamma
                max_value_old = -np.inf
                for alpha_vec in Gamma:
                    if isinstance(alpha_vec, tuple):
                        alpha, _ = alpha_vec
                    else:
                        alpha = alpha_vec
                    value = np.dot(alpha, b_check)
                    if value > max_value_old:
                        max_value_old = value
                
                # Keep if not sufficiently improved
                if max_value_new < max_value_old:
                    B_prime_new.append(b_check)
            
            B_prime = B_prime_new
        
        return Gamma_prime
    
    def solve(self, pomdp: POMDP) -> LookaheadAlphaVectorPolicy:
        """
        Solve the POMDP using randomized point-based value iteration.
        
        Args:
            pomdp: POMDP instance
        
        Returns:
            Policy based on learned alpha vectors
        """
        # Initialize with single lower bound
        Gamma = [(baws_lowerbound(pomdp), pomdp.A[0])]
        
        # Iterate
        Gamma = alphavector_iteration(pomdp, self, Gamma)
        
        return LookaheadAlphaVectorPolicy(pomdp, Gamma)


def expand_beliefs(pomdp: POMDP, B: List[np.ndarray], num_new_points: int, 
                   num_steps: int = 10) -> List[np.ndarray]:
    """
    Expand belief set by simulation.
    
    Args:
        pomdp: POMDP instance
        B: Current belief set
        num_new_points: Number of new points to add
        num_steps: Steps to simulate for each point
    
    Returns:
        Expanded belief set
    """
    B_expanded = B.copy()
    
    for _ in range(num_new_points):
        # Start from random existing belief
        if len(B) > 0:
            b = B[np.random.randint(len(B))].copy()
        else:
            b = np.ones(pomdp.n_states) / pomdp.n_states
        
        # Simulate forward
        for _ in range(num_steps):
            a = pomdp.A[np.random.randint(len(pomdp.A))]
            o = pomdp.O[np.random.randint(len(pomdp.O))]
            b = update(b, pomdp, a, o)
        
        B_expanded.append(b)
    
    return B_expanded


def generate_initial_beliefs(pomdp: POMDP, num_points: int, method: str = 'random') -> List[np.ndarray]:
    """
    Generate initial belief points for PBVI.
    
    Args:
        pomdp: POMDP instance
        num_points: Number of belief points to generate
        method: Method for generation ('random', 'corners', 'uniform')
    
    Returns:
        List of belief points
    """
    B = []
    
    if method == 'corners':
        # Add corner beliefs (deterministic states)
        for i in range(min(num_points, pomdp.n_states)):
            b = np.zeros(pomdp.n_states)
            b[i] = 1.0
            B.append(b)
        num_points -= len(B)
    
    if method == 'uniform' or (method == 'corners' and num_points > 0):
        # Add uniform belief
        B.append(np.ones(pomdp.n_states) / pomdp.n_states)
        num_points -= 1
    
    # Add random beliefs
    for _ in range(num_points):
        b = np.random.dirichlet(np.ones(pomdp.n_states))
        B.append(b)
    
    return B

