"""
EM Algorithm for Learning Bayesian Networks with Missing Data
Implements Section 4.4.2 from "Algorithms for Decision Making"
"""

import numpy as np
import networkx as nx
from typing import List, Tuple
from .bayesian_network import Variable, statistics, sub2ind
import itertools


def get_missing_mask(D: np.ndarray) -> np.ndarray:
    """
    Create boolean mask indicating missing values (NaN).
    
    Args:
        D: Data matrix (n_vars x n_samples)
    
    Returns:
        Boolean mask (True = missing)
    """
    return np.isnan(D)


def get_parent_instantiation_index(values: List[int], r_parents: List[int]) -> int:
    """
    Get the index for a specific parent instantiation.
    
    Args:
        values: Parent values (1-indexed)
        r_parents: Parent cardinalities
    
    Returns:
        Parent instantiation index (0-indexed for array access)
    """
    if len(values) == 0:
        return 0
    return sub2ind(r_parents, values) - 1


def expected_statistics(vars: List[Variable], G: nx.DiGraph, D: np.ndarray, 
                       cpds: List[np.ndarray]) -> List[np.ndarray]:
    """
    E-step: Compute expected sufficient statistics given current parameters.
    For complete cases, count as before. For incomplete cases, distribute
    counts according to current belief.
    
    Args:
        vars: List of Variable objects
        G: Directed graph
        D: Data matrix with NaN for missing values
        cpds: Current conditional probability distributions
    
    Returns:
        Expected count statistics M
    """
    n = len(vars)
    r = [vars[i].r for i in range(n)]
    
    # Compute q (number of parent instantiations)
    q = []
    for i in range(n):
        parents = list(G.predecessors(i))
        if len(parents) == 0:
            q.append(1)
        else:
            q.append(int(np.prod([r[j] for j in parents])))
    
    # Initialize expected statistics
    M = [np.zeros((q[i], r[i])) for i in range(n)]
    missing_mask = get_missing_mask(D)
    
    # Process each sample
    for col_idx in range(D.shape[1]):
        o = D[:, col_idx]
        sample_missing = missing_mask[:, col_idx]
        
        if not np.any(sample_missing):
            # Complete case: count as usual
            for i in range(n):
                k = int(o[i]) - 1
                parents = list(G.predecessors(i))
                j = 0
                
                if len(parents) > 0:
                    parent_vals = [int(o[p]) for p in parents]
                    parent_r = [r[p] for p in parents]
                    j = get_parent_instantiation_index(parent_vals, parent_r)
                
                M[i][j, k] += 1.0
        else:
            # Incomplete case: use EM to distribute counts
            # Compute probability for each possible completion
            missing_vars = np.where(sample_missing)[0]
            observed_vars = np.where(~sample_missing)[0]
            
            # Generate all possible completions of missing variables
            missing_ranges = [range(1, r[v] + 1) for v in missing_vars]
            
            # Compute unnormalized probabilities for each completion
            completion_probs = []
            completions = []
            
            for completion in itertools.product(*missing_ranges):
                # Create full sample with this completion
                o_complete = o.copy()
                for idx, var_idx in enumerate(missing_vars):
                    o_complete[var_idx] = completion[idx]
                
                # Compute probability of this completion
                prob = 1.0
                for i in range(n):
                    parents = list(G.predecessors(i))
                    k = int(o_complete[i]) - 1
                    j = 0
                    
                    if len(parents) > 0:
                        parent_vals = [int(o_complete[p]) for p in parents]
                        parent_r = [r[p] for p in parents]
                        j = get_parent_instantiation_index(parent_vals, parent_r)
                    
                    prob *= cpds[i][j, k]
                
                completion_probs.append(prob)
                completions.append(o_complete)
            
            # Normalize probabilities
            total_prob = sum(completion_probs)
            if total_prob > 0:
                completion_probs = [p / total_prob for p in completion_probs]
            else:
                # Uniform if all zero
                completion_probs = [1.0 / len(completion_probs)] * len(completion_probs)
            
            # Add expected counts
            for prob, o_complete in zip(completion_probs, completions):
                for i in range(n):
                    k = int(o_complete[i]) - 1
                    parents = list(G.predecessors(i))
                    j = 0
                    
                    if len(parents) > 0:
                        parent_vals = [int(o_complete[p]) for p in parents]
                        parent_r = [r[p] for p in parents]
                        j = get_parent_instantiation_index(parent_vals, parent_r)
                    
                    M[i][j, k] += prob
    
    return M


def maximize_parameters(M: List[np.ndarray], alpha: List[np.ndarray]) -> List[np.ndarray]:
    """
    M-step: Maximize parameters given expected sufficient statistics.
    
    Args:
        M: Expected count statistics
        alpha: Prior hyperparameters
    
    Returns:
        Updated CPDs
    """
    cpds = []
    for i in range(len(M)):
        counts = M[i] + alpha[i]
        row_sums = counts.sum(axis=1, keepdims=True)
        cpd = counts / row_sums
        cpds.append(cpd)
    
    return cpds


def log_likelihood(vars: List[Variable], G: nx.DiGraph, D: np.ndarray, 
                  cpds: List[np.ndarray]) -> float:
    """
    Compute log-likelihood of data given parameters.
    Only uses complete cases for likelihood computation.
    
    Args:
        vars: List of Variable objects
        G: Directed graph
        D: Data matrix
        cpds: Current CPDs
    
    Returns:
        Log-likelihood
    """
    n = len(vars)
    r = [vars[i].r for i in range(n)]
    missing_mask = get_missing_mask(D)
    
    ll = 0.0
    
    for col_idx in range(D.shape[1]):
        o = D[:, col_idx]
        sample_missing = missing_mask[:, col_idx]
        
        if not np.any(sample_missing):
            # Complete case
            sample_ll = 0.0
            for i in range(n):
                parents = list(G.predecessors(i))
                k = int(o[i]) - 1
                j = 0
                
                if len(parents) > 0:
                    parent_vals = [int(o[p]) for p in parents]
                    parent_r = [r[p] for p in parents]
                    j = get_parent_instantiation_index(parent_vals, parent_r)
                
                prob = cpds[i][j, k]
                if prob > 0:
                    sample_ll += np.log(prob)
                else:
                    sample_ll += -1e10  # Very negative for zero probability
            
            ll += sample_ll
    
    return ll


def em_learning(vars: List[Variable], G: nx.DiGraph, D: np.ndarray, 
                max_iter: int = 100, tol: float = 1e-4, 
                alpha_default: float = 1.0) -> Tuple[List[np.ndarray], List[float]]:
    """
    Full EM algorithm for parameter learning with missing data.
    
    Args:
        vars: List of Variable objects
        G: Directed graph
        D: Data matrix with NaN for missing values
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        alpha_default: Prior pseudocount
    
    Returns:
        Tuple of (learned CPDs, log-likelihood history)
    """
    n = len(vars)
    r = [vars[i].r for i in range(n)]
    
    # Initialize parameters uniformly
    cpds = []
    for i in range(n):
        parents = list(G.predecessors(i))
        if len(parents) == 0:
            q_i = 1
        else:
            q_i = int(np.prod([r[j] for j in parents]))
        
        cpds.append(np.ones((q_i, r[i])) / r[i])
    
    # Prior
    alpha = []
    for i in range(n):
        parents = list(G.predecessors(i))
        if len(parents) == 0:
            q_i = 1
        else:
            q_i = int(np.prod([r[j] for j in parents]))
        alpha.append(np.ones((q_i, r[i])) * alpha_default)
    
    ll_history = []
    prev_ll = -np.inf
    
    for iteration in range(max_iter):
        # E-step
        M = expected_statistics(vars, G, D, cpds)
        
        # M-step
        cpds = maximize_parameters(M, alpha)
        
        # Compute log-likelihood
        ll = log_likelihood(vars, G, D, cpds)
        ll_history.append(ll)
        
        # Check convergence
        if abs(ll - prev_ll) < tol:
            print(f"EM converged after {iteration + 1} iterations")
            break
        
        prev_ll = ll
    
    return cpds, ll_history

