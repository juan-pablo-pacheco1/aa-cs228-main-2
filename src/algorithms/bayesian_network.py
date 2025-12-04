"""
Bayesian Network Structure Learning Algorithms
Implements Algorithms 4.1, 5.2-5.3, 5.4 from "Algorithms for Decision Making"
"""

import numpy as np
import networkx as nx
from scipy.special import loggamma
from typing import List, Dict, Tuple
import copy


class Variable:
    """Represents a discrete random variable"""
    def __init__(self, name: str, r: int):
        """
        Args:
            name: Variable name
            r: Cardinality (number of possible values)
        """
        self.name = name
        self.r = r
    
    def __repr__(self):
        return f"Variable(name={self.name}, r={self.r})"


def sub2ind(siz: List[int], x: List[int]) -> int:
    """
    Convert multi-dimensional subscript to linear index.
    Implements the sub2ind helper function.
    
    Args:
        siz: Dimensions array
        x: Multi-dimensional indices (1-indexed, will be converted)
    
    Returns:
        Linear index (1-indexed to match Julia convention)
    """
    if len(siz) == 0:
        return 1
    
    k = [1] + list(np.cumprod(siz[:-1]))
    return int(np.dot(k, np.array(x) - 1)) + 1


def statistics(vars: List[Variable], G: nx.DiGraph, D: np.ndarray) -> List[np.ndarray]:
    """
    Algorithm 4.1: Compute sufficient statistics from data.
    Computes the count statistics M[i][j,k] where:
    - i is the node index
    - j is the parent instantiation index
    - k is the node value
    
    Args:
        vars: List of Variable objects
        G: Directed graph (NetworkX DiGraph)
        D: Data matrix (n_vars x n_samples) with integer values in [1, r_i]
    
    Returns:
        List of M matrices, one per variable
    """
    n = len(vars)
    r = [vars[i].r for i in range(n)]
    
    # q[i] = number of parent instantiations for variable i
    q = []
    for i in range(n):
        parents = list(G.predecessors(i))
        if len(parents) == 0:
            q.append(1)
        else:
            q.append(int(np.prod([r[j] for j in parents])))
    
    # Initialize M matrices
    M = [np.zeros((q[i], r[i])) for i in range(n)]
    
    # Count occurrences
    for col_idx in range(D.shape[1]):
        o = D[:, col_idx]
        for i in range(n):
            k = int(o[i]) - 1  # Convert to 0-indexed for array access
            parents = list(G.predecessors(i))
            j = 0  # Default parent instantiation index
            
            if len(parents) > 0:
                parent_vals = [int(o[p]) for p in parents]
                parent_r = [r[p] for p in parents]
                j = sub2ind(parent_r, parent_vals) - 1  # Convert to 0-indexed
            
            M[i][j, k] += 1.0
    
    return M


def prior(vars: List[Variable], G: nx.DiGraph, alpha_default: float = 1.0) -> List[np.ndarray]:
    """
    Define prior hyperparameters (Dirichlet priors) for Bayesian scoring.
    
    Args:
        vars: List of Variable objects
        G: Directed graph
        alpha_default: Default prior pseudocount
    
    Returns:
        List of alpha matrices (same structure as M from statistics)
    """
    n = len(vars)
    r = [vars[i].r for i in range(n)]
    
    alpha = []
    for i in range(n):
        parents = list(G.predecessors(i))
        if len(parents) == 0:
            q_i = 1
        else:
            q_i = int(np.prod([r[j] for j in parents]))
        
        alpha.append(np.ones((q_i, r[i])) * alpha_default)
    
    return alpha


def bayesian_score_component(M: np.ndarray, alpha: np.ndarray) -> float:
    """
    Compute Bayesian score for a single node.
    
    Args:
        M: Count statistics matrix (q x r)
        alpha: Prior hyperparameter matrix (q x r)
    
    Returns:
        Log Bayesian score component
    """
    p = np.sum(loggamma(alpha + M))
    p -= np.sum(loggamma(alpha))
    p += np.sum(loggamma(np.sum(alpha, axis=1)))
    p -= np.sum(loggamma(np.sum(alpha, axis=1) + np.sum(M, axis=1)))
    return p


def bayesian_score(vars: List[Variable], G: nx.DiGraph, D: np.ndarray, 
                   alpha_default: float = 1.0) -> float:
    """
    Compute the Bayesian score for the entire network.
    
    Args:
        vars: List of Variable objects
        G: Directed graph
        D: Data matrix
        alpha_default: Default prior pseudocount
    
    Returns:
        Log Bayesian score
    """
    n = len(vars)
    M = statistics(vars, G, D)
    alpha = prior(vars, G, alpha_default)
    return sum(bayesian_score_component(M[i], alpha[i]) for i in range(n))


class K2Search:
    """
    Algorithm 5.2-5.3: K2 Structure Learning Algorithm
    Learns Bayesian network structure given a variable ordering.
    """
    
    def __init__(self, ordering: List[int]):
        """
        Args:
            ordering: Variable ordering (list of variable indices)
        """
        self.ordering = ordering
    
    def fit(self, vars: List[Variable], D: np.ndarray, alpha_default: float = 1.0) -> nx.DiGraph:
        """
        Learn Bayesian network structure using K2 algorithm.
        
        Args:
            vars: List of Variable objects
            D: Data matrix (n_vars x n_samples)
            alpha_default: Prior pseudocount
        
        Returns:
            Learned directed acyclic graph
        """
        n = len(vars)
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        
        for k, i in enumerate(self.ordering[1:], start=1):
            y = bayesian_score(vars, G, D, alpha_default)
            
            while True:
                y_best = -np.inf
                j_best = None
                
                # Try adding each potential parent
                for j in self.ordering[:k]:
                    if not G.has_edge(j, i):
                        G.add_edge(j, i)
                        y_prime = bayesian_score(vars, G, D, alpha_default)
                        
                        if y_prime > y_best:
                            y_best = y_prime
                            j_best = j
                        
                        G.remove_edge(j, i)
                
                # If improvement found, add the edge; otherwise stop
                if y_best > y:
                    y = y_best
                    G.add_edge(j_best, i)
                else:
                    break
        
        return G


class LocalDirectedGraphSearch:
    """
    Algorithm 5.4: Local Search for Bayesian Network Structure Learning
    Uses random edge additions/deletions with acyclicity constraint.
    """
    
    def __init__(self, G_init: nx.DiGraph, k_max: int):
        """
        Args:
            G_init: Initial graph
            k_max: Maximum number of iterations
        """
        self.G = G_init
        self.k_max = k_max
    
    def rand_graph_neighbor(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Generate a random neighbor graph by adding or removing one edge.
        
        Args:
            G: Current graph
        
        Returns:
            Neighbor graph
        """
        n = G.number_of_nodes()
        i = np.random.randint(0, n)
        j = (i + np.random.randint(1, n)) % n
        
        G_prime = G.copy()
        
        if G.has_edge(i, j):
            G_prime.remove_edge(i, j)
        else:
            G_prime.add_edge(i, j)
        
        return G_prime
    
    def fit(self, vars: List[Variable], D: np.ndarray, alpha_default: float = 1.0) -> nx.DiGraph:
        """
        Learn Bayesian network structure using local search.
        
        Args:
            vars: List of Variable objects
            D: Data matrix
            alpha_default: Prior pseudocount
        
        Returns:
            Learned directed acyclic graph
        """
        G = self.G.copy()
        y = bayesian_score(vars, G, D, alpha_default)
        
        for k in range(self.k_max):
            G_prime = self.rand_graph_neighbor(G)
            
            # Reject if creates cycle
            if not nx.is_directed_acyclic_graph(G_prime):
                y_prime = -np.inf
            else:
                y_prime = bayesian_score(vars, G_prime, D, alpha_default)
            
            # Accept if improvement
            if y_prime > y:
                y = y_prime
                G = G_prime
        
        return G


def estimate_parameters(vars: List[Variable], G: nx.DiGraph, D: np.ndarray, 
                       alpha_default: float = 1.0) -> List[np.ndarray]:
    """
    Estimate conditional probability parameters using maximum likelihood.
    Returns the CPDs (conditional probability distributions) for each variable.
    
    Args:
        vars: List of Variable objects
        G: Directed graph
        D: Data matrix
        alpha_default: Prior pseudocount for smoothing
    
    Returns:
        List of CPD arrays (normalized M + alpha)
    """
    M = statistics(vars, G, D)
    alpha = prior(vars, G, alpha_default)
    
    # Compute CPDs with Bayesian smoothing
    cpds = []
    for i in range(len(vars)):
        # Add prior and normalize
        counts = M[i] + alpha[i]
        # Normalize each row (parent instantiation)
        row_sums = counts.sum(axis=1, keepdims=True)
        cpd = counts / row_sums
        cpds.append(cpd)
    
    return cpds

