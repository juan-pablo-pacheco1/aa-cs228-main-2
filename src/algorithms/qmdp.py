"""
QMDP Alternative Implementation for OCD Treatment Analysis
This module implements a QMDP (Quick Model-based Decision Process) approach
as an alternative to POMDP for OCD treatment planning.
"""

import numpy as np
from typing import Dict, Tuple, List

class QMDP:
    def __init__(self, states: List[str], actions: List[str], transition_model: Dict[str, Dict[str, float]], reward_model: Dict[str, float], discount_factor: float = 0.9):
        """
        Initialize the QMDP model.

        :param states: List of states in the model.
        :param actions: List of actions available.
        :param transition_model: Transition probabilities for each action.
        :param reward_model: Reward values for each state.
        :param discount_factor: Discount factor for future rewards.
        """
        self.states = states
        self.actions = actions
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.discount_factor = discount_factor
        self.q_values = {state: {action: 0.0 for action in actions} for state in states}

    def compute_q_values(self, max_iterations: int = 100, tolerance: float = 1e-3):
        """
        Compute Q-values using the QMDP algorithm.

        :param max_iterations: Maximum number of iterations for convergence.
        :param tolerance: Convergence tolerance.
        """
        for iteration in range(max_iterations):
            delta = 0
            for state in self.states:
                for action in self.actions:
                    q_value = self.reward_model[state] + self.discount_factor * sum(
                        self.transition_model[state][next_state] * max(self.q_values[next_state].values())
                        for next_state in self.states
                    )
                    delta = max(delta, abs(self.q_values[state][action] - q_value))
                    self.q_values[state][action] = q_value
            if delta < tolerance:
                break

    def select_action(self, belief: Dict[str, float]) -> str:
        """
        Select the best action based on the current belief state.

        :param belief: Belief distribution over states.
        :return: Selected action.
        """
        action_values = {action: 0.0 for action in self.actions}
        for state, prob in belief.items():
            for action in self.actions:
                action_values[action] += prob * self.q_values[state][action]
        return max(action_values, key=action_values.get)
