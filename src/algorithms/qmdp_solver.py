"""
QMDP Solver for OCD Treatment Planning
This module implements a QMDP solver that integrates with the existing data pipeline
and saves the resulting policy for future use.
"""

import numpy as np
import pandas as pd
import json
import os
from typing import List, Dict, Callable

class QMDP:
    def __init__(self, states: List[str], actions: List[str], transition_model: Dict[str, Dict[str, float]], reward_model: Dict[str, float], discount_factor: float = 0.9):
        self.states = states
        self.actions = actions
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.discount_factor = discount_factor
        self.q_values = {state: {action: 0.0 for action in actions} for state in states}

    def compute_q_values(self, max_iterations: int = 100, tolerance: float = 1e-3):
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
        action_values = {action: 0.0 for action in self.actions}
        for state, prob in belief.items():
            for action in self.actions:
                action_values[action] += prob * self.q_values[state][action]
        return max(action_values, key=action_values.get)

def load_data():
    data_path = os.path.join(os.path.dirname(__file__), '../data/processed/train_discrete.csv')
    variable_definitions_path = os.path.join(os.path.dirname(__file__), '../data/processed/variable_definitions.json')

    data = pd.read_csv(data_path)
    with open(variable_definitions_path, 'r') as f:
        variable_definitions = json.load(f)

    return data, variable_definitions

def define_qmdp_components(data: pd.DataFrame, variable_definitions: Dict):
    states = data['state'].unique().tolist()
    actions = variable_definitions['actions']

    transition_model = {state: {next_state: 1 / len(states) for next_state in states} for state in states}
    reward_model = {state: np.random.uniform(-10, 10) for state in states}  # Placeholder rewards

    return states, actions, transition_model, reward_model

def save_policy(qmdp: QMDP):
    policy_path = os.path.join(os.path.dirname(__file__), '../models/qmdp_policy.json')
    policy = {
        'q_values': qmdp.q_values,
        'states': qmdp.states,
        'actions': qmdp.actions
    }
    with open(policy_path, 'w') as f:
        json.dump(policy, f, indent=4)

def main():
    data, variable_definitions = load_data()
    states, actions, transition_model, reward_model = define_qmdp_components(data, variable_definitions)

    qmdp = QMDP(states, actions, transition_model, reward_model)
    qmdp.compute_q_values()

    save_policy(qmdp)

if __name__ == "__main__":
    main()