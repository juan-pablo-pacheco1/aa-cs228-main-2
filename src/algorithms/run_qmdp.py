import os
import pandas as pd
import numpy as np
import pickle
from qmdp import QMDP

def load_train_data():
    """Load and preprocess the training data."""
    data_path = os.path.join(os.path.dirname(__file__), '../../data/processed/train_discrete.csv')
    data = pd.read_csv(data_path)
    return data

def define_qmdp_components(data):
    """Define states, actions, transition model, and reward model."""
    # Define states as unique combinations of features
    states = data[['Age_Group', 'YBOCS_Severity_Baseline']].drop_duplicates().apply(tuple, axis=1).tolist()

    # Define actions from the Treatment column
    actions = data['Treatment'].unique().tolist()

    # Transition model: Assume uniform transitions for simplicity
    transition_model = {state: {next_state: 1 / len(states) for next_state in states} for state in states}

    # Reward model: Map Response to rewards, using only state-defining columns
    response_to_reward = {1: 10, 2: 5, 3: -5, 4: -10}  # Example mapping
    reward_model = {
        tuple(row[['Age_Group', 'YBOCS_Severity_Baseline']]): response_to_reward[row['Response']]
        for _, row in data.dropna(subset=['Response']).iterrows()
    }

    return states, actions, transition_model, reward_model

def save_policy(qmdp):
    """Save the QMDP policy to the models folder as a .pkl file."""
    policy_path = os.path.join(os.path.dirname(__file__), '../../models/qmdp_policy.pkl')
    policy = {
        'q_values': qmdp.q_values,
        'states': qmdp.states,
        'actions': qmdp.actions
    }
    with open(policy_path, 'wb') as f:
        pickle.dump(policy, f)

def main():
    # Load and preprocess data
    data = load_train_data()

    # Define QMDP components
    states, actions, transition_model, reward_model = define_qmdp_components(data)

    # Initialize and compute QMDP policy
    qmdp = QMDP(states, actions, transition_model, reward_model)
    qmdp.compute_q_values()

    # Save the policy
    save_policy(qmdp)

if __name__ == "__main__":
    main()