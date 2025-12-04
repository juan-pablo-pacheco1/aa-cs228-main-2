"""
Solve OCD Treatment POMDP using Simulation Parameters
This script integrates the simulation logic from OCD_SIMULATOR_FINAL.py
with the POMDP solving pipeline.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from algorithms.pomdp import POMDP
from models.solve_pomdp import solve_with_pbvi, evaluate_policy
from src.OCD_SIMULATOR_FINAL import CLINICIAN

# Transition function
def T(s, a, s_prime):
    # Define transitions based on simulation logic
    return 1 / len(states)  # Example: uniform transitions

# Reward function
def R(s, a):
    # Define rewards based on YBOCS reduction
    treatment_effects = CLINICIAN['ocd_treatment_ybocs']
    if a in treatment_effects:
        return treatment_effects[a]['mean_change']
    return -10  # Penalty for invalid actions

# Observation function
def O(a, s_prime, o):
    # Define observations (e.g., response categories)
    return 1 / len(observations)  # Example: uniform observations

def create_pomdp_from_simulation():
    """Create a POMDP model using simulation parameters."""
    # Define states based on age groups and subtypes
    global states, observations  # Make these accessible to T, R, O
    states = []
    for age_group, _ in CLINICIAN['age_groups']:
        for subtype, _ in CLINICIAN['subtype_probabilities']:
            states.append((age_group, subtype))

    # Define actions
    actions = ["SSRI", "CBT", "SSRI_plus_CBT"]

    # Define observations (e.g., response categories)
    observations = ["Remission", "Partial Response", "No Response"]

    # Discount factor
    gamma = 0.95

    return POMDP(gamma, states, actions, observations, T, R, O)

def save_policy(policy, pomdp, avg_reward, solve_time):
    """Save the policy to the models folder."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    policy_data = {
        'policy': policy,
        'states': pomdp.S,
        'actions': pomdp.A,
        'avg_reward': avg_reward,
        'solve_time': solve_time,
        'method': 'pbvi_simulation'
    }

    policy_path = os.path.join(models_dir, 'pomdp_policy_simulator.pkl')
    with open(policy_path, 'wb') as f:
        import pickle
        pickle.dump(policy_data, f)

    print(f"\nPolicy saved to: {policy_path}")

def main():
    """Main training pipeline."""
    print("Creating POMDP from simulation parameters...")
    pomdp = create_pomdp_from_simulation()

    # Solve and evaluate
    policy, solve_time = solve_with_pbvi(pomdp, num_beliefs=10, k_max=2)
    avg_reward, rewards = evaluate_policy(pomdp, policy, num_episodes=5, max_steps=2)

    # Save the policy
    save_policy(policy, pomdp, avg_reward, solve_time)

if __name__ == "__main__":
    main()