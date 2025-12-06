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
import json
from algorithms.pomdp import POMDP
from models.solve_pomdp import solve_with_pbvi, evaluate_policy
from src.OCD_SIMULATOR_FINAL import CLINICIAN, APA_ALGORITHM, assess_response
from src.unified_calculate_transition_probabilities import calculate_transition_probabilities, load_data
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
observation_prob_path = os.path.join(base_dir, "data/processed/observation_probabilities.json")

# Debugging: Print resolved path for observation probabilities
print(f"Resolved path for observation probabilities: {observation_prob_path}")

with open(observation_prob_path, "r") as f:
    observation_probabilities = json.load(f)

# observation_prob_path points to a JSON file containing observation probabilities.
# This file is created by analyzing the `ocd_with_treatments.csv` file in the `data/processed` directory.
# The analysis involves calculating the likelihood of each observation category (e.g., Remission, Partial Response, No Response)
# based on the treatment outcomes recorded in the dataset.

# Load dynamic transition probabilities
data = load_data()
actions = ["CBT", "SSRI", "TMS", "Ketamine", "CBT+SSRI", "CBT+TMS", "SSRI+TMS", "SNRI"]
transition_probabilities = calculate_transition_probabilities(data, actions)

# Debugging: Summarize transition probabilities for each action
for action, probs in transition_probabilities.items():
    print(f"Action: {action}, Transition Probabilities: {probs}")

# Transition probabilities are dynamically calculated based on the dataset
# using the `unified_calculate_transition_probabilities.py` script.

# Define possible observations combining medication effects and broader outcomes
observations = [
    {"response": "Remission", "side_effects": "None", "adherence": "High"},
    {"response": "Partial Response", "side_effects": "Mild", "adherence": "Medium"},
    {"response": "No Response", "side_effects": "Severe", "adherence": "Low"},
    # Add more combinations as needed
]

# Transition function
undefined_transitions = set()  # Track undefined transitions to avoid repeated logging
total_transitions = 0  # Track total transitions checked
def T(s, a, s_prime):
    """
    Transition function: Define the probability of transitioning from state `s` to `s_prime`
    given action `a`.
    """
    global total_transitions
    total_transitions += 1  # Increment total transitions checked

    # Extract state components
    age_group, subtype = s
    next_age_group, next_subtype = s_prime

    if a in transition_probabilities:
        if next_subtype in transition_probabilities[a]:
            prob = transition_probabilities[a][next_subtype]
            # print(f"Transition: s={s}, a={a}, s'={s_prime}, prob={prob}")  # Log transition probability
            return prob

    # Debugging: Check for undefined transitions
    undefined_key = (a, next_subtype)
    if undefined_key not in undefined_transitions:
        # print(f"Undefined transition: action={a}, next_subtype={next_subtype}, available={list(transition_probabilities[a].keys())}")
        undefined_transitions.add(undefined_key)  # Log only once

    return 1 / len(states)  # Uniform probability fallback

# Reward function
# Replace the R function with reward_model to include observations in the reward calculation
def R(s, a):
    """Reward function: Assign rewards based on the state, action, and observation."""
    # Use the observation function to determine the observation
    observation = max(
        observations, 
        key=lambda o: O(a, s, o)  # Select the observation with the highest probability
    )
    return reward_model(s, a, observation)

# Update the reward model to assign rewards based on observations
def reward_model(state, action, observation):
    """Assign rewards based on the state, action, and observation."""
    if observation["response"] == "Remission":
        return 10  # High reward for remission
    elif observation["response"] == "Partial Response":
        return 5  # Moderate reward for partial response
    elif observation["response"] == "No Response":
        return -10  # Penalty for no response
    return 0  # Default reward

# Observation function
def O(a, s_prime, o):
    """
    Observation function: Define the probability of observing `o` given action `a` and resulting state `s_prime`.
    """
    # Define observation probabilities based on response categories
    response_category = s_prime[1]  # Assuming `s_prime` contains the response category

    # Ensure valid probabilities by assigning uniform probabilities as a fallback
    if response_category not in observation_probabilities or o not in observation_probabilities[response_category]:
        return 1 / len(observations)  # Uniform probability fallback

    return observation_probabilities[response_category].get(o, 0)

def create_pomdp_from_simulation():
    """Create a POMDP model using simulation parameters."""
    # Define states based on age groups and subtypes
    global states, observations  # Make these accessible to T, R, O
    states = [(age_group, subtype) for age_group, _ in CLINICIAN['age_groups'] for subtype, _ in CLINICIAN['subtype_probabilities']]

    # Define actions
    actions = ["CBT", "SSRI", "TMS", "Ketamine", "CBT+SSRI", "CBT+TMS", "SSRI+TMS", "SNRI"]  # Updated actions

    # Define observations (e.g., response categories)
    observations = ["Remission", "Partial Response", "No Response"]  # Aligned with observation_probabilities.json

    # Debugging: Print all actions in the POMDP model
    # print(f"Actions in POMDP model: {actions}")
    # print(f"Actions in transition_probabilities: {list(transition_probabilities.keys())}")

    # Adjusted discount factor
    gamma = 0.99

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

def evaluate_policy_debug(pomdp, policy, num_episodes, max_steps):
    """Evaluate the policy with detailed logging."""
    total_reward = 0
    for episode in range(1, num_episodes + 1):
        state = np.random.choice(pomdp.S)  # Start with a random state
        episode_reward = 0
        # print(f"\nEpisode {episode}/{num_episodes}: Starting state: {state}")
        for step in range(1, max_steps + 1):
            action = policy[state]
            next_state = np.random.choice(pomdp.S, p=[pomdp.T(state, action, s_prime) for s_prime in pomdp.S])
            reward = pomdp.R(state, action)
            episode_reward += reward
            # print(f"Step {step}: state={state}, action={action}, next_state={next_state}, reward={reward}")
            state = next_state
        total_reward += episode_reward
        # print(f"Episode {episode} reward: {episode_reward}")
    avg_reward = total_reward / num_episodes
    # print(f"\nAverage reward: {avg_reward}")
    return avg_reward

def validate_action(state, action):
    """Validate if the action is valid for the given state."""
    # Debugging invalid actions
    logging.debug(f"Validating action: {action} for state: {state}")
    # Example placeholder for validation logic
    if action not in valid_actions_for_state(state):
        logging.warning(f"Invalid action: {action} for state: {state}")
        return False
    return True

# Update the valid_actions_for_state function to validate actions based on the dataset
def valid_actions_for_state(state):
    """Return a list of valid actions for the given state."""
    # Define all possible actions
    all_actions = ["CBT", "SSRI", "CBT+SSRI", "SSRI+TMS", "TMS", "Ketamine", "Benzodiazepine", "SNRI"]

    # Take all actions to be valid regardless of state
    return all_actions

# Add the missing function definition for print_undefined_transition_percentage
def print_undefined_transition_percentage():
    """Calculate and print the percentage of undefined transitions."""
    undefined_count = len(undefined_transitions)
    percentage = (undefined_count / total_transitions) * 100 if total_transitions > 0 else 0
    print(f"Total transitions checked: {total_transitions}")
    print(f"Undefined transitions: {undefined_count} ({percentage:.2f}%)")

# Replace the evaluate_policy call in main with evaluate_policy_debug
def main():
    """Main training pipeline."""
    print("Creating POMDP from simulation parameters...")
    pomdp = create_pomdp_from_simulation()

    # Solve and evaluate with reduced parameters
    policy, solve_time = solve_with_pbvi(pomdp, num_beliefs=100, k_max=20)  # Reduced belief points and iterations
    avg_reward = evaluate_policy_debug(pomdp, policy, num_episodes=5, max_steps=3)  # Reduced episodes and steps

    # Save the policy
    save_policy(policy, pomdp, avg_reward, solve_time)

    # Print the percentage of undefined transitions
    print_undefined_transition_percentage()

if __name__ == "__main__":
    main()