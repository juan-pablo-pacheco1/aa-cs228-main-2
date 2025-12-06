import os
import pandas as pd
from collections import defaultdict
from src.OCD_SIMULATOR_FINAL import CLINICIAN

# This script combines the two calculate_transition_probabilities.py files to include SSRI, SNRI, Benzodiazepine, CBT and CBT+SSRI

def load_data():
    """Load OCD patient dataset."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'ocd_with_treatments.csv')
    return pd.read_csv(data_path)

def define_states():
    """Define states based on YBOCS thresholds."""
    baseline_mean = CLINICIAN['baseline_ybocs']['mean']
    baseline_sd = CLINICIAN['baseline_ybocs']['sd']

    no_response_threshold = baseline_mean - baseline_sd
    partial_response_threshold = baseline_mean - 2 * baseline_sd

    return {
        'No Response': lambda score: score >= no_response_threshold,
        'Partial Response': lambda score: partial_response_threshold <= score < no_response_threshold,
        'Remission': lambda score: score < partial_response_threshold
    }

def calculate_transition_probabilities(data, actions):
    """Calculate transition probabilities for specified actions."""
    transition_probabilities = {}
    states = define_states()

    for action in actions:
        action_data = data[data['Treatment'] == action]

        # Calculate total Y-BOCS change
        action_data['Total Y-BOCS Change'] = (
            action_data['Y-BOCS Score (Obsessions)'] + action_data['Y-BOCS Score (Compulsions)']
        )

        # Count transitions into states
        state_counts = {state: 0 for state in states}
        for _, row in action_data.iterrows():
            total_change = row['Total Y-BOCS Change']
            for state, condition in states.items():
                if condition(total_change):
                    state_counts[state] += 1
                    break

        # Calculate probabilities
        total_transitions = sum(state_counts.values())
        if total_transitions == 0:
            print(f"No transitions found for action {action}. Assigning uniform probabilities.")
            total_transitions = len(states)  # Avoid division by zero
            state_counts = {state: 1 for state in states}  # Uniform distribution

        transition_probabilities[action] = {
            state: count / total_transitions for state, count in state_counts.items()
        }

        # Debugging: Log transition probabilities for each action
        print(f"Action: {action}, Transition Probabilities: {transition_probabilities[action]}")

    return transition_probabilities

def main():
    data = load_data()
    # Update actions to include only the specified treatments
    actions = ["CBT", "SSRI", "TMS", "Ketamine", "CBT+SSRI", "CBT+TMS", "SSRI+TMS"]
    transition_probabilities = calculate_transition_probabilities(data, actions)

    # Output the results
    for action, probabilities in transition_probabilities.items():
        print(f"Action: {action}")
        for state, probability in probabilities.items():
            print(f"  {state}: {probability:.2%}")

if __name__ == "__main__":
    main()