# Add the project root to the Python path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

# Updated data path to use the processed dataset
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'ocd_with_treatments.csv')

def load_data():
    """Load OCD patient dataset."""
    import pandas as pd
    return pd.read_csv(data_path)

# Load the dataset
data = load_data()

# Updated define_states to use thresholds from OCD_SIMULATOR_FINAL
from src.OCD_SIMULATOR_FINAL import CLINICIAN

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

# Update actions to match dataset entries
actions = ["SSRI", "SNRI", "Benzodiazepine"]

# Initialize a dictionary to store transition probabilities
transition_probabilities = {}

# Iterate over each action to calculate probabilities
for action in actions:
    # Filter data for the specific action
    action_data = data[data['Medications'] == action]

    # Use .loc to avoid SettingWithCopyWarning
    action_data.loc[:, 'Total Y-BOCS Change'] = (
        action_data['Y-BOCS Score (Obsessions)'] + action_data['Y-BOCS Score (Compulsions)']
    )

    # Classify transitions into states
    state_counts = {state: 0 for state in define_states()}
    for _, row in action_data.iterrows():
        total_change = row['Total Y-BOCS Change']
        for state, condition in define_states().items():
            if condition(total_change):
                state_counts[state] += 1
                break

    # Calculate probabilities for each state
    total_transitions = sum(state_counts.values())
    if total_transitions == 0:
        print(f"No transitions found for the given action or state. Assigning uniform probabilities.")
        total_transitions = 1  # Avoid division by zero by assigning a default value
    transition_probabilities[action] = {
        state: count / total_transitions for state, count in state_counts.items()
    }

# Output the results
for action, probabilities in transition_probabilities.items():
    print(f"Action: {action}")
    for state, probability in probabilities.items():
        print(f"  {state}: {probability:.2%}")
