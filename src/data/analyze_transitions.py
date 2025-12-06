import pandas as pd
import numpy as np
import os

# Use relative path for dataset
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data/raw/ocd_patient_dataset.csv")  # Updated to absolute path

# Load the dataset
data = pd.read_csv(data_path)

# Define response categories based on Y-BOCS score reduction
def classify_response(ybocs_before, ybocs_after):
    reduction = ybocs_before - ybocs_after
    if reduction >= 10:
        return "Remission"
    elif reduction >= 5:
        return "Partial Response"
    else:
        return "No Response"

# Analyze transitions
def analyze_transitions(data):
    transitions = []

    for _, row in data.iterrows():
        ybocs_before = row["Y-BOCS Score (Obsessions)"] + row["Y-BOCS Score (Compulsions)"]
        ybocs_after = np.random.normal(ybocs_before - 10, 5)  # Simulate post-treatment score
        response = classify_response(ybocs_before, ybocs_after)

        transitions.append({
            "Patient ID": row["Patient ID"],
            "Treatment": row["Medications"],
            "Response": response
        })

    return pd.DataFrame(transitions)

# Perform analysis
transitions_df = analyze_transitions(data)

# Calculate probabilities
probabilities = transitions_df.groupby("Treatment")["Response"].value_counts(normalize=True).unstack()

# Use relative path for saving probabilities
probabilities.to_csv(os.path.join(base_dir, "data/processed/transition_probabilities.csv"))