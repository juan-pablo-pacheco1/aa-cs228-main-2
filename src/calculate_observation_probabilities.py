import pandas as pd
import os

# Use relative paths for input and output files
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(base_dir, "data/processed/ocd_with_treatments.csv")
output_path = os.path.join(base_dir, "data/processed/observation_probabilities.json")

# Load the dataset
data = pd.read_csv(file_path)

# Filter rows with valid response categories and observations
data = data.dropna(subset=["Response_Category"])

# Calculate observation probabilities based on response categories
observation_counts = data["Response_Category"].value_counts(normalize=True)

# Define the observation categories
observations = ["Remission", "Partial Response", "No Response", "Dropout"]

# Create a dictionary for observation probabilities
observation_probabilities = {}
for response in observations:
    observation_probabilities[response] = {}
    for obs in observations:
        if response == obs:
            observation_probabilities[response][obs] = observation_counts.get(response, 0.0)
        else:
            # Distribute remaining probability among other categories
            remaining_prob = 1 - observation_counts.get(response, 0.0)
            observation_probabilities[response][obs] = remaining_prob / (len(observations) - 1)

# Print the observation probabilities
print("Observation Probabilities:")
print(observation_probabilities)

# Save the observation probabilities to a JSON file
import json
with open(output_path, "w") as f:
    json.dump(observation_probabilities, f, indent=4)
print(f"Observation probabilities saved to {output_path}")