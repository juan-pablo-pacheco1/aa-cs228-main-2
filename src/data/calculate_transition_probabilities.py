import os
import pandas as pd
from collections import defaultdict

# Load the dataset
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'ocd_with_treatments.csv')
data = pd.read_csv(data_path)

# Filter relevant treatments
treatments_of_interest = ["CBT", "CBT+SSRI"]
filtered_data = data[data['Treatment'].isin(treatments_of_interest)]

# Initialize transition probability storage
transition_probabilities = defaultdict(lambda: defaultdict(int))

# Calculate transition probabilities
for treatment in treatments_of_interest:
    treatment_data = filtered_data[filtered_data['Treatment'] == treatment]
    total_patients = len(treatment_data)

    if total_patients > 0:
        response_counts = treatment_data['Response_Category'].value_counts()
        for response, count in response_counts.items():
            transition_probabilities[treatment][response] = count / total_patients

# Display the calculated probabilities
for treatment, probabilities in transition_probabilities.items():
    print(f"Transition probabilities for {treatment}:")
    for response, probability in probabilities.items():
        print(f"  {response}: {probability:.2f}")