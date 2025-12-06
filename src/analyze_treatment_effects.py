import pandas as pd
import os

# Use relative paths for input and output files
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(base_dir, "data/processed/ocd_with_treatments.csv")
output_path = os.path.join(base_dir, "data/processed/treatment_statistics.csv")

# Load the dataset
data = pd.read_csv(file_path)

# Filter rows with valid YBOCS scores at Week 0 and Week 12
data = data.dropna(subset=["YBOCS_Total_Week0", "YBOCS_Total_Week12", "Treatment"])

# Calculate YBOCS reduction for each treatment type
data["YBOCS_Reduction"] = data["YBOCS_Total_Week0"] - data["YBOCS_Total_Week12"]

# Group by treatment and calculate mean and standard deviation of reductions
stats = data.groupby("Treatment")["YBOCS_Reduction"].agg(["mean", "std"])

# Print the results
print(stats)

# Save the statistics to a CSV file
stats.to_csv(output_path)
print(f"Statistics saved to {output_path}")