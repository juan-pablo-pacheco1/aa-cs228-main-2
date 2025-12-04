"""
Data Preprocessing for OCD Treatment Analysis
Discretizes variables and prepares data for Bayesian network learning
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.bayesian_network import Variable


def discretize_age(age: float) -> int:
    """Discretize age into 5 categories (1-indexed)"""
    if age < 25:
        return 1  # Young adult
    elif age < 40:
        return 2  # Early adulthood
    elif age < 55:
        return 3  # Middle age
    elif age < 70:
        return 4  # Older adult
    else:
        return 5  # Senior


def discretize_ybocs(score: float) -> int:
    """Discretize Y-BOCS total score (0-40) into 5 severity levels (1-indexed)"""
    if np.isnan(score):
        return np.nan
    score = np.clip(score, 0, 40)  # Ensure within valid range
    if score < 8:
        return 1  # Subclinical
    elif score < 16:
        return 2  # Mild
    elif score < 24:
        return 3  # Moderate
    elif score < 32:
        return 4  # Severe
    else:
        return 5  # Extreme


def discretize_duration(months: float) -> int:
    """Discretize symptom duration into 4 categories (1-indexed)"""
    if months < 12:
        return 1  # <1 year
    elif months < 60:
        return 2  # 1-5 years
    elif months < 120:
        return 3  # 5-10 years
    else:
        return 4  # 10+ years


def map_obsession_type(obs_type: str) -> int:
    """Map obsession type to integer (1-indexed)"""
    mapping = {
        'Contamination': 1,
        'Harm-related': 2,
        'Symmetry': 3,
        'Religious': 4,
        'Hoarding': 5,
        'Existential': 6
    }
    return mapping.get(obs_type, 1)


def map_compulsion_type(comp_type: str) -> int:
    """Map compulsion type to integer (1-indexed)"""
    mapping = {
        'Washing': 1,
        'Checking': 2,
        'Counting': 3,
        'Ordering': 4,
        'Praying': 5,
        'Mental': 6
    }
    return mapping.get(comp_type, 1)


def map_gender(gender: str) -> int:
    """Map gender to integer (1-indexed)"""
    return 1 if gender == 'Female' else 2


def map_ethnicity(ethnicity: str) -> int:
    """Map ethnicity to integer (1-indexed)"""
    mapping = {
        'African': 1,
        'Asian': 2,
        'Caucasian': 3,
        'Hispanic': 4
    }
    return mapping.get(ethnicity, 3)


def map_yes_no(value: str) -> int:
    """Map Yes/No to integer (1-indexed)"""
    return 1 if value == 'Yes' else 2


def map_treatment(treatment: str) -> int:
    """Map treatment type to integer (1-indexed)"""
    mapping = {
        'CBT': 1,
        'SSRI': 2,
        'TMS': 3,
        'Ketamine': 4,
        'CBT+SSRI': 5,
        'CBT+TMS': 6,
        'SSRI+TMS': 7
    }
    return mapping.get(treatment, 1)


def map_adherence(adherence: str) -> int:
    """Map adherence level to integer (1-indexed)"""
    mapping = {'High': 1, 'Medium': 2, 'Low': 3}
    return mapping.get(adherence, 2)


def map_side_effects(side_effects: str) -> int:
    """Map side effect severity to integer (1-indexed)"""
    mapping = {'None': 1, 'Mild': 2, 'Moderate': 3, 'Severe': 4}
    return mapping.get(side_effects, 1)


def map_response(response: str) -> int:
    """Map response category to integer (1-indexed)"""
    mapping = {
        'Remission': 1,
        'Partial Response': 2,
        'No Response': 3,
        'Dropout': 4
    }
    return mapping.get(response, 3)


def create_discrete_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Variable]]:
    """
    Create discretized dataset suitable for Bayesian network learning.
    
    Args:
        df: Raw dataframe with treatment data
    
    Returns:
        Tuple of (discretized dataframe, list of Variable objects)
    """
    df_discrete = pd.DataFrame()
    
    # Demographic variables
    df_discrete['Age_Group'] = df['Age'].apply(discretize_age)
    df_discrete['Gender'] = df['Gender'].apply(map_gender)
    df_discrete['Ethnicity'] = df['Ethnicity'].apply(map_ethnicity)
    
    # Clinical variables (baseline)
    df_discrete['Duration'] = df['Duration of Symptoms (months)'].apply(discretize_duration)
    df_discrete['YBOCS_Severity_Baseline'] = (
        df['Y-BOCS Score (Obsessions)'] + df['Y-BOCS Score (Compulsions)']
    ).apply(discretize_ybocs)
    df_discrete['Obsession_Type'] = df['Obsession Type'].apply(map_obsession_type)
    df_discrete['Compulsion_Type'] = df['Compulsion Type'].apply(map_compulsion_type)
    
    # Comorbidities
    df_discrete['Depression'] = df['Depression Diagnosis'].apply(map_yes_no)
    df_discrete['Anxiety'] = df['Anxiety Diagnosis'].apply(map_yes_no)
    df_discrete['Family_History'] = df['Family History of OCD'].apply(map_yes_no)
    
    # Treatment variables
    df_discrete['Treatment'] = df['Treatment'].apply(map_treatment)
    df_discrete['Adherence'] = df['Adherence'].apply(map_adherence)
    
    # Outcome variables
    df_discrete['Side_Effects'] = df['Side_Effects'].apply(map_side_effects)
    df_discrete['YBOCS_Severity_Week12'] = df['YBOCS_Total_Week12'].apply(discretize_ybocs)
    df_discrete['Response'] = df['Response_Category'].apply(map_response)
    
    # Define variable objects
    variables = [
        Variable('Age_Group', 5),
        Variable('Gender', 2),
        Variable('Ethnicity', 4),
        Variable('Duration', 4),
        Variable('YBOCS_Severity_Baseline', 5),
        Variable('Obsession_Type', 5),  # Only 5 types in data
        Variable('Compulsion_Type', 5),  # Only 5 types in data
        Variable('Depression', 2),
        Variable('Anxiety', 2),
        Variable('Family_History', 2),
        Variable('Treatment', 7),
        Variable('Adherence', 3),
        Variable('Side_Effects', 4),
        Variable('YBOCS_Severity_Week12', 5),
        Variable('Response', 4)
    ]
    
    return df_discrete, variables


def split_data(df: pd.DataFrame, train_ratio: float = 0.7, 
               val_ratio: float = 0.15, random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test sets.
    
    Args:
        df: Input dataframe
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    np.random.seed(random_seed)
    n = len(df)
    indices = np.random.permutation(n)
    
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
    return train_df, val_df, test_df


def introduce_missing_data(df: pd.DataFrame, missing_rate: float = 0.05, 
                          random_seed: int = 42) -> pd.DataFrame:
    """
    Introduce missing data to simulate realistic scenarios.
    
    Args:
        df: Input dataframe
        missing_rate: Proportion of values to set as missing
        random_seed: Random seed
    
    Returns:
        DataFrame with missing values (NaN)
    """
    np.random.seed(random_seed)
    df_missing = df.copy()
    
    # Don't introduce missing values in baseline demographics or treatment assignment
    protected_cols = ['Age_Group', 'Gender', 'Ethnicity', 'Duration', 
                     'YBOCS_Severity_Baseline', 'Treatment']
    
    # Can have missing in outcomes, adherence, etc.
    for col in df_missing.columns:
        if col not in protected_cols:
            mask = np.random.random(len(df_missing)) < missing_rate
            df_missing.loc[mask, col] = np.nan
    
    return df_missing


def preprocess_data(input_file: str, output_dir: str, missing_rate: float = 0.05):
    """
    Full preprocessing pipeline.
    
    Args:
        input_file: Path to input CSV with treatment data
        output_dir: Directory to save processed data
        missing_rate: Rate of missing data to introduce
    """
    print("Loading treatment data...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} patients")
    
    print("\nDiscretizing variables...")
    df_discrete, variables = create_discrete_dataset(df)
    
    print(f"Created {len(variables)} discrete variables:")
    for var in variables:
        print(f"  - {var.name}: {var.r} categories")
    
    print("\nSplitting data...")
    train_df, val_df, test_df = split_data(df_discrete)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Introduce missing data in training set only
    print(f"\nIntroducing {missing_rate*100:.1f}% missing data to training set...")
    train_df_missing = introduce_missing_data(train_df, missing_rate)
    
    # Calculate statistics
    missing_count = train_df_missing.isna().sum().sum()
    total_count = train_df_missing.size
    actual_missing_rate = missing_count / total_count
    print(f"Actual missing rate: {actual_missing_rate*100:.2f}%")
    
    # Save data
    print("\nSaving processed data...")
    os.makedirs(output_dir, exist_ok=True)
    
    train_df_missing.to_csv(os.path.join(output_dir, 'train_discrete.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_discrete.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_discrete.csv'), index=False)
    
    # Save variable definitions
    import json
    var_defs = {var.name: var.r for var in variables}
    with open(os.path.join(output_dir, 'variable_definitions.json'), 'w') as f:
        json.dump(var_defs, f, indent=2)
    
    print("Preprocessing complete!")
    
    return train_df_missing, val_df, test_df, variables


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(base_dir, "data/processed/ocd_with_treatments.csv")
    output_dir = os.path.join(base_dir, "data/processed")
    
    train_df, val_df, test_df, variables = preprocess_data(input_file, output_dir)

