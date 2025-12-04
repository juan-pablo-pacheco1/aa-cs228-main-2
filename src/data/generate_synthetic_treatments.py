"""
Generate Synthetic Treatment Outcome Data for OCD Patients
Based on literature-informed treatment efficacy rates and clinical patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class TreatmentSimulator:
    """
    Simulates OCD treatment outcomes based on literature parameters.
    
    Treatment efficacy rates (from literature):
    - CBT: 60-70% response rate, works better for contamination/checking
    - SSRI: 40-60% response rate, 8-12 week onset
    - TMS: 30-40% response rate for treatment-resistant
    - Ketamine: 50% rapid response for severe cases
    - Combined CBT+SSRI: 70-80% response rate
    """
    
    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
        
        # Treatment response rates (baseline)
        self.response_rates = {
            'CBT': 0.65,
            'SSRI': 0.50,
            'TMS': 0.35,
            'Ketamine': 0.50,
            'CBT+SSRI': 0.75,
            'CBT+TMS': 0.55,
            'SSRI+TMS': 0.50
        }
        
        # Side effect rates
        self.side_effect_rates = {
            'CBT': 0.05,  # Minimal (temporary anxiety)
            'SSRI': 0.30,  # GI issues, sexual dysfunction
            'TMS': 0.15,  # Headache, scalp discomfort
            'Ketamine': 0.40,  # Dissociation, nausea
            'CBT+SSRI': 0.30,
            'CBT+TMS': 0.15,
            'SSRI+TMS': 0.35
        }
        
        # Dropout rates
        self.dropout_rates = {
            'CBT': 0.20,  # Time commitment
            'SSRI': 0.25,  # Side effects, delayed onset
            'TMS': 0.15,  # Accessibility
            'Ketamine': 0.20,  # Cost, stigma
            'CBT+SSRI': 0.22,
            'CBT+TMS': 0.18,
            'SSRI+TMS': 0.20
        }
    
    def assign_treatment(self, row: pd.Series) -> str:
        """
        Assign treatment based on patient characteristics.
        Simulates realistic clinical decision-making.
        """
        ybocs = row['Y-BOCS Score (Obsessions)'] + row['Y-BOCS Score (Compulsions)']
        has_depression = row['Depression Diagnosis'] == 'Yes'
        has_anxiety = row['Anxiety Diagnosis'] == 'Yes'
        duration = row['Duration of Symptoms (months)']
        
        # Treatment assignment logic
        if ybocs < 24:  # Mild to moderate
            # First-line: CBT or SSRI
            if np.random.random() < 0.6:
                return 'CBT'
            else:
                return 'SSRI'
        elif ybocs < 32:  # Moderate to severe
            # Often combined treatment
            if has_depression or has_anxiety:
                if np.random.random() < 0.5:
                    return 'CBT+SSRI'
                else:
                    return np.random.choice(['CBT', 'SSRI'])
            else:
                return np.random.choice(['CBT', 'SSRI', 'CBT+SSRI'], p=[0.3, 0.3, 0.4])
        else:  # Severe
            # May include TMS or Ketamine for treatment-resistant
            if duration > 120:  # Chronic (10+ years)
                return np.random.choice(['CBT+SSRI', 'SSRI+TMS', 'TMS', 'Ketamine'], 
                                       p=[0.4, 0.25, 0.2, 0.15])
            else:
                return np.random.choice(['CBT+SSRI', 'SSRI', 'CBT'], p=[0.5, 0.3, 0.2])
    
    def compute_response_probability(self, row: pd.Series, treatment: str) -> float:
        """
        Compute probability of treatment response based on patient factors.
        """
        base_rate = self.response_rates[treatment]
        
        # Modifiers based on patient characteristics
        obsession_type = row['Obsession Type']
        compulsion_type = row['Compulsion Type']
        ybocs = row['Y-BOCS Score (Obsessions)'] + row['Y-BOCS Score (Compulsions)']
        has_depression = row['Depression Diagnosis'] == 'Yes'
        has_anxiety = row['Anxiety Diagnosis'] == 'Yes'
        
        # CBT works better for certain types
        if 'CBT' in treatment:
            if obsession_type in ['Contamination', 'Harm-related'] and \
               compulsion_type in ['Washing', 'Checking']:
                base_rate *= 1.15
            elif obsession_type == 'Religious':
                base_rate *= 0.90
        
        # SSRIs work better with comorbid anxiety/depression
        if 'SSRI' in treatment:
            if has_depression or has_anxiety:
                base_rate *= 1.10
        
        # More severe cases respond less
        if ybocs > 32:
            base_rate *= 0.85
        elif ybocs < 20:
            base_rate *= 1.10
        
        # Age effects
        age = row['Age']
        if age < 25:
            base_rate *= 1.05  # Better plasticity
        elif age > 60:
            base_rate *= 0.95
        
        return np.clip(base_rate, 0.1, 0.95)
    
    def simulate_outcome(self, row: pd.Series, treatment: str) -> Dict:
        """
        Simulate treatment outcome over 12 weeks.
        Returns measurements at weeks 0, 4, 8, 12.
        """
        # Initial Y-BOCS
        ybocs_obs_0 = row['Y-BOCS Score (Obsessions)']
        ybocs_comp_0 = row['Y-BOCS Score (Compulsions)']
        ybocs_total_0 = ybocs_obs_0 + ybocs_comp_0
        
        # Determine if patient drops out
        dropped_out = np.random.random() < self.dropout_rates[treatment]
        dropout_week = np.random.choice([4, 8]) if dropped_out else None
        
        # Determine if patient responds to treatment
        response_prob = self.compute_response_probability(row, treatment)
        responds = np.random.random() < response_prob
        
        # Determine if patient has side effects
        has_side_effects = np.random.random() < self.side_effect_rates[treatment]
        side_effect_severity = np.random.choice(['None', 'Mild', 'Moderate', 'Severe'],
                                                p=[0.7, 0.2, 0.08, 0.02] if not has_side_effects 
                                                else [0.0, 0.5, 0.35, 0.15])
        
        # Simulate adherence (affects response)
        adherence = np.random.choice(['High', 'Medium', 'Low'], p=[0.5, 0.35, 0.15])
        if adherence == 'Low':
            responds = responds and (np.random.random() < 0.4)
        elif adherence == 'Medium':
            responds = responds and (np.random.random() < 0.7)
        
        # Simulate Y-BOCS trajectory
        if responds:
            # Response: gradual improvement
            if 'SSRI' in treatment and 'CBT' not in treatment:
                # SSRIs have delayed onset
                reduction_week4 = np.random.uniform(0.05, 0.15)
                reduction_week8 = np.random.uniform(0.20, 0.35)
                reduction_week12 = np.random.uniform(0.35, 0.55)
            elif 'Ketamine' in treatment:
                # Rapid but potentially less sustained
                reduction_week4 = np.random.uniform(0.30, 0.45)
                reduction_week8 = np.random.uniform(0.35, 0.50)
                reduction_week12 = np.random.uniform(0.30, 0.50)
            else:
                # CBT: more linear improvement
                reduction_week4 = np.random.uniform(0.15, 0.25)
                reduction_week8 = np.random.uniform(0.30, 0.45)
                reduction_week12 = np.random.uniform(0.40, 0.60)
        else:
            # Non-response: minimal or no improvement
            reduction_week4 = np.random.uniform(0.0, 0.10)
            reduction_week8 = np.random.uniform(0.0, 0.15)
            reduction_week12 = np.random.uniform(0.0, 0.20)
        
        # Apply reductions
        if dropout_week == 4:
            ybocs_4 = ybocs_total_0 * (1 - reduction_week4 * 0.5)
            ybocs_8 = np.nan
            ybocs_12 = np.nan
        elif dropout_week == 8:
            ybocs_4 = ybocs_total_0 * (1 - reduction_week4)
            ybocs_8 = ybocs_total_0 * (1 - reduction_week8 * 0.7)
            ybocs_12 = np.nan
        else:
            ybocs_4 = ybocs_total_0 * (1 - reduction_week4)
            ybocs_8 = ybocs_total_0 * (1 - reduction_week8)
            ybocs_12 = ybocs_total_0 * (1 - reduction_week12)
        
        # Add measurement noise
        ybocs_4 = ybocs_4 + np.random.normal(0, 2) if not np.isnan(ybocs_4) else ybocs_4
        ybocs_8 = ybocs_8 + np.random.normal(0, 2) if not np.isnan(ybocs_8) else ybocs_8
        ybocs_12 = ybocs_12 + np.random.normal(0, 2) if not np.isnan(ybocs_12) else ybocs_12
        
        # Clip to valid range
        ybocs_4 = np.clip(ybocs_4, 0, 40) if not np.isnan(ybocs_4) else ybocs_4
        ybocs_8 = np.clip(ybocs_8, 0, 40) if not np.isnan(ybocs_8) else ybocs_8
        ybocs_12 = np.clip(ybocs_12, 0, 40) if not np.isnan(ybocs_12) else ybocs_12
        
        # Determine response category (≥35% reduction = response, ≥50% = remission)
        if not np.isnan(ybocs_12):
            reduction = (ybocs_total_0 - ybocs_12) / ybocs_total_0
            if reduction >= 0.50 and ybocs_12 <= 12:
                response_category = 'Remission'
            elif reduction >= 0.35:
                response_category = 'Partial Response'
            else:
                response_category = 'No Response'
        else:
            response_category = 'Dropout'
        
        return {
            'Treatment': treatment,
            'YBOCS_Total_Week0': ybocs_total_0,
            'YBOCS_Total_Week4': ybocs_4,
            'YBOCS_Total_Week8': ybocs_8,
            'YBOCS_Total_Week12': ybocs_12,
            'Response_Category': response_category,
            'Side_Effects': side_effect_severity,
            'Adherence': adherence,
            'Dropped_Out': dropped_out
        }


def generate_treatment_data(input_file: str, output_file: str, random_seed: int = 42):
    """
    Generate synthetic treatment data for OCD patients.
    
    Args:
        input_file: Path to original OCD patient dataset
        output_file: Path to save enhanced dataset with treatments
        random_seed: Random seed for reproducibility
    """
    print("Loading OCD patient data...")
    df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df)} patients")
    print(f"Columns: {df.columns.tolist()}")
    
    # Initialize simulator
    simulator = TreatmentSimulator(random_seed=random_seed)
    
    # Generate treatment data for each patient
    print("\nGenerating synthetic treatment data...")
    treatment_data = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing patient {idx + 1}/{len(df)}...")
        
        # Assign treatment
        treatment = simulator.assign_treatment(row)
        
        # Simulate outcome
        outcome = simulator.simulate_outcome(row, treatment)
        
        # Combine original and treatment data
        patient_data = row.to_dict()
        patient_data.update(outcome)
        
        treatment_data.append(patient_data)
    
    # Create DataFrame
    df_with_treatment = pd.DataFrame(treatment_data)
    
    # Save to file
    print(f"\nSaving to {output_file}...")
    df_with_treatment.to_csv(output_file, index=False)
    
    # Print summary statistics
    print("\n=== Treatment Assignment Summary ===")
    print(df_with_treatment['Treatment'].value_counts())
    
    print("\n=== Response Category Summary ===")
    print(df_with_treatment['Response_Category'].value_counts())
    
    print("\n=== Side Effects Summary ===")
    print(df_with_treatment['Side_Effects'].value_counts())
    
    print("\n=== Adherence Summary ===")
    print(df_with_treatment['Adherence'].value_counts())
    
    print(f"\nDropout rate: {df_with_treatment['Dropped_Out'].sum() / len(df_with_treatment):.1%}")
    
    print("\nData generation complete!")
    
    return df_with_treatment


if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(base_dir, "data/raw/ocd_patient_dataset.csv")
    output_file = os.path.join(base_dir, "data/processed/ocd_with_treatments.csv")
    
    df = generate_treatment_data(input_file, output_file)

