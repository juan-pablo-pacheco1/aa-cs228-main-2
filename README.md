# Personalized OCD Treatment Predictions Using Bayesian Networks and POMDPs

**CS228 Final Project**  
**Authors:** Anushka Rawat, Juan Pacheco, Eugenie Shi  
**Institution:** Stanford University

## Project Overview

This project implements a decision-making system for personalized Obsessive-Compulsive Disorder (OCD) treatment recommendations using probabilistic graphical models and partially observable Markov decision processes (POMDPs). The system:

1. Learns a Bayesian network from OCD patient data to capture probabilistic relationships between patient characteristics and treatment outcomes
2. Formulates OCD treatment selection as a POMDP to handle uncertainty and make sequential treatment decisions
3. Solves the POMDP using point-based value iteration to generate personalized treatment policies

## Key Features

- **Bayesian Network Structure Learning**: Implements K2 (Algorithms 5.2-5.3) and Local Search (Algorithm 5.4) from "Algorithms for Decision Making"
- **EM Parameter Learning**: Handles missing data using Expectation-Maximization (Section 4.4.2)
- **POMDP Formulation**: Models OCD treatment as a sequential decision problem under partial observability
- **Point-Based Value Iteration**: Solves POMDPs using Algorithms 21.6 and 21.8 (standard and randomized variants)
- **Synthetic Treatment Data**: Generates realistic treatment outcomes based on literature-informed efficacy rates

## Project Structure

```
cs228 final/
├── data/
│   ├── raw/                          # Original OCD patient dataset
│   └── processed/                    # Discretized and preprocessed data
├── src/
│   ├── algorithms/                   # Core algorithm implementations
│   │   ├── bayesian_network.py      # Algorithms 4.1, 5.2-5.4
│   │   ├── em_learning.py           # Section 4.4.2
│   │   ├── pomdp.py                 # Algorithms 19.1, 21.1
│   │   └── point_based_vi.py        # Algorithms 21.6, 21.7, 21.8
│   ├── data/                         # Data processing
│   │   ├── generate_synthetic_treatments.py
│   │   └── preprocess.py
│   └── models/                       # OCD-specific models
│       ├── bn_model.py               # Bayesian network for OCD
│       ├── ocd_pomdp.py             # POMDP formulation
│       └── solve_pomdp.py           # POMDP solver
├── notebooks/                        # Jupyter notebooks for analysis
├── models/                           # Saved trained models
├── results/                          # Figures and evaluation results
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. **Clone or navigate to the project directory:**
```bash
cd "/Users/anushkarawat/cs228 final"
```

2. **Create a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Generation and Preprocessing

Generate synthetic treatment outcome data and preprocess:

```bash
# Generate synthetic treatments
python src/data/generate_synthetic_treatments.py

# Preprocess and discretize data
python src/data/preprocess.py
```

### 2. Learn Bayesian Network

Train the Bayesian network to learn relationships between patient features and outcomes:

```bash
python src/models/bn_model.py
```

This will:
- Learn network structure using K2 algorithm
- Learn parameters using EM (handles missing data)
- Evaluate on test set
- Save model to `models/bayesian_network.pkl`

### 3. Formulate and Solve POMDP

Create and solve the OCD treatment POMDP:

```bash
# Test POMDP creation
python src/models/ocd_pomdp.py

# Solve POMDP using point-based value iteration
python src/models/solve_pomdp.py
```

This will:
- Create POMDP with learned transition dynamics
- Solve using Point-Based Value Iteration
- Evaluate policy through simulation
- Save policy to `models/pomdp_policy.pkl`

### 4. Example: Make Treatment Recommendations

```python
import pickle
import numpy as np

# Load trained models
with open('models/bayesian_network.pkl', 'rb') as f:
    bn_model = pickle.load(f)

with open('models/pomdp_policy.pkl', 'rb') as f:
    policy_data = pickle.load(f)
    policy = policy_data['policy']

# Example patient profile
patient = {
    'age_group': 3,          # Middle age
    'ybocs_baseline': 3,     # Moderate severity
    'depression': 1,         # No depression
    'anxiety': 1,            # No anxiety
    # ... other features
}

# Get treatment recommendation from policy
initial_belief = ...  # Construct from patient profile
recommended_action = policy.action(initial_belief)
print(f"Recommended treatment: {recommended_action}")
```

## Algorithms Implemented

All algorithms are from **"Algorithms for Decision Making"** by Kochenderfer, Wheeler, and Wray (2022).

### Bayesian Network Learning

| Algorithm | File | Description |
|-----------|------|-------------|
| **4.1** | `bayesian_network.py` | Sufficient Statistics Computation |
| **5.2-5.3** | `bayesian_network.py` | K2 Structure Learning |
| **5.4** | `bayesian_network.py` | Local Search for Structure Learning |
| **Section 4.4.2** | `em_learning.py` | EM Parameter Learning with Missing Data |

### POMDP Solving

| Algorithm | File | Description |
|-----------|------|-------------|
| **19.1** | `pomdp.py` | POMDP Definition |
| **21.1** | `pomdp.py` | Bayesian Belief Update |
| **21.6** | `point_based_vi.py` | Point-Based Value Iteration |
| **21.7** | `point_based_vi.py` | Backup Operation |
| **21.8** | `point_based_vi.py` | Randomized Point-Based Value Iteration |

## Data

### Original Dataset

- **Source**: [OCD Patient Analysis on GitHub](https://github.com/Sinthuya/OCD_Patient_Analysis)
- **Size**: 1,500 patients
- **Features**: Demographics, Y-BOCS scores, obsession/compulsion types, comorbidities

### Synthetic Treatment Data

We generate synthetic treatment outcomes for 4 treatment modalities:
- **CBT** (Cognitive Behavioral Therapy): 60-70% response rate
- **SSRIs** (Selective Serotonin Reuptake Inhibitors): 40-60% response rate
- **TMS** (Transcranial Magnetic Stimulation): 30-40% response rate for treatment-resistant cases
- **Ketamine**: 50% rapid response for severe cases

Response rates are modulated by:
- Symptom severity
- Obsession/compulsion types
- Comorbid conditions
- Treatment adherence
- Duration of treatment

## Model Details

### Bayesian Network

**Variables (15 total):**
- Demographics: Age, Gender, Ethnicity
- Clinical History: Duration, Family History
- Baseline Symptoms: Y-BOCS, Obsession Type, Compulsion Type
- Comorbidities: Depression, Anxiety
- Treatment: Type, Adherence
- Outcomes: Y-BOCS at Week 12, Response Category, Side Effects

**Structure Learning:**
- Uses domain-informed variable ordering for K2
- Validates with local search
- Handles missing data via EM

### POMDP Formulation

**State Space (1,920 states):**
- Y-BOCS Severity: 5 levels
- Depression: Yes/No
- Anxiety: Yes/No
- Weeks on Treatment: 0, 4, 8, 12
- Current Treatment: None + 7 types
- Adherence Level: High/Medium/Low

**Action Space (9 actions):**
- Continue current treatment
- Stop treatment
- Start/switch to: CBT, SSRI, TMS, Ketamine, CBT+SSRI, CBT+TMS, SSRI+TMS

**Observation Space (40 observations):**
- Observed Y-BOCS (with measurement noise)
- Reported Side Effects: None/Mild/Moderate/Severe
- Attendance: Yes/No

**Reward Function:**
```
R(s, a) = -10 * ybocs_severity
          - treatment_cost / 100
          - adherence_penalty
          - switching_penalty
          + remission_bonus
```

## Results

### Bayesian Network Performance

- **Test Accuracy**: ~XX% (see results from training)
- **Learned edges**: Captures clinically meaningful relationships
- **EM Convergence**: Typically converges in 10-30 iterations

### POMDP Policy

- **Solution Method**: Point-Based Value Iteration
- **Belief Points**: 30-50 sampled beliefs
- **Iterations**: 5-10 for convergence
- **Average Reward**: Evaluated through simulation

## References

1. Kochenderfer, M. J., Wheeler, T. A., & Wray, K. H. (2022). *Algorithms for Decision Making*. MIT Press.
2. Beck, A. T. (1963). Thinking and depression: I. Idiosyncratic content and cognitive distortions. *Archives of General Psychiatry*, 9(4), 324-333.
3. Goodman, W. K., et al. (1989). The Yale-Brown Obsessive Compulsive Scale (Y-BOCS). *Archives of General Psychiatry*, 46(11), 1006-1011.
4. Kroenke, K., Spitzer, R. L., & Williams, J. B. (2001). The PHQ-9. *Journal of General Internal Medicine*, 16(9), 606-613.
5. Spitzer, R. L., et al. (2006). A brief measure for assessing generalized anxiety disorder (GAD-7). *Archives of Internal Medicine*, 166(10), 1092-1097.

## License

This project is for educational purposes as part of CS228 at Stanford University.

## Contact

- Anushka Rawat: anushkar@stanford.edu
- Juan Pacheco: pacheco7@stanford.edu
- Eugenie Shi: yqshi@stanford.edu

## Acknowledgments

- CS228 course staff for guidance and feedback
- Stanford OCD Lab for domain expertise
- Original dataset authors for making data publicly available

