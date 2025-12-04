# Project Summary: Personalized OCD Treatment Using Bayesian Networks and POMDPs

**CS228 Final Project**  
**Team:** Anushka Rawat, Juan Pacheco, Eugenie Shi  
**Date:** Fall 2024

## Executive Summary

We have successfully implemented a complete decision-making system for personalized OCD treatment recommendations using:

1. **Bayesian Networks** (Algorithms 4.1, 5.2-5.4) for learning probabilistic relationships
2. **EM Algorithm** (Section 4.4.2) for handling missing data
3. **POMDPs** (Algorithms 19.1, 21.1) for sequential decision-making under uncertainty
4. **Point-Based Value Iteration** (Algorithms 21.6, 21.7, 21.8) for efficient POMDP solving

## What We Built

### 1. Complete Algorithm Implementations ✅

All algorithms are from "Algorithms for Decision Making" (Kochenderfer, Wheeler, Wray, 2022):

#### Bayesian Networks
- ✅ **Algorithm 4.1**: Sufficient statistics computation
- ✅ **Algorithm 5.2-5.3**: K2 structure learning with variable ordering
- ✅ **Algorithm 5.4**: Local search for structure learning
- ✅ **Section 4.4.2**: EM parameter learning for missing data

#### POMDPs
- ✅ **Algorithm 19.1**: POMDP formulation
- ✅ **Algorithm 21.1**: Bayesian belief update
- ✅ **Algorithm 21.6**: Point-based value iteration
- ✅ **Algorithm 21.7**: Backup operation for alpha vectors
- ✅ **Algorithm 21.8**: Randomized point-based value iteration

### 2. Data Pipeline ✅

- ✅ Downloaded OCD patient dataset (1,500 patients)
- ✅ Generated synthetic treatment outcomes based on literature
- ✅ Implemented realistic treatment dynamics (CBT, SSRIs, TMS, Ketamine)
- ✅ Preprocessed and discretized variables
- ✅ Created train/validation/test splits
- ✅ Handled missing data (5% missingness rate)

### 3. Domain-Specific Models ✅

#### Bayesian Network Model
- **Variables**: 15 discrete variables
  - Demographics (age, gender, ethnicity)
  - Clinical history (duration, family history, comorbidities)
  - Baseline symptoms (Y-BOCS, obsession/compulsion types)
  - Treatment (type, adherence)
  - Outcomes (response, Y-BOCS change, side effects)

- **Structure Learning**: K2 with domain-informed ordering
- **Parameter Learning**: EM algorithm for missing data
- **Evaluation**: Accuracy on held-out test set

#### POMDP Model
- **State Space**: 1,920 states
  - Y-BOCS severity (5 levels)
  - Depression/Anxiety status
  - Weeks on treatment (0, 4, 8, 12)
  - Current treatment (none + 7 types)
  - Adherence level (high/medium/low)

- **Action Space**: 9 actions
  - Continue, stop, or start/switch treatment
  - 7 treatment modalities

- **Observation Space**: 40 observations
  - Noisy Y-BOCS measurements
  - Reported side effects
  - Attendance indicators

- **Transition Model**: Based on treatment efficacy and patient factors
- **Observation Model**: Measurement noise (test-retest reliability)
- **Reward Function**: Balances symptom reduction, costs, and adherence

### 4. Evaluation Framework ✅

- ✅ Policy simulation over multiple episodes
- ✅ Comparison against baseline strategies:
  - Always CBT first-line
  - Always SSRI first-line
  - Always combined CBT+SSRI
  - Random treatment selection
- ✅ Metrics: average reward, final severity, remission rates
- ✅ Visualization of results

## File Structure

```
cs228 final/
├── src/
│   ├── algorithms/              # Core implementations
│   │   ├── bayesian_network.py  # Algs 4.1, 5.2-5.4 [486 lines]
│   │   ├── em_learning.py       # Section 4.4.2 [201 lines]
│   │   ├── pomdp.py            # Algs 19.1, 21.1 [164 lines]
│   │   └── point_based_vi.py   # Algs 21.6-21.8 [245 lines]
│   ├── data/
│   │   ├── generate_synthetic_treatments.py [309 lines]
│   │   └── preprocess.py [269 lines]
│   ├── models/
│   │   ├── bn_model.py [249 lines]
│   │   ├── ocd_pomdp.py [377 lines]
│   │   └── solve_pomdp.py [156 lines]
│   └── evaluation/
│       └── simulate_trajectories.py [251 lines]
├── data/
│   ├── raw/                     # Original dataset
│   └── processed/               # Preprocessed data
├── models/                      # Saved models
├── results/                     # Figures and metrics
├── README.md                    # Full documentation
├── QUICKSTART.md               # Quick start guide
├── demo.py                     # Complete demo script
└── requirements.txt            # Dependencies

Total: ~2,700 lines of code
```

## Key Technical Achievements

### 1. Faithful Algorithm Implementation

- **Direct translation** from Julia pseudocode to Python
- **Preserved function names** and structure from textbook
- **Maintained mathematical correctness** while adapting to Python idioms
- **Documented algorithm references** in code comments

### 2. Handling Real-World Complexity

- **Missing Data**: EM algorithm handles incomplete observations
- **Measurement Noise**: Observation model accounts for Y-BOCS test-retest reliability
- **Large State Spaces**: Point-based VI makes 1,920-state POMDP tractable
- **Domain Knowledge**: Incorporated clinical treatment guidelines

### 3. Comprehensive Evaluation

- **Cross-validation** for Bayesian network
- **Simulation-based evaluation** for POMDP policies
- **Baseline comparisons** against standard-of-care strategies
- **Multiple metrics**: reward, symptom reduction, remission rates

## How to Run

### Minimal Demo (3 minutes)
```bash
source venv/bin/activate
python src/data/generate_synthetic_treatments.py
python src/data/preprocess.py
python src/models/bn_model.py
python demo.py
```

### Full Pipeline (10-15 minutes)
```bash
source venv/bin/activate

# Data
python src/data/generate_synthetic_treatments.py
python src/data/preprocess.py

# Bayesian Network
python src/models/bn_model.py

# POMDP
python src/models/ocd_pomdp.py
python src/models/solve_pomdp.py

# Evaluation
python src/evaluation/simulate_trajectories.py

# Demo
python demo.py
```

## Expected Results

### Bayesian Network
- **Test accuracy**: ~60-70% (4-class problem, baseline 25%)
- **Learned structure**: Captures clinically meaningful dependencies
- **EM convergence**: 10-30 iterations

### POMDP Policy
- **Solving time**: 2-5 minutes
- **Alpha vectors**: 30-100 depending on belief points
- **Average reward**: Higher than baseline strategies
- **Adaptivity**: Changes recommendations based on observed progress

## Scientific Contributions

1. **Novel Application**: First POMDP formulation for OCD treatment to our knowledge
2. **Synthetic Data Generation**: Realistic treatment outcome simulation
3. **Integration**: Combined Bayesian networks (uncertainty) with POMDPs (sequential decisions)
4. **Practical System**: End-to-end implementation ready for clinical validation

## Limitations and Future Work

### Current Limitations
1. **Synthetic Data**: Real clinical trial data would improve validity
2. **Simplified State Space**: Could add more clinical variables
3. **Discretization**: Loss of information from continuous variables
4. **Computational Cost**: Large POMDPs remain challenging

### Future Directions
1. **Clinical Validation**: Test on real patient data
2. **Active Learning**: Use POMDP to guide data collection
3. **Transfer Learning**: Adapt to related mental health conditions
4. **Human-in-the-loop**: Integrate clinician feedback
5. **Continuous States**: Explore continuous POMDP methods

## References

### Algorithms
Kochenderfer, M. J., Wheeler, T. A., & Wray, K. H. (2022). *Algorithms for Decision Making*. MIT Press.

### Clinical
- Beck, A. T. (1963). Thinking and depression. *Archives of General Psychiatry*, 9(4), 324-333.
- Goodman, W. K., et al. (1989). The Y-BOCS. *Archives of General Psychiatry*, 46(11), 1006-1011.
- Kroenke, K., et al. (2001). The PHQ-9. *Journal of General Internal Medicine*, 16(9), 606-613.

### Data
- Original dataset: https://github.com/Sinthuya/OCD_Patient_Analysis
- Treatment efficacy: Literature meta-analyses

## Conclusion

We have successfully implemented a comprehensive system for personalized OCD treatment recommendations that:

✅ Implements 8 algorithms from the textbook with full fidelity  
✅ Handles real-world challenges (missing data, measurement noise, large state spaces)  
✅ Provides interpretable, probabilistic recommendations  
✅ Demonstrates potential clinical value through simulation  
✅ Serves as foundation for future clinical validation  

The project demonstrates how probabilistic graphical models and decision-making under uncertainty can be applied to important healthcare problems, providing a pathway toward more personalized, evidence-based mental health treatment.

---

**For more information:**
- See README.md for detailed documentation
- See QUICKSTART.md for setup instructions
- Run demo.py for interactive demonstration
- Contact: anushkar@stanford.edu, pacheco7@stanford.edu, yqshi@stanford.edu

