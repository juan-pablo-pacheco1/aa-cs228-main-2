# Quick Start Guide

## Get Running in 5 Minutes

### 1. Setup (1 minute)

```bash
cd "/Users/anushkarawat/cs228 final"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Data (30 seconds)

```bash
python src/data/generate_synthetic_treatments.py
python src/data/preprocess.py
```

**Output:**
- 1,500 patients with synthetic treatment outcomes
- Train/val/test splits
- Discretized variables

### 3. Train Bayesian Network (2-3 minutes)

```bash
python src/models/bn_model.py
```

**What it does:**
- Learns structure using K2 algorithm (Algorithm 5.2-5.3)
- Learns parameters using EM (Section 4.4.2)
- Evaluates on test set
- Saves to `models/bayesian_network.pkl`

**Expected output:**
```
K2 learned XX edges
Final log-likelihood: -XXXX.XX
Accuracy: XX.XX%
```

### 4. Run Demo (30 seconds)

```bash
python demo.py
```

**Shows:**
- Bayesian network predictions
- Example sequential treatment planning
- Key results and findings

## Optional: Full Pipeline

### 5. Create POMDP (10 seconds)

```bash
python src/models/ocd_pomdp.py
```

**Creates:**
- State space: 1,920 states
- Action space: 9 actions
- Observation space: 40 observations

### 6. Solve POMDP (2-5 minutes)

```bash
python src/models/solve_pomdp.py
```

**What it does:**
- Generates belief points
- Runs Point-Based Value Iteration (Algorithm 21.6)
- Evaluates policy through simulation
- Saves to `models/pomdp_policy.pkl`

**Note:** This step is optional but required for POMDP policy comparison.

### 7. Evaluate and Compare (1-2 minutes)

```bash
python src/evaluation/simulate_trajectories.py
```

**Compares:**
- POMDP Policy (learned)
- CBT First-Line
- SSRI First-Line
- CBT+SSRI First-Line
- Random selection

**Generates:**
- `results/policy_comparison.png`
- `results/evaluation_results.pkl`

## Files Created

After running the full pipeline:

```
data/processed/
├── ocd_with_treatments.csv      # Synthetic treatment data
├── train_discrete.csv            # Training data
├── val_discrete.csv              # Validation data
├── test_discrete.csv             # Test data
└── variable_definitions.json     # Variable metadata

models/
├── bayesian_network.pkl          # Trained BN
└── pomdp_policy.pkl             # Trained policy (optional)

results/
├── bn_structure.png              # Network visualization
├── policy_comparison.png         # Evaluation plot
└── evaluation_results.pkl        # Evaluation metrics
```

## Troubleshooting

### "Module not found" error
```bash
# Make sure you're in the project directory
cd "/Users/anushkarawat/cs228 final"
source venv/bin/activate
```

### "File not found" error
```bash
# Run steps in order - later steps depend on earlier outputs
# Start from step 2 if you get file not found errors
```

### Long runtime
- Bayesian network training: 2-3 minutes (normal)
- POMDP solving: 2-5 minutes (normal)
- If >10 minutes, interrupt and restart

### Import errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

## Understanding the Output

### Bayesian Network Results

```
Accuracy: 65.33%
```
- **Interpretation:** Model correctly predicts treatment outcome 65% of the time
- **Baseline:** Random guessing would be ~25% (4 outcome categories)

### POMDP Policy Results

```
Average reward: -45.23 ± 12.45
```
- **Interpretation:** Higher (less negative) is better
- **Reward includes:** Symptom severity, treatment cost, adherence, switching penalty
- **Compare across policies** to see relative performance

### Policy Comparison

Look for:
- **Higher average reward** = Better overall strategy
- **Lower final Y-BOCS** = Better symptom reduction
- **Higher remission rate** = More patients achieving recovery

## Next Steps

1. **Modify parameters:** Edit hyperparameters in the scripts
2. **Try different orderings:** Change K2 ordering in `bn_model.py`
3. **Adjust POMDP rewards:** Modify reward function in `ocd_pomdp.py`
4. **Add more belief points:** Increase `num_beliefs` in `solve_pomdp.py`
5. **Extended evaluation:** Increase `num_episodes` in `simulate_trajectories.py`

## Getting Help

- **README.md**: Full documentation
- **Code comments**: Each file has detailed comments
- **Algorithm references**: See "Algorithms for Decision Making" textbook

## Quick Reference

| Task | Command | Time |
|------|---------|------|
| Generate data | `python src/data/generate_synthetic_treatments.py` | 30s |
| Preprocess | `python src/data/preprocess.py` | 10s |
| Train BN | `python src/models/bn_model.py` | 2-3m |
| Create POMDP | `python src/models/ocd_pomdp.py` | 10s |
| Solve POMDP | `python src/models/solve_pomdp.py` | 2-5m |
| Evaluate | `python src/evaluation/simulate_trajectories.py` | 1-2m |
| Demo | `python demo.py` | 30s |

**Total time for full pipeline:** ~10-15 minutes

