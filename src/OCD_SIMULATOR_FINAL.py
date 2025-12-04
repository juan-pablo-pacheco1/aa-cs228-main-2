import numpy as np
import pandas as pd
import random
import math 

# ============================================================
# 0. CLINICIAN-CONFIGURABLE PARAMETERS (ALL EXPLICIT)
# ============================================================

CLINICIAN = {

    # --------------------------------------------------------
    # AGE DISTRIBUTION
    # checked
    # --------------------------------------------------------
    # Source:
    # - https://www.nimh.nih.gov/health/statistics/obsessive-compulsive-disorder-ocd?utm_source=chatgpt.com
    # exact: figure 1
    "age_groups": [
    ((18, 29), 1.5/4.5),
    ((30, 44), 1.4/4.5),
    ((45, 59), 1.1/4.5),
    ((60, 90), 0.5/4.5),
    ], 

    # --------------------------------------------------------
    # SEX DISTRIBUTION
    # checked
    # --------------------------------------------------------
    # https://pubmed.ncbi.nlm.nih.gov/32603559/
    # exact:  In a typical sample, women were 1.6 times more 
    # likely to experience OCD compared to men, with lifetime prevalence rates of 1.5% in women and 1.0% in men. 
    "female_probability": 1.5/2.5,  

    # --------------------------------------------------------
    # OCD SUBTYPE PREVALENCE
    # checked
    # --------------------------------------------------------
    # Source:
    # - https://pmc.ncbi.nlm.nih.gov/articles/PMC2797569/
    # Table 1: "Prevalence of each O/C in the subsample assessed for OCD"
    #
    # Exact percentages (column 1, in order appearing):
    #   Contamination:      2.9%
    #   Checking:          15.4%
    #   Ordering:           9.1%
    #   Hoarding:          14.4%   <-- EXCLUDED based on clinician feedback
    #   Sexual/religious:   2.3%
    #   Moral:              4.2%
    #   Harming:            1.7%
    #   Illness:            1.8%
    #   Other O/C:          1.1%
    #   Any of the above    28.2
    #
    # Total (excluding hoarding) = 2.9+15.4+9.1+2.3+4.2+1.7+1.8+1.1 + = 66.7
    #
    "subtype_probabilities": [
        ("contamination",     2.9/66.7),
        ("checking",         15.4/66.7),
        ("ordering",          9.1/66.7),
        ("sexual_religious",  2.3/66.7),
        ("moral",             4.2/66.7),
        ("harming",           1.7/66.7),
        ("illness",           1.8/66.7),
        ("other_oc",          1.1/66.7),
        ("any of the above",  28.2/66.7),
    ],

# --------------------------------------------------------
# SSRIs - YBOCS CHANGE FROM BASELINE
# checked
# --------------------------------------------------------
# Source:
# - https://pmc.ncbi.nlm.nih.gov/articles/PMC7025764/
# - "WMD for citalopram was ‐3.63 (95% CI ‐5.20 to ‐2.06, n=401), WMD for fluoxetine was ‐3.07 (95% CI ‐5.32 to ‐0.82, n=606), WMD for fluvoxamine was ‐3.87 (95% CI ‐5.69 to ‐2.04, n=566), WMD for paroxetine was ‐3.36 (95% CI ‐4.55 to ‐2.17, n=833) and WMD for sertraline was ‐2.45 (95% CI ‐3.54 to ‐1.35, n=691)."
# Source for ERP_SRI:
    # - https://www.sciencedirect.com/science/article/abs/pii/S0005796716301218
"ocd_treatment_ybocs": {
    "fluoxetine": {
        "mean_change": -3.07,
        "ci_95": (-5.32, -0.82),
    },
    "sertraline": {
        "mean_change": -2.45,
        "ci_95": (-3.54, -1.35),
    },
    "paroxetine": {
        "mean_change": -3.36,
        "ci_95": (-4.55, -2.17),
    },
    "fluvoxamine": {
        "mean_change": -3.87,
        "ci_95": (-5.69, -2.04),
    },
    "citalopram": {
        "mean_change": -3.63,
        "ci_95": (-5.20, -2.06),
    },
    "ERP_and_SRI": {
        "mean_change": -14.03,
        "sd": 7.11,
    },
},
    # --------------------------------------------------------
    # YBOCS 
    # checked
    # Source: Table 1: https://pmc.ncbi.nlm.nih.gov/articles/PMC3272757/
    # --------------------------------------------------------
    "baseline_ybocs": {
        "mean": 20.26,
        "sd": 8.4,
    }
}




# ============================================================
# APA ALGORITHM (SIMPLIFIED) checked!
# ============================================================
# Source: https://psychiatryonline.org/pb/assets/raw/sitewide/practice_guidelines/guidelines/ocd-1410197738287.pdf
# sOURCE:  https://med.stanford.edu/ocd/about/diagnosis.html?utm_source=chatgpt.com
#  In out experience, patients experience a 25% decrease in a Y-BOCS score as mild to moderate improvement....In controlled treatment trials, a decrease of greater than or equal to 35% is widely accepted as indicating a clinically meaningful 
# response and translates into a global improvement rating of much or very much improved
# Source for remission: https://www.psychiatrist.com/jcp/response-versus-remission-obsessive-compulsive-disorder/?utm_source=chatgpt.com
# quote-- remission rates for YBOCS < = 12:

APA_ALGORITHM = {
    "first_line": ["SSRI", "ERP_and_SRI"],
    
    "second_line": ["switch_different_SSRI"],
    
    "third_line": ["switch_different_SSRI"],
    
    "response_thresholds": {
        "clinically_meaningful": 0.35,      # ≥35% YBOCS reduction
        "moderate": 0.25,      # 25-35% reduction
        "remission_ybocs": 12  # YBOCS ≤12
    }
}


APA_ALGORITHM = {
    # Source: https://www.aafp.org/pubs/afp/issues/2015/1115/p896.html
    # Quote: "A trial of SSRI therapy should continue for 8 to 12 weeks, with at least 4 to 6 weeks at the maximal tolerable dosage."
    "first_line": {
        "options": ["SSRI", "CBT", "SSRI_plus_CBT"],
        "duration_weeks": "8-12 total (4-6 at maximal dose)",
        "note": "All three are equally valid first-line per APA"
    },
    

    # Source: https://www.aafp.org/pubs/afp/issues/2015/1115/p896.html
    # Quote1: "If there is no response to trials of at least two SSRIs, the patient should be referred to a psychiatrist. Clomipramine is an option in these patients."
    # Quote2: "Addition of an atypical antipsychotic is effective for some patients with inadequate response to SSRI therapy."
    "second_line": {
        "if_poor_response": [
            "switch_different_SSRI",  # First choice
            "switch_clomipramine",
            "augment_antipsychotic"
        ],
        "if_moderate_response": [
            "augment_antipsychotic",  # First choice for partial responders
            "add_CBT_if_not_provided"
        ]
    },
    
    # Source: https://www.aafp.org/pubs/afp/issues/2015/1115/p896.html
    "third_line": [
        "switch_different_augmenting_antipsychotic",
        "switch_different_SRI",
        "augment_clomipramine",
        "augment_glutamate_modulator"
    ],
    
    # Source: https://pubmed.ncbi.nlm.nih.gov/26833615/
    "response_thresholds": {
        "adequate": 0.35,      # ≥35% YBOCS reduction (Pallanti 2002, FDA standard)
        "moderate": 0.25,      # 25-35% reduction (Clinical trial convention)
        "remission_ybocs": 12  # YBOCS ≤12 (Consensus definition)
    }
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def weighted_choice(weighted_items):
    """
    weighted_items: list of (value, weight) tuples
    e.g. [ ((18, 29), 1.5/4.5), ((30, 44), 1.4/4.5), ... ]
    Returns ONE (value, weight) tuple sampled according to weights.
    """
    values, weights = zip(*weighted_items)
    # choose one element from weighted_items, respecting weights
    chosen = random.choices(weighted_items, weights=weights, k=1)[0]
    return chosen


def sample_age():
    (age_min, age_max), _ = weighted_choice(CLINICIAN["age_groups"])
    return int(np.random.randint(age_min, age_max + 1))

def classify_severity(ybocs_score):
    if ybocs_score <= 7: return "subclinical"
    if ybocs_score <= 15: return "mild"
    if ybocs_score <= 23: return "moderate"
    if ybocs_score <= 31: return "severe"
    return "extreme"

# CHECK THIS
def assess_response(ybocs_baseline, ybocs_current):
    if ybocs_current is None:
        return {"category": "dropout", "pct_change": None, "remission": False}
    
    change = ybocs_baseline - ybocs_current
    pct_change = (change / ybocs_baseline) if ybocs_baseline > 0 else 0
    
    if pct_change >= APA_ALGORITHM["response_thresholds"]["adequate"]:
        category = "adequate"
    elif pct_change >= APA_ALGORITHM["response_thresholds"]["moderate"]:
        category = "moderate"
    else:
        category = "poor"
    
    return {
        "category": category,
        "pct_change": pct_change,
        "remission": ybocs_current <= APA_ALGORITHM["response_thresholds"]["remission_ybocs"]
    }

# CHECK THIS
def get_valid_actions(treatment_history, last_response):
    if len(treatment_history) == 0:
        # Return just the treatment options, not the entire dict
        return APA_ALGORITHM["first_line"]["options"]
    elif last_response == "adequate":
        return ["continue"]
    elif len(treatment_history) == 1:
        # Second line returns different structures depending on response
        second_line = APA_ALGORITHM["second_line"]
        if last_response == "poor":
            return second_line["if_poor_response"]
        elif last_response == "moderate":
            return second_line["if_moderate_response"]
        else:
            # Fallback: combine both poor and moderate options
            return second_line["if_poor_response"]
    else:
        return APA_ALGORITHM["third_line"]
    

# *** ADD THE NEW FUNCTION HERE ***
def guideline_based_decision(valid_actions, treatment_history, patient, last_response, trajectory):
    """
    Implements APA Practice Guidelines for OCD (Figure shown in upload).
    This is what clinicians SHOULD do according to evidence-based guidelines.
    """
    
    # ========================================
    # FIRST-LINE (Step 0): No prior treatment
    # ========================================
    if len(treatment_history) == 0:
        # Per guidelines: All three are equally valid first-line
        # Clinicians often prefer combination for moderate-severe cases
        if patient['severity'] in ['severe', 'extreme']:
            if 'SSRI_plus_CBT' in valid_actions:
                return 'SSRI_plus_CBT'
        
        # Otherwise, slight preference for monotherapy due to access/cost
        # 60% SSRI, 25% CBT, 15% combination (based on real-world practice patterns)
        weights = []
        actions_subset = []
        
        if 'SSRI' in valid_actions:
            actions_subset.append('SSRI')
            weights.append(0.60)
        if 'CBT' in valid_actions:
            actions_subset.append('CBT')
            weights.append(0.25)
        if 'SSRI_plus_CBT' in valid_actions:
            actions_subset.append('SSRI_plus_CBT')
            weights.append(0.15)
        
        if actions_subset:
            # Normalize weights
            total = sum(weights)
            weights = [w/total for w in weights]
            return np.random.choice(actions_subset, p=weights)
        
        return valid_actions[0]  # Fallback
    
    # ========================================
    # AFTER FIRST TREATMENT: Response-dependent
    # ========================================
    
    # If adequate response, continue treatment
    if last_response == "adequate":
        return "continue"
    
    # ========================================
    # MODERATE RESPONSE (25-35% improvement)
    # ========================================
    if last_response == "moderate":
        # Per guidelines: "Augment with second-generation antipsychotic or add CBT (ERP)"
        
        # Check if patient already on CBT
        has_cbt = any(tx in ['CBT', 'SSRI_plus_CBT', 'ERP_and_SRI', 'add_CBT_if_not_provided'] 
                     for tx in treatment_history)
        
        # Prefer adding CBT if not already provided (less side effects)
        if not has_cbt and 'add_CBT_if_not_provided' in valid_actions:
            return 'add_CBT_if_not_provided'
        
        # Otherwise augment with antipsychotic
        if 'augment_antipsychotic' in valid_actions:
            return 'augment_antipsychotic'
        
        # Fallback to switching
        if 'switch_different_SSRI' in valid_actions:
            return 'switch_different_SSRI'
    
    # ========================================
    # POOR/NO RESPONSE (<25% improvement)
    # ========================================
    if last_response == "poor":
        num_ssri_trials = sum(1 for tx in treatment_history 
                              if tx in ['SSRI', 'switch_different_SSRI', 'switch_different_SRI'])
        
        # Per guidelines: Try at least 2 different SSRIs before other options
        if num_ssri_trials < 2:
            if 'switch_different_SSRI' in valid_actions:
                return 'switch_different_SSRI'
        
        # After 2 SSRI failures: Multiple options per guideline
        # Priority: switch to clomipramine, then augment antipsychotic, then venlafaxine
        
        if 'switch_clomipramine' in valid_actions:
            # 40% chance - considered after 2 SSRI failures
            if np.random.rand() < 0.40:
                return 'switch_clomipramine'
        
        if 'augment_antipsychotic' in valid_actions:
            # 35% chance
            if np.random.rand() < 0.35:
                return 'augment_antipsychotic'
        
        if 'switch_different_SSRI' in valid_actions:
            # 25% chance - try another SSRI
            return 'switch_different_SSRI'
    
    # ========================================
    # THIRD-LINE: After multiple failures
    # ========================================
    if len(treatment_history) >= 2:
        # Per guideline: "Switch to different augmenting antipsychotic, different SRI, 
        # augment with clomipramine, or glutamate antagonist"
        
        # Randomize among valid third-line options
        third_line_options = []
        
        for action in valid_actions:
            if action in ['switch_different_augmenting_antipsychotic', 
                         'switch_different_SRI',
                         'augment_clomipramine',
                         'augment_glutamate_modulator']:
                third_line_options.append(action)
        
        if third_line_options:
            return np.random.choice(third_line_options)
        
        # Fallback
        if 'switch_different_SSRI' in valid_actions:
            return 'switch_different_SSRI'
    
    # Final fallback
    return valid_actions[0] if valid_actions else None

# ============================================================
# POMDP BLACK BOX FUNCTIONS 
# ... (rest of your code continues)

# ============================================================
# POMDP BLACK BOX FUNCTIONS 
# NEED TO TIE IN THE ALREADY IMPLEMENTED ONES!
# ============================================================

def POMDP_INITIALIZE_BELIEF(patient_profile):
    baseline = float(patient_profile["ybocs_baseline"])
    belief = {
        "patient": patient_profile,
        "step": 0,
        "last_ybocs": baseline,
        # Online estimate of per-action improvement (reward)
        "action_stats": {},  # action -> {"n": int, "mean_delta": float}
        "history": [],
        "dropped_out": False,
    }
    return belief


def POMDP_GET_TRUE_STATE(patient_profile):
    baseline = float(patient_profile["ybocs_baseline"])
    responsiveness = float(np.random.lognormal(mean=0.0, sigma=0.35))
    return {
        "profile": patient_profile,
        "true_ybocs": baseline,
        "responsiveness": responsiveness,
        "step": 0,
        "dropped_out": False,
        "last_action": None,
    }


def POMDP_TRANSITION(current_state, action):
    # Copy to avoid mutating caller's reference except for simple types.
    next_state = dict(current_state)
    next_state["step"] = int(next_state.get("step", 0) + 1)
    next_state["last_action"] = action

    # If the patient has already dropped out, remain in an absorbing state.
    if next_state.get("dropped_out", False):
        return next_state

    baseline = float(next_state["profile"]["ybocs_baseline"])
    true_ybocs = float(next_state.get("true_ybocs", baseline))

    # Sample a patient-specific responsiveness factor once.
    if "responsiveness" not in next_state:
        next_state["responsiveness"] = float(np.random.lognormal(mean=0.0, sigma=0.35))
    responsiveness = float(next_state["responsiveness"])

    # Local helper: map high-level APA action to a Y-BOCS change distribution.
    def _effect_distribution(act):
        data = CLINICIAN["ocd_treatment_ybocs"]

        # SSRI-like actions: pick a random SSRI from meta-analysis
        if act in {"SSRI", "switch_different_SSRI", "switch_different_SRI", "switch_clomipramine"}:
            ssri_keys = [k for k, v in data.items() if "ci_95" in v]
            drug = random.choice(ssri_keys)
            mean = float(data[drug]["mean_change"])
            lo, hi = data[drug]["ci_95"]
            sd = float(abs(hi - lo) / (2.0 * 1.96))
            return mean, sd

        # ERP + SRI or CBT combinations: use ERP_and_SRI effect
        if act in {"ERP_and_SRI", "SSRI_plus_CBT", "add_CBT_if_not_provided"}:
            mean = float(data["ERP_and_SRI"]["mean_change"])
            sd = float(data["ERP_and_SRI"]["sd"])
            return mean, sd

        # CBT alone: approximate as ~half of ERP+SRI effect
        if act == "CBT":
            mean = float(data["ERP_and_SRI"]["mean_change"]) / 2.0
            sd = float(data["ERP_and_SRI"]["sd"])
            return mean, sd

        # Continuing current regimen: expect little additional structured change
        if act == "continue":
            return 0.0, 1.0

        # Augmentation and other second/third-line strategies: modest additional gain
        if act in {
            "augment_antipsychotic",
            "switch_different_augmenting_antipsychotic",
            "augment_clomipramine",
            "augment_glutamate_modulator",
        }:
            return -3.0, 2.0  # in Y-BOCS points

        # Fallback: very small generic effect
        return -1.0, 1.0

    mean_delta, sd_delta = _effect_distribution(action)

    # Scale by patient responsiveness; negative delta = improvement.
    expected_delta = mean_delta * responsiveness
    sampled_delta = float(np.random.normal(expected_delta, sd_delta))

    new_true_ybocs = max(0.0, true_ybocs + sampled_delta)
    next_state["true_ybocs"] = float(new_true_ybocs)

    return next_state


def POMDP_OBSERVATION(state, action):
    baseline = float(state["profile"]["ybocs_baseline"])
    true_ybocs = float(state.get("true_ybocs", baseline))

    # If already in dropout state, stay dropped out and do not observe Y-BOCS.
    if state.get("dropped_out", False):
        return {"ybocs_score": None, "dropout": True}

    # Dropout probability increases with residual severity.
    if baseline > 0:
        severity_ratio = true_ybocs / baseline
    else:
        severity_ratio = 0.0
    dropout_prob = min(0.05 + 0.3 * severity_ratio, 0.5)
    dropped_now = bool(np.random.rand() < dropout_prob)

    if dropped_now:
        state["dropped_out"] = True  # make dropout absorbing in the true state
        return {"ybocs_score": None, "dropout": True}

    # Measurement noise around the true (latent) Y-BOCS.
    measured = float(np.random.normal(true_ybocs, 1.0))
    measured = float(np.clip(measured, 0.0, 40.0))

    return {"ybocs_score": measured, "dropout": False}


def POMDP_REWARD(state, action, next_state, observation):
    # Large penalty for dropout regardless of state details.
    if observation.get("dropout", False):
        return -20.0

    # If we have explicit latent Y-BOCS in the states, reward improvement plus
    # a remission bonus. Otherwise, fall back to baseline-vs-observation reward.
    has_latent = ("true_ybocs" in state) and ("true_ybocs" in next_state)

    if has_latent:
        baseline = float(state["profile"]["ybocs_baseline"])
        prev_y = float(state.get("true_ybocs", baseline))
        new_y = float(next_state.get("true_ybocs", prev_y))
        improvement = max(0.0, prev_y - new_y)

        reward = improvement

        # Bonus for reaching remission.
        remission_threshold = float(APA_ALGORITHM["response_thresholds"]["remission_ybocs"])
        if new_y <= remission_threshold:
            reward += 10.0

        # Small penalty for higher-burden augmentation strategies.
        if action in {
            "augment_antipsychotic",
            "switch_different_augmenting_antipsychotic",
            "augment_clomipramine",
            "augment_glutamate_modulator",
        }:
            reward -= 2.0

        return float(reward)

    # --- Fallback: simple reward based on baseline vs observed Y-BOCS. ---
    if observation.get("dropout"):
        return -10.0
    baseline = float(state["profile"]["ybocs_baseline"])
    current = float(observation.get("ybocs_score", baseline))
    reduction = baseline - current
    return float(reduction)


def POMDP_UPDATE_BELIEF(belief, action, observation):
    # Shallow copy, then deep-copy mutable substructures we modify.
    new_belief = dict(belief)
    new_belief["history"] = list(belief.get("history", []))
    new_belief["action_stats"] = dict(belief.get("action_stats", {}))

    last_y = belief.get("last_ybocs")
    dropout = bool(observation.get("dropout", False))
    y_obs = observation.get("ybocs_score", last_y)

    # Update running per-action improvement estimate when we see a non-dropout score.
    if (not dropout) and (y_obs is not None) and (last_y is not None):
        delta = float(last_y - y_obs)  # positive = improvement in Y-BOCS points
        stats = new_belief["action_stats"].get(action, {"n": 0, "mean_delta": 0.0})
        n_old = int(stats["n"])
        mean_old = float(stats["mean_delta"])
        n_new = n_old + 1
        mean_new = mean_old + (delta - mean_old) / float(n_new)
        new_belief["action_stats"][action] = {"n": n_new, "mean_delta": mean_new}
        new_belief["last_ybocs"] = float(y_obs)

    # Record history and bookkeeping.
    new_belief["history"].append({"action": action, "observation": observation})
    new_belief["step"] = int(belief.get("step", 0) + 1)
    new_belief["dropped_out"] = bool(dropout or belief.get("dropped_out", False))

    return new_belief


def POMDP_SOLVE(belief, valid_actions):
    # Normalize valid_actions to a list of strings.
    if isinstance(valid_actions, dict):
        if "options" in valid_actions:
            actions = list(valid_actions["options"])
        else:
            actions = list(valid_actions.keys())
    else:
        actions = list(valid_actions)

    if not actions:
        raise ValueError("POMDP_SOLVE called with no valid actions")

    stats = belief.get("action_stats", {})
    total_tries = sum(info["n"] for info in stats.values()) + 1e-6

    # Local helper to get a literature-based prior on expected improvement
    # (in positive Y-BOCS points) for each abstract action.
    def _prior_expected_improvement(act):
        data = CLINICIAN["ocd_treatment_ybocs"]

        if act in {"SSRI", "switch_different_SSRI", "switch_different_SRI", "switch_clomipramine"}:
            ssri_keys = [k for k, v in data.items() if "ci_95" in v]
            means = [data[k]["mean_change"] for k in ssri_keys]
            mean = float(sum(means) / len(means))
            return -mean  # negative mean_change => positive improvement

        if act in {"ERP_and_SRI", "SSRI_plus_CBT", "add_CBT_if_not_provided"}:
            mean = float(data["ERP_and_SRI"]["mean_change"])
            return -mean

        if act == "CBT":
            mean = float(data["ERP_and_SRI"]["mean_change"]) / 2.0
            return -mean

        if act == "continue":
            return 0.5  # tiny expected gain from natural course/maintenance

        if act in {
            "augment_antipsychotic",
            "switch_different_augmenting_antipsychotic",
            "augment_clomipramine",
            "augment_glutamate_modulator",
        }:
            return 3.0  # modest expected additional improvement

        # Fallback for any unrecognized action label.
        return 1.0

    # Use an upper-confidence-bound style rule (chapter 15) over actions.
    c = 1.0  # exploration constant

    best_action = None
    best_score = -float("inf")

    for a in actions:
        a_stats = stats.get(a, {"n": 0, "mean_delta": 0.0})
        n_a = int(a_stats["n"])
        empirical = float(a_stats["mean_delta"])
        prior = float(_prior_expected_improvement(a))

        if n_a == 0:
            mean_est = prior
        else:
            # Simple shrinkage toward prior: average empirical and prior.
            mean_est = 0.5 * prior + 0.5 * empirical

        exploration_bonus = c * math.sqrt(math.log(total_tries + 1.0) / (n_a + 1.0))
        score = mean_est + exploration_bonus

        if score > best_score:
            best_score = score
            best_action = a

    return best_action

# ============================================================
# SIMULATE ONE PATIENT
# ============================================================

def simulate_one_patient(use_pomdp=True):
    patient = {}
    patient["age"] = sample_age()
    patient["gender"] = "female" if random.random() < CLINICIAN["female_probability"] else "male"
    patient["subtype"] = weighted_choice(CLINICIAN["subtype_probabilities"])
    
    baseline = float(np.random.normal(
        CLINICIAN["baseline_ybocs"]["mean"],
        CLINICIAN["baseline_ybocs"]["sd"]
    ))
    baseline = float(np.clip(baseline, 0.0, 40.0))
    patient["ybocs_baseline"] = baseline
    patient["severity"] = classify_severity(baseline)
    
    if use_pomdp:
        belief = POMDP_INITIALIZE_BELIEF(patient)
        true_state = POMDP_GET_TRUE_STATE(patient)
    else:
        belief = None
        true_state = {"profile": patient}
    
    trajectory = []
    treatment_history = []
    current_ybocs = baseline
    
    for step in range(3):
        if step == 0:
            last_response = None
        else:
            response = assess_response(
                trajectory[-1]["ybocs_before"],
                trajectory[-1]["ybocs_after"]
            )
            last_response = response["category"]

        valid_actions = get_valid_actions(treatment_history, last_response)
        
        if last_response == "adequate":
            break
        
        if use_pomdp:
            action = POMDP_SOLVE(belief, valid_actions)
        else:
            # NEW - Use guideline-based decision making
            action = guideline_based_decision(
                valid_actions, 
                treatment_history, 
                patient, 
                last_response,
                trajectory
            )

        if use_pomdp:
            next_state = POMDP_TRANSITION(true_state, action)
            obs = POMDP_OBSERVATION(next_state, action)
            reward = POMDP_REWARD(true_state, action, next_state, obs)
            belief = POMDP_UPDATE_BELIEF(belief, action, obs)
            true_state = next_state
        else:
            # Standard care also needs state transitions
            next_state = POMDP_TRANSITION(true_state, action)
            obs = POMDP_OBSERVATION(next_state, action)
            reward = POMDP_REWARD(true_state, action, next_state, obs)
            true_state = next_state


        

        
        trajectory.append({
            "step": step,
            "action": action,
            "ybocs_before": current_ybocs,
            "ybocs_after": obs.get("ybocs_score"),
            "dropout": obs.get("dropout", False),
            "reward": reward
        })
        
        treatment_history.append(action)
        current_ybocs = obs.get("ybocs_score", current_ybocs)
        
        if obs.get("dropout", False):
            break
    
    patient["num_treatment_steps"] = len(trajectory)
    patient["treatments_tried"] = treatment_history
    patient["final_ybocs"] = trajectory[-1]["ybocs_after"] if trajectory else None
    patient["total_reward"] = sum(t["reward"] for t in trajectory)
    
    if patient["final_ybocs"] is not None:
        final = assess_response(baseline, patient["final_ybocs"])
        patient["final_response"] = final["category"]
        patient["pct_improvement"] = final["pct_change"] * 100
        patient["achieved_remission"] = final["remission"]
    else:
        patient["final_response"] = "dropout"
        patient["pct_improvement"] = None
        patient["achieved_remission"] = False
    
    return patient

# ============================================================
# SIMULATE DATASET
# ============================================================

def simulate_dataset(n_patients=500, use_pomdp=True):
    mode = "POMDP" if use_pomdp else "STANDARD CARE"
    print(f"Simulating {n_patients} patients with {mode}...")
    
    patients = []
    for i in range(n_patients):
        if (i + 1) % 50 == 0:
            print(f"  Simulated {i + 1}/{n_patients} patients...")
        patients.append(simulate_one_patient(use_pomdp=use_pomdp))
    
    df = pd.DataFrame(patients)
    return df

# ============================================================
# MVP SUCCESS METRICS
# ============================================================

def calculate_success_metrics(df):
    """Calculate 5 key success metrics."""
    return {
        'remission_rate': (df['achieved_remission'].sum() / len(df)) * 100,
        'response_rate': ((df['final_response'] == 'adequate').sum() / len(df)) * 100,
        'any_benefit_rate': ((df['final_response'].isin(['adequate', 'moderate'])).sum() / len(df)) * 100,
        'mean_improvement': df['pct_improvement'].dropna().mean(),
        'mean_steps': df['num_treatment_steps'].mean()
    }

def compare_approaches(df_standard, df_pomdp):
    """Print comparison of key metrics."""
    std = calculate_success_metrics(df_standard)
    pomdp = calculate_success_metrics(df_pomdp)
    
    print("\n" + "=" * 70)
    print("SUCCESS METRICS COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<30} {'Standard':>12} {'POMDP':>12} {'Δ':>12}")
    print("-" * 70)
    print(f"{'Remission (Y-BOCS≤12)':<30} {std['remission_rate']:>11.1f}% {pomdp['remission_rate']:>11.1f}% {pomdp['remission_rate']-std['remission_rate']:>11.1f}")
    print(f"{'Response (≥35% improve)':<30} {std['response_rate']:>11.1f}% {pomdp['response_rate']:>11.1f}% {pomdp['response_rate']-std['response_rate']:>11.1f}")
    print(f"{'Any Benefit (≥25%)':<30} {std['any_benefit_rate']:>11.1f}% {pomdp['any_benefit_rate']:>11.1f}% {pomdp['any_benefit_rate']-std['any_benefit_rate']:>11.1f}")
    print(f"{'Mean % Improvement':<30} {std['mean_improvement']:>11.1f}% {pomdp['mean_improvement']:>11.1f}% {pomdp['mean_improvement']-std['mean_improvement']:>11.1f}")
    print(f"{'Mean Treatment Steps':<30} {std['mean_steps']:>11.2f} {pomdp['mean_steps']:>11.2f} {pomdp['mean_steps']-std['mean_steps']:>11.2f}")
    
    # --- Approximate statistical significance for difference in mean % improvement ---
    imp_std = df_standard['pct_improvement'].dropna().values
    imp_pomdp = df_pomdp['pct_improvement'].dropna().values

    if len(imp_std) > 1 and len(imp_pomdp) > 1:
        n1, n2 = len(imp_std), len(imp_pomdp)
        mean_diff = imp_pomdp.mean() - imp_std.mean()

        # Sample variances with Bessel's correction
        var1 = imp_std.var(ddof=1)
        var2 = imp_pomdp.var(ddof=1)
        se_diff = np.sqrt(var1 / n1 + var2 / n2)

        if se_diff > 0:
            z = mean_diff / se_diff

            # Standard normal CDF via error function
            def norm_cdf(x):
                return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))

            p_two_sided = 2.0 * (1.0 - norm_cdf(abs(z)))
        else:
            z = float('nan')
            p_two_sided = float('nan')

        print("\nApproximate significance test for mean % improvement (POMDP - Standard):")
        print(f"  Mean difference: {mean_diff:.2f} percentage points")
        print(f"  z ≈ {z:.2f}, two-sided p ≈ {p_two_sided:.4f}")
        if p_two_sided < 0.05:
            print("  ➜ Statistically significant at α = 0.05.")
        else:
            print("  ➜ Not statistically significant at α = 0.05.")
    else:
        print("\nNot enough data to compute a significance test.")
    
    # Quick verdict based on clinical metrics
    wins = sum([
        pomdp['remission_rate'] > std['remission_rate'],
        pomdp['response_rate'] > std['response_rate'],
        pomdp['any_benefit_rate'] > std['any_benefit_rate'],
        pomdp['mean_improvement'] > std['mean_improvement'],
        pomdp['mean_steps'] < std['mean_steps']
    ])
    
    print("\n" + "=" * 70)
    print(f"POMDP wins on {wins}/5 metrics")
    if wins >= 4:
        print("STRONG EVIDENCE of clinical value")
    elif wins >= 3:
        print("MODERATE EVIDENCE of clinical value")
    else:
        print("WEAK EVIDENCE of clinical value")
    print("=" * 70)


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

def create_visualizations(df_standard, df_pomdp):
    """Create comprehensive visualizations comparing POMDP vs Standard Care."""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # --------------------------------------------------------
    # 1. Key Metrics Comparison (Bar Chart)
    # --------------------------------------------------------
    ax1 = plt.subplot(2, 3, 1)
    metrics_std = calculate_success_metrics(df_standard)
    metrics_pomdp = calculate_success_metrics(df_pomdp)
    
    metrics_names = ['Remission\nRate', 'Response\nRate', 'Any Benefit\nRate']
    std_values = [metrics_std['remission_rate'], metrics_std['response_rate'], metrics_std['any_benefit_rate']]
    pomdp_values = [metrics_pomdp['remission_rate'], metrics_pomdp['response_rate'], metrics_pomdp['any_benefit_rate']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, std_values, width, label='Standard Care', alpha=0.8, color='#e74c3c')
    bars2 = ax1.bar(x + width/2, pomdp_values, width, label='POMDP', alpha=0.8, color='#3498db')
    
    ax1.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Treatment Success Rates', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # --------------------------------------------------------
    # 2. Y-BOCS Improvement Distribution (Violin Plot)
    # --------------------------------------------------------
    ax2 = plt.subplot(2, 3, 2)
    
    # Prepare data for violin plot
    improvement_data = []
    labels = []
    
    for imp in df_standard['pct_improvement'].dropna():
        improvement_data.append(imp)
        labels.append('Standard Care')
    
    for imp in df_pomdp['pct_improvement'].dropna():
        improvement_data.append(imp)
        labels.append('POMDP')
    
    plot_df = pd.DataFrame({'Improvement (%)': improvement_data, 'Method': labels})
    
    parts = ax2.violinplot([df_standard['pct_improvement'].dropna(), 
                            df_pomdp['pct_improvement'].dropna()],
                          positions=[0, 1], widths=0.7, showmeans=True, showmedians=True)
    
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Standard Care', 'POMDP'])
    ax2.set_ylabel('% Improvement', fontsize=11, fontweight='bold')
    ax2.set_title('Distribution of Y-BOCS Improvement', fontsize=13, fontweight='bold')
    ax2.axhline(y=35, color='green', linestyle='--', alpha=0.5, label='Adequate Response (35%)')
    ax2.axhline(y=25, color='orange', linestyle='--', alpha=0.5, label='Moderate Response (25%)')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # --------------------------------------------------------
    # 3. Treatment Response Categories (Stacked Bar)
    # --------------------------------------------------------
    ax3 = plt.subplot(2, 3, 3)
    
    response_counts_std = df_standard['final_response'].value_counts()
    response_counts_pomdp = df_pomdp['final_response'].value_counts()
    
    categories = ['adequate', 'moderate', 'poor', 'dropout']
    std_pcts = [(response_counts_std.get(cat, 0) / len(df_standard)) * 100 for cat in categories]
    pomdp_pcts = [(response_counts_pomdp.get(cat, 0) / len(df_pomdp)) * 100 for cat in categories]
    
    x_pos = [0, 1]
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']
    
    bottom_std = 0
    bottom_pomdp = 0
    
    for i, cat in enumerate(categories):
        ax3.bar(0, std_pcts[i], bottom=bottom_std, color=colors[i], alpha=0.8, width=0.6)
        ax3.bar(1, pomdp_pcts[i], bottom=bottom_pomdp, color=colors[i], alpha=0.8, width=0.6, 
                label=cat.capitalize())
        
        # Add percentage labels
        if std_pcts[i] > 5:
            ax3.text(0, bottom_std + std_pcts[i]/2, f'{std_pcts[i]:.1f}%', 
                    ha='center', va='center', fontsize=9, fontweight='bold')
        if pomdp_pcts[i] > 5:
            ax3.text(1, bottom_pomdp + pomdp_pcts[i]/2, f'{pomdp_pcts[i]:.1f}%', 
                    ha='center', va='center', fontsize=9, fontweight='bold')
        
        bottom_std += std_pcts[i]
        bottom_pomdp += pomdp_pcts[i]
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['Standard Care', 'POMDP'])
    ax3.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Treatment Response Distribution', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_ylim(0, 100)
    
    # --------------------------------------------------------
    # 4. Number of Treatment Steps (Box Plot)
    # --------------------------------------------------------
    ax4 = plt.subplot(2, 3, 4)
    
    bp = ax4.boxplot([df_standard['num_treatment_steps'], df_pomdp['num_treatment_steps']],
                     labels=['Standard Care', 'POMDP'],
                     patch_artist=True,
                     medianprops=dict(color='red', linewidth=2),
                     boxprops=dict(facecolor='lightblue', alpha=0.7))
    
    ax4.set_ylabel('Number of Treatment Steps', fontsize=11, fontweight='bold')
    ax4.set_title('Treatment Duration Comparison', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add mean markers
    means = [df_standard['num_treatment_steps'].mean(), df_pomdp['num_treatment_steps'].mean()]
    ax4.plot([1, 2], means, 'D', color='green', markersize=8, label='Mean', zorder=3)
    ax4.legend(fontsize=9)
    
    # --------------------------------------------------------
    # 5. Baseline Y-BOCS vs Final Y-BOCS (Scatter)
    # --------------------------------------------------------
    ax5 = plt.subplot(2, 3, 5)
    
    # Standard Care
    std_valid = df_standard[df_standard['final_ybocs'].notna()]
    ax5.scatter(std_valid['ybocs_baseline'], std_valid['final_ybocs'], 
               alpha=0.5, s=30, c='#e74c3c', label='Standard Care', edgecolors='black', linewidth=0.5)
    
    # POMDP
    pomdp_valid = df_pomdp[df_pomdp['final_ybocs'].notna()]
    ax5.scatter(pomdp_valid['ybocs_baseline'], pomdp_valid['final_ybocs'], 
               alpha=0.5, s=30, c='#3498db', label='POMDP', edgecolors='black', linewidth=0.5)
    
    # Add diagonal line (no improvement)
    max_val = max(df_standard['ybocs_baseline'].max(), df_pomdp['ybocs_baseline'].max())
    ax5.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='No Change')
    
    # Add remission line
    ax5.axhline(y=12, color='green', linestyle='--', alpha=0.5, label='Remission (Y-BOCS≤12)')
    
    ax5.set_xlabel('Baseline Y-BOCS', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Final Y-BOCS', fontsize=11, fontweight='bold')
    ax5.set_title('Baseline vs Final Y-BOCS Scores', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)
    
    # --------------------------------------------------------
    # 6. Treatment Effectiveness by Subtype (Heatmap-style)
    # --------------------------------------------------------
    ax6 = plt.subplot(2, 3, 6)
    
    # Calculate mean improvement by subtype for both methods
    subtypes = []
    std_improvements = []
    pomdp_improvements = []
    
    for subtype in df_standard['subtype'].unique():
        subtype_name = subtype[0] if isinstance(subtype, tuple) else subtype
        
        std_mask = df_standard['subtype'].apply(lambda x: x[0] if isinstance(x, tuple) else x) == subtype_name
        pomdp_mask = df_pomdp['subtype'].apply(lambda x: x[0] if isinstance(x, tuple) else x) == subtype_name
        
        std_imp = df_standard[std_mask]['pct_improvement'].dropna().mean()
        pomdp_imp = df_pomdp[pomdp_mask]['pct_improvement'].dropna().mean()
        
        if not np.isnan(std_imp) and not np.isnan(pomdp_imp):
            subtypes.append(subtype_name.replace('_', ' ').title()[:15])
            std_improvements.append(std_imp)
            pomdp_improvements.append(pomdp_imp)
    
    x = np.arange(len(subtypes))
    width = 0.35
    
    bars1 = ax6.barh(x - width/2, std_improvements, width, label='Standard Care', alpha=0.8, color='#e74c3c')
    bars2 = ax6.barh(x + width/2, pomdp_improvements, width, label='POMDP', alpha=0.8, color='#3498db')
    
    ax6.set_yticks(x)
    ax6.set_yticklabels(subtypes, fontsize=9)
    ax6.set_xlabel('Mean % Improvement', fontsize=11, fontweight='bold')
    ax6.set_title('Effectiveness by OCD Subtype', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            width_val = bar.get_width()
            ax6.text(width_val, bar.get_y() + bar.get_height()/2.,
                    f'{width_val:.1f}%', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('ocd_simulation_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved visualization: ocd_simulation_results.png")
    
    # --------------------------------------------------------
    # Additional Figure: Detailed Treatment Pathways
    # --------------------------------------------------------
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Treatment actions frequency
    ax_left = axes[0]
    
    all_treatments_std = []
    for treatments in df_standard['treatments_tried']:
        all_treatments_std.extend(treatments)
    
    all_treatments_pomdp = []
    for treatments in df_pomdp['treatments_tried']:
        all_treatments_pomdp.extend(treatments)
    
    from collections import Counter
    counter_std = Counter(all_treatments_std)
    counter_pomdp = Counter(all_treatments_pomdp)
    
    all_actions = set(counter_std.keys()) | set(counter_pomdp.keys())
    all_actions = sorted(all_actions)
    
    std_counts = [counter_std.get(action, 0) for action in all_actions]
    pomdp_counts = [counter_pomdp.get(action, 0) for action in all_actions]
    
    x = np.arange(len(all_actions))
    width = 0.35
    
    ax_left.bar(x - width/2, std_counts, width, label='Standard Care', alpha=0.8, color='#e74c3c')
    ax_left.bar(x + width/2, pomdp_counts, width, label='POMDP', alpha=0.8, color='#3498db')
    
    ax_left.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax_left.set_title('Treatment Actions Used', fontsize=13, fontweight='bold')
    ax_left.set_xticks(x)
    ax_left.set_xticklabels([a.replace('_', '\n') for a in all_actions], rotation=45, ha='right', fontsize=8)
    ax_left.legend(fontsize=10)
    ax_left.grid(axis='y', alpha=0.3)
    
    # Remission rate by severity
    ax_right = axes[1]
    
    severities = ['subclinical', 'mild', 'moderate', 'severe', 'extreme']
    std_remission_by_sev = []
    pomdp_remission_by_sev = []
    
    for sev in severities:
        std_sev = df_standard[df_standard['severity'] == sev]
        pomdp_sev = df_pomdp[df_pomdp['severity'] == sev]
        
        if len(std_sev) > 0:
            std_remission_by_sev.append((std_sev['achieved_remission'].sum() / len(std_sev)) * 100)
        else:
            std_remission_by_sev.append(0)
        
        if len(pomdp_sev) > 0:
            pomdp_remission_by_sev.append((pomdp_sev['achieved_remission'].sum() / len(pomdp_sev)) * 100)
        else:
            pomdp_remission_by_sev.append(0)
    
    x = np.arange(len(severities))
    width = 0.35
    
    ax_right.bar(x - width/2, std_remission_by_sev, width, label='Standard Care', alpha=0.8, color='#e74c3c')
    ax_right.bar(x + width/2, pomdp_remission_by_sev, width, label='POMDP', alpha=0.8, color='#3498db')
    
    ax_right.set_ylabel('Remission Rate (%)', fontsize=11, fontweight='bold')
    ax_right.set_title('Remission Rate by Baseline Severity', fontsize=13, fontweight='bold')
    ax_right.set_xticks(x)
    ax_right.set_xticklabels([s.capitalize() for s in severities], rotation=45, ha='right')
    ax_right.legend(fontsize=10)
    ax_right.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ocd_treatment_pathways.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization: ocd_treatment_pathways.png")
    
    plt.show()
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    
    print("=" * 70)
    print("OCD SIMULATOR: POMDP vs STANDARD CARE")
    print("=" * 70)
    
    df_standard = simulate_dataset(n_patients=500, use_pomdp=False)
    df_pomdp = simulate_dataset(n_patients=500, use_pomdp=True)
    
    compare_approaches(df_standard, df_pomdp)
    
    df_standard.to_csv('synthetic_ocd_standard.csv', index=False)
    df_pomdp.to_csv('synthetic_ocd_pomdp.csv', index=False)
    print("\n✓ Saved: synthetic_ocd_standard.csv, synthetic_ocd_pomdp.csv")
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    create_visualizations(df_standard, df_pomdp)


