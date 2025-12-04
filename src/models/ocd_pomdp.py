"""
OCD Treatment POMDP Formulation
Defines state, action, observation spaces and transition/observation/reward models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Tuple, Any
import pickle
from algorithms.pomdp import POMDP
from algorithms.bayesian_network import Variable, sub2ind


class OCDTreatmentState:
    """State representation for OCD treatment POMDP"""
    def __init__(self, ybocs_severity, depression, anxiety, weeks_on_tx, current_tx, adherence_level):
        self.ybocs_severity = ybocs_severity  # 1-5 (subclinical to extreme)
        self.depression = depression  # 1-2 (no, yes)
        self.anxiety = anxiety  # 1-2 (no, yes)
        self.weeks_on_tx = weeks_on_tx  # 0, 4, 8, 12
        self.current_tx = current_tx  # 0 (none), 1-7 (treatment types)
        self.adherence_level = adherence_level  # 1-3 (high, medium, low)
    
    def __eq__(self, other):
        return (self.ybocs_severity == other.ybocs_severity and
                self.depression == other.depression and
                self.anxiety == other.anxiety and
                self.weeks_on_tx == other.weeks_on_tx and
                self.current_tx == other.current_tx and
                self.adherence_level == other.adherence_level)
    
    def __hash__(self):
        return hash((self.ybocs_severity, self.depression, self.anxiety,
                    self.weeks_on_tx, self.current_tx, self.adherence_level))
    
    def __repr__(self):
        return f"State(ybocs={self.ybocs_severity}, dep={self.depression}, anx={self.anxiety}, " + \
               f"weeks={self.weeks_on_tx}, tx={self.current_tx}, adh={self.adherence_level})"


class OCDTreatmentAction:
    """Action representation"""
    def __init__(self, action_type, treatment_id=None):
        self.action_type = action_type  # 'continue', 'start', 'switch', 'stop'
        self.treatment_id = treatment_id  # 1-7 for start/switch
    
    def __eq__(self, other):
        return self.action_type == other.action_type and self.treatment_id == other.treatment_id
    
    def __hash__(self):
        return hash((self.action_type, self.treatment_id))
    
    def __repr__(self):
        if self.treatment_id:
            tx_names = ['CBT', 'SSRI', 'TMS', 'Ketamine', 'CBT+SSRI', 'CBT+TMS', 'SSRI+TMS']
            return f"Action({self.action_type}, {tx_names[self.treatment_id-1]})"
        return f"Action({self.action_type})"


class OCDTreatmentObservation:
    """Observation representation"""
    def __init__(self, observed_ybocs, reported_side_effects, attendance):
        self.observed_ybocs = observed_ybocs  # 1-5 (with measurement noise)
        self.reported_side_effects = reported_side_effects  # 1-4 (none to severe)
        self.attendance = attendance  # 1-2 (attended, missed)
    
    def __eq__(self, other):
        return (self.observed_ybocs == other.observed_ybocs and
                self.reported_side_effects == other.reported_side_effects and
                self.attendance == other.attendance)
    
    def __hash__(self):
        return hash((self.observed_ybocs, self.reported_side_effects, self.attendance))
    
    def __repr__(self):
        return f"Obs(ybocs={self.observed_ybocs}, se={self.reported_side_effects}, att={self.attendance})"


def create_state_space():
    """Create discretized state space"""
    states = []
    for ybocs in range(1, 6):  # 5 severity levels
        for dep in range(1, 3):  # 2 levels
            for anx in range(1, 3):  # 2 levels
                for weeks in [0, 4, 8, 12]:  # 4 time points
                    for tx in range(0, 8):  # 0=none, 1-7=treatments
                        for adh in range(1, 4):  # 3 levels
                            states.append(OCDTreatmentState(ybocs, dep, anx, weeks, tx, adh))
    return states


def create_action_space():
    """Create action space"""
    actions = [OCDTreatmentAction('continue')]
    actions.append(OCDTreatmentAction('stop'))
    
    # Start/switch to each treatment
    for tx_id in range(1, 8):
        actions.append(OCDTreatmentAction('start', tx_id))
    
    return actions


def create_observation_space():
    """Create observation space"""
    observations = []
    for ybocs in range(1, 6):
        for se in range(1, 5):
            for att in range(1, 3):
                observations.append(OCDTreatmentObservation(ybocs, se, att))
    return observations


class OCDTreatmentPOMDP:
    """OCD Treatment POMDP with learned dynamics from Bayesian network"""
    
    def __init__(self, bn_cpds=None, bn_graph=None, bn_variables=None):
        """
        Args:
            bn_cpds: Learned CPDs from Bayesian network
            bn_graph: Learned graph structure
            bn_variables: Variable definitions
        """
        self.bn_cpds = bn_cpds
        self.bn_graph = bn_graph
        self.bn_variables = bn_variables
        
        if bn_variables:
            self.var_names = [v.name for v in bn_variables]
        
        # Treatment efficacy parameters (from literature/data)
        self.treatment_efficacy = {
            0: 0.0,  # No treatment
            1: 0.65,  # CBT
            2: 0.50,  # SSRI
            3: 0.35,  # TMS
            4: 0.50,  # Ketamine
            5: 0.75,  # CBT+SSRI
            6: 0.55,  # CBT+TMS
            7: 0.50   # SSRI+TMS
        }
        
        # Treatment costs (relative)
        self.treatment_costs = {
            0: 0,
            1: 50,   # CBT (session cost)
            2: 20,   # SSRI (medication cost)
            3: 200,  # TMS (expensive)
            4: 300,  # Ketamine (very expensive)
            5: 60,   # CBT+SSRI
            6: 220,  # CBT+TMS
            7: 210   # SSRI+TMS
        }
    
    def transition(self, s: OCDTreatmentState, a: OCDTreatmentAction, s_prime: OCDTreatmentState) -> float:
        """
        Transition probability T(s, a, s').
        Models treatment effects and natural progression.
        """
        # Update treatment status
        if a.action_type == 'stop':
            expected_tx = 0
        elif a.action_type in ['start', 'switch']:
            expected_tx = a.treatment_id
        else:  # continue
            expected_tx = s.current_tx
        
        if s_prime.current_tx != expected_tx:
            return 0.0
        
        # Update weeks on treatment
        if a.action_type in ['start', 'switch']:
            expected_weeks = 0
        else:
            expected_weeks = min(s.weeks_on_tx + 4, 12)
        
        if s_prime.weeks_on_tx != expected_weeks:
            return 0.0
        
        # Comorbidities are relatively stable
        if s_prime.depression != s.depression:
            dep_change_prob = 0.1
        else:
            dep_change_prob = 0.9
        
        if s_prime.anxiety != s.anxiety:
            anx_change_prob = 0.1
        else:
            anx_change_prob = 0.9
        
        # Adherence changes based on side effects and treatment burden
        adh_prob = 0.8 if s_prime.adherence_level == s.adherence_level else 0.1
        
        # Y-BOCS severity changes based on treatment
        if s_prime.current_tx == 0:
            # No treatment: likely stays same or worsens
            ybocs_probs = np.zeros(5)
            ybocs_probs[s.ybocs_severity - 1] = 0.6  # Stay same
            if s.ybocs_severity < 5:
                ybocs_probs[s.ybocs_severity] = 0.3  # Worsen
            if s.ybocs_severity > 1:
                ybocs_probs[s.ybocs_severity - 2] = 0.1  # Improve slightly
            ybocs_probs = ybocs_probs / ybocs_probs.sum()
            ybocs_prob = ybocs_probs[s_prime.ybocs_severity - 1]
        else:
            # On treatment: probability of improvement
            efficacy = self.treatment_efficacy[s_prime.current_tx]
            
            # Adjust for adherence
            if s_prime.adherence_level == 1:  # High
                efficacy *= 1.0
            elif s_prime.adherence_level == 2:  # Medium
                efficacy *= 0.7
            else:  # Low
                efficacy *= 0.4
            
            # Adjust for time on treatment
            time_factor = min(s_prime.weeks_on_tx / 12.0, 1.0)
            efficacy *= (0.3 + 0.7 * time_factor)  # Gradual improvement
            
            # Compute Y-BOCS transition probabilities
            ybocs_probs = np.zeros(5)
            current_severity = s.ybocs_severity
            
            # Response distribution
            if np.random.random() < efficacy:
                # Responding: shift toward lower severity
                target = max(1, current_severity - 2)
                for i in range(1, 6):
                    if i <= target:
                        ybocs_probs[i-1] = 0.3 / target
                    elif i == target + 1:
                        ybocs_probs[i-1] = 0.5
                    else:
                        ybocs_probs[i-1] = 0.2 / (5 - target - 1)
            else:
                # Not responding: stays similar
                ybocs_probs[current_severity - 1] = 0.7
                if current_severity > 1:
                    ybocs_probs[current_severity - 2] = 0.15
                if current_severity < 5:
                    ybocs_probs[current_severity] = 0.15
            
            ybocs_probs = ybocs_probs / ybocs_probs.sum()
            ybocs_prob = ybocs_probs[s_prime.ybocs_severity - 1]
        
        # Total probability
        prob = ybocs_prob * dep_change_prob * anx_change_prob * adh_prob
        return prob
    
    def observation_func(self, a: OCDTreatmentAction, s_prime: OCDTreatmentState, 
                        o: OCDTreatmentObservation) -> float:
        """
        Observation probability O(a, s', o).
        Models noisy measurements and reporting.
        """
        # Y-BOCS observation with measurement noise (test-retest reliability ~ 0.85)
        ybocs_noise_std = 0.5  # In terms of discrete levels
        ybocs_obs_probs = np.zeros(5)
        for i in range(1, 6):
            diff = abs(i - s_prime.ybocs_severity)
            ybocs_obs_probs[i-1] = np.exp(-diff**2 / (2 * ybocs_noise_std**2))
        ybocs_obs_probs = ybocs_obs_probs / ybocs_obs_probs.sum()
        ybocs_prob = ybocs_obs_probs[o.observed_ybocs - 1]
        
        # Side effects depend on treatment and adherence
        # Simplified: higher prob of reporting SE if actually experiencing them
        se_prob = 0.25  # Uniform for now (could be improved with BN)
        
        # Attendance depends on adherence level
        if s_prime.adherence_level == 1:  # High adherence
            att_prob = 0.95 if o.attendance == 1 else 0.05
        elif s_prime.adherence_level == 2:  # Medium
            att_prob = 0.80 if o.attendance == 1 else 0.20
        else:  # Low
            att_prob = 0.60 if o.attendance == 1 else 0.40
        
        return ybocs_prob * se_prob * att_prob
    
    def reward(self, s: OCDTreatmentState, a: OCDTreatmentAction) -> float:
        """
        Reward function R(s, a).
        Encourages symptom reduction while minimizing cost and burden.
        """
        # Symptom severity cost (higher severity = more negative)
        severity_cost = -10 * s.ybocs_severity
        
        # Treatment cost
        tx_cost = -self.treatment_costs[s.current_tx] / 100.0  # Normalize
        
        # Low adherence penalty
        adherence_bonus = 0 if s.adherence_level == 1 else -2 * (s.adherence_level - 1)
        
        # Switching penalty (to encourage stability)
        switch_penalty = 0
        if a.action_type in ['start', 'switch']:
            switch_penalty = -5
        
        # Remission bonus
        remission_bonus = 20 if s.ybocs_severity == 1 else 0
        
        total_reward = severity_cost + tx_cost + adherence_bonus + switch_penalty + remission_bonus
        return total_reward
    
    def create_pomdp(self, gamma=0.95):
        """Create POMDP object"""
        states = create_state_space()
        actions = create_action_space()
        observations = create_observation_space()
        
        print(f"State space size: {len(states)}")
        print(f"Action space size: {len(actions)}")
        print(f"Observation space size: {len(observations)}")
        
        pomdp = POMDP(
            gamma=gamma,
            S=states,
            A=actions,
            O=observations,
            T=self.transition,
            R=self.reward,
            O_func=self.observation_func
        )
        
        return pomdp


def load_bn_and_create_pomdp(bn_model_path, gamma=0.95):
    """Load trained BN and create POMDP"""
    print("Loading Bayesian network model...")
    with open(bn_model_path, 'rb') as f:
        bn_data = pickle.load(f)
    
    pomdp_model = OCDTreatmentPOMDP(
        bn_cpds=bn_data['cpds'],
        bn_graph=bn_data['graph'],
        bn_variables=bn_data['variables']
    )
    
    print("Creating POMDP...")
    pomdp = pomdp_model.create_pomdp(gamma=gamma)
    
    return pomdp, pomdp_model


if __name__ == "__main__":
    print("="*60)
    print("OCD Treatment POMDP Formulation")
    print("="*60)
    
    # Create simplified POMDP (without BN for now)
    print("\nCreating POMDP without learned BN...")
    pomdp_model = OCDTreatmentPOMDP()
    pomdp = pomdp_model.create_pomdp()
    
    print("\n✓ POMDP created successfully!")
    print(f"  Discount factor: {pomdp.gamma}")
    print(f"  State space: {pomdp.n_states} states")
    print(f"  Action space: {pomdp.n_actions} actions")
    print(f"  Observation space: {pomdp.n_observations} observations")
    
    print("\n" + "="*60)
    print("POMDP formulation complete!")
    print("="*60)

