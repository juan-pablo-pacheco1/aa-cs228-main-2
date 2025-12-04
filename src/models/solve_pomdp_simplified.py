"""
Simplified POMDP Solver - Tractable Version
Uses a reduced state space for demonstration purposes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from algorithms.pomdp import POMDP
from algorithms.point_based_vi import (
    PointBasedValueIteration,
    generate_initial_beliefs
)
import time


class SimplifiedOCDState:
    """Simplified state: just severity and treatment"""
    def __init__(self, severity, treatment):
        self.severity = severity  # 1-3 (mild, moderate, severe)
        self.treatment = treatment  # 0-2 (none, CBT, SSRI)
    
    def __eq__(self, other):
        return self.severity == other.severity and self.treatment == other.treatment
    
    def __hash__(self):
        return hash((self.severity, self.treatment))
    
    def __repr__(self):
        sev = ['mild', 'moderate', 'severe'][self.severity-1]
        tx = ['none', 'CBT', 'SSRI'][self.treatment]
        return f"State({sev}, {tx})"


class SimplifiedAction:
    def __init__(self, action_id):
        self.action_id = action_id  # 0=continue, 1=CBT, 2=SSRI
    
    def __eq__(self, other):
        return self.action_id == other.action_id
    
    def __hash__(self):
        return hash(self.action_id)
    
    def __repr__(self):
        names = ['continue', 'start_CBT', 'start_SSRI']
        return f"Action({names[self.action_id]})"


class SimplifiedObs:
    def __init__(self, obs_severity):
        self.obs_severity = obs_severity  # 1-3 (with noise)
    
    def __eq__(self, other):
        return self.obs_severity == other.obs_severity
    
    def __hash__(self):
        return hash(self.obs_severity)
    
    def __repr__(self):
        return f"Obs({self.obs_severity})"


def create_simplified_pomdp():
    """Create small POMDP for demonstration"""
    # State space: 3 severities × 3 treatments = 9 states
    states = []
    for sev in range(1, 4):
        for tx in range(3):
            states.append(SimplifiedOCDState(sev, tx))
    
    # Action space: continue, start CBT, start SSRI
    actions = [SimplifiedAction(i) for i in range(3)]
    
    # Observation space: 3 severity levels (with noise)
    observations = [SimplifiedObs(i) for i in range(1, 4)]
    
    print(f"Simplified POMDP:")
    print(f"  States: {len(states)}")
    print(f"  Actions: {len(actions)}")
    print(f"  Observations: {len(observations)}")
    
    # Treatment efficacy
    efficacy = {0: 0.0, 1: 0.65, 2: 0.50}  # CBT, SSRI
    
    def transition(s, a, s_prime):
        # Determine expected treatment
        if a.action_id == 0:  # Continue
            exp_tx = s.treatment
        else:  # Start new treatment
            exp_tx = a.action_id
        
        if s_prime.treatment != exp_tx:
            return 0.0
        
        # Severity transition
        if s_prime.treatment == 0:  # No treatment
            # Likely stays same or worsens
            if s_prime.severity == s.severity:
                return 0.7
            elif s_prime.severity == min(s.severity + 1, 3):
                return 0.2
            elif s_prime.severity == max(s.severity - 1, 1):
                return 0.1
            else:
                return 0.0
        else:
            # On treatment: may improve
            eff = efficacy[s_prime.treatment]
            if np.random.random() < eff:
                # Responding: move toward lower severity
                if s_prime.severity < s.severity:
                    return 0.6
                elif s_prime.severity == s.severity:
                    return 0.3
                else:
                    return 0.1
            else:
                # Not responding: stays similar
                if s_prime.severity == s.severity:
                    return 0.8
                else:
                    return 0.1
        
        return 0.1  # Small uniform probability
    
    def observation_func(a, s_prime, o):
        # Noisy observation of severity
        if o.obs_severity == s_prime.severity:
            return 0.7
        elif abs(o.obs_severity - s_prime.severity) == 1:
            return 0.15
        else:
            return 0.0
    
    def reward(s, a):
        # Reward based on severity (lower is better)
        severity_cost = -10 * s.severity
        
        # Treatment cost
        tx_cost = {0: 0, 1: -2, 2: -1}  # CBT more expensive
        
        # Switching penalty
        switch_penalty = -5 if a.action_id > 0 else 0
        
        return severity_cost + tx_cost[s.treatment] + switch_penalty
    
    pomdp = POMDP(
        gamma=0.9,
        S=states,
        A=actions,
        O=observations,
        T=transition,
        R=reward,
        O_func=observation_func
    )
    
    return pomdp


def main():
    print("="*60)
    print("Simplified OCD Treatment POMDP (Fast Version)")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Create simplified POMDP
    print("\n1. Creating simplified POMDP...")
    pomdp = create_simplified_pomdp()
    
    # Generate belief points
    print("\n2. Generating belief points...")
    B = generate_initial_beliefs(pomdp, num_points=5, method='corners')
    print(f"   Generated {len(B)} beliefs")
    
    # Solve with PBVI
    print("\n3. Solving with Point-Based VI...")
    print("   Iterations: 3")
    
    start_time = time.time()
    solver = PointBasedValueIteration(B, k_max=3)
    policy = solver.solve(pomdp)
    solve_time = time.time() - start_time
    
    print(f"\n   ✓ Solved in {solve_time:.2f} seconds")
    print(f"   Alpha vectors: {len(policy.Gamma)}")
    
    # Quick evaluation
    print("\n4. Evaluating policy...")
    from algorithms.pomdp import update, create_uniform_belief, simulate_step
    
    rewards = []
    for ep in range(5):
        s_idx = np.random.randint(pomdp.n_states)
        s = pomdp.S[s_idx]
        b = create_uniform_belief(pomdp)
        
        episode_reward = 0
        discount = 1.0
        
        for step in range(3):
            a = policy.action(b)
            r = pomdp.R(s, a)
            episode_reward += discount * r
            discount *= pomdp.gamma
            
            s_next, _, o = simulate_step(pomdp, s, a)
            b = update(b, pomdp, a, o)
            s = s_next
        
        rewards.append(episode_reward)
    
    print(f"   Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    
    # Save
    print("\n5. Saving policy...")
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    policy_data = {
        'avg_reward': np.mean(rewards),
        'solve_time': solve_time,
        'method': 'pbvi_simplified',
        'n_states': len(pomdp.S),
        'n_actions': len(pomdp.A),
        'n_alpha_vectors': len(policy.Gamma),
        'note': 'Simplified 9-state POMDP for demonstration'
    }
    
    # Save just the statistics (policy can't be pickled due to local functions)
    with open(os.path.join(models_dir, 'pomdp_policy.pkl'), 'wb') as f:
        pickle.dump(policy_data, f)
    
    print(f"   ✓ Saved results to models/pomdp_policy.pkl")
    
    print("\n" + "="*60)
    print("POMDP solving complete!")
    print("="*60)
    print("\nNote: This is a simplified version (9 states) for demonstration.")
    print("The full version (1,920 states) requires significant computation.")
    
    return policy


if __name__ == "__main__":
    policy = main()

