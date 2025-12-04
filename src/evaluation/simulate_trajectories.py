"""
Simulate Treatment Trajectories and Evaluate Policies
Compares POMDP policy against baseline strategies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import pickle

from algorithms.pomdp import update, create_uniform_belief, simulate_step
from models.ocd_pomdp import OCDTreatmentState, OCDTreatmentAction, OCDTreatmentObservation


class BaselinePolicy:
    """Baseline policy for comparison"""
    
    def __init__(self, strategy='always_cbt'):
        self.strategy = strategy
        self.treatment_map = {
            'cbt': 1,
            'ssri': 2,
            'tms': 3,
            'ketamine': 4,
            'cbt_ssri': 5
        }
    
    def action(self, b=None, obs_history=None):
        """Select action based on baseline strategy"""
        if self.strategy == 'always_cbt':
            return OCDTreatmentAction('start', self.treatment_map['cbt'])
        elif self.strategy == 'always_ssri':
            return OCDTreatmentAction('start', self.treatment_map['ssri'])
        elif self.strategy == 'always_cbt_ssri':
            return OCDTreatmentAction('start', self.treatment_map['cbt_ssri'])
        elif self.strategy == 'random':
            tx = np.random.choice([1, 2, 5])  # CBT, SSRI, or CBT+SSRI
            return OCDTreatmentAction('start', tx)
        else:
            return OCDTreatmentAction('continue')


def simulate_episode(pomdp, policy, initial_state_idx, max_steps=5, verbose=False):
    """
    Simulate one treatment episode.
    
    Args:
        pomdp: POMDP instance
        policy: Policy to evaluate
        initial_state_idx: Index of initial state
        max_steps: Maximum number of decision points
        verbose: Print details
    
    Returns:
        total_reward: Cumulative discounted reward
        trajectory: List of (state, action, reward, observation) tuples
    """
    s = pomdp.S[initial_state_idx]
    b = create_uniform_belief(pomdp)
    
    total_reward = 0
    discount = 1.0
    trajectory = []
    
    for step in range(max_steps):
        # Select action
        if hasattr(policy, 'action'):
            a = policy.action(b)
        else:
            a = policy.action()  # Baseline
        
        # Get reward
        r = pomdp.R(s, a)
        total_reward += discount * r
        
        # Simulate step
        s_next, _, o = simulate_step(pomdp, s, a)
        
        trajectory.append((s, a, r, o))
        
        if verbose:
            print(f"  Step {step}: {a} -> reward={r:.1f}, ybocs={s_next.ybocs_severity}")
        
        # Update belief
        b = update(b, pomdp, a, o)
        
        # Update state
        s = s_next
        discount *= pomdp.gamma
    
    return total_reward, trajectory


def evaluate_policies(pomdp, policies_dict, num_episodes=50, max_steps=5):
    """
    Evaluate multiple policies.
    
    Args:
        pomdp: POMDP instance
        policies_dict: Dictionary of {policy_name: policy}
        num_episodes: Number of episodes per policy
        max_steps: Steps per episode
    
    Returns:
        results: Dictionary of evaluation metrics per policy
    """
    print("="*60)
    print("Evaluating Policies")
    print("="*60)
    
    results = {}
    
    for policy_name, policy in policies_dict.items():
        print(f"\n{policy_name}:")
        
        rewards = []
        final_severities = []
        
        for ep in range(num_episodes):
            # Random initial state (moderate-severe OCD)
            eligible_states = [i for i, s in enumerate(pomdp.S) 
                             if s.ybocs_severity >= 3 and s.current_tx == 0]
            s_idx = np.random.choice(eligible_states)
            
            total_reward, trajectory = simulate_episode(pomdp, policy, s_idx, max_steps)
            
            rewards.append(total_reward)
            final_state = trajectory[-1][0]
            final_severities.append(final_state.ybocs_severity)
            
            if (ep + 1) % 10 == 0:
                print(f"  Episodes {ep + 1}/{num_episodes}: "
                      f"avg reward={np.mean(rewards):.2f}, "
                      f"avg final ybocs={np.mean(final_severities):.2f}")
        
        results[policy_name] = {
            'rewards': rewards,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'final_severities': final_severities,
            'mean_severity': np.mean(final_severities),
            'remission_rate': np.mean([s == 1 for s in final_severities])
        }
        
        print(f"  Mean reward: {results[policy_name]['mean_reward']:.2f} ± "
              f"{results[policy_name]['std_reward']:.2f}")
        print(f"  Mean final Y-BOCS: {results[policy_name]['mean_severity']:.2f}")
        print(f"  Remission rate: {results[policy_name]['remission_rate']*100:.1f}%")
    
    return results


def plot_results(results, save_path=None):
    """Plot comparison of policies"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    policy_names = list(results.keys())
    
    # 1. Mean rewards
    ax = axes[0]
    means = [results[p]['mean_reward'] for p in policy_names]
    stds = [results[p]['std_reward'] for p in policy_names]
    
    ax.bar(range(len(policy_names)), means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xticks(range(len(policy_names)))
    ax.set_xticklabels(policy_names, rotation=45, ha='right')
    ax.set_ylabel('Average Cumulative Reward')
    ax.set_title('Policy Performance')
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Final Y-BOCS severity
    ax = axes[1]
    severities_data = [results[p]['final_severities'] for p in policy_names]
    
    bp = ax.boxplot(severities_data, labels=policy_names)
    ax.set_ylabel('Final Y-BOCS Severity')
    ax.set_title('Treatment Outcomes')
    ax.set_xticklabels(policy_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Remission rates
    ax = axes[2]
    remission_rates = [results[p]['remission_rate'] * 100 for p in policy_names]
    
    ax.bar(range(len(policy_names)), remission_rates, alpha=0.7, color='green')
    ax.set_xticks(range(len(policy_names)))
    ax.set_xticklabels(policy_names, rotation=45, ha='right')
    ax.set_ylabel('Remission Rate (%)')
    ax.set_title('Remission Rates (Y-BOCS ≤ 8)')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
    
    plt.show()
    
    return fig


def main():
    """Main evaluation pipeline"""
    print("="*60)
    print("Treatment Trajectory Simulation")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Create POMDP
    print("\n1. Loading POMDP...")
    from models.ocd_pomdp import OCDTreatmentPOMDP
    pomdp_model = OCDTreatmentPOMDP()
    pomdp = pomdp_model.create_pomdp(gamma=0.95)
    
    # Define policies to compare
    print("\n2. Setting up policies...")
    policies = {
        'CBT First-Line': BaselinePolicy('always_cbt'),
        'SSRI First-Line': BaselinePolicy('always_ssri'),
        'CBT+SSRI First-Line': BaselinePolicy('always_cbt_ssri'),
        'Random': BaselinePolicy('random'),
    }
    
    # Try to load trained POMDP policy
    policy_path = os.path.join(base_dir, 'models/pomdp_policy.pkl')
    if os.path.exists(policy_path):
        print("   Loading trained POMDP policy...")
        with open(policy_path, 'rb') as f:
            policy_data = pickle.load(f)
            # Check if it contains actual policy or just stats
            if 'policy' in policy_data:
                policies['POMDP Policy'] = policy_data['policy']
                print("   ✓ POMDP policy loaded")
            else:
                print("   ⚠ POMDP policy file found but contains simplified version")
                print("   Continuing with baseline policies only...")
    else:
        print("   ⚠ No trained POMDP policy found.")
        print("   Continuing with baseline policies only...")
    
    # Evaluate
    print("\n3. Running simulations...")
    results = evaluate_policies(pomdp, policies, num_episodes=30, max_steps=4)
    
    # Plot
    print("\n4. Generating plots...")
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    fig = plot_results(results, save_path=os.path.join(results_dir, 'policy_comparison.png'))
    
    # Save results
    results_file = os.path.join(results_dir, 'evaluation_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_file}")
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = main()

