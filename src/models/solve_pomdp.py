"""
Solve OCD Treatment POMDP using Point-Based Value Iteration
Implements Algorithms 21.6 and 21.8 from "Algorithms for Decision Making"
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from algorithms.point_based_vi import (
    PointBasedValueIteration, RandomizedPointBasedValueIteration,
    generate_initial_beliefs, expand_beliefs
)
from models.ocd_pomdp import load_bn_and_create_pomdp, OCDTreatmentPOMDP
import time


def solve_with_pbvi(pomdp, num_beliefs=50, k_max=10, method='standard'):
    """
    Solve POMDP using Point-Based Value Iteration.
    
    Args:
        pomdp: POMDP instance
        num_beliefs: Number of belief points to sample
        k_max: Number of iterations
        method: 'standard' or 'randomized'
    
    Returns:
        policy: Learned policy
        solve_time: Time taken to solve
    """
    print(f"\n{'='*60}")
    print(f"Solving with {'Randomized' if method == 'randomized' else 'Standard'} PBVI")
    print(f"{'='*60}")
    
    # Generate initial belief points
    print(f"\n1. Generating {num_beliefs} belief points...")
    B = generate_initial_beliefs(pomdp, num_beliefs, method='corners')
    print(f"   Generated {len(B)} beliefs")
    
    # Optionally expand with simulated beliefs
    if num_beliefs > pomdp.n_states:
        print(f"   Expanding with simulation...")
        B = expand_beliefs(pomdp, B[:min(10, pomdp.n_states)], 
                          num_beliefs - min(10, pomdp.n_states), num_steps=5)
        print(f"   Total beliefs: {len(B)}")
    
    # Solve
    print(f"\n2. Running {'Randomized' if method == 'randomized' else 'Standard'} PBVI...")
    print(f"   Iterations: {k_max}")
    
    start_time = time.time()
    
    if method == 'randomized':
        solver = RandomizedPointBasedValueIteration(B, k_max)
    else:
        solver = PointBasedValueIteration(B, k_max)
    
    policy = solver.solve(pomdp)
    
    solve_time = time.time() - start_time
    
    print(f"\n3. Solving complete!")
    print(f"   Time: {solve_time:.2f} seconds")
    print(f"   Alpha vectors: {len(policy.Gamma)}")
    
    return policy, solve_time


def evaluate_policy(pomdp, policy, num_episodes=20, max_steps=5):
    """
    Evaluate policy through simulation.
    
    Args:
        pomdp: POMDP instance
        policy: Learned policy
        num_episodes: Number of episodes to simulate
        max_steps: Maximum steps per episode
    
    Returns:
        avg_reward: Average cumulative reward
        rewards: List of episode rewards
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Policy")
    print(f"{'='*60}")
    
    from algorithms.pomdp import update, create_uniform_belief
    
    rewards = []
    
    for ep in range(num_episodes):
        # Random initial state
        s_idx = np.random.randint(pomdp.n_states)
        s = pomdp.S[s_idx]
        
        # Start with uniform belief
        b = create_uniform_belief(pomdp)
        
        episode_reward = 0
        discount = 1.0
        
        for step in range(max_steps):
            # Select action
            a = policy.action(b)
            
            # Get reward
            r = pomdp.R(s, a)
            episode_reward += discount * r
            discount *= pomdp.gamma
            
            # Simulate next state and observation
            from algorithms.pomdp import simulate_step
            s_next, _, o = simulate_step(pomdp, s, a)
            
            # Update belief
            b = update(b, pomdp, a, o)
            
            # Move to next state
            s = s_next
        
        rewards.append(episode_reward)
        
        if (ep + 1) % 5 == 0:
            print(f"   Episode {ep + 1}/{num_episodes}: avg reward = {np.mean(rewards):.2f}")
    
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    print(f"\n   Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    
    return avg_reward, rewards


def main():
    """Main training pipeline"""
    print("="*60)
    print("OCD Treatment POMDP Solver")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Create POMDP
    print("\n1. Creating POMDP...")
    pomdp_model = OCDTreatmentPOMDP()
    pomdp = pomdp_model.create_pomdp(gamma=0.95)
    
    # Solve with standard PBVI (reduced for tractability)
    policy_std, time_std = solve_with_pbvi(
        pomdp, 
        num_beliefs=min(10, pomdp.n_states),  # Reduced from 30 to 10
        k_max=2,  # Reduced from 5 to 2 iterations
        method='standard'
    )
    
    # Evaluate (reduced for speed)
    avg_reward_std, rewards_std = evaluate_policy(pomdp, policy_std, num_episodes=5, max_steps=2)
    
    # Save policy
    print(f"\n{'='*60}")
    print("Saving Policy")
    print(f"{'='*60}")
    
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    policy_data = {
        'policy': policy_std,
        'pomdp_model': pomdp_model,
        'avg_reward': avg_reward_std,
        'solve_time': time_std,
        'method': 'pbvi_standard'
    }
    
    policy_path = os.path.join(models_dir, 'pomdp_policy.pkl')
    with open(policy_path, 'wb') as f:
        pickle.dump(policy_data, f)
    
    print(f"\n   Policy saved to: {policy_path}")
    
    print("\n" + "="*60)
    print("POMDP solving complete!")
    print("="*60)
    
    return policy_std, pomdp


if __name__ == "__main__":
    policy, pomdp = main()

