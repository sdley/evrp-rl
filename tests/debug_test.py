#!/usr/bin/env python3
"""Quick test of EVRP environment."""

from evrp_rl.env import EVRPEnvironment
import numpy as np

env = EVRPEnvironment(num_customers=3, num_chargers=1, max_battery=500.0)
obs, _ = env.reset()

print("Node indices:")
print(f"  Customer range: {env.customer_start_idx} - {env.customer_end_idx}")
print(f"  Charger range: {env.charger_start_idx} - {env.charger_end_idx}")

valid_actions = obs["valid_actions_mask"]
valid_customers = [i for i in range(env.customer_start_idx, env.customer_end_idx + 1)
                  if valid_actions[i]]

print(f"Valid customers: {valid_customers}")
print(f"Valid actions: {np.where(valid_actions)[0]}")

if valid_customers:
    customer1 = valid_customers[0]
    print(f"\nVisiting customer {customer1}")
    obs, reward, terminated, truncated, info = env.step(customer1)
    print(f"  Visited: {env.visited_mask}")
    print(f"  Current node: {env.current_node}")
    print(f"  Reward: {reward}")
