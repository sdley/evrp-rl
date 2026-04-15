#!/usr/bin/env python3
"""Verification script for EVRP environment."""

from evrp_rl.env import EVRPEnvironment
import numpy as np

print("=" * 60)
print("EVRP Environment Verification")
print("=" * 60)

# Create environment
env = EVRPEnvironment(
    num_customers=5,
    num_chargers=2,
    max_battery=100.0,
    max_cargo=50.0,
)

print("\n[1] Environment Properties")
print(f"  Nodes: {env.num_nodes}")
print(f"  Customers: {env.num_customers}")
print(f"  Chargers: {env.num_chargers}")
print(f"  Max Battery: {env.max_battery} Wh")
print(f"  Max Cargo: {env.max_cargo} kg")

# Reset and get observation
obs, info = env.reset()

print("\n[2] Initial State")
print(f"  Current node: {info['current_node']} (depot)")
print(f"  Battery: {info['current_battery']:.1f} Wh")
print(f"  Cargo: {info['current_cargo']:.1f} kg")
print(f"  Customers visited: {info['visited_customers']}")

# Take a few steps
print("\n[3] Episode Execution")
total_reward = 0
step_count = 0

for _ in range(5):
    valid_mask = obs["valid_actions_mask"]
    valid_actions = [i for i, v in enumerate(valid_mask) if v]
    
    if not valid_actions:
        break
    
    action = np.random.choice(valid_actions)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    step_count += 1
    
    print(f"  Step {step_count}: Action {action} -> Reward {reward:.2f}")

print("\n[4] Final State")
print(f"  Steps taken: {step_count}")
print(f"  Total reward: {total_reward:.2f}")
print(f"  Distance: {info['total_distance']:.2f} km")
print(f"  Battery: {info['current_battery']:.1f} Wh")
print(f"  Cargo: {info['current_cargo']:.1f} kg")

print("\n[5] Graph Utilities")
graph = env.get_graph()
coords = env.get_node_coordinates()
dist_matrix = env.get_distance_matrix()
print(f"  Graph nodes: {graph.number_of_nodes()}")
print(f"  Coordinates shape: {coords.shape}")
print(f"  Distance matrix shape: {dist_matrix.shape}")

print("\n" + "=" * 60)
print("✓ EVRP Environment Operational")
print("=" * 60)
