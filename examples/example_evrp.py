"""
Example script demonstrating the EVRP Environment usage.

This script shows:
1. Environment initialization
2. Basic interaction (reset, step, render)
3. Random policy execution
4. Environment visualization
"""

import numpy as np
from evrp_rl.env import EVRPEnvironment


def run_random_episode():
    """Run a single episode with random policy."""
    print("=" * 60)
    print("EVRP Environment - Random Policy Episode")
    print("=" * 60)
    
    # Create environment
    env = EVRPEnvironment(
        num_customers=8,
        num_chargers=3,
        max_battery=150.0,
        max_cargo=100.0,
        energy_consumption_rate=0.8,
        time_limit=50,
        render_mode="human",
    )
    
    # Reset environment
    print("\n[1] Resetting environment...")
    observation, info = env.reset()
    print(f"Initial state:")
    print(f"  Current node: {info['current_node']} (depot)")
    print(f"  Battery: {info['current_battery']:.1f} / {env.max_battery}")
    print(f"  Cargo: {info['current_cargo']:.1f} / {env.max_cargo}")
    print(f"  Time limit: {info['time_limit']} steps")
    
    total_reward = 0.0
    
    # Run episode
    print("\n[2] Running episode with random policy...")
    while True:
        # Get valid actions
        valid_actions = observation["valid_actions_mask"]
        valid_action_indices = np.where(valid_actions)[0]
        
        if len(valid_action_indices) == 0:
            print("No valid actions! Episode terminated.")
            break
        
        # Random action from valid actions
        action = np.random.choice(valid_action_indices)
        
        # Get node names for readability
        node_names = {
            0: "Depot",
            **{i: f"Customer-{i}" for i in range(1, env.num_customers + 1)},
            **{
                i: f"Charger-{i - env.num_customers}"
                for i in range(env.num_customers + 1, env.num_nodes)
            },
        }
        
        current_name = node_names[env.current_node]
        next_name = node_names[action]
        
        # Step
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Print step info
        print(
            f"\nStep {info['current_step']}: {current_name} -> {next_name}"
        )
        print(
            f"  Reward: {reward:+.2f}, Total: {total_reward:+.2f}"
        )
        print(
            f"  Battery: {info['current_battery']:.1f} / {env.max_battery}"
        )
        print(
            f"  Cargo: {info['current_cargo']:.1f} / {env.max_cargo}"
        )
        print(
            f"  Visited customers: {info['visited_customers']} / {env.num_customers}"
        )
        print(
            f"  Distance traveled: {info['total_distance']:.2f}"
        )
        
        # Check termination
        if terminated:
            print(f"\n✓ Episode terminated (goal reached or infeasible)")
            break
        
        if truncated:
            print(f"\n✗ Episode truncated (time limit exceeded)")
            break
    
    # Final statistics
    print("\n" + "=" * 60)
    print("Episode Summary")
    print("=" * 60)
    print(f"Total steps: {info['current_step']}")
    print(f"Total distance: {info['total_distance']:.2f} km")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Visited customers: {info['visited_customers']} / {env.num_customers}")
    print(f"Depot visits: {info['depot_visits']}")
    print(f"Charger visits: {info['charger_visits']}")
    print(f"Success: {info['visited_customers'] == env.num_customers}")
    
    # Render final state
    print("\n[3] Rendering environment state...")
    env.render()
    
    env.close()


def demonstrate_graph_properties():
    """Demonstrate graph and problem properties."""
    print("\n" + "=" * 60)
    print("EVRP Environment - Problem Properties")
    print("=" * 60)
    
    env = EVRPEnvironment(
        num_customers=5,
        num_chargers=2,
        max_battery=100.0,
    )
    env.reset()
    
    print(f"\nProblem Instance:")
    print(f"  Nodes: {env.num_nodes}")
    print(f"  Customers: {env.num_customers}")
    print(f"  Charging stations: {env.num_chargers}")
    print(f"  Max battery: {env.max_battery} Wh")
    print(f"  Max cargo: {env.max_cargo} kg")
    
    # Get graph
    graph = env.get_graph()
    print(f"\nGraph Properties:")
    print(f"  Number of nodes: {graph.number_of_nodes()}")
    print(f"  Number of edges: {graph.number_of_edges()}")
    print(f"  Is complete: {graph.number_of_edges() == env.num_nodes * (env.num_nodes - 1) / 2}")
    
    # Distance statistics
    coords = env.get_node_coordinates()
    dist_matrix = env.get_distance_matrix()
    
    # Remove diagonal
    distances = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    
    print(f"\nDistance Statistics:")
    print(f"  Min: {distances.min():.2f} km")
    print(f"  Max: {distances.max():.2f} km")
    print(f"  Mean: {distances.mean():.2f} km")
    print(f"  Median: {np.median(distances):.2f} km")
    
    # Customer demands
    customer_demands = env.node_demands[env.customer_start_idx:env.customer_end_idx + 1]
    print(f"\nCustomer Demands:")
    print(f"  Total: {customer_demands.sum():.1f} kg")
    print(f"  Min: {customer_demands.min():.1f} kg")
    print(f"  Max: {customer_demands.max():.1f} kg")
    print(f"  Mean: {customer_demands.mean():.1f} kg")
    
    env.close()


def test_battery_and_cargo_mechanics():
    """Test battery and cargo mechanics."""
    print("\n" + "=" * 60)
    print("EVRP Environment - Battery & Cargo Mechanics")
    print("=" * 60)
    
    env = EVRPEnvironment(
        num_customers=3,
        num_chargers=1,
        max_battery=100.0,
        max_cargo=50.0,
        energy_consumption_rate=1.0,
    )
    
    obs, info = env.reset()
    
    print(f"\nInitial State:")
    print(f"  Current node: {info['current_node']}")
    print(f"  Battery: {info['current_battery']:.1f}")
    print(f"  Cargo: {info['current_cargo']:.1f}")
    
    # Move to customer
    print(f"\nAction: Move to Customer-1 (node 1)")
    obs, reward, terminated, truncated, info = env.step(1)
    print(f"  Battery: {info['current_battery']:.1f} (decreased)")
    print(f"  Cargo: {info['current_cargo']:.1f} (increased by demand)")
    
    # Move to charger
    print(f"\nAction: Move to Charger (node {env.charger_start_idx})")
    obs, reward, terminated, truncated, info = env.step(env.charger_start_idx)
    print(f"  Battery: {info['current_battery']:.1f} (refilled to max)")
    print(f"  Cargo: {info['current_cargo']:.1f} (unchanged)")
    
    # Return to depot
    print(f"\nAction: Return to Depot (node 0)")
    obs, reward, terminated, truncated, info = env.step(0)
    print(f"  Battery: {info['current_battery']:.1f} (decreased by travel)")
    print(f"  Cargo: {info['current_cargo']:.1f} (unloaded at depot)")
    
    env.close()


if __name__ == "__main__":
    # Run demonstrations
    run_random_episode()
    demonstrate_graph_properties()
    test_battery_and_cargo_mechanics()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)
