"""
Example usage of GAT and MLP encoders with EVRP environment.

This script demonstrates:
1. Creating encoders
2. Encoding EVRP instances
3. Comparing GAT vs MLP outputs
4. Batch processing
"""

import torch
import numpy as np
from evrp_rl.env import EVRPEnvironment
from evrp_rl.encoders import GATEncoder, MLPEncoder


def example_single_encoding():
    """Example 1: Encode a single EVRP instance."""
    print("=" * 70)
    print("Example 1: Single Instance Encoding")
    print("=" * 70)
    
    # Create environment and encoder
    env = EVRPEnvironment(num_customers=10, num_chargers=3)
    gat_encoder = GATEncoder(embed_dim=128, num_layers=3, num_heads=8)
    mlp_encoder = MLPEncoder(embed_dim=128, hidden_dim=256, num_layers=3)
    
    # Get observation
    obs, info = env.reset()
    print(f"Problem size: {obs['node_coords'].shape[0]} nodes")
    print(f"  - 1 depot")
    print(f"  - 10 customers")
    print(f"  - 3 charging stations")
    
    # Prepare graph data
    graph_data = {
        'node_coords': torch.from_numpy(obs['node_coords']).unsqueeze(0).float(),
        'node_demands': torch.from_numpy(obs['node_demands']).unsqueeze(0).float(),
        'node_types': torch.from_numpy(obs['node_types']).unsqueeze(0).float(),
        'distance_matrix': torch.from_numpy(obs['distance_matrix']).unsqueeze(0).float(),
    }
    
    # Encode with both encoders
    gat_encoder.eval()
    mlp_encoder.eval()
    
    with torch.no_grad():
        gat_node_embeds, gat_graph_embed = gat_encoder(graph_data)
        mlp_node_embeds, mlp_graph_embed = mlp_encoder(graph_data)
    
    print(f"\nGAT Encoder:")
    print(f"  Node embeddings: {gat_node_embeds.shape}")
    print(f"  Graph embedding: {gat_graph_embed.shape}")
    print(f"  Parameters: {sum(p.numel() for p in gat_encoder.parameters()):,}")
    
    print(f"\nMLP Encoder:")
    print(f"  Node embeddings: {mlp_node_embeds.shape}")
    print(f"  Graph embedding: {mlp_graph_embed.shape}")
    print(f"  Parameters: {sum(p.numel() for p in mlp_encoder.parameters()):,}")
    
    # Compare embeddings
    print(f"\nEmbedding Statistics:")
    print(f"  GAT graph embed mean: {gat_graph_embed.mean():.4f}, std: {gat_graph_embed.std():.4f}")
    print(f"  MLP graph embed mean: {mlp_graph_embed.mean():.4f}, std: {mlp_graph_embed.std():.4f}")
    print()


def example_batch_encoding():
    """Example 2: Batch encoding of multiple instances."""
    print("=" * 70)
    print("Example 2: Batch Encoding")
    print("=" * 70)
    
    batch_size = 8
    num_customers = 10
    num_chargers = 3
    
    # Create encoder
    encoder = GATEncoder(embed_dim=64, num_layers=2, num_heads=4)
    encoder.eval()
    
    # Create multiple environments
    envs = [EVRPEnvironment(num_customers=num_customers, num_chargers=num_chargers) 
            for _ in range(batch_size)]
    observations = [env.reset()[0] for env in envs]
    
    print(f"Batch size: {batch_size}")
    print(f"Problem size: {num_customers + num_chargers + 1} nodes each")
    
    # Stack into batched tensors
    graph_data = {
        'node_coords': torch.stack([
            torch.from_numpy(obs['node_coords']).float() 
            for obs in observations
        ]),
        'node_demands': torch.stack([
            torch.from_numpy(obs['node_demands']).float() 
            for obs in observations
        ]),
        'node_types': torch.stack([
            torch.from_numpy(obs['node_types']).float() 
            for obs in observations
        ]),
        'distance_matrix': torch.stack([
            torch.from_numpy(obs['distance_matrix']).float() 
            for obs in observations
        ]),
    }
    
    # Encode batch
    with torch.no_grad():
        node_embeds, graph_embeds = encoder(graph_data)
    
    print(f"\nOutput shapes:")
    print(f"  Node embeddings: {node_embeds.shape}")
    print(f"  Graph embeddings: {graph_embeds.shape}")
    
    # Analyze embeddings
    print(f"\nBatch statistics:")
    for i in range(min(3, batch_size)):
        print(f"  Instance {i}: graph_embed mean={graph_embeds[i].mean():.4f}, "
              f"std={graph_embeds[i].std():.4f}")
    print()


def example_episode_encoding():
    """Example 3: Encode states during an episode."""
    print("=" * 70)
    print("Example 3: Episode State Encoding")
    print("=" * 70)
    
    # Create environment and encoder
    env = EVRPEnvironment(num_customers=5, num_chargers=2, seed=42)
    encoder = MLPEncoder(embed_dim=64, hidden_dim=128, num_layers=2)
    encoder.eval()
    
    obs, info = env.reset()
    embeddings_history = []
    actions_taken = []
    
    print("Taking random actions and encoding each state...")
    
    for step in range(10):
        # Prepare graph data
        graph_data = {
            'node_coords': torch.from_numpy(obs['node_coords']).unsqueeze(0).float(),
            'node_demands': torch.from_numpy(obs['node_demands']).unsqueeze(0).float(),
            'node_types': torch.from_numpy(obs['node_types']).unsqueeze(0).float(),
            'distance_matrix': torch.from_numpy(obs['distance_matrix']).unsqueeze(0).float(),
        }
        
        # Encode current state
        with torch.no_grad():
            node_embeds, graph_embed = encoder(graph_data)
        
        embeddings_history.append(graph_embed.squeeze(0).numpy())
        
        # Take random action
        valid_actions = np.where(obs['valid_actions_mask'])[0]
        if len(valid_actions) == 0:
            print(f"  Step {step}: No valid actions, episode terminated")
            break
        
        action = np.random.choice(valid_actions)
        actions_taken.append(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step < 5:
            print(f"  Step {step}: action={action}, reward={reward:.2f}, "
                  f"embed_mean={graph_embed.mean():.4f}")
        
        if terminated or truncated:
            print(f"  Episode finished at step {step}")
            break
    
    print(f"\nCollected {len(embeddings_history)} state embeddings")
    embeddings_array = np.stack(embeddings_history)
    print(f"Embeddings shape: {embeddings_array.shape}")
    print(f"Overall mean: {embeddings_array.mean():.4f}, std: {embeddings_array.std():.4f}")
    print()


def example_encoder_comparison():
    """Example 4: Compare GAT and MLP performance."""
    print("=" * 70)
    print("Example 4: GAT vs MLP Comparison")
    print("=" * 70)
    
    # Create encoders with same embed_dim
    embed_dim = 128
    gat = GATEncoder(embed_dim=embed_dim, num_layers=3, num_heads=4)
    mlp = MLPEncoder(embed_dim=embed_dim, hidden_dim=256, num_layers=3)
    
    # Count parameters
    gat_params = sum(p.numel() for p in gat.parameters())
    mlp_params = sum(p.numel() for p in mlp.parameters())
    
    print(f"Model Complexity:")
    print(f"  GAT: {gat_params:,} parameters")
    print(f"  MLP: {mlp_params:,} parameters")
    print(f"  Ratio: {gat_params / mlp_params:.2f}x more parameters in GAT")
    
    # Timing comparison
    import time
    
    # Create test data
    batch_size = 16
    num_nodes = 20
    graph_data = {
        'node_coords': torch.rand(batch_size, num_nodes, 2),
        'node_demands': torch.rand(batch_size, num_nodes),
        'node_types': torch.randint(0, 3, (batch_size, num_nodes)),
        'distance_matrix': torch.rand(batch_size, num_nodes, num_nodes),
    }
    
    # Warm-up
    gat.eval()
    mlp.eval()
    with torch.no_grad():
        _ = gat(graph_data)
        _ = mlp(graph_data)
    
    # Time GAT
    num_runs = 50
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = gat(graph_data)
    gat_time = (time.time() - start) / num_runs * 1000
    
    # Time MLP
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = mlp(graph_data)
    mlp_time = (time.time() - start) / num_runs * 1000
    
    print(f"\nInference Time (batch_size={batch_size}, num_nodes={num_nodes}):")
    print(f"  GAT: {gat_time:.2f} ms/batch")
    print(f"  MLP: {mlp_time:.2f} ms/batch")
    print(f"  Speedup: {gat_time / mlp_time:.2f}x faster with MLP")
    
    print(f"\nTrade-offs:")
    print(f"  GAT: More expressive (uses graph structure), slower, more parameters")
    print(f"  MLP: Simple baseline, faster, fewer parameters, ignores structure")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("EVRP Encoder Examples")
    print("=" * 70 + "\n")
    
    example_single_encoding()
    example_batch_encoding()
    example_episode_encoding()
    example_encoder_comparison()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
