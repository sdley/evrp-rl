# Encoder Modules for EVRP

This document describes the encoder architectures for embedding Electric Vehicle Routing Problem (EVRP) instances into continuous representations suitable for reinforcement learning agents.

## Overview

Encoders transform graph-structured EVRP instances into vector embeddings that capture the problem structure and relationships between nodes. These embeddings serve as input to RL agents for decision-making.

**Available Encoders:**

1. **GATEncoder**: Graph Attention Network with edge-aware attention
2. **MLPEncoder**: Simple MLP baseline for independent node embedding

## Architecture Comparison

### Graph Attention Network (GAT) Encoder

**Key Features:**

- Graph-aware processing using attention mechanisms
- Edge features (distances) incorporated into attention computation
- Multi-head attention for capturing diverse relationships
- Message passing aggregates neighborhood information

**Architecture:**

```
Input Node Features (6D)
    ↓
Input Projection → Embedding Space
    ↓
GAT Layer 1 (Edge-aware Attention)
    ↓ [Residual Connection + LayerNorm]
GAT Layer 2 (Edge-aware Attention)
    ↓ [Residual Connection + LayerNorm]
    ...
GAT Layer L (Edge-aware Attention)
    ↓
Node Embeddings (num_nodes × embed_dim)
    ↓ [Mean Pooling]
Graph Embedding (embed_dim)
```

**Attention Mechanism:**

```
For edge (i, j):
α_ij = LeakyReLU(LeakyReLU(W[h_i || h_j || e_ij]))

where:
- h_i: embedding of node i
- h_j: embedding of node j
- e_ij: edge distance between i and j
- ||: concatenation
- W: learnable weight matrix
```

**When to Use:**

- Complex routing problems with spatial structure
- When node relationships matter (nearby nodes affect decisions)
- Training data is sufficient for larger models
- Inference time is not critical

### MLP Encoder

**Key Features:**

- Simple feed-forward architecture
- Processes each node independently
- No graph structure utilization
- Fast and lightweight

**Architecture:**

```
Input Node Features (6D)
    ↓
Linear → Hidden Dim
    ↓ [ReLU + Dropout]
Linear → Hidden Dim
    ↓ [LayerNorm + ReLU + Dropout]
    ...
Linear → Embed Dim
    ↓
Node Embeddings (num_nodes × embed_dim)
    ↓ [Mean Pooling]
Graph Embedding (embed_dim)
```

**When to Use:**

- Fast baseline comparisons
- Small-scale problems
- Limited computational resources
- Graph structure is less important

## Input Format

All encoders expect a dictionary with the following keys:

```python
graph_data = {
    'node_coords': Tensor[batch, num_nodes, 2],      # (x, y) coordinates
    'node_demands': Tensor[batch, num_nodes],        # demand at each node
    'node_types': Tensor[batch, num_nodes, 3],       # one-hot: [depot, customer, charger]
    'distance_matrix': Tensor[batch, num_nodes, num_nodes],  # pairwise distances
}
```

**Node Features (6D):**

1. x-coordinate (normalized)
2. y-coordinate (normalized)
3. Demand value (normalized)
4. Is depot (0 or 1)
5. Is customer (0 or 1)
6. Is charging station (0 or 1)

## Output Format

Both encoders return a tuple `(node_embeddings, graph_embedding)`:

```python
node_embeddings: Tensor[batch, num_nodes, embed_dim]
graph_embedding: Tensor[batch, embed_dim]
```

- **Node Embeddings**: Individual vector for each node, useful for attention-based action selection
- **Graph Embedding**: Single vector representing the entire problem instance

## Usage Examples

### Basic Usage

```python
from src.encoders import GATEncoder, MLPEncoder
import torch

# Initialize encoder
encoder = GATEncoder(
    embed_dim=128,
    num_layers=3,
    num_heads=8,
    dropout=0.1
)

# Prepare graph data
graph_data = {
    'node_coords': torch.rand(4, 20, 2),      # batch=4, nodes=20
    'node_demands': torch.rand(4, 20),
    'node_types': torch.rand(4, 20, 3),
    'distance_matrix': torch.rand(4, 20, 20),
}

# Encode
node_embeds, graph_embed = encoder(graph_data)
print(f"Node embeddings: {node_embeds.shape}")    # (4, 20, 128)
print(f"Graph embedding: {graph_embed.shape}")    # (4, 128)
```

### Integration with EVRP Environment

```python
from src.env import EVRPEnvironment
from src.encoders import GATEncoder
import torch

# Create environment and encoder
env = EVRPEnvironment(num_customers=10, num_chargers=3)
encoder = GATEncoder(embed_dim=128)

# Get observation
obs, info = env.reset()

# Convert to torch tensors and add batch dimension
graph_data = {
    'node_coords': torch.from_numpy(obs['node_coords']).unsqueeze(0).float(),
    'node_demands': torch.from_numpy(obs['node_demands']).unsqueeze(0).float(),
    'node_types': torch.from_numpy(obs['node_types']).unsqueeze(0).float(),
    'distance_matrix': torch.from_numpy(obs['distance_matrix']).unsqueeze(0).float(),
}

# Encode state
encoder.eval()
with torch.no_grad():
    node_embeds, graph_embed = encoder(graph_data)

# Use embeddings for RL agent
# ... (action selection based on node_embeds)
```

### Batch Processing Multiple Episodes

```python
# Collect observations from multiple environments
batch_size = 8
envs = [EVRPEnvironment() for _ in range(batch_size)]
observations = [env.reset()[0] for env in envs]

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
node_embeds, graph_embeds = encoder(graph_data)
# Shape: node_embeds=(8, num_nodes, 128), graph_embeds=(8, 128)
```

## Hyperparameter Guide

### GATEncoder

```python
GATEncoder(
    embed_dim=128,        # Embedding dimension: 64-256 typical
    num_layers=3,         # Number of GAT layers: 2-4 recommended
    num_heads=8,          # Attention heads: 4-16 typical
    dropout=0.1,          # Dropout: 0.0-0.2
    negative_slope=0.2,   # LeakyReLU slope: 0.1-0.3
    concat_heads=True,    # Concatenate heads (more capacity)
)
```

**Parameter Trade-offs:**

- **embed_dim**: Higher = more expressive, but slower and needs more data
- **num_layers**: More layers = larger receptive field, but risk over-smoothing
- **num_heads**: More heads = diverse attention patterns, but more parameters
- **dropout**: Higher = better regularization, but may underfit

### MLPEncoder

```python
MLPEncoder(
    embed_dim=128,        # Output embedding dimension
    hidden_dim=256,       # Hidden layer dimension: 128-512 typical
    num_layers=3,         # Number of MLP layers: 2-4 recommended
    dropout=0.1,          # Dropout: 0.0-0.2
)
```

**Parameter Trade-offs:**

- **hidden_dim**: Higher = more capacity, but more parameters
- **num_layers**: Deeper = more non-linearity, but harder to train

## Performance Comparison

### Expected Behavior

| Aspect              | GAT                         | MLP                          |
| ------------------- | --------------------------- | ---------------------------- |
| **Expressiveness**  | High (uses graph structure) | Medium (ignores structure)   |
| **Parameters**      | More (attention mechanisms) | Fewer (simple layers)        |
| **Inference Speed** | Slower (message passing)    | Faster (parallel processing) |
| **Training Time**   | Longer                      | Shorter                      |
| **Data Efficiency** | Better on complex problems  | Better on simple problems    |
| **Scalability**     | O(N²) edges                 | O(N) nodes                   |

### Typical Use Cases

**Use GAT when:**

- Problem has strong spatial/graph structure
- Performance is critical over speed
- Training data is abundant
- Node relationships are important

**Use MLP when:**

- Need fast baseline
- Limited computational resources
- Problem structure is simple
- Quick experiments/prototyping

## Implementation Details

### Edge-Aware Attention

The key innovation in GATEncoder is edge-aware attention. Unlike standard GAT that only uses node features, our implementation incorporates edge distances:

```python
# Standard GAT attention (node features only)
α_ij = softmax(LeakyReLU(a^T [W·h_i || W·h_j]))

# Edge-aware attention (our implementation)
α_ij = softmax(LeakyReLU(LeakyReLU(W[h_i || h_j || e_ij])))
```

This allows the attention mechanism to consider both node features and their distances, which is crucial for routing problems.

### Memory Efficiency

For large graphs, consider:

1. **Sparse Graphs**: If using sparse adjacency, modify `_build_edge_index()` to only include k-nearest neighbors
2. **Mini-batching**: Process graphs one at a time if memory is limited
3. **Gradient Checkpointing**: Trade compute for memory using `torch.utils.checkpoint`

### Training Tips

1. **Normalization**: Normalize coordinates to [0, 1] and demands by max capacity
2. **Learning Rate**: Start with 1e-4 for GAT, 3e-4 for MLP
3. **Warm-up**: Use learning rate warm-up for first 1000 steps
4. **Regularization**: Apply dropout and weight decay (1e-5)
5. **Batch Size**: 32-128 for GAT, 64-256 for MLP

## Testing

Run the comprehensive test suite:

```bash
# All encoder tests
pytest tests/test_encoders.py -v

# Specific test class
pytest tests/test_encoders.py::TestGATEncoder -v

# Integration tests
pytest tests/test_encoders.py::TestEncoderIntegration -v
```

## API Reference

### Encoder (Abstract Base Class)

```python
class Encoder(ABC, nn.Module):
    def __init__(self, embed_dim: int)

    @abstractmethod
    def forward(self, graph_data: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]

    def get_embed_dim(self) -> int
```

### GATEncoder

```python
class GATEncoder(Encoder):
    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        concat_heads: bool = True,
    )

    def forward(
        self,
        graph_data: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor]
```

### MLPEncoder

```python
class MLPEncoder(Encoder):
    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    )

    def forward(
        self,
        graph_data: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor]

    def get_num_parameters(self) -> int
```

## References

- **Paper**: "Reinforce-model-Paper.pdf" Section V-A1 (GAT architecture)
- **Graph Attention Networks**: Veličković et al. (2018)
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/

## Future Extensions

Potential improvements:

1. **Sparse Attention**: Only attend to k-nearest neighbors for scalability
2. **Hierarchical Encoding**: Multi-scale representations
3. **Dynamic Graphs**: Handle time-varying graphs
4. **Learned Aggregation**: Replace mean pooling with attention-based aggregation
5. **Transformer Encoder**: Full self-attention over all nodes

---

For questions or issues, please refer to the main project README or open an issue on the repository.
