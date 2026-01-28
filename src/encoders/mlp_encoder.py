"""
Simple MLP encoder baseline for EVRP.

Embeds nodes independently without graph structure, serving as a baseline
to compare against graph-based methods like GAT.
"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder


class MLPEncoder(Encoder):
    """
    MLP-based encoder for EVRP.
    
    This is a simple baseline that encodes each node independently using
    a multi-layer perceptron, without considering graph structure or
    relationships between nodes. Each node is embedded based solely on
    its own features (coordinates, demand, type).
    
    Comparison to GAT:
    ------------------
    Unlike the Graph Attention Network (GATEncoder), this encoder:
    
    1. **No Graph Structure**: Does not use edge information or node relationships.
       GAT leverages attention over neighbors to capture spatial relationships.
    
    2. **Independent Embeddings**: Each node is processed in isolation.
       GAT aggregates information from neighboring nodes through attention.
    
    3. **No Attention Mechanism**: Simple feed-forward processing.
       GAT uses edge-aware attention: alpha_ij = LeakyReLU(LeakyReLU(W[x_i||x_j||e_ij]))
    
    4. **Simpler & Faster**: Fewer parameters and faster computation.
       GAT is more expressive but computationally expensive.
    
    5. **Limited Expressiveness**: Cannot capture complex spatial patterns.
       GAT can learn adaptive attention based on problem structure.
    
    When to Use MLP vs GAT:
    -----------------------
    - MLP: Fast baseline, small problems, when graph structure is less important
    - GAT: Better performance expected on structured problems where spatial
           relationships matter (typical for routing problems like EVRP)
    
    Architecture:
    1. Input projection: Maps 6D node features to hidden dimension
    2. Hidden layers: Multiple MLP layers with ReLU activation
    3. Output projection: Maps to final embedding dimension
    4. Graph embedding: Mean pooling over all node embeddings
    
    Node Features:
    - Coordinates (x, y): 2D
    - Demand: 1D
    - Node type (depot/customer/charger): 3D one-hot
    Total: 6D input features
    
    Args:
        embed_dim: Dimension of output embeddings
        hidden_dim: Hidden layer dimension
        num_layers: Number of MLP layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """Initialize MLP encoder."""
        super().__init__(embed_dim)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Node feature dimension: coords (2) + demand (1) + type (3) = 6
        self.node_feat_dim = 6
        
        # Build MLP layers with proper initialization
        layers = []
        
        # Input layer with proper initialization
        input_layer = nn.Linear(self.node_feat_dim, hidden_dim)
        nn.init.orthogonal_(input_layer.weight, gain=np.sqrt(2))  # Adjust gain
        nn.init.zeros_(input_layer.bias)
        layers.append(input_layer)
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            hidden_layer = nn.Linear(hidden_dim, hidden_dim)
            nn.init.orthogonal_(hidden_layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(hidden_layer.bias)
            layers.append(hidden_layer)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer with careful initialization
        output_layer = nn.Linear(hidden_dim, embed_dim)
        nn.init.orthogonal_(output_layer.weight, gain=1.0)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        
        self.mlp = nn.Sequential(*layers)
        
        # Optional: Learned aggregation weights for graph embedding
        self.use_learned_pooling = False
        if self.use_learned_pooling:
            self.pool_weights = nn.Linear(embed_dim, 1)
    
    def _prepare_node_features(
        self,
        node_coords: torch.Tensor,
        node_demands: torch.Tensor,
        node_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenate and normalize node features into a single tensor.
        
        Args:
            node_coords: (batch, num_nodes, 2)
            node_demands: (batch, num_nodes) or (batch, num_nodes, 1)
            node_types: (batch, num_nodes, 3) or (batch, num_nodes) with integer labels
        
        Returns:
            Node features (batch, num_nodes, 6)
        """
        # Normalize coordinates to [0, 1] range
        # Use min-max normalization per batch to handle varying scales
        coords_min = node_coords.min()
        coords_max = node_coords.max()
        if coords_max > coords_min:
            node_coords_norm = (node_coords - coords_min) / (coords_max - coords_min + 1e-8)
        else:
            node_coords_norm = torch.zeros_like(node_coords)
        
        # Ensure demands have correct shape
        if node_demands.dim() == 2:
            node_demands = node_demands.unsqueeze(-1)
        
        # Normalize demands to [0, 1] range
        demand_max = node_demands.max()
        if demand_max > 0:
            node_demands_norm = node_demands / (demand_max + 1e-8)
        else:
            node_demands_norm = torch.zeros_like(node_demands)
        
        # Convert node_types to one-hot if needed (from integer labels)
        if node_types.dim() == 2 and node_types.shape[-1] != 3:
            # Integer labels: convert to one-hot
            batch_size, num_nodes = node_types.shape
            node_types_onehot = torch.zeros(batch_size, num_nodes, 3, device=node_types.device, dtype=node_coords.dtype)
            for i in range(3):
                node_types_onehot[:, :, i] = (node_types == i).float()
            node_types = node_types_onehot
        elif node_types.dim() == 2:
            # (batch, num_nodes) -> assume need one-hot encoding
            batch_size, num_nodes = node_types.shape
            node_types_onehot = torch.zeros(batch_size, num_nodes, 3, device=node_types.device, dtype=node_coords.dtype)
            node_types_long = node_types.long()
            for i in range(3):
                node_types_onehot[:, :, i] = (node_types_long == i).float()
            node_types = node_types_onehot
        
        # Concatenate all features
        node_features = torch.cat([
            node_coords_norm,     # (batch, num_nodes, 2)
            node_demands_norm,    # (batch, num_nodes, 1)
            node_types,           # (batch, num_nodes, 3)
        ], dim=-1)
        
        return node_features
    
    def forward(
        self,
        graph_data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode EVRP graph using MLP.
        
        Args:
            graph_data: Dictionary with keys:
                - 'node_coords': (batch, num_nodes, 2)
                - 'node_demands': (batch, num_nodes)
                - 'node_types': (batch, num_nodes, 3)
                - 'distance_matrix': (batch, num_nodes, num_nodes) [not used]
        
        Returns:
            Tuple of (node_embeddings, graph_embedding):
                - node_embeddings: (batch, num_nodes, embed_dim)
                - graph_embedding: (batch, embed_dim)
        """
        node_coords = graph_data['node_coords']
        node_demands = graph_data['node_demands']
        node_types = graph_data['node_types']
        
        batch_size, num_nodes = node_coords.shape[:2]
        
        # Prepare node features (batch, num_nodes, 6)
        node_features = self._prepare_node_features(node_coords, node_demands, node_types)
        
        # Apply MLP to each node independently
        # Reshape: (batch * num_nodes, 6)
        x = node_features.view(-1, self.node_feat_dim)
        
        # Forward through MLP
        x = self.mlp(x)
        
        # Reshape back: (batch, num_nodes, embed_dim)
        node_embeddings = x.view(batch_size, num_nodes, self.embed_dim)
        
        # Graph-level embedding via pooling
        if self.use_learned_pooling:
            # Learned attention-based pooling
            pool_logits = self.pool_weights(node_embeddings).squeeze(-1)  # (batch, num_nodes)
            pool_weights = F.softmax(pool_logits, dim=1).unsqueeze(-1)   # (batch, num_nodes, 1)
            graph_embedding = (node_embeddings * pool_weights).sum(dim=1) # (batch, embed_dim)
        else:
            # Simple mean pooling
            graph_embedding = node_embeddings.mean(dim=1)
        
        return node_embeddings, graph_embedding
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
