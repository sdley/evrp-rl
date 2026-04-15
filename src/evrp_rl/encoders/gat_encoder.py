"""
Graph Attention Network (GAT) encoder for EVRP.

Implementation based on Reinforce-model-Paper.pdf Section V-A1.
Uses edge-aware attention mechanism to encode EVRP problem instances.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from .encoder import Encoder


class EdgeAwareGATConv(MessagePassing):
    """
    Edge-aware Graph Attention layer.
    
    Implements attention mechanism with edge features:
    alpha_ij = LeakyReLU(LeakyReLU(W[x_i || x_j || e_ij]))
    
    This differs from standard GAT by incorporating edge features (distances)
    directly into the attention computation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """
        Initialize Edge-aware GAT layer.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension per head
            edge_dim: Edge feature dimension (typically 1 for distance)
            heads: Number of attention heads
            concat: If True, concatenate head outputs; else average
            negative_slope: LeakyReLU negative slope
            dropout: Dropout probability for attention weights
            bias: Whether to use bias in linear layers
        """
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        # Linear transformations for node features
        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Linear transformation for edge features
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        
        # Attention mechanism: W[x_i || x_j || e_ij]
        # Total input dim: (out_channels * 3) per head
        self.att = nn.Parameter(torch.Tensor(1, heads, 3 * out_channels))
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of edge-aware GAT layer.
        
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge features (num_edges, edge_dim)
        
        Returns:
            Updated node features (num_nodes, heads * out_channels) if concat
                                or (num_nodes, out_channels) if not concat
        """
        # Transform node features
        x_src = self.lin_src(x).view(-1, self.heads, self.out_channels)
        x_dst = self.lin_dst(x).view(-1, self.heads, self.out_channels)
        
        # Transform edge features
        edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        
        # Propagate messages
        out = self.propagate(
            edge_index,
            x=(x_src, x_dst),
            edge_attr=edge_attr,
        )
        
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index_i: torch.Tensor,
        size_i: Optional[int],
    ) -> torch.Tensor:
        """
        Compute messages with edge-aware attention.
        
        Args:
            x_i: Target node features (num_edges, heads, out_channels)
            x_j: Source node features (num_edges, heads, out_channels)
            edge_attr: Edge features (num_edges, heads, out_channels)
            edge_index_i: Target node indices
            size_i: Number of target nodes
        
        Returns:
            Attended messages (num_edges, heads, out_channels)
        """
        # Concatenate node and edge features: [x_i || x_j || e_ij]
        concat_features = torch.cat([x_i, x_j, edge_attr], dim=-1)
        
        # Apply LeakyReLU twice as per specification
        # alpha_ij = LeakyReLU(LeakyReLU(W[x_i||x_j||e_ij]))
        alpha = F.leaky_relu(concat_features, self.negative_slope)
        alpha = (alpha * self.att).sum(dim=-1)  # (num_edges, heads)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Softmax over all edges connected to same target node
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        
        # Apply dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Multiply attention weights with source features
        return x_j * alpha.unsqueeze(-1)


class GATEncoder(Encoder):
    """
    Graph Attention Network (GAT) encoder for EVRP.
    
    Encodes EVRP problem instances using L stacked GAT layers with edge-aware
    attention. The attention mechanism incorporates both node features and edge
    distances to compute adaptive attention weights.
    
    Architecture:
    1. Input projection: Maps node features to initial embeddings
    2. L GAT layers: Apply edge-aware attention with multi-head mechanism
    3. Output: Node embeddings + graph embedding (mean pooling)
    
    Node Features:
    - Coordinates (x, y): 2D
    - Demand: 1D
    - Node type (depot/customer/charger): 3D one-hot
    Total: 6D input features
    
    Edge Features:
    - Euclidean distance: 1D
    
    Attention Mechanism:
    For each edge (i, j), compute attention as:
        alpha_ij = LeakyReLU(LeakyReLU(W[h_i || h_j || e_ij]))
    where h_i, h_j are node embeddings and e_ij is the edge distance.
    
    Args:
        embed_dim: Dimension of output embeddings (dx in paper)
        num_layers: Number of GAT layers (L in paper)
        num_heads: Number of attention heads per layer
        dropout: Dropout probability
        negative_slope: LeakyReLU negative slope
        concat_heads: If True, concatenate head outputs; else average
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        concat_heads: bool = True,
    ):
        """Initialize GAT encoder."""
        super().__init__(embed_dim)
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.concat_heads = concat_heads
        
        # Node feature dimension: coords (2) + demand (1) + type (3) = 6
        self.node_feat_dim = 6
        # Edge feature dimension: distance (1)
        self.edge_feat_dim = 1
        
        # Input projection
        self.input_proj = nn.Linear(self.node_feat_dim, embed_dim)
        
        # Build GAT layers
        self.gat_layers = nn.ModuleList()
        
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                # First layer
                in_dim = embed_dim
            else:
                # Subsequent layers
                if concat_heads:
                    in_dim = embed_dim * num_heads
                else:
                    in_dim = embed_dim
            
            # Last layer should not concatenate heads
            concat = concat_heads if layer_idx < num_layers - 1 else False
            
            # Output dimension per head
            if layer_idx == num_layers - 1:
                # Last layer outputs final embed_dim
                out_dim = embed_dim
            else:
                out_dim = embed_dim
            
            self.gat_layers.append(
                EdgeAwareGATConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    edge_dim=self.edge_feat_dim,
                    heads=num_heads,
                    concat=concat,
                    negative_slope=negative_slope,
                    dropout=dropout,
                )
            )
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim * num_heads if (concat_heads and i < num_layers - 1) 
                        else embed_dim)
            for i in range(num_layers)
        ])
    
    def _prepare_node_features(
        self,
        node_coords: torch.Tensor,
        node_demands: torch.Tensor,
        node_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenate node features into a single tensor.
        
        Args:
            node_coords: (batch, num_nodes, 2)
            node_demands: (batch, num_nodes) or (batch, num_nodes, 1)
            node_types: (batch, num_nodes, 3) or (batch, num_nodes) with integer labels
        
        Returns:
            Node features (batch, num_nodes, 6)
        """
        # Ensure demands have correct shape
        if node_demands.dim() == 2:
            node_demands = node_demands.unsqueeze(-1)
        
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
            node_coords,      # (batch, num_nodes, 2)
            node_demands,     # (batch, num_nodes, 1)
            node_types,       # (batch, num_nodes, 3)
        ], dim=-1)
        
        return node_features
    
    def _build_edge_index(
        self,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build fully connected edge index for batch of graphs.
        
        Args:
            batch_size: Number of graphs in batch
            num_nodes: Number of nodes per graph
            device: Device to place tensor on
        
        Returns:
            Edge index (2, batch_size * num_nodes * num_nodes)
        """
        # Create edge index for single graph (fully connected)
        edge_index_single = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # No self-loops
                    edge_index_single.append([i, j])
        
        edge_index_single = torch.tensor(edge_index_single, dtype=torch.long).t()
        num_edges_per_graph = edge_index_single.size(1)
        
        # Replicate for batch
        edge_indices = []
        for batch_idx in range(batch_size):
            offset = batch_idx * num_nodes
            edge_indices.append(edge_index_single + offset)
        
        edge_index = torch.cat(edge_indices, dim=1).to(device)
        
        return edge_index, num_edges_per_graph
    
    def forward(
        self,
        graph_data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode EVRP graph using GAT layers.
        
        Args:
            graph_data: Dictionary with keys:
                - 'node_coords': (batch, num_nodes, 2)
                - 'node_demands': (batch, num_nodes)
                - 'node_types': (batch, num_nodes, 3)
                - 'distance_matrix': (batch, num_nodes, num_nodes)
        
        Returns:
            Tuple of (node_embeddings, graph_embedding):
                - node_embeddings: (batch, num_nodes, embed_dim)
                - graph_embedding: (batch, embed_dim)
        """
        node_coords = graph_data['node_coords']
        node_demands = graph_data['node_demands']
        node_types = graph_data['node_types']
        distance_matrix = graph_data['distance_matrix']
        
        batch_size, num_nodes = node_coords.shape[:2]
        device = node_coords.device
        
        # Prepare node features (batch, num_nodes, 6)
        node_features = self._prepare_node_features(node_coords, node_demands, node_types)
        
        # Flatten batch dimension: (batch * num_nodes, 6)
        x = node_features.view(-1, self.node_feat_dim)
        
        # Project to embedding space
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Build edge index and extract edge features
        edge_index, num_edges_per_graph = self._build_edge_index(
            batch_size, num_nodes, device
        )
        
        # Extract edge attributes from distance matrix
        # (batch, num_nodes, num_nodes) -> (batch * num_edges, 1)
        edge_attr_list = []
        for batch_idx in range(batch_size):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edge_attr_list.append(distance_matrix[batch_idx, i, j])
        
        edge_attr = torch.stack(edge_attr_list).unsqueeze(-1)  # (num_edges, 1)
        
        # Apply GAT layers
        for layer_idx, (gat_layer, layer_norm) in enumerate(
            zip(self.gat_layers, self.layer_norms)
        ):
            x_residual = x
            x = gat_layer(x, edge_index, edge_attr)
            x = layer_norm(x)
            
            # Residual connection (if dimensions match)
            if x_residual.size(-1) == x.size(-1):
                x = x + x_residual
            
            # Apply activation and dropout (except last layer)
            if layer_idx < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Reshape back to (batch, num_nodes, embed_dim)
        node_embeddings = x.view(batch_size, num_nodes, -1)
        
        # Graph-level embedding via mean pooling
        graph_embedding = node_embeddings.mean(dim=1)
        
        return node_embeddings, graph_embedding
