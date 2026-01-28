"""
Abstract base class for EVRP graph encoders.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import torch
import torch.nn as nn


class Encoder(ABC, nn.Module):
    """
    Abstract base class for graph encoders in EVRP.
    
    All encoder implementations should inherit from this class and implement
    the forward method to encode graph-structured EVRP instances into
    embeddings suitable for downstream RL agents.
    
    The encoder takes a graph representation (nodes with features, edges with distances)
    and produces:
    1. Node embeddings: Feature vectors for each node in the graph
    2. Graph embedding: A single vector representing the entire graph
    
    Input Format:
        The forward method expects a dictionary with the following keys:
        - 'node_coords': Tensor of shape (batch, num_nodes, 2) - x, y coordinates
        - 'node_demands': Tensor of shape (batch, num_nodes) - demand at each node
        - 'node_types': Tensor of shape (batch, num_nodes, 3) - one-hot [depot, customer, charger]
        - 'distance_matrix': Tensor of shape (batch, num_nodes, num_nodes) - pairwise distances
        - 'edge_index': Optional - Tensor of shape (2, num_edges) for sparse graphs
        
    Output Format:
        Returns a tuple (node_embeddings, graph_embedding) where:
        - node_embeddings: Tensor of shape (batch, num_nodes, embed_dim)
        - graph_embedding: Tensor of shape (batch, embed_dim)
    """
    
    def __init__(self, embed_dim: int):
        """
        Initialize the encoder.
        
        Args:
            embed_dim: Dimensionality of the output embeddings
        """
        super().__init__()
        self.embed_dim = embed_dim
    
    @abstractmethod
    def forward(self, graph_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a graph-structured EVRP instance.
        
        Args:
            graph_data: Dictionary containing graph information with keys:
                - 'node_coords': (batch, num_nodes, 2)
                - 'node_demands': (batch, num_nodes)
                - 'node_types': (batch, num_nodes, 3)
                - 'distance_matrix': (batch, num_nodes, num_nodes)
                - 'edge_index': Optional (2, num_edges)
        
        Returns:
            Tuple of (node_embeddings, graph_embedding):
                - node_embeddings: (batch, num_nodes, embed_dim)
                - graph_embedding: (batch, embed_dim)
        """
        pass
    
    def get_embed_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embed_dim
