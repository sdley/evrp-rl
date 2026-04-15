"""
Encoder modules for graph-based state representations in EVRP.

This package provides various encoder architectures for embedding EVRP problem instances:
- Encoder: Abstract base class defining the encoder interface
- GATEncoder: Graph Attention Network with edge-aware attention
- MLPEncoder: Simple MLP baseline for independent node embedding
"""

from .encoder import Encoder
from .gat_encoder import GATEncoder
from .mlp_encoder import MLPEncoder

__all__ = ["Encoder", "GATEncoder", "MLPEncoder"]
