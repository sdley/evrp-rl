"""
Comprehensive tests for EVRP encoder modules.

Tests both GAT and MLP encoders using synthetic EVRP instances.
"""

import pytest
import torch
import numpy as np
from evrp_rl.encoders import Encoder, GATEncoder, MLPEncoder
from evrp_rl.env import EVRPEnvironment


class TestEncoderBase:
    """Test abstract Encoder base class."""
    
    def test_encoder_is_abstract(self):
        """Verify that Encoder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Encoder(embed_dim=128)
    
    def test_encoder_inheritance(self):
        """Test that concrete encoders inherit from Encoder."""
        gat = GATEncoder(embed_dim=128)
        mlp = MLPEncoder(embed_dim=128)
        
        assert isinstance(gat, Encoder)
        assert isinstance(mlp, Encoder)
    
    def test_get_embed_dim(self):
        """Test embed_dim accessor."""
        gat = GATEncoder(embed_dim=64)
        mlp = MLPEncoder(embed_dim=128)
        
        assert gat.get_embed_dim() == 64
        assert mlp.get_embed_dim() == 128


class TestGATEncoder:
    """Test GAT encoder implementation."""
    
    @pytest.fixture
    def encoder(self):
        """Create GAT encoder instance."""
        return GATEncoder(
            embed_dim=128,
            num_layers=3,
            num_heads=4,
            dropout=0.1,
        )
    
    @pytest.fixture
    def sample_graph(self):
        """Create synthetic EVRP graph data."""
        batch_size = 2
        num_nodes = 10  # 1 depot + 7 customers + 2 chargers
        
        # Node coordinates (random)
        node_coords = torch.rand(batch_size, num_nodes, 2)
        
        # Node demands (depot and chargers have 0 demand)
        node_demands = torch.rand(batch_size, num_nodes)
        node_demands[:, 0] = 0  # depot
        node_demands[:, 8:] = 0  # chargers
        
        # Node types (one-hot: depot, customer, charger)
        node_types = torch.zeros(batch_size, num_nodes, 3)
        node_types[:, 0, 0] = 1  # depot
        node_types[:, 1:8, 1] = 1  # customers
        node_types[:, 8:, 2] = 1  # chargers
        
        # Distance matrix (Euclidean)
        distance_matrix = torch.cdist(node_coords, node_coords, p=2)
        
        return {
            'node_coords': node_coords,
            'node_demands': node_demands,
            'node_types': node_types,
            'distance_matrix': distance_matrix,
        }
    
    def test_gat_initialization(self, encoder):
        """Test GAT encoder initializes correctly."""
        assert encoder.embed_dim == 128
        assert encoder.num_layers == 3
        assert encoder.num_heads == 4
        assert len(encoder.gat_layers) == 3
        assert len(encoder.layer_norms) == 3
    
    def test_gat_forward(self, encoder, sample_graph):
        """Test GAT forward pass."""
        encoder.eval()
        with torch.no_grad():
            node_embeds, graph_embed = encoder(sample_graph)
        
        batch_size = sample_graph['node_coords'].shape[0]
        num_nodes = sample_graph['node_coords'].shape[1]
        
        # Check output shapes
        assert node_embeds.shape == (batch_size, num_nodes, 128)
        assert graph_embed.shape == (batch_size, 128)
    
    def test_gat_forward_single_instance(self, encoder):
        """Test GAT with single graph instance (batch_size=1)."""
        # Create single instance
        num_nodes = 5
        graph_data = {
            'node_coords': torch.rand(1, num_nodes, 2),
            'node_demands': torch.rand(1, num_nodes),
            'node_types': torch.eye(3).unsqueeze(0).repeat(1, num_nodes // 3 + 1, 1)[:, :num_nodes],
            'distance_matrix': torch.rand(1, num_nodes, num_nodes),
        }
        
        encoder.eval()
        with torch.no_grad():
            node_embeds, graph_embed = encoder(graph_data)
        
        assert node_embeds.shape == (1, num_nodes, 128)
        assert graph_embed.shape == (1, 128)
    
    def test_gat_different_sizes(self, encoder):
        """Test GAT handles different graph sizes."""
        for num_nodes in [5, 10, 20]:
            graph_data = {
                'node_coords': torch.rand(1, num_nodes, 2),
                'node_demands': torch.rand(1, num_nodes),
                'node_types': torch.rand(1, num_nodes, 3),
                'distance_matrix': torch.rand(1, num_nodes, num_nodes),
            }
            
            encoder.eval()
            with torch.no_grad():
                node_embeds, graph_embed = encoder(graph_data)
            
            assert node_embeds.shape == (1, num_nodes, 128)
            assert graph_embed.shape == (1, 128)
    
    def test_gat_gradient_flow(self, encoder, sample_graph):
        """Test that gradients flow through GAT."""
        encoder.train()
        
        node_embeds, graph_embed = encoder(sample_graph)
        
        # Dummy loss
        loss = graph_embed.sum()
        loss.backward()
        
        # Check that gradients exist
        for param in encoder.parameters():
            assert param.grad is not None
    
    def test_gat_deterministic(self, encoder, sample_graph):
        """Test GAT produces deterministic outputs in eval mode."""
        encoder.eval()
        
        with torch.no_grad():
            output1 = encoder(sample_graph)
            output2 = encoder(sample_graph)
        
        assert torch.allclose(output1[0], output2[0], atol=1e-6)
        assert torch.allclose(output1[1], output2[1], atol=1e-6)
    
    def test_gat_with_evrp_env(self, encoder):
        """Test GAT encoder with actual EVRP environment."""
        env = EVRPEnvironment(num_customers=5, num_chargers=2)
        obs, info = env.reset()
        
        # Prepare graph data from observation
        graph_data = {
            'node_coords': torch.from_numpy(obs['node_coords']).unsqueeze(0).float(),
            'node_demands': torch.from_numpy(obs['node_demands']).unsqueeze(0).float(),
            'node_types': torch.from_numpy(obs['node_types']).unsqueeze(0).float(),
            'distance_matrix': torch.from_numpy(obs['distance_matrix']).unsqueeze(0).float(),
        }
        
        encoder.eval()
        with torch.no_grad():
            node_embeds, graph_embed = encoder(graph_data)
        
        num_nodes = obs['node_coords'].shape[0]
        assert node_embeds.shape == (1, num_nodes, 128)
        assert graph_embed.shape == (1, 128)


class TestMLPEncoder:
    """Test MLP encoder implementation."""
    
    @pytest.fixture
    def encoder(self):
        """Create MLP encoder instance."""
        return MLPEncoder(
            embed_dim=128,
            hidden_dim=256,
            num_layers=3,
            dropout=0.1,
        )
    
    @pytest.fixture
    def sample_graph(self):
        """Create synthetic EVRP graph data."""
        batch_size = 2
        num_nodes = 10
        
        return {
            'node_coords': torch.rand(batch_size, num_nodes, 2),
            'node_demands': torch.rand(batch_size, num_nodes),
            'node_types': torch.rand(batch_size, num_nodes, 3),
            'distance_matrix': torch.rand(batch_size, num_nodes, num_nodes),
        }
    
    def test_mlp_initialization(self, encoder):
        """Test MLP encoder initializes correctly."""
        assert encoder.embed_dim == 128
        assert encoder.hidden_dim == 256
        assert encoder.num_layers == 3
    
    def test_mlp_forward(self, encoder, sample_graph):
        """Test MLP forward pass."""
        encoder.eval()
        with torch.no_grad():
            node_embeds, graph_embed = encoder(sample_graph)
        
        batch_size = sample_graph['node_coords'].shape[0]
        num_nodes = sample_graph['node_coords'].shape[1]
        
        # Check output shapes
        assert node_embeds.shape == (batch_size, num_nodes, 128)
        assert graph_embed.shape == (batch_size, 128)
    
    def test_mlp_different_sizes(self, encoder):
        """Test MLP handles different graph sizes."""
        for num_nodes in [5, 10, 20]:
            graph_data = {
                'node_coords': torch.rand(1, num_nodes, 2),
                'node_demands': torch.rand(1, num_nodes),
                'node_types': torch.rand(1, num_nodes, 3),
                'distance_matrix': torch.rand(1, num_nodes, num_nodes),
            }
            
            encoder.eval()
            with torch.no_grad():
                node_embeds, graph_embed = encoder(graph_data)
            
            assert node_embeds.shape == (1, num_nodes, 128)
            assert graph_embed.shape == (1, 128)
    
    def test_mlp_gradient_flow(self, encoder, sample_graph):
        """Test that gradients flow through MLP."""
        encoder.train()
        
        node_embeds, graph_embed = encoder(sample_graph)
        
        # Dummy loss
        loss = graph_embed.sum()
        loss.backward()
        
        # Check that gradients exist
        for param in encoder.parameters():
            assert param.grad is not None
    
    def test_mlp_with_evrp_env(self, encoder):
        """Test MLP encoder with actual EVRP environment."""
        env = EVRPEnvironment(num_customers=5, num_chargers=2)
        obs, info = env.reset()
        
        # Prepare graph data from observation
        graph_data = {
            'node_coords': torch.from_numpy(obs['node_coords']).unsqueeze(0).float(),
            'node_demands': torch.from_numpy(obs['node_demands']).unsqueeze(0).float(),
            'node_types': torch.from_numpy(obs['node_types']).unsqueeze(0).float(),
            'distance_matrix': torch.from_numpy(obs['distance_matrix']).unsqueeze(0).float(),
        }
        
        encoder.eval()
        with torch.no_grad():
            node_embeds, graph_embed = encoder(graph_data)
        
        num_nodes = obs['node_coords'].shape[0]
        assert node_embeds.shape == (1, num_nodes, 128)
        assert graph_embed.shape == (1, 128)
    
    def test_mlp_num_parameters(self, encoder):
        """Test parameter counting."""
        num_params = encoder.get_num_parameters()
        assert num_params > 0
        
        # Verify count is correct
        expected = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        assert num_params == expected


class TestEncoderComparison:
    """Compare GAT and MLP encoders."""
    
    @pytest.fixture
    def sample_graph(self):
        """Create synthetic EVRP graph data."""
        return {
            'node_coords': torch.rand(2, 10, 2),
            'node_demands': torch.rand(2, 10),
            'node_types': torch.rand(2, 10, 3),
            'distance_matrix': torch.rand(2, 10, 10),
        }
    
    def test_same_output_shape(self, sample_graph):
        """Verify both encoders produce same output shape."""
        gat = GATEncoder(embed_dim=128)
        mlp = MLPEncoder(embed_dim=128)
        
        gat.eval()
        mlp.eval()
        
        with torch.no_grad():
            gat_out = gat(sample_graph)
            mlp_out = mlp(sample_graph)
        
        assert gat_out[0].shape == mlp_out[0].shape
        assert gat_out[1].shape == mlp_out[1].shape
    
    def test_different_outputs(self, sample_graph):
        """Verify encoders produce different embeddings (as expected)."""
        gat = GATEncoder(embed_dim=128)
        mlp = MLPEncoder(embed_dim=128)
        
        gat.eval()
        mlp.eval()
        
        with torch.no_grad():
            gat_out = gat(sample_graph)
            mlp_out = mlp(sample_graph)
        
        # Outputs should be different due to different architectures
        assert not torch.allclose(gat_out[0], mlp_out[0])
        assert not torch.allclose(gat_out[1], mlp_out[1])
    
    def test_gat_more_parameters(self):
        """Verify GAT has more parameters than MLP (due to attention)."""
        gat = GATEncoder(embed_dim=128, num_layers=3, num_heads=4)
        mlp = MLPEncoder(embed_dim=128, num_layers=3)
        
        gat_params = sum(p.numel() for p in gat.parameters())
        mlp_params = sum(p.numel() for p in mlp.parameters())
        
        # GAT should have more parameters due to attention mechanisms
        assert gat_params > mlp_params


class TestEncoderIntegration:
    """Integration tests with EVRP environment."""
    
    def test_encode_evrp_episode(self):
        """Test encoding an entire EVRP episode."""
        env = EVRPEnvironment(num_customers=5, num_chargers=2)
        encoder = GATEncoder(embed_dim=64)
        encoder.eval()
        
        obs, info = env.reset()
        embeddings_list = []
        
        for step in range(10):
            # Prepare graph data
            graph_data = {
                'node_coords': torch.from_numpy(obs['node_coords']).unsqueeze(0).float(),
                'node_demands': torch.from_numpy(obs['node_demands']).unsqueeze(0).float(),
                'node_types': torch.from_numpy(obs['node_types']).unsqueeze(0).float(),
                'distance_matrix': torch.from_numpy(obs['distance_matrix']).unsqueeze(0).float(),
            }
            
            with torch.no_grad():
                _, graph_embed = encoder(graph_data)
            
            embeddings_list.append(graph_embed)
            
            # Take random action
            valid_actions = np.where(obs['valid_actions_mask'])[0]
            if len(valid_actions) == 0:
                break
            action = np.random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        # Should have collected some embeddings
        assert len(embeddings_list) > 0
        
        # All embeddings should have same shape
        for embed in embeddings_list:
            assert embed.shape == (1, 64)
    
    def test_batch_processing(self):
        """Test encoding multiple EVRP instances in batch."""
        batch_size = 4
        num_customers = 5
        num_chargers = 2
        
        encoder = MLPEncoder(embed_dim=64)
        encoder.eval()
        
        # Create multiple environments and collect observations
        envs = [EVRPEnvironment(num_customers=num_customers, num_chargers=num_chargers) 
                for _ in range(batch_size)]
        
        observations = [env.reset()[0] for env in envs]
        
        # Stack into batch
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
        
        with torch.no_grad():
            node_embeds, graph_embeds = encoder(graph_data)
        
        num_nodes = num_customers + num_chargers + 1  # +1 for depot
        assert node_embeds.shape == (batch_size, num_nodes, 64)
        assert graph_embeds.shape == (batch_size, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
