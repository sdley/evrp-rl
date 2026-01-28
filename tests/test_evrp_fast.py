"""
Fast unit tests for EVRP Environment - smoke tests.
"""

import pytest
import numpy as np
from src.env import EVRPEnvironment


class TestEVRPEnvironmentBasic:
    """Basic smoke tests for environment."""
    
    def test_env_creation(self):
        """Test basic environment creation."""
        env = EVRPEnvironment(
            num_customers=3,
            num_chargers=1,
            max_battery=100.0,
            max_cargo=50.0,
        )
        
        assert env.num_customers == 3
        assert env.num_chargers == 1
        assert env.num_nodes == 5
        assert env.max_battery == 100.0
        assert env.max_cargo == 50.0
    
    def test_reset(self):
        """Test environment reset."""
        env = EVRPEnvironment(num_customers=3, num_chargers=1)
        obs, info = env.reset()
        
        assert env.current_node == 0
        assert env.current_battery == env.max_battery
        assert env.current_cargo == 0.0
        assert env.visited_customers == 0
        assert len(obs) > 0
        assert len(info) > 0
    
    def test_step_and_valid_actions(self):
        """Test taking a step with valid action."""
        env = EVRPEnvironment(
            num_customers=3, 
            num_chargers=1,
            max_battery=500.0,
        )
        obs, _ = env.reset()
        
        # Get first valid customer action
        valid_customers = [
            i for i in range(env.customer_start_idx, env.customer_end_idx + 1) 
            if obs["valid_actions_mask"][i]
        ]
        
        assert len(valid_customers) > 0, "Should have valid customer actions"
        
        action = valid_customers[0]
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert env.current_node == action
        assert env.visited_mask[action]
    
    def test_battery_mechanics(self):
        """Test battery depletion and charging."""
        env = EVRPEnvironment(
            num_customers=2,
            num_chargers=1,
            max_battery=500.0,
        )
        obs, _ = env.reset()
        
        initial_battery = env.current_battery
        
        # Move to customer
        valid_customers = [
            i for i in range(env.customer_start_idx, env.customer_end_idx + 1) 
            if obs["valid_actions_mask"][i]
        ]
        
        if valid_customers:
            obs, _, _, _, _ = env.step(valid_customers[0])
            assert env.current_battery < initial_battery
    
    def test_cargo_mechanics(self):
        """Test cargo pickup and dropoff."""
        env = EVRPEnvironment(
            num_customers=2,
            num_chargers=1,
            max_battery=500.0,
        )
        obs, _ = env.reset()
        
        # Move to customer
        valid_customers = [
            i for i in range(env.customer_start_idx, env.customer_end_idx + 1) 
            if obs["valid_actions_mask"][i]
        ]
        
        if valid_customers:
            customer_idx = valid_customers[0]
            demand = env.node_demands[customer_idx]
            
            obs, _, _, _, _ = env.step(customer_idx)
            assert env.current_cargo == demand
            
            # Return to depot
            if obs["valid_actions_mask"][0]:
                obs, _, _, _, _ = env.step(0)
                assert env.current_cargo == 0.0
    
    def test_graph_utilities(self):
        """Test graph utility methods."""
        env = EVRPEnvironment(num_customers=3, num_chargers=1)
        obs, _ = env.reset()
        
        graph = env.get_graph()
        assert graph.number_of_nodes() == 5
        
        coords = env.get_node_coordinates()
        assert coords.shape == (5, 2)
        assert np.allclose(coords[0], [0, 0])
        
        dist_matrix = env.get_distance_matrix()
        assert dist_matrix.shape == (5, 5)
        assert np.allclose(np.diag(dist_matrix), 0)
    
    def test_observation_structure(self):
        """Test observation dictionary structure."""
        env = EVRPEnvironment(num_customers=3, num_chargers=1)
        obs, _ = env.reset()
        
        required_keys = [
            "node_coords", "distance_matrix", "node_demands", "node_types",
            "current_node", "current_battery", "current_cargo",
            "visited_mask", "valid_actions_mask"
        ]
        
        for key in required_keys:
            assert key in obs, f"Missing key: {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
