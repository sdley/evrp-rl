"""
Unit tests for EVRP Environment.
"""

import pytest
import numpy as np
from src.env import EVRPEnvironment


class TestEVRPEnvironmentInitialization:
    """Test environment initialization."""
    
    def test_env_creation(self):
        """Test basic environment creation."""
        env = EVRPEnvironment(
            num_customers=5,
            num_chargers=2,
            max_battery=100.0,
            max_cargo=50.0,
        )
        
        assert env.num_customers == 5
        assert env.num_chargers == 2
        assert env.num_nodes == 8  # depot + 5 customers + 2 chargers
        assert env.max_battery == 100.0
        assert env.max_cargo == 50.0
    
    def test_observation_space(self):
        """Test observation space structure."""
        env = EVRPEnvironment(num_customers=5, num_chargers=2)
        
        obs, _ = env.reset()
        
        assert "node_coords" in obs
        assert "distance_matrix" in obs
        assert "node_demands" in obs
        assert "node_types" in obs
        assert "current_node" in obs
        assert "current_battery" in obs
        assert "current_cargo" in obs
        assert "visited_mask" in obs
        assert "valid_actions_mask" in obs
        
        assert obs["node_coords"].shape == (8, 2)
        assert obs["distance_matrix"].shape == (8, 8)
        assert obs["node_demands"].shape == (8,)
        assert obs["node_types"].shape == (8,)
    
    def test_action_space(self):
        """Test action space."""
        env = EVRPEnvironment(num_customers=5, num_chargers=2)
        
        assert env.action_space.n == 8
        
        # Test valid action sampling
        for _ in range(10):
            action = env.action_space.sample()
            assert 0 <= action < 8


class TestEVRPEnvironmentReset:
    """Test environment reset."""
    
    def test_reset_initial_state(self):
        """Test reset returns valid initial state."""
        env = EVRPEnvironment(num_customers=5, num_chargers=2)
        obs, info = env.reset()
        
        # Check initial position is depot
        assert env.current_node == 0
        
        # Check initial battery is full
        assert env.current_battery == env.max_battery
        
        # Check initial cargo is empty
        assert env.current_cargo == 0.0
        
        # Check no customers visited
        assert env.visited_customers == 0
        assert not env.visited_mask.any()
        
        # Check route starts at depot
        assert env.route == [0]
        
        # Check info structure
        assert "current_node" in info
        assert "current_battery" in info
        assert "current_cargo" in info
        assert "visited_customers" in info


class TestEVRPEnvironmentStep:
    """Test environment step function."""
    
    def test_invalid_action_penalty(self):
        """Test that invalid actions receive penalty."""
        env = EVRPEnvironment(num_customers=5, num_chargers=2)
        obs, _ = env.reset()
        
        # Find a customer and mark it as visited
        env.visited_mask[1] = True
        
        # Try to visit already visited customer (invalid action)
        obs, reward, terminated, truncated, info = env.step(action=1)
        
        # Invalid action should give penalty
        assert reward <= 0
    
    def test_valid_action_depot_to_customer(self):
        """Test valid move from depot to customer."""
        env = EVRPEnvironment(
            num_customers=5, 
            num_chargers=2,
            max_battery=500.0,  # High battery to ensure action is valid
        )
        obs, _ = env.reset()
        
        # Get valid actions
        valid_actions = obs["valid_actions_mask"]
        valid_action_indices = [i for i, v in enumerate(valid_actions) if v]
        
        # Should have some valid actions beyond depot
        valid_customers = [i for i in valid_action_indices if i > 0]
        assert len(valid_customers) > 0, "Should have valid customer actions"
        
        # Move to first valid customer
        action = valid_customers[0]
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert env.current_node == action
        assert env.visited_mask[action]
        assert reward < 0  # Should have distance cost
    
    def test_battery_depletion(self):
        """Test battery depletion over movement."""
        env = EVRPEnvironment(
            num_customers=2,
            num_chargers=1,
            max_battery=500.0,
            energy_consumption_rate=1.0,
        )
        obs, _ = env.reset()
        
        initial_battery = env.current_battery
        
        # Get valid action
        valid_actions = obs["valid_actions_mask"]
        valid_action_indices = [i for i, v in enumerate(valid_actions) if v and i > 0]
        
        if valid_action_indices:
            action = valid_action_indices[0]
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Battery should decrease
            assert env.current_battery < initial_battery
    
    def test_charging_station_battery_refill(self):
        """Test battery refills at charging station."""
        env = EVRPEnvironment(
            num_customers=2,
            num_chargers=1,
            max_battery=500.0,
        )
        obs, _ = env.reset()
        
        # Get valid action to move somewhere
        valid_actions = obs["valid_actions_mask"]
        valid_action_indices = [i for i, v in enumerate(valid_actions) if v and i > 0]
        
        if valid_action_indices:
            action = valid_action_indices[0]
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check if we can reach charger
            valid_actions = obs["valid_actions_mask"]
            charger_idx = env.charger_start_idx
            
            if charger_idx < len(valid_actions) and valid_actions[charger_idx]:
                depleted_battery = env.current_battery
                
                # Move to charger
                obs, reward, terminated, truncated, info = env.step(charger_idx)
                
                # Battery should be fully charged
                assert env.current_battery == env.max_battery
    
    def test_cargo_pickup_and_dropoff(self):
        """Test cargo pickup at customer and dropoff at depot."""
        env = EVRPEnvironment(
            num_customers=3, 
            num_chargers=1,
            max_battery=500.0,  # High battery
        )
        obs, _ = env.reset()
        
        initial_cargo = env.current_cargo
        
        # Get valid customer action
        valid_actions = obs["valid_actions_mask"]
        valid_customers = [i for i in range(env.customer_start_idx, env.customer_end_idx + 1) 
                          if valid_actions[i]]
        
        if valid_customers:
            customer_idx = valid_customers[0]
            customer_demand = env.node_demands[customer_idx]
            
            obs, reward, terminated, truncated, info = env.step(customer_idx)
            
            # Cargo should increase
            assert env.current_cargo == customer_demand
            
            # Return to depot (if valid)
            valid_actions = obs["valid_actions_mask"]
            if valid_actions[0]:
                obs, reward, terminated, truncated, info = env.step(0)
                
                # Cargo should be unloaded
                assert env.current_cargo == 0.0
    
    def test_visited_customers_mask(self):
        """Test visited customers mask updates."""
        env = EVRPEnvironment(
            num_customers=3, 
            num_chargers=1,
            max_battery=500.0,
        )
        obs, _ = env.reset()
        
        # Get valid customer actions
        valid_actions = obs["valid_actions_mask"]
        valid_customers = [i for i in range(env.customer_start_idx, env.customer_end_idx + 1) 
                          if valid_actions[i]]
        
        if len(valid_customers) >= 2:
            # Visit first customer
            customer1 = valid_customers[0]
            obs, reward, terminated, truncated, info = env.step(customer1)
            assert env.visited_mask[customer1]
            
            # Visit second customer
            valid_actions = obs["valid_actions_mask"]
            valid_customers = [i for i in range(env.customer_start_idx, env.customer_end_idx + 1) 
                              if valid_actions[i]]
            if valid_customers:
                customer2 = valid_customers[0]
                obs, reward, terminated, truncated, info = env.step(customer2)
                assert env.visited_mask[customer2]
                
                # Both should be marked as visited
                assert env.visited_mask[customer1] and env.visited_mask[customer2]


class TestEVRPEnvironmentValidity:
    """Test action validity checks."""
    
    def test_cannot_revisit_customer(self):
        """Test that visited customers cannot be revisited."""
        env = EVRPEnvironment(num_customers=3, num_chargers=1)
        obs, _ = env.reset()
        
        # Visit customer 1
        obs, reward, terminated, truncated, info = env.step(action=1)
        
        # Return to depot
        obs, reward, terminated, truncated, info = env.step(action=0)
        
        # Check valid actions: customer 1 should not be valid
        valid_actions = env._get_valid_actions()
        assert not valid_actions[1]
    
    def test_battery_feasibility(self):
        """Test battery feasibility check for actions."""
        env = EVRPEnvironment(
            num_customers=2,
            num_chargers=1,
            max_battery=10.0,  # Very small battery
            energy_consumption_rate=1.0,
        )
        obs, _ = env.reset()
        
        # Deplete battery
        for _ in range(5):
            valid_actions = env._get_valid_actions()
            valid_action = np.where(valid_actions)[0]
            if len(valid_action) > 0:
                obs, reward, terminated, truncated, info = env.step(valid_action[0])
            if terminated:
                break


class TestEVRPEnvironmentRender:
    """Test rendering functionality."""
    
    def test_render_modes(self):
        """Test different render modes."""
        env = EVRPEnvironment(num_customers=3, num_chargers=1, render_mode="human")
        obs, _ = env.reset()
        
        # Should not raise error
        env.render()
        
        env.close()


class TestEVRPEnvironmentGraph:
    """Test graph utilities."""
    
    def test_get_graph(self):
        """Test getting graph representation."""
        env = EVRPEnvironment(num_customers=3, num_chargers=1)
        obs, _ = env.reset()
        
        graph = env.get_graph()
        
        assert graph.number_of_nodes() == 5
        assert graph.number_of_edges() == 10  # Complete graph
    
    def test_get_coordinates(self):
        """Test getting node coordinates."""
        env = EVRPEnvironment(num_customers=3, num_chargers=1)
        obs, _ = env.reset()
        
        coords = env.get_node_coordinates()
        
        assert coords.shape == (5, 2)
        # Depot should be at origin
        assert np.allclose(coords[0], [0, 0])
    
    def test_get_distance_matrix(self):
        """Test getting distance matrix."""
        env = EVRPEnvironment(num_customers=3, num_chargers=1)
        obs, _ = env.reset()
        
        dist_matrix = env.get_distance_matrix()
        
        assert dist_matrix.shape == (5, 5)
        # Diagonal should be zero
        assert np.allclose(np.diag(dist_matrix), 0)
        # Should be symmetric
        assert np.allclose(dist_matrix, dist_matrix.T)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
