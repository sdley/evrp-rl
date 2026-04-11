"""
Gymnasium-compatible Electric Vehicle Routing Problem (EVRP) Environment.

This environment models the EVRP with battery constraints, charging stations,
time windows, and cargo capacity. The agent controls a fleet of electric vehicles
to serve customers while optimizing route efficiency.

Reference:
    - Modular Reinforcement Learning Framework for Electric Vehicle Routing Problems
    - Reinforce-model-Paper.pdf Section III
"""

import numpy as np
import networkx as nx
from gymnasium import Env, spaces
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import warnings


class EVRPEnvironment(Env):
    """
    Gymnasium-compatible EVRP environment with battery and charging constraints.
    
    Node Structure:
        - Node 0: Depot (start/end point)
        - Nodes 1 to m: Customers
        - Nodes m+1 to g: Charging stations
    
    State Space:
        - Static graph features: node coordinates, distances, demands, types
        - Dynamic state: remaining battery, cargo, visited mask
    
    Action Space:
        - Discrete: select next node (customer/charger/depot)
    
    Reward Structure:
        - Distance cost: -distance traveled
        - Charging penalty: -num_charger_visits
        - Depot penalty: -num_depot_revisits
        - Infeasibility penalty: -max(0, negative_battery)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        num_customers: int = 10,
        num_chargers: int = 3,
        max_battery: float = 100.0,
        max_cargo: float = 100.0,
        energy_consumption_rate: float = 1.0,
        charger_cost: float = 0.5,
        depot_revisit_cost: float = 1.0,
        time_limit: int = 100,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the EVRP environment.
        
        Args:
            num_customers: Number of customers to serve
            num_chargers: Number of charging stations
            max_battery: Maximum battery capacity (Wh)
            max_cargo: Maximum cargo capacity (kg)
            energy_consumption_rate: Energy per unit distance (Wh/km)
            charger_cost: Cost coefficient for visiting charging stations
            depot_revisit_cost: Cost coefficient for revisiting depot
            time_limit: Maximum number of steps per episode
            seed: Random seed for reproducibility
            render_mode: Rendering mode ('human' or 'rgb_array')
        """
        super().__init__()
        
        # Problem parameters
        self.num_customers = num_customers
        self.num_chargers = num_chargers
        self.num_nodes = 1 + num_customers + num_chargers  # depot + customers + chargers
        self.max_battery = max_battery
        self.max_cargo = max_cargo
        self.energy_consumption_rate = energy_consumption_rate
        self.charger_cost = charger_cost
        self.depot_revisit_cost = depot_revisit_cost
        self.time_limit = time_limit
        self.render_mode = render_mode
        
        # Node indices
        self.depot_idx = 0
        self.customer_start_idx = 1
        self.customer_end_idx = num_customers
        self.charger_start_idx = num_customers + 1
        self.charger_end_idx = num_customers + num_chargers
        
        # Initialize random state
        self.rng = np.random.RandomState(seed)
        
        # Graph representation
        self.graph = None
        self.node_coords = None
        self.node_demands = None
        self.node_types = None  # 0: depot, 1: customer, 2: charger
        self.distance_matrix = None
        
        # Episode state
        self.current_node = None
        self.current_battery = None
        self.current_cargo = None
        self.visited_customers = None
        self.visited_mask = None
        self.depot_visits = None
        self.charger_visits = None
        self.current_step = None
        self.route = None
        self.total_distance = None
        self.infeasibility_penalty = None
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(self.num_nodes)
        
        # Observation space: dictionary with static and dynamic components
        self.observation_space = spaces.Dict({
            "node_coords": spaces.Box(
                low=-1000, high=1000, shape=(self.num_nodes, 2), dtype=np.float32
            ),
            "distance_matrix": spaces.Box(
                low=0, high=10000, shape=(self.num_nodes, self.num_nodes), dtype=np.float32
            ),
            "node_demands": spaces.Box(
                low=0, high=max_cargo, shape=(self.num_nodes,), dtype=np.float32
            ),
            "node_types": spaces.Box(
                low=0, high=2, shape=(self.num_nodes,), dtype=np.int32
            ),
            "current_node": spaces.Discrete(self.num_nodes),
            "current_battery": spaces.Box(low=0, high=max_battery, shape=(1,), dtype=np.float32),
            "current_cargo": spaces.Box(low=0, high=max_cargo, shape=(1,), dtype=np.float32),
            "visited_mask": spaces.MultiBinary(self.num_nodes),
            "valid_actions_mask": spaces.MultiBinary(self.num_nodes),
        })
        
        # Create initial graph
        self._generate_problem()
    
    def _generate_problem(self):
        """Generate a random EVRP instance."""
        # Create nodes with random positions in [0, 100]
        self.node_coords = self.rng.uniform(0, 100, size=(self.num_nodes, 2))
        
        # Set depot at origin
        self.node_coords[self.depot_idx] = [0, 0]
        
        # Initialize node properties
        self.node_demands = np.zeros(self.num_nodes, dtype=np.float32)
        self.node_types = np.zeros(self.num_nodes, dtype=np.int32)
        
        # Set node types: 0=depot, 1=customer, 2=charger
        self.node_types[self.depot_idx] = 0
        self.node_types[self.customer_start_idx:self.customer_end_idx + 1] = 1
        self.node_types[self.charger_start_idx:self.charger_end_idx + 1] = 2
        
        # Random demands for customers (0 for depot and chargers)
        self.node_demands[self.customer_start_idx:self.customer_end_idx + 1] = (
            self.rng.uniform(5, 30, size=self.num_customers)
        )
        
        # Compute distance matrix (Euclidean)
        self._compute_distance_matrix()
        
        # Create NetworkX graph
        self.graph = self._create_graph()
    
    def _compute_distance_matrix(self):
        """Compute Euclidean distance matrix between all nodes."""
        n = self.num_nodes
        self.distance_matrix = np.zeros((n, n), dtype=np.float32)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(self.node_coords[i] - self.node_coords[j])
                    self.distance_matrix[i, j] = dist
    
    def _create_graph(self) -> nx.Graph:
        """Create NetworkX graph representation of the problem."""
        graph = nx.complete_graph(self.num_nodes)
        
        # Add node attributes
        for node in graph.nodes():
            graph.nodes[node]["coords"] = self.node_coords[node]
            graph.nodes[node]["demand"] = self.node_demands[node]
            graph.nodes[node]["type"] = self.node_types[node]
        
        # Add edge weights (distances)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    graph[i][j]["distance"] = self.distance_matrix[i, j]
        
        return graph
    
    def _compute_energy_cost(self, from_node: int, to_node: int) -> float:
        """
        Compute energy cost for traveling from one node to another.
        
        Energy cost: f(i, j) = euclidean_distance(i, j) * energy_consumption_rate
        
        Args:
            from_node: Source node index
            to_node: Destination node index
            
        Returns:
            Energy cost in Wh
        """
        distance = self.distance_matrix[from_node, to_node]
        return distance * self.energy_consumption_rate
    
    def _get_valid_actions(self) -> np.ndarray:
        """
        Compute valid action mask based on current state.
        
        Invalid actions:
        - Already visited customers
        - Nodes where battery insufficient to reach + return to depot
        - Infeasible cargo (would exceed max_cargo)
        
        Returns:
            Boolean mask of shape (num_nodes,)
        """
        valid_actions = np.ones(self.num_nodes, dtype=bool)
        
        # Cannot revisit visited customers
        for customer_idx in range(self.customer_start_idx, self.customer_end_idx + 1):
            if self.visited_mask[customer_idx]:
                valid_actions[customer_idx] = False
        
        # Check battery feasibility: need enough battery to reach node + return to depot
        for node_idx in range(self.num_nodes):
            if not valid_actions[node_idx]:
                continue
            
            # Energy to go to node + energy to return to depot
            energy_required = (
                self._compute_energy_cost(self.current_node, node_idx) +
                self._compute_energy_cost(node_idx, self.depot_idx)
            )
            
            if energy_required > self.current_battery:
                valid_actions[node_idx] = False
        
        # Check cargo feasibility for customer visits
        for customer_idx in range(self.customer_start_idx, self.customer_end_idx + 1):
            if not valid_actions[customer_idx]:
                continue
            
            demand = self.node_demands[customer_idx]
            if self.current_cargo + demand > self.max_cargo:
                valid_actions[customer_idx] = False
        
        # CRITICAL FIX: If all customers visited, ONLY depot should be valid
        # This prevents agents from getting stuck in charging loops
        if self.visited_customers == self.num_customers:
            valid_actions[:] = False
            valid_actions[self.depot_idx] = True

        # Avoid trivial self-loops on depot/charger unless there is no alternative.
        # This removes a frequent failure mode where the policy keeps selecting the
        # current charging station (or depot) for many consecutive steps.
        if self._is_depot(self.current_node) or self._is_charger(self.current_node):
            if valid_actions[self.current_node]:
                alternate_actions = valid_actions.copy()
                alternate_actions[self.current_node] = False
                if alternate_actions.any():
                    valid_actions[self.current_node] = False
        
        # Allow depot as fallback if at least one valid action exists
        # (ensures agent can always return to depot)
        if not valid_actions.any():
            valid_actions[self.depot_idx] = True
        
        return valid_actions.astype(np.int32)
    
    def _update_state(self, next_node: int):
        """
        Update environment state after taking action.
        
        Updates:
        - Current battery: b_{t+1} = B if charger, else b_t - f(t, t+1)
        - Current cargo: q_{t+1} = Q if depot, else q_t - d_t if customer
        - Visited mask: mark customer as visited
        - Route and distance tracking
        """
        # Compute energy cost for movement
        energy_cost = self._compute_energy_cost(self.current_node, next_node)
        
        # Update battery
        if self._is_charger(next_node):
            self.current_battery = self.max_battery  # Full charge at charger
            self.charger_visits += 1
        else:
            self.current_battery -= energy_cost
        
        # Update cargo
        if self._is_depot(next_node):
            self.current_cargo = 0  # Unload cargo at depot
            self.depot_visits += 1
        elif self._is_customer(next_node):
            demand = self.node_demands[next_node]
            self.current_cargo += demand  # Load customer demand
        
        # Mark customer as visited
        if self._is_customer(next_node):
            self.visited_mask[next_node] = True
            self.visited_customers += 1
        
        # Update route tracking
        step_distance = self.distance_matrix[self.current_node, next_node]
        self.last_step_distance = float(step_distance)
        self.route.append(next_node)
        self.total_distance += step_distance
        
        # Move to next node
        self.current_node = next_node
        self.current_step += 1
    
    def _compute_reward(self, next_node: int) -> float:
        """
        Compute reward with sparse shaping to encourage efficient single-visit routing.
        
        NORMALIZED REWARD STRUCTURE:
        - +1.0 for visiting unserved customer 
        - -0.5 for revisiting served customer (discourage loops)
        - -0.01 per step (time penalty for efficiency)
        - +2.0 bonus for completing all customers (goal achievement)
        
        Total range: approximately -2 to +22 for a full episode
        Normalized to be more stable for neural network learning.
        
        Args:
            next_node: Next node index
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # -0.01 per step (small time penalty)
        reward -= 0.01

        # Discourage non-productive revisits to depot/chargers that can cause loops.
        if self._is_charger(next_node):
            reward -= 0.05
        if self._is_depot(next_node) and self.visited_customers < self.num_customers:
            reward -= 0.05
        if next_node == self.current_node and (self._is_depot(next_node) or self._is_charger(next_node)):
            reward -= 0.25
        
        # Check if visiting a new customer (check BEFORE state is updated)
        is_new_customer = self._is_customer(next_node) and not self.visited_mask[next_node]
        is_revisited_customer = self._is_customer(next_node) and self.visited_mask[next_node]
        
        # +1.0 for visiting unserved customer
        if is_new_customer:
            reward += 1.0
        
        # -0.5 for revisiting served customer (in case mask fails)
        elif is_revisited_customer:
            reward -= 0.5
        
        # Check if all customers will be visited after this action
        customers_after = self.visited_customers + (1 if is_new_customer else 0)
        all_customers_visited = customers_after == self.num_customers
        
        # +2.0 bonus for completing all customers
        if all_customers_visited and not hasattr(self, '_completion_bonus_given'):
            reward += 2.0
            # Mark bonus as given (prevent double-counting)
            self._completion_bonus_given = True
        
        # Small penalty for battery depletion (safety constraint)
        if self.current_battery < 0:
            reward -= 1.0
        
        return float(reward)
    
    def _is_depot(self, node_idx: int) -> bool:
        """Check if node is the depot."""
        return node_idx == self.depot_idx
    
    def _is_customer(self, node_idx: int) -> bool:
        """Check if node is a customer."""
        return self.customer_start_idx <= node_idx <= self.customer_end_idx
    
    def _is_charger(self, node_idx: int) -> bool:
        """Check if node is a charging station."""
        return self.charger_start_idx <= node_idx <= self.charger_end_idx
    
    def _check_episode_done(self) -> bool:
        """
        Check if episode is done.
        
        Episode ends when:
        - All customers visited and returned to depot
        - Time limit exceeded
        - Battery becomes negative and cannot reach depot
        """
        if self.current_step >= self.time_limit:
            return True
        
        # All customers visited and back at depot
        if (self.visited_customers == self.num_customers and
            self._is_depot(self.current_node)):
            return True
        
        # Battery critical (cannot reach depot from current position)
        energy_to_depot = self._compute_energy_cost(self.current_node, self.depot_idx)
        if self.current_battery < energy_to_depot and not self._is_charger(self.current_node):
            return True
        
        return False
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        # Generate new problem instance
        self._generate_problem()
        
        # Initialize episode state
        self.current_node = self.depot_idx
        self.current_battery = self.max_battery
        self.current_cargo = 0.0
        self.visited_customers = 0
        self.visited_mask = np.zeros(self.num_nodes, dtype=bool)
        self.depot_visits = 0
        self.charger_visits = 0
        self.current_step = 0
        self.route = [self.depot_idx]
        self.total_distance = 0.0
        self.last_step_distance = 0.0
        self.infeasibility_penalty = 0.0
        
        # Reset completion bonus flag for new reward structure
        if hasattr(self, '_completion_bonus_given'):
            delattr(self, '_completion_bonus_given')
        
        info = self._get_info()
        observation = self._get_observation()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step of the environment.
        
        Args:
            action: Node index to visit next
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Validate action
        if action < 0 or action >= self.num_nodes:
            raise ValueError(f"Invalid action {action}. Must be in [0, {self.num_nodes - 1}]")
        
        # Check if action is valid
        valid_actions = self._get_valid_actions()
        if not valid_actions[action]:
            # Invalid action: apply smaller penalty (scaled to match new reward range)
            invalid_action_penalty = -1.0
            terminated = False
            truncated = self.current_step >= self.time_limit
            self.last_step_distance = 0.0
            self.current_step += 1  # Still count the step
            
            info = self._get_info()
            observation = self._get_observation()
            
            return observation, invalid_action_penalty, terminated, truncated, info
        
        # CRITICAL: Compute reward BEFORE updating state
        # This ensures customer rewards are given immediately when visiting new customers
        reward = self._compute_reward(action)
        
        # Update state (this changes visited_mask)
        self._update_state(action)
        
        # Check termination
        terminated = self._check_episode_done()
        truncated = self.current_step >= self.time_limit
        
        # Add timeout penalty if truncated without completion (scaled to new range)
        if truncated and not terminated:
            reward -= 2.0
        
        info = self._get_info()
        observation = self._get_observation()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict:
        """
        Get current observation as dictionary with static and dynamic state.
        Includes normalization for stable learning.
        
        Returns:
            Dictionary with observation components
        """
        valid_actions = self._get_valid_actions()
        
        # Normalize distance matrix to [0, 1] range
        max_dist = np.max(self.distance_matrix) if np.max(self.distance_matrix) > 0 else 1.0
        normalized_distances = self.distance_matrix / max_dist
        
        # Normalize coordinates to [0, 1] range
        coord_max = np.max(self.node_coords) if np.max(self.node_coords) > 0 else 1.0
        normalized_coords = self.node_coords / coord_max
        
        observation = {
            "node_coords": normalized_coords.astype(np.float32),
            "distance_matrix": normalized_distances.astype(np.float32),
            "node_demands": self.node_demands.astype(np.float32),
            "node_types": self.node_types.astype(np.int32),
            "current_node": np.array(self.current_node, dtype=np.int32),
            "current_battery": np.array([self.current_battery / self.max_battery], dtype=np.float32),
            "current_cargo": np.array([self.current_cargo / self.max_cargo], dtype=np.float32),
            "visited_mask": self.visited_mask.astype(np.int32),
            "valid_actions_mask": valid_actions,
        }
        
        return observation
    
    def _get_info(self) -> Dict:
        """
        Get additional information about the environment state.
        
        Returns:
            Dictionary with info
        """
        if self._is_depot(self.current_node):
            node_type = "depot"
        elif self._is_customer(self.current_node):
            node_type = "customer"
        elif self._is_charger(self.current_node):
            node_type = "charger"
        else:
            node_type = "unknown"

        all_customers_visited = self.visited_customers == self.num_customers
        success = all_customers_visited and self._is_depot(self.current_node)

        info = {
            "current_node": self.current_node,
            "current_battery": float(self.current_battery),
            "current_cargo": float(self.current_cargo),
            "visited_customers": int(self.visited_customers),
            "total_distance": float(self.total_distance),
            "distance": float(self.last_step_distance),
            "node_type": node_type,
            "all_customers_visited": bool(all_customers_visited),
            "success": bool(success),
            "depot_visits": int(self.depot_visits),
            "charger_visits": int(self.charger_visits),
            "current_step": int(self.current_step),
            "time_limit": int(self.time_limit),
        }
        
        return info
    
    def render(self):
        """
        Render the current environment state.
        
        Visualizes:
        - Node positions and types (depot, customers, chargers)
        - Current route
        - Current vehicle position
        - Battery and cargo state
        """
        if self.render_mode is None:
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Plot nodes
        # Depot
        ax.scatter(*self.node_coords[self.depot_idx], color="red", s=200, marker="s",
                  label="Depot", zorder=5)
        
        # Customers
        customer_coords = self.node_coords[self.customer_start_idx:self.customer_end_idx + 1]
        customer_visited = self.visited_mask[self.customer_start_idx:self.customer_end_idx + 1]
        
        ax.scatter(customer_coords[~customer_visited, 0],
                  customer_coords[~customer_visited, 1],
                  color="blue", s=100, marker="o", label="Customers (unvisited)", zorder=4)
        ax.scatter(customer_coords[customer_visited, 0],
                  customer_coords[customer_visited, 1],
                  color="lightblue", s=100, marker="o", label="Customers (visited)",
                  alpha=0.5, zorder=4)
        
        # Chargers
        charger_coords = self.node_coords[self.charger_start_idx:self.charger_end_idx + 1]
        ax.scatter(charger_coords[:, 0], charger_coords[:, 1], color="green", s=150,
                  marker="^", label="Chargers", zorder=4)
        
        # Plot route
        if len(self.route) > 1:
            route_coords = self.node_coords[self.route]
            ax.plot(route_coords[:, 0], route_coords[:, 1], "k--", alpha=0.5, zorder=2)
        
        # Highlight current position
        ax.scatter(*self.node_coords[self.current_node], color="purple", s=300,
                  marker="*", label="Current position", zorder=6)
        
        # Add labels
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(
            f"EVRP Visualization (Step {self.current_step}, "
            f"Battery: {self.current_battery:.1f}/{self.max_battery}, "
            f"Cargo: {self.current_cargo:.1f}/{self.max_cargo})"
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        
        if self.render_mode == "human":
            plt.show()
        elif self.render_mode == "rgb_array":
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return image
    
    def close(self):
        """Close the environment."""
        pass
    
    def get_graph(self) -> nx.Graph:
        """
        Get the NetworkX graph representation.
        
        Returns:
            NetworkX Graph object
        """
        return self.graph.copy()
    
    def get_node_coordinates(self) -> np.ndarray:
        """
        Get node coordinates.
        
        Returns:
            Array of shape (num_nodes, 2)
        """
        return self.node_coords.copy()
    
    def get_distance_matrix(self) -> np.ndarray:
        """
        Get distance matrix.
        
        Returns:
            Array of shape (num_nodes, num_nodes)
        """
        return self.distance_matrix.copy()
