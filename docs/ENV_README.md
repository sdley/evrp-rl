# EVRP Environment Implementation

## Overview

This module implements a **Gymnasium-compatible Electric Vehicle Routing Problem (EVRP) environment** with battery constraints, charging stations, time windows, and cargo capacity. The environment is designed for deep reinforcement learning research on realistic EV routing scenarios.

## Key Features

### Node Structure

- **Node 0**: Depot (start/end point, cargo unloading point)
- **Nodes 1 to m**: Customers with random demands
- **Nodes m+1 to g**: Charging stations for battery recharging

### State Representation (Dictionary)

```python
observation = {
    "node_coords": np.array,           # (num_nodes, 2) - node coordinates
    "distance_matrix": np.array,       # (num_nodes, num_nodes) - euclidean distances
    "node_demands": np.array,          # (num_nodes,) - customer demands
    "node_types": np.array,            # (num_nodes,) - node type (0=depot, 1=customer, 2=charger)
    "current_node": int,               # current vehicle position
    "current_battery": float,          # remaining battery (Wh)
    "current_cargo": float,            # current cargo load (kg)
    "visited_mask": np.array,          # (num_nodes,) - visited customers
    "valid_actions_mask": np.array,    # (num_nodes,) - feasible actions
}
```

### Dynamics

#### Battery Update

$$
b_{t+1} = \begin{cases}
B & \text{if node is charger} \\
b_t - f(i, j) & \text{otherwise}
\end{cases}
$$

Where:

- $f(i, j) = \text{distance}(i, j) \times \text{energy\_consumption\_rate}$ (Wh)
- $B$ = maximum battery capacity

#### Cargo Update

$$
q_{t+1} = \begin{cases}
0 & \text{if node is depot} \\
q_t + d_j & \text{if customer } j \text{ unvisited} \\
q_t & \text{otherwise}
\end{cases}
$$

### Reward Structure

$$r_t = -d_{ij} - \lambda_c \cdot \mathbb{1}_{\text{charger}} - \lambda_d \cdot \mathbb{1}_{\text{depot-revisit}} - \text{infeasibility}$$

Where:

- $-d_{ij}$: Distance cost
- $\lambda_c$: Charger visit penalty
- $\lambda_d$: Depot revisit penalty
- Infeasibility: $\max(0, -b_t)$ when battery would be negative

### Action Masking (Transient & Permanent Constraints)

**Permanent (Cannot revisit):**

- Already visited customers

**Transient (Battery feasibility):**

- Nodes unreachable due to insufficient battery: requires $f(i, j) + f(j, \text{depot}) \leq b_i$
- Cargo capacity violation: $q_t + d_j > Q_{\max}$

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- numpy >= 1.21.0
- gymnasium >= 0.28.0
- networkx >= 2.6
- torch >= 2.0.0 (for RL training)
- matplotlib >= 3.5.0 (visualization)

## Usage

### Basic Example

```python
from src.env import EVRPEnvironment

# Create environment
env = EVRPEnvironment(
    num_customers=10,
    num_chargers=3,
    max_battery=150.0,
    max_cargo=100.0,
    energy_consumption_rate=0.8,
    time_limit=100,
)

# Reset and get initial observation
obs, info = env.reset()

# Take a step
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### Running an Episode with Random Policy

```python
env = EVRPEnvironment(
    num_customers=8,
    num_chargers=3,
    render_mode="human"
)

obs, info = env.reset()
total_reward = 0

while True:
    # Get valid actions
    valid_mask = obs["valid_actions_mask"]
    valid_actions = [i for i, v in enumerate(valid_mask) if v]

    # Random valid action
    action = np.random.choice(valid_actions)

    # Step
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        break

print(f"Total reward: {total_reward}")
print(f"Customers served: {info['visited_customers']}")
print(f"Distance traveled: {info['total_distance']:.2f}")

env.render()
env.close()
```

## API Reference

### Initialization Parameters

```python
EVRPEnvironment(
    num_customers: int = 10,
    num_chargers: int = 3,
    max_battery: float = 100.0,           # Maximum battery capacity (Wh)
    max_cargo: float = 100.0,             # Maximum cargo capacity (kg)
    energy_consumption_rate: float = 1.0, # Energy per unit distance (Wh/km)
    charger_cost: float = 0.5,            # Reward penalty for charger visits
    depot_revisit_cost: float = 1.0,      # Reward penalty for depot revisits
    time_limit: int = 100,                # Maximum steps per episode
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
)
```

### Core Methods

#### `reset(seed=None, options=None) -> (obs, info)`

Initialize episode to initial state. Returns observation and info.

#### `step(action: int) -> (obs, reward, terminated, truncated, info)`

Execute one action. Returns:

- `obs`: New observation
- `reward`: Immediate reward
- `terminated`: Episode ended (goal reached/infeasible)
- `truncated`: Time limit exceeded
- `info`: Auxiliary information

#### `render()`

Visualize environment state with matplotlib:

- Red square: Depot
- Blue circles: Unvisited customers
- Light blue circles: Visited customers
- Green triangles: Charging stations
- Purple star: Current position

#### `get_graph() -> nx.Graph`

Get NetworkX graph representation of problem.

#### `get_node_coordinates() -> np.ndarray`

Get (num_nodes, 2) array of node coordinates.

#### `get_distance_matrix() -> np.ndarray`

Get (num_nodes, num_nodes) distance matrix.

### Gymnasium-Compatible Interface

```python
action_space: spaces.Discrete(num_nodes)

observation_space: spaces.Dict({
    "node_coords": Box,
    "distance_matrix": Box,
    "node_demands": Box,
    "node_types": Box,
    "current_node": Discrete,
    "current_battery": Box,
    "current_cargo": Box,
    "visited_mask": MultiBinary,
    "valid_actions_mask": MultiBinary,
})
```

## Testing

### Run Fast Tests

```bash
python3 -m pytest test_evrp_fast.py -v
```

### Run Example

```bash
python3 example_evrp.py
```

## Environment Properties

### Graph

- Complete graph: All nodes connected
- Metric property: Triangle inequality holds (Euclidean distance)
- Symmetric distances

### Problem Characteristics

- **Dynamically generated**: New instance on each `reset()`
- **Configurable scale**: Adjustable number of customers/chargers
- **Realistic constraints**: Battery, cargo, time windows
- **Sparse rewards**: Only distance-based immediate feedback

## Implementation Notes

### Design Decisions

1. **Action Validity**: Invalid actions return penalty (-10) without advancing state
2. **Battery Feasibility**: Conservative check requires battery for round-trip to depot
3. **Cargo Capacity**: Hard constraint (cannot exceed max_cargo)
4. **Graph Generation**: Random coordinates in [0, 100] × [0, 100]
5. **Depot Location**: Fixed at origin [0, 0] for reproducibility

### Computational Complexity

- State computation: O(num_nodes²) for distance matrix
- Valid action computation: O(num_nodes²) per step
- Memory: O(num_nodes²) for distance matrix storage

## References

- Gymnasium: https://gymnasium.farl.io/
- EVRP Formulation: Based on IEEE JIOT 2024 paper
- Graph Structure: NetworkX library for graph operations

## Example Output

```
============================================================
EVRP Visualization (Step 15, Battery: 45.3/150.0, Cargo: 25.0/100.0)
============================================================

Total steps: 24
Total distance: 487.63 km
Total reward: -523.45
Visited customers: 8 / 10
Depot visits: 3
Charger visits: 2
Success: False
```

## Future Enhancements

1. **Time Windows**: Add customer time window constraints
2. **Multiple Vehicles**: Fleet management with multiple vehicles
3. **Real Maps**: Integration with OpenStreetMap data
4. **Stochastic Demands**: Random customer demands across episodes
5. **Traffic Patterns**: Time-dependent travel costs
6. **Heterogeneous Chargers**: Different charging speeds/costs

## License

MIT License - See LICENSE.txt

## Contributing

Contributions welcome! Please ensure code follows best practices:

- PEP 8 style guide
- Type hints for all functions
- Comprehensive docstrings
- Unit tests for new features
