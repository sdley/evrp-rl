# Quick Reference: Configuration Options

## Environment Configuration

```yaml
env:
  # Required
  num_customers: 20 # Number of delivery locations
  num_chargers: 5 # Number of charging stations

  # Capacity constraints (use either naming convention)
  max_battery: 100.0 # OR battery_capacity
  max_cargo: 50.0 # OR cargo_capacity

  # Optional parameters
  energy_consumption_rate: 1.0 # Energy per distance unit
  charger_cost: 0.5 # Penalty for visiting chargers
  depot_revisit_cost: 1.0 # Penalty for depot revisits
  time_limit: 100 # Max steps per episode
  seed: 42 # Random seed
  render_mode: null # 'human' or 'rgb_array'
```

## Agent Configuration

### A2C Agent

```yaml
agent:
  type: "a2c"

  encoder:
    type: "mlp" # 'mlp' or 'gat'
    embed_dim: 128 # Embedding dimension
    hidden_dim: 256 # Hidden layer size (MLP only)
    num_layers: 3 # Number of layers
    dropout: 0.1 # Dropout probability

    # GAT-specific (when type: 'gat')
    num_heads: 8 # Number of attention heads
    negative_slope: 0.2 # LeakyReLU slope
    concat_heads: true # Concatenate or average heads

  hyperparameters:
    lr: 3e-4 # Learning rate
    gamma: 0.99 # Discount factor
    entropy_coef: 0.01 # Entropy regularization
    hidden_dim: 256 # Actor/critic hidden size
```

### SAC Agent

```yaml
agent:
  type: "sac"

  encoder:
    type: "gat" # 'mlp' or 'gat'
    embed_dim: 128
    num_layers: 3
    num_heads: 8
    dropout: 0.1

  hyperparameters:
    lr: 3e-4
    gamma: 0.99
    tau: 0.005 # Soft update coefficient
    alpha: "auto" # Entropy coefficient ('auto' or float)
    hidden_dim: 256
    buffer_size: 10000 # Replay buffer capacity
    batch_size: 32 # Training batch size
    learning_starts: 100 # Steps before training starts
```

## Run Configuration

```yaml
run:
  name: "experiment_name" # Experiment identifier
  epochs: 100 # Number of training epochs
  eval_frequency: 10 # Evaluate every N epochs
  save_frequency: 25 # Save checkpoint every N epochs
  max_steps_per_episode: 100 # Max steps before truncation
  num_eval_episodes: 5 # Episodes to run during eval
  device: "cuda" # 'cuda' or 'cpu'
  seed: 42 # Random seed for reproducibility
```

## Reward Shaping (Optional)

```yaml
reward:
  # Battery penalties
  battery_penalty_coef: 0.1 # Penalty coefficient
  battery_threshold: 20.0 # Battery level threshold

  # Cargo penalties
  cargo_penalty_coef: 0.05 # Penalty for unused capacity

  # Completion rewards
  completion_bonus: 10.0 # Bonus for serving all customers
  early_termination_penalty: -5.0 # Penalty for early stop
```

## Action Masking (Optional)

```yaml
mask:
  # Battery constraints
  mask_low_battery: true # Mask actions with insufficient charge
  battery_threshold: 15.0 # Minimum battery to consider action

  # Cargo constraints
  mask_excess_cargo: true # Mask actions exceeding capacity
  safety_margin: 1.1 # Multiply threshold by margin
```

## Complete Example

```yaml
env:
  num_customers: 20
  num_chargers: 5
  max_battery: 100.0
  max_cargo: 50.0
  seed: 42

agent:
  type: "sac"
  encoder:
    type: "gat"
    embed_dim: 128
    num_layers: 3
    num_heads: 8
  hyperparameters:
    lr: 3e-4
    gamma: 0.99
    tau: 0.005
    alpha: "auto"
    buffer_size: 10000
    batch_size: 32

reward:
  battery_penalty_coef: 0.1
  battery_threshold: 20.0
  completion_bonus: 10.0

mask:
  mask_low_battery: true
  battery_threshold: 15.0

run:
  name: "sac_gat_exp"
  epochs: 100
  eval_frequency: 10
  save_frequency: 25
  device: "cuda"
  seed: 42
```

## Programmatic API

```python
from src.framework import create_experiment_config

config = create_experiment_config(
    env_config={
        'num_customers': 20,
        'num_chargers': 5,
        'max_battery': 100.0,
        'max_cargo': 50.0,
    },
    agent_config={
        'type': 'sac',
        'encoder': {
            'type': 'gat',
            'embed_dim': 128,
            'num_layers': 3,
            'num_heads': 8,
        },
        'hyperparameters': {
            'lr': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 'auto',
            'buffer_size': 10000,
            'batch_size': 32,
        },
    },
    run_config={
        'name': 'sac_gat_exp',
        'epochs': 100,
        'eval_frequency': 10,
        'device': 'cuda',
    },
)
```

## Common Configurations

### Quick Test (Small, Fast)

```yaml
env:
  num_customers: 10
  num_chargers: 3

agent:
  type: "a2c"
  encoder:
    type: "mlp"
    embed_dim: 64
    num_layers: 2

run:
  epochs: 20
  eval_frequency: 5
  device: "cpu"
```

### Production (Large, Thorough)

```yaml
env:
  num_customers: 50
  num_chargers: 10

agent:
  type: "sac"
  encoder:
    type: "gat"
    embed_dim: 256
    num_layers: 4
    num_heads: 16

run:
  epochs: 500
  eval_frequency: 20
  save_frequency: 50
  device: "cuda"
```

### Ablation Study

```python
configs = {
    'A2C+MLP': create_experiment_config(
        env_config={'num_customers': 15, 'num_chargers': 4},
        agent_config={'type': 'a2c', 'encoder': {'type': 'mlp'}},
        run_config={'epochs': 50, 'name': 'a2c_mlp'}
    ),
    'A2C+GAT': create_experiment_config(
        env_config={'num_customers': 15, 'num_chargers': 4},
        agent_config={'type': 'a2c', 'encoder': {'type': 'gat'}},
        run_config={'epochs': 50, 'name': 'a2c_gat'}
    ),
    'SAC+MLP': create_experiment_config(
        env_config={'num_customers': 15, 'num_chargers': 4},
        agent_config={'type': 'sac', 'encoder': {'type': 'mlp'}},
        run_config={'epochs': 50, 'name': 'sac_mlp'}
    ),
    'SAC+GAT': create_experiment_config(
        env_config={'num_customers': 15, 'num_chargers': 4},
        agent_config={'type': 'sac', 'encoder': {'type': 'gat'}},
        run_config={'epochs': 50, 'name': 'sac_gat'}
    ),
}
```

## Tips

1. **Start Small**: Test with small problem sizes (`num_customers=10`) first
2. **Use CPU for Small Problems**: Switch to GPU only for larger instances
3. **Tune Learning Rate**: Typical range is 1e-5 to 1e-3
4. **Monitor Entropy**: Ensure agent explores enough (entropy > 0)
5. **Check Success Rate**: Aim for >80% in evaluation
6. **Save Frequently**: Set `save_frequency` to avoid losing progress
7. **Use Reward Shaping**: Can significantly improve learning
8. **Enable Action Masking**: Prevents invalid actions, speeds up learning
9. **Set Seeds**: Always set `seed` for reproducibility
10. **Evaluate Often**: Set `eval_frequency` to catch issues early

## Validation

The framework validates all configurations and provides helpful error messages:

```python
from src.framework import ConfigLoader

config = ConfigLoader.load('my_config.yaml')  # Auto-validates
config = ConfigLoader.validate(config_dict)   # Manual validation
```

Required fields:

- `env.num_customers`
- `env.num_chargers`
- `agent.type`
- `agent.encoder.type`
- `run.name`
- `run.epochs`

All other fields have sensible defaults.
