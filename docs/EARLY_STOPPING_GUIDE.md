# Early Stopping Guide

## Overview

Early stopping is an intelligent training termination technique that prevents unnecessary training iterations while maintaining high performance. It monitors evaluation metrics and stops training when the model has converged.

## How It Works

### Key Components

1. **Evaluation Tracking**: After every `EVAL_INTERVAL` episodes, the agent is evaluated on a fixed environment
2. **Best Reward Tracking**: Keeps track of the highest evaluation reward achieved
3. **Patience Counter**: Counts consecutive evaluation intervals without improvement
4. **Convergence Check**: When patience exceeds the threshold, training stops automatically

### Algorithm

```python
for episode in range(MAX_EPISODES):
    # ... training ...

    if (episode + 1) % EVAL_INTERVAL == 0:
        eval_reward = evaluate_agent()

        if eval_reward > best_reward:
            best_reward = eval_reward
            patience_counter = 0  # Reset on improvement
        else:
            if episode + 1 >= WARMUP_EPISODES:
                patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            stop_training()  # Exit loop
```

## Configuration Parameters

### `EARLY_STOPPING_PATIENCE` (Default: 100)

- **Type**: Integer (number of evaluation intervals)
- **Meaning**: Stop if no improvement for 100 evaluation intervals
- **Calculation**: 100 evals × 20 episodes/eval = 2,000 steps before stopping
- **Recommendation**:
  - **100-150**: For most RL tasks (A2C, SAC)
  - **50-100**: For simpler environments with faster convergence
  - **200+**: For complex tasks that need more exploration time

### `EARLY_STOPPING_MIN_DELTA` (Default: 0.05)

- **Type**: Float (reward improvement threshold)
- **Meaning**: Minimum reward increase to count as "improvement"
- **Usage**: Can be integrated for stricter convergence criteria
- **Recommendation**: 0.05 for EVRP environment

### `WARMUP_EPISODES` (Default: 200)

- **Type**: Integer (number of episodes)
- **Meaning**: Episodes before early stopping is active
- **Purpose**: Allows initial exploration without triggering early stop
- **Calculation**: With EVAL_INTERVAL=20, this is ~10 evaluation intervals
- **Recommendation**:
  - **200**: Recommended for A2C/SAC agents
  - **100-300**: Depending on environment complexity

### `EVAL_INTERVAL` (Default: 20)

- **Type**: Integer (episodes between evaluations)
- **Meaning**: How often to evaluate and check for improvement
- **Recommendation**: 20 for balanced checking

### `TRAIN_EPISODES` (Default: 10,000)

- **Type**: Integer (maximum episodes)
- **Meaning**: Maximum training episodes (hard limit before early stopping)
- **Role**: Safety cap to prevent infinite training

## Example Scenarios

### Scenario 1: Fast Convergence (EVRP with GAT)

```python
WARMUP_EPISODES = 200          # Initial exploration
EARLY_STOPPING_PATIENCE = 80   # Stop after 80 * 20 = 1,600 episodes of no improvement
EVAL_INTERVAL = 20
```

**Expected behavior**: Stops around 2,000-3,000 episodes if converged quickly

### Scenario 2: Slow Convergence (Complex environment)

```python
WARMUP_EPISODES = 500          # Longer warmup
EARLY_STOPPING_PATIENCE = 150  # More patient
EVAL_INTERVAL = 20
```

**Expected behavior**: Allows up to 3,500 episodes of no improvement

### Scenario 3: Very Stable Training

```python
WARMUP_EPISODES = 100          # Short warmup
EARLY_STOPPING_PATIENCE = 50   # Quick stopping
EVAL_INTERVAL = 50             # Less frequent checks
```

**Expected behavior**: Stops quickly if agent converges rapidly

## Monitoring Early Stopping

### Console Output

When training, watch for:

```
Episode   100: Loss=0.2511 ↓ | Reward=18.50 ↑ | GradNorm=0.234 | LR=0.000300
    → New best eval: 14.52 ⭐
...
Episode  2800: Loss=0.1203 → | Reward=19.10 ↑ | GradNorm=0.089 | LR=0.000300

🛑 EARLY STOPPING: No improvement for 100 evaluations
   Best eval reward: 15.23
   Episodes trained: 2850/10000
```

### History Tracking

Each trained model's history includes:

```python
{
    'early_stopped': True,           # Whether early stopping was triggered
    'total_episodes_trained': 2850,  # Actual episodes trained (< 10,000)
    'eval_history': [...]            # All evaluation rewards during training
}
```

### Analysis Cell Output

The analysis cell shows:

```
A2C:
  Seed 42: 2850/10000 episodes | Saved 7150 (71.5%) | Best eval=15.23 ⭐
  Seed 123: 10000/10000 episodes | Full training | Best eval=14.89

  Summary:
    • Early stopped: 2/3 seeds
    • Avg episodes saved: 7150
    • Total efficiency gain: 47.7%
```

## When Early Stopping Works Best

✅ **Works well for**:

- Convergent problems (EVRP routing)
- Stable training curves
- When you have enough evaluation data
- Production training with time constraints

❌ **May not work well for**:

- Very noisy/unstable environments
- Highly stochastic problems
- When evaluation is expensive
- Multi-phase training (curriculum learning)

## Adjusting for Your Problem

### If stopping too early

- Increase `EARLY_STOPPING_PATIENCE` (e.g., 100 → 150)
- Increase `WARMUP_EPISODES` (e.g., 200 → 500)
- Decrease `EVAL_INTERVAL` (e.g., 20 → 10) for more frequent checks

### If not stopping early enough

- Decrease `EARLY_STOPPING_PATIENCE` (e.g., 100 → 50)
- Decrease `WARMUP_EPISODES` (e.g., 200 → 100)
- Increase `EVAL_INTERVAL` (e.g., 20 → 30) for less frequent checks

### If evaluation is noisy

- Increase `EARLY_EPISODES` (e.g., 20 → 10) to average over more samples
- Increase patience to handle natural fluctuations
- Use validation reward smoothing

## Performance Impact

Typical efficiency gains for EVRP:

- **A2C**: 40-70% episodes saved (2,000-7,000 episodes stopped)
- **SAC**: 30-60% episodes saved (3,000-7,000 episodes stopped)

Performance maintained:

- Early stopped models: ~95-99% of full training performance
- Time savings: Proportional to episodes saved

## Troubleshooting

| Issue           | Symptom               | Solution                                    |
| --------------- | --------------------- | ------------------------------------------- |
| Stops too early | Agent unstable        | Increase WARMUP_EPISODES or PATIENCE        |
| Never stops     | Runs full 10,000      | Decrease PATIENCE or increase EVAL_INTERVAL |
| Noisy stopping  | Inconsistent behavior | Increase EVAL_EPISODES                      |
| Wrong agent?    | Config mismatch       | Check agent name (a2c vs sac)               |

## See Also

- [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) - Full configuration options
- [FRAMEWORK_SUMMARY.md](FRAMEWORK_SUMMARY.md) - Training framework details
- Training notebook: `examples/agent_benchmark_evrp_a2c_sac.ipynb`
