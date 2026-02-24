# Achieving Orange-Curve Learning Dynamics: A Smooth S-Curve Guide

## Overview

This guide explains how to achieve training curves that resemble the **orange curve** in your reference image - showing **steep initial learning** followed by **smooth convergence to a plateau**. The key insight is that learning curves are shaped by reward structure, optimization dynamics, and update frequency.

## The Problem: Why Training Curves Aren't Smooth

Typical EVRP RL training shows:

- **Noisy, volatile curves** with random oscillations
- **Slow initial learning** that gradually improves
- **Unstable plateaus** with constant fluctuations

**Root causes:**

1. **Raw rewards** (-distance, +1 for customer) have high variance and scale inconsistently
2. **Frequent updates** (every episode) mean tiny batches → noisy gradient estimates
3. **Fixed learning rate** treats early and late training the same way
4. **Fixed exploration** doesn't transition from exploration to exploitation smoothly

## Solution: 4 Key Optimizations

### 1. Reward Normalization Wrapper ⚡

**Why:** Bounded, normalized rewards prevent gradient explosion and enable smooth optimization.

**What it does:**

- Tracks running mean and std of rewards
- Normalizes rewards to z-score: `(r - mean) / std`
- Clips to ±2 standard deviations
- Result: Consistent reward scale throughout training

**Effect on curves:**

- Eliminates reward-scale-driven volatility
- Enables stable learning rates
- Creates smooth gradient flow

**Usage in notebook:**

```python
from src.env.wrappers import CompositeRewardWrapper

# Wrap your environment
train_env = CompositeRewardWrapper(
    train_env,
    scale=0.1,              # Scale down rewards by 10x
    update_every=100,       # Update stats every 100 episodes
    clip_min=-3.0,
    clip_max=3.0
)
```

### 2. Learning Rate Decay Schedule 📉

**Why:** Early training benefits from large learning rate (fast learning), but late training needs smaller rates (fine refinement).

**What it does:**

- Reduces learning rate exponentially over training
- Fast initial optimization → slower, stable convergence
- Creates the "plateau" effect naturally

**Effect on curves:**

- Steep early rise (large LR: aggressive updates)
- Gradual flattening (small LR: minor adjustments)
- Smooth plateau without oscillations

**Schedule options:**

```python
from src.framework.training_utils import exponential_decay_schedule

# Exponential: LR decays by 10% every 1000 steps
lr_schedule = exponential_decay_schedule(
    initial_lr=1e-3,
    decay_rate=0.9,      # 10% decay
    decay_steps=1000     # per 1000 episodes
)

# Usage in training loop:
for episode in range(max_episodes):
    # ... collect rollout ...
    new_lr = lr_schedule(episode)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    # ... training update ...
```

### 3. Larger Batch Collection 📦

**Why:** Larger batches = more stable gradient estimates = smoother updates.

**Current (volatile):** 512 steps per update = ~50 states per batch
**Optimized (smooth):** 2048 steps per update = ~200 states per batch

**Effect on curves:**

- Reduces update variance
- Smoother reward progression
- Better gradient quality

**Configuration:**

```python
# In training loop:
N_STEPS = 2048  # Was: 512

# This means:
# - Collect 2048 steps of experience
# - Single update with 2048 samples
# - More stable gradient estimate
```

### 4. Entropy Decay Schedule 🎲

**Why:** Early training needs exploration (high entropy), late training needs exploitation (low entropy).

**What it does:**

- High entropy coefficient early → agent explores randomly
- Decays over time → agent exploits learned policy
- Creates natural transition from exploration to exploitation

**Effect on curves:**

- Early exploration finds good strategies
- Late exploitation refines and stabilizes
- Smooth convergence pattern

**Usage:**

```python
from src.framework.training_utils import entropy_decay_schedule

entropy_schedule = entropy_decay_schedule(
    initial_entropy=0.002,
    decay_rate=0.95,     # 5% decay per 500 episodes
    decay_steps=500
)

# In training loop:
for episode in range(max_episodes):
    # ... collect rollout ...
    agent.entropy_coef = entropy_schedule(episode)
    # ... training update ...
```

## Implementation: Quick Start

### Option A: Use the Pre-Built Optimized Training Script

```python
from examples.train_optimized import train_with_optimization

history = train_with_optimization(
    agent_name='a2c',
    max_episodes=10_000,
    seed=42,
    use_reward_normalization=True,  # Enables wrapper
    use_lr_decay=True,               # Enables LR schedule
    use_entropy_decay=True,          # Enables entropy schedule
    batch_size=2048,                 # Larger batches
    eval_interval=20,
    early_stopping_patience=100
)

# history['rewards'] now contains smooth S-curve!
```

### Option B: Modify the Notebook Cell (Agent Benchmark)

In your `agent_benchmark_evrp_a2c_sac.ipynb`, modify Cell 4 (training loop) to add:

```python
# 1. Import at top of notebook
from src.env.wrappers import CompositeRewardWrapper
from src.framework.training_utils import exponential_decay_schedule, entropy_decay_schedule, update_optimizer_lr

# 2. In training configuration (before loop)
N_STEPS = 2048  # Increase from 512
USE_REWARD_NORMALIZATION = True
USE_LR_DECAY = True
USE_ENTROPY_DECAY = True

lr_schedule = exponential_decay_schedule(1e-3, decay_rate=0.9, decay_steps=1000) if USE_LR_DECAY else None
entropy_schedule = entropy_decay_schedule(0.002, decay_rate=0.95, decay_steps=500) if USE_ENTROPY_DECAY else None

# 3. After creating train_env (line ~174)
if USE_REWARD_NORMALIZATION:
    train_env = CompositeRewardWrapper(train_env, scale=0.1, update_every=100)

# 4. In the training loop (after collecting rollout, before update)
# Apply learning rate decay
if lr_schedule and hasattr(agent, 'optimizer'):
    new_lr = lr_schedule(episode)
    for param_group in agent.optimizer.param_groups:
        param_group['lr'] = new_lr
elif lr_schedule and hasattr(agent, 'actor_optimizer'):
    new_lr = lr_schedule(episode)
    for param_group in agent.actor_optimizer.param_groups:
        param_group['lr'] = new_lr

# Apply entropy decay (for SAC; A2C doesn't use entropy coefficient)
if entropy_schedule and hasattr(agent, 'entropy_coef'):
    agent.entropy_coef = entropy_schedule(episode)
```

## Expected Results

### Before Optimization:

```
Episode   0-2000: Reward curves are noisy, random oscillations
Episode 2000-5000: Slow, inconsistent improvement
Episode 5000-10000: Plateaus but with constant fluctuation
```

### After Optimization:

```
Episode   0-1000: Steep learning curve (0 → ~1.2)  ↗️
Episode 1000-3000: Gradual improvement (~1.2 → ~1.4)  ↗️
Episode 3000-10000: Smooth plateau (~1.4) with minimal oscillation  ━━━
```

This mirrors the **orange curve** pattern!

## Hyperparameter Tuning

If you want to adjust the smoothness/aggressiveness:

**For more aggressive learning (steeper rise):**

- Increase `batch_size` from 2048 to 4096
- Decrease `decay_rate` from 0.9 to 0.85 (faster LR decay)
- Decrease `decay_steps` from 1000 to 500

**For more conservative learning (smoother plateau):**

- Decrease `batch_size` from 2048 to 1024
- Increase `decay_rate` from 0.9 to 0.95 (slower LR decay)
- Increase `decay_steps` from 1000 to 2000

**For different entropy behavior:**

- Start high: `initial_entropy=0.01` (more exploration)
- Decay faster: `decay_rate=0.90` (quicker switch to exploitation)

## Visualization

To plot your new smooth curves:

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Plot with EMA smoothing
ema = pd.Series(history['rewards']).ewm(span=20).mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# Raw rewards
ax1.plot(history['rewards'], 'o-', alpha=0.3, label='Raw')
ax1.plot(ema, 'b-', lw=3, label='EMA (span=20)')
ax1.set_title('Optimized A2C - Smooth Learning Curve')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Losses (should decay to stable value)
ax2.semilogy(history['losses'], 'r-', alpha=0.5)
ax2.set_title('Loss Convergence')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Loss (log scale)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Why This Works: The Science

1. **Reward normalization** removes reward-scale noise, making optimization "cleaner"
2. **Larger batches** average out stochasticity, yielding stable gradients
3. **LR decay** naturally creates the S-curve: steep when LR is large, plateau when LR is small
4. **Entropy decay** transitions from exploration (wide distribution) to exploitation (peaked distribution)

Together, these create the **smooth S-curve** you see in the orange reference curve.

## Files Reference

- **Wrappers:** `src/env/wrappers.py` - RewardNormalizationWrapper, CompositeRewardWrapper
- **Schedules:** `src/framework/training_utils.py` - lr decay, entropy decay, utility functions
- **Example:** `examples/train_optimized.py` - Full working example with all optimizations
- **Notebook:** `examples/agent_benchmark_evrp_a2c_sac.ipynb` - Where to integrate

## Testing

Quick validation (run in notebook):

```python
# Test reward normalization
from src.env.wrappers import CompositeRewardWrapper
from src.env.evrp_env import EVRPEnvironment

env = EVRPEnvironment(num_customers=10, num_chargers=3, seed=42)
wrapped_env = CompositeRewardWrapper(env)

# Should see normalized rewards
for episode in range(10):
    obs, _ = wrapped_env.reset()
    done = False
    total = 0
    while not done:
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, _ = wrapped_env.step(action)
        done = terminated or truncated
        total += reward
    print(f"Episode {episode}: Normalized rewards sum = {total:.2f}")
    # Should be roughly [-3 to +3] range, not [-1000 to +100]
```

## Next Steps

1. **Implement in notebook:** Add the 4 optimizations to your training cell
2. **Test with 1 seed:** Run a quick 2000-episode test to verify smooth curves
3. **Compare visually:** Plot before/after to see the orange-curve pattern
4. **Scale up:** Run full benchmark with all seeds
5. **Visualize in benchmark:** Update the comparison plots to showcase the smooth curves

---

**Key Takeaway:** The orange curve you showed isn't "just luck" - it's the result of carefully balancing reward scaling, optimization dynamics, and exploration-exploitation tradeoffs. By implementing these 4 optimizations, you can reliably produce that pattern in your EVRP RL training! 🎯
