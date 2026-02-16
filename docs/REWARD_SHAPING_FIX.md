# Reward Shaping Fix for Efficient EVRP Routing

## Problem Identified

The trained agents were exhibiting inefficient routing behavior:

- **Excessive revisits**: Visiting same customers multiple times (120 visits for 20 customers)
- **Charger overuse**: 31 charger visits for 20-customer routes
- **Long episodes**: 162-177 steps instead of optimal ~25-35 steps
- **Complex routes**: Spider-web patterns with many loops

### Root Cause

The previous reward structure was **too focused on distance minimization** with weak signals for customer service:

- Large negative rewards for distance traveled
- Small positive rewards (+1) for visiting new customers
- Weak penalties for revisits
- Complex multi-component reward with competing signals

This caused agents to:

1. Get confused about the primary objective (serve all customers efficiently)
2. Fall into local minima (visit nearby customers repeatedly)
3. Not learn clear distinctions between good and bad actions

## Solution: Sparse Reward Shaping

Implemented a **clearer, sparser reward structure** based on RL best practices:

### New Reward Components

```python
+10   # Visiting unserved customer (STRONG positive signal)
-1    # Revisiting served customer (clear penalty for loops)
-0.1  # Per step (time penalty for efficiency)
+50   # Bonus for completing all customers (terminal goal)
```

### Design Rationale

1. **Strong Customer Incentive (+10)**
   - Makes serving new customers the primary objective
   - Dominates step penalty (-0.1), ensuring agents prioritize coverage over distance
   - Clear signal: "Serving customers is good"

2. **Revisit Penalty (-1)**
   - Prevents agents from getting stuck in loops
   - 10x stronger than step penalty, discouraging redundant visits
   - Clear signal: "Don't go back to served customers"

3. **Step Penalty (-0.1)**
   - Encourages shorter routes after coverage objectives met
   - Subtle enough not to interfere with exploration
   - Prevents agents from wandering aimlessly

4. **Completion Bonus (+50)**
   - Strong terminal reward for achieving goal
   - Provides clear value function target
   - Equivalent to serving 5 customers (makes completion attractive)

### Expected Improvements

| Metric            | Old (Distance-Based) | New (Reward-Shaped)   |
| ----------------- | -------------------- | --------------------- |
| Route Length      | 88-177 steps         | 25-35 steps (optimal) |
| Customer Revisits | 6x average           | 0 (single-visit)      |
| Charger Visits    | 31 (excessive)       | 2-3 (efficient)       |
| Coverage          | 100% (after loops)   | 100% (direct)         |
| Training Signal   | Noisy, conflicting   | Clear, consistent     |

## Implementation Details

### Files Modified

**`src/env/evrp_env.py`** - `_compute_reward()` method:

```python
def _compute_reward(self, next_node: int) -> float:
    reward = 0.0

    # -0.1 per step (time penalty)
    reward -= 0.1

    # Check if visiting a new customer
    is_new_customer = self._is_customer(next_node) and not self.visited_mask[next_node]
    is_revisited_customer = self._is_customer(next_node) and self.visited_mask[next_node]

    # +10 for visiting unserved customer
    if is_new_customer:
        reward += 10.0

    # -1 for revisiting served customer
    elif is_revisited_customer:
        reward -= 1.0

    # Check if all customers will be visited after this action
    customers_after = self.visited_customers + (1 if is_new_customer else 0)
    all_customers_visited = customers_after == self.num_customers

    # +50 bonus for completing all customers (given once)
    if all_customers_visited and not hasattr(self, '_completion_bonus_given'):
        reward += 50.0
        self._completion_bonus_given = True

    # Battery penalty (safety constraint)
    if self.current_battery < 0:
        reward -= 5.0

    return float(reward)
```

### Notebook Updates

Added new cells in `examples/agent_benchmark_evrp_a2c_sac.ipynb`:

- **Cell 8b**: Retrain agents with new reward structure
- **Cell 8c**: Compare old vs new agent behavior side-by-side

## Training Instructions

### Quick Test (10k episodes)

```python
# Run Cell 8b in notebook
TRAIN_EPISODES_V2 = 10_000  # Fast validation
```

### Production Training (50k episodes recommended)

```python
TRAIN_EPISODES_V2 = 50_000  # Full convergence
```

### Expected Training Metrics

**Reward trajectory should show**:

- Initial exploration: -5 to +5 (random)
- Early learning: +20 to +50 (discovering customers)
- Mid training: +100 to +150 (efficient partial coverage)
- Converged: +180 to +200 (optimal 20-customer routes)

**Calculation for optimal episode**:

```
20 customers × 10 = +200
Completion bonus = +50
~30 steps × -0.1 = -3
Total = +247 (theoretical maximum)
```

Practical convergence: **+180 to +220** (accounting for battery constraints and charger visits)

## Validation Checklist

After retraining, verify:

- [ ] Route length: 25-35 steps (not 88+)
- [ ] Customer revisits: 0 (all unique visits)
- [ ] Charger visits: 2-4 (realistic for battery constraints)
- [ ] Coverage: 100% (all customers served)
- [ ] Route visualization: Clean paths (no spider webs)
- [ ] Training rewards: Increasing from -5 to +180+

## Theoretical Background

This reward structure follows established RL principles:

1. **Sparse Rewards**: Large rewards for critical events (customer visits, completion)
2. **Reward Shaping**: Small dense signal (step penalty) guides exploration between sparse rewards
3. **Potential-Based Shaping**: Step penalty is equivalent to potential difference (closer to completion = higher potential)
4. **Clear Credit Assignment**: Immediate +10 when visiting new customer (no delayed rewards)

### References

- Ng, A. Y., Harada, D., & Russell, S. (1999). "Policy invariance under reward transformations"
- Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning" (sparse rewards in Atari)
- Schulman, J., et al. (2017). "Proximal Policy Optimization" (advantage-based learning with shaped rewards)

## Troubleshooting

### If agents still show revisits after retraining:

1. **Check environment is using new reward**:

   ```python
   env = EnvFactory.create(config)
   obs, _ = env.reset()
   obs, reward, _, _, _ = env.step(1)  # Visit customer 1
   print(f"First customer visit reward: {reward}")  # Should be ~+9.9 (+10 -0.1)
   ```

2. **Verify action masking is working**:

   ```python
   # In evrp_env.py, _get_valid_actions() should mask visited customers
   valid_actions = env._get_valid_actions()
   print(f"Can revisit customer 1: {valid_actions[1]}")  # Should be False
   ```

3. **Increase training episodes**: 10k may not be sufficient, try 30k-50k

4. **Check hyperparameters**: Ensure entropy coefficient > 0 (exploration) and learning rate appropriate (3e-4 for A2C/SAC)

## Future Enhancements

Consider adding:

1. **Curriculum learning**: Start with 5 customers, gradually increase to 20
2. **Auxiliary rewards**: +0.5 for visiting nearest unvisited customer (exploration guidance)
3. **Adaptive penalties**: Increase revisit penalty over training (-1 → -2 → -5)
4. **Multi-objective rewards**: Separate distance and service objectives with dynamic weighting

---

**Status**: ✅ Implemented and ready for testing  
**Date**: 2026-02-16  
**Impact**: Expected 3-4x improvement in route efficiency
