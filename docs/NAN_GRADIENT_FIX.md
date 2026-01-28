# NaN Gradient Fix Summary

## Problem

The A2C agent was producing NaN gradients during backpropagation when training with multi-step rollouts (>2 steps). The NaN gradients would appear in:

- Encoder layers (MLP)
- Actor network layers
- Critic network layers

This prevented the agent from training successfully.

## Root Cause

The primary issue was in the **entropy calculation** within the A2C agent's update method. Specifically:

1. **Entropy Computation with torch.where()**: The original code used `torch.where()` to handle cases where probability was 0:

   ```python
   entropy = torch.where(
       probs > 0,
       -probs * log_probs,
       torch.zeros_like(probs)
   ).sum(dim=-1).mean()
   ```

   This approach created problems during backpropagation because:

   - When action masking sets some logits to `-inf`, the corresponding probabilities become 0
   - `torch.where()` with varying conditions across the batch can cause gradient flow issues
   - The condition `probs > 0` creates a discontinuity that PyTorch's autograd struggles with

2. **Secondary Issue - 0 \* (-inf) = NaN**: When computing `-probs * log_probs` directly, PyTorch evaluates `0 * (-inf)` as `NaN` rather than 0, even though mathematically the limit should be 0.

## Solution

Replace the `torch.where()` approach with **clamping** the log probabilities:

```python
# Clamp log_probs to avoid -inf before multiplication
log_probs_safe = torch.clamp(log_probs, min=-20.0)  # e^(-20) ≈ 2e-9, effectively zero
entropy = -(probs * log_probs_safe).sum(dim=-1).mean()
```

**Why this works:**

- Clamping `-inf` to `-20` prevents NaN from ever being created
- For masked actions (prob ≈ 0), the contribution is negligible: `0 * (-20) = 0`
- The gradient flow is smooth and continuous
- `-20` is conservative enough that `e^(-20) ≈ 2×10^-9` is effectively zero for RL purposes

## Additional Improvements Made

While debugging, several other improvements were made to enhance numerical stability:

### 1. Network Initialization

**MLP Encoder:**

- Changed from default initialization to orthogonal initialization with `gain=√2`
- Added zero initialization for biases
- Removed LayerNorm (which can cause issues with zero variance)

**Actor-Critic Networks:**

- Orthogonal initialization with `gain=√2` for hidden layers
- Small gain (`gain=0.01`) for output layers to prevent large initial predictions
- Changed activation from ReLU to Tanh for better gradient flow

### 2. Input Normalization

**MLP Encoder Feature Preparation:**

- Changed from fixed scaling (divide by 100) to adaptive min-max normalization:

  ```python
  # Coordinates
  coords_min = node_coords.min()
  coords_max = node_coords.max()
  if coords_max > coords_min:
      node_coords_norm = (node_coords - coords_min) / (coords_max - coords_min + 1e-8)

  # Demands
  demand_max = node_demands.max()
  if demand_max > 0:
      node_demands_norm = node_demands / (demand_max + 1e-8)
  ```

- This ensures features are always in [0, 1] range regardless of problem scale

### 3. Advantage Normalization

Added variance check before normalization:

```python
if len(advantages) > 1:
    adv_std = advantages.std()
    if adv_std > 1e-4:  # Only normalize if there's sufficient variance
        advantages = (advantages - advantages.mean()) / (adv_std + 1e-6)
    else:
        advantages = advantages - advantages.mean()
```

### 4. Loss Function

Changed from MSE to Huber loss for critic:

```python
critic_loss = F.smooth_l1_loss(state_values, returns)
```

Huber loss is more robust to outliers than MSE.

## Testing Results

### Before Fix

```
Warning: Bad gradient in encoder.mlp.0.weight: 306 NaN out of 384
Warning: Bad gradient in actor.0.weight: 44118 NaN out of 66048
... (all layers had NaN gradients)
Skipping update due to NaN/Inf gradients
```

### After Fix

```
Episode 1:
  Steps: 20, Total Reward: -19.00
  Actor Loss: -0.2972, Critic Loss: 9.0475
  Mean Value: -0.0027, Entropy: 0.3178
Episode 2:
  Steps: 20, Total Reward: -19.00
  Actor Loss: -0.0757, Critic Loss: 9.3089
  Mean Value: -0.0476, Entropy: 0.0549
...
✓ Training completed successfully!
```

## Files Modified

1. **src/encoders/mlp_encoder.py**

   - Improved weight initialization
   - Better input normalization
   - Removed LayerNorm

2. **src/agents/a2c_agent.py**
   - Fixed entropy computation (main fix)
   - Improved network initialization
   - Better advantage normalization
   - Switched to Huber loss for critic

## Verification

All tests now pass:

- ✅ `test_nan_minimal.py` - 10-step rollout trains without NaN
- ✅ `test_training_success.py` - Multi-episode training works correctly
- ✅ `test_critic_only.py` - Critic trains independently
- ✅ `test_actor_only.py` - Actor trains independently

## Recommendations for Future Development

1. **Always clamp log probabilities** when computing entropy with action masking
2. **Use orthogonal initialization** for RL networks (especially with Tanh activations)
3. **Prefer adaptive normalization** over fixed scaling for input features
4. **Test individual loss components** separately when debugging NaN gradients
5. **Consider Huber loss** over MSE for value function regression in RL

## References

- Stable Baselines3 implementation: Uses similar entropy computation with clamping
- OpenAI Spinning Up: Recommends orthogonal initialization for policy networks
- PPO paper: Discusses importance of proper value function initialization
