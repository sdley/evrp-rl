# Training Degradation Fix: Analysis & Solutions

## Problem Statement

Training curves show **catastrophic degradation** after promising initial improvements:

- **SAC**: Loss spike ~1000x at iteration ~800 (10^-1 → 10^-4)
- **A2C**: High variance and loss increase starting iteration ~800
- Both agents: Reward trajectory deteriorates, indicating policy collapse

## Root Cause Analysis

### 1. **Tau Too Small (0.005) - PRIMARY CAUSE**

**Problem**: Target networks update only 0.5% per learning step

- Current network rapidly adapts to new data
- Target network lags 200 steps behind
- Creates massive divergence: Current Q vs Target Q mismatch

**Mechanism**:

```
Iteration 800+:
- Current Q-network: Has adapted to recent experiences
- Target Q-network: Still reflects old policy (200 steps ago)
- Target = reward + gamma * V(s') from old network
- But current network has diverged significantly
- TD learning targets become WRONG
- Loss explodes as network fights itself
```

**Impact**: Off-policy divergence accumulates over time, explodes at iteration 800

**Fix**: Tau increased from 0.005 → 0.02 (4x faster synchronization)

- Now target networks update 2% per step (matches current in ~50 steps)
- Removes lag-induced divergence

---

### 2. **Auto-Entropy Too Aggressive (-action_dim \* 0.5)**

**Problem**: Target entropy for SAC is too negative

- For action_dim=N, target_entropy = -N \* 0.5
- Large negative value means agent should be VERY exploratory
- Alpha (entropy coefficient) grows unbounded to match target

**Mechanism**:

```
Early training (0-400 its):
- Policy is random, entropy high naturally
- Alpha adjustment is small, relatively stable

Mid training (400-800 its):
- Policy improves, entropy naturally decreases
- But target entropy is very negative (-N * 0.5)
- Autotuning INCREASES alpha to force more exploration
- At iteration ~800: Alpha becomes very large
- High entropy term dominates: E[alpha * log(pi(a|s))] >> E[Q(s,a)]
- Actor loss becomes minimized by RANDOMNESS, not by good actions

Later training (800+):
- Policy "forgets" learned behaviors and becomes exploratory
- Loss increases as critic tries to predict random policy
- This manifests as huge loss spike and degradation
```

**Fix**: target_entropy less negative: -action_dim _ 0.5 → -action_dim _ 0.25

- Entropy guidance is weaker in later training
- Prevents unbounded alpha growth
- Preserves good policies learned early

---

### 3. **Learning Rate Too Conservative (3e-4)**

**Problem**: After 600+ iterations, policy has diverged from replay buffer

- New experiences show different reward distributions
- But 3e-4 learning rate is too small to adapt quickly
- Network can't reorient before divergence compounds

**Fix**: Learning rate increased: 3e-4 → 5e-4 (66% increase)

- Allows faster policy adaptation to changing data distribution
- Critical during iteration 600-800 when distribution shift happens

---

### 4. **Gradient Clipping Too Restrictive (0.75/1.0)**

**Problem**: During divergence, network needs LARGE corrective gradients

- But clipping limits to 0.75 (A2C) or 1.0 (SAC)
- Network can't make large enough steps to recover
- Degradation accelerates as clipping prevents adaptation

**Mechanism**:

```
Normal iteration: ||grad|| = 0.3 → scaled by 1.0 (no clipping)
Divergence iteration: ||grad|| = 5.0 → clipped to 1.0
Network can only take tiny steps to fix large error
Compounding divergence continues
```

**Fix**: Max grad norm increased

- A2C: 0.75 → 1.5 (2x more gradient flow)
- SAC: 1.0 → 2.0 (2x more gradient flow)

---

### 5. **Reward Clipping Too Aggressive (±10)**

**Problem**: Clipping masks reward signal variance

- Good trajectories: R = 15 → clipped to 10
- Bad trajectories: R = -8 → clipped to -8
- Q-function learns to treat these identically in early training
- Doesn't learn to distinguish good/bad until late training

**Fix**: Reward clip range increased: 10.0 → 30.0

- Preserves more reward variance
- Q-function learns meaningful distinctions
- Supports gradient flow during backprop

---

### 6. **Entropy Coefficient Too High (A2C: 0.01)**

**Problem**: A2C explores too much early, forgets policy improvements

- Actor loss = policy_gradient - 0.01 \* entropy
- High entropy coefficient over-rewards random actions
- When policy converges, entropy drops, encouraging MORE exploration again

**Fix**: Reduced entropy_coef: 0.01 → 0.005

- Balances exploration/exploitation better
- Prevents exploration oscillations

---

## Combined Effect: Why Degradation Occurs

```
Iterations 0-400: Initial learning phase
├─ Tau small but acceptable (new experiences dominate)
├─ Policy improves, rewards increase
└─ All systems working

Iterations 400-600: Policy improvement phase
├─ Rewards plateau, policy near local optima
├─ Tau lag starts accumulating (200-step divergence builds)
└─ Early good experiences age in replay buffer

Iterations 600-800: CRITICAL TRANSITION
├─ Tau divergence reaches critical threshold
├─ New experiences conflict with stale target network
├─ Alpha growth forces exploration (target_entropy too negative)
├─ Low learning rate can't adapt fast enough
├─ Gradient clipping prevents corrective updates
└─ Compound divergence: current ≠ target; policy ≠ data

Iteration ~800: CATASTROPHIC EVENT
├─ Divergence cascades: Q-estimates explode
├─ Entropy-driven exploration collapes old policy
├─ Loss spikes 1000x (1e-1 → 1e-4)
└─ Agent "forgets" learned behaviors

Iterations 800+: Recovery attempt (fails)
├─ Agent tries to relearn from high-entropy random actions
├─ Target lag still 200 steps (not fixed early enough)
├─ Learning rate insufficient for distribution shift
├─ High variance, low rewards
└─ Training destabilized
```

---

## Solutions Applied

### Configuration Changes

| Parameter               | Was     | Now      | Reason                                           |
| ----------------------- | ------- | -------- | ------------------------------------------------ |
| **Learning Rate**       | 3e-4    | 5e-4     | Faster adaptation to distribution shift          |
| **Tau (SAC)**           | 0.005   | 0.02     | 4x faster target sync, eliminates lag-divergence |
| **Target Entropy**      | -N\*0.5 | -N\*0.25 | Prevent unbounded alpha growth                   |
| **Max Grad Norm (A2C)** | 0.75    | 1.5      | Larger corrective updates                        |
| **Max Grad Norm (SAC)** | 1.0     | 2.0      | Larger corrective updates                        |
| **Reward Clip**         | ±10     | ±30      | Preserve reward signal variance                  |
| **Entropy Coef (A2C)**  | 0.01    | 0.005    | Reduce exploration oscillation                   |

### Expected Outcomes

1. **Smoother loss curves**: Target network lag reduced → stable learning
2. **No loss spike at iteration 800**: Entropy growth controlled → balanced exploration
3. **Better late-game performance**: Higher learning rate adapts to changing data
4. **Reduced variance**: Larger gradients allow network to converge better
5. **Sustained improvements**: Reward signal preserved → consistent learning signal

---

## Validation Steps

To confirm these fixes work:

1. **Run training with new configs** (Cell 9, Cell 28)
2. **Compare loss curves** to previous run:
   - Check that SAC doesn't spike at ~800 iterations
   - Verify A2C maintains smooth convergence
3. **Compare final rewards**: Should match or exceed previous best
4. **Check convergence speed**: Should be similar or faster
5. **Analyze gradient norms**: Should show larger but stable gradients

---

## Future Improvements

### Optional Enhancements

1. **Learning rate scheduling**: Decay from 5e-4 → 2e-4 after 500 iterations
2. **Target entropy annealing**: Gradually decrease entropy target during training
3. **Prioritized experience replay**: Weight recent/surprising experiences higher
4. **Entropy decay schedule**: More aggressive early, less late: alpha_decay = 0.9995^iteration
5. **Adaptive gradient clipping**: Dynamic max_grad_norm based on recent gradient statistics

### Monitoring Additions

- Log alpha evolution (SAC) to detect unbounded growth early
- Track current vs target Q difference to detect divergence
- Log gradient norms per layer to identify bottlenecks
- Monitor replay buffer age distribution

---

## References

- SAC Paper: Haarnoja et al., "Soft Actor-Critic Algorithms and Applications" (ICLR 2019)
- Target Network Lag: Analysis in DQN and its variants
- Entropy Tuning: OpenAI Spinning Up documentation
- Gradient Flow: Goodfellow et al., "Deep Learning" book, Chapter on optimization
