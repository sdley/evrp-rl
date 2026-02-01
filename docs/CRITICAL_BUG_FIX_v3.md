# Critical Bug Fix v3.0 - Reward Timing Issue

## 🚨 CRITICAL BUG DISCOVERED

### The Problem

Customer rewards (+10 each) were being computed AFTER state update, causing ALL customer rewards to be deferred to the final completion step instead of being given immediately.

### Evidence from Diagnostic

```
BEFORE FIX (v2.0):
  Step 1: Reward= -1.10  ← Should be +9.95
  Step 2: Reward= -1.10  ← Should be +9.95
  Step 3: Reward= -1.10  ← Should be +9.95
  Step 4: Reward= -1.10  ← Should be +9.95
  Step 5: Reward= -1.10  ← Should be +9.95
  Step 6: Reward=+92.90  ← All 50 points given here!
  Total: +87.40
```

This created a **sparse reward problem** where agents received no feedback for progress, only a massive delayed reward at the end.

### Root Cause

**File**: `src/env/evrp_env.py` - `step()` method

**Buggy Code Flow**:

```python
def step(self, action):
    # 1. Update state FIRST
    self._update_state(action)  # ← Sets visited_mask[action] = True

    # 2. Compute reward AFTER
    reward = self._compute_reward(action)  # ← Checks visited_mask[action]
    # Problem: visited_mask[action] is already True, so is_new_customer = False!
```

In `_compute_reward()`:

```python
is_new_customer = self._is_customer(next_node) and not self.visited_mask[next_node]
# visited_mask was ALREADY updated, so this is always False!
```

### The Fix

**Solution**: Compute reward BEFORE updating state

```python
def step(self, action):
    # ... validation ...

    # FIXED: Compute reward BEFORE state update
    reward = self._compute_reward(action)  # Check old visited_mask

    # THEN update state
    self._update_state(action)  # Update visited_mask

    # ... rest of code ...
```

Now `_compute_reward()` sees the OLD visited_mask before it's updated, correctly identifying new customer visits.

### Expected Results After Fix

```
AFTER FIX (v3.0):
  Step 1: Reward= +9.95  ← +10 customer -0.05 step
  Step 2: Reward= +9.95  ← +10 customer -0.05 step
  Step 3: Reward= +9.95  ← +10 customer -0.05 step
  Step 4: Reward= +9.95  ← +10 customer -0.05 step
  Step 5: Reward= +9.95  ← +10 customer -0.05 step
  Step 6: Reward=+~29    ← +30 completion +efficiency -0.05 step
  Total: ~+79 to +84
```

**Dense rewards now guide learning at every step!**

## 🎯 Additional Improvements in v3.0

### 1. Adjusted Reward Magnitudes

Since customer rewards are now immediate:

| Component        | Before         | After          | Reason                                  |
| ---------------- | -------------- | -------------- | --------------------------------------- |
| Step penalty     | -0.1           | -0.05          | Smoother gradients                      |
| Customer reward  | +10            | +10            | (now given immediately!)                |
| Completion bonus | +50            | +30            | Reduced since we got +50 from customers |
| Efficiency bonus | max(0, 50-len) | max(0, 30-len) | Proportionally reduced                  |

### 2. Reduced Entropy Coefficients

Agents were exploring too much, not exploiting learned knowledge:

| Agent | Entropy Before | After | Impact                     |
| ----- | -------------- | ----- | -------------------------- |
| A2C   | 0.1            | 0.03  | 3× less random exploration |
| SAC   | 0.3            | 0.1   | 3× less random exploration |

High entropy was causing:

- A2C: High variance (±40-54 reward)
- SAC: Complete failure (charging loops)
- Both: Poor train→eval transfer

### 3. Faster Learning Rates

| Parameter           | Before | After | Reason                                   |
| ------------------- | ------ | ----- | ---------------------------------------- |
| A2C lr              | 1e-4   | 3e-4  | Dense rewards → faster learning possible |
| SAC learning_starts | 1000   | 500   | Start learning sooner                    |
| SAC target_update   | ?      | 1     | Update every step for faster adaptation  |
| Epochs              | 100    | 50    | Faster iteration for testing             |

### 4. Added Value Loss Coefficient (A2C)

```python
'value_loss_coef': 0.5  # Balance actor and critic learning
```

## 📊 Expected Performance Improvements

### Before (v2.0 with bug):

| Agent   | Success | Steps | Training Reward | Eval Reward        |
| ------- | ------- | ----- | --------------- | ------------------ |
| A2C+MLP | 100%    | 329   | 65-80           | 87.4               |
| A2C+GAT | 0%      | 0     | 50-80           | -50                |
| SAC+MLP | 0%      | 128   | 65-80           | -51 (499 charges!) |
| SAC+GAT | 0%      | 156   | 60-80           | -52 (498 charges!) |

**Problems**:

- A2C: Succeeded but 55× slower than optimal
- A2C+GAT: Policy collapse (terminates immediately)
- SAC: Charging loop trap (infinite charging)
- High variance: ±40-54 reward

### After (v3.0 with fix):

| Agent   | Success | Steps | Training Reward | Eval Reward |
| ------- | ------- | ----- | --------------- | ----------- |
| A2C+MLP | 100%    | 8-15  | 75-85           | 80-85       |
| A2C+GAT | 90-100% | 8-15  | 75-85           | 80-85       |
| SAC+MLP | 80-100% | 8-20  | 70-80           | 75-80       |
| SAC+GAT | 80-100% | 8-20  | 70-80           | 75-80       |

**Expected improvements**:

- ✅ All agents succeed (vs 25% before)
- ✅ Efficient routes: 8-15 steps (vs 329)
- ✅ Stable learning: ±5-10 variance (vs ±40-54)
- ✅ No charging loops
- ✅ Dense feedback guides learning

## 🧪 Validation Checklist

- [ ] Run Cell 2: Reload modules
- [ ] Run Cell 4: Load v3.0 configs (should show improvement summary)
- [ ] Run Cell 7: Diagnostic test
  - [ ] Steps 1-5 show ~+9.95 reward each (not -1.10!)
  - [ ] Step 6 shows ~+29 reward (not +92.90!)
  - [ ] Total reward: 79-84 (similar to before)
- [ ] Run Cell 8: Train all agents
  - [ ] A2C agents: Success in <20 steps
  - [ ] SAC agents: >80% success (not 0%!)
  - [ ] Training rewards: smooth increase
  - [ ] No charging loops (charge visits < 5)
- [ ] Run Cells 9-14: Generate analysis
  - [ ] Compare before/after in visualizations

## 🎓 Lessons Learned

### 1. Reward Timing Matters

Even if total reward is correct, **when** rewards are given drastically affects learning:

- **Immediate feedback** → agent learns what actions lead to progress
- **Delayed feedback** → sparse reward, credit assignment problem

### 2. Order of Operations is Critical

```python
# WRONG: State update first
update_state()  # Changes state
reward = compute_reward()  # Sees new state, misses transitions

# RIGHT: Reward first
reward = compute_reward()  # Sees old state, captures transition
update_state()  # Then change state
```

### 3. Exploration vs Exploitation Balance

Too much entropy (0.1-0.3) prevents agents from exploiting learned policies:

- Training: High entropy finds solutions through random exploration
- Eval: Deterministic policy fails because exploration was doing the work
- Solution: Lower entropy (0.03-0.1) forces agents to learn robust policies

### 4. Diagnostic Tools are Essential

The detailed step-by-step diagnostic revealed the bug immediately:

```python
# Without diagnostic: "Agents not learning, not sure why"
# With diagnostic: "Rewards only at end step - timing bug!"
```

## 📁 Files Modified

1. `src/env/evrp_env.py`

   - Line ~477: Moved `reward = self._compute_reward(action)` BEFORE `self._update_state(action)`
   - Lines 320-365: Adjusted reward magnitudes (completion 50→30, step 0.1→0.05)

2. `examples/ablation_study.ipynb`

   - Cell 4: Updated all hyperparameters (entropy, learning rates, epochs)
   - Cell 7: Enhanced diagnostic with expected vs actual comparison

3. `docs/CRITICAL_BUG_FIX_v3.md` (this file)

## 🚀 Implementation Status

- ✅ Reward timing bug fixed
- ✅ Entropy coefficients reduced
- ✅ Learning rates adjusted
- ✅ Completion bonus adjusted
- ✅ Training epochs reduced (faster iteration)
- ✅ Diagnostic enhanced with validation checks
- ⏳ **Ready for testing!**

---

**Version**: 3.0  
**Date**: January 31, 2026  
**Status**: ✅ Implemented, awaiting validation  
**Priority**: CRITICAL - This fix should resolve all major training failures
