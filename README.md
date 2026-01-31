# Applying Zero-Shot Reinforcement Learning to Light Transport

This repository applies **Zero-Shot Reinforcement Learning (ZSRL)** methods from the paper  
**_Zero-Shot Reinforcement Learning from Low-Quality Data_** to a **light transport** problem.

The goal is to demonstrate how ZSRL-style Forward–Backward (FB / CFB) representations can be used
to reason about **radiance distributions** and **directional rewards** without task-specific retraining.

---

## Problem Overview: Light Transport

The **light transport** problem can be described as follows:

- A virtual camera is placed inside a 3D scene.
- For each pixel in a 2D image plane, a ray is traced into the scene.
- The ray may bounce multiple times before reaching a light source or terminating.
- The **pixel intensity** is computed from the accumulated radiance along the ray.

From an RL perspective:

| Component | Interpretation |
|---------|----------------|
| State   | Camera position + surface normal (6D) |
| Action  | Ray direction (3D) |
| Reward  | RGB radiance (or luminance) |
| Policy  | Direction selection strategy |
| Value   | Expected radiance from a given direction |

This formulation allows us to apply **offline RL and ZSRL** methods to light transport data.

---

## Repository Structure

### Core Files

| File | Description |
|-----|-------------|
| `main_exorl.py` | Training pipeline for offline RL / ZSRL agents |
| `eval_zsrl.py` | Zero-shot evaluation on held-out light transport scenes |
| `data_loader.py` | Loads `.npz` light transport datasets |
| `agents/` | Implementation of CQL, FB, CFB, TD3, SF, etc. |
| `agents/workspaces_lt.py` | Training loop and logging utilities |

---

## Algorithms Implemented

- **CQL** – Conservative Q-Learning  
- **TD3** – Twin Delayed DDPG  
- **FB** – Forward–Backward Representation Learning  
- **CFB (VCFB / MCFB)** – Value-Conservative Forward–Backward  
- **GCIQL** – Generalized Conservative IQL  
- **SF** – Successor Features  

Primary algorithm used in experiments:

> **VCFB — Value-Conservative Forward Backward**

---

## Dataset Format

Datasets consist of `.npz` files:

```
rl_reward_light<ID>_batch_<B>.npz
```

Each file contains:
- `S`: states `(N, 6)`
- `A`: actions `(N, 3)`
- `R`: RGB rewards `(N, 3)`
- `S_next`: next states `(N, 6)`

---

## Training

### Example Command

```bash
CUDA_VISIBLE_DEVICES=0 nohup python main_exorl.py vcfb --seed 42 > test_run_full.log 2>&1 &
```

---

## Evaluation (Zero-Shot)

```bash
nohup python eval_zsrl.py \
  --eval_npz eval_dataset.npz \
  --model_path agents/cfb/saved_models/<run_name> \
  --device cuda \
  --use_reward_weighted_z \
  --chunk_size 32768 \
  > eval_zsrl.out 2>&1 &
```

### Metrics
- MSE
- Mean Spearman correlation
- Top-1 direction accuracy

---

## Normalization

- Dataset-wide normalization is computed during training
- Statistics are saved to `norm_stats.npz`
- Evaluation **must reuse the same stats**

---
