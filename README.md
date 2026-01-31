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

## 📥 Download Dataset

To generate the light transport dataset, please use the official rendering repository:

👉 **Repository:** https://github.com/IntelligentDecisionLab/RL_light_transport  

### Steps

1. Open the script: rendering/scripts/render.py

2. Choose which data to generate by commenting out the corresponding sections:

- **Training data**
  ```python
  # Comment out lines 531–563
  ```
  This disables evaluation-only rendering and produces **training data**.

- **Evaluation data**
  ```python
  # Comment out lines 617–633
  ```
  This disables training-only rendering and produces **evaluation data**.

3. Run the rendering script as instructed in the repository to generate the `.npz` files.

### Notes
- Training and evaluation data are generated **separately** to avoid data leakage.
- Ensure the correct section is commented out **before** running the renderer.
- The resulting datasets should be placed in the `dataset/` directory used by the training and evaluation scripts.

## Dependencies

Set up a Conda environment or a python venv with **Python 3.9**:

```bash
conda create --name zsrl python=3.9.16
conda activate zsrl
pip install -r requirements.txt
```

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

## 🧪 Evaluation (Zero-Shot)

During zero-shot evaluation, inputs are normalized using the **same mean and standard deviation computed from the training dataset**.

Normalization statistics are **precomputed offline** using `compute_norm_stats.py` (formerly `compute_mean.py`) and saved as `norm_stats.npz`.  
At evaluation time, these saved statistics are loaded and applied directly to states and actions before inference.

This ensures:
- consistent normalization between training and evaluation
- no need to load the full dataset during evaluation
- CPU-only, memory-efficient preprocessing
- no interference with GPU resources used for inference


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

## 🛠️ TODO

**Action required for co-authors**

The current evaluation script needs to be **revised and improved** to provide more reliable and informative metrics for the light transport setting.

Specifically, we need to:
- Review and improve the evaluation metrics used in zero-shot testing
- Ensure metrics properly reflect **directional distribution quality**, not just pointwise errors
- Verify consistency between training objectives and evaluation criteria
- Potentially add additional diagnostics (e.g., rank-based metrics, top-k accuracy, distributional similarity)

