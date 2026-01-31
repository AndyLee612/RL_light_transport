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

# Open Questions and Proposed Solutions

#### Q1: Should the latent task vector $z$ be normalized during inference?

In the CFB / ZSRL framework, the latent task embedding
$z \in \mathbb{R}^{d_z}$ is used through an inner product
$$Q(s,a,z) = \langle F(s,a,z), z \rangle,$$ which makes the scale of $z$
directly affect the magnitude of predicted $Q$ values. During
evaluation, $z$ is either sampled randomly or inferred from data.

Need to Inspect the training code and verify whether: $z$ is explicitly normalized (e.g., $\|z\|_2 = 1$)

If $z$ is normalized during training, then evaluation must apply the same normalization to ensure consistency between training and inference.

#### Q2: Reward-weighted inference $z = B(s) \cdot R$ assumes $R$ is an immediate reward, but the evaluation dataset provides cumulative returns.

**Issue.** The ZSRL formulation infers task embeddings via
$$z \propto \mathbb{E}[B(s) \cdot R],$$ where $R$ denotes the
*immediate* reward signal associated with a transition. However, in the
current light transport evaluation setup, $r_{\text{gt}}$ is computed
using path tracing and represents an approximation of the *cumulative
return* (i.e., radiance integrated over multiple bounces), which is
conceptually closer to $Q^\ast(s,a)$ than to the per-step reward.

Using cumulative return in place of immediate reward breaks the
theoretical assumption underlying reward-weighted $z$ inference.

**Proposed Solution.** Extend the dataset to explicitly record both:

-   **Immediate reward** $r(s,a)$ (e.g., per-bounce radiance
    contribution, consistent with the offline training dataset), and

-   **Cumulative return** $\hat{Q}(s,a)$ (e.g., hemispherical radiance
    estimated via path tracing).

Then:

-   Use the immediate reward $r(s,a)$ for reward-weighted $z$ inference.

-   Use the cumulative return $\hat{Q}(s,a)$ exclusively for evaluating
    predicted $Q(s,a,z)$.

#### Q3: How should reward-weighted $z$ be inferred when actions are continuous and rewards are directional?

**Issue.** The expression $z = B(s) \cdot R$. In light transport, rewards depend on both
surface state $s$ and outgoing direction $a$. Simply taking a maximum
over directions or using a fixed grid does not reflect how rewards are
actually encountered under a policy.

**Proposed Solution.** Infer $z$ using a policy-consistent procedure:

1.  Sample a small set of states $\{s_i\}$ from the evaluation dataset.

2.  For each $s_i$, sample an action $a_i \sim \pi(a \mid s_i, z)$ using
    the learned policy.

3.  Obtain the corresponding *immediate* reward $r(s_i,a_i)$ by lookup
    or interpolation from the evaluation dataset.

4.  Infer the task embedding via
    $$z \propto \frac{1}{N} \sum_{i=1}^N B(s_i)\, r(s_i,a_i),$$ followed
    by normalization if required.

This procedure aligns reward-weighted $z$ inference with both the
theoretical ZSRL assumptions and the practical semantics of the light
transport environment.

### 4️⃣ What evaluation metrics are appropriate for continuous directional actions?

**Issue**

Mean Squared Error (MSE) between predicted `Q(s,a,z)` and ground-truth cumulative return `Q̂(s,a)` is straightforward and scale-aware, and is likely a reasonable baseline metric.

However, ranking-based metrics (e.g., Spearman correlation, top-1 direction match) require more careful consideration:

- Directions are discretized into a fixed grid (currently 1024 directions)
- The directional grid resolution itself influences the metric
- Compute full ranking metrics over 1024 directions is not reasonable

As a result, ranking metrics may over- or under-estimate performance in a way that is sensitive to discretization artifacts rather than model quality.

**Proposed Solution**

- Treat **MSE** as the primary evaluation metric
- Reconsider the use of ranking metrics, or modify them to be geometry-aware:
  - Compare only top-k directions instead of full rankings
  - Measure angular distance between predicted best direction and ground-truth best direction
- If rank-based metrics are reported, clearly state the discretization scheme and its limitations

---
