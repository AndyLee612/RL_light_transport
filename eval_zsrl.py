# agents/cfb/saved_models/sweep-z50-alpha0_01--seed42/best_model.pt/best-step40000.pickle

# eval_zsrl.py
# Evaluate a saved FB/CFB ZSRL-style model on eval_dataset.npz (GT radiance distribution)
# Compares GT luminance rewards over 1024 hemisphere dirs vs predicted Q(s,a,z).

import os
import glob
import argparse
from pathlib import Path

import numpy as np
import torch

import yaml

from data_loader import load_light_transport_npz

# Agent class is CFB (vcfb/mcfb)
from agents.cfb.agent import CFB


def luminance(rgb: np.ndarray) -> np.ndarray:
    """rgb: (..., 3) -> (...,)"""
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)


def compute_norm_stats_from_dataset(root_dir: str):
    """
    Recompute (state_mean, state_std, action_mean, action_std) from the LT training dataset,
    to apply the same normalization at eval time.
    """
    pattern = os.path.join(root_dir, "rl_reward_light*_batch_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No dataset files found under: {pattern}")

    all_s, all_a = [], []
    for fp in files:
        S, A, _, _ = load_light_transport_npz(fp)  # S: [N,6], A:[N,3]
        all_s.append(S)
        all_a.append(A)
    states = np.concatenate(all_s, axis=0)
    actions = np.concatenate(all_a, axis=0)

    state_mean = states.mean(axis=0, keepdims=True)
    state_std = states.std(axis=0, keepdims=True) + 1e-6
    action_mean = actions.mean(axis=0, keepdims=True)
    action_std = actions.std(axis=0, keepdims=True) + 1e-6

    return state_mean, state_std, action_mean, action_std


def normalize_eval(pos: np.ndarray, normal: np.ndarray, dirs: np.ndarray,
                   state_mean, state_std, action_mean, action_std):
    """
    pos: (K,3), normal:(K,3), dirs:(D,3)
    returns:
      states_norm: (K,6)
      dirs_norm: (D,3)
    """
    states = np.concatenate([pos, normal], axis=-1).astype(np.float32)  # (K,6)
    states_norm = (states - state_mean) / state_std

    dirs = dirs.astype(np.float32)
    dirs_norm = (dirs - action_mean) / action_std

    return states_norm.astype(np.float32), dirs_norm.astype(np.float32)


def load_best_pickle(path: str) -> str:
    """
    Accepts either:
      - direct path to *.pickle
      - a directory containing *.pickle
      - a "best_model.pt" directory containing *.pickle
    Returns a single pickle path.
    """
    p = Path(path)
    if p.is_file() and p.suffix == ".pickle":
        return str(p)

    if p.is_dir():
        cands = sorted(p.rglob("*.pickle"))
        if not cands:
            raise FileNotFoundError(f"No .pickle found under directory: {p}")
        # choose the last one (often best-stepXXXX.pickle)
        return str(cands[-1])

    raise FileNotFoundError(f"Invalid model path: {path}")


def try_load_agent_weights(agent, pickle_path: str, device: torch.device):
    """
    Tries a few common patterns:
      1) agent.load(pickle_path)
      2) torch.load -> state_dict
    """
    if hasattr(agent, "load"):
        try:
            agent.load(pickle_path)
            agent.to(device) if hasattr(agent, "to") else None
            return
        except Exception as e:
            print("[Warn] agent.load() failed, falling back to torch.load. Error:", e)

    obj = torch.load(pickle_path, map_location=device)

    # If it’s already a state dict:
    if isinstance(obj, dict):
        # common keys
        for key in ["state_dict", "model", "agent", "fb", "FB"]:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

        # Try loading into agent / agent.FB
        loaded = False
        if hasattr(agent, "load_state_dict"):
            try:
                agent.load_state_dict(obj, strict=False)
                loaded = True
            except Exception:
                pass

        if (not loaded) and hasattr(agent, "FB") and hasattr(agent.FB, "load_state_dict"):
            agent.FB.load_state_dict(obj, strict=False)
            loaded = True

        if not loaded:
            raise RuntimeError(
                "Could not load weights. Your pickle format doesn’t match common patterns.\n"
                "Open the pickle keys and adjust try_load_agent_weights()."
            )
        return

    raise RuntimeError(f"Unexpected pickle content type: {type(obj)}")


@torch.no_grad()
def infer_z_reward_weighted(agent, states_norm_t: torch.Tensor, r_scalar_t: torch.Tensor) -> torch.Tensor:
    """
    Simple z estimate: z ∝ mean( B(s) * r ).
    Requires agent.FB.backward_representation(states) -> (K, z_dim) or similar.
    """
    if not hasattr(agent, "FB") or not hasattr(agent.FB, "backward_representation"):
        raise AttributeError("Agent does not expose FB.backward_representation().")

    B = agent.FB.backward_representation(states_norm_t)  # (K, z_dim)
    # r_scalar_t: (K,) or (K,1)
    r = r_scalar_t.view(-1, 1)
    z = (B * r).mean(dim=0)  # (z_dim,)
    z = z / (z.norm() + 1e-8)
    return z


@torch.no_grad()
def compute_q_distribution(agent, states_norm: np.ndarray, dirs_norm: np.ndarray, z: torch.Tensor, device):
    """
    states_norm: (K,6)
    dirs_norm:   (D,3)
    z: (z_dim,)
    returns q: (K,D) using Q(s,a,z) = <F(s,a,z), z> (and average heads if two)
    """
    K = states_norm.shape[0]
    D = dirs_norm.shape[0]

    s = torch.tensor(states_norm, dtype=torch.float32, device=device)          # (K,6)
    a = torch.tensor(dirs_norm, dtype=torch.float32, device=device)            # (D,3)

    # Expand to all (K*D)
    s_rep = s[:, None, :].expand(K, D, 6).reshape(K * D, 6)                    # (K*D,6)
    a_rep = a[None, :, :].expand(K, D, 3).reshape(K * D, 3)                    # (K*D,3)
    z_rep = z[None, :].expand(K * D, -1)                                       # (K*D,z)

    # Forward representation -> (F1, F2) or single
    out = agent.FB.forward_representation(s_rep, a_rep, z_rep)

    if isinstance(out, (tuple, list)) and len(out) >= 2:
        F1, F2 = out[0], out[1]
        q1 = (F1 * z_rep).sum(dim=-1)
        q2 = (F2 * z_rep).sum(dim=-1)
        q = 0.5 * (q1 + q2)
    else:
        F = out
        q = (F * z_rep).sum(dim=-1)

    q = q.view(K, D).detach().cpu().numpy().astype(np.float32)
    return q


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman correlation for 1D arrays without scipy (rank corr)."""
    x_rank = x.argsort().argsort().astype(np.float32)
    y_rank = y.argsort().argsort().astype(np.float32)
    x_rank = (x_rank - x_rank.mean()) / (x_rank.std() + 1e-8)
    y_rank = (y_rank - y_rank.mean()) / (y_rank.std() + 1e-8)
    return float((x_rank * y_rank).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_npz", type=str, default="eval_dataset.npz")
    ap.add_argument("--dataset_root", type=str, default="./dataset")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--use_reward_weighted_z", action="store_true",
                    help="Infer z via z ∝ mean(B(s)*r) using GT radiance at best direction per state.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("[Info] device:", device)

    # -------------------------
    # Load eval dataset
    # -------------------------
    data = np.load(args.eval_npz)
    pos = data["pos"].astype(np.float32)        # (K,3)
    normal = data["normal"].astype(np.float32)  # (K,3)
    dirs = data["dirs"].astype(np.float32)      # (D,3)
    rad = data["radiance_gt"].astype(np.float32)  # (K,res,res,3)

    K = pos.shape[0]
    res = rad.shape[1]
    D = res * res
    rad_flat = rad.reshape(K, D, 3)
    r_gt = luminance(rad_flat)                  # (K,D)

    print("[Eval] K:", K, "D:", D, "dirs:", dirs.shape, "rad:", rad.shape)

    # -------------------------
    # Norm stats (match training)
    # -------------------------
    state_mean, state_std, action_mean, action_std = compute_norm_stats_from_dataset(args.dataset_root)
    states_norm, dirs_norm = normalize_eval(pos, normal, dirs, state_mean, state_std, action_mean, action_std)

    # -------------------------
    # Build agent skeleton (match your training hyperparams!)
    # -------------------------
    # Set these to the same as training config
    config_path = "agents/cfb/config.yaml"
    
    with open(config_path, "rb") as f:
        config = yaml.safe_load(f)

    observation_length = 6      # pos(3) + normal(3)
    action_length = 3           # direction vector (x, y, z)

    agent = CFB(
        observation_length=observation_length,
        action_length=action_length,
        preprocessor_hidden_dimension=config["preprocessor_hidden_dimension"],
        preprocessor_output_dimension=config["preprocessor_output_dimension"],
        preprocessor_hidden_layers=config["preprocessor_hidden_layers"],
        forward_hidden_dimension=config["forward_hidden_dimension"],
        forward_hidden_layers=config["forward_hidden_layers"],
        forward_number_of_features=config["forward_number_of_features"],
        backward_hidden_dimension=config["backward_hidden_dimension"],
        backward_hidden_layers=config["backward_hidden_layers"],
        actor_hidden_dimension=config["actor_hidden_dimension"],
        actor_hidden_layers=config["actor_hidden_layers"],
        preprocessor_activation=config["preprocessor_activation"],
        forward_activation=config["forward_activation"],
        backward_activation=config["backward_activation"],
        actor_activation=config["actor_activation"],
        z_dimension=config["z_dimension"],
        actor_learning_rate=config["actor_learning_rate"],
        critic_learning_rate=config["critic_learning_rate"],
        learning_rate_coefficient=config["learning_rate_coefficient"],
        orthonormalisation_coefficient=config["orthonormalisation_coefficient"],
        discount=config["discount"],
        batch_size=config["batch_size"],
        z_mix_ratio=config["z_mix_ratio"],
        gaussian_actor=config["gaussian_actor"],
        std_dev_clip=config["std_dev_clip"],
        std_dev_schedule=config["std_dev_schedule"],
        tau=config["tau"],
        device=config["device"],
        vcfb=config["vcfb"],
        mcfb=config["mcfb"],
        total_action_samples=config["total_action_samples"],
        ood_action_weight=config["ood_action_weight"],
        alpha=config["alpha"],
        target_conservative_penalty=config["target_conservative_penalty"],
        lagrange=config["lagrange"],
    )

    # -------------------------
    # Load best checkpoint
    # -------------------------
    pickle_path = load_best_pickle(args.model_path)
    print("[Load] using checkpoint:", pickle_path)
    try_load_agent_weights(agent, pickle_path, device)
    agent.train(False) if hasattr(agent, "train") else None

    # -------------------------
    # Choose / infer z
    # -------------------------
    # Default: a random z (useful just to sanity-check plumbing)
    z = torch.randn(50, device=device)
    z = z / (z.norm() + 1e-8)

    if args.use_reward_weighted_z:
        K = states_norm.shape[0]
        n = 16 # can use other values (number of sampled states)
        idx = np.random.choice(K, size=n, replace=False)

        r_state = r_gt.max(axis=1).astype(np.float32)  # (K,)
        s_t = torch.tensor(states_norm[idx], dtype=torch.float32, device=device)
        r_t = torch.tensor(r_state[idx], dtype=torch.float32, device=device)

        z = infer_z_reward_weighted(agent, s_t, r_t)
        print("[z] inferred reward-weighted z (subset), norm=", float(z.norm().cpu()))


    # -------------------------
    # Predict Q distribution
    # -------------------------
    q_pred = compute_q_distribution(agent, states_norm, dirs_norm, z, device)  # (K,D)

    # -------------------------
    # Metrics
    # -------------------------
    mse = float(((q_pred - r_gt) ** 2).mean())

    # Spearman per-state then mean
    spears = [spearman_corr(q_pred[i], r_gt[i]) for i in range(K)]
    spear_mean = float(np.mean(spears))

    # Top-1 agreement
    top1_q = q_pred.argmax(axis=1)
    top1_r = r_gt.argmax(axis=1)
    top1_acc = float((top1_q == top1_r).mean())

    print("\n=== EVAL RESULTS ===")
    print("Q vs GT reward (luminance) distribution")
    print("MSE:", mse)
    print("Mean Spearman:", spear_mean)
    print("Top-1 direction match rate:", top1_acc)
    print("====================\n")

    # Optional: dump a small debug npz
    out = "eval_results_debug.npz"
    np.savez(
        out,
        states_raw=np.concatenate([pos, normal], axis=-1),
        dirs_raw=dirs,
        rewards_gt=r_gt,
        q_pred=q_pred,
        z=z.detach().cpu().numpy(),
    )
    print("[Saved]", out)


if __name__ == "__main__":
    main()
