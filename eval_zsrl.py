import argparse
from pathlib import Path
import numpy as np
import torch
import yaml
from tqdm import tqdm

from agents.cfb.agent import CFB


def luminance(rgb: np.ndarray) -> np.ndarray:
    """rgb: (..., 3) -> (...,)"""
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)


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


def try_load_agent_weights(agent, pickle_path: str, device: torch.device) -> None:
    """
    Tries a few common patterns:
      1) agent.load(pickle_path)
      2) torch.load -> state_dict
    """
    if hasattr(agent, "load"):
        try:
            agent.load(pickle_path)
            if hasattr(agent, "to"):
                agent.to(device)
            return
        except Exception as e:
            print("[Warn] agent.load() failed, falling back to torch.load. Error:", e)

    obj = torch.load(pickle_path, map_location=device)

    if isinstance(obj, dict):
        # common nested keys
        for key in ["state_dict", "model", "agent", "fb", "FB"]:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

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
            keys = list(obj.keys())[:50] if isinstance(obj, dict) else []
            raise RuntimeError(
                "Could not load weights. Your pickle format doesn’t match common patterns.\n"
                f"Top-level keys (up to 50): {keys}\n"
                "Open the pickle keys and adjust try_load_agent_weights()."
            )
        return

    raise RuntimeError(f"Unexpected pickle content type: {type(obj)}")


def load_norm_stats(file_path: str):
    """
    Load precomputed mean and std from a file.
    """
    data = np.load(file_path)
    state_mean = data['state_mean']
    state_std = data['state_std']
    action_mean = data['action_mean']
    action_std = data['action_std']
    
    return state_mean, state_std, action_mean, action_std


def normalize_eval(pos: np.ndarray, normal: np.ndarray, dirs: np.ndarray,
                   state_mean, state_std, action_mean, action_std):
    """
    pos: (K,3), normal:(K,3), dirs:(D,3)
    returns:
      states_norm: (K,6)
      dirs_norm: (D,3)
    """
    # Normalize states using the precomputed mean and std
    states = np.concatenate([pos, normal], axis=-1).astype(np.float32)  # (K,6)
    states_norm = (states - state_mean) / state_std

    dirs = dirs.astype(np.float32)
    dirs_norm = (dirs - action_mean) / action_std

    return states_norm.astype(np.float32), dirs_norm.astype(np.float32)


@torch.no_grad()
def infer_z_reward_weighted(
    agent, states_t: torch.Tensor, r_scalar_t: torch.Tensor
) -> torch.Tensor:
    """
    Simple z estimate: z ∝ mean( B(s) * r ).
    Requires agent.FB.backward_representation(states) -> (K, z_dim) or similar.
    NOTE: states_t is RAW states (no normalization).
    """
    if not hasattr(agent, "FB") or not hasattr(agent.FB, "backward_representation"):
        raise AttributeError("Agent does not expose FB.backward_representation().")

    B = agent.FB.backward_representation(states_t)  # (K, z_dim)
    r = r_scalar_t.view(-1, 1)                      # (K, 1)
    z = (B * r).mean(dim=0)                         # (z_dim,)
    z = z / (z.norm() + 1e-8)
    return z


@torch.no_grad()
def compute_q_distribution_chunked(
    agent,
    states: np.ndarray,
    dirs: np.ndarray,
    z: torch.Tensor,
    device: torch.device,
    chunk_size: int = 131072,
    use_amp: bool = True,
) -> np.ndarray:
    """
    states: (K,6) RAW
    dirs:   (D,3) RAW
    z: (z_dim,)
    returns q: (K,D) using Q(s,a,z) = <F(s,a,z), z> (and average heads if two)

    Chunked over flattened (K*D).
    """
    K = states.shape[0]
    D = dirs.shape[0]

    s = torch.tensor(states, dtype=torch.float32, device=device)  # (K,6)
    a = torch.tensor(dirs, dtype=torch.float32, device=device)    # (D,3)

    total = K * D
    q_out = torch.empty((total,), dtype=torch.float32, device="cpu")

    pbar = tqdm(range(0, total, chunk_size), desc="[Infer] Q(s,a,z) chunks", unit="chunk")
    for start in pbar:
        end = min(start + chunk_size, total)
        p = torch.arange(start, end, device=device)

        i = torch.div(p, D, rounding_mode="floor")  # (B,)
        j = p - i * D                                # (B,)

        s_batch = s[i]                               # (B,6)
        a_batch = a[j]                               # (B,3)
        z_batch = z.expand(end - start, -1)           # (B,z)

        if use_amp and device.type == "cuda":
            autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            # disabled autocast on non-cuda or when requested
            autocast_ctx = torch.autocast(device_type=device.type, enabled=False)

        with autocast_ctx:
            out = agent.FB.forward_representation(s_batch, a_batch, z_batch)

            if isinstance(out, (tuple, list)) and len(out) >= 2:
                F1, F2 = out[0], out[1]
                q1 = (F1 * z_batch).sum(dim=-1)
                q2 = (F2 * z_batch).sum(dim=-1)
                q = 0.5 * (q1 + q2)
            else:
                F = out
                q = (F * z_batch).sum(dim=-1)

        q_out[start:end] = q.detach().float().cpu()

    return q_out.view(K, D).numpy().astype(np.float32)


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
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--use_reward_weighted_z",
        action="store_true",
        help="Infer z via z ∝ mean(B(s)*r) using GT radiance at best direction per state.",
    )
    ap.add_argument("--z_dim", type=int, default=None, help="Override z dim (default: config z_dimension).")
    ap.add_argument("--z_seed", type=int, default=0, help="Seed for random z when not inferring.")
    ap.add_argument("--z_infer_n", type=int, default=16, help="Number of sampled states for reward-weighted z.")
    ap.add_argument("--chunk_size", type=int, default=131072, help="Chunk size for (K*D) inference.")
    ap.add_argument("--no_amp", action="store_true", help="Disable AMP autocast on CUDA.")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print("[Info] device:", device)

    # -------------------------
    # Load eval dataset
    # -------------------------
    data = np.load(args.eval_npz)
    pos = data["pos"].astype(np.float32)              # (K,3)
    normal = data["normal"].astype(np.float32)        # (K,3)
    dirs = data["dirs"].astype(np.float32)            # (D,3)
    rad = data["radiance_gt"].astype(np.float32)      # (K,res,res,3)

    # RAW state/action (no normalization)
    states = np.concatenate([pos, normal], axis=-1).astype(np.float32)  # (K,6)

    K = states.shape[0]
    res = rad.shape[1]
    D_dirs = dirs.shape[0]
    D_rad = res * res

    if D_dirs != D_rad:
        raise ValueError(
            f"[Eval] Mismatch: dirs has D={D_dirs}, but radiance_gt implies res*res={D_rad} (res={res}). "
            "Your eval_npz must be consistent."
        )

    rad_flat = rad.reshape(K, D_rad, 3)
    r_gt = luminance(rad_flat)                        # (K,D)

    print("[Eval] K:", K, "D:", D_dirs, "states:", states.shape, "dirs:", dirs.shape, "rad:", rad.shape)

    # -------------------------
    # Load normalization stats (from training data)
    # -------------------------
    state_mean, state_std, action_mean, action_std = load_norm_stats('norm_stats.npz')

    # -------------------------
    # Build agent skeleton (match training hyperparams!)
    # -------------------------
    config_path = "agents/cfb/config.yaml"
    with open(config_path, "rb") as f:
        config = yaml.safe_load(f)

    observation_length = 6
    action_length = 3

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
        device=device,
        vcfb=True,
        mcfb=False,
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
    if hasattr(agent, "train"):
        agent.train(False)

    # -------------------------
    # Choose / infer z
    # -------------------------
    torch.manual_seed(args.z_seed)
    np.random.seed(args.z_seed)

    z_dim = int(args.z_dim) if args.z_dim is not None else int(config["z_dimension"])

    z = torch.randn(z_dim, device=device)
    z = z / (z.norm() + 1e-8)

    if args.use_reward_weighted_z:
        n = min(args.z_infer_n, K)
        idx = np.random.choice(K, size=n, replace=False)

        # one scalar reward per state: best-direction GT luminance
        r_state = r_gt.max(axis=1).astype(np.float32)  # (K,)
        s_t = torch.tensor(states[idx], dtype=torch.float32, device=device)  # RAW states
        r_t = torch.tensor(r_state[idx], dtype=torch.float32, device=device)

        z = infer_z_reward_weighted(agent, s_t, r_t)
        print("[z] inferred reward-weighted z (subset), norm=", float(z.norm().detach().cpu()))

    # -------------------------
    # Predict Q distribution (chunked + tqdm)
    # -------------------------
    q_pred = compute_q_distribution_chunked(
        agent=agent,
        states=states,          # RAW
        dirs=dirs,              # RAW
        z=z,
        device=device,
        chunk_size=args.chunk_size,
        use_amp=(not args.no_amp),
    )  # (K,D)

    # -------------------------
    # Metrics
    # -------------------------
    mse = float(((q_pred - r_gt) ** 2).mean())

    spears = []
    for i in tqdm(range(K), desc="[Metric] Spearman per-state", unit="state"):
        spears.append(spearman_corr(q_pred[i], r_gt[i]))
    spear_mean = float(np.mean(spears))

    top1_q = q_pred.argmax(axis=1)
    top1_r = r_gt.argmax(axis=1)
    top1_acc = float((top1_q == top1_r).mean())

    print("\n=== EVAL RESULTS ===")
    print("Q vs GT reward (luminance) distribution")
    print("MSE:", mse)
    print("Mean Spearman:", spear_mean)
    print("Top-1 direction match rate:", top1_acc)
    print("====================\n")

    # -------------------------
    # Save debug
    # -------------------------
    out = "eval_results_debug.npz"
    np.savez(
        out,
        states_raw=states,     # already pos+normal
        dirs_raw=dirs,
        rewards_gt=r_gt,
        q_pred=q_pred,
        z=z.detach().cpu().numpy(),
    )
    print("[Saved]", out)


if __name__ == "__main__":
    main()

