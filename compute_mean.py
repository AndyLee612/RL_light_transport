import os, glob
import numpy as np
from tqdm import tqdm

def load_sa_for_norm(file_path: str):
    # mmap_mode helps reduce peak RAM usage
    data = np.load(file_path, mmap_mode="r")

    positions  = data["positions"]   # (N,3)
    normals    = data["normals"]     # (N,3)
    directions = data["directions"]  # (N,3)

    # Build (N,6) state; cast to float32 once
    S = np.concatenate([positions, normals], axis=1).astype(np.float32, copy=False)
    A = directions.astype(np.float32, copy=False)

    return S, A

def _merge_mean_var(n, mean, M2, nb, mean_b, M2_b):
    """
    Merge (n, mean, M2) with (nb, mean_b, M2_b)
    where M2 = sum (x-mean)^2  (per-dimension).
    """
    if n == 0:
        return nb, mean_b, M2_b
    delta = mean_b - mean
    nt = n + nb
    mean_new = mean + delta * (nb / nt)
    M2_new = M2 + M2_b + (delta * delta) * (n * nb / nt)
    return nt, mean_new, M2_new

def compute_norm_stats_streaming(dataset_root: str, out_path: str):
    pattern = os.path.join(dataset_root, "rl_reward_light*_batch_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")

    # Global accumulators for states
    nS = 0
    meanS = None
    M2S = None

    # Global accumulators for actions
    nA = 0
    meanA = None
    M2A = None

    for fp in tqdm(files, desc="[Norm] streaming stats", unit="file"):
        S, A = load_sa_for_norm(fp)  # lightweight loader above

        # states batch stats
        Sb = S.astype(np.float64, copy=False)  # float64 improves numerical stability
        nbS = Sb.shape[0]
        meanSb = Sb.mean(axis=0)
        M2Sb = ((Sb - meanSb) ** 2).sum(axis=0)

        if meanS is None:
            meanS = meanSb
            M2S = M2Sb
            nS = nbS
        else:
            nS, meanS, M2S = _merge_mean_var(nS, meanS, M2S, nbS, meanSb, M2Sb)

        # actions batch stats
        Ab = A.astype(np.float64, copy=False)
        nbA = Ab.shape[0]
        meanAb = Ab.mean(axis=0)
        M2Ab = ((Ab - meanAb) ** 2).sum(axis=0)

        if meanA is None:
            meanA = meanAb
            M2A = M2Ab
            nA = nbA
        else:
            nA, meanA, M2A = _merge_mean_var(nA, meanA, M2A, nbA, meanAb, M2Ab)

    # Convert to std (population std like np.std default ddof=0)
    varS = M2S / max(nS, 1)
    varA = M2A / max(nA, 1)

    state_mean = meanS.astype(np.float32)[None, :]
    state_std  = (np.sqrt(varS) + 1e-6).astype(np.float32)[None, :]

    action_mean = meanA.astype(np.float32)[None, :]
    action_std  = (np.sqrt(varA) + 1e-6).astype(np.float32)[None, :]

    np.savez(out_path,
             state_mean=state_mean, state_std=state_std,
             action_mean=action_mean, action_std=action_std)

    print(f"[Saved] {out_path}")
    print(" state_mean:", state_mean.shape, "state_std:", state_std.shape)
    print(" action_mean:", action_mean.shape, "action_std:", action_std.shape)

compute_norm_stats_streaming("./dataset", "norm_stats.npz")

