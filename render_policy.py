"""
Render Cornell box using a trained FB (Forward-Backward) policy.

Pipeline:
  1. Load FB checkpoint (actor + backward representation B).
  2. Compute task vector z from an eval dataset:
        z = mean_i  B(obs_i, action_i) * reward_i        (then L2-normalized)
  3. Render: at each surface hit, query actor(obs, z) -> 3D action,
     tanh+normalize -> local hemisphere direction -> next ray.
  4. Save HDR image to .npz.

Usage:
    python render_with_policy.py \
        --ckpt   /path/to/best-step20000.pickle \
        --eval   /path/to/eval_dataset.npz \
        --out    image.npz \
        --spp    64
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import mitsuba as mi
import drjit as dr
from PIL import Image

# Variant is set inside main() based on --variant flag


# =============================================================================
# 0. Normalization (must match training-time stats from norm_stats.npz)
# =============================================================================
class Normalizer:
    """Applies (x - mean) / std using stats computed on the training set."""
    def __init__(self, stats_path, device):
        s = np.load(stats_path)
        self.s_mean = torch.from_numpy(s["state_mean"]).to(device)   # (1, 6)
        self.s_std  = torch.from_numpy(s["state_std"]).to(device)
        self.a_mean = torch.from_numpy(s["action_mean"]).to(device)  # (1, 3)
        self.a_std  = torch.from_numpy(s["action_std"]).to(device)
        print(f"[+] Loaded norm stats from {stats_path}")
        print(f"    state_mean={self.s_mean.cpu().numpy().ravel()}")
        print(f"    state_std ={self.s_std.cpu().numpy().ravel()}")

    def norm_state(self, s):   return (s - self.s_mean) / self.s_std
    def norm_action(self, a):  return (a - self.a_mean) / self.a_std


# =============================================================================
# 1. Network modules — shapes taken directly from the checkpoint
# =============================================================================
def mlp(dims, activation="relu", layernorm_first=False, trailing_act=False):
    """Sequential MLP matching the official OneStepFB architecture.

    activation:      'relu' or 'gelu' between hidden Linear layers.
    layernorm_first: insert (LayerNorm, Tanh) right after the FIRST Linear
                     (used by the actor head).
    trailing_act:    append an activation AFTER the last Linear
                     (used by obs_preprocessor / obs_z_preprocessor:
                      Linear → ReLU → Linear → ReLU → Linear → ReLU).
    """
    act_cls = {"relu": nn.ReLU, "gelu": nn.GELU}[activation]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = (i == len(dims) - 2)
        if i == 0 and layernorm_first:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.Tanh())
        elif not is_last:
            layers.append(act_cls())
        elif is_last and trailing_act:
            layers.append(act_cls())
    return nn.Sequential(*layers)


class Trunk(nn.Module):
    """Wrapper so state_dict keys match `<name>.trunk.<idx>.weight`."""
    def __init__(self, dims, activation="relu",
                 layernorm_first=False, trailing_act=False):
        super().__init__()
        self.trunk = mlp(dims, activation=activation,
                         layernorm_first=layernorm_first,
                         trailing_act=trailing_act)

    def forward(self, x):
        return self.trunk(x)


class Actor(nn.Module):
    """Matches the official ActorModel exactly."""
    def __init__(self, obs_dim=6, z_dim=50, action_dim=3):
        super().__init__()
        # obs_preprocessor:    Linear→ReLU→Linear→ReLU→Linear→ReLU
        self.obs_preprocessor   = Trunk([obs_dim,         1024, 1024, 512],
                                        activation="relu", trailing_act=True)
        self.obs_z_preprocessor = Trunk([obs_dim + z_dim, 1024, 1024, 512],
                                        activation="relu", trailing_act=True)
        # actor head: Linear→LayerNorm→Tanh→Linear→GELU→...→Linear
        self.actor              = Trunk([1024, 512, 512, 512, 512, action_dim],
                                        activation="gelu",
                                        layernorm_first=True)

    def forward(self, obs, z):
        h_obs   = self.obs_preprocessor(obs)
        h_obs_z = self.obs_z_preprocessor(torch.cat([obs, z], dim=-1))
        h = torch.cat([h_obs, h_obs_z], dim=-1)
        return self.actor(h)


class BackwardNet(nn.Module):
    """BackwardModel: 9 → 512 → 512 → 512 → 512 → 50, all GELU between."""
    def __init__(self, in_dim=9, z_dim=50):
        super().__init__()
        self.B = Trunk([in_dim, 512, 512, 512, 512, z_dim], activation="gelu")

    def forward(self, x):
        return self.B(x)


# =============================================================================
# 2. Loading: pick the actor + B subtrees out of the full checkpoint
# =============================================================================
def load_fb_components(ckpt_path, device):
    full = torch.load(ckpt_path, map_location=device, weights_only=False)
    if hasattr(full, "state_dict"):
        full = full.state_dict()

    def subtree(prefix):
        return {k[len(prefix):]: v for k, v in full.items() if k.startswith(prefix)}

    actor = Actor().to(device).eval()
    actor.load_state_dict(subtree("actor."))

    B = BackwardNet().to(device).eval()
    B.load_state_dict(subtree("FB.backward_representation."))

    print("[+] Loaded actor and B network from checkpoint.")
    return actor, B


# =============================================================================
# 3. Compute z from eval dataset (OSFB inference recipe)
#
#    Standard FB / one-step FB inference:
#        z_raw = E_(s,a)~D [ B(s, a) * r(s, a) ]
#        z     = z_raw * sqrt(d) / ||z_raw||         (project onto sqrt(d)-sphere)
#
#    where:
#        D       = labeled eval dataset (positions, normals, directions, radiance)
#        r(s,a)  = scalar reward (we use mean radiance as a luminance proxy)
#        d       = z_dim (50 here)
#
#    The expectation is a Monte Carlo average over all (point × direction) pairs
#    in the eval dataset.
# =============================================================================
@torch.no_grad()
def compute_z(B, eval_npz_path, normalizer, device, z_dim=50, chunk_size=65536):
    d = np.load(eval_npz_path)
    required = ["pos", "normal", "dirs", "radiance_gt"]
    for k in required:
        assert k in d.files, f"eval npz missing key '{k}'. found: {d.files}"

    pos    = d["pos"].astype(np.float32)            # (K, 3)
    normal = d["normal"].astype(np.float32)         # (K, 3)
    dirs   = d["dirs"].astype(np.float32)           # (D, 3)
    rad_gt = d["radiance_gt"].astype(np.float32)    # (K, res, res, 3)

    K, D = pos.shape[0], dirs.shape[0]
    rad = rad_gt.reshape(K, D, 3)

    state_raw  = np.concatenate([pos, normal], -1)
    state_raw  = np.broadcast_to(state_raw[:, None, :], (K, D, 6))
    action_raw = np.broadcast_to(dirs[None, :, :], (K, D, 3))

    state_t  = torch.from_numpy(np.ascontiguousarray(state_raw )).to(device).reshape(-1, 6)
    action_t = torch.from_numpy(np.ascontiguousarray(action_raw)).to(device).reshape(-1, 3)
    state_n  = normalizer.norm_state(state_t)
    action_n = normalizer.norm_action(action_t)
    inp_t    = torch.cat([state_n, action_n], dim=-1)               # (K*D, 9)
    reward_t = torch.from_numpy(rad.mean(-1).reshape(-1, 1)).to(device)

    # Chunked B forward pass to bound memory
    z_acc = torch.zeros(z_dim, device=device, dtype=inp_t.dtype)
    n_total = inp_t.shape[0]
    for i in range(0, n_total, chunk_size):
        b_out = B(inp_t[i : i + chunk_size])                        # (chunk, z_dim)
        z_acc += (b_out * reward_t[i : i + chunk_size]).sum(dim=0)
    z_raw = z_acc / n_total
    z     = z_raw * np.sqrt(z_dim) / (z_raw.norm() + 1e-8)

    print(f"[+] z computed from {K} eval pts × {D} dirs ({n_total} samples)")
    print(f"    raw  ||z||={z_raw.norm():.3f}   mean reward={reward_t.mean():.3f}")
    print(f"    final ||z||={z.norm():.3f}  (target sqrt(d)={np.sqrt(z_dim):.3f})")
    return z


# =============================================================================
# 4. Renderer
# =============================================================================
def vec3_to_np(v):
    return np.stack([np.array(v.x), np.array(v.y), np.array(v.z)], axis=-1)

def np_to_vec3(arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return mi.Vector3f(mi.Float(arr[:, 0].copy()),
                       mi.Float(arr[:, 1].copy()),
                       mi.Float(arr[:, 2].copy()))


@torch.no_grad()
def actor_forward_chunked(actor, obs, z, chunk_size=65536):
    """Run actor on `obs` in chunks to bound peak memory."""
    if obs.shape[0] <= chunk_size:
        z_b = z.unsqueeze(0).expand(obs.shape[0], -1)
        return actor(obs, z_b)
    out = []
    for i in range(0, obs.shape[0], chunk_size):
        sl  = obs[i : i + chunk_size]
        z_b = z.unsqueeze(0).expand(sl.shape[0], -1)
        out.append(actor(sl, z_b))
    return torch.cat(out, dim=0)


@torch.no_grad()
def policy_directions(actor, normalizer, state_np, z, chunk_size, device):
    """Pure numpy in / numpy out wrapper around the actor.
    Keeps torch tensors local; nothing torch-related survives this call."""
    state_t = torch.from_numpy(state_np).to(device)
    obs     = normalizer.norm_state(state_t)
    a_raw   = torch.tanh(actor_forward_chunked(actor, obs, z, chunk_size))
    a       = a_raw * normalizer.a_std + normalizer.a_mean
    a       = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    a[..., 2] = a[..., 2].abs()
    out = a.detach().cpu().numpy().astype(np.float32, copy=True)
    del state_t, obs, a_raw, a
    return out


@torch.no_grad()
def render_tile(scene, actor, z, normalizer, x0, y0, tw, th, W, H,
                spp, max_depth, device, chunk_size):
    """Render a single (tw x th) tile starting at pixel (x0, y0)."""
    n_lanes = tw * th * spp

    sampler = mi.load_dict({"type": "independent"})
    sampler.set_sample_count(spp)
    sampler.seed(x0 * 1000003 + y0, wavefront_size=n_lanes)

    idx = dr.arange(mi.UInt32, n_lanes)
    pix = idx // spp
    lx, ly = pix % tw, pix // tw
    j = sampler.next_2d()
    pos2d = mi.Vector2f(mi.Float(lx) + j.x + x0,
                        mi.Float(ly) + j.y + y0) / mi.ScalarVector2f(W, H)

    sensor = scene.sensors()[0]
    rays, _ = sensor.sample_ray(0.0, 0.0, pos2d, mi.Vector2f(0.0))

    L          = mi.Spectrum(0)
    throughput = mi.Spectrum(1)
    active     = mi.Bool(True)

    si = scene.ray_intersect(rays, active)
    active &= si.is_valid()

    for depth in range(max_depth):
        L = L + throughput * si.emitter(scene).eval(si)
        if depth == max_depth - 1:
            break

        # ---- drjit -> numpy (fully detached) ----
        pos_np = vec3_to_np(si.p).astype(np.float32, copy=True)
        nrm_np = vec3_to_np(si.sh_frame.n).astype(np.float32, copy=True)
        state_np = np.ascontiguousarray(np.concatenate([pos_np, nrm_np], -1))

        # ---- torch (no drjit references kept) ----
        local_dir = policy_directions(actor, normalizer, state_np,
                                      z, chunk_size, device)

        # ---- numpy -> drjit ----
        wo_local = np_to_vec3(local_dir)
        wo_world = si.sh_frame.to_world(wo_local)

        bsdf     = si.bsdf(rays)
        bsdf_val = bsdf.eval(mi.BSDFContext(), si, wo_local, active)
        throughput = throughput * bsdf_val / (1.0 / (2.0 * np.pi))

        rays = si.spawn_ray(wo_world)
        si   = scene.ray_intersect(rays, active)
        active &= si.is_valid()

    tile = vec3_to_np(L).reshape(th, tw, spp, 3).mean(axis=2)
    return tile.astype(np.float32)


@torch.no_grad()
def render(scene, actor, z, normalizer, spp=64, max_depth=4,
           device="cpu", chunk_size=65536, tile=64):
    sensor = scene.sensors()[0]
    W, H = sensor.film().crop_size()
    img = np.zeros((H, W, 3), dtype=np.float32)

    n_tiles = ((H + tile - 1) // tile) * ((W + tile - 1) // tile)
    done = 0
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            th = min(tile, H - y0)
            tw = min(tile, W - x0)
            img[y0:y0+th, x0:x0+tw] = render_tile(
                scene, actor, z, normalizer, x0, y0, tw, th, W, H,
                spp, max_depth, device, chunk_size)
            done += 1
            print(f"  tile {done}/{n_tiles}  ({x0},{y0}) {tw}x{th}")
    return img


# =============================================================================
# 5. Tonemap HDR -> 8-bit sRGB-ish PNG
# =============================================================================
def save_png(img_hdr, path, gamma=2.2):
    x = np.clip(img_hdr, 0, None)
    x = (x / (1 + x)) ** (1 / gamma)         # Reinhard + gamma
    x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(x).save(path)
    print(f"[+] Saved {path}")


# =============================================================================
# 6. Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Path to FB checkpoint (.pickle)")
    ap.add_argument("--norm", required=True,
                    help="Path to norm_stats.npz (training-time mean/std)")
    ap.add_argument("--eval", default=None,
                    help="Path to eval_dataset.npz (required unless --z is given)")
    ap.add_argument("--z", default=None,
                    help="Path to a precomputed z .npy file (skips eval computation)")
    ap.add_argument("--save-z", default=None,
                    help="If set, save the computed z to this .npy path")
    ap.add_argument("--out",  default="image.npz",
                    help="Output HDR npz path (PNG saved alongside)")
    ap.add_argument("--spp",  type=int, default=64)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=65536,
                    help="Max rows per network forward pass")
    ap.add_argument("--tile", type=int, default=64,
                    help="Tile size in pixels (smaller -> less memory, slower)")
    ap.add_argument("--variant", default="llvm_ad_rgb",
                    choices=["llvm_ad_rgb", "cuda_ad_rgb"],
                    help="Mitsuba variant. llvm_ad_rgb = CPU vectorized (default, "
                         "stable with small tiles). cuda_ad_rgb = GPU (fastest, "
                         "needs careful memory mgmt with torch+CUDA).")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="Where to run the policy network. Use 'cpu' if mitsuba+torch+CUDA segfault together.")
    args = ap.parse_args()

    # Set Mitsuba variant before any scene/sensor calls
    mi.set_variant(args.variant)
    print(f"[+] Mitsuba variant: {args.variant}")

    if args.eval is None and args.z is None:
        ap.error("must provide either --eval (compute z) or --z (load z)")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[+] Device: {device}")

    normalizer = Normalizer(args.norm, device)
    actor, B = load_fb_components(args.ckpt, device)

    # --- Get z: either load from disk or compute from eval dataset ---
    if args.z is not None:
        z = torch.from_numpy(np.load(args.z).astype(np.float32)).to(device)
        print(f"[+] Loaded z from {args.z}  ||z||={z.norm():.3f}")
    else:
        z = compute_z(B, args.eval, normalizer, device, chunk_size=args.chunk_size)
        if args.save_z is not None:
            np.save(args.save_z, z.cpu().numpy())
            print(f"[+] Saved z to {args.save_z}")

    scene = mi.load_dict(mi.cornell_box())
    img   = render(scene, actor, z, normalizer,
                   spp=args.spp, max_depth=args.max_depth,
                   device=device, chunk_size=args.chunk_size,
                   tile=args.tile)

    np.savez(args.out, image=img)
    print(f"[+] Saved {args.out}  shape={img.shape}  dtype={img.dtype}")

    # Also save a viewable PNG next to the npz
    png_path = args.out.rsplit(".", 1)[0] + ".png"
    save_png(img, png_path)


if __name__ == "__main__":
    main()