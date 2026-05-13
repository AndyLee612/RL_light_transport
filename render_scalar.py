"""
Scalar-mode renderer: one ray at a time. Slow but bulletproof.
No vectorized drjit, no JIT, no torch/drjit conflict.

Usage:
    CUDA_VISIBLE_DEVICES=-1 python render_scalar.py \
        --ckpt ./agents/osfb/saved_models/.../best-step20000.pickle \
        --norm ./norm_stats.npz \
        --z    ./output/z.npy \
        --out  ./output/image.npz \
        --spp  4
"""
import argparse
import numpy as np
import torch
import mitsuba as mi
from PIL import Image

# scalar variant — one ray per call, no JIT
mi.set_variant("scalar_rgb")

# Reuse network classes from render_policy.py (must be importable)
import sys
sys.path.insert(0, ".")
from render_policy import (
    Normalizer, load_fb_components, actor_forward_chunked, save_png,
)


@torch.no_grad()
def policy_dir_batch(actor, normalizer, states, z, device, chunk_size=8192):
    """states: (N, 6) numpy -> world-space directions (N, 3) numpy.
    The actor was trained to emit world-space directions."""
    s_t = torch.from_numpy(states).to(device)
    obs = normalizer.norm_state(s_t)
    a   = torch.tanh(actor_forward_chunked(actor, obs, z, chunk_size))
    a   = a * normalizer.a_std + normalizer.a_mean
    a   = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    return a.cpu().numpy().astype(np.float32, copy=True)


def local_to_world(local, normal):
    """Transform a single (3,) local direction by an orthonormal frame
    built from `normal` (3,)."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 \
             else np.array([0.0, 1.0, 0.0])
    t = np.cross(helper, normal)
    t /= np.linalg.norm(t) + 1e-8
    b = np.cross(normal, t)
    return local[0] * t + local[1] * b + local[2] * normal


def render_scalar(scene, actor, z, normalizer, spp, max_depth, device):
    sensor = scene.sensors()[0]
    W, H = sensor.film().crop_size()
    img = np.zeros((H, W, 3), dtype=np.float32)

    sampler = mi.load_dict({"type": "independent"})

    for py in range(H):
        for px in range(W):
            pix_color = np.zeros(3, dtype=np.float32)
            for s in range(spp):
                # Random subpixel jitter
                jx, jy = np.random.rand(2)
                pos2d = mi.Point2f((px + jx) / W, (py + jy) / H)
                ray, _ = sensor.sample_ray(0.0, 0.0, pos2d, mi.Point2f(0.0))

                throughput = np.ones(3, dtype=np.float32)
                radiance   = np.zeros(3, dtype=np.float32)

                for depth in range(max_depth):
                    si = scene.ray_intersect(ray)
                    if not si.is_valid():
                        break

                    # Emission (scalar mode -> mi.Color3f or similar)
                    em = si.emitter(scene)
                    if em is not None:
                        e_dr = em.eval(si)
                        e = np.array([float(e_dr[0]), float(e_dr[1]), float(e_dr[2])],
                                     dtype=np.float32)
                        radiance += throughput * e

                    if depth == max_depth - 1:
                        break

                    pos = np.array([float(si.p.x), float(si.p.y), float(si.p.z)],
                                   dtype=np.float32)
                    nrm = np.array([float(si.sh_frame.n.x),
                                    float(si.sh_frame.n.y),
                                    float(si.sh_frame.n.z)], dtype=np.float32)
                    state = np.concatenate([pos, nrm])[None, :]   # (1, 6)

                    # Policy outputs WORLD-SPACE direction (no local_to_world).
                    world_dir = policy_dir_batch(actor, normalizer,
                                                 state, z, device)[0]

                    # Jitter the policy direction with a uniform cone around it
                    # to sample a *distribution* (path tracing needs stochasticity).
                    # Cone half-angle: ~30 degrees (cos = 0.866).
                    cos_max = 0.866
                    u1, u2 = np.random.rand(2)
                    cos_t  = 1.0 - u1 * (1.0 - cos_max)
                    sin_t  = np.sqrt(max(0.0, 1.0 - cos_t * cos_t))
                    phi    = 2.0 * np.pi * u2
                    # Local jitter direction (z aligned with policy direction)
                    j_local = np.array([sin_t * np.cos(phi),
                                        sin_t * np.sin(phi),
                                        cos_t], dtype=np.float32)
                    # Build frame around world_dir
                    pd = world_dir
                    helper_p = (np.array([1.0, 0, 0], dtype=np.float32)
                                if abs(pd[0]) < 0.9
                                else np.array([0.0, 1, 0], dtype=np.float32))
                    tp = np.cross(helper_p, pd); tp /= np.linalg.norm(tp) + 1e-8
                    bp = np.cross(pd, tp)
                    world_dir = j_local[0]*tp + j_local[1]*bp + j_local[2]*pd

                    # Force into upper hemisphere (relative to surface normal).
                    if np.dot(world_dir, nrm) < 0:
                        world_dir = world_dir - 2.0 * np.dot(world_dir, nrm) * nrm
                    world_dir = world_dir / (np.linalg.norm(world_dir) + 1e-8)

                    # BSDF needs the direction in the LOCAL frame for evaluation.
                    # Mitsuba's BSDF works in shading frame: z = normal.
                    # Reconstruct local-frame direction by projecting world_dir
                    # against the shading frame.
                    # Build orthonormal frame from nrm:
                    helper = (np.array([1.0, 0, 0], dtype=np.float32)
                              if abs(nrm[0]) < 0.9
                              else np.array([0.0, 1, 0], dtype=np.float32))
                    t = np.cross(helper, nrm); t /= np.linalg.norm(t) + 1e-8
                    b = np.cross(nrm, t)
                    local_dir = np.array([
                        float(np.dot(world_dir, t)),
                        float(np.dot(world_dir, b)),
                        float(np.dot(world_dir, nrm)),
                    ], dtype=np.float32)

                    wo_local = mi.Vector3f(float(local_dir[0]),
                                           float(local_dir[1]),
                                           float(local_dir[2]))
                    bv_dr = si.bsdf().eval(mi.BSDFContext(), si, wo_local)
                    bv = np.array([float(bv_dr[0]), float(bv_dr[1]), float(bv_dr[2])],
                                  dtype=np.float32)
                    throughput *= bv / (1.0 / (2.0 * np.pi))

                    # Spawn next ray in world space
                    new_o = pos + 1e-4 * world_dir
                    ray = mi.Ray3f(o=mi.Point3f(float(new_o[0]),
                                                float(new_o[1]),
                                                float(new_o[2])),
                                   d=mi.Vector3f(float(world_dir[0]),
                                                 float(world_dir[1]),
                                                 float(world_dir[2])))

                pix_color += radiance

            img[py, px] = pix_color / spp

        if (py + 1) % 8 == 0:
            print(f"  row {py+1}/{H}", flush=True)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--norm", required=True)
    ap.add_argument("--z", required=True)
    ap.add_argument("--out", default="image.npz")
    ap.add_argument("--spp", type=int, default=4)
    ap.add_argument("--max-depth", type=int, default=3)
    args = ap.parse_args()

    device = "cpu"
    print(f"[+] Device: {device}  (scalar mitsuba, single-ray)")

    normalizer = Normalizer(args.norm, device)
    actor, _   = load_fb_components(args.ckpt, device)
    z = torch.from_numpy(np.load(args.z).astype(np.float32)).to(device)
    print(f"[+] z norm: {z.norm().item():.3f}")

    scene = mi.load_dict(mi.cornell_box())
    img   = render_scalar(scene, actor, z, normalizer,
                          spp=args.spp, max_depth=args.max_depth,
                          device=device)

    np.savez(args.out, image=img)
    print(f"[+] Saved {args.out}  shape={img.shape}")
    save_png(img, args.out.rsplit(".", 1)[0] + ".png")


if __name__ == "__main__":
    main()