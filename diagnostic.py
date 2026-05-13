"""Test render_policy.render_tile() / render() in isolation."""
import sys, numpy as np, torch
import mitsuba as mi
import drjit as dr
mi.set_variant('llvm_ad_rgb')
print('[diag] imports ok', flush=True)

sys.path.insert(0, '.')
from render_policy import (
    Normalizer, load_fb_components, render_tile, render,
)

device = 'cpu'

print('[diag] loading checkpoint + normalizer + z...', flush=True)
actor, B = load_fb_components(
    './agents/osfb/saved_models/sweep-z50-alpha0_01--seed42/best_model.pt/best-step20000.pickle',
    device)
normalizer = Normalizer('./norm_stats.npz', device)
z = torch.from_numpy(np.load('./output/z.npy').astype(np.float32)).to(device)

scene = mi.load_dict(mi.cornell_box())
W, H = scene.sensors()[0].film().crop_size()
print(f'[diag] film size {W}x{H}', flush=True)

# -----------------------------------------------------------------
print('\n=== TEST 1: single render_tile (32x32, spp=16, depth=2) ===', flush=True)
tile = render_tile(scene, actor, z, normalizer,
                   x0=0, y0=0, tw=32, th=32, W=W, H=H,
                   spp=16, max_depth=2,
                   device=device, chunk_size=8192)
print(f'  PASS  shape={tile.shape}  mean={tile.mean():.4f}', flush=True)

# -----------------------------------------------------------------
print('\n=== TEST 2: two render_tile calls in a row ===', flush=True)
t1 = render_tile(scene, actor, z, normalizer,
                 x0=0, y0=0, tw=32, th=32, W=W, H=H,
                 spp=16, max_depth=2, device=device, chunk_size=8192)
print('  tile 1 ok', flush=True)
t2 = render_tile(scene, actor, z, normalizer,
                 x0=32, y0=0, tw=32, th=32, W=W, H=H,
                 spp=16, max_depth=2, device=device, chunk_size=8192)
print('  tile 2 ok', flush=True)
print('  PASS', flush=True)

# -----------------------------------------------------------------
print('\n=== TEST 3: full render() small (spp=16, depth=2, tile=64) ===', flush=True)
img = render(scene, actor, z, normalizer,
             spp=16, max_depth=2,
             device=device, chunk_size=8192, tile=64)
print(f'  PASS  shape={img.shape}  mean={img.mean():.4f}', flush=True)

# -----------------------------------------------------------------
print('\n=== TEST 4: full render() with REAL params (spp=64, depth=4) ===', flush=True)
img = render(scene, actor, z, normalizer,
             spp=64, max_depth=4,
             device=device, chunk_size=32768, tile=64)
print(f'  PASS  shape={img.shape}  mean={img.mean():.4f}', flush=True)

print('\nALL TESTS PASSED')