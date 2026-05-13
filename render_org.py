"""Render a Cornell box image two ways for comparison:

    1. Custom path tracer (simulate_batch_pt) — your hand-written sampler.
    2. Mitsuba's built-in mi.render — for sanity-check reference.

Writes BOTH .exr (HDR) AND .png (tonemapped) for each, so you can open the
PNGs directly in any image viewer.
"""

import os
import mitsuba as mi
import numpy as np
import cv2

# Pick the variant BEFORE importing modules that use Mitsuba.
mi.set_variant("llvm_ad_rgb")

# Your file render.py with simulate_batch_pt + tonemap.
# If your filename is different (e.g. data_gen.py), change `render` below.
from render import simulate_batch_pt, tonemap


if __name__ == "__main__":
    scene = mi.load_dict(mi.cornell_box())
    os.makedirs("./output", exist_ok=True)

    # ---- 1. Custom path tracer (your code) ----
    # simulate_batch_pt writes BOTH custom.exr AND custom.png internally.
    # spp_ must be a multiple of 256 (SPP_PER_BATCH).
    simulate_batch_pt(scene, spp_=16, out_path="./output/custom", GT=True)

    # ---- 2. Mitsuba's built-in renderer (reference) ----
    img = mi.render(scene, spp=16)
    mi.util.write_bitmap("./output/builtin.exr", img)

    # Also save a tonemapped PNG so you can open it directly.
    img_np = np.array(img)                       # (H, W, 3), HDR linear
    img_png = tonemap(img_np) * 255.0            # tonemap + scale to 0-255
    cv2.imwrite("./output/builtin.png", img_png)

    print("\n[Done] custom  -> ./output/custom.exr  + ./output/custom.png")
    print("[Done] builtin -> ./output/builtin.exr + ./output/builtin.png")