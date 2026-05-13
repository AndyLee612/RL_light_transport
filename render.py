import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import cv2
import numpy as np
# setting up the scene and render the image
import torch 
import os, sys

def move_light(scene_dict, xyz):
    if "light" not in scene_dict:
        raise KeyError("No 'light' node found in scene dictionary.")
    
    old = scene_dict["light"]["to_world"]

    # Convert DrJit matrix to numpy
    M = np.array(old.matrix)

    M[:3, 3] = xyz

    # Build new transform with same rotation and new translation
    scene_dict["light"]["to_world"] = mi.ScalarTransform4f(M)

    return scene_dict

# Example Usage
# scene_dict = mi.cornell_box()
# scene_dict = move_light(scene_dict, xyz=(0.3, 1.5, -0.2))
# scene = mi.load_dict(scene_dict)

def tonemap(input, gamma=2.2):

    img = np.copy(input)
    print(img.shape)
    img[...,0] = input[...,2]
    img[...,2] = input[...,0]
    if type(img) == torch.Tensor:
        img = torch.clamp(img, min=0.0, max=None)
        img = (img / (1 + img)) ** (1.0 / gamma)
    elif type(img) == np.ndarray:
        img = np.clip(img, 0.0, None)
        img = (img / (1 + img)) ** (1.0 / gamma)
    else:
        assert(False)

    return img

def render_final_image():
    print(mi.variants())
    mi.set_variant('llvm_ad_rgb')
    scene = mi.load_dict(mi.cornell_box())
    image = mi.render(scene, spp=64)

    print(image.shape)
    plt.axis("off")
    plt.imshow(image ** (1.0 / 2.2)); # approximate sRGB tonemapping
    plt.show()
    mi.util.write_bitmap("cbox.png", image)
    mi.util.write_bitmap("cbox.exr", image)

# dump the data for RL training

INV_PI = 0.31830988618
M_PI = 3.14159265359
### some global settings 
SCALE = 2.0

def generate_primary_rays(scene, sensor, sampler, spp = 4,if_stratify=False, sample2=None):
    film = sensor.film()
    film_size = film.crop_size()
    rfilter = film.rfilter()
    border_size = rfilter.border_size()

    if film.sample_border():
        film_size += 2 * border_size
    # spp = sampler.sample_count()
    # print("============================ spp. sampler", spp)

    # Compute discrete sample position
    idx = dr.arange(mi.UInt32, dr.prod(film_size) * spp)
    # Try to avoid a division by an unknown constant if we can help it
    log_spp = dr.log2i(spp)
    if 1 << log_spp == spp:
        idx >>= dr.opaque(mi.UInt32, log_spp)
    else:
        idx //= dr.opaque(mi.UInt32, spp)

    # Compute the position on the image plane
    pos = mi.Vector2i()
    pos.y = idx // film_size[0]
    pos.x = dr.fma(mi.Int32(-film_size[0]), pos.y, idx)
    if film.sample_border():
        pos -= border_size

    pos += mi.Vector2i(film.crop_offset())

    # Cast to floating point and add random offset
    # print(pos)
    # print(sampler.next_2d())
    if not if_stratify:
        pos_f = mi.Vector2f(pos) + sampler.next_2d()
    else:
        pos_f = mi.Vector2f(pos) + sample2
# Re-scale the position to [0, 1]^2
    scale = dr.rcp(mi.ScalarVector2f(film.crop_size()))
    offset = -mi.ScalarVector2f(film.crop_offset()) * scale
    pos_adjusted = dr.fma(pos_f, scale, offset)

    aperture_sample = mi.Vector2f(0.0)
    # if sensor.needs_aperture_sample():
    #     aperture_sample = sampler.next_2d()

    time = sensor.shutter_open()
    # if sensor.shutter_open_time() > 0:
    #     time += sampler.next_1d() * sensor.shutter_open_time()

    wavelength_sample = 0
    # if mi.is_spectral:
    #     wavelength_sample = sampler.next_1d()

    # with dr.resume_grad():  ###TODO
    ray, weight = sensor.sample_ray_differential(
        time=time,
        sample1=wavelength_sample,
        sample2=pos_adjusted,
        sample3=aperture_sample
    )

    reparam_det = 1.0
    # With box filter, ignore random offset to prevent numerical instabilities
    splatting_pos = mi.Vector2f(pos) if rfilter.is_box_filter() else pos_f

    return ray, weight, splatting_pos  # , reparam_det


def mis_weight(pdf_a, pdf_b):
    a2 = dr.sqr(pdf_a)
    return dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0)

def stratified_sampling_2d(n_pixels, spp_):
    #round spp to square number
    side = 1
    while (side*side< spp_):
        side += 1
    us = torch.arange(0, side)/side 
    vs = torch.arange(0, side)/side
    u, v = torch.meshgrid(us, vs)
    uv = torch.stack([u,v], dim = -1)
    uv = uv.reshape(-1,2)[:spp_,...]
    uv = uv[torch.randperm(uv.shape[0]), ...]
    uv_all_pixels = uv.unsqueeze(0).repeat(n_pixels, 1,1).reshape(-1,2)
    wavefront_size = n_pixels * spp_
    jitter = torch.rand((wavefront_size, 2))/side
    return uv_all_pixels + jitter 


def uniform_sampling(sample2):
    theta = sample2[0] * np.pi
    phi = sample2[1] * np.pi * 2.0 - np.pi
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    return mi.Vector3f(np.stack([x,y, np.sqrt(1 -np.clip(x*x +y*y,0.0,1.0))], axis=-1))
          
def test_sampler(scene_path, spp_ = 128, out_path="./output/pt.exr", neumip_ckpt_path = "../tortoise_shell.ckpt", use_offset = True):
    # settings
    mi.set_variant("llvm_ad_rgb")
    max_depth = 2
    SEED = 43
    np.random.seed(SEED)
    ####################################
    scene = mi.load_file(scene_path)
    sensors = scene.sensors()
    sensor = sensors[0]
    film_size = sensor.film().crop_size()
    n_pixels = dr.prod(film_size)
    sampler = mi.load_dict({'type' : 'independent'})
    # sampler = mi.load_dict({'type' : 'ldsampler'})
    sampler.set_sample_count(spp_)
    # spp = sampler.sample_count()
    spp = spp_
    sampler.seed(SEED, wavefront_size=dr.prod(film_size)*spp)  # important!! bug for longtime
    # Standard BSDF evaluation context for path tracing+
    # generate the batch of primary rays
    sample_buffer_2d = mi.Point2f(np.array(stratified_sampling_2d(film_size[0]*film_size[1], spp)))
    prim_rays, weights, splatting_pos = generate_primary_rays(
        scene, sensor, sampler, spp, True, sampler.next_2d())
    si_pre = scene.ray_intersect_preliminary(
        prim_rays)  # primary intersections
    si = si_pre.compute_surface_interaction(prim_rays)
    valid = si.is_valid()  # lanes of bool
    rays = mi.Ray3f(prim_rays)
    depth = 0 #depth = mi.UInt32(0)                          # Depth of current vertex
    L = mi.Spectrum(0)                            # Radiance accumulator
    throughput = mi.Spectrum(1)
    active = mi.Bool(valid)                      # Active SIMD lanes

    num_lanes = sampler.wavefront_size()
    assert(num_lanes == (n_pixels *spp))  #   print(" %d lanes...." % num_lanes)
    while depth < max_depth:
        # -----------------------------Direct sampling -----------------------
        Le = throughput * si.emitter(scene).eval(si)
        L = L + Le
        depth += 1
        if (depth >= max_depth): break

         # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()
        bsdf = si.bsdf(rays)
        
        #------------------ bsdf sampling ------------------------
        tmp_2d_samples = np.array(stratified_sampling_2d(n_pixels, spp))
        sample_buffer_2d = mi.Point2f(tmp_2d_samples)
        # sample_buffer_1d = mi.Float(np.array(stratified_sampling_1d(n_pixels, spp)))
        sample_buffer_1d = mi.Float(tmp_2d_samples[...,0])

        bs, bsdf_val = bsdf.sample(bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active)  # only for general materials
        # bs, bsdf_val = bsdf.sample(bsdf_ctx, si, sampler.next_1d(), sample_buffer_2d, active)  # only for general materials
        # new_dir = bs.wo
        throughput[active] *= bsdf_val 
        # bs, bsdf_val = bsdf.sample(bsdf_ctx, si, sample_buffer_1d, sample_buffer_2d, active)  # only for general materials
        new_dir = uniform_sampling(sample_buffer_2d)
        # Update throughput and rays for next bounce
           
        
        rays[active] = si.spawn_ray(si.sh_frame.to_world(new_dir))
        # pdf = bsdf.pdf(bsdf_ctx, si, new_dir) #TODO
        #######--------------------for next interaction ---------------
        si_pre = scene.ray_intersect_preliminary(rays, active)
        # update the neumip material mask at current intersections
        si = si_pre.compute_surface_interaction(rays)
        active &= si.is_valid()
        
    print("=================== cosine sampling for output image =========================")
    # image = mi.util.convert_to_bitmap(L)
    L = np.array(L).reshape(film_size[0], film_size[1], spp, 3)
    L = np.mean(L, axis=2)
    image = mi.TensorXf(L)
    mi.util.write_bitmap(out_path+".exr", image)
    cv2.imwrite(out_path+".png", tonemap(L)*255.0)
    
    
MIS = True
STRATIFY_BSDF = False

def simulate_batch_pt(scene, spp_ = 1024, out_path="./output/pt", GT = True, light_id=0):
    # settings
    # mi.set_variant("llvm_ad_rgb")  # set outside before calling
    max_depth = 5
    torch.manual_seed(42) #important
    ####################################
    # scene = mi.load_file(scene_path)
    sensors = scene.sensors()
    sensor = sensors[0]
    film_size = sensor.film().crop_size()
    print("film_size: ", film_size)
    n_pixels = dr.prod(film_size)
    print("n_pixels: ", n_pixels)
    sampler = sensor.sampler()
    H = film_size[0]
    W = film_size[1]
    print("================= %d spp in total===================" % (spp_))
    SPP_PER_BATCH = 256
    if spp_ < SPP_PER_BATCH:
        SPP_PER_BATCH = spp_
    assert(spp_ % SPP_PER_BATCH == 0)
    n_batches = spp_ // SPP_PER_BATCH
    seeds = [42, 38, 50, 64, 23, 75, 88,39,12, 13, 14, 15, 16, 17,18, 19,20,21,22,24,25,26,27,28,38, 50, 64, 23, 75, 88,39,12, 13, 14, 15, 16, 17,18, 19,20,21,22,24,25,26,27,28,22,24,25,26,27,28]
    if GT:
        seeds = seeds[2:]
    L_total = np.zeros((3, film_size[0], film_size[1], SPP_PER_BATCH))
    for i_batch in range(n_batches): 
        # sampler = mi.load_dict({'type' : 'stratified'}) #ldsampler
        # sampler = mi.load_dict({'type' : 'independent'})
        sampler.set_sample_count(SPP_PER_BATCH)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SPP_PER_BATCH = ", SPP_PER_BATCH)
        # spp = sampler.sample_count()
        spp = SPP_PER_BATCH
        print("================= batch %d with spp %d ===================" % (i_batch, spp))
        sampler.seed(seeds[i_batch], wavefront_size=dr.prod(film_size)
                    * SPP_PER_BATCH)  # important!! bug for longtime
        # Standard BSDF evaluation context for path tracing+
        # generate the batch of primary rays
        prim_rays, weights, splatting_pos = generate_primary_rays(
            scene, sensor, sampler, spp)
        si_pre = scene.ray_intersect_preliminary(prim_rays)  # primary intersections
        si = si_pre.compute_surface_interaction(prim_rays)
        valid = si.is_valid()  # lanes of bool        
        
        rays = mi.Ray3f(prim_rays)
        depth = 0 #depth = mi.UInt32(0)                          # Depth of current vertex
        L = mi.Spectrum(0)                            # Radiance accumulator
        throughput = mi.Spectrum(1)
        active = mi.Bool(valid)                      # Active SIMD lanes
    
        # variables caching information from the previous bounce
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        num_lanes = sampler.wavefront_size()
        assert(num_lanes == (n_pixels * SPP_PER_BATCH))  #   print(" %d lanes...." % num_lanes)

        # ---- NEW: stable ray IDs for this batch ----
        ray_ids = dr.arange(mi.UInt32, num_lanes)

        all_positions = []
        all_normals = []
        all_directions = []
        all_radiance = []
        all_ray_ids = []     # NEW
        all_bounce_ids = []  # NEW

        while depth < max_depth:
            # Le = mi.Spectrum(0)
            if MIS and depth > 0:
                # Compute the MIS weight for emitter sample from previous bounce 
                ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)
                mis = mis_weight(prev_bsdf_pdf, scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta))
                Le = throughput * si.emitter(scene).eval(si) * mis
            else:
                Le = throughput * si.emitter(scene).eval(si)
            throughput[~active] = mi.Spectrum(0.0)
            L = L + Le
            
            depth += 1
            if (depth >= max_depth):
                break

            # Standard BSDF evaluation context for path tracing
            bsdf_ctx = mi.BSDFContext()
            bsdf = si.bsdf(rays)

            if MIS:
                ######## [MIS] ##############
                # ---------------------------Emitter sampling --------------------------
                # is emitter sampling even possible on the current vertex?
                active_em = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth) 
                
                # If so, randomly sample an emitter without derivative tracking.
                ds, em_weight = scene.sample_emitter_direction(
                    si, sampler.next_2d(), True, active_em)
                active_em &= ds.pdf != 0.0

                ds.d = dr.normalize(ds.p - si.p)
                em_val = scene.eval_emitter_direction(si, ds, active_em)
                em_weight = dr.select(ds.pdf != 0.0, em_val / ds.pdf, 0)
                dr.disable_grad(ds.d)

                wo = si.to_local(ds.d)
                bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
                mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
                Lr_dir = throughput * mis_em * bsdf_value_em * em_weight
                L = L + Lr_dir
            ######## [MIS] ##############

            ##################### brdf sampling ##############################
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            if STRATIFY_BSDF:
                sample_2d_strata = mi.Point2f(np.array(stratified_sampling_2d(n_pixels, SPP_PER_BATCH)))

            start.record()
            if STRATIFY_BSDF:
                bs, bsdf_val = bsdf.sample(bsdf_ctx, si, sampler.next_1d(), sample_2d_strata, active)  # only for general materials
            else:
                bs, bsdf_val = bsdf.sample(bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active)  # only for general materials
            end.record()
            torch.cuda.synchronize()
            print("[COSINE SAMPLE] time elapsed for importance sampling of samples .....",
                  start.elapsed_time(end)/1000, n_pixels*spp)
            # timings_sample.append(start.elapsed_time(end)/1000)

            
            # for data dumping
            dr.eval(si) # to evaluate the lazily built vectors
            dr.eval(bs) 
            wo_world = si.sh_frame.to_world(bs.wo)  # this is the direction sampled from BRDF

            # Record current bounce info (before updating si)
            try:
                indices = dr.compress(active)

                def dump_vec_info(name, v):
                    print(f"{name}.x.shape = {v.x.shape}")
                    print(f"{name}.y.shape = {v.y.shape}")
                    print(f"{name}.z.shape = {v.z.shape}")

                def gather_vec3(v, name):
                    assert hasattr(v, 'x') and hasattr(v, 'y') and hasattr(v, 'z'), f"Invalid vec3: {v}"
                    assert v.x.shape == v.y.shape == v.z.shape, f"Inconsistent vec shape: {v.x.shape}, {v.y.shape}, {v.z.shape}"
                    print(f"gathering {name}")
                    return np.stack([
                        np.array(dr.gather(mi.Float, v.x, indices)),
                        np.array(dr.gather(mi.Float, v.y, indices)),
                        np.array(dr.gather(mi.Float, v.z, indices))
                    ], axis=-1)
                    

                # Gather geometry + radiance
                positions = gather_vec3(si.p, "positions")
                normals = gather_vec3(si.sh_frame.n, "normals")
                directions = gather_vec3(wo_world, "directions")
                radiance_step = gather_vec3(throughput * bsdf_val, "radiance")

                # Gather ray IDs for active rays
                ray_ids_step = np.array(dr.gather(mi.UInt32, ray_ids, indices))
                bounce_ids_step = np.full(ray_ids_step.shape[0], depth, dtype=np.int32)

                print(f"posision.shape {positions.shape}, {normals.shape}, {directions.shape}, {radiance_step.shape}")
                all_positions.append(np.array(positions))
                all_normals.append(np.array(normals))
                all_directions.append(np.array(directions))
                all_radiance.append(np.array(radiance_step))
                all_ray_ids.append(ray_ids_step)
                all_bounce_ids.append(bounce_ids_step)
            
                print(f"Recorded {positions.shape[0]} samples at depth {depth}")
            except Exception as e:
                print("Error while extracting data:", e)
            

            # Now update for the next bounce
            throughput[active] *= bsdf_val    
            rays[active] = si.spawn_ray(wo_world)
            
            ########[MIS] update the information needed by next iteration
            if MIS:
                prev_si = dr.detach(si, True)
                prev_bsdf_pdf = bs.pdf #TODO
                prev_bsdf_delta = mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
            ########[MIS]##############

            ###########################################################################
            si_pre = scene.ray_intersect_preliminary(rays, active)
            # update the neumip material mask at current intersections
            si = si_pre.compute_surface_interaction(rays)
            active &= si.is_valid()

        # image = mi.util.convert_to_bitmap(L)
        print(L.shape)
        L = np.array(L).reshape(3, film_size[0], film_size[1], spp)
        L_total = L_total + L
        
        # Save per-batch RL data with correspondences
        np.savez(
            f"rl_reward_light{light_id}_batch_{i_batch}.npz",
            positions=np.concatenate(all_positions, axis=0),
            normals=np.concatenate(all_normals, axis=0),
            directions=np.concatenate(all_directions, axis=0),
            radiance=np.concatenate(all_radiance, axis=0),
            ray_ids=np.concatenate(all_ray_ids, axis=0),
            bounce_ids=np.concatenate(all_bounce_ids, axis=0),
            bounce_lengths=[len(x) for x in all_positions]
        )


    print("=================== cosine sampling for output image =========================")
    L_total = np.mean(L_total, axis=3) / n_batches
    L_total = np.transpose(L_total, (1,2,0))
    image = mi.TensorXf(L_total)
    mi.util.write_bitmap(out_path+".exr", image)
    cv2.imwrite(out_path+".png", tonemap(L_total)*255.0)


def render_gt_indirect_radiance_at_points(scene, positions, normals):
    sensors = scene.sensors()
    sensor = sensors[0]
    film_size = sensor.film().crop_size()
    print("film_size: ", film_size)
    n_pixels = dr.prod(film_size)
    print("n_pixels: ", n_pixels)
    sampler = sensor.sampler()
    H = film_size[0]
    W = film_size[1]
    
    integrator = mi.load_dict({'type': 'path'})  # use standard PT
    sampler = mi.load_dict({'type': 'independent', 'sample_count': 1})
    positions_np  = np.array(positions)  # shape (N, 3)
    normals_np = np.array(normals)    # shape (N, 3)
    N = positions.shape[0]
    M = 64  # directions per point
    
    res = 32
    D = res * res

    # === 1. Sample fixed local hemisphere directions ===
    u, v = np.meshgrid(np.linspace(0, 1, res), np.linspace(0, 1, res), indexing='ij')
    u2d = mi.Point2f(mi.Float(u.flatten()), mi.Float(v.flatten()))
    local_dirs = mi.warp.square_to_uniform_hemisphere(u2d)              # (D, 3)

    # === 2. Expand points and normals ===
    # Convert numpy → drjit arrays
    pos_x = mi.Float(positions_np[:, 0])
    pos_y = mi.Float(positions_np[:, 1])
    pos_z = mi.Float(positions_np[:, 2])

    nrm_x = mi.Float(normals_np[:, 0])
    nrm_y = mi.Float(normals_np[:, 1])
    nrm_z = mi.Float(normals_np[:, 2])

    # Repeat N times for each of the D directions
    pos_x = dr.repeat(pos_x, D)
    pos_y = dr.repeat(pos_y, D)
    pos_z = dr.repeat(pos_z, D)

    nrm_x = dr.repeat(nrm_x, D)
    nrm_y = dr.repeat(nrm_y, D)
    nrm_z = dr.repeat(nrm_z, D)

    # Build Mitsuba vector types
    positions = mi.Point3f(pos_x, pos_y, pos_z)     # shape (N*D)
    normals   = mi.Vector3f(nrm_x, nrm_y, nrm_z)     # shape (N*D)

    local_dirs = dr.tile(local_dirs, N)                             # (N*D, 3)

    # === 3. Create rays ===
    frames = mi.Frame3f(normals)
    world_dirs = frames.to_world(local_dirs)
    eps = 1e-4
    ray_origins = positions + eps * world_dirs
    rays = mi.Ray3f(o=ray_origins, d=world_dirs)

    # === 4. Trace all rays ===
    result = integrator.sample(scene, sampler, rays)

    # convert Spectrum → NumPy
    radiance_np = np.array(result[0]).reshape(N, res, res, 3)

    return radiance_np

# if __name__ == "__main__":
#     mi.set_variant('llvm_ad_rgb')

#     # 3 required positions (with y fixed to 1.0 for Cornell box)
#     light_positions = [
#         (0, 1, 0),
#         (1, 0, 0), 
#         (0, 0, 1),  
#     ]

#     outdir = "./output"
#     os.makedirs(outdir, exist_ok=True)

#     for i, xyz in enumerate(light_positions):
#         print(f"=== Rendering dataset {i} with light at {xyz} ===")

#         # Load fresh Cornell box each time
#         scene_dict = mi.cornell_box()
#         scene_dict = move_light(scene_dict, xyz)
#         scene = mi.load_dict(scene_dict)

#         # Output path per variant
#         out_path = os.path.join(outdir, f"pt_light_{i}")

#         simulate_batch_pt(
#             scene,
#             spp_=256,
#             out_path=out_path,
#             GT=True,
#             light_id=i
#         )

#     print("=== All 3 datasets completed successfully! ===")

def sample_eval_points(npz_path, K):
    """Sample K (position, normal) pairs from offline dataset."""
    data = np.load(npz_path)
    positions = data["positions"]
    normals = data["normals"]

    idx = np.random.choice(len(positions), size=K, replace=False)

    return positions[idx], normals[idx]


def build_eval_dataset(scene, offline_npz, K=128, res=32, out_path="eval_dataset.npz"):
    """Sample K evaluation points and compute GT radiance."""
    
    print(f"[Eval] Loading offline dataset: {offline_npz}")
    pos_eval, normal_eval = sample_eval_points(offline_npz, K)

    print(f"[Eval] Sampled {K} surface points")

    print("[Eval] Computing ground truth hemispherical radiance...")
    radiance_gt = render_gt_indirect_radiance_at_points(
        scene, pos_eval, normal_eval
    )  # shape: (K, res, res, 3)

    print("[Eval] GT Radiance shape:", radiance_gt.shape)

    # === Rebuild hemisphere directions used by GT function ===
    u, v = np.meshgrid(
        np.linspace(0, 1, res, dtype=np.float32),
        np.linspace(0, 1, res, dtype=np.float32),
        indexing="ij",
    )

    u2d = mi.Point2f(u.ravel(), v.ravel())
    local_dirs = mi.warp.square_to_uniform_hemisphere(u2d)  # DrJit Vector3f with D lanes

    # Force numpy (D, 3)
    dirs_eval = np.stack(
        [np.array(local_dirs.x), np.array(local_dirs.y), np.array(local_dirs.z)],
        axis=-1,
    ).astype(np.float32)  # (D, 3)

    print("[Eval] Saving evaluation dataset to:", out_path)

    np.savez(out_path,
             pos=pos_eval,
             normal=normal_eval,
             dirs=dirs_eval,
             radiance_gt=radiance_gt)

    print("[Eval] Done! Saved:", out_path)

if __name__ == "__main__":
    mi.set_variant("llvm_ad_rgb")

    # Load Cornell Box
    scene_dict = mi.cornell_box()
    scene = mi.load_dict(scene_dict)

    # Offline dataset path
    offline_npz = "rl_reward_light0_batch_0.npz"

    # Output evaluation dataset
    out_path = "eval_dataset.npz"

    # Number of evaluation samples
    K = 128

    build_eval_dataset(scene, offline_npz, K=K, res=32, out_path=out_path)

# if __name__ == "__main__":
#     # render_final_image()
#     mi.set_variant('llvm_ad_rgb')
#     scene = mi.load_dict(mi.cornell_box())
#     outdir = "./output"
#     os.makedirs(outdir,  exist_ok=True)
#     simulate_batch_pt(scene, spp_=256, out_path=os.path.join(outdir, "pt"), GT=True)
