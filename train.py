#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


import os
import sys
import torch
import socket
from random import randint, Random
from utils.loss_utils import l1_loss, ssim, l2_loss, tv_loss, lp_loss, DepthRelLoss, mse_loss
from gaussian_renderer import render, network_gui, render_mask, render_lightweight
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, parse_cfg
import cv2
import copy
import uuid
import json
import time
import yaml
import hashlib
import functools
import torchvision
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import torchvision.transforms.functional as F
from utils.image_utils import psnr, save_image
from scene.cameras import camName_from_Path, imageName_from_Path
from argparse import ArgumentParser, Namespace
from utils.general_utils import DecayScheduler, kthvalue
from utils.graphics_utils import adjust_depths
from utils.image_utils import resize_image, downsample_image, blur_image, get_mask, write_depth, coords_grid, flow_warp, coords_grid_proj, get_depth, resize_dims, add_metrics_label_to_image
from utils.loader_utils import MultiViewVideoDataset
from utils.loader_utils import SequentialMultiviewSampler, MultiViewVideoDataset
from arguments import ModelParams, PipelineParams, OptimizationParams, QuantizeParams, OptimizationParamsInitial, OptimizationParamsRest
from scene.decoders import LatentDecoder, LatentDecoderRes, Gate
from scripts.adaptive_hierarchy import (AdaptiveHierarchyMotion,
                                         AdaptiveOffsetGate,
                                         project_gaussians, sample_pixel_map)
from scripts.adaptive_gate_partition import (
    build_adaptive_gate_partition, build_compact_gate_payload,
    decode_compact_gate_payload, finalize_gate_partition,
    decode_sqres_reopen_mask, encode_sqres_reopen_mask,
    refine_adaptive_gate_partition, inherit_gate_parameters)
from scripts.first_frame_snapshot import (
    load_frame1_snapshot, save_frame1_snapshot)
from generate_video_all import symlink
from scipy import ndimage
from torch_scatter import scatter_max

# Disable tqdm to make pdb easier to use
# Set to False to disable progress bars for debugging
enable_tqdm = True
enable_debug = False

EPS = 1.0e-7


@torch.no_grad()
def refine_v7b_gate_frame(gaussians, dataset, evidence, frozen_roots,
                          frozen_anchors, frozen_origin, frozen_voxel_size,
                          args, target_level):
    """Run one V7b refinement stage and preserve per-point Gate values."""
    old_gate = gaussians.gate_atts
    if old_gate is None or target_level + 1 >= args.adaptive_gate_max_levels:
        return None
    old_inverse = gaussians.adaptive_gate_partition["point_group_ids"]
    old_point_gates = old_gate.sample_gate_Gumbel_Softmax(
        stochastic=False).index_select(0, old_inverse)
    parent_active = old_gate.sample_gate_Gumbel_Softmax(
        stochastic=False) > 0.05
    residual_scores = gaussians._ungated_offset_res.norm(dim=-1)
    partition, parent_group_ids, stats = refine_adaptive_gate_partition(
        gaussians.adaptive_gate_partition,
        frozen_roots,
        frozen_anchors,
        evidence,
        residual_scores,
        parent_active,
        frozen_voxel_size,
        init_pos=frozen_origin,
        target_level=target_level,
        min_points=args.adaptive_gate_min_points,
        offset_bits_per_point=args.adaptive_gate_offset_bits,
        residual_ratio=args.adaptive_gate_residual_ratio,
        active_threshold=0.05)
    if stats["rate_accepted_groups"] == 0:
        return stats

    new_gate = Gate(
        partition["terminal_group_count"],
        gamma=dataset.gate_gamma,
        eta=dataset.gate_eta,
        lr=dataset.gate_lr,
        temp=dataset.gate_temp,
        lambda_l2=dataset.gate_lambda_l2,
        lambda_l0=dataset.gate_lambda_l0,
        inverse_indices=partition["point_group_ids"])
    inherit_gate_parameters(old_gate, new_gate, parent_group_ids)
    new_gate.train()
    new_point_gates = new_gate.sample_gate_Gumbel_Softmax(
        stochastic=False).index_select(0, partition["point_group_ids"])
    if not torch.equal(old_point_gates, new_point_gates):
        raise RuntimeError(
            "Adaptive V7b split changed deterministic per-point Gate values")

    gaussians.adaptive_gate_partition = partition
    gaussians.adaptive_gate_topology_bits = int(partition["topology_bits"])
    gaussians.gate_atts = new_gate
    stats["old_group_count"] = int(old_gate.num_gates)
    stats["new_group_count"] = int(new_gate.num_gates)
    stats["continuity_verified"] = True
    return stats


@torch.no_grad()
def hierarchy_region_probe(hierarchy, gaussians, cameras, pipe, background,
                           iteration, level, snapshots=None):
    """Compare the same projected pixels before/after a child-motion trial."""
    mapping = getattr(hierarchy, f"point_to_node_{level}")
    total = hierarchy.dynamic_scores.new_zeros(hierarchy.node_counts[level])
    count = total.clone()
    if snapshots is None:
        snapshots = []
    for index, camera in enumerate(cameras):
        pkg = render(camera, gaussians, pipe, background,
                     iteration, "RGB")
        error_map = (pkg["render"] - camera.original_image).abs().mean(dim=0)
        if index >= len(snapshots):
            xy = pkg["viewspace_points"].squeeze(0).detach()
            visible = pkg["visibility_filter"].detach()
            snapshots.append((xy, visible))
        else:
            xy, visible = snapshots[index]
        values, valid = sample_pixel_map(xy, error_map, visible)
        total.scatter_add_(0, mapping, values)
        count.scatter_add_(0, mapping, valid.to(count.dtype))
    return total / count.clamp_min(1.0), snapshots


def zero_gaussian_gradients(gaussians):
    """Clear probe gradients even when a restored snapshot has no optimizer."""
    if gaussians.optimizer is not None:
        gaussians.optimizer.zero_grad(set_to_none=True)
        return
    for parameter in gaussians.parameters():
        parameter.grad = None
    gate = getattr(gaussians, "gate_atts", None)
    if gate is not None:
        gate.optimizer.zero_grad(set_to_none=True)
try:
    from torch.utils.tensorboard import SummaryWriter
    if not ('SLURM_PROCID' in os.environ and os.environ['SLURM_PROCID']!='0'):
        TENSORBOARD_FOUND = True
    else:
        TENSORBOARD_FOUND = False
except ImportError:
    TENSORBOARD_FOUND = False

try:
    import wandb
    if not ('SLURM_PROCID' in os.environ and os.environ['SLURM_PROCID']!='0'):
        WANDB_FOUND = True
    else:
        WANDB_FOUND = False
except ImportError:
    WANDB_FOUND = False

def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, qp:QuantizeParams, opt_lod, testing_iterations: list, 
             saving_iterations: list, checkpoint_iterations, checkpoint: str, debug_from, args):
    """Main training function for Structured Sparse Gaussian splatting (S²GS)."""
    use_adaptive_gate = bool(args.adaptive_gate_partition
                             or args.adaptive_gate_partition_v7b)
    use_adaptive_gate_v7b = bool(args.adaptive_gate_partition_v7b)
    if args.adaptive_gate_partition and args.adaptive_gate_partition_v7b:
        raise ValueError(
            "Use either legacy --adaptive_gate_partition or "
            "--adaptive_gate_partition_v7b, not both")
    if args.adaptive_hierarchy and dataset.update_mask != "viewspace_diff":
        raise ValueError("Adaptive hierarchy requires update_mask=viewspace_diff")
    if use_adaptive_gate and dataset.update_mask != "viewspace_diff":
        raise ValueError(
            "Adaptive Gate partitioning requires update_mask=viewspace_diff")
    if args.adaptive_hierarchy and use_adaptive_gate:
        raise ValueError(
            "--adaptive_hierarchy and adaptive Gate partitioning are separate modes")
    if (use_adaptive_gate
            and not any(gating != "none" for gating in qp.gate_params)):
        raise ValueError(
            "Adaptive Gate partitioning requires at least one configured S²GS Gate")
    if args.save_frame1_snapshot and args.resume_frame1_snapshot:
        raise ValueError("Save and resume frame-one snapshot are mutually exclusive")
    if args.stop_after_frame1_snapshot and not args.save_frame1_snapshot:
        raise ValueError(
            "--stop_after_frame1_snapshot requires --save_frame1_snapshot")
    wandb_enabled = WANDB_FOUND and dataset.use_wandb
    tb_writer = prepare_output_and_logger(args)
    generator = Random(dataset.seed)
    qp.seed = dataset.seed

    qp.use_shift = [bool(el) for el in qp.use_shift]

    # Create dataset and loader for training and testing at each time instance
    train_image_dataset = MultiViewVideoDataset(dataset.source_path, split='train', test_indices=dataset.test_indices,
                                                max_frames=dataset.max_frames, start_idx=dataset.start_idx, img_format=dataset.img_fmt)
    if (args.save_frame1_snapshot
            and train_image_dataset.n_frames < 2):
        raise ValueError("Saving a frame-one snapshot requires at least two frames")
    test_image_dataset = MultiViewVideoDataset(dataset.source_path, split='test', test_indices=dataset.test_indices, 
                                               max_frames=dataset.max_frames, start_idx=dataset.start_idx, 
                                               img_format=dataset.img_fmt)

    train_sampler = SequentialMultiviewSampler(train_image_dataset)
    if test_image_dataset.n_cams > 0:
        test_sampler = SequentialMultiviewSampler(test_image_dataset)

    train_loader = iter(torch.utils.data.DataLoader(train_image_dataset, batch_size=train_image_dataset.n_cams, 
                                                    sampler=train_sampler, num_workers=4))
    if test_image_dataset.n_cams > 0:
        test_loader = iter(torch.utils.data.DataLoader(test_image_dataset, batch_size=test_image_dataset.n_cams, 
                                                        sampler=test_sampler, num_workers=4))
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    print(f"training(): dataset.white_background set to {dataset.white_background}")
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    # Initial set of images to initialize camera and camera parameters
    # Image dimensions should remain constant throughout the video
    print(f"training(): loading data for the first frame...")
    tic = time.time()
    train_data = next(train_loader)
    train_images, train_paths = train_data
    resume_frame2_data = None
    if args.resume_frame1_snapshot:
        if train_image_dataset.n_frames < 2:
            raise ValueError("Frame-one snapshot resume requires at least two frames")
        try:
            resume_frame2_data = next(train_loader)
        except StopIteration as exc:
            raise ValueError(
                "Could not pre-consume frame two for snapshot resume") from exc
    if test_image_dataset.n_cams > 0:
        test_data = next(test_loader)
        test_images, test_paths = test_data
        test_image_data = {'image':test_images.cuda(),'path':test_paths,'frame_idx':0}
    else:
        print('No test cameras found, disabling testing.')
        test_images, test_paths = None, None
        test_image_data = {'image':None,'path':None,'frame_idx':0}

    train_image_data = {'image':train_images.cuda(),'path':train_paths,'frame_idx':0}
    
    print(f"training(): data loaded in {float(time.time() - tic):.2f} sec")

    # Create the gaussian model and scene, initialized with frame 1 images from dataset
    gaussians = GaussianModel(dataset.sh_degree, qp, dataset, opt_lod, use_offset_legacy=args.use_xyz_legacy)

    max_frames = args.max_frames
    scene = Scene(
        dataset,
        gaussians,
        train_image_data= train_image_data,
        test_image_data=test_image_data,
        N_video_views=max_frames
    )
    # Setup training arguments
    gaussians.set_coarse_interval(opt_lod)
    gaussians.training_setup(opt, opt_lod)

    # Spiral cameras
    video_cameras = scene.getVideoCameras()
    # video_cameras = None

    # Metadata used by various components
    train_cameras = scene.getTrainCameras()
    n_frames, n_cams = train_image_dataset.n_frames, train_image_dataset.n_cams
    print(f"training(): running with {n_frames} frames from {n_cams} cameras")
    opt.iterations = opt.epochs
    print(f"training(): opt.iterations set to {opt.iterations}")
    _,H,W = train_cameras[0].original_image.shape

    cur_frame_views = train_image_data['image']
    prev_frame_views = cur_frame_views
    if resume_frame2_data is not None:
        cur_frame_views = resume_frame2_data[0].cuda()
        cur_train_images = cur_frame_views
        cur_train_paths = resume_frame2_data[1]

    # Vary number of iterations based on frame difference in json file
    if dataset.adaptive_iters and n_frames>1:
        frame_diff = json.load(open(os.path.join(dataset.source_path,'frame_diff.json'),'r'))['l2']
        frame_diff = np.array(frame_diff[:n_frames-1])
        epochs_rest = opt.opt_rest['epochs_rest']
        mult = np.clip(frame_diff/frame_diff.mean(),1/4,4) # between 0.25 to 4
        mult = mult/mult.mean()
        frame_epochs = np.ceil((mult*epochs_rest)).astype(np.int32)
        frame_iters = np.concatenate((np.array([opt.iterations]),frame_epochs*n_cams))
    else:
        frame_iters = np.array([opt.iterations]+[opt.opt_rest['epochs_rest']*n_cams]*(n_frames-1))
    if args.adaptive_hierarchy:
        frame_iters[1:] = (args.hierarchy_stage1_steps
                           + args.hierarchy_gs_steps
                           + args.hierarchy_growth_steps)

    # Progressive training scheduler - OBSOLETE: Remove in future cleanup
    resize_scale_sched = DecayScheduler(
                                        total_steps=int(opt.resize_period*(opt.iterations+1)),
                                        decay_name='cosine',
                                        start=opt.resize_scale,
                                        end=1.0,
                                        )

    start_frame_idx = 1
    if args.resume_frame1_snapshot:
        snapshot_info = load_frame1_snapshot(
            args.resume_frame1_snapshot, gaussians, generator,
            restore_rng=True)
        if snapshot_info["num_gaussians"] != gaussians.get_anchor.shape[0]:
            raise RuntimeError("Frame-one snapshot Gaussian count was not restored")
        prev_xyz = gaussians._offset.clone()
        # In an uninterrupted run the first frame's last iteration value is
        # still live when the frame-two viewspace probe begins.
        iteration = int(frame_iters[0])
        start_frame_idx = 2
        print(f"Resuming at frame 2 from frame-one snapshot: "
              f"{snapshot_info['path']}")
    gaussians.adaptive_gate_partition_enabled = bool(
        use_adaptive_gate)
    gaussians.adaptive_gate_partition_v7b_enabled = use_adaptive_gate_v7b
    training_metrics = []
    net_elapsed_time = 0.0
    net_iter_time = 0.0


    training_start = time.time()

    if opt.lambda_flow > 0.0:
        grid = coords_grid(1,H,W, device='cuda')

    if enable_tqdm:
        progress_bar_frame = tqdm(range(1, n_frames+1), desc="Training progress")
        progress_bar_frame.update(start_frame_idx-1)
    else:
        progress_bar_frame = None
        frame_counter = 0

    # The previous frame's compact Gate values are retained only to diagnose
    # any difference from Gate.get_gates()' cached last-forward sample.
    last_sidecar_point_gates = None

    # start frame index loop
    for frame_idx in range(start_frame_idx, n_frames+1):

        first_iter = 1
        frame_hierarchy_payload = None
        frame_gate_payload = None
        frame_gate_stats = {}
        stage_probe_psnr = {}
        frame_trial_candidates = 0
        frame_trial_accepted = 0
        frame_gradient_cue_points = 0
        frame_image_cue_points = 0
        frame_sqres_cue_reopened_points = 0
        frame_sqres_reopen_mask = None
        frame_sqres_reopen_mask_payload = None
        frame_sqres_reopen_mask_raw_bytes = 0
        frame_sqres_reopen_mask_bytes = 0
        frame_sqres_previous_point_gate = None
        frame_previous_gate_cache_mismatch_points = 0
        frame_previous_gate_cache_max_abs_diff = 0.0
        frame_sidecar_decoder_gate_mismatch_points = 0
        frame_sidecar_decoder_gate_max_abs_diff = 0.0
        point_evidence = None
        gradient_scores = None
        image_change_scores = None
        image_cue = None
        frame_gate_evidence = None
        frame_partition_root_indices = None
        frame_partition_anchors = None
        frame_partition_origin = None
        frame_partition_voxel_size = None
        frame_gate_stage_stats = []

        # The V7d mask cost is known before the frame's final size() report.
        gaussians.adaptive_gate_sqres_reopen_mask_bits = 0
        
        if dataset.log_ply:
            scene.model_path = os.path.join(args.model_path,'frames',str(dataset.start_idx + frame_idx).zfill(4))
            os.makedirs(scene.model_path,exist_ok=True)

        ema_loss_for_log, cur_size, best_psnr = 0.0, 0.0, 0.0
        metrics = {'val':{'psnr':0.0, 'loss':0.0, 'fps': 0.0, 'storage': 0.0}, 'test':{'psnr':0.0, 'loss': 0.0, 'fps': 0.0, 'storage': 0.0}}
        camera_idx_stack = []
        report = None

        if dataset.timed:
            torch.cuda.synchronize()
        frame_start_io = time.time()
        frame_time_io = 0.0
        frame_time_init = 0.0

        try: 
            # Pre-load data for next frame
            next_train_data = next(train_loader)
            next_train_images, next_train_paths = next_train_data[0].cuda(), next_train_data[1]
            next_frame_views = next_train_images

        except StopIteration:
            assert frame_idx == n_frames
            opt.lambda_flow = 0.0

        if dataset.timed:
            torch.cuda.synchronize()
        frame_start = time.time()
        frame_time = 0.0

        # Freeze the previous-frame coordinate system before any current-frame
        # update. All V7b topology stages use these anchors/root IDs so the
        # compact split paths remain reconstructible after training moves GS.
        if use_adaptive_gate_v7b and frame_idx > 1:
            frame_partition_root_indices = gaussians.get_root_indices.detach().clone()
            frame_partition_anchors = gaussians.get_anchor.detach().clone()
            frame_partition_origin = gaussians.init_pos.detach().clone()
            frame_partition_voxel_size = gaussians.voxel_size.detach().clone()

        # Update a bunch of variables and models for each new frame
        if frame_idx > 1:
            gaussians.frame_idx = frame_idx
            # Initialize gate probabilities based on gradient differences or frame differences
            if dataset.update_mask == "viewspace_diff":
                # Compute viewspace gradient differences for gate initialization
                grad_diff = torch.zeros(gaussians.get_offset.shape[0],1).to(gaussians._offset)
                denom = torch.zeros(gaussians.get_offset.shape[0],1).to(gaussians._offset)
                if args.adaptive_hierarchy or use_adaptive_gate:
                    image_change_sum = torch.zeros_like(grad_diff)
                    image_change_count = torch.zeros_like(grad_diff)
                zero_gaussian_gradients(gaussians)
                for cam_idx, camera in enumerate(train_cameras):
                    render_pkg = render_mask(
                        camera, gaussians, pipe, bg, iteration,
                        image_shape=cur_frame_views[cam_idx].shape)
                    camera.prev_rendered = render_pkg["render"].detach()
                    image, viewspace_point_tensor = render_pkg["render"], render_pkg["viewspace_points"]
                    anchor_visible_mask = render_pkg["visible_mask"]
                    visibility_filter = render_pkg["visibility_filter"]
                    cur_gt_image = cur_frame_views[cam_idx]
                    prev_gt_image = prev_frame_views[cam_idx]
                    if dataset.update_loss == "mae":
                        Ll1 = mse_loss(image, cur_gt_image)
                        Ll1_prev = mse_loss(image, prev_gt_image)
                    elif dataset.update_loss == "mse":
                        Ll1 = mse_loss(image, cur_gt_image)
                        Ll1_prev = mse_loss(image, prev_gt_image)
                    elif dataset.update_loss == "ssim":
                        Ll1 = 1.0-ssim(image, cur_gt_image)
                        Ll1_prev = 1.0-ssim(image, prev_gt_image)
                    elif dataset.update_loss == "mae_orig":
                        Ll1 = l1_loss(image, cur_gt_image)
                        Ll1_prev = l1_loss(image, prev_gt_image)
                    cur_loss = Ll1-Ll1_prev
                    cur_loss.backward()
                    cur_grad = viewspace_point_tensor.grad[visibility_filter,:2].clone()
                    with torch.no_grad():
                        viewspace_point_tensor.grad *= 0
                    zero_gaussian_gradients(gaussians)

                    combined_mask = anchor_visible_mask.clone()
                    combined_mask[anchor_visible_mask] = visibility_filter
                    grad_diff[combined_mask] += torch.norm(cur_grad,dim=-1,keepdim=True)
                    denom[combined_mask] += 1

                    if args.adaptive_hierarchy or use_adaptive_gate:
                        temporal_map = (cur_gt_image - prev_gt_image).abs().mean(dim=0)
                        error_increase = ((image.detach() - cur_gt_image).abs().mean(dim=0)
                                          - (image.detach() - prev_gt_image).abs().mean(dim=0))
                        evidence_map = torch.maximum(
                            0.5 * temporal_map, error_increase.clamp_min(0.0))
                        xy, in_front = project_gaussians(
                            render_pkg["gaussian_xyz"], camera)
                        values, valid = sample_pixel_map(
                            xy, evidence_map,
                            render_pkg["visibility_filter"] & in_front)
                        image_change_sum[valid, 0] += values[valid]
                        image_change_count[valid, 0] += 1

                    # torchvision.utils.save_image(render_pkg['render'], os.path.join(scene.model_path, f"cam{str(camera.uid+1)}_dynamic_gs.png"))

                grad_diff[grad_diff.isnan()] = 0.0

                with torch.no_grad():
                    if dataset.adaptive_render and dataset.adaptive_update_period>0.0:
                        for camera in train_cameras:
                            grad_mask = (grad_diff.flatten()>dataset.pixel_update_thresh)
                            render_pkg = render_mask(camera, scene.gaussians, pipe, bg, iteration, 
                                                    gaussian_mask=grad_mask)
                            alphamask = (render_pkg["alpha"]>0.9).float()

                            # 连通区域分析，去除噪点
                            mask_np = alphamask.cpu().numpy() if alphamask.is_cuda else alphamask.numpy()
                            labeled_array, num_features = ndimage.label(mask_np)
                            component_sizes = np.bincount(labeled_array.ravel())
                            component_sizes[0] = 0
                            min_size = 40
                            filtered_mask = np.isin(labeled_array, np.where(component_sizes >= min_size)[0])
                            alphamask = torch.from_numpy(filtered_mask).to(alphamask.device).float()

                            camera.orig_mask = alphamask
                            mask_down = torch.nn.functional.max_pool2d(alphamask.unsqueeze(0), (dataset.dilate_size,dataset.dilate_size))
                            mask_dilate = torch.nn.functional.interpolate(mask_down, size=(alphamask.shape[-2],alphamask.shape[-1]))
                            camera.mask = (mask_dilate.squeeze(0).squeeze(0)>0).float()

                gaussian_mask = grad_diff>dataset.gaussian_update_thresh

                if use_adaptive_gate:
                    # Compute this once before update_residuals so V7d can
                    # selectively reopen old-OFF sq_res inputs. Reuse the
                    # identical evidence below for the adaptive partition.
                    image_change_scores = (
                        image_change_sum
                        / image_change_count.clamp_min(1.0)).flatten()
                    gradient_scores = grad_diff.flatten()
                    image_cue = image_change_scores > (
                        args.adaptive_gate_image_change_threshold)
                    point_evidence = (
                        (gradient_scores > dataset.gaussian_update_thresh)
                        | image_cue)
                    if (use_adaptive_gate_v7b
                            and args.adaptive_gate_sqres_cue_floor > 0.0):
                        if gaussians.gate_atts is not None:
                            cached_gate = gaussians.gate_atts.get_gates()
                            if cached_gate is not None:
                                frame_sqres_previous_point_gate = (
                                    cached_gate.detach().clone().flatten())
                        if frame_sqres_previous_point_gate is None:
                            frame_sqres_reopen_mask = torch.zeros_like(
                                point_evidence, dtype=torch.bool)
                        else:
                            if (frame_sqres_previous_point_gate.numel()
                                    != point_evidence.numel()):
                                raise RuntimeError(
                                    "Previous offset Gate and cue must have "
                                    "one value per Gaussian")
                            if last_sidecar_point_gates is not None:
                                if (last_sidecar_point_gates.numel()
                                        != frame_sqres_previous_point_gate.numel()):
                                    raise RuntimeError(
                                        "Previous sidecar Gate and current "
                                        "Gaussian order do not match")
                                cache_diff = (
                                    frame_sqres_previous_point_gate
                                    - last_sidecar_point_gates.to(
                                        frame_sqres_previous_point_gate))
                                frame_previous_gate_cache_mismatch_points = int(
                                    torch.count_nonzero(cache_diff).item())
                                frame_previous_gate_cache_max_abs_diff = float(
                                    cache_diff.abs().max().item())
                            frame_sqres_reopen_mask = (
                                (frame_sqres_previous_point_gate <= 0.05)
                                & point_evidence)
                        frame_sqres_cue_reopened_points = int(
                            frame_sqres_reopen_mask.sum().item())
                        frame_sqres_reopen_mask_payload = (
                            encode_sqres_reopen_mask(
                                frame_sqres_reopen_mask))
                        frame_sqres_reopen_mask_raw_bytes = int(
                            frame_sqres_reopen_mask_payload["raw_bytes"])
                        frame_sqres_reopen_mask_bytes = len(
                            frame_sqres_reopen_mask_payload["blob"])
                        gaussians.adaptive_gate_sqres_reopen_mask_bits = (
                            frame_sqres_reopen_mask_bytes * 8)


            with torch.no_grad():

                # Load optimizer hyperparams (initial or rest) based on frame index
                opt.set_params(frame_idx)
                opt.iterations = frame_iters[frame_idx-1]
                opt.epochs = (opt.iterations//n_cams)
                gaussians.frame_idx = frame_idx
                # Create decoder and latents for quantized residuals if first time
                # Else reset latent values to 0
                if (use_adaptive_gate and frame_idx > 2
                        and gaussians.gate_atts is not None):
                    # update_residuals() immediately snapshots this old
                    # per-point gate into gate_before before the current
                    # frame installs its newly partitioned group map.
                    previous_point_gates = gaussians.gate_atts.get_gates()
                    if (previous_point_gates is None
                            or previous_point_gates.numel()
                            != gaussians.get_offset.shape[0]):
                        raise RuntimeError(
                            "Previous S²GS Gate did not retain one value per "
                            "Gaussian before adaptive regrouping")
                gaussians.update_residuals(
                    dataset,
                    adaptive_gate_sqres_floor=(
                        args.adaptive_gate_sqres_floor
                        if use_adaptive_gate_v7b else 0.0),
                    adaptive_gate_sqres_cue_floor=(
                        args.adaptive_gate_sqres_cue_floor
                        if use_adaptive_gate_v7b else 0.0),
                    point_evidence=(
                        point_evidence
                        if (use_adaptive_gate_v7b
                            and args.adaptive_gate_sqres_cue_floor > 0.0)
                        else None))
                if (use_adaptive_gate_v7b
                        and args.adaptive_gate_sqres_cue_floor > 0.0
                        and point_evidence is not None):
                    previous_gate = getattr(gaussians, "gate_before", None)
                    if frame_sqres_previous_point_gate is None:
                        if previous_gate is not None:
                            raise RuntimeError(
                                "F2 sq_res Gate should remain unmasked when "
                                "no previous offset Gate exists")
                    else:
                        if (previous_gate is None
                                or previous_gate.numel()
                                != point_evidence.numel()):
                            raise RuntimeError(
                                "Previous offset Gate and frame evidence "
                                "must have one value per Gaussian")
                        if not torch.equal(
                                previous_gate.flatten(),
                                frame_sqres_previous_point_gate):
                            raise RuntimeError(
                                "update_residuals changed the previous "
                                "offset Gate used to form the V7d mask")
                # Redefine the optimizer and other tracked variables for the gaussian model
                gaussians.training_setup(opt, opt_lod)
                if args.adaptive_hierarchy:
                    gaussians.prepare_hierarchy_frame()
                # Load the current test data (Preloaded data for next frame is only for training)
                train_images, train_paths = cur_train_images, cur_train_paths
                if dataset.timed:
                    torch.cuda.synchronize()
                frame_time += time.time() - frame_start
                if test_image_dataset.n_cams > 0:
                    test_data = next(test_loader)
                    test_images, test_paths = test_data[0].cuda(), test_data[1]
                else:
                    if frame_idx == start_frame_idx:
                        print('No test cameras found, disabling testing.')
                    test_images, test_paths = None, None
                if dataset.timed:
                    torch.cuda.synchronize()
                frame_start = time.time()
                train_image_data = {'image':train_images,'path':train_paths}
                test_image_data = {'image':test_images,'path':test_paths}
            
                # Update the images and paths for all cameras in the scene with new frame index
                scene.updateCameraImages(args, train_image_data, test_image_data, frame_idx, resolution_scales=[1.0])
                train_cameras = scene.getTrainCameras()
                
                gaussians.update_masks(dataset, None if dataset.update_mask == "none" else gaussian_mask)
                gaussians.freeze_atts(dataset)

                if dataset.adaptive_render and dataset.adaptive_update_period>0.0:
                    adaptive_update_epochs = np.ceil(opt.epochs*dataset.adaptive_update_period).astype(np.int32)
                    pix_thresh_vals = torch.ones(adaptive_update_epochs*n_cams)*dataset.pixel_update_thresh

                    if opt.iterations>pix_thresh_vals.shape[0]:
                        addn_pix_vals = torch.zeros(opt.iterations-pix_thresh_vals.shape[0]).to(pix_thresh_vals)
                        pix_thresh_vals = torch.cat((pix_thresh_vals,addn_pix_vals),dim=0)
                    assert pix_thresh_vals.shape[0] == opt.iterations
                else:
                    pix_thresh_vals = None

                # Initialize gate probabilities based on computed differences
                if (not args.adaptive_hierarchy
                        and not use_adaptive_gate
                        and any([gating!="none" for gating in qp.gate_params])):
                    if dataset.update_mask == "viewspace_diff":
                        # Use gradient differences for gate initialization
                        init_probs = grad_diff/(grad_diff+grad_diff.median())
                        gaussians.init_probs = init_probs.flatten()
                    num_gates, inverse_indices = torch.unique(gaussians.get_root_indices, dim=0, return_inverse=True)
                    gaussians.init_probs = scatter_max(gaussians.init_probs, inverse_indices, dim=0)[0]
                    if gaussians.gate_atts is None:
                        gaussians.gate_atts = Gate(num_gates.shape[0], 
                                                  gamma=dataset.gate_gamma,
                                                  eta=dataset.gate_eta,
                                                  lr = dataset.gate_lr, 
                                                  temp=dataset.gate_temp,
                                                  lambda_l2=dataset.gate_lambda_l2, 
                                                  lambda_l0=dataset.gate_lambda_l0, 
                                                  init_probs=gaussians.init_probs,
                                                  inverse_indices=inverse_indices)
                        gaussians.gate_atts.train()
                    else:
                        gaussians.gate_atts.reset_params(init_probs=gaussians.init_probs)
                        gaussians.gate_atts.train()

                if use_adaptive_gate:
                    # Keep the original S²GS gate initialization formula so
                    # this mode changes grouping, not the learned prior.
                    gradient_median = gradient_scores.median()
                    point_init_probs = gradient_scores / (
                        gradient_scores + gradient_median)
                    point_init_probs = torch.nan_to_num(
                        point_init_probs, nan=0.0, posinf=1.0,
                        neginf=0.0).clamp_(0.0, 1.0)
                    gaussians.init_probs = point_init_probs
                    frame_gate_evidence = point_evidence.detach().clone()
                    frame_gradient_cue_points = int((
                        gradient_scores > dataset.gaussian_update_thresh
                    ).sum().item())
                    frame_image_cue_points = int(image_cue.sum().item())
                    partition = build_adaptive_gate_partition(
                        (frame_partition_root_indices
                         if use_adaptive_gate_v7b
                         else gaussians.get_root_indices),
                        (frame_partition_anchors
                         if use_adaptive_gate_v7b
                         else gaussians.get_anchor.detach()),
                        point_evidence,
                        (frame_partition_voxel_size
                         if use_adaptive_gate_v7b
                         else gaussians.voxel_size),
                        init_pos=(frame_partition_origin
                                  if use_adaptive_gate_v7b
                                  else gaussians.init_pos),
                        max_levels=(1 if use_adaptive_gate_v7b
                                    else args.adaptive_gate_max_levels),
                        min_points=args.adaptive_gate_min_points,
                        offset_bits_per_point=args.adaptive_gate_offset_bits)
                    if use_adaptive_gate_v7b:
                        # Stage one trains the exact S²GS root Gate; deeper
                        # levels are opened only at 100/200/300-step probes.
                        partition["max_levels"] = args.adaptive_gate_max_levels
                        partition["method"] = "adaptive_gate_partition_v7b"
                        partition["refinement_stats"] = []
                    point_group_ids = partition["point_group_ids"]
                    group_init_probs = scatter_max(
                        point_init_probs, point_group_ids, dim=0)[0]
                    gaussians.adaptive_gate_partition = partition
                    gaussians.adaptive_gate_topology_bits = int(
                        partition["topology_bits"])
                    gaussians.gate_atts = Gate(
                        partition["terminal_group_count"],
                        gamma=dataset.gate_gamma,
                        eta=dataset.gate_eta,
                        lr=dataset.gate_lr,
                        temp=dataset.gate_temp,
                        lambda_l2=dataset.gate_lambda_l2,
                        lambda_l0=dataset.gate_lambda_l0,
                        init_probs=group_init_probs,
                        inverse_indices=point_group_ids)
                    gaussians.gate_atts.train()
                    print(
                        f"Frame {frame_idx}: adaptive Gate roots "
                        f"{partition['root_count']}, mixed "
                        f"{partition['mixed_roots_eligible']}, splits "
                        f"{partition['split_count']}, terminal groups "
                        f"{partition['terminal_group_count']}, evidence "
                        f"{int(point_evidence.sum().item())}")

                if args.adaptive_hierarchy:
                    image_change_scores = (image_change_sum
                                           / image_change_count.clamp_min(1.0))
                    frame_gradient_cue_points = int((
                        grad_diff.flatten() > dataset.gaussian_update_thresh
                    ).sum().item())
                    frame_image_cue_points = int((
                        image_change_scores.flatten()
                        > args.hierarchy_image_change_threshold
                    ).sum().item())
                    gaussians.current_frame_shared_bits = 0
                    gaussians.gate_atts = None
                    gaussians.shared_motion = AdaptiveHierarchyMotion(
                        gaussians.get_anchor.detach(),
                        gaussians.init_pos,
                        gaussians.voxel_size,
                        gaussians.fork,
                        grad_diff.flatten(),
                        dataset.gaussian_update_thresh,
                        image_change_scores=image_change_scores.flatten(),
                        image_change_threshold=args.hierarchy_image_change_threshold,
                        levels=args.hierarchy_levels,
                        root_scale=args.hierarchy_root_scale,
                        lr=args.hierarchy_motion_lr,
                        lambda_deform=args.hierarchy_lambda_deform,
                        off_ratio=args.hierarchy_off_ratio,
                        update_ratio=args.hierarchy_update_ratio,
                        split_residual_ratio=args.hierarchy_split_residual_ratio,
                        split_heterogeneity=args.hierarchy_split_heterogeneity,
                        split_min_gain=args.hierarchy_split_min_gain,
                        split_relative_gain=args.hierarchy_split_relative_gain,
                    )
                    gaussians.set_hierarchy_update_mask(
                        gaussians.shared_motion.update_point_mask)
                    stage2_param_states = [(param, param.requires_grad)
                                           for param in gaussians.parameters()]
                    for param, _ in stage2_param_states:
                        param.requires_grad_(False)
                    hierarchy_probe_before = None
                    hierarchy_probe_snapshots = None
                    frame_new_gaussians = 0
                    print(f"Frame {frame_idx}: hierarchy regions per level "
                          f"{gaussians.shared_motion.node_counts}; initial states "
                          f"{gaussians.shared_motion.state_summary()}")

                if dataset.flow_update and opt.lambda_flow>0.0:
                    gaussians.update_points_flow()
                prev_frame_views = cur_frame_views

        if enable_tqdm and frame_idx == 1:
            progress_bar_iter = tqdm(range(first_iter, opt.iterations+1), 
                                     desc="Frame iteration progress")
        else:
            progress_bar_iter = None

        if dataset.timed:
            torch.cuda.synchronize()
        frame_time += time.time()- frame_start
        frame_start = time.time()
        frame_time_io += time.time() - frame_start_io
        frame_time_init += time.time() - frame_start_io
        frame_start_io = time.time()
        frame_time_training = 0.0

        # Start training iteration loop for current frame
        for iteration in range(first_iter, opt.iterations + 1):        

            hierarchy_stage1 = (args.adaptive_hierarchy and frame_idx > 1
                                and iteration <= args.hierarchy_stage1_steps)
            hierarchy_growth = (args.adaptive_hierarchy and frame_idx > 1
                                and iteration > args.hierarchy_stage1_steps
                                + args.hierarchy_gs_steps)
            if args.adaptive_hierarchy and frame_idx > 1 \
                    and iteration == args.hierarchy_stage1_steps + 1:
                for param, was_trainable in stage2_param_states:
                    param.requires_grad_(was_trainable)
                gaussians.shared_motion.requires_grad_(False)
                hierarchy = gaussians.shared_motion
                if torch.any(hierarchy.update_point_mask):
                    with torch.no_grad():
                        evidence = hierarchy._point_evidence()
                        initial_probs = torch.where(
                            evidence,
                            hierarchy.dynamic_scores.new_full(evidence.shape, 0.8),
                            hierarchy.dynamic_scores.new_full(evidence.shape, 0.04))
                    gaussians.hierarchy_residual_gate = AdaptiveOffsetGate(
                        hierarchy.update_point_mask, initial_probs,
                        lr=args.hierarchy_offset_gate_lr,
                        temperature=dataset.gate_temp,
                        lambda_l0=args.hierarchy_offset_gate_lambda_l0,
                        lambda_l2=args.hierarchy_offset_gate_lambda_l2)
                    # First fit the residual against a fixed, evidence-based
                    # gate.  Per-point sparsification from step one collapsed
                    # nearly every offset in the initial v5 run.
                    gaussians.hierarchy_residual_gate.logits.requires_grad_(False)
                print(f"Frame {frame_idx}: hierarchy fixed; updating "
                      f"{int(gaussians.hierarchy_update_mask.sum().item())} "
                      "UPDATE Gaussians with sparse offset gates; "
                      "other attributes follow S²GS")
            if (args.adaptive_hierarchy and frame_idx > 1
                    and iteration == args.hierarchy_stage1_steps
                    + args.hierarchy_gate_warmup_steps + 1
                    and gaussians.hierarchy_residual_gate is not None):
                gaussians.hierarchy_residual_gate.logits.requires_grad_(True)
                print(f"Frame {frame_idx}: offset gate learning enabled; "
                      f"{int(gaussians.hierarchy_residual_gate.active_mask().sum().item())} "
                      "offsets active after warmup")
            if (args.adaptive_hierarchy and frame_idx > 1
                    and iteration == args.hierarchy_stage1_steps
                    + args.hierarchy_gs_steps + 1):
                frame_hierarchy_payload = gaussians.shared_motion.frame_payload()
                gaussians.commit_shared_motion()
                print(f"Frame {frame_idx}: shared motion committed; "
                      "starting S²GS-style growth in UPDATE regions")

            if enable_debug:
                print(f"DEBUG: started iteration {iteration}")

            if dataset.timed:
                torch.cuda.synchronize()
            iter_start = time.time()

            gaussians.update_learning_rate(iteration, qp)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not camera_idx_stack:
                camera_idx_stack = list(range(n_cams))
            cam_idx = camera_idx_stack.pop(generator.randint(0, len(camera_idx_stack)-1))
            viewpoint_cam = train_cameras[cam_idx]

            # Render

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            # Loss
            gt_image = viewpoint_cam.original_image
            if opt.transform == "resize":
                gt_image = resize_image(gt_image, resize_scale_sched(iteration))
            elif "blur" in opt.transform and resize_scale_sched(iteration)!=1.0:
                if (iteration-1) % 100 == 0:
                    transform = blur_image(resize_scale_sched(iteration), opt.transform)
                gt_image = transform(gt_image)
            elif opt.transform == "downsample":
                gt_image = downsample_image(gt_image, resize_scale_sched(iteration))

            color_rw_mask = None

            # Initialize pixel_mask to None by default
            pixel_mask = None
            if frame_idx>1 and pix_thresh_vals is not None:
                if pix_thresh_vals[iteration-1]>0:
                    pixel_mask = viewpoint_cam.mask
            
            # render
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, iteration, "RGB")

            image, viewspace_point_tensor = render_pkg["render"], render_pkg["viewspace_points"]
            visibility_filter= render_pkg["visibility_filter"]
            anchor_visible_mask, offset_selection_mask = render_pkg["visible_mask"], render_pkg["selection_mask"]

            # Compute main reconstruction losses
            loss, Ll1 = torch.Tensor([0.0]).to(image.device), torch.Tensor([0.0]).to(image.device)
            if iteration>opt.color_from_iter:
                if pixel_mask is not None:
                    # Apply pixel mask for selective training
                    Ll1 = l1_loss(image*pixel_mask.unsqueeze(0), gt_image*pixel_mask.unsqueeze(0))
                    Lssim = ssim(image*pixel_mask.unsqueeze(0), gt_image*pixel_mask.unsqueeze(0))
                else:
                    Ll1 = l1_loss(image, gt_image)
                    Lssim = ssim(image, gt_image)

                loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - Lssim)

            # Add regularization losses
            if opt.weight_decay>0.0:
                loss += opt.weight_decay * gaussians.std_reg()

            if gaussians.gate_atts is not None and gaussians.gate_atts.training:
                loss += gaussians.gate_atts.reg_loss(gaussians._ungated_offset_res)
            if (gaussians.hierarchy_residual_gate is not None
                    and gaussians.hierarchy_residual_gate.training
                    and gaussians.hierarchy_residual_gate.logits.requires_grad):
                loss += gaussians.hierarchy_residual_gate.regularization(
                    gaussians._ungated_offset_res)

            if opt.lambda_posres>0.0:
                residual = gaussians.get_offset-prev_xyz.detach()
                loss += opt.lambda_posres*torch.abs(residual).mean()
            if iteration > opt.alpha_from_iter and opt.lambda_alpha>0.0:
                loss += opt.lambda_alpha * l2_loss(render_pkg["alpha"],1.0)

            # Temporal consistency loss
            if opt.lambda_consistency>0.0:
                prev_image = viewpoint_cam.prev_rendered
                cur_image = render_pkg["render"]
                gt_diff = viewpoint_cam.image_diff
                # High consistency loss for low varying regions
                gt_diff = 1/(gt_diff+gt_diff.mean()) 
                # Normalize
                gt_diff = gt_diff/gt_diff.mean()
                consistency_loss = 1- l1_loss(prev_image*gt_diff, cur_image*gt_diff)
                loss += opt.lambda_consistency*consistency_loss
            if hierarchy_stage1:
                loss += gaussians.shared_motion.regularization()
            loss.backward()
            if (hierarchy_stage1 and iteration == 1
                    and torch.any(gaussians.hierarchy_update_mask)):
                if not any(param.grad is not None and torch.any(param.grad != 0)
                           for param in gaussians.shared_motion.translations):
                    raise RuntimeError("Hierarchy motion received no gradient")
            if hierarchy_stage1:
                error_map = (image.detach() - gt_image).abs().mean(dim=0)
                point_error, point_visible = sample_pixel_map(
                    viewspace_point_tensor.squeeze(0), error_map,
                    visibility_filter)
                gaussians.shared_motion.accumulate_residual(
                    point_visible, point_error[point_visible])
            if enable_debug:
                print(f'DEBUG ({iteration}): backpropagated')

            with torch.no_grad():
                if dataset.timed:
                    torch.cuda.synchronize()
                frame_time += time.time() - iter_start
                frame_time_io += time.time() - iter_start
                frame_time_training += time.time() - iter_start
                net_elapsed_time = time.time() - training_start
                # Log and save
                if dataset.test_interval>0:
                    is_test = (iteration % dataset.test_interval == 0) and frame_idx == 1
                else:
                    is_test = (iteration in testing_iterations) and frame_idx == 1
                if iteration == opt.iterations:
                    is_test = True
                
                report = training_report(tb_writer, wandb_enabled, dataset, frame_idx, iteration, Ll1, loss, 
                                         l1_loss, cur_size, frame_time, is_test, scene, 
                                         render, (pipe, background, iteration, "RGB"), prev_report=report)

                if report:
                    if 'val' in report.keys():
                        report_configs = ['test','val']
                    else:
                        report_configs = ['test']
                    for config_name in report_configs:
                        metrics[config_name]['psnr'] = report[config_name]['psnr']
                        metrics[config_name]['loss'] = report[config_name]['l1']
                        metrics[config_name]['fps'] = report[config_name]['fps']
                        metrics[config_name]['storage'] = report[config_name]['storage']
                    if metrics['test']['psnr'] > best_psnr:
                        best_psnr = metrics['test']['psnr']

                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

                if (iteration) % dataset.log_interval == 0 or iteration == opt.iterations:
                    log_dict = {
                                "Loss": f"{ema_loss_for_log:.{5}f}",
                                "Num points": f"{gaussians._offset.shape[0]}",
                        "Update points": int(gaussians.hierarchy_update_mask.sum().item())
                                                if (frame_idx > 1 and gaussians.hierarchy_update_mask is not None)
                                                else gaussians._offset.shape[0],
                                "Active offsets": int(gaussians.hierarchy_residual_gate.active_mask().sum().item())
                                                  if gaussians.hierarchy_residual_gate is not None
                                                  else 0,
                                "Size (MB)": f"{metrics['test']['storage']:.{2}f}",
                                "FPS": f"{metrics['test']['fps']:.{2}f}",
                                "PSNR (Test)": f"{metrics['test']['psnr']:.{2}f}",
                                "PSNR (Val)": f"{metrics['val']['psnr']:.{2}f}",
                                }
                    if progress_bar_iter:
                        progress_bar_iter.set_postfix(log_dict)
                        progress_bar_iter.update(dataset.log_interval)
                if iteration == opt.iterations and progress_bar_iter:
                    progress_bar_iter.close()

                if dataset.timed:
                    torch.cuda.synchronize()
                iter_start = time.time()
                frame_time_densify = 0.0

                # Gaussian Densification
                if (iteration <= (np.ceil(opt.densify_until_epoch*n_cams*opt.iterations))
                        and iteration > (opt.calc_dense_stats*n_cams)
                        and (not args.adaptive_hierarchy or frame_idx == 1
                             or hierarchy_growth)):
                    # Track max radii in image-space for pruning
                    gaussians.add_densification_stats(render_pkg, image.shape[2], image.shape[1])

                    if (args.adaptive_hierarchy and frame_idx > 1):
                        growth_step = (iteration - args.hierarchy_stage1_steps
                                       - args.hierarchy_gs_steps)
                        if growth_step == args.hierarchy_growth_steps // 2:
                            added = gaussians.grow_hierarchy_dynamic(
                                iteration, opt_lod,
                                args.hierarchy_growth_candidates)
                            frame_new_gaussians += added
                            print(f"Frame {frame_idx}: added {added} GS; "
                                  f"total {gaussians.get_anchor.shape[0]}")
                    elif iteration > opt_lod.update_from and iteration % opt_lod.update_interval == 0:
                        if frame_idx == 1:
                            # Standard densification for first frame
                            gaussians.run_densify(iteration, opt_lod)
                        # else:
                        #     # Dynamic densification for subsequent frames
                            gaussians.densify_dynamic(iteration, opt_lod)

                    if enable_debug:
                        print(f'DEBUG ({iteration}): densification done')

                if dataset.timed:
                    torch.cuda.synchronize()
                frame_time += time.time()-iter_start
                frame_time_io += time.time()-iter_start
                frame_time_densify += time.time()-iter_start

                with torch.no_grad():
                    if (opt.iterations - iteration) < (2*n_cams): # Save most recent render for final epochs
                        viewpoint_cam.prev_rendered = render_pkg["render"].detach()

                    if (opt.iterations - iteration)<(n_cams) and cam_idx == 1 and (dataset.log_images or dataset.log_compressed or dataset.log_ply):
                        
                        if dataset.log_images:
                            save_image(gt_image,os.path.join(scene.model_path, "gt.png"))

                        # if dataset.log_ply:
                        #     scene.save(iteration, save_point_cloud=True)
                        
                        # if dataset.log_compressed:
                        #     if frame_idx == 1:
                        #         scene.save(frame_idx, save_point_cloud=True)                                

                        # if frame_idx>1 and (dataset.adaptive_render and dataset.adaptive_update_period>0.0) and dataset.update_mask!="none":
                        #     torchvision.utils.save_image(train_cameras[cam_idx].mask.unsqueeze(0)*gt_image,
                        #                                  os.path.join(scene.model_path, "mask.png"))
                        #     torchvision.utils.save_image(train_cameras[cam_idx].orig_mask.unsqueeze(0)*gt_image,
                        #                                  os.path.join(scene.model_path, "orig_mask.png"))
                        if dataset.log_images:
                            video_camera = video_cameras[frame_idx-1]
                            spiral_img = render(video_camera, gaussians, pipe, bg, iteration, "RGB")["render"]
                            if frame_idx == 1:
                                os.makedirs(os.path.join(dataset.model_path,"spiral"), exist_ok=True)
                            save_image(torch.clip(spiral_img, 0.0, 1.0),os.path.join(dataset.model_path, "spiral", f"{str(dataset.start_idx + frame_idx).zfill(4)}.png"))

                        # if frame_idx == 1:
                        #     with torch.no_grad():
                        #         render_pkg = render_mask(viewpoint_cam, gaussians, pipe, bg, iteration, image_shape=gt_image.shape, 
                        #                                  color_mask=color_rw_mask, render_depth=True)
                        #         pred_depth = render_pkg["depth"]
                        #         render_depth = pred_depth.detach().cpu().numpy()

                        #     if opt.lambda_depth>0.0 or dataset.depth_init:
                        #         gt_depth = viewpoint_cam.gt_depth
                        #         gt_depth = gt_depth.detach().cpu().numpy()
                        #         gt_depth = (gt_depth-gt_depth.min())/(gt_depth.max()-gt_depth.min())
                        #         render_depth = (render_depth-render_depth.min())/(render_depth.max()-render_depth.min())
                        #         depth_ssim = ssim(torch.tensor(render_depth).cuda().unsqueeze(0), torch.tensor(gt_depth).cuda().unsqueeze(0)).item()
                        #         depth_psnr = psnr(torch.tensor(gt_depth).cuda().unsqueeze(0), torch.tensor(render_depth).cuda().unsqueeze(0)).item()
                        #         depth_err = np.abs(render_depth-gt_depth)
                        #         depth_err = torch.abs(render_pkg["depth"]-viewpoint_cam.gt_depth).detach().cpu().numpy()
                        #         torchvision.utils.save_image(torch.tensor(depth_err).unsqueeze(0),os.path.join(dataset.model_path,'err_depth_gray.png'))

                # Optimizer step
                if dataset.timed:
                    torch.cuda.synchronize()
                iter_start = time.time()
                frame_time_grad = 0.0
                if iteration <= opt.iterations:
                    # gaussians.update_grads()
                    if hierarchy_stage1:
                        gaussians.shared_motion.optimizer.step()
                        gaussians.shared_motion.optimizer.zero_grad(set_to_none=True)
                    else:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none = True)
                        if gaussians.gate_atts is not None and gaussians.gate_atts.training:
                            gaussians.gate_atts.step()
                            gaussians.gate_atts.clamp_params()
                        if (gaussians.hierarchy_residual_gate is not None
                                and gaussians.hierarchy_residual_gate.training
                                and gaussians.hierarchy_residual_gate.logits.requires_grad):
                            gaussians.hierarchy_residual_gate.optimizer.step()
                            gaussians.hierarchy_residual_gate.optimizer.zero_grad(set_to_none=True)
                    if (use_adaptive_gate_v7b and frame_idx > 1
                            and frame_gate_evidence is not None
                            and iteration in (100, 200, 300)):
                        target_level = iteration // 100 - 1
                        stage_stats = refine_v7b_gate_frame(
                            gaussians, dataset, frame_gate_evidence,
                            frame_partition_root_indices,
                            frame_partition_anchors, frame_partition_origin,
                            frame_partition_voxel_size, args, target_level)
                        if stage_stats is not None:
                            frame_gate_stage_stats.append(stage_stats)
                            print(
                                f"Frame {frame_idx} step {iteration}: V7b "
                                f"level {target_level} candidates "
                                f"{stage_stats['eligible_groups']}, active "
                                f"mixed {stage_stats['cue_mixed_groups']}, "
                                f"residual-qualified "
                                f"{stage_stats['residual_heterogeneous_groups']}, "
                                f"split {stage_stats['rate_accepted_groups']}, "
                                f"active points {stage_stats['active_parent_points']}, "
                                f"static child points "
                                f"{stage_stats['static_child_points']}, saved/topology/"
                                f"gate-budget bits "
                                f"{stage_stats['predicted_saved_bits']}/"
                                f"{stage_stats['topology_bits']}/"
                                f"{stage_stats['group_gate_overhead_bits']}")
                    if hierarchy_stage1:
                        level_steps = max(
                            1, args.hierarchy_stage1_steps
                            // args.hierarchy_levels)
                        level_step = (iteration - 1) % level_steps + 1
                        trial_start = max(1, level_steps - args.hierarchy_trial_steps)
                        hierarchy = gaussians.shared_motion
                        if (level_step == trial_start
                                and hierarchy.current_level + 1 < hierarchy.levels
                                and not hierarchy.finished):
                            probe_cameras = [
                                train_cameras[(frame_idx + hierarchy.current_level) % n_cams],
                                train_cameras[(frame_idx + hierarchy.current_level
                                               + n_cams // 2) % n_cams],
                            ]
                            probe_level = hierarchy.current_level
                            hierarchy_probe_before, hierarchy_probe_snapshots = (
                                hierarchy_region_probe(
                                    hierarchy, gaussians, probe_cameras, pipe,
                                    background, iteration, probe_level))
                            if hierarchy.begin_split_trial():
                                frame_trial_candidates += hierarchy.last_trial_candidates
                                print(f"Frame {frame_idx} level {probe_level}: "
                                      f"trial splits {hierarchy.last_trial_candidates}")
                                hierarchy.translations[probe_level].requires_grad_(False)
                                hierarchy.rotations[probe_level].requires_grad_(False)
                                gaussians.set_hierarchy_update_mask(
                                    hierarchy.update_point_mask)
                            else:
                                hierarchy_probe_before = None
                        if level_step == level_steps or iteration == args.hierarchy_stage1_steps:
                            if hierarchy.trial_nodes is not None:
                                probe_level = hierarchy.current_level - 1
                                after_error, _ = hierarchy_region_probe(
                                    hierarchy, gaussians, probe_cameras, pipe,
                                    background, iteration, probe_level,
                                    hierarchy_probe_snapshots)
                                hierarchy_states = hierarchy.finish_split_trial(
                                    hierarchy_probe_before, after_error)
                                frame_trial_accepted += hierarchy.last_trial_accepted
                                print(f"Frame {frame_idx} level {probe_level}: "
                                      f"accepted splits {hierarchy.last_trial_accepted} "
                                      f"/ {hierarchy.last_trial_candidates}")
                                hierarchy.translations[probe_level].requires_grad_(True)
                                hierarchy.rotations[probe_level].requires_grad_(True)
                                hierarchy_probe_before = None
                                hierarchy_probe_snapshots = None
                            if iteration == args.hierarchy_stage1_steps:
                                hierarchy_states = hierarchy.finalize()
                            gaussians.set_hierarchy_update_mask(
                                hierarchy.update_point_mask)
                            print(f"Frame {frame_idx} step {iteration}: "
                                  f"hierarchy states {hierarchy.state_summary()}; "
                                  f"update GS "
                                  f"{int(gaussians.hierarchy_update_mask.sum().item())}")
                    if (args.adaptive_hierarchy and 1 < frame_idx <= 10
                            and iteration in (
                                args.hierarchy_stage1_steps,
                                args.hierarchy_stage1_steps + args.hierarchy_gs_steps,
                                opt.iterations)
                            and scene.getTestCameras()):
                        probe_camera = scene.getTestCameras()[0]
                        probe_image = render(
                            probe_camera, gaussians, pipe, background,
                            iteration, "RGB")["render"]
                        probe_psnr = psnr(
                            probe_image, probe_camera.original_image).mean().item()
                        stage_probe_psnr[str(iteration)] = round(probe_psnr, 3)
                        print(f"Frame {frame_idx} step {iteration}: "
                              f"single-view test PSNR {probe_psnr:.3f}")
                if dataset.timed:
                    torch.cuda.synchronize()
                frame_time += time.time()-iter_start
                frame_time_io += time.time()-iter_start
                frame_time_grad += time.time()-iter_start
                if enable_debug:
                    print(f'DEBUG ({iteration}): Optimizer step done')
        # end training loop for this frame

        if dataset.timed:
            torch.cuda.synchronize()
        frame_start = time.time()
        frame_time_save_update = 0.0

        hierarchy_bits = 0
        hierarchy_states = []
        updated_gaussians = 0
        if use_adaptive_gate and frame_idx > 1:
            sq_res_open_points = {}
            for param_index, param_name in enumerate(gaussians.param_names):
                if qp.quant_type[param_index] != "sq_res":
                    continue
                decoder_gate = getattr(
                    gaussians.latent_decoders[param_name], "gates", None)
                if decoder_gate is None:
                    open_points = int(gaussians.get_anchor.shape[0])
                else:
                    open_points = int((decoder_gate.flatten() > 0.05).sum().item())
                sq_res_open_points[param_name] = open_points
            group_gates = gaussians.gate_atts.sample_gate_Gumbel_Softmax(
                stochastic=False)
            gate_dir = os.path.join(args.model_path,
                                    "adaptive_gate_partition")
            os.makedirs(gate_dir, exist_ok=True)
            partition = gaussians.adaptive_gate_partition
            compact_payload_bytes = 0
            debug_payload_bytes = 0
            if use_adaptive_gate_v7b:
                frame_gate_payload = build_compact_gate_payload(
                    partition, group_gates, frame_idx,
                    frame_partition_origin, frame_partition_voxel_size,
                    active_threshold=0.05,
                    sqres_reopen_mask_payload=(
                        frame_sqres_reopen_mask_payload
                        if args.adaptive_gate_sqres_cue_floor > 0.0
                        else None))
                # Validate the actual serialized representation against the
                # current frozen frame inputs before saving it. These checks
                # do not touch the optimizer or any training state.
                rebuilt_groups, rebuilt_group_gates, rebuilt_point_gates = (
                    decode_compact_gate_payload(
                        frame_gate_payload, frame_partition_root_indices,
                        frame_partition_anchors))
                expected_point_gates = group_gates.index_select(
                    0, partition["point_group_ids"])
                if not torch.equal(rebuilt_groups,
                                   partition["point_group_ids"]):
                    raise RuntimeError(
                        "Compact V7b sidecar changed point-to-group mapping")
                if (not torch.equal(rebuilt_group_gates, group_gates)
                        or not torch.equal(rebuilt_point_gates,
                                           expected_point_gates)):
                    group_equal = torch.equal(rebuilt_group_gates, group_gates)
                    point_equal = torch.equal(
                        rebuilt_point_gates, expected_point_gates)
                    group_max_error = float((
                        rebuilt_group_gates - group_gates).abs().max().item())
                    point_max_error = float((
                        rebuilt_point_gates - expected_point_gates
                    ).abs().max().item())
                    raise RuntimeError(
                        "Compact V7b sidecar changed Gate values: "
                        f"group_equal={group_equal}, point_equal={point_equal}, "
                        f"group_max_error={group_max_error}, "
                        f"point_max_error={point_max_error}")
                if args.adaptive_gate_sqres_cue_floor > 0.0:
                    mask_payload = frame_gate_payload.get(
                        "sq_res_reopen_mask")
                    if mask_payload is None or frame_sqres_reopen_mask is None:
                        raise RuntimeError(
                            "V7d sidecar is missing its sq_res reopen mask")
                    decoded_reopen_mask = decode_sqres_reopen_mask(
                        mask_payload,
                        expected_point_count=partition[
                            "point_group_ids"].numel(),
                        device=frame_sqres_reopen_mask.device)
                    if not torch.equal(decoded_reopen_mask,
                                       frame_sqres_reopen_mask):
                        raise RuntimeError(
                            "V7d reopen-mask bitstream changed point order or bits")

                    old_gate = getattr(gaussians, "gate_before", None)
                    if frame_sqres_previous_point_gate is None:
                        if old_gate is not None:
                            raise RuntimeError(
                                "Unexpected old offset Gate during F2 V7d decode")
                        expected_decoder_gate = None
                    else:
                        if old_gate is None:
                            raise RuntimeError(
                                "V7d decoder lost its previous offset Gate")
                        expected_decoder_gate = torch.where(
                            decoded_reopen_mask,
                            torch.clamp(
                                old_gate.flatten(),
                                min=args.adaptive_gate_sqres_cue_floor),
                            old_gate.flatten())
                    sqres_gate_names = {"f_dc", "sc", "rot", "op"}
                    actual_decoder_gate = None
                    for param_index, param_name in enumerate(
                            gaussians.param_names):
                        if (param_name not in sqres_gate_names
                                or qp.quant_type[param_index] != "sq_res"
                                or gaussians.gate_params.get(
                                    param_name, False)):
                            continue
                        actual = getattr(
                            gaussians.latent_decoders[param_name], "gates", None)
                        if expected_decoder_gate is None:
                            if actual is not None:
                                raise RuntimeError(
                                    f"F2 {param_name} decoder should remain "
                                    "ungated without a previous Gate")
                        else:
                            if gaussians.gate_params.get(param_name, False):
                                expected_for_attribute = old_gate.flatten()
                            else:
                                expected_for_attribute = expected_decoder_gate
                            if (actual is None or not torch.equal(
                                    actual.flatten(),
                                    expected_for_attribute)):
                                raise RuntimeError(
                                    f"V7d {param_name} decoder Gate does not "
                                    "round-trip from its previous Gate policy")
                        if (actual_decoder_gate is None
                                and not gaussians.gate_params.get(
                                    param_name, False)):
                            actual_decoder_gate = actual

                    # This diagnostic isolates any pre-existing difference
                    # between the cached last-forward Gate used by training
                    # and the previous frame's compact Gate used by decoding.
                    if (actual_decoder_gate is not None
                            and last_sidecar_point_gates is not None):
                        sidecar_old_gate = last_sidecar_point_gates.to(
                            actual_decoder_gate).flatten()
                        sidecar_rebuilt = torch.where(
                            decoded_reopen_mask,
                            torch.clamp(
                                sidecar_old_gate,
                                min=args.adaptive_gate_sqres_cue_floor),
                            sidecar_old_gate)
                        restore_diff = (
                            actual_decoder_gate.flatten() - sidecar_rebuilt)
                        frame_sidecar_decoder_gate_mismatch_points = int(
                            torch.count_nonzero(restore_diff).item())
                        frame_sidecar_decoder_gate_max_abs_diff = float(
                            restore_diff.abs().max().item())
                compact_path = os.path.join(
                    gate_dir, f"frame_{frame_idx:04d}.pt")
                torch.save(frame_gate_payload, compact_path)
                compact_payload_bytes = os.path.getsize(compact_path)
                last_sidecar_point_gates = rebuilt_point_gates.detach().clone()

                if args.save_adaptive_gate_debug_payload:
                    debug_payload = finalize_gate_partition(
                        partition, group_gates)
                    debug_payload.update({
                        "frame_idx": frame_idx,
                        "coordinate_rule": (
                            "previous_frame_decoded_anchor_octree"),
                        "method": "adaptive_gate_partition_v7b",
                        "root_cell_side": float(frame_partition_voxel_size),
                        "child_cell_side_base": float(
                            frame_partition_voxel_size / 2.0),
                        "grid_origin": frame_partition_origin.detach().cpu(),
                        "root_values": partition[
                            "root_values"].detach().cpu(),
                        "sq_res_decoder_open_points": sq_res_open_points,
                        "refinement_stats": frame_gate_stage_stats,
                    })
                    debug_payload["format"] = "adaptive_gate_partition_v7b"
                    debug_payload["topology_anchor_source"] = (
                        "frozen_previous_frame_decoded_anchors")
                    debug_dir = os.path.join(
                        args.model_path, "adaptive_gate_debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    debug_path = os.path.join(
                        debug_dir, f"frame_{frame_idx:04d}.pt")
                    torch.save(debug_payload, debug_path)
                    debug_payload_bytes = os.path.getsize(debug_path)
            else:
                # Preserve the legacy V7 payload and save path verbatim.
                frame_gate_payload = finalize_gate_partition(
                    partition, group_gates)
                frame_gate_payload.update({
                    "frame_idx": frame_idx,
                    "coordinate_rule": (
                        "previous_frame_decoded_anchor_octree"),
                    "method": "adaptive_gate_partition_v7_legacy",
                    "root_cell_side": float(gaussians.voxel_size),
                    "child_cell_side_base": float(
                        gaussians.voxel_size / 2.0),
                    "grid_origin": gaussians.init_pos.detach().cpu(),
                    "root_values": partition["root_values"].detach().cpu(),
                    "sq_res_decoder_open_points": sq_res_open_points,
                })
                torch.save(
                    frame_gate_payload,
                    os.path.join(gate_dir, f"frame_{frame_idx:04d}.pt"))
            active_offset_mask = gaussians.adaptive_gate_active_mask()
            active_offsets = (
                int(active_offset_mask.sum().item())
                if (active_offset_mask is not None
                    and gaussians.gate_params.get("offset", False))
                else int(gaussians.get_offset.shape[0]))
            frame_gate_stats = {
                "roots": int(gaussians.adaptive_gate_partition["root_count"]),
                "mixed_roots": int(gaussians.adaptive_gate_partition[
                    "mixed_roots_eligible"]),
                "terminal_groups": int(partition["terminal_group_count"]),
                "splits": int(partition["split_count"]),
                "active_groups": int((group_gates > 0.05).sum().item()),
                "active_offsets": active_offsets,
                "sq_res_decoder_open_points": sq_res_open_points,
                "topology_bits": int(partition["topology_bits"]),
                "group_gate_overhead_bits": int(partition[
                    "group_gate_overhead_bits"]),
                "predicted_saved_bits": int(partition[
                    "predicted_saved_bits"]),
                "net_predicted_saved_bits": int(partition[
                    "net_predicted_saved_bits"]),
                "refinement_stats": frame_gate_stage_stats,
                "compact_payload_bytes": int(compact_payload_bytes),
                "debug_payload_bytes": int(debug_payload_bytes),
            }
            if args.adaptive_gate_sqres_cue_floor > 0.0:
                frame_gate_stats["sqres_cue_reopened_points"] = int(
                    frame_sqres_cue_reopened_points)
                frame_gate_stats["sqres_reopen_mask_bytes"] = int(
                    frame_sqres_reopen_mask_bytes)
                frame_gate_stats["sqres_reopen_mask_raw_bytes"] = int(
                    frame_sqres_reopen_mask_raw_bytes)
                frame_gate_stats["previous_gate_cache_mismatch_points"] = int(
                    frame_previous_gate_cache_mismatch_points)
                frame_gate_stats["previous_gate_cache_max_abs_diff"] = float(
                    frame_previous_gate_cache_max_abs_diff)
                frame_gate_stats["sidecar_decoder_gate_mismatch_points"] = int(
                    frame_sidecar_decoder_gate_mismatch_points)
                frame_gate_stats["sidecar_decoder_gate_max_abs_diff"] = float(
                    frame_sidecar_decoder_gate_max_abs_diff)
            print(
                f"Frame {frame_idx}: adaptive Gate payload roots "
                f"{frame_gate_stats['roots']}, terminal groups "
                f"{frame_gate_stats['terminal_groups']}, splits "
                f"{frame_gate_stats['splits']}, active offsets "
                f"{frame_gate_stats['active_offsets']}, predicted saved/topology "
                f"bits {frame_gate_stats['predicted_saved_bits']}/"
                f"{frame_gate_stats['topology_bits']}, extra gate-group "
                f"budget {frame_gate_stats['group_gate_overhead_bits']}")
            if args.adaptive_gate_sqres_cue_floor > 0.0:
                print(
                    f"Frame {frame_idx}: sq_res cue-reopened points "
                    f"{frame_gate_stats['sqres_cue_reopened_points']}, mask "
                    f"coded/raw bytes "
                    f"{frame_gate_stats['sqres_reopen_mask_bytes']}/"
                    f"{frame_gate_stats['sqres_reopen_mask_raw_bytes']}, "
                    f"prior-cache differences "
                    f"{frame_gate_stats['previous_gate_cache_mismatch_points']} "
                    f"(max {frame_gate_stats['previous_gate_cache_max_abs_diff']:.7g}), "
                    f"sidecar restore differences "
                    f"{frame_gate_stats['sidecar_decoder_gate_mismatch_points']} "
                    f"(max {frame_gate_stats['sidecar_decoder_gate_max_abs_diff']:.7g})")
        if args.adaptive_hierarchy and frame_idx > 1:
            if frame_hierarchy_payload is None:
                frame_hierarchy_payload = gaussians.shared_motion.frame_payload()
                gaussians.commit_shared_motion()
            frame_hierarchy_payload["new_gaussians"] = frame_new_gaussians
            if gaussians.hierarchy_residual_gate is not None:
                frame_hierarchy_payload["offset_gate"] = (
                    gaussians.hierarchy_residual_gate.frame_payload())
            hierarchy_bits = frame_hierarchy_payload["estimated_bits"]
            hierarchy_states = [
                {"off": int((level["region_states"] == 0).sum().item()),
                 "update": int((level["region_states"] == 1).sum().item()),
                 "split": int((level["region_states"] == 2).sum().item())}
                for level in frame_hierarchy_payload["levels"]]
            updated_gaussians = int(gaussians.hierarchy_update_mask.sum().item())
            hierarchy_dir = os.path.join(args.model_path, "adaptive_hierarchy")
            os.makedirs(hierarchy_dir, exist_ok=True)
            torch.save(
                frame_hierarchy_payload,
                os.path.join(hierarchy_dir, f"frame_{frame_idx:04d}.pt"))

        # Update previous frame's attributes and latents for next frame's residual encoding
        if frame_idx != n_frames:
            # Used for residual encoding of next frame
            with torch.no_grad():
                for att_name in gaussians.get_atts:
                    prev_atts = gaussians.get_decoded_atts[att_name].clone()
                    prev_latents = gaussians.get_atts[att_name].clone()
                    gaussians.prev_atts[att_name] = prev_atts
                    gaussians.prev_latents[att_name] = prev_latents
                    gaussians.prev_atts[att_name].requires_grad_(False)
                    gaussians.prev_latents[att_name].requires_grad_(False)
                    gaussians.prev_atts_initial[att_name] = prev_atts.clone()
            cur_frame_views = next_frame_views
            cur_train_images = next_train_images
            cur_train_paths = next_train_paths
            prev_xyz = gaussians._offset.clone()

        if args.save_frame1_snapshot and frame_idx == 1:
            snapshot_info = save_frame1_snapshot(
                args.save_frame1_snapshot, gaussians, generator)
            print(f"Saved frame-one snapshot: {snapshot_info['path']}")

        if dataset.timed:
            torch.cuda.synchronize()
        frame_time += time.time()-frame_start
        frame_time_io += time.time()-frame_start
        frame_time_save_update += time.time()-frame_start

        # Collect frame metrics for logging
        if test_image_dataset.n_cams > 0:
            frame_metrics = {
                "Frame index": frame_idx,
                "Loss": round(ema_loss_for_log,5),
                "Loss (Test)": round(metrics['test']['loss'].item(),5),
                "Num points": gaussians._offset.shape[0],
                "Update points": int(gaussians.hierarchy_update_mask.sum().item())
                                    if (frame_idx > 1 and gaussians.hierarchy_update_mask is not None)
                                    else gaussians._offset.shape[0],
                "Active gates": (frame_gate_stats.get("active_groups", 0)
                                 if use_adaptive_gate
                                 else sum(level["update"] for level in hierarchy_states)),
                "Size (MB)": round(metrics['test']['storage'],5),
                "FPS": round(metrics['test']['fps'],5),
                "PSNR (Test)": round(metrics['test']['psnr'].item(),2),
                "Frame time": round(frame_time,2),
                "Frame time IO": round(frame_time_io,2),           # 较于Frame time, 额外包括计算机在执行数据输入和输出操作(I/O)时花费的时间
                "Training time elapsed": round(net_elapsed_time,2),
                "Frame time initialization": round(frame_time_init, 2),
                "Frame time training": round(frame_time_training, 2),
                "Frame time densification": round(frame_time_densify),
                "Frame time grad": round(frame_time_grad),
                "Frame time save & update": round(frame_time_save_update),
            }
        else:
            # Not using test cameras
            frame_metrics = {
                "Frame index": frame_idx,
                "Loss": round(ema_loss_for_log,5),
                "Num points": gaussians._offset.shape[0],
                "Update points": int(gaussians.hierarchy_update_mask.sum().item())
                                    if (frame_idx > 1 and gaussians.hierarchy_update_mask is not None)
                                    else gaussians._offset.shape[0],
                "Active gates": (frame_gate_stats.get("active_groups", 0)
                                 if use_adaptive_gate
                                 else sum(level["update"] for level in hierarchy_states)),
                "Size (MB)": round(metrics['test']['storage'],5),
                "FPS": round(metrics['test']['fps'].item(),2),
                "Frame time": round(frame_time,2),
                "Frame time IO": round(frame_time_io,2),
                "Training time elapsed": round(net_elapsed_time,2),
                "Frame time initialization": round(frame_time_init, 2),
                "Frame time training": round(frame_time_training, 2),
                "Frame time densification": round(frame_time_densify),
                "Frame time grad": round(frame_time_grad),
                "Frame time save & update": round(frame_time_save_update),
            }

        training_metrics.append(frame_metrics)
        if args.adaptive_hierarchy:
            frame_metrics["Hierarchy states"] = hierarchy_states
            frame_metrics["Hierarchy estimated MB"] = round(
                hierarchy_bits / 8 / 1e6, 5)
            frame_metrics["Hierarchy update Gaussians"] = updated_gaussians
            frame_metrics["Active offset Gaussians"] = (
                int(gaussians.hierarchy_residual_gate.active_mask().sum().item())
                if frame_idx > 1 and gaussians.hierarchy_residual_gate is not None
                else 0)
            frame_metrics["Offset gate estimated MB"] = (
                round(gaussians.hierarchy_residual_gate.size_bits() / 8 / 1e6, 5)
                if frame_idx > 1 and gaussians.hierarchy_residual_gate is not None
                else 0.0)
            frame_metrics["Hierarchy trial splits"] = frame_trial_candidates
            frame_metrics["Hierarchy accepted splits"] = frame_trial_accepted
            frame_metrics["Hierarchy new Gaussians"] = (
                frame_new_gaussians if frame_idx > 1 else 0)
            frame_metrics["Gradient cue Gaussians"] = frame_gradient_cue_points
            frame_metrics["Image cue Gaussians"] = frame_image_cue_points
            frame_metrics["Stage single-view PSNR"] = stage_probe_psnr
        if use_adaptive_gate:
            frame_metrics["Adaptive gate method"] = (
                ("adaptive_gate_partition_v7d"
                 if args.adaptive_gate_sqres_cue_floor > 0.0
                 else "adaptive_gate_partition_v7b")
                if use_adaptive_gate_v7b
                else "adaptive_gate_partition_v7_legacy")
            frame_metrics["Adaptive refinement stages"] = frame_gate_stats.get(
                "refinement_stats", [])
            frame_metrics["S2GS root groups"] = frame_gate_stats.get("roots", 0)
            frame_metrics["Adaptive terminal groups"] = frame_gate_stats.get(
                "terminal_groups", 0)
            frame_metrics["Adaptive split parents"] = frame_gate_stats.get(
                "splits", 0)
            frame_metrics["Cue-mixed S2GS roots"] = frame_gate_stats.get(
                "mixed_roots", 0)
            frame_metrics["Active adaptive groups"] = frame_gate_stats.get(
                "active_groups", 0)
            frame_metrics["Active offset Gaussians"] = frame_gate_stats.get(
                "active_offsets", 0)
            frame_metrics["sq_res decoder open points"] = frame_gate_stats.get(
                "sq_res_decoder_open_points", {})
            if args.adaptive_gate_sqres_cue_floor > 0.0:
                frame_metrics["sq_res cue-reopened points"] = (
                    frame_gate_stats.get("sqres_cue_reopened_points", 0))
                frame_metrics["sq_res cue mask coded bytes"] = (
                    frame_gate_stats.get("sqres_reopen_mask_bytes", 0))
                frame_metrics["sq_res cue mask raw bytes"] = (
                    frame_gate_stats.get("sqres_reopen_mask_raw_bytes", 0))
                frame_metrics["Previous Gate cache mismatch points"] = (
                    frame_gate_stats.get(
                        "previous_gate_cache_mismatch_points", 0))
                frame_metrics["Previous Gate cache max abs diff"] = (
                    frame_gate_stats.get(
                        "previous_gate_cache_max_abs_diff", 0.0))
                frame_metrics["sq_res sidecar restore mismatch points"] = (
                    frame_gate_stats.get(
                        "sidecar_decoder_gate_mismatch_points", 0))
                frame_metrics["sq_res sidecar restore max abs diff"] = (
                    frame_gate_stats.get(
                        "sidecar_decoder_gate_max_abs_diff", 0.0))
            frame_metrics["Adaptive topology MB"] = round(
                frame_gate_stats.get("topology_bits", 0) / 8 / 1e6, 5)
            frame_metrics["Extra gate-group budget MB"] = round(
                frame_gate_stats.get("group_gate_overhead_bits", 0)
                / 8 / 1e6, 5)
            frame_metrics["Predicted offset savings MB"] = round(
                frame_gate_stats.get("predicted_saved_bits", 0) / 8 / 1e6, 5)
            frame_metrics["Predicted net savings MB"] = round(
                frame_gate_stats.get("net_predicted_saved_bits", 0)
                / 8 / 1e6, 5)
            if use_adaptive_gate_v7b:
                frame_metrics["Adaptive compact sidecar bytes"] = int(
                    frame_gate_stats.get("compact_payload_bytes", 0))
                frame_metrics["Adaptive debug payload bytes"] = int(
                    frame_gate_stats.get("debug_payload_bytes", 0))
            frame_metrics["Gradient cue Gaussians"] = frame_gradient_cue_points
            frame_metrics["Image cue Gaussians"] = frame_image_cue_points
        # Metric labels for video demos
        if dataset.log_images and dataset.log_labels:
            # For the test view
            test_view_path = os.path.join(dataset.model_path, "test", "renders", "cam00", f"{str(dataset.start_idx + frame_idx).zfill(4)}.png")
            add_metrics_label_to_image(test_view_path, frame_metrics)
            # For the spiral views
            spiral_path = os.path.join(dataset.model_path, "spiral", f"{str(dataset.start_idx + frame_idx).zfill(4)}.png")
            add_metrics_label_to_image(spiral_path, frame_metrics)

        # Compute and display average metrics
        if test_image_dataset.n_cams > 0:
            avg_metrics = {
                "Loss (Test)": round(sum([fm["Loss (Test)"] for fm in training_metrics])/len(training_metrics),5),
                "PSNR (Test)": round(sum([fm["PSNR (Test)"] for fm in training_metrics])/len(training_metrics),2),
                "Size (MB)": round(sum([fm["Size (MB)"] for fm in training_metrics])/len(training_metrics),5),
                "FPS": round(sum([fm["FPS"] for fm in training_metrics])/len(training_metrics),2),
                "Num points": round(sum([fm["Num points"] for fm in training_metrics])/len(training_metrics),2),
                "Active gates": 0 if len(training_metrics) == 1 else int(sum([fm["Active gates"] for fm in training_metrics])/(len(training_metrics)-1)),
                "Frame time": round(sum([fm["Frame time"] for fm in training_metrics])/len(training_metrics),2),
                "Frame time I/O": round(sum([fm["Frame time IO"] for fm in training_metrics])/len(training_metrics),2),
                "Elapsed time": round(frame_metrics["Training time elapsed"],2),
            }
        else:
            avg_metrics = {
                "Loss (Test)": round(sum([fm["Loss (Test)"] for fm in training_metrics])/len(training_metrics),5),
                "PSNR (Test)": round(sum([fm["PSNR (Test)"] for fm in training_metrics])/len(training_metrics),2),
                "Size (MB)": round(sum([fm["Size (MB)"] for fm in training_metrics])/len(training_metrics),5),
                "Num points": round(sum([fm["Num points"] for fm in training_metrics])/len(training_metrics),2),
                "Active gates": 0 if len(training_metrics) == 1 else int(sum([fm["Active gates"] for fm in training_metrics])/(len(training_metrics)-1)),
                "Frame time": round(sum([fm["Frame time"] for fm in training_metrics])/len(training_metrics),2),
                "Elapsed time": round(frame_metrics["Training time elapsed"],2),
            }

        # Update progress display
        del frame_metrics["Training time elapsed"]
        if enable_tqdm:
            progress_bar_frame.set_postfix(frame_metrics)
            progress_bar_frame.update(1)
        else:
            frame_counter += 1
            print(f"frame {frame_counter} frame_metrics: {frame_metrics}")

        if args.stop_after_frame1_snapshot and frame_idx == 1:
            break

        # End frame index loop
          
    with open(os.path.join(args.model_path,'training_metrics.json'),'w') as f:
        json.dump(training_metrics, f, indent=4) 

    with open(os.path.join(args.model_path, 'avg_metrics.json'),'w') as f:
        json.dump(avg_metrics, f) 

    if enable_tqdm:
        progress_bar_frame.close()

    # Display final results
    print('\nFinal average training metrics:')
    for k,v in avg_metrics.items():
        print(k+":"+ str(v))

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")


    # Create wandb logger
    if WANDB_FOUND and args.use_wandb:
        wandb_project = args.wandb_project
        wandb_run_name = args.wandb_run_name
        wandb_entity = args.wandb_entity
        wandb_mode = args.wandb_mode
        id = hashlib.md5(wandb_run_name.encode('utf-8')).hexdigest()
        name = os.path.basename(args.model_path) if wandb_run_name is None else wandb_run_name
        wandb.init(
            project=wandb_project,
            name=name,
            entity=wandb_entity,
            config=args,
            sync_tensorboard=False,
            dir=args.model_path,
            mode=wandb_mode,
            id=id,
            resume=True
        )

    return tb_writer

def training_report(tb_writer, wandb_enabled, model_args, frame_idx, iteration, Ll1, loss, l1_loss, size, 
                    elapsed, is_test, scene : Scene, renderFunc, renderArgs, prev_report=None):
    # if tb_writer:
    #     tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
    #     tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
    #     tb_writer.add_scalar('elapsed', elapsed, iteration)
    #     tb_writer.add_scalar('size', size, iteration)
        
    # Report test and samples of training set
    if is_test:
        torch.cuda.empty_cache()
        # validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
        #                       {'name': 'val', 'cameras' : scene.getTrainCameras()[1:14]})  # hack: hardcoded val views indices

        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},)  # hack: hardcoded val views indices

        report = {}
        for config in validation_configs:
            metrics = {}
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                fps_time = 0.0
                if model_args.log_images: 
                    os.makedirs(os.path.join(model_args.model_path,config['name'],"gt"),exist_ok=True)
                    os.makedirs(os.path.join(model_args.model_path,config['name'],"renders"),exist_ok=True)
                for idx, viewpoint in enumerate(config['cameras']):
                    
                    gaussian_param = {
                        'xyz': torch.empty(0, 3),         
                        'color': torch.empty(0, 3),     
                        'opacity': torch.empty(0, 1),     
                        'scaling': torch.empty(0, 3),      
                        'rot': torch.empty(0, 4),          
                        'selection_mask': torch.empty(0, dtype=torch.bool)  
                    }
                    (xyz, color, opacity, scaling, rot, selection_mask) = scene.gaussians.generate_lod_gaussians(viewpoint, None)
                    gaussian_param.update({
                        'xyz': xyz,
                        'color': color,
                        'opacity': opacity,
                        'scaling': scaling,
                        'rot': rot,
                        'selection_mask': selection_mask
                    })
                    torch.cuda.synchronize()
                    fps_start = time.time()
                    # render_results = renderFunc(viewpoint, scene.gaussians, *renderArgs, gaussian_param=gaussian_param)
                    render_results = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_results["render"], 0.0, 1.0)
                    torch.cuda.synchronize()
                    fps_end = time.time()
                    time_pre, time_render, time_post = render_results["time_pre"], render_results["time_ren"], render_results["time_post"]
                    gt_image = torch.clamp(viewpoint.original_image, 0.0, 1.0)
                    # if tb_writer and (idx < 5):
                    #     tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), 
                    #                          image[None], global_step=iteration)
                    #     if prev_report is None: # First time logging
                    #         tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), 
                    #                              gt_image[None], global_step=iteration)
                            
                    if model_args.log_images:
                        # Not logging GTs
                        os.makedirs(os.path.join(model_args.model_path,config['name'],"renders", 
                                                 camName_from_Path(viewpoint.image_path)),exist_ok=True)
                        if prev_report is None:
                            os.makedirs(os.path.join(model_args.model_path,config['name'],"gt", 
                                                     camName_from_Path(viewpoint.image_path)),exist_ok=True)
                            if os.path.exists(os.path.join(
                                                model_args.model_path,config['name'],"gt", 
                                                camName_from_Path(viewpoint.image_path),str(model_args.start_idx+frame_idx).zfill(4)+".png"
                                                )
                                            ):
                                os.remove(os.path.join(model_args.model_path,config['name'],"gt", 
                                                       camName_from_Path(viewpoint.image_path),str(model_args.start_idx+frame_idx).zfill(4)+".png"))
                            os.symlink(viewpoint.image_path,os.path.join(model_args.model_path,config['name'],"gt", 
                                                                         camName_from_Path(viewpoint.image_path),str(model_args.start_idx+frame_idx).zfill(4)+".png"))

                        save_image(image,os.path.join(model_args.model_path,config['name'],"renders", 
                                                      camName_from_Path(viewpoint.image_path),str(model_args.start_idx+frame_idx).zfill(4)+".png"))


                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    fps_time = fps_time + (fps_end - fps_start)

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                fps_test =  1/ (fps_time / len(config['cameras']))
                storage = scene.gaussians.size()/8/(10**6)  # in MB         
                # if tb_writer:
                #     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                #     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                metrics['l1'] = l1_test
                metrics['psnr'] = psnr_test
                metrics['fps'] = fps_test
                metrics['storage'] = storage
                report[config['name']] = metrics

        report['iteration'] = iteration
        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_offset.shape[0], iteration)
        torch.cuda.empty_cache()
        return report
    else:
        return None

if __name__ == "__main__":

    print('Running on ', socket.gethostname())
    # Config file is used for argument defaults. Command line arguments override config file.
    config_path = sys.argv[sys.argv.index("--config")+1] if "--config" in sys.argv else None
    if config_path:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = {}
    config = defaultdict(lambda: {}, config)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    lp = ModelParams(parser, config['model_params'])
    op_i = OptimizationParamsInitial(parser, config['opt_params_initial'])
    op_r = OptimizationParamsRest(parser, config['opt_params_rest'])
    pp = PipelineParams(parser, config['pipe_params'])
    qp = QuantizeParams(parser, config['quantize_params'])
    op_lod = parse_cfg(config)

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_format", type=str, default='ply')
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--use_xyz_legacy', action='store_true', default=False, help='If set, use legacy xyz decoding in GaussianModel (_xyz_legacy) to reproduce paper numbers. To save compressed pkl\'s, leave unset or set to False. Default: False (use _xyz_fixed).')
    parser.add_argument('--adaptive_gate_partition', action='store_true',
                        help='Run legacy pre-split V7 adaptive Gate mode for reproducibility.')
    parser.add_argument('--adaptive_gate_partition_v7b', action='store_true',
                        help='Use staged root-Gate-first V7b partitioning on dynamic frames.')
    parser.add_argument('--adaptive_gate_sqres_floor', type=float, default=0.0,
                        help='V7b-only minimum previous-offset Gate value for unconfigured f_dc/sc/rot/op sq_res decoders (0 keeps V7b behavior).')
    parser.add_argument('--adaptive_gate_sqres_cue_floor', type=float, default=0.0,
                        help='V7d-only floor for old-OFF f_dc/sc/rot/op sq_res inputs at points positive under the current gradient-or-image cue (0 keeps prior V7 behavior).')
    parser.add_argument('--save_adaptive_gate_debug_payload', action='store_true',
                        help='Also save the verbose V7b topology dump under adaptive_gate_debug/.')
    parser.add_argument('--adaptive_gate_max_levels', type=int, default=4)
    parser.add_argument('--adaptive_gate_min_points', type=int, default=2)
    parser.add_argument('--adaptive_gate_residual_ratio', type=float, default=0.8,
                        help='Split only if static-child mean offset residual is at most this fraction of dynamic-child residual.')
    parser.add_argument('--adaptive_gate_image_change_threshold', type=float,
                        default=0.005)
    parser.add_argument('--adaptive_gate_offset_bits', type=int, default=96,
                        help='Estimated offset bits avoided per OFF Gaussian for the split rate guard.')
    parser.add_argument('--save_frame1_snapshot', type=str, default=None,
                        help='Optionally save a paired-run frame-one snapshot to this path.')
    parser.add_argument('--stop_after_frame1_snapshot', action='store_true',
                        help='After saving the requested frame-one snapshot, stop the run.')
    parser.add_argument('--resume_frame1_snapshot', type=str, default=None,
                        help='Restore a frame-one snapshot and start directly at frame two.')
    parser.add_argument('--adaptive_hierarchy', action='store_true')
    parser.add_argument('--hierarchy_levels', type=int, default=4)
    parser.add_argument('--hierarchy_root_scale', type=float, default=2.0)
    parser.add_argument('--hierarchy_stage1_steps', type=int, default=100)
    parser.add_argument('--hierarchy_gs_steps', type=int, default=200)
    parser.add_argument('--hierarchy_growth_steps', type=int, default=100)
    parser.add_argument('--hierarchy_trial_steps', type=int, default=7)
    parser.add_argument('--hierarchy_motion_lr', type=float, default=5e-4)
    parser.add_argument('--hierarchy_lambda_deform', type=float, default=1.0)
    parser.add_argument('--hierarchy_image_change_threshold', type=float,
                        default=0.005)
    parser.add_argument('--hierarchy_off_ratio', type=float, default=0.0)
    parser.add_argument('--hierarchy_update_ratio', type=float, default=0.60)
    parser.add_argument('--hierarchy_split_residual_ratio', type=float, default=0.008)
    parser.add_argument('--hierarchy_split_heterogeneity', type=float, default=0.35)
    parser.add_argument('--hierarchy_split_min_gain', type=float, default=0.0003)
    parser.add_argument('--hierarchy_split_relative_gain', type=float, default=0.02)
    parser.add_argument('--hierarchy_growth_candidates', type=int, default=32)
    parser.add_argument('--hierarchy_gate_warmup_steps', type=int, default=100)
    parser.add_argument('--hierarchy_offset_gate_lr', type=float, default=0.005)
    parser.add_argument('--hierarchy_offset_gate_lambda_l0', type=float,
                        default=0.0005)
    parser.add_argument('--hierarchy_offset_gate_lambda_l2', type=float,
                        default=0.001)
    args = parser.parse_args(sys.argv[1:])
    if args.stop_after_frame1_snapshot and not args.save_frame1_snapshot:
        parser.error('--stop_after_frame1_snapshot requires --save_frame1_snapshot')
    if not 1 <= args.adaptive_gate_max_levels <= 4:
        parser.error('--adaptive_gate_max_levels must be between one and four')
    if args.adaptive_gate_min_points < 2:
        parser.error('--adaptive_gate_min_points must be at least two')
    if not 0.0 <= args.adaptive_gate_residual_ratio < 1.0:
        parser.error('--adaptive_gate_residual_ratio must be in [0, 1)')
    if (args.save_adaptive_gate_debug_payload
            and not args.adaptive_gate_partition_v7b):
        parser.error(
            '--save_adaptive_gate_debug_payload requires '
            '--adaptive_gate_partition_v7b')
    if not 0.0 <= args.adaptive_gate_sqres_floor <= 1.0:
        parser.error('--adaptive_gate_sqres_floor must be in [0, 1]')
    if (args.adaptive_gate_sqres_floor > 0.0
            and not args.adaptive_gate_partition_v7b):
        parser.error(
            '--adaptive_gate_sqres_floor requires '
            '--adaptive_gate_partition_v7b')
    if not 0.0 <= args.adaptive_gate_sqres_cue_floor <= 1.0:
        parser.error('--adaptive_gate_sqres_cue_floor must be in [0, 1]')
    if (args.adaptive_gate_sqres_cue_floor > 0.0
            and not args.adaptive_gate_partition_v7b):
        parser.error(
            '--adaptive_gate_sqres_cue_floor requires '
            '--adaptive_gate_partition_v7b')
    if (args.adaptive_gate_sqres_floor > 0.0
            and args.adaptive_gate_sqres_cue_floor > 0.0):
        parser.error(
            '--adaptive_gate_sqres_floor and '
            '--adaptive_gate_sqres_cue_floor are mutually exclusive')
    if args.adaptive_gate_image_change_threshold < 0:
        parser.error('--adaptive_gate_image_change_threshold cannot be negative')
    if args.adaptive_gate_offset_bits < 0:
        parser.error('--adaptive_gate_offset_bits cannot be negative')
    if args.hierarchy_levels < 1:
        parser.error('--hierarchy_levels must be positive')
    if args.hierarchy_stage1_steps < args.hierarchy_levels:
        parser.error('--hierarchy_stage1_steps must cover every hierarchy level')
    if not 0 < args.hierarchy_trial_steps < args.hierarchy_stage1_steps // args.hierarchy_levels:
        parser.error('--hierarchy_trial_steps must fit inside each hierarchy level')
    if args.hierarchy_gs_steps < 1 or args.hierarchy_growth_steps < 2:
        parser.error('GS and growth stages must have positive step counts')
    if not 0 <= args.hierarchy_gate_warmup_steps < args.hierarchy_gs_steps:
        parser.error('--hierarchy_gate_warmup_steps must fit inside the GS stage')
    if args.hierarchy_offset_gate_lr <= 0:
        parser.error('--hierarchy_offset_gate_lr must be positive')
    if (args.hierarchy_offset_gate_lambda_l0 < 0
            or args.hierarchy_offset_gate_lambda_l2 < 0):
        parser.error('offset gate regularization weights cannot be negative')
    # args.save_iterations.append(args.iterations)
    
    # Merge optimization args for initial and rest and change accordingly
    op = OptimizationParams(op_i.extract(args), op_r.extract(args))

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    lp_args = lp.extract(args)
    pp_args = pp.extract(args)
    qp_args = qp.extract(args)

    # Check for incompatible options
    if args.use_xyz_legacy and getattr(lp_args, 'log_compressed', False):
        print('Error: must use xyz_fixed with log_compressed (do not use --use-xyz-legacy with --log_compressed)')
        sys.exit(1)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp_args, op, pp_args, qp_args, op_lod, args.test_iterations, args.save_iterations, 
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
