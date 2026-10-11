#!/usr/bin/env python3
"""Replay and benchmark exact ReCon-GS packet streams on the renderer."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.archive.run_recongs_native_stream import decode_topology_index, normalize_transition  # noqa: E402


def read_records(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run_dir / "stream_metrics/frames.jsonl").read_text().splitlines()
        if line.strip()
    ]


def source_cam00_metrics(record: dict[str, Any]) -> dict[str, float]:
    views = record["evaluation"]["test"]["views"]
    row = next((item for item in views if "cam00" in item["view"].lower()), None)
    if row is None:
        raise ValueError(f"frame {record['frame']}: source metrics have no cam00 view")
    return {key: float(row[key]) for key in ("l1", "psnr", "ssim")}


def capture_arrays(model) -> dict[str, np.ndarray]:
    return {
        "xyz": model._xyz.detach().cpu().numpy(),
        "features_dc": model._features_dc.detach().cpu().numpy(),
        "features_rest": model._features_rest.detach().cpu().numpy(),
        "scaling": model._scaling.detach().cpu().numpy(),
        "rotation": model._rotation.detach().cpu().numpy(),
        "opacity": model._opacity.detach().cpu().numpy(),
    }


def capture_tuple_arrays(capture) -> dict[str, np.ndarray]:
    return {
        "xyz": capture[1].detach().cpu().numpy(),
        "features_dc": capture[2].detach().cpu().numpy(),
        "features_rest": capture[3].detach().cpu().numpy(),
        "scaling": capture[4].detach().cpu().numpy(),
        "rotation": capture[5].detach().cpu().numpy(),
        "opacity": capture[6].detach().cpu().numpy(),
    }


def compare_arrays(actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> dict[str, float]:
    maxima = {}
    for name in actual:
        if actual[name].shape != expected[name].shape:
            raise ValueError(
                f"final checkpoint {name} shape differs: "
                f"{actual[name].shape} != {expected[name].shape}"
            )
        difference = np.abs(actual[name].astype(np.float64) - expected[name].astype(np.float64))
        maxima[name] = float(difference.max()) if difference.size else 0.0
        if not np.allclose(actual[name], expected[name], atol=2e-5, rtol=2e-6, equal_nan=True):
            raise ValueError(f"final checkpoint {name} differs, max abs={maxima[name]}")
    return maxima


def make_pipeline(config_path: Path, source_root: Path, checkpoint: Path, temp_dir: Path):
    import torch
    from omegaconf import OmegaConf

    sys.path.insert(0, str(source_root.resolve()))
    from libgs.pipeline.loader import load_pipeline
    from libgs.utils.config import to_structured
    import libgs.data.dataset.base as dataset_base

    # This warning's implementation assumes a DataLoader worker, but playback
    # intentionally loads one frame view synchronously in the main process.
    dataset_base.warning = lambda: None
    Config, Pipeline = load_pipeline("ReCon-GS")
    if not hasattr(Config().module, 'motion_mode'):
        raise RuntimeError('Historical playback needs its archived source snapshot')
    config, _ = to_structured(OmegaConf.load(config_path), Config)
    if hasattr(config, 'gui_enabled'):
        config.gui_enabled = False
    else:
        from libgs.renderer import network_gui
        network_gui.init = lambda *args, **kwargs: None
    config.mode = "train"
    config.ckpt_path = checkpoint
    config.output_dir = temp_dir
    config.experiment_name = "exact_stream_playback_benchmark"
    config.data.shuffle = False
    config.data.extra_dataloader_kwargs = {"num_workers": 0, "pin_memory": False}
    pipeline = Pipeline(config)
    pipeline.trainer.setup(pipeline.module, ckpt_path=checkpoint)
    return pipeline, torch, Config, Pipeline


def reset_frame0(pipeline, frame0_checkpoint, GaussianModel):
    module, trainer = pipeline.module, pipeline.trainer
    module.gaussians.restore(frame0_checkpoint["gaussians"], module.config.gaussian)
    module.gaussians_incr = GaussianModel(module.config.sh_degree)
    # Playback has no optimizer. Leaving its training Adam state live would
    # measure optimizer tensor migration during extend/prune instead of decode.
    module.gaussians.optimizer = None
    module.renderer.gaussians = module.gaussians
    pipeline.datamodule.current_frame = 0
    trainer.global_step = int(frame0_checkpoint["global_step"])


def get_cam00_view(pipeline, frame: int, device):
    pipeline.datamodule.current_frame = int(frame)
    dataset = pipeline.datamodule.get_test_dataset()
    candidates = [Path(str(item.path)).name for item in dataset.items]
    index = next((i for i, name in enumerate(candidates) if "cam00" in name.lower()), None)
    if index is None:
        raise ValueError(f"frame {frame}: test dataset has no cam00 image")
    return dataset[index].to(device), candidates[index]


def replay_one_stream(
    pipeline,
    torch,
    GaussianModel,
    MergedGaussianModel,
    F,
    metrics,
    frame0_checkpoint,
    final_checkpoint,
    reference_records,
    run_dir: Path,
    renderer_frames: set[int],
    pass_name: str,
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    torch.cuda.synchronize(device)
    reset_frame0(pipeline, frame0_checkpoint, GaussianModel)
    module, trainer = pipeline.module, pipeline.trainer
    records = read_records(run_dir)
    previous_index = None
    previous_birth = None
    frame_times = []
    decode_times = []
    quality = {}
    renderer_benchmarks = {}
    final_birth = None
    camera_prepare_times = []
    gaussian_fields = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
    quaternion_multiply = importlib.import_module("libgs.utils.general").quaternion_multiply

    for record in records:
        frame = int(record["frame"])
        pipeline.datamodule.current_frame = frame
        trainer.global_step = int(record["steps"])
        camera_started = time.perf_counter()
        viewpoint, camera_name = get_cam00_view(pipeline, frame, device)
        torch.cuda.synchronize(device)
        camera_prepare_times.append((time.perf_counter() - camera_started) * 1000.0)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        decode_started = time.perf_counter()
        packet_path = run_dir / record["packet"]["path"]
        transition_path = run_dir / record["transition_metadata"]["path"]
        birth_path = run_dir / record["birth_ply"]["path"]
        transition_metadata = json.loads(transition_path.read_text())
        with np.load(packet_path, allow_pickle=False) as packet:
            delta_np = np.asarray(packet["delta"]).copy()
            deleted_np = np.asarray(packet["deleted_source_indices"]).copy()
            transition = normalize_transition(
                int(transition_metadata["old_base_rows"]),
                int(transition_metadata["previous_birth_rows"]),
                deleted_np,
            )
            index_np = decode_topology_index(packet, previous_index, transition)
        decode_times.append(time.perf_counter() - decode_started)

        if int(module.gaussians._xyz.shape[0]) != transition["old_base_rows"]:
            raise ValueError(f"frame {frame}: live base row count differs from transition")
        if int(0 if previous_birth is None else previous_birth._xyz.shape[0]) != transition["previous_birth_rows"]:
            raise ValueError(f"frame {frame}: live birth row count differs from transition")
        source_rows = transition["source_rows"]
        keep = torch.ones(source_rows, dtype=torch.bool, device=device)
        if deleted_np.size:
            deleted = torch.as_tensor(deleted_np, dtype=torch.long, device=device)
            keep[deleted] = False
        keep_indices = torch.nonzero(keep, as_tuple=False).flatten()
        for field_name in gaussian_fields:
            old_tensor = getattr(module.gaussians, field_name).detach()
            if previous_birth is None:
                source_tensor = old_tensor
            else:
                birth_tensor = getattr(previous_birth, field_name).detach()
                source_tensor = torch.cat((old_tensor, birth_tensor), dim=0)
            updated_tensor = source_tensor.index_select(0, keep_indices).contiguous()
            setattr(
                module.gaussians,
                field_name,
                torch.nn.Parameter(updated_tensor, requires_grad=False),
            )
        if int(module.gaussians._xyz.shape[0]) != transition["survivor_rows"]:
            raise ValueError(f"frame {frame}: live survivor row count differs from transition")

        delta = torch.as_tensor(delta_np, dtype=torch.float32, device=device)
        index = torch.as_tensor(index_np, dtype=torch.long, device=device)
        field = delta[index].sum(dim=0)
        delta_rotation = F.normalize(field[:, 3:], dim=-1)
        updated_xyz = module.gaussians._xyz.detach() + field[:, :3]
        updated_rotation = quaternion_multiply(
            module.gaussians.get_rotation, delta_rotation
        )
        module.gaussians._xyz = torch.nn.Parameter(updated_xyz.contiguous(), requires_grad=False)
        module.gaussians._rotation = torch.nn.Parameter(
            updated_rotation.contiguous(), requires_grad=False
        )

        birth = GaussianModel(module.config.sh_degree)
        birth.load_ply(str(birth_path))
        final_birth = birth
        module.renderer.gaussians = MergedGaussianModel([module.gaussians, birth])
        with torch.inference_mode():
            rendered = module.forward(viewpoint, training=False)["render"].clamp(0.0, 1.0)
        torch.cuda.synchronize(device)
        frame_times.append((time.perf_counter() - started) * 1000.0)

        with torch.inference_mode():
            measured = {
                "l1": float(metrics.l1_loss(rendered, viewpoint.image).item()),
                "psnr": float(metrics.psnr(rendered, viewpoint.image).item()),
                "ssim": float(metrics.ssim(rendered, viewpoint.image).item()),
            }
        torch.cuda.synchronize(device)

        source = source_cam00_metrics(reference_records[frame - 1])
        quality[frame] = {
            "camera": camera_name,
            "playback": measured,
            "reference_training_eval": source,
            "difference": {key: measured[key] - source[key] for key in measured},
        }
        if frame in renderer_frames:
            for _ in range(5):
                module.forward(viewpoint, training=False)
            torch.cuda.synchronize(device)
            render_started = time.perf_counter()
            for _ in range(30):
                module.forward(viewpoint, training=False)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - render_started
            renderer_benchmarks[frame] = {
                "camera": camera_name,
                "warmups": 5,
                "renders": 30,
                "wall_seconds": elapsed,
                "fps": 30 / elapsed,
                "reference_training_fps": float(reference_records[frame - 1]["renderer_benchmark"]["fps"]),
            }
        previous_index = index_np.copy()
        previous_birth = birth

    base_checkpoint_diffs = compare_arrays(
        capture_arrays(module.gaussians), capture_tuple_arrays(final_checkpoint["gaussians"])
    )
    birth_checkpoint_diffs = compare_arrays(
        capture_arrays(final_birth), capture_tuple_arrays(final_checkpoint["gaussians_incr"])
    )
    values = np.asarray(frame_times, dtype=np.float64)
    decode_values = np.asarray(decode_times, dtype=np.float64)
    return {
        "run_dir": str(run_dir),
        "frames_rendered": len(records),
        "frame_total_scope": "packet+transition read/decode, optimizer-free row append/prune, anchor upload and field application, incremental PLY load/upload, and one synchronized forward+clamp; cam00 image preparation/metrics are timed and reported separately; excludes one-time pipeline/frame-0 setup and extra 5+30 renderer timing calls",
        "end_to_end_playback_fps": 1000.0 / float(values.mean()),
        "per_frame_total_ms_mean": float(values.mean()),
        "per_frame_total_ms_median": float(np.median(values)),
        "per_frame_total_ms_p95": float(np.percentile(values, 95)),
        "packet_read_decode_ms_mean": float(decode_values.mean() * 1000.0),
        "camera_ground_truth_image_prep_ms_mean_excluded_from_playback": float(np.mean(camera_prepare_times)),
        "fps_if_camera_ground_truth_prep_is_included": 1000.0 / float(values.mean() + np.mean(camera_prepare_times)),
        "renderer_only_samples": renderer_benchmarks,
        "quality_comparison_all_frames": quality,
        "quality_abs_difference_max": {
            key: max(abs(row["difference"][key]) for row in quality.values())
            for key in ("l1", "psnr", "ssim")
        },
        "final_checkpoint_state_max_abs_difference": {
            "base": base_checkpoint_diffs,
            "birth": birth_checkpoint_diffs,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_run", type=Path)
    parser.add_argument("candidate_runs", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/recongs_flame_steak_native_stream_20261002.yaml")
    parser.add_argument("--source-root", type=Path, default=Path("/tmp/recongs_native_source_snapshot_20261002"))
    parser.add_argument("--frame0-checkpoint", type=Path, default=Path("/home/gamer-02/Documents/DeepLearning/ReCon-GS/output/flame_steak_fullres/20260926T225223_dc5baf/ckpt-0-15000.pth"))
    parser.add_argument("--final-checkpoint", type=Path)
    parser.add_argument("--renderer-frames", default="1,150,299")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from torch.nn import functional as F
    from libgs.metric import psnr, ssim
    from libgs.model.merged_gaussian import MergedGaussianModel

    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    metric_module = type("Metrics", (), {
        "l1_loss": staticmethod(F.l1_loss),
        "psnr": staticmethod(psnr),
        "ssim": staticmethod(ssim),
    })
    recon_model = importlib.import_module("pipeline.ReCon-GS.archive.model.gaussian")
    GaussianModel = recon_model.GaussianModel
    frame0_checkpoint = torch.load(args.frame0_checkpoint, map_location=device, weights_only=False)
    final_checkpoint_path = args.final_checkpoint or (
        args.reference_run / "ckpt-299-200.pth"
    )
    final_checkpoint = torch.load(final_checkpoint_path, map_location="cpu", weights_only=False)
    reference_records = read_records(args.reference_run.resolve())
    renderer_frames = {int(value) for value in args.renderer_frames.split(",") if value.strip()}
    if not renderer_frames.issubset(set(range(1, len(reference_records) + 1))):
        raise ValueError("renderer frame selection is outside the reference sequence")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    setup_dir = Path("/tmp") / f"recongs_native_playback_setup_{time.strftime('%Y%m%dT%H%M%S')}"
    pipeline, torch, _, _ = make_pipeline(
        args.config.resolve(), args.source_root.resolve(), args.frame0_checkpoint.resolve(),
        setup_dir,
    )
    results: dict[str, Any] = {
        "device": torch.cuda.get_device_name(device),
        "reference_run": str(args.reference_run.resolve()),
        "candidate_order": [],
        "renderer_only_scope": "same ReCon-GS Module.forward; no clamp; cam00; 5 warmups then 30 CUDA-synchronized calls",
        "data_cache_note": "second pass may use the local OS page cache; pass order alternates between candidate streams",
        "streams": {},
    }
    runs = [path.resolve() for path in args.candidate_runs]
    ordered_runs = runs + list(reversed(runs))
    for pass_index, run_dir in enumerate(ordered_runs):
        label = run_dir.name
        results["candidate_order"].append(label)
        result = replay_one_stream(
            pipeline,
            torch,
            GaussianModel,
            MergedGaussianModel,
            F,
            metric_module,
            frame0_checkpoint,
            final_checkpoint,
            reference_records,
            run_dir,
            renderer_frames,
            f"pass{pass_index + 1}",
        )
        results["streams"].setdefault(label, []).append(result)
    serialized = json.dumps(results, indent=2, sort_keys=True) + "\n"
    args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
