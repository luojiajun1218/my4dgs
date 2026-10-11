"""Read-only first-frame opacity-pruning diagnosis for the rigid-tree reference.

Every threshold is fixed before evaluating the held-out cam00 view. Report
quality changes over *all* training views; cam00 is a separate diagnostic.
FPS uses the full active Module.forward plus image clamp at 1600x1199.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_rigid_residual_tree import (  # noqa: E402
    RAW_FIELDS, _cam00, _install_raw, _load_config, _load_reference,
    _make_module, _time_materialized, check_gpu_exclusivity,
)


THRESHOLDS = (0.0075, 0.01, 0.015, 0.02)


def _validate_image(view, name):
    image = view.image
    if (tuple(image.shape) != (3, 1199, 1600) or
            not bool(torch.isfinite(image).all()) or
            bool((image < -1e-6).any()) or
            bool((image > 1 + 1e-6).any())):
        raise ValueError(f'{name}: expected finite 1600x1199 RGB in [0,1]')


def _render_psnr(module, view):
    from libgs.metric import psnr

    image = module.forward(view)['render'].clamp(0., 1.)
    if tuple(image.shape) != tuple(view.image.shape):
        raise ValueError(f'Render shape differs for {view.path.name}')
    return float(psnr(image, view.image))


def _quality(module, train_views, heldout):
    with torch.no_grad():
        train = {view.path.name: _render_psnr(module, view)
                 for view in train_views}
        cam00 = _render_psnr(module, heldout)
    return train, cam00


def _summary(full_train, candidate_train, full_cam00, candidate_cam00):
    deltas = {name: candidate_train[name] - full_train[name]
              for name in full_train}
    values = list(deltas.values())
    worst = min(deltas, key=deltas.get)
    return dict(
        training_psnr_mean=sum(candidate_train.values()) / len(candidate_train),
        training_psnr_delta_mean=sum(values) / len(values),
        training_psnr_delta_worst=deltas[worst],
        training_worst_camera=worst,
        training_psnr_by_camera=candidate_train,
        training_psnr_delta_by_camera=deltas,
        heldout_cam00_psnr=candidate_cam00,
        heldout_cam00_psnr_delta=candidate_cam00 - full_cam00,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True,
                        help='Existing rigid-tree run with a frame0 reference')
    parser.add_argument('--reference', type=Path,
                        help='Defaults to run/frame_storage/rigid_tree/reference_frame0.npz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fps-warmups', type=int, default=10)
    parser.add_argument('--fps-renders', type=int, default=100)
    parser.add_argument('--fps-rounds', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.fps_warmups, args.fps_renders, args.fps_rounds) < 1:
        parser.error('FPS counts must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for full-render FPS')

    gpu_check = check_gpu_exclusivity()
    run = args.run.resolve()
    config = _load_config(run)
    if (config.data.resolution != -1 or
            config.data.expected_image_size != [1600, 1199]):
        raise ValueError('Diagnosis requires the 1600x1199 full-resolution run')
    reference = (args.reference or
        run / 'frame_storage' / 'rigid_tree' / 'reference_frame0.npz').resolve()
    state, raw = _load_reference(reference, config.module.rigid_tree.region_count)
    if state.frame_index != 0:
        raise ValueError('This diagnosis must use the untouched first-frame reference')

    device = torch.device('cuda')
    module, data = _make_module(config, device)
    data.current_frame = 0
    dataset = data.get_train_dataset()
    if dataset is None or len(dataset) < 1:
        raise ValueError('No frame0 training views')
    train_views = [dataset[index].to(device) for index in range(len(dataset))]
    train_views.sort(key=lambda view: view.path.name)
    names = [view.path.name for view in train_views]
    if len(names) != len(set(names)):
        raise ValueError('Training camera names are not unique')
    heldout, _ = _cam00(data, 0, device)
    if heldout.path.name in names:
        raise ValueError('cam00 must be held out from training views')
    for view in train_views:
        _validate_image(view, view.path.name)
    _validate_image(heldout, 'cam00')

    opacity = torch.sigmoid(raw['opacity'].flatten())
    count = len(opacity)
    reference_report = dict(
        name='unfiltered', threshold=None, gaussian_count=count,
        removed_count=0, removed_fraction=0.)
    _install_raw(module, raw, device)
    full_train, full_cam00 = _quality(module, train_views, heldout)
    reference_report['quality'] = _summary(
        full_train, full_train, full_cam00, full_cam00)
    reference_report['fps'] = _time_materialized(
        module, heldout, args.fps_warmups, args.fps_renders,
        args.fps_rounds)
    print(f"reference: {count} GS, training mean "
          f"{reference_report['quality']['training_psnr_mean']:.4f} dB, "
          f"cam00 FPS {reference_report['fps']['fps']:.2f}", flush=True)

    variants = []
    for threshold in THRESHOLDS:
        keep = opacity >= threshold
        filtered = {name: raw[name][keep] for name in RAW_FIELDS}
        report = dict(name=f'opacity_ge_{threshold:g}', threshold=threshold,
            gaussian_count=int(keep.sum()), removed_count=int((~keep).sum()),
            removed_fraction=float((~keep).float().mean()))
        _install_raw(module, filtered, device)
        train, cam00 = _quality(module, train_views, heldout)
        report['quality'] = _summary(full_train, train, full_cam00, cam00)
        report['fps'] = _time_materialized(
            module, heldout, args.fps_warmups, args.fps_renders,
            args.fps_rounds)
        variants.append(report)
        print(f"threshold={threshold:g}: removed={report['removed_count']}, "
              f"train mean delta={report['quality']['training_psnr_delta_mean']:+.4f} dB, "
              f"worst={report['quality']['training_psnr_delta_worst']:+.4f} dB, "
              f"cam00 FPS={report['fps']['fps']:.2f}", flush=True)

    result = dict(schema='rigid_tree_first_reference_pruning_diagnosis_v1',
        run=str(run), reference=str(reference), frame=0,
        thresholds_fixed_before_heldout_evaluation=list(THRESHOLDS),
        training_cameras=names, heldout_camera=heldout.path.name,
        image_shape=list(heldout.image.shape),
        quality_metric='PSNR of clamp(Module.forward(view)[render],0,1) versus GT',
        fps_scope='CUDA-synchronized full Module.forward plus clamp on held-out cam00; '
                  'preloaded viewpoint and materialized GS; no file decode or metrics',
        gpu_check=gpu_check, reference_result=reference_report,
        variants=variants, no_training_updates=True, no_checkpoint_changes=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
