"""Read-only verification of the completed experiment and its measurements."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import yaml

root = Path('/home/gamer-02/Documents/DeepLearning/my4dgs')
control = root / 'output/fs_goal_half_budget_20261019'
run = control / 'absolute_bytes_full300/20261019T023630_678e85'
measure = control / 'measurement_full300'
source = control / 'source_v1'
baseline = root / 'docs/experiment/evidence/flame_steak_half_fresh_full300_20261018'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with path.open() as file:
        return list(csv.DictReader(file))


summary = load(measure / 'summary.json')
decoded = load(measure / 'decoded_sequence.json')
rows = read_csv(measure / 'per_frame_metrics.csv')
dynamic = rows[1:]
prior = load(baseline / 'summary.json')
config = yaml.safe_load((run / 'config.yaml').read_text())
fresh_config = yaml.safe_load((root / 'config/n3dv/flame_steak_half_byte_budget_20261019.yaml').read_text())
assert summary['status'] == 'complete' and summary['full_sequence_evaluation']
assert decoded['sequence_complete'] and decoded['no_training_updates']
assert decoded['measured_frames'] == decoded['planned_frames'] == 300
assert [int(row['frame']) for row in rows] == list(range(300))
assert [row['frame'] for row in decoded['frames']] == list(range(300))
assert len(read_csv(measure / 'per_30_frames.csv')) == 10
for key in ('module', 'data', 'trainer'):
    assert config[key] == fresh_config[key], key
assert config['random_seed'] == fresh_config['random_seed']
assert all(int(row['train_steps']) == 300 for row in dynamic)
assert sum(int(row['new_gaussians']) - int(row['retired_gaussians']) for row in dynamic) == summary['final_gaussians'] - summary['initial_gaussians']
assert all(float(row['gs_rotation_blocks']) == 0 for row in dynamic)
for row in rows:
    components = sum(int(row[key]) for key in ('motion_bytes', 'opacity_bytes', 'feature_bytes', 'birth_bytes', 'manifest_bytes'))
    assert components == int(row['complete_update_bytes'])
mean_psnr = statistics.mean(float(row['psnr']) for row in rows)
mean_bytes = statistics.mean(int(row['complete_update_bytes']) for row in dynamic)
assert abs(mean_psnr - summary['psnr']) < 1e-12
assert abs(mean_bytes / 1000 - summary['dynamic_complete_kb']) < 1e-12
assert sum(int(row['complete_update_bytes']) for row in dynamic) == decoded['storage']['complete_update_bytes_total']
assert summary['total_stream_bytes'] == summary['shared_reference_bytes'] + sum(int(row['complete_update_bytes']) for row in dynamic)
assert all(value == 0 for value in summary['max_saved_quality_differences'].values())
assert summary['max_cache_pixel_difference'] == 0
assert summary['gpu_exclusivity_verified'] and all(item['status']['verified'] for item in decoded['gpu_checks'])
assert decoded['timing_protocol']['counted_operation'] == 'Module.forward(cam00)[render].clamp(0,1)'
assert decoded['timing_protocol']['warmups_per_frame'] == 10
assert decoded['timing_protocol']['timed_full_renders_per_frame'] == 50
fps = sum(item['timed_renders'] for item in decoded['frames']) / sum(item['render_seconds'] for item in decoded['frames'])
assert abs(fps - summary['aggregate_materialized_fps']) < 1e-10

manifest = load(source / 'source_manifest.json')
assert all(sha(source / path) == digest for path, digest in manifest['hashes'].items())
training_files = {path: digest for path, digest in manifest['hashes'].items()
                  if path.startswith(('pipeline/', 'extensions/')) or path == 'main.py'}
assert all(sha(root / path) == digest for path, digest in training_files.items())
prefix = load(run / 'prefix_provenance.json')
assert prefix['resume_frame'] == 29 and prefix['previous_frames'] == 30
assert all(sha(run / item['relative_path']) == item['sha256'] for item in prefix['files'])
checkpoint = Path(fresh_config['ckpt_path'])
checkpoint_sha = sha(checkpoint)
assert checkpoint_sha == '61f38b3ff4ffc40655b1bad75834ce92c1540537962f0a65438b5a2646b4384e'
initial_config = yaml.safe_load((control / 'absolute_bytes_v1/20261019T015438_7f571f/config.yaml').read_text())
assert initial_config['ckpt_path'] == str(checkpoint)
resume_checkpoint = Path(config['ckpt_path'])
assert resume_checkpoint == Path(prefix['source_run']) / 'rigid-ckpt-29-300.pth'
assert sha(run / 'frame_storage/rigid_tree/reference_frame0.npz') == sha(Path(prior['run']) / 'frame_storage/rigid_tree/reference_frame0.npz')

old_config = yaml.safe_load((Path(prior['run']) / 'config.yaml').read_text())
rate_changes = {}
checkpoint_schedule_changes = {}
for key in ('module', 'data', 'trainer'):
    before = json.loads(json.dumps(old_config[key]))
    after = json.loads(json.dumps(config[key]))
    if key == 'module':
        for field in ('rate_loss_weight', 'rate_normalization', 'rate_budget_bytes'):
            rate_changes[field] = {'baseline': before['rigid_tree'].pop(field, None), 'candidate': after['rigid_tree'].pop(field, None)}
    if key == 'trainer':
        field = 'saving_ckpt_frames'
        checkpoint_schedule_changes[field] = {'baseline': before.pop(field), 'candidate': after.pop(field)}
        assert set(checkpoint_schedule_changes[field]['candidate']) - set(checkpoint_schedule_changes[field]['baseline']) == {3}
    assert before == after, key

byte_stats = [int(row['complete_update_bytes']) for row in dynamic]
components = {key: {'mean_kb': value, 'percent': value / summary['dynamic_complete_kb'] * 100}
              for key, value in summary['storage_components_mean_kb'].items()}
worst = min(rows, key=lambda row: float(row['psnr']))
best = max(rows, key=lambda row: float(row['psnr']))
result = dict(
    status='verified', frames=300, dynamic_frames=299,
    run=str(run), independent_measurement=str(measure / 'decoded_sequence.json'),
    training_config=str(run / 'config.yaml'), fresh_reproduction_config=str(root / 'config/n3dv/flame_steak_half_byte_budget_20261019.yaml'),
    reproduction_module_data_trainer_match=True,
    initial_checkpoint=str(checkpoint), initial_checkpoint_sha256=checkpoint_sha,
    continuation_checkpoint=str(resume_checkpoint), continuation_checkpoint_sha256=sha(resume_checkpoint),
    initial_reference_identical_to_half_baseline=True,
    frozen_source_files_verified=len(manifest['hashes']), current_training_source_files_match=len(training_files),
    prefix_files_unchanged=len(prefix['files']),
    rate_configuration_changes=rate_changes,
    checkpoint_schedule_changes=checkpoint_schedule_changes,
    baseline_byte_price=0.002 / (24 * summary['initial_gaussians']),
    candidate_byte_price=3e-5 / 50000,
    byte_price_increase_percent=((3e-5 / 50000) / (0.002 / (24 * summary['initial_gaussians'])) - 1) * 100,
    comparison={key: {'baseline': prior[key], 'candidate': summary[key], 'delta': summary[key] - prior[key]}
                for key in ('psnr', 'ssim', 'lpips', 'dynamic_complete_kb', 'aggregate_materialized_fps', 'mean_train_wall_seconds')},
    storage_reduction_percent=(1 - summary['dynamic_complete_kb'] / prior['dynamic_complete_kb']) * 100,
    complete_update_kb=dict(mean=mean_bytes / 1000, median=statistics.median(byte_stats) / 1000,
                           p95=float(np.percentile(byte_stats, 95)) / 1000, maximum=max(byte_stats) / 1000,
                           frames_at_most_50kb=sum(value <= 50000 for value in byte_stats)),
    total_stream_mb=summary['total_stream_bytes'] / 1e6,
    reference_amortized_kb_per_frame=summary['total_stream_bytes'] / 300 / 1000,
    storage_components=components,
    worst_quality_frame=dict(frame=int(worst['frame']), psnr=float(worst['psnr'])),
    best_quality_frame=dict(frame=int(best['frame']), psnr=float(best['psnr'])),
    frames_with_effective_subdivisions=summary['frames_with_effective_subdivisions'],
    old_gs_without_local_residual_fraction=summary['old_gs_without_local_residual_fraction'],
    rotation_residual_blocks_zero=True,
    numerical_and_packet_verification_passed=True,
    goals=dict(psnr_at_least_34_2=summary['psnr'] >= 34.2,
               dynamic_complete_strictly_below_50kb=mean_bytes < 50000,
               over_50kb_percent=(mean_bytes / 50000 - 1) * 100,
               pure_render_fps_at_least_200=fps >= 200),
    measurement_provenance=load(measure / 'measurement_provenance.json'),
    limitations='Around 50 KB is achieved, strict <=50 KB is not. FPS excludes CPU decode/verify, upload, image I/O and quality metrics. Dynamic LPIPS uses saved final decoded-state evaluations. GPU power restoration remains separately blocked by OS authentication.',
)
(measure / 'full300_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
