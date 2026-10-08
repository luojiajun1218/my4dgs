"""Read complete common/contrast runs and compare their verified protocol.

No renderer, training state or historical output is modified. Reports require
all requested dynamic frames, a final checkpoint and a full sequential decode
audit. The output directory must be new.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import struct

import yaml


HEADER = struct.Struct('<4sBBH6If32s32s32s')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect_run(directory, count):
    directory = directory.resolve()
    config = yaml.safe_load((directory / 'config.yaml').read_text())
    parameters = config['module']['common_contrast_binary']
    require(config['module']['motion_mode'] == 'common_contrast_binary',
            'Only common_contrast_binary runs have a comparable codec here')
    require(config['data']['extra_dataset_kwargs']['num_frames'] == count + 1,
            'Requested dynamic frame count differs from the saved configuration')
    expected = list(range(1, count + 1))
    summary_dir = directory / 'frame_summary'
    available = sorted(int(path.stem.rsplit('_', 1)[1])
                       for path in summary_dir.glob('quantized_frame_*.json'))
    require(available == expected,
            f'Incomplete or unexpected frames in {directory}: {len(available)}/{count}')
    steps = config['trainer']['num_incr_steps']
    final_checkpoint = directory / f'ckpt-{count}-{steps}.pth'
    require(final_checkpoint.is_file(), 'Final checkpoint is missing')
    audit_path = directory / f'sequential_decode_audit_frame001_{count:03d}.json'
    require(audit_path.is_file(), 'Full sequential decoder audit is missing')
    audit = read_json(audit_path)
    require([row['frame'] for row in audit['frames']] == expected,
            'Decoder audit does not cover every requested frame in order')
    require(bool(audit.get('final_state_sha256')), 'Decoder audit lacks a final state hash')
    require(audit.get('final_state_match') is True
            and audit['final_state_sha256'] == audit.get('final_checkpoint_state_sha256')
            and set(audit.get('final_fields_exact', {})) ==
            {'xyz', 'sh_dc', 'sh_rest', 'logscale', 'rotation', 'opacity'}
            and all(value is True for value in audit['final_fields_exact'].values()),
            'Decoder final state was not verified against every checkpoint field')

    manifest_path = directory / 'source_snapshot/manifest.json'
    manifest = read_json(manifest_path)
    source_hashes = {}
    for item in manifest:
        snapshot = directory / 'source_snapshot' / item['snapshot']
        require(digest(snapshot) == item['sha256'], f'Source snapshot changed: {snapshot}')
        source_hashes[Path(item['source']).name] = item['sha256']

    rows, camera_sequences = [], {}
    input_hashes = {str(path): digest(path) for path in
                    (directory / 'config.yaml', audit_path, manifest_path, final_checkpoint)}
    previous = None
    for frame, decoded in zip(expected, audit['frames']):
        detail_path = summary_dir / f'quantized_frame_{frame:03d}.json'
        frame_path = summary_dir / f'frame_{frame:03d}.json'
        eval_path = summary_dir / f'evaluation_{frame:03d}_step{steps}.json'
        views_path = summary_dir / f'training_views_{frame:03d}.json'
        detail, report = read_json(detail_path), read_json(frame_path)
        evaluation, views = read_json(eval_path), read_json(views_path)
        require(detail['frame'] == report['frame'] == evaluation['frame'] == frame,
                f'Frame identity mismatch at {frame}')
        require(evaluation['step'] == steps, f'Wrong evaluation budget at frame {frame}')
        require(len(views['views']) == steps and 'cam00.png' not in views['views'],
                f'Wrong training cameras or budget at frame {frame}')
        require(set(views['views']) == {f'cam{i:02d}.png' for i in range(1, 21)},
                f'Training camera set changed at frame {frame}')
        require(len(evaluation['views']) == 1
                and evaluation['views'][0]['camera'] == 'cam00.png'
                and evaluation['views'][0]['split'] == 'test',
                f'Unexpected final evaluation cameras at frame {frame}')
        camera_sequences[frame] = views['views']
        motion_path = directory / 'frame_storage/common_contrast_binary' / f'motion_{frame:03d}.bin'
        packet = motion_path.read_bytes()
        require(len(packet) >= HEADER.size, f'Truncated header at frame {frame}')
        magic, version, flags, header_size, identity, prev_count, base_count, deleted, births, bits, _, _, _, topology = HEADER.unpack(packet[:HEADER.size])
        require((magic, version, flags, header_size, identity) ==
                (b'CCRB', 1, 0, HEADER.size, frame), f'Invalid codec header at frame {frame}')
        require(len(packet) == header_size + (bits + 7) // 8,
                f'Payload length mismatch at frame {frame}')
        require(base_count == prev_count - deleted, f'Deletion count mismatch at frame {frame}')
        if previous is not None:
            require(deleted == previous['births'] and
                    prev_count == previous['base'] + previous['births'],
                    f'Sequential count mismatch at frame {frame}')
        previous = dict(base=base_count, births=births)
        require(decoded['coded_payload_bits'] == bits
                and decoded['birth_count'] == births
                and decoded['topology_digest'] == topology.hex()
                and len(decoded['deleted_rows']) == deleted,
                f'Decoder audit disagrees with the actual stream at frame {frame}')
        birth_export = detail.get('incremental_export')
        birth_path = directory / 'frame_storage/common_contrast_binary' / f'incremental_{frame:03d}.ply'
        birth_bytes = birth_path.stat().st_size if birth_export else 0
        require((birth_export['rows'] if birth_export else 0) == births,
                f'Birth export count mismatch at frame {frame}')
        actual_bytes = len(packet) + birth_bytes
        require(actual_bytes == detail['inference_file_bytes'] == detail['hard_storage_bytes'],
                f'Actual files disagree with hard accounting at frame {frame}')
        require(detail['motion_export']['archive_bytes'] == len(packet),
                f'Motion file size mismatch at frame {frame}')
        require(detail['tree_setup']['tree_rebuilt'] == decoded['tree_rebuilt'] ==
                ((frame - 1) % parameters['tree_rebuild_interval'] == 0),
                f'Tree lifecycle mismatch at frame {frame}')
        quality = evaluation['evaluation']['test']
        require(report['quantized_evaluation']['test'] == quality,
                f'Quality summaries disagree at frame {frame}')
        require(all(math.isfinite(quality[key]) for key in ('psnr', 'l1', 'ssim')),
                f'Nonfinite quality at frame {frame}')
        distortion = ((1 - config['module']['lambda_dssim']) * quality['l1'] +
                      config['module']['lambda_dssim'] * (1 - quality['ssim']))
        reference_bytes = detail['reference_storage_bytes']
        require(reference_bytes > 0, f'Invalid reference cost at frame {frame}')
        levels = detail['activities']['levels']
        roots = levels[0]['reachable_nodes']
        opened = sum(level.get('refine_open', 0) for level in levels)
        zero_open = opened - sum(level.get('nonzero_contrast_records', 0) for level in levels)
        row = dict(frame=frame, psnr=quality['psnr'], l1=quality['l1'], ssim=quality['ssim'],
                   test_image_objective=distortion,
                   test_rd_objective=distortion + parameters['storage_loss_weight'] *
                   actual_bytes / reference_bytes, reference_bytes=reference_bytes,
                   motion_bytes=len(packet), birth_bytes=birth_bytes, total_bytes=actual_bytes,
                   training_seconds=report['timing']['training'], wall_seconds=report['timing']['wall'],
                   root_groups=roots, open_splits=opened, terminal_groups=roots + opened,
                   open_splits_without_local_payload=zero_open, rebuilt=decoded['tree_rebuilt'])
        for level in levels[:-1]:
            row[f"level{level['level']}_open"] = level.get('refine_open', 0)
            row[f"level{level['level']}_reached_gates"] = level.get('reachable_refine_gates', 0)
        rows.append(row)
        for path in (detail_path, frame_path, eval_path, views_path, motion_path):
            input_hashes[str(path)] = digest(path)
        if birth_export:
            input_hashes[str(birth_path)] = digest(birth_path)
    result = dict(run=str(directory), dynamic_frames=count,
                  psnr_mean=statistics.mean(row['psnr'] for row in rows),
                  l1_mean=statistics.mean(row['l1'] for row in rows),
                  ssim_mean=statistics.mean(row['ssim'] for row in rows),
                  test_rd_objective_mean=statistics.mean(row['test_rd_objective'] for row in rows),
                  total_bytes_mean=statistics.mean(row['total_bytes'] for row in rows),
                  total_bytes_sum=sum(row['total_bytes'] for row in rows),
                  motion_bytes_mean=statistics.mean(row['motion_bytes'] for row in rows),
                  birth_bytes_mean=statistics.mean(row['birth_bytes'] for row in rows),
                  training_seconds_median=statistics.median(row['training_seconds'] for row in rows),
                  wall_seconds_median=statistics.median(row['wall_seconds'] for row in rows),
                  decoder_device=audit['device'], final_state_sha256=audit['final_state_sha256'],
                  final_checkpoint_match=True,
                  final_checkpoint=str(final_checkpoint), source_hashes=source_hashes,
                  limits=['One scene and one training seed; no significance estimate.',
                          'Bytes exclude the shared first frame, checkpoints and diagnostic files.',
                          'Test RD uses the predefined training lambda; cam00 is final evaluation only.',
                          'Decoder verification is limited to the recorded device and numerical path.'])
    return result, rows, config, camera_sequences, input_hashes


def compare_protocol(left, right):
    a, b = left[2], right[2]
    for key in ('ckpt_path', 'random_seed', 'data', 'trainer'):
        require(a[key] == b[key], f'Comparison protocol differs at {key}')
    a_module, b_module = dict(a['module']), dict(b['module'])
    a_parameters = dict(a_module.pop('common_contrast_binary'))
    b_parameters = dict(b_module.pop('common_contrast_binary'))
    require(a_parameters.pop('force_all_gates_open') is False
            and b_parameters.pop('force_all_gates_open') is True,
            'Expected learned-gate run followed by an all-open control')
    require(a_parameters == b_parameters and a_module == b_module,
            'Model settings differ beyond the gate intervention')
    require(left[3] == right[3], 'Actual per-frame camera sequences differ')
    require([row['reference_bytes'] for row in left[1]] ==
            [row['reference_bytes'] for row in right[1]],
            'Storage normalization differs between arms')
    for filename in left[0]['source_hashes'].keys() & right[0]['source_hashes'].keys():
        if filename.endswith('.py'):
            require(left[0]['source_hashes'][filename] == right[0]['source_hashes'][filename],
                    f'Implementation changed between comparison arms: {filename}')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_pair(directory, learned, baseline, bits):
    """Plot per-frame measurements, not a rate-distortion frontier."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True,
                                constrained_layout=True)
    frames = [row['frame'] for row in learned]
    metrics = [('psnr', 'Test PSNR (dB)', 1),
               ('total_bytes', 'Motion + birth files (KB)', 1000),
               ('terminal_groups', 'Terminal motion groups', 1)]
    for axis, (key, label, scale) in zip(axes, metrics):
        for rows, color, legend in [(learned, '#167d8d', 'Learned subtree gates'),
                                    (baseline, '#ce7128', 'All splits open')]:
            axis.plot(frames, [row[key] / scale for row in rows],
                      color=color, linewidth=1.3, label=legend)
        axis.set_ylabel(label)
        axis.grid(axis='y', alpha=.22, linewidth=.5)
        axis.spines[['top', 'right']].set_visible(False)
        if key != 'psnr':
            axis.set_ylim(bottom=0)
    axes[0].legend(frameon=False, loc='best')
    axes[-1].set_xlabel('Dynamic frame')
    axes[-1].set_xlim(frames[0], frames[-1])
    figure.suptitle(f'Flame steak: matched training budget, fixed {bits}-bit motion', fontsize=12)
    figure.savefig(directory / 'paired_results.png', dpi=180)
    figure.savefig(directory / 'paired_results.pdf')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--dynamic-frames', type=int, default=99)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true', help='Export matched per-frame PNG/PDF figures')
    args = parser.parse_args()
    require(args.dynamic_frames > 0, 'Frame count must be positive')
    require(not args.output.exists(), 'Analysis output must be a new directory')
    require(not args.plot or args.baseline is not None, 'Paired plots require a verified baseline')
    left = inspect_run(args.run, args.dynamic_frames)
    results = dict(learned=left[0])
    paired = None
    right = None
    if args.baseline:
        right = inspect_run(args.baseline, args.dynamic_frames)
        compare_protocol(left, right)
        results['all_open'] = right[0]
        paired = [dict(frame=a['frame'], psnr_gain=a['psnr'] - b['psnr'],
                       l1_difference=a['l1'] - b['l1'],
                       test_rd_gain=b['test_rd_objective'] - a['test_rd_objective'],
                       bytes_difference=a['total_bytes'] - b['total_bytes'],
                       terminal_groups_difference=a['terminal_groups'] - b['terminal_groups'])
                  for a, b in zip(left[1], right[1])]
        results['paired'] = dict(psnr_gain_mean=statistics.mean(row['psnr_gain'] for row in paired),
            test_rd_gain_mean=statistics.mean(row['test_rd_gain'] for row in paired),
            l1_difference_mean=statistics.mean(row['l1_difference'] for row in paired),
            bytes_difference_mean=statistics.mean(row['bytes_difference'] for row in paired),
            byte_ratio_learned_over_all_open=left[0]['total_bytes_sum'] / right[0]['total_bytes_sum'],
            protocol_verified=True)
    args.output.mkdir(parents=True)
    (args.output / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    write_csv(args.output / 'learned_frames.csv', left[1])
    if right:
        write_csv(args.output / 'all_open_frames.csv', right[1])
        write_csv(args.output / 'paired_frames.csv', paired)
    hashes = dict(left[4])
    if right:
        hashes.update(right[4])
    hashes[str(Path(__file__).resolve())] = digest(Path(__file__))
    analysis_copy = args.output / 'analysis_script.py'
    shutil.copy2(Path(__file__), analysis_copy)
    hashes[str(analysis_copy.resolve())] = digest(analysis_copy)
    (args.output / 'source_sha256.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    if args.plot:
        plot_pair(args.output, left[1], right[1],
                  left[2]['module']['common_contrast_binary']['fixed_motion_bits'])
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
