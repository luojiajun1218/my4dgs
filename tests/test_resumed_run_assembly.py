"""Synthetic provenance/selection checks; no datasets or actual checkpoints."""
import json
from pathlib import Path
import sys
import tempfile

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from assemble_resumed_rd_run import assemble


def test_resumed_run_assembly():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        prefix, suffix = root / 'prefix' / 'run', root / 'suffix' / 'run'
        for run, frames in ((prefix, (1, 2, 3)), (suffix, (2, 3))):
            (run / 'frame_storage').mkdir(parents=True)
            config = dict(experiment_name=run.parent.name, ckpt_path=str(root / 'shared0'),
                          data=dict(extra_dataset_kwargs=dict(num_frames=4)),
                          trainer=dict(num_incr_steps=2))
            if run == suffix:
                config['ckpt_path'] = str(prefix / 'ckpt-2-2.pth')
            (run / 'config.yaml').write_text(yaml.safe_dump(config))
            manifest = dict(seed=0, source_sha256={'module': 'fixed'},
                            frame_seed='frame', sampler_seed='sampler')
            (run / 'controlled_manifest.json').write_text(json.dumps(manifest))
            for name in ('source_snapshot.zip', 'source_snapshot_manifest.json'):
                (run / name).write_text('synthetic')
            for name, count in (('training_views.jsonl', 2), ('evaluation.jsonl', 21),
                                ('frame_times.jsonl', 1)):
                rows = [dict(frame=f, step=i + 1, source=run.parent.name)
                        for f in frames for i in range(count)]
                (run / name).write_text(''.join(json.dumps(r) + '\n' for r in rows))
            for frame in frames:
                (run / f'ckpt-{frame}-2.pth').write_text('synthetic checkpoint')
                for name in (f'frame_{frame:03d}.json', f'motion_{frame:03d}.rdp',
                             f'motion_{frame:03d}.rdp.json', f'incremental_{frame:03d}.ply'):
                    (run / 'frame_storage' / name).write_text(run.parent.name)
        output = assemble(prefix, suffix, 2)
        rows = [json.loads(r) for r in (output / 'training_views.jsonl').read_text().splitlines()]
        assert [r['frame'] for r in rows] == [1, 1, 2, 2, 3, 3]
        assert [r['source'] for r in rows] == ['prefix'] * 4 + ['suffix'] * 2
        assert (output / 'ckpt-2-2.pth').resolve() == prefix / 'ckpt-2-2.pth'
        assert (output / 'ckpt-3-2.pth').resolve() == suffix / 'ckpt-3-2.pth'
        assert not json.loads((output / 'resume_lineage.json').read_text())['uninterrupted_training']
        manifest['seed'] = 1
        (suffix / 'controlled_manifest.json').write_text(json.dumps(manifest))
        try:
            assemble(prefix, suffix, 2)
        except ValueError as error:
            assert 'seed' in str(error)
        else:
            raise AssertionError('Mismatched seed accepted')


if __name__ == '__main__':
    test_resumed_run_assembly()
