"""Verify independent frame009 replay, then resume 260 and continue frame010."""
from pathlib import Path
import json
import sys
import tempfile
import warnings

import torch
from omegaconf import OmegaConf
from absl import flags, logging

ROOT = Path('/home/gamer-02/Documents/DeepLearning/my4dgs')
sys.path.insert(0, str(ROOT))
from libgs.utils.config import to_structured
from pipeline.entry import Config, Pipeline
from pipeline.model.birth_quantization import RAW_FIELDS, model_raw
from scripts.replay_rigid_tree import replay

SOURCE = Path('/tmp/my4dgs_multiview_formal9_gauvauq3/output/frame009_multiview_formal/20261024T000412_74857f')


def checkpoint_raw(checkpoint):
    return dict(zip(('xyz', 'sh_dc', 'sh_rest', 'logscale', 'rotation', 'opacity'),
                    checkpoint['gaussians'][1:7]))


def verify_stream(run, frame):
    stream = run / 'frame_storage/rigid_tree'
    reference = next(stream.glob('reference_frame*.npz'))
    state, raw = replay(stream, reference, through=frame)
    ckpt = torch.load(run / f'rigid-ckpt-{frame}-300.pth', map_location='cpu', weights_only=False)
    expected = checkpoint_raw(ckpt)
    exact = {name: torch.equal(raw[name], expected[name]) for name in RAW_FIELDS}
    assert all(exact.values()), exact
    return dict(frame=frame, all_attributes_bit_exact=True, fields=exact,
                count=len(state.xyz), reference=str(reference))


def main():
    flags.FLAGS(['verify-multiview-formal'])
    logging.set_verbosity(logging.INFO)
    torch.set_num_threads(4)
    original = verify_stream(SOURCE, 9)
    boundary = torch.load(SOURCE / 'rigid-ckpt-9-260.pth', map_location='cpu', weights_only=False)
    final = torch.load(SOURCE / 'rigid-ckpt-9-300.pth', map_location='cpu', weights_only=False)
    old = checkpoint_raw(boundary)
    full = checkpoint_raw(final)
    exact_old = {name: torch.equal(value, full[name][:len(value)]) for name, value in old.items()}
    assert all(exact_old.values()), exact_old
    assert len(full['xyz']) == len(old['xyz']) + len(boundary['birth_source_rows'])
    old_motion = boundary['rigid_tree_motion']['tensors']
    final_motion = final['rigid_tree_motion']['tensors']
    unchanged_motion = {name: torch.equal(value, final_motion[name])
                        for name, value in old_motion.items() if torch.is_tensor(value)
                        and name in final_motion and value.shape == final_motion[name].shape}
    assert all(unchanged_motion.values()), unchanged_motion

    workspace = Path(tempfile.mkdtemp(prefix='my4dgs_multiview_resume_'))
    Path('/tmp/my4dgs_multiview_resume_latest_path').write_text(str(workspace)+'\n')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        config, _ = to_structured(OmegaConf.load(ROOT / 'config/n3dv/flame_steak_1020_full300_multiview.yaml'), Config)
    config.output_dir = workspace / 'output'
    config.experiment_name = 'resume009_continue010'
    config.ckpt_path = SOURCE / 'rigid-ckpt-9-260.pth'
    pipeline = Pipeline(config)
    trainer = pipeline.trainer
    trainer.setup(pipeline.module, config.ckpt_path)
    pipeline.save_config()
    assert trainer.current_frame == 9 and trainer.global_step == 260
    resumed9 = trainer.frame_training_loop()
    resumed_replay9 = verify_stream(trainer.output_dir, 9)
    packets = ['motion_009.bin', 'birth_009.bin', 'opacity_009.bin', 'frame_009.json']
    packet_exact = {name: (SOURCE/'frame_storage/rigid_tree'/name).read_bytes()
                    == (trainer.output_dir/'frame_storage/rigid_tree'/name).read_bytes()
                    for name in packets}
    assert all(packet_exact.values()), packet_exact
    # Finalized frame transition exercises next-frame birth lineage/history.
    trainer._restore_rng(trainer._resume_rng)
    trainer._resume_rng = None
    pipeline.module.begin_next_frame()
    trainer.global_step = 0
    pipeline.datamodule.current_frame = 10
    report10 = trainer.frame_training_loop()
    trainer.save_checkpoint()
    replay10 = verify_stream(trainer.output_dir, 10)
    metadata = dict(source_run=str(SOURCE), run_dir=str(trainer.output_dir),
        source_replay=original, old_attributes_fixed_260_to_300=exact_old,
        motion_tensors_fixed_260_to_300=unchanged_motion,
        resumed_frame009_stages=resumed9['stage_image_metrics'],
        resumed_frame009_replay=resumed_replay9, frame010_replay=replay10,
        actual_resume_packets_bit_exact=packet_exact,
        frame010_stages=report10['stage_image_metrics'],
        scope='Strict native frame009 step260 resume and one frame010 continuation; no full300 run.')
    (workspace / 'verification.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('VERIFICATION', json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
