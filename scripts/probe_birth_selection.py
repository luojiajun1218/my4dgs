"""Explicit recipe forks at saved motion boundaries, using the real birth/codec path."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import warnings

import torch
from absl import flags, logging
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from libgs.utils.config import to_structured
from pipeline.entry import Config, Pipeline
from pipeline.model.birth_quantization import RAW_FIELDS
from scripts.replay_rigid_tree import replay


def check_replay(run, frame):
    stream = run/'frame_storage/rigid_tree'
    reference = next(stream.glob('reference_frame*.npz'))
    _, decoded = replay(stream, reference, through=frame)
    checkpoint = torch.load(run/f'rigid-ckpt-{frame}-300.pth', map_location='cpu', weights_only=False)
    raw = dict(zip(('xyz', 'sh_dc', 'sh_rest', 'logscale', 'rotation', 'opacity'),
                   checkpoint['gaussians'][1:7]))
    fields = {name: torch.equal(decoded[name], raw[name]) for name in RAW_FIELDS}
    if not all(fields.values()):
        raise AssertionError(fields)
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+', default=[1, 9, 29])
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--continue-frames', type=int, default=0,
                        help='Additional contiguous frames after the last probe boundary')
    args = parser.parse_args()
    flags.FLAGS(['birth-selection-probe'])
    logging.set_verbosity(logging.INFO)
    torch.set_num_threads(4)
    workspace = args.output_root or Path(tempfile.mkdtemp(prefix='my4dgs_birth_selection_'))
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace/'results.json').write_text('[]\n')
    records = []
    for frame in args.frames:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            old, _ = to_structured(OmegaConf.load(args.run_dir/'config.yaml'), Config)
            new, _ = to_structured(OmegaConf.load(args.config), Config)
        old.output_dir = workspace/'output'
        old.experiment_name = f'frame{frame:03d}_selection'
        old.ckpt_path = args.run_dir/f'rigid-ckpt-{frame}-180.pth'
        pipeline = Pipeline(old)
        trainer = pipeline.trainer
        trainer.setup(pipeline.module, old.ckpt_path)
        if trainer.global_step != 180 or not pipeline.module.motion_committed:
            raise ValueError('This probe requires a completed motion boundary')
        # Deliberate experimental fork; ordinary resume remains strict.
        pipeline.module.config.birth = copy.deepcopy(new.module.birth)
        pipeline.module.config.gaussian_stage2 = copy.deepcopy(new.module.gaussian_stage2)
        pipeline.module.config.birth.validate(pipeline.module.motion_steps)
        trainer.config = copy.deepcopy(new.trainer)
        pipeline.config.trainer = trainer.config
        pipeline.save_config()
        extra = args.continue_frames if frame == args.frames[-1] else 0
        for offset in range(extra+1):
            current = frame+offset
            if offset:
                pipeline.module.begin_next_frame()
                trainer.global_step = 0
                pipeline.datamodule.current_frame = current
            report = trainer.frame_training_loop()
            trainer.save_checkpoint()
            record = dict(frame=current, source_boundary=str(old.ckpt_path),
                run_dir=str(trainer.output_dir), stages=report['stage_image_metrics'],
                new=report['birth_count'], deleted=report['retired_count'],
                encoded_bytes=report['encoded_bytes'], evaluation=report['evaluation'],
                admission=report['birth_control'].get('birth_admission'),
                decoded_fields_bit_exact=check_replay(trainer.output_dir, current),
                scope='Fixed saved motion followed by actual birth selection, refit and codec'
                      if not offset else 'One contiguous next-frame continuation')
            records.append(record)
            (workspace/'results.json').write_text(json.dumps(records, indent=2)+'\n')
            small = {key:record[key] for key in ('frame', 'new', 'deleted', 'encoded_bytes')}
            small['stages'] = {s['step']:s['psnr'] for s in record['stages']}
            print('BIRTH_SELECTION_RESULT', json.dumps(small), flush=True)
        del pipeline, trainer
    print('RESULTS', workspace/'results.json', flush=True)


if __name__ == '__main__':
    main()
