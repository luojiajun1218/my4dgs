"""Check the published recipe starts directly from the canonical native frame0."""
from pathlib import Path
import hashlib
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
from pipeline.model.birth_quantization import RAW_FIELDS
from scripts.replay_rigid_tree import replay


def main():
    flags.FLAGS(['multiview-native-first-frame'])
    logging.set_verbosity(logging.INFO)
    torch.set_num_threads(4)
    workspace = Path(tempfile.mkdtemp(prefix='my4dgs_multiview_frame001_'))
    Path('/tmp/my4dgs_multiview_frame001_latest_path').write_text(str(workspace)+'\n')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        cfg, _ = to_structured(OmegaConf.load(ROOT/'config/n3dv/flame_steak_1020_full300_multiview.yaml'), Config)
    cfg.output_dir = workspace/'output'
    cfg.experiment_name = 'native_frame001_multiview'
    native_path = Path(cfg.ckpt_path)
    native_sha = hashlib.sha256(native_path.read_bytes()).hexdigest()
    assert native_sha == 'fc1c7822198f265b0c0641671411b4d952b4cbc53fa25ff748fc0dbb1c1f534f'
    pipeline = Pipeline(cfg)
    trainer = pipeline.trainer
    trainer.setup(pipeline.module, cfg.ckpt_path)
    assert trainer.is_first_frame and len(pipeline.module.gaussians._xyz) == 128963
    # Match Trainer.training_loop's native bootstrap branch: no dynamic
    # frame exists yet, so there is no finalized frame to advance from.
    trainer.global_step = 0
    pipeline.datamodule.current_frame = 1
    report = trainer.frame_training_loop()
    ckpt = torch.load(trainer.output_dir/'rigid-ckpt-1-300.pth', map_location='cpu', weights_only=False)
    expected = dict(zip(('xyz','sh_dc','sh_rest','logscale','rotation','opacity'), ckpt['gaussians'][1:7]))
    stream = trainer.output_dir/'frame_storage/rigid_tree'
    _, decoded = replay(stream, stream/'reference_frame0.npz', through=1)
    exact = {name: torch.equal(expected[name], decoded[name]) for name in RAW_FIELDS}
    assert all(exact.values()), exact
    assert hashlib.sha256(native_path.read_bytes()).hexdigest() == native_sha
    metadata = dict(run_dir=str(trainer.output_dir), source_checkpoint=str(cfg.ckpt_path),
        source_checkpoint_sha256=native_sha, source_checkpoint_unchanged=True,
        native_count=132586, retained_initial_count=128963,
        stages=report['stage_image_metrics'], decoded_fields_bit_exact=exact,
        scope='Published recipe, native frame0 bootstrap and one dynamic frame001; no full300 run.')
    (workspace/'verification.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('NATIVE_SMOKE', json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
