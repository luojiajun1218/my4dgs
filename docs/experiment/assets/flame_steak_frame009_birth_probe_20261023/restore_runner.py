"""Restore frame 3 and reconstruct frames 4..9; isolate frame-9 birth input."""
from pathlib import Path
import json
import sys
import tempfile
import warnings
import torch
from omegaconf import OmegaConf
from absl import flags,logging

ROOT=Path('/home/gamer-02/Documents/DeepLearning/my4dgs')
sys.path.insert(0,str(ROOT))
from libgs.utils.config import to_structured
from pipeline.entry import Config,Pipeline

SOURCE=ROOT/'output/n3dv/flame_steak_1020/full300_130k_birthdriven10pct_180_80_40/20261022T142925_c8bbee'

def main():
    flags.FLAGS(['frame009-diagnostic'])
    logging.set_verbosity(logging.INFO)
    torch.set_num_threads(4)
    workspace=Path(tempfile.mkdtemp(prefix='my4dgs_frame009_'))
    Path('/tmp/my4dgs_frame009_latest_path').write_text(str(workspace)+'\n')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        cfg,_=to_structured(OmegaConf.load(SOURCE/'config.yaml'),Config)
    cfg.output_dir=workspace/'output'
    cfg.experiment_name='restore_frame009'
    cfg.ckpt_path=SOURCE/'rigid-ckpt-3-300.pth'
    pipeline=Pipeline(cfg)
    trainer=pipeline.trainer
    trainer.setup(pipeline.module,cfg.ckpt_path)
    assert trainer.current_frame==3 and trainer.global_step==300
    trainer._checkpoint_due=lambda: ((trainer.current_frame==8 and trainer.global_step==300)
        or (trainer.current_frame==9 and trainer.global_step in (180,260,300)))
    trainer._restore_rng(trainer._resume_rng)
    trainer._resume_rng=None
    pipeline.module.begin_next_frame()
    trainer.global_step=0
    reports=[]
    for frame in range(4,10):
        pipeline.datamodule.current_frame=frame
        reports.append(trainer.frame_training_loop())
        if frame<9:
            pipeline.module.begin_next_frame()
            trainer.global_step=0
        print('COMPLETED_FRAME',frame,flush=True)
    diagnostic=OmegaConf.load(ROOT/'config/diagnostic/flame_steak_birth_revision.yaml')
    diagnostic.run_dir=str(trainer.output_dir)
    diagnostic.frames=[9]
    diagnostic.result_json=str(workspace/'frame009_birth_probe.json')
    OmegaConf.save(diagnostic,workspace/'frame009_birth_probe.yaml')
    metadata=dict(source_run=str(SOURCE),restore_checkpoint=str(cfg.ckpt_path),
        reconstruction_frames=list(range(4,10)),run_dir=str(trainer.output_dir),
        diagnostic_config=str(workspace/'frame009_birth_probe.yaml'),
        diagnostic_result=str(workspace/'frame009_birth_probe.json'),
        scope='Restored original frame-3 optimizer/motion/RNG state, retrained frames 4..9 with baseline config. CUDA training is not guaranteed bitwise identical to the archived run.')
    (workspace/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('FRAME009_METADATA',json.dumps(metadata),flush=True)

if __name__=='__main__':main()
