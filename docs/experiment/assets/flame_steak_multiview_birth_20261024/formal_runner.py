"""Explicit completed-frame branch to test a new birth recipe on frame009."""
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

PREVIOUS=Path('/tmp/my4dgs_frame009_amdd4mjt/output/restore_frame009/20261023T232410_e675b7')

def main():
    flags.FLAGS(['frame009-multiview-formal'])
    logging.set_verbosity(logging.INFO)
    torch.set_num_threads(4)
    workspace=Path(tempfile.mkdtemp(prefix='my4dgs_multiview_formal9_'))
    Path('/tmp/my4dgs_multiview_formal9_latest_path').write_text(str(workspace)+'\n')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        old,_=to_structured(OmegaConf.load(PREVIOUS/'config.yaml'),Config)
        new,_=to_structured(OmegaConf.load(ROOT/'config/n3dv/flame_steak_1020_full300_multiview.yaml'),Config)
    old.output_dir=workspace/'output'
    old.experiment_name='frame009_multiview_formal'
    old.ckpt_path=PREVIOUS/'rigid-ckpt-8-300.pth'
    pipeline=Pipeline(old)
    trainer=pipeline.trainer
    trainer.setup(pipeline.module,old.ckpt_path)
    assert trainer.current_frame==8 and trainer.global_step==300
    # This is an explicit experimental recipe fork at a finalized boundary.
    # Ordinary resume remains strict and will not silently migrate a recipe.
    pipeline.module.config.birth=new.module.birth
    pipeline.module.config.gaussian_stage2=new.module.gaussian_stage2
    pipeline.module.config.opacity_update.joint_with_birth=False
    pipeline.module.config.birth.validate(180)
    trainer.config=new.trainer
    pipeline.config.trainer=new.trainer
    pipeline.save_config()
    trainer._restore_rng(trainer._resume_rng)
    trainer._resume_rng=None
    pipeline.module.begin_next_frame()
    trainer.global_step=0
    pipeline.datamodule.current_frame=9
    report=trainer.frame_training_loop()
    metadata=dict(previous_run=str(PREVIOUS),previous_checkpoint=str(old.ckpt_path),
        config=str(ROOT/'config/n3dv/flame_steak_1020_full300_multiview.yaml'),
        run_dir=str(trainer.output_dir),scope='Real 180+80+40 frame009 pipeline, branched explicitly at finalized frame008. No full sequence run.')
    (workspace/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('FORMAL9_RESULT',json.dumps(dict(metadata=metadata,
        stages=report['stage_image_metrics'],new=report['birth_count'],deleted=report['retired_count'],
        encoded_bytes=report['encoded_bytes'],admission=report['birth_control'].get('birth_admission'))),flush=True)

if __name__=='__main__':main()
