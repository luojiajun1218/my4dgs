"""Strict subset-policy resume and two contiguous post-29 frames."""
from pathlib import Path
import json
import sys
import warnings
import torch
from absl import flags, logging
from omegaconf import OmegaConf

ROOT=Path('/home/gamer-02/Documents/DeepLearning/my4dgs')
sys.path.insert(0,str(ROOT))
from libgs.utils.config import to_structured
from pipeline.entry import Config,Pipeline
from pipeline.model.birth_quantization import model_raw
from scripts.probe_birth_selection import check_replay


def make_pipeline(source, workspace, experiment):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        cfg,_=to_structured(OmegaConf.load(ROOT/'config/n3dv/flame_steak_1024_multiview_select.yaml'),Config)
    cfg.output_dir=workspace/'output'
    cfg.experiment_name=experiment
    cfg.ckpt_path=source
    pipeline=Pipeline(cfg)
    pipeline.trainer.setup(pipeline.module,source)
    pipeline.save_config()
    return pipeline


def main():
    flags.FLAGS(['birth-select-resume'])
    logging.set_verbosity(logging.INFO)
    torch.set_num_threads(4)
    workspace=Path('/tmp/my4dgs_birth_selection_1024_verify')
    workspace.mkdir(parents=True,exist_ok=True)
    results=json.loads(Path('/tmp/my4dgs_birth_selection_1024_v2/results.json').read_text())
    original=Path(next(r['run_dir'] for r in results if r['frame']==9))
    pipeline=make_pipeline(original/'rigid-ckpt-9-260.pth',workspace,'resume009')
    tr=pipeline.trainer
    old=model_raw(pipeline.module.gaussians)
    report=tr.frame_training_loop()
    final=model_raw(pipeline.module.gaussians)
    fixed={name:torch.equal(value,final[name][:len(value)]) for name,value in old.items()}
    assert all(fixed.values()),fixed
    replay9=check_replay(tr.output_dir,9)
    files=['motion_009.bin','birth_009.bin','opacity_009.bin','frame_009.json']
    exact={name:(original/'frame_storage/rigid_tree'/name).read_bytes()
                ==(tr.output_dir/'frame_storage/rigid_tree'/name).read_bytes() for name in files}
    assert all(exact.values()),exact
    del pipeline,tr,old,final
    source29=Path(next(r['run_dir'] for r in results if r['frame']==29))
    pipeline=make_pipeline(source29/'rigid-ckpt-29-300.pth',workspace,'continue030_031')
    tr=pipeline.trainer
    tr._restore_rng(tr._resume_rng)
    tr._resume_rng=None
    continuation=[]
    for frame in (30,31):
        pipeline.module.begin_next_frame()
        pipeline.datamodule.current_frame=frame
        tr.global_step=0
        report=tr.frame_training_loop()
        tr.save_checkpoint()
        check=check_replay(tr.output_dir,frame)
        k,d=report['birth_count'],report['retired_count']
        assert abs(k-d)<=int(.1*k+1e-9)
        retired_recent=report['birth_control']['retirement']['retired_previous_birth_count']
        assert retired_recent==0
        row=dict(frame=frame,run_dir=str(tr.output_dir),stages=report['stage_image_metrics'],
            new=k,deleted=d,encoded_bytes=report['encoded_bytes'],
            previous_births=report['birth_control']['previous_birth_count'],
            retired_recent_births=retired_recent,decoded_fields_bit_exact=check)
        continuation.append(row)
        print('CONTINUATION',json.dumps(row),flush=True)
    value=dict(resume009=dict(run_dir=str(original),old_attributes_fixed=fixed,
        decoded_fields_bit_exact=replay9,packets_bit_exact=exact),continuation=continuation)
    (workspace/'verification.json').write_text(json.dumps(value,indent=2)+'\n')
    print('VERIFIED',workspace/'verification.json',flush=True)


if __name__=='__main__':main()
