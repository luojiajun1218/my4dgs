"""Run the original ReCon-GS static trainer with safe mid-frame clock setup.

The external source tree is read only. The local Trainer subclass starts its
frame timer on resume; native optimization, rendering and checkpointing stay
in the original implementation.
"""
import argparse
import importlib
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--name')
    parser.add_argument('--recon-root',type=Path,
        default=Path('/home/gamer-02/Documents/DeepLearning/ReCon-GS'))
    args=parser.parse_args()
    sys.path.insert(0,str(args.recon_root.resolve()))
    from omegaconf import OmegaConf
    from libgs.pipeline.loader import load_pipeline
    from libgs.utils.config import to_structured
    Config,Pipeline=load_pipeline('ReCon-GS')
    native=importlib.import_module(Pipeline.__module__)

    class ResumeTrainer(native.Trainer):
        def frame_training_loop(self):
            if self.is_first_frame and self.global_step>0:
                self.module.timer.clock('frame').start()
            return super().frame_training_loop()

    native.Trainer=ResumeTrainer
    config,extra=to_structured(OmegaConf.load(args.config.resolve()),Config)
    if extra:raise ValueError(f'Unrecognized first-frame config: {extra}')
    if config.data.extra_dataset_kwargs.get('num_frames')!=1:
        raise ValueError('This entry is limited to one static first frame')
    if args.checkpoint is not None:config.ckpt_path=args.checkpoint.resolve()
    if args.name is not None:config.experiment_name=args.name
    Pipeline(config).run()


if __name__=='__main__':main()
