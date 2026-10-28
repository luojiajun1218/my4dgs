"""Run the original ReCon-GS static trainer with safe mid-frame clock setup.

The external source tree is read only. The local Trainer subclass starts its
frame timer on resume. --colour-only freezes geometry and opacity while native
SH fitting, rendering and checkpointing continue in the original implementation.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


STATIC_FIELDS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')
COLOUR_FIELDS = ('_features_dc', '_features_rest')


def configure_static_parameters(model, colour_only):
    """Set trainability on restored leaves without changing optimizer references."""
    for name in STATIC_FIELDS:
        value = getattr(model, name)
        if not value.is_leaf:
            raise ValueError(f'Static checkpoint field is not a leaf: {name}')
        value.grad = None
        value.requires_grad_(not colour_only or name in COLOUR_FIELDS)
        if not any(value is parameter for group in model.optimizer.param_groups
                   for parameter in group['params']):
            raise ValueError(f'Static optimizer does not reference restored field: {name}')


def static_field_digests(model):
    return {name: hashlib.sha256(getattr(model, name).detach().contiguous().cpu().numpy().tobytes()).hexdigest()
            for name in STATIC_FIELDS}


def verify_colour_only(before, after):
    changed = [name for name in STATIC_FIELDS if before[name] != after[name]]
    if any(name not in COLOUR_FIELDS for name in changed):
        raise ValueError(f'Colour-only fit changed frozen attributes: {changed}')
    if not changed:
        raise ValueError('Colour-only fit did not change any SH weights')
    return changed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--name')
    parser.add_argument('--colour-only', action='store_true',
        help='Fit initial SH only; keep xyz, scale, rotation and opacity bit-exact')
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
            if not self.is_first_frame:
                raise ValueError('This entry is limited to the static first frame')
            start_step, end_step = self.global_step, self.num_steps
            if start_step >= end_step:
                raise ValueError('Static fitting needs remaining training steps')
            configure_static_parameters(self.module.gaussians, args.colour_only)
            before = static_field_digests(self.module.gaussians)
            trainable = [name for name in STATIC_FIELDS
                         if getattr(self.module.gaussians, name).requires_grad]
            if self.is_first_frame and self.global_step>0:
                self.module.timer.clock('frame').start()
            result = super().frame_training_loop()
            after = static_field_digests(self.module.gaussians)
            changed = (verify_colour_only(before, after) if args.colour_only else
                       [name for name in STATIC_FIELDS if before[name] != after[name]])
            report = dict(colour_only=args.colour_only, start_step=start_step,
                end_step=end_step, trainable_fields=trainable, changed_fields=changed,
                before_sha256=before, after_sha256=after,
                frozen_fields_bit_exact=all(before[name] == after[name]
                    for name in STATIC_FIELDS if name not in trainable))
            (self.output_dir / 'first_frame_trainability.json').write_text(json.dumps(report, indent=2)+'\n')
            print('STATIC_TRAINABILITY_VERIFIED', json.dumps(report), flush=True)
            return result

    native.Trainer=ResumeTrainer
    config,extra=to_structured(OmegaConf.load(args.config.resolve()),Config)
    if extra:raise ValueError(f'Unrecognized first-frame config: {extra}')
    if config.data.extra_dataset_kwargs.get('num_frames')!=1:
        raise ValueError('This entry is limited to one static first frame')
    if args.colour_only and (config.module.noise_scale is not None or
                            config.module.densify.until_step != 0):
        raise ValueError('Colour-only fitting requires noise_scale=null and densify.until_step=0')
    if args.checkpoint is not None:config.ckpt_path=args.checkpoint.resolve()
    if args.name is not None:config.experiment_name=args.name
    Pipeline(config).run()


if __name__=='__main__':main()
