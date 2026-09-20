"""Create a fresh, fully resolved config and launch a controlled experiment."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import uuid

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['binary', 'reference', 'nested'], default='binary')
    parser.add_argument('--frames', type=int, default=6, help='Total frames including shared frame 0')
    parser.add_argument('--name', required=True)
    parser.add_argument('--set', nargs='*', default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / 'config/flame_steak_recongs_100_controlled.yaml')
    config.experiment_name = args.name
    config.data.extra_dataset_kwargs.num_frames = args.frames
    config.gui_port = 6020
    if args.method == 'binary':
        config.module.deformation.hierarchy_mode = 'binary'
        config.module.lambda_deform = .1
        config.module.deformation.binary_gradient_weight = .5
        config.module.deformation.binary_lambda_rate = .0001
    elif args.method == 'nested':
        config.module.deformation.hierarchy_mode = 'nested'
        config.module.deformation.refinement_mode = 'gradhet'
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.set))
    plans = root / 'output/experiment_configs'
    plans.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:6]
    path = plans / f'{args.name}_{stamp}.yaml'
    OmegaConf.save(config, path)
    source = root if args.method != 'reference' else root.parent / 'ReCon-GS'
    command = [sys.executable, str(root / 'scripts/run_controlled.py'), '--source', str(source)]
    if args.method == 'reference':
        command.append('--reference')
    command += ['--pipeline=ReCon-GS', '--config=' + str(path)]
    print('Resolved config:', path, flush=True)
    raise SystemExit(subprocess.call(command, cwd=root))


if __name__ == '__main__':
    main()
