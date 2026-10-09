"""Formal training entry for the common/contrast motion backend.

This keeps the standard libgs configuration/checkpoint path and saves the
source files needed to reproduce the selected motion backend.
"""

import sys
import warnings
from pathlib import Path
import hashlib
import json
import shutil
from typing import List, Tuple, Type

warnings.filterwarnings('ignore', category=UserWarning, module='tinycudann')
warnings.filterwarnings('ignore', category=UserWarning, message='No seed found')

# A script below the repository root does not receive the root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from absl import app, flags, logging
from omegaconf import DictConfig
from omegaconf import OmegaConf as oc

from libgs.pipeline.loader import load_pipeline
from libgs.utils.config import to_structured, to_yaml
from libgs.utils.git import get_git_commit_hash


flags.DEFINE_string('pipeline', 'ReCon-GS', 'pipeline name')
flags.DEFINE_string('config', '', 'config file path')
flags.DEFINE_bool('print', False, 'print config and exit')
flags.set_default(logging.ALSOLOGTOSTDERR, True)
flags.set_default(logging.SHOWPREFIXFORINFO, False)


def resolve_config(argv: List[str], Config: Type) -> Tuple['Config', DictConfig]:
    args = oc.from_cli(argv[1:])
    if flags.FLAGS.config:
        args = oc.merge(oc.load(flags.FLAGS.config), args)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning,
            module=r'libgs\.utils\.config',
            message=r"(?:Try|Failed) to cast '.*' to typing\.Optional\[(?:pathlib\.Path|str)\]$")
        cfg, args = to_structured(args, Config)
    if flags.FLAGS.print:
        print(to_yaml(cfg))
        sys.exit(0)
    if cfg.module.motion_mode != 'common_contrast_binary':
        raise ValueError('This entry accepts only module.motion_mode=common_contrast_binary')
    return cfg, args


def main(argv):
    Config, Pipeline = load_pipeline(flags.FLAGS.pipeline)
    cfg, args = resolve_config(argv, Config)
    if args:
        logging.info('Arguments not specified in Config: %s', args)
    pipeline = Pipeline(cfg, **oc.to_container(args))
    if pipeline.is_global_zero:
        logging.get_absl_handler().use_absl_log_file('logging', pipeline.output_dir)
        source_dir = Path(pipeline.output_dir) / 'source_snapshot'
        source_dir.mkdir(parents=True, exist_ok=False)
        sources = [Path(__file__).resolve(), Path(flags.FLAGS.config).resolve(),
            Path('pipeline/ReCon-GS/model/common_contrast.py').resolve(),
            Path('pipeline/ReCon-GS/model/node_motion.py').resolve(),
            Path('pipeline/ReCon-GS/model/binary_refinement.py').resolve(),
            Path('pipeline/ReCon-GS/model/storage_aware_binary.py').resolve(),
            Path('pipeline/ReCon-GS/model/residual_motion.py').resolve(),
            Path('pipeline/ReCon-GS/model/direct_tree.py').resolve(),
            Path('pipeline/ReCon-GS/model/tree_edit_codec.py').resolve(),
            Path('pipeline/ReCon-GS/model/split_region_scores.py').resolve(),
            Path('pipeline/ReCon-GS/direct_transaction.py').resolve(),
            Path('pipeline/ReCon-GS/module.py').resolve(),
            Path('pipeline/ReCon-GS/trainer.py').resolve()]
        manifest = []
        for index, source in enumerate(sources):
            target = source_dir / f'{index:02d}_{source.name}'
            shutil.copy2(source, target)
            manifest.append(dict(source=str(source), snapshot=target.name,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        (source_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    logging.info('Git commit hash: %s', get_git_commit_hash())
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning,
            module=r'libgs\.pipeline',
            message=r'You are using `torch\.load` with `weights_only=False`.*')
        result = pipeline.run()
    logging.info('Finished common/contrast pipeline.')
    return result


if __name__ == '__main__':
    app.run(main)
