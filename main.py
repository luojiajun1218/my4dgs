import sys
import warnings
from typing import List, Tuple, Type

warnings.filterwarnings("ignore", category=UserWarning, module="tinycudann")
warnings.filterwarnings("ignore", category=UserWarning, message="No seed found")

from absl import app, flags, logging
from omegaconf import DictConfig
from omegaconf import OmegaConf as oc

from libgs.pipeline.loader import load_pipeline
from libgs.utils.config import to_structured, to_yaml
from libgs.utils.git import get_git_commit_hash

#默认启动环境从base改成了ReConGS
flags.DEFINE_string("pipeline", "my4dgs", "pipeline name")
flags.DEFINE_string("config", "", "config file path")
flags.DEFINE_bool("print", False, "print config to stdout and exit")
flags.set_default(logging.ALSOLOGTOSTDERR, True)
flags.set_default(logging.SHOWPREFIXFORINFO, False)


def resolve_config(argv: List[str], Config: Type) -> Tuple["Config", DictConfig]:
    args = oc.from_cli(argv[1:])
    if flags.FLAGS.config:
        args = oc.merge(oc.load(flags.FLAGS.config), args)

    # libgs retains valid strings after trying to call typing.Optional itself.
    # Silence only these known Path/str conversion notices, not other warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, module=r"libgs\.utils\.config",
            message=r"(?:Try|Failed) to cast '.*' to typing\.Optional\[(?:pathlib\.Path|str)\]$",
        )
        cfg, args = to_structured(args, Config)
    if flags.FLAGS.print:
        print(to_yaml(cfg))
        sys.exit(0)

    return cfg, args


def main(argv):
    Config, Pipeline = load_pipeline(flags.FLAGS.pipeline)
    cfg, args = resolve_config(argv, Config)
    if args:
        logging.info(f"Arguments not specified in Config: {args}")
    pipeline = Pipeline(cfg, **oc.to_container(args))
    if pipeline.is_global_zero:
        log_dir = pipeline.output_dir
        logging.get_absl_handler().use_absl_log_file("logging", log_dir)
    logging.info(f"Git commit hash: {get_git_commit_hash()}")
    logging.info(f"Launch pipeline {flags.FLAGS.pipeline} ...")
    # The existing full-state checkpoint loader deliberately uses pickle.
    # Keep its current behavior; hide only PyTorch's verbose default-change notice.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=FutureWarning, module=r"libgs\.pipeline",
            message=r"You are using `torch\.load` with `weights_only=False`.*",
        )
        pipeline.run()
    logging.info(f"Finished pipeline {flags.FLAGS.pipeline}.")


if __name__ == "__main__":
    app.run(main)
