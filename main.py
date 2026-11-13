import os
import sys
import warnings
from typing import List, Tuple, Type

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

warnings.filterwarnings("ignore", category=UserWarning, module="tinycudann")
warnings.filterwarnings("ignore", category=UserWarning, message="No seed found")

from absl import app, flags, logging
from omegaconf import DictConfig
from omegaconf import OmegaConf as oc

from libgs.pipeline.loader import load_pipeline
from libgs.utils.config import to_structured, to_yaml
from libgs.utils.git import get_git_commit_hash

flags.DEFINE_string("pipeline", "pipeline", "pipeline name")
flags.DEFINE_string("config", "", "config file path")
flags.DEFINE_bool("print", False, "print config to stdout and exit")
flags.set_default(logging.ALSOLOGTOSTDERR, True)
flags.set_default(logging.SHOWPREFIXFORINFO, False)


def resolve_config(argv: List[str], Config: Type) -> Tuple["Config", DictConfig]:
    args = oc.from_cli(argv[1:])
    if flags.FLAGS.config:
        args = oc.merge(oc.load(flags.FLAGS.config), args)

    # Removed production policies must not silently survive as libgs extras.
    birth = oc.select(args, "module.birth", default={}) or {}
    removed = sorted(key for key in birth if key.startswith("tail_commit_"))
    if removed:
        raise ValueError(f"Removed ABCD/periodic attribute policies: {removed}. "
                         "Remove these keys and start a new experiment from the shared first frame.")

    full_update = oc.select(args, "module.full_update", default={}) or {}
    if oc.select(args, "module.rigid_tree.target_update_bytes", default=0) != 0:
        raise ValueError("Byte-budget feedback is retired; set a fixed module.rigid_tree.rate_loss_weight")
    if (oc.select(args, "module.rigid_tree.initial_rate_multiplier", default=1) != 1 or
            oc.select(args, "module.rigid_tree.min_rate_multiplier", default=1) != 1):
        raise ValueError("Rate multipliers are retired; use a fixed rate_loss_weight")
    if full_update.get("enabled", False):
        supported = {"enabled", "audit_camera_sequence",
                     "gradient_diagnostics", "logscale_step", "sh_dc_step", "sh_rest_step",
                     "storage_loss_enabled", "per_attribute_l1_weight", "point_dc_rate_scale", "point_logscale_rate_multiplier", "point_sh_rest_rate_multiplier",
                     "repair_temporal_weight",
                     "repair_temporal_scale", "repair_temporal_lag", "repair_rmse_mix", "motion_rmse_mix",
                     "early_joint_repair", "dense_color_200", "regional_full_200", "attribute_aware_gate",
                     "regional_opacity_l1_weight", "shared_region_orientation"}
        unsupported = sorted(set(full_update) - supported)
        if unsupported:
            raise ValueError(f"Retired full-update training options: {unsupported}. "
                             "Use the current 200-step full-attribute plan.")
    retired = [name for name in ("learnable_gate", "point_gate_enabled",
                                 "gate_probability_weighted_proximal",
                                 "temporal_focus_strength", "tail_rmse_mix")
               if full_update.get(name, False)]
    if retired:
        raise ValueError(f"Removed learnable attribute-repair options: {retired}. "
                         "Use the image-based regional repair configuration instead.")
    retired_resume = [name for name in ("resume_allow_repair_gate_change",
                                        "resume_allow_temporal_focus_change",
                                        "resume_allow_tail_rmse_change")
                      if args.get(name, False)]
    if retired_resume:
        raise ValueError(f"Removed attribute-repair resume options: {retired_resume}")

    # libgs retains valid strings after trying to call typing.Optional itself.
    # Silence only these known Path/str conversion notices, not other warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, module=r"libgs\.utils\.config",
            message=r"(?:Try|Failed) to cast '.*' to typing\.Optional\[(?:pathlib\.Path|str)\]$",
        )
        cfg, args = to_structured(args, Config)
    normalize_views = getattr(cfg.trainer, 'use_single_training_view', None)
    if cfg.mode == 'train' and normalize_views is not None:
        normalize_views()
    if flags.FLAGS.print:
        print(to_yaml(cfg))
        sys.exit(0)

    return cfg, args


def main(argv):
    if flags.FLAGS.pipeline == 'pipeline':
        from pipeline import Config, Pipeline
    else:
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
