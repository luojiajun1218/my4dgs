"""训练管线的配置与组装入口。

定义顶层配置，创建并连接数据模块、模型和训练器，供 libgs 调用；
具体数据处理、模型计算及训练调度分别由对应文件负责。
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import torch

from libgs.pipeline import Config as BaseConfig
from libgs.pipeline import Pipeline as BasePipeline
from libgs.utils.general import safe_state

from .data import DataConfig, DataModule
from .module import Module, ModuleConfig
from .trainer import Trainer, TrainerConfig


@dataclass
class Config(BaseConfig):
    output_dir: Path = Path("output")
    experiment_name: str = datetime.now().strftime("%Y-%m-%d")
    mode: Literal["train", "validate"] = "train"
    ckpt_path: Optional[Path] = None
    resume_prefix_dir: Optional[Path] = None
    resume_completed_prefix_frames: Optional[int] = None
    resume_last_frame: Optional[int] = None
    # Opt in only when a historical dataset root was removed after relocation.
    resume_allow_unverified_data_root: bool = False
    diagnostic_train_camera: Optional[str] = None
    gate_refine_image_ste: bool = False
    gui_ip: str = "127.0.0.1"
    gui_port: int = 6009
    gui_enabled: bool = True
    detect_anomaly: bool = False
    quiet: bool = False
    random_seed: int = 0
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    module: ModuleConfig = field(default_factory=ModuleConfig)


class Pipeline(BasePipeline):
    datamodule: DataModule
    module: Module
    trainer: Trainer

    def setup(self):
        if self.config.mode == 'train':
            self.config.trainer.use_single_training_view()
        safe_state(self.config.quiet, seed=self.config.random_seed)
        if self.config.gui_enabled:
            from libgs.renderer import network_gui
            network_gui.init(self.config.gui_ip, self.config.gui_port)
        torch.autograd.set_detect_anomaly(self.config.detect_anomaly)
        return super().setup()

    def setup_trainer_and_modules(self):
        output_dir = Path(self.config.output_dir) / self.config.experiment_name
        self.datamodule = DataModule(self.config.data)
        self.module = Module(self.config.module, datamodule=self.datamodule)
        self.module.gui_enabled = self.config.gui_enabled
        if not isinstance(self.config.gate_refine_image_ste, bool):
            raise ValueError('Gate STE flag must be boolean')
        gate=self.config.module.rigid_tree.spatial_gate
        if self.config.gate_refine_image_ste and not (gate.frame_training_plan and gate.shared_image_recode and
                gate.joint_image_gradients and not gate.hierarchical_private and not gate.private_full_motion and not gate.predictor_image_weight):
            raise ValueError('Gate image STE supports only the physical-recoding frame plan')
        self.module.gate_refine_image_ste = self.config.gate_refine_image_ste
        self.trainer = Trainer(self.config.trainer, output_dir=output_dir)
        if self.config.resume_prefix_dir is not None and self.config.mode != 'train':
            raise ValueError('Completed-prefix extension is a training entry point')
        if (self.config.resume_completed_prefix_frames is not None and
                self.config.resume_prefix_dir is None):
            raise ValueError('An explicit completed prefix length needs resume_prefix_dir')
        if self.config.resume_last_frame is not None and self.config.resume_prefix_dir is None:
            raise ValueError('A frame limit requires a completed prefix')
        self.trainer.resume_prefix_dir = self.config.resume_prefix_dir
        self.trainer.resume_completed_prefix_frames = self.config.resume_completed_prefix_frames
        self.trainer.resume_last_frame = self.config.resume_last_frame
        if not isinstance(self.config.resume_allow_unverified_data_root, bool):
            raise ValueError('resume_allow_unverified_data_root must be boolean')
        self.trainer.resume_allow_unverified_data_root = self.config.resume_allow_unverified_data_root
        self.trainer.diagnostic_train_camera = self.config.diagnostic_train_camera
