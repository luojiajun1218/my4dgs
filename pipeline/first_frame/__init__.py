"""首帧静态训练管线的配置和组装入口。

定义首帧 Config、初始化运行环境，并组装共享数据模块、首帧模型和训练调度器。
静态训练逻辑分别位于同目录 module.py、trainer.py；动态帧由项目主管线负责。
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import torch
from libgs.pipeline import Config as BaseConfig, Pipeline as BasePipeline
from libgs.renderer import network_gui
from libgs.utils.general import safe_state

from ..data import DataConfig, DataModule
from .module import Module, ModuleConfig
from .trainer import Trainer, TrainerConfig


@dataclass
class Config(BaseConfig):
    output_dir: Path = Path('output')
    experiment_name: str = datetime.now().strftime('%Y-%m-%d')
    mode: Literal['train', 'validate'] = 'train'
    ckpt_path: Optional[Path] = None
    gui_ip: str = '127.0.0.1'
    gui_port: int = 6009
    detect_anomaly: bool = False
    quiet: bool = False
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    module: ModuleConfig = field(default_factory=ModuleConfig)


class Pipeline(BasePipeline):
    def setup(self):
        safe_state(self.config.quiet)
        network_gui.init(self.config.gui_ip, self.config.gui_port)
        torch.autograd.set_detect_anomaly(self.config.detect_anomaly)
        return super().setup()

    def setup_trainer_and_modules(self):
        output_dir = Path(self.config.output_dir) / self.config.experiment_name
        self.datamodule = DataModule(self.config.data)
        self.module = Module(self.config.module, datamodule=self.datamodule)
        self.trainer = Trainer(self.config.trainer, output_dir=output_dir)
