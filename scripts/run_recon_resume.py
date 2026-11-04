"""Run ReCon-GS with its frame timer started when resuming the first frame.

The upstream trainer starts this timer only at step 1, but stops it at the
final step even if training was resumed later. This wrapper changes timing
only; the training loop, optimizer and checkpoint remain the original ones.
"""

import importlib
from pathlib import Path
import runpy
import sys


RECON_ROOT = Path('/home/gamer-02/Documents/DeepLearning/ReCon-GS')
sys.path.insert(0, str(RECON_ROOT))
trainer = importlib.import_module('pipeline.ReCon-GS.trainer').Trainer
original_frame_training_loop = trainer.frame_training_loop


def resume_safe_frame_training_loop(self):
    if self.global_step > 0 and self.datamodule.current_frame == 0:
        self.module.timer.clock('frame').start()
    return original_frame_training_loop(self)


trainer.frame_training_loop = resume_safe_frame_training_loop
runpy.run_path(str(RECON_ROOT / 'main.py'), run_name='__main__')
