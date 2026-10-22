"""Resume native frame 0 while initializing its wall-clock timer.

The native loop normally starts the frame timer at step 1. A completed-frame
resume skips that step, so initialize the timer before entering the loop.
"""

import argparse
import importlib
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    native = Path(__file__).resolve().parents[2] / 'ReCon-GS'
    sys.path.insert(0, str(native))
    trainer = importlib.import_module('pipeline.ReCon-GS.trainer').Trainer
    original = trainer.frame_training_loop

    def resume_frame(self):
        if self.global_step > 0:
            self.module.timer.clock('frame').start()
        return original(self)

    trainer.frame_training_loop = resume_frame
    sys.argv = [str(native / 'main.py'), '--config=' + str(args.config.resolve())]
    runpy.run_path(str(native / 'main.py'), run_name='__main__')


if __name__ == '__main__':
    main()
