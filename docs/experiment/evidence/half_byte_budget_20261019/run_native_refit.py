"""Initialize the native frame timer when resuming inside frame0."""
import importlib
from pathlib import Path
import runpy
import sys
native = Path('/home/gamer-02/Documents/DeepLearning/ReCon-GS')
sys.path.insert(0,str(native))
Trainer = importlib.import_module('pipeline.ReCon-GS.trainer').Trainer
original = Trainer.frame_training_loop

def resume_frame(self):
    self.module.timer.clock('frame').start()
    return original(self)

Trainer.frame_training_loop = resume_frame
config = sys.argv[1]
sys.argv = [str(native/'main.py'),'--config='+config]
runpy.run_path(str(native/'main.py'),run_name='__main__')
