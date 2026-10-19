"""Partition synchronized frame wall time without synchronizing every step."""
from contextlib import contextmanager
import time


class FrameTimer:
    def __init__(self, synchronize=None, clock=time.perf_counter):
        self.synchronize = synchronize or (lambda: None)
        self.clock = clock
        self.synchronize()
        self.started = self.updated = self.clock()
        self.current = 'preparation'
        self.seconds = {}
        self.finished = False

    def switch(self, phase):
        if self.finished:
            raise RuntimeError('Frame timer is already finished')
        if phase == self.current:
            return
        self.synchronize()
        now = self.clock()
        self.seconds[self.current] = self.seconds.get(self.current, 0.) + now - self.updated
        self.updated, self.current = now, phase

    @contextmanager
    def phase(self, name):
        previous = self.current
        self.switch(name)
        try:
            yield
        finally:
            self.switch(previous)

    def finish(self):
        if self.finished:
            raise RuntimeError('Frame timer is already finished')
        self.switch('finished')
        self.finished = True
        phases = dict(self.seconds)
        get = lambda key: phases.get(key, 0.)
        return dict(version=1, phases=phases,
            training_seconds=get('preparation') + get('motion') + get('birth'),
            optimization_seconds=get('motion') + get('birth'),
            evaluation_seconds=get('stage_evaluation') + get('validation'),
            checkpoint_seconds=get('checkpoint'),
            bookkeeping_seconds=get('bookkeeping'),
            frame_wall_seconds=self.updated - self.started,
            scope='Synchronized wall time. Training includes frame/data preparation, '
                  'optimization, mandatory quantization/encoding and progress metrics; '
                  'excludes heldout stage renders, validation and checkpoint writes. '
                  'Frame wall time ends before the summary JSON write and final log.')
