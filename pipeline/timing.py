"""Partition synchronized frame wall time without synchronizing every step."""
from contextlib import contextmanager
import math
import time


def summarize_frame_timings(reports):
    """Report measured phases without relabeling legacy combined elapsed time."""
    dynamic = [row for row in reports if row['frame'] > 0]
    ids = [row['frame'] for row in dynamic]
    if len(set(ids)) != len(ids):
        raise ValueError('Frame timing reports must have unique frame IDs')
    fields = ('training_seconds', 'optimization_seconds', 'evaluation_seconds',
              'checkpoint_seconds', 'bookkeeping_seconds', 'frame_wall_seconds')

    def group(rows):
        measured = []
        missing = []
        for row in rows:
            timing = row.get('timing') or {}
            if timing.get('version') not in (1, 2) or any(key not in timing for key in fields):
                missing.append(row['frame'])
                continue
            values = {key: float(timing[key]) for key in fields}
            if any(not math.isfinite(value) or value < 0 for value in values.values()):
                raise ValueError('Measured frame times must be finite and nonnegative')
            measured.append(values)
        complete = bool(rows) and not missing
        return dict(frame_count=len(rows), measured_frame_count=len(measured),
            all_frames_measured=complete, missing_frames=sorted(missing),
            mean_seconds={key: (sum(row[key] for row in measured) / len(measured)
                                if complete else None) for key in fields},
            measured_only_mean_seconds={key: (sum(row[key] for row in measured) / len(measured)
                                              if measured else None) for key in fields})

    model_rows = [row for row in dynamic if row.get('timing', {}).get('version') == 2]
    model_group = group(model_rows)
    model_group.update(frame_count=len(dynamic),
                       all_frames_measured=bool(dynamic) and len(model_rows) == len(dynamic),
                       missing_frames=sorted(set(ids) - {row['frame'] for row in model_rows}))
    if not model_group['all_frames_measured']:
        model_group['mean_seconds'] = {key: None for key in fields}
    return dict(all_dynamic=group(dynamic),
        excluding_first_dynamic=group([row for row in dynamic if row['frame'] > 1]),
        model_training=model_group,
        scope='Measured synchronized phase times only; no fallback to legacy '
              'elapsed_seconds, which combines training and evaluation. '
              'Excluding-first group omits frame 1, which may include cold startup.')


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
        model = get('motion') + get('birth') + get('refit')
        preparation = get('preparation') + get('model_setup')
        return dict(version=2, phases=phases,
            training_seconds=model,
            optimization_seconds=model,
            preparation_seconds=preparation,
            data_loading_seconds=get('data_loading'),
            encoding_seconds=get('encoding'),
            diagnostics_seconds=get('diagnostics'),
            training_loop_seconds=(model + preparation + get('data_loading') +
                                   get('encoding') + get('diagnostics')),
            evaluation_seconds=get('stage_evaluation') + get('validation'),
            checkpoint_seconds=get('checkpoint'),
            bookkeeping_seconds=get('bookkeeping'),
            frame_wall_seconds=self.updated - self.started,
            scope='Synchronized wall time. Training is model forward/backward, '
                  'optimizer updates and training regularization/quantization. '
                  'Frame setup, image loading/transfers, final stream encoding, '
                  'progress diagnostics, heldout renders, validation and checkpoint '
                  'writes are separate. Offline video/COLMAP preprocessing is outside this timer. '
                  'Frame wall time ends before the summary JSON write and final log.')
