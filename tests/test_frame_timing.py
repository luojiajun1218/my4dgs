"""Frame wall-time accounting must exclude evaluation and checkpoint stalls."""
import unittest

from pipeline.timing import FrameTimer


class Clock:
    def __init__(self):
        self.now = 0.
        self.pending = 0.
        self.synchronizations = 0

    def __call__(self):
        return self.now

    def synchronize(self):
        self.now += self.pending
        self.pending = 0.
        self.synchronizations += 1


class FrameTimingTests(unittest.TestCase):
    def test_async_work_is_charged_to_the_phase_that_launched_it(self):
        clock = Clock()
        clock.pending = 99.  # Outstanding work from the previous frame is excluded.
        timer = FrameTimer(clock.synchronize, clock)
        clock.now += 1.
        timer.switch('motion')
        clock.pending = 7.
        with timer.phase('stage_evaluation'):
            clock.pending = 3.
        clock.pending = 2.
        with timer.phase('checkpoint'):
            clock.now += 5.
        timer.switch('birth')
        clock.pending = 4.
        timer.switch('validation')
        clock.pending = 6.
        timer.switch('bookkeeping')
        clock.now += .5
        timing = timer.finish()
        self.assertEqual(timing['training_seconds'], 14.)
        self.assertEqual(timing['optimization_seconds'], 13.)
        self.assertEqual(timing['evaluation_seconds'], 9.)
        self.assertEqual(timing['checkpoint_seconds'], 5.)
        self.assertEqual(timing['frame_wall_seconds'], 28.5)
        self.assertEqual(sum(timing['phases'].values()), timing['frame_wall_seconds'])

    def test_same_phase_does_not_add_per_step_synchronization(self):
        clock = Clock()
        timer = FrameTimer(clock.synchronize, clock)
        timer.switch('motion')
        before = clock.synchronizations
        for _ in range(200):
            timer.switch('motion')
            clock.pending += 1.
        self.assertEqual(clock.synchronizations, before)
        self.assertEqual(timer.finish()['training_seconds'], 200.)
        with self.assertRaisesRegex(RuntimeError, 'finished'):
            timer.finish()

    def test_context_restores_phase_on_exception(self):
        clock = Clock()
        timer = FrameTimer(clock.synchronize, clock)
        timer.switch('motion')
        with self.assertRaises(ValueError):
            with timer.phase('checkpoint'):
                clock.now += 3.
                raise ValueError('save failed')
        self.assertEqual(timer.current, 'motion')
        self.assertEqual(timer.finish()['checkpoint_seconds'], 3.)


if __name__ == '__main__':
    unittest.main()
