"""Frame wall-time accounting must exclude evaluation and checkpoint stalls."""
import unittest

from pipeline.timing import FrameTimer, summarize_frame_timings


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

    def test_aggregation_separates_cold_first_frame_and_ignores_reference(self):
        clock = Clock()
        timer = FrameTimer(clock.synchronize, clock)
        timer.switch('motion')
        clock.pending = 10.
        base = timer.finish()
        rows = [dict(frame=0, timing=base), dict(frame=1, timing=base),
                dict(frame=2, timing={**base, 'training_seconds': 2.})]
        result = summarize_frame_timings(rows)
        self.assertEqual(result['all_dynamic']['frame_count'], 2)
        self.assertEqual(result['all_dynamic']['mean_seconds']['training_seconds'], 6.)
        self.assertEqual(result['excluding_first_dynamic']['mean_seconds']['training_seconds'], 2.)

    def test_aggregation_does_not_substitute_legacy_combined_time_for_training(self):
        legacy = dict(frame=1, elapsed_seconds=99.)
        result = summarize_frame_timings([legacy])['all_dynamic']
        self.assertFalse(result['all_frames_measured'])
        self.assertEqual(result['missing_frames'], [1])
        self.assertIsNone(result['mean_seconds']['training_seconds'])
        self.assertIsNone(result['measured_only_mean_seconds']['training_seconds'])

    def test_partial_coverage_and_invalid_measurements_are_explicit(self):
        clock = Clock()
        timing = FrameTimer(clock.synchronize, clock).finish()
        rows = [dict(frame=1, elapsed_seconds=20.), dict(frame=2, timing=timing)]
        result = summarize_frame_timings(rows)['all_dynamic']
        self.assertEqual(result['measured_frame_count'], 1)
        self.assertIsNone(result['mean_seconds']['training_seconds'])
        self.assertEqual(result['measured_only_mean_seconds']['training_seconds'], 0.)
        with self.assertRaisesRegex(ValueError, 'unique'):
            summarize_frame_timings([rows[1], rows[1]])
        rows[1]['timing']['training_seconds'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            summarize_frame_timings(rows)


if __name__ == '__main__':
    unittest.main()
