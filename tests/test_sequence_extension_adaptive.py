import copy
import unittest

from pipeline.sequence_extension import _validate_adaptive_birth_completion


def completed_adaptive_checkpoint(source=4, committed=2):
    updates = 230 if source else 0
    overlap = 130 if source else 0
    control = dict(source_count=source, training_population=source, population_cap=10,
                   low_alpha_newborns_removed=source-committed,
                   global_iterations=300, motion_iterations=200,
                   newborn_iterations=updates, joint_overlap_iterations=overlap,
                   final_population_iterations=100, new_alpha_iterations=updates,
                   population_screening=False, codec_trials200=0,
                   finish_render_calls=0, extra_SGD_steps=0)
    checkpoint = dict(rigid_tree_frame=dict(birth_control=dict(
        proposal_count=source, committed_population_count=committed,
        early_replacement=control)),
        early_replacement_state=dict(initialized=True, shared_steps=300,
                                     birth_updates=updates), dual_birth_state=None)
    manifest = dict(birth_count=committed, training_step_accounting=dict(
        warm_gate=70, joint=130, final_population=100, global_iterations=300,
        shared_adam_steps=300, motion_participation=200, birth_participation=updates,
        joint_overlap=overlap, extra_SGD_steps=0, topology_born_step=70,
        population_frozen_step=70, final_alpha_cleanup_step=300))
    return checkpoint, manifest


class AdaptiveCompletionTests(unittest.TestCase):
    def test_completed_no_screening_and_empty_population_are_accepted(self):
        for source, committed in ((4, 2), (4, 0), (0, 0)):
            with self.subTest(source=source, committed=committed):
                checkpoint, manifest = completed_adaptive_checkpoint(source, committed)
                _validate_adaptive_birth_completion(checkpoint, manifest, 300)

    def test_inconsistent_population_or_training_ledger_is_rejected(self):
        original, original_manifest = completed_adaptive_checkpoint()
        cases = (
            ('early_replacement_state', 'birth_updates', 229),
            ('early_replacement_state', 'shared_steps', 299),
            ('early_replacement', 'population_screening', True),
            ('early_replacement', 'training_population', 3),
            ('early_replacement', 'low_alpha_newborns_removed', 1),
            ('early_replacement', 'source_count', True),
            ('manifest', 'birth_count', 3),
            ('ledger', 'population_frozen_step', 200),
            ('ledger', 'extra_SGD_steps', 1),
        )
        for group, key, value in cases:
            with self.subTest(group=group, key=key):
                checkpoint, manifest = copy.deepcopy((original, original_manifest))
                if group == 'early_replacement_state':
                    checkpoint[group][key] = value
                elif group == 'early_replacement':
                    checkpoint['rigid_tree_frame']['birth_control'][group][key] = value
                elif group == 'manifest':
                    manifest[key] = value
                else:
                    manifest['training_step_accounting'][key] = value
                with self.assertRaises(ValueError):
                    _validate_adaptive_birth_completion(checkpoint, manifest, 300)

    def test_midframe_checkpoint_is_not_accepted(self):
        checkpoint, manifest = completed_adaptive_checkpoint()
        with self.assertRaises(ValueError):
            _validate_adaptive_birth_completion(checkpoint, manifest, 200)


if __name__ == '__main__':
    unittest.main()
