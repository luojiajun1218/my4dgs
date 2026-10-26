"""Check the full adaptive initializer in oriented, anisotropic coordinates."""
import math
import tempfile
import unittest
from unittest.mock import patch

import torch

from pipeline.early_replacement import EarlyReplacementController
from tests.test_adaptive_birth import controller_fixture


class BirthSeedAxesTests(unittest.TestCase):
    def initialize(self, quaternion):
        original = EarlyReplacementController._early_current_old
        source = {}

        def oriented(module, step):
            raw, alpha = original(module, step)
            raw = {name: value.clone() for name, value in raw.items()}
            raw['rotation'] = quaternion.to(raw['rotation']).expand_as(raw['rotation']).clone()
            raw['logscale'][:] = raw['logscale'].new_tensor([.6, .2, 1e-4]).log()
            source.update(raw)
            return raw, alpha

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(EarlyReplacementController, '_early_current_old', oriented):
            model, _ = controller_fixture(directory)
            rows = model.births.parent_rows
            seed = {name: value.detach().clone() for name, value in model.births._seed_raw.items()}
            center = source['xyz'][rows]
            # Binding the live source must preserve the initialized world pose.
            torch.testing.assert_close(model.births.raw()['xyz'], seed['xyz'], rtol=0, atol=2e-7)
            return seed, seed['xyz'] - center, model.birth_control['proposal_count']

    def test_rotating_the_parent_rotates_the_seed_displacement(self):
        _, unrotated, count = self.initialize(torch.tensor([1., 0., 0., 0.]))
        half = math.sqrt(.5)
        _, rotated, rotated_count = self.initialize(torch.tensor([half, 0., half, 0.]))
        # A quarter turn around +Y sends local (x,y,z) to world (z,y,-x).
        expected = torch.stack((unrotated[:, 2], unrotated[:, 1], -unrotated[:, 0]), -1)
        self.assertEqual(count, rotated_count)
        self.assertGreater(count, 1)
        torch.testing.assert_close(rotated, expected, rtol=0, atol=2e-7)

    def test_a_flat_rotated_source_keeps_new_centers_inside_its_thin_axis(self):
        half = math.sqrt(.5)
        seed, offset, _ = self.initialize(torch.tensor([half, 0., half, 0.]))
        # Its thin local Z axis points along world X. This fails for a world-
        # axis jitter that uses the much larger local X scale on world X.
        self.assertLessEqual(float(offset[:, 0].abs().max()), .25e-4 + 2e-7)
        self.assertGreater(float(offset[:, 2].abs().max()), .05)
        torch.testing.assert_close(seed['logscale'].exp(),
            seed['logscale'].new_tensor([.42, .14, .7e-4]).expand_as(seed['logscale']),
            rtol=1e-6, atol=1e-8)

    def test_quaternion_sign_and_scale_do_not_change_the_seed_geometry(self):
        q = torch.tensor([.7, .2, -.3, .6])
        reference, _, count = self.initialize(q)
        for equivalent in (-q, 3*q):
            with self.subTest(quaternion=equivalent.tolist()):
                seed, _, other_count = self.initialize(equivalent)
                self.assertEqual(count, other_count)
                torch.testing.assert_close(seed['xyz'], reference['xyz'], rtol=0, atol=2e-7)
                for name in ('logscale', 'opacity', 'sh_dc', 'sh_rest'):
                    torch.testing.assert_close(seed[name], reference[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
