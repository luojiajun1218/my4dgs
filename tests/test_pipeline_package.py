"""Flattened package entry points, historical class identity, and CUDA sources."""

import importlib
from pathlib import Path
import pickle
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class PipelinePackageTests(unittest.TestCase):
    def run_python(self, code):
        return subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                              check=True, capture_output=True, text=True)

    def test_old_and_new_imports_share_classes_in_either_import_order(self):
        names = ['data', 'module', 'trainer', 'sequence_extension', 'model',
                 'model.gaussian', 'model.rigid_tree_codec',
                 'model.rigid_tree_birth_codec', 'model.legacy_opacity_codec',
                 'model.birth_quantization', 'model.rigid_residual_tree',
                 'model.spatial_gate', 'model.spatial_gated_motion',
                 'model.contribution_rasterizer', 'model.contribution_renderer']
        for legacy_first in (True, False):
            with self.subTest(legacy_first=legacy_first):
                self.run_python(f'''
import importlib
names = {names!r}
for name in names:
    canonical = 'pipeline.' + name
    for prefix in ('pipeline.ReCon-GS.', 'pipeline.my4dgs.'):
        first, second = (prefix + name, canonical) if {legacy_first!r} else (canonical, prefix + name)
        a = importlib.import_module(first)
        b = importlib.import_module(second)
        assert a is b, (first, second)
        assert a.__spec__.name == canonical, a.__spec__.name
''')

    def test_pure_model_import_does_not_load_the_training_entry(self):
        self.run_python('''
import importlib, sys
importlib.import_module('pipeline.model.rigid_tree_codec')
assert 'pipeline.entry' not in sys.modules
assert 'pipeline.module' not in sys.modules
''')

    def test_libgs_loads_both_names_and_historical_pickles(self):
        from libgs.pipeline.loader import load_pipeline
        config, pipeline = load_pipeline('my4dgs')
        self.assertEqual((config, pipeline), load_pipeline('ReCon-GS'))
        self.assertEqual((config, pipeline), load_pipeline('pipeline'))
        self.assertIs(pickle.loads(b'cpipeline.ReCon-GS\nConfig\n.'), config)
        codec = importlib.import_module('pipeline.model.rigid_tree_codec')
        self.assertIs(pickle.loads(
            b'cpipeline.ReCon-GS.model.rigid_tree_codec\nDecodedState\n.'), codec.DecodedState)

    def test_cli_print_supports_new_default_and_previous_pipeline_name(self):
        outputs = []
        for flags in ([], ['--pipeline=ReCon-GS']):
            result = subprocess.run([sys.executable, 'main.py',
                '--config=config/fs_spatial_gate_robustmedian_200_first30_20261016.yaml',
                '--print', *flags], cwd=ROOT, check=True, capture_output=True, text=True)
            outputs.append(result.stdout)
        self.assertEqual(outputs[0], outputs[1])
        self.assertIn('initialization: robust_median', outputs[0])

    def test_cuda_extension_sources_resolve_after_the_directory_move(self):
        observed = importlib.import_module('pipeline.model.contribution_rasterizer')
        observed._backend.cache_clear()
        try:
            with patch('torch.utils.cpp_extension.load', return_value=object()) as load:
                observed._backend()
            kwargs = load.call_args.kwargs
            expected = ROOT / 'extensions/observed_rasterizer'
            for source in kwargs['sources']:
                path = Path(source)
                self.assertTrue(path.is_relative_to(expected), source)
                self.assertTrue(path.is_file(), source)
            self.assertTrue(Path(kwargs['extra_include_paths'][0]).is_dir())
        finally:
            observed._backend.cache_clear()


if __name__ == '__main__':
    unittest.main()
