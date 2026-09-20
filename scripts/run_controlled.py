"""Run either source tree with identical per-frame view streams and audit logs.

The reference implementation is imported unchanged. Instrumentation adds an
independent sampler RNG, per-frame seeds, metrics, and reference motion exports.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import runpy
import sys
import time

import numpy as np
import torch
from archive_run_sources import archive_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args, remaining = parser.parse_known_args()
    source = args.source.resolve()
    os.chdir(source)
    sys.path.insert(0, str(source))
    package = 'pipeline.ReCon-GS'
    data_cls = importlib.import_module(package + '.data').DataModule
    trainer_cls = importlib.import_module(package + '.trainer').Trainer
    module_cls = importlib.import_module(package + '.module').Module
    original_loader = data_cls.train_dataloader
    original_frame = trainer_cls.frame_training_loop
    original_train = module_cls.training_step
    original_validation = module_cls.validation_step
    original_post = module_cls.post_training_step

    def append(path, row):
        with path.open('a') as stream:
            stream.write(json.dumps(row) + '\n')

    def controlled_loader(self, *pos, **kw):
        loader = original_loader(self, *pos, **kw)
        if isinstance(loader.sampler, torch.utils.data.RandomSampler):
            loader.sampler.generator = torch.Generator().manual_seed(
                args.seed + 100000 + max(0, self.current_frame))
        loader.generator = torch.Generator().manual_seed(
            args.seed + 200000 + max(0, self.current_frame))
        return loader

    def controlled_frame(self):
        seed = args.seed + 1000 * max(0, self.current_frame)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        manifest_path = self.output_dir / 'controlled_manifest.json'
        if not manifest_path.exists():
            files = [source / 'main.py', *sorted((source / 'pipeline').rglob('*.py'))]
            manifest = dict(source=str(source), reference=args.reference,
                            seed=args.seed, argv=remaining,
                            frame_seed='seed + 1000 * frame',
                            sampler_seed='seed + 100000 + frame',
                            source_sha256={str(p.relative_to(source)):
                                           hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in files})
            manifest_path.write_text(json.dumps(manifest, indent=2))
            archive_sources(self.output_dir)
        frame = self.current_frame
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original_frame(self)
        torch.cuda.synchronize()
        append(self.output_dir / 'frame_times.jsonl',
               dict(frame=frame, wall_seconds=time.perf_counter() - started))
        return result

    def controlled_train(self, viewpoint, current_step):
        append(self.output_dir / 'training_views.jsonl',
               dict(frame=self.current_frame, step=current_step, path=str(viewpoint.path)))
        return original_train(self, viewpoint, current_step)

    def controlled_validation(self, viewpoint, idx, loader_idx):
        metrics = original_validation(self, viewpoint, idx, loader_idx)
        append(self.output_dir / 'evaluation.jsonl',
               dict(frame=self.current_frame, step=self.global_step,
                    split=self.datamodule.eval_names[loader_idx], path=str(viewpoint.path),
                    **{k: float(v.mean()) if torch.is_tensor(v) else float(v)
                       for k, v in metrics.items()}))
        return metrics

    def controlled_post(self, render_pkg):
        if (args.reference and not self.trainer.is_first_frame and
                self.global_step == self.deform.config.num_stage1_steps):
            # Upstream applies deformation BEFORE its final optimizer update.
            # Export the exact pre-update state used for that application.
            directory = self.output_dir / 'frame_storage'
            directory.mkdir(exist_ok=True)
            path = directory / f'motion_{self.current_frame:03d}.npz'
            np.savez_compressed(path, representation=np.array('recongs_original'),
                                delta=self.deform.delta.detach().cpu().numpy(),
                                index=self.deform.index.int().cpu().numpy())
        result = original_post(self, render_pkg)
        if args.reference and not self.trainer.is_first_frame and self.trainer.is_last_step:
            directory = self.output_dir / 'frame_storage'
            path = directory / f'incremental_{self.current_frame:03d}.ply'
            self.gaussians_incr.save_ply(path)
            motion = directory / f'motion_{self.current_frame:03d}.npz'
            stats = dict(frame=self.current_frame, motion_bytes=motion.stat().st_size,
                         incremental_bytes=path.stat().st_size,
                         total_bytes=motion.stat().st_size + path.stat().st_size,
                         anchors=self.deform.delta.shape[0])
            (directory / f'frame_{self.current_frame:03d}.json').write_text(
                json.dumps(stats, indent=2))
        return result

    data_cls.train_dataloader = controlled_loader
    trainer_cls.frame_training_loop = controlled_frame
    module_cls.training_step = controlled_train
    module_cls.validation_step = controlled_validation
    module_cls.post_training_step = controlled_post
    sys.argv = [str(source / 'main.py'), *remaining]
    runpy.run_path(str(source / 'main.py'), run_name='__main__')


if __name__ == '__main__':
    main()
