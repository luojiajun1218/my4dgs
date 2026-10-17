import json
import random
from itertools import cycle
from functools import partial
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from torch.utils.data import ConcatDataset, DataLoader, RandomSampler, SequentialSampler, Subset

from libgs.data import Dataset, SceneInfo, load_scene
from libgs.pipeline import DataConfig as BaseDataConfig
from libgs.pipeline import DataModule as BaseDataModule
from libgs.data.utils.fetch import Fetcher


SELECTIVE_PAIR_WORKER_TIMEOUT_SECONDS = 30


def _collate_single_view(batch, expected_image_size=None):
    view = batch[0]
    if expected_image_size is not None:
        width, height = expected_image_size
        if tuple(view.image.shape[-2:]) != (height, width):
            raise ValueError(f'Loaded image size {tuple(view.image.shape[-2:])} '
                             f'does not match expected height/width {(height, width)}')
    return view


class PairedFetcher(Fetcher):
    """Same cache cycling as Fetcher, with a private camera-order RNG."""
    def __init__(self, loader, cache, move_to_device, seed):
        self.order_rng = random.Random(seed)
        super().__init__(loader, cache, move_to_device)

    def next(self):
        item = next(self.loader_iter)
        if self.move_to_device:
            item = self.move_to_device(item)
        if self.cache and len(self.cached_items) < self.num_items:
            self.cached_items.append(item)
            if len(self.cached_items) == self.num_items:
                if isinstance(self.loader.sampler, RandomSampler):
                    self.order_rng.shuffle(self.cached_items)
                self.loader_iter = cycle(self.cached_items)
                self.cached_items = []
        return item


@dataclass
class DataConfig(BaseDataConfig):
    root: str = ""
    resolution: int = -1
    expected_image_size: Optional[List[int]] = None  # [width, height] after loading
    white_background: bool = False
    split_train_test: bool = True
    shuffle: bool = True
    training_view_seed: Optional[int] = None
    canonical_training_view_order: bool = False
    training_camera_names: Optional[List[str]] = None
    validation_camera_names: Optional[List[str]] = None
    eval_train: bool = True
    first_frame_eval_train: Optional[bool] = None
    eval_test: bool = True
    extra_dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    extra_dataloader_kwargs: Dict[str, Any] = field(default_factory=dict)


class DataModule(BaseDataModule):
    scene: SceneInfo
    eval_names: List[str]
    current_frame: int = -1

    def __init__(self, config: DataConfig):
        super().__init__(config)
        expected = config.expected_image_size
        if expected is not None and (len(expected) != 2 or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in expected)):
            raise ValueError('expected_image_size must contain positive integer width/height')

        args = (Path(config.root), config.resolution)
        kwargs = {
            "split_train_test": config.split_train_test,
            "white_background": config.white_background,
            **config.extra_dataset_kwargs,
        }
        self.scene = load_scene(*args, **kwargs)
        self._selective_pairs_cache = None

    @property
    def cameras_extent(self) -> float:
        return self.scene.nerf_normalization["radius"]

    @property
    def num_frames(self) -> int:
        return self.scene.train_dataset.num_frames

    def setup(self, save_dir: Optional[Path] = None):
        self.scene.train_dataset.setup()
        if self.scene.test_dataset is not None:
            self.scene.test_dataset.setup()
        if self.config.shuffle:  # Multi-res consistent random shuffling
            self.scene.train_dataset.shuffle()
            if self.scene.test_dataset is not None:
                self.scene.test_dataset.shuffle()
        if save_dir:
            self.save_scene_info(save_dir)

    def on_save_checkpoint(self, ckpt: dict):
        ckpt["current_frame"] = self.current_frame

    def on_load_checkpoint(self, ckpt: dict):
        if "current_frame" in ckpt:
            self.current_frame = ckpt["current_frame"]

    def save_scene_info(self, save_dir: Path):
        if not self.scene.ply_path.exists():
            print(f"[WARNING] ply file not exists: {self.scene.ply_path}")
            return

        input_ply_path = save_dir / "input.ply"
        input_ply_path.write_bytes(self.scene.ply_path.read_bytes())

        json_cams, camlist = [], []
        if self.scene.test_dataset:
            camlist.extend(self.scene.test_dataset.items)
        if self.scene.train_dataset:
            camlist.extend(self.scene.train_dataset.items)

        json_cams = [camera.to_json(id=id) for id, camera in enumerate(camlist)]
        with (save_dir / "cameras.json").open("w") as file:
            json.dump(json_cams, file)

    def get_train_dataset(self, scale: float = 1.0) -> Dataset:
        train_dataset = self.scene.train_dataset
        if self.current_frame >= 0:
            train_dataset = train_dataset.get_frame_dataset(self.current_frame)

        if scale != 1:
            train_dataset = deepcopy(train_dataset)
        train_dataset.resolution_scale = scale
        return train_dataset

    def get_test_dataset(self, scale: float = 1.0) -> Optional[Dataset]:
        test_dataset = self.scene.test_dataset
        if self.current_frame >= 0:
            test_dataset = test_dataset.get_frame_dataset(self.current_frame)
        if test_dataset:
            if scale != 1:
                test_dataset = deepcopy(test_dataset)
            test_dataset.resolution_scale = scale
        return test_dataset

    @staticmethod
    def _camera_indices(dataset, names: Optional[Sequence[str]]) -> Optional[List[int]]:
        if names is None:
            return None
        requested = list(names)
        if not requested or len(requested) != len(set(requested)):
            raise ValueError('Camera-name split must be nonempty and contain no duplicates')
        camera_to_index = {}
        for index, item in enumerate(dataset.items):
            name = item.path.name
            if name in camera_to_index:
                raise ValueError(f'Training camera filenames must be unique: {name}')
            camera_to_index[name] = index
        missing = [name for name in requested if name not in camera_to_index]
        if missing:
            raise ValueError(f'Camera-name split contains missing views: {missing}')
        return [camera_to_index[name] for name in requested]

    def _configured_camera_split(self):
        training = getattr(self.config, 'training_camera_names', None)
        validation = getattr(self.config, 'validation_camera_names', None)
        if training is None and validation is None:
            return
        if training is None or validation is None:
            raise ValueError('Training and validation camera names must be configured together')
        overlap = set(training) & set(validation)
        if overlap:
            raise ValueError(f'Training and validation camera splits overlap: {sorted(overlap)}')

    def _view_collate(self):
        return partial(_collate_single_view,
                       expected_image_size=getattr(self.config, 'expected_image_size', None))

    def selective_training_pairs(self, names, device, camera_atol):
        """Explicit filename selection is stable across dataset shuffling/resume."""
        if self.current_frame <= 0:
            raise ValueError('Selective comparisons require a previous frame')
        cache_key = (self.current_frame, tuple(names), str(device), camera_atol)
        cached = getattr(self, '_selective_pairs_cache', None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        # Drop the previous frame's GPU images before loading this frame's pairs.
        self._selective_pairs_cache = None
        current = self.scene.train_dataset.get_frame_dataset(self.current_frame)
        previous = self.scene.train_dataset.get_frame_dataset(self.current_frame - 1)
        def mapping(dataset):
            result = {}
            for index, item in enumerate(dataset.items):
                if item.path.name in result:
                    raise ValueError('Training camera filenames must be unique within a frame')
                result[item.path.name] = index
            return result
        current_map, previous_map = mapping(current), mapping(previous)
        for name in names:
            if name not in current_map or name not in previous_map:
                raise ValueError(f'Comparison camera {name} is not in both training frames')
        dataset = ConcatDataset([Subset(current, [current_map[name] for name in names]),
                                 Subset(previous, [previous_map[name] for name in names])])
        kwargs = dict(self.config.extra_dataloader_kwargs)
        if kwargs.get('num_workers', 0) < 1:
            raise ValueError('Selective paired views require data.extra_dataloader_kwargs.num_workers >= 1')
        # libgs full-resolution downsampling expects get_worker_info().id.
        # Keep reading inside workers, using the same worker settings as training.
        kwargs.update(batch_size=1, shuffle=False, drop_last=False,
                      sampler=SequentialSampler(dataset), collate_fn=self._view_collate(),
                      timeout=SELECTIVE_PAIR_WORKER_TIMEOUT_SECONDS)
        loader = DataLoader(dataset, **kwargs)
        loader_iter = iter(loader)
        views = []
        try:
            for _ in range(len(dataset)):
                views.append(next(loader_iter).to(device))
        finally:
            # This one-shot loader must release its workers on success and error.
            shutdown_workers = getattr(loader_iter, '_shutdown_workers', None)
            if shutdown_workers is not None:
                shutdown_workers()
        count = len(names)
        pairs = []
        for now, old in zip(views[:count], views[count:]):
            if now.image.shape != old.image.shape or not torch.allclose(
                    now.full_proj_transform, old.full_proj_transform, rtol=0, atol=camera_atol):
                raise ValueError(f'Image evidence needs matched stationary camera: {name}')
            pairs.append((now, old))
        self._selective_pairs_cache = (cache_key, pairs)
        return pairs

    def train_dataloader(
        self,
        scale: float = 1.0,
        random: bool = True,
        indices: Optional[Sequence[int]] = None,
    ) -> DataLoader:
        dataset = self.get_train_dataset(scale)
        self._configured_camera_split()
        if indices is None:
            indices = self._camera_indices(
                dataset, getattr(self.config, 'training_camera_names', None))
        seed = self.config.training_view_seed
        if getattr(self.config, 'canonical_training_view_order', False):
            if random and seed is None:
                raise ValueError('Canonical training order requires training_view_seed')
            if not hasattr(dataset, 'items'):
                raise ValueError('Canonical training order requires camera metadata')
            selected = (range(len(dataset)) if indices is None or len(indices) == 0
                        else indices)
            # The parent dataset shuffles all frames before filtering this one.
            # Sort a Subset index, preserving its cached views and source order.
            def camera_key(index):
                item = dataset.items[index]
                frame = getattr(item, 'frame', self.current_frame)
                path = Path(item.path)
                return frame, path.name, str(path)

            indices = sorted(selected, key=camera_key)
        if indices:
            dataset = Subset(dataset, indices)
        collate_fn = self._view_collate()
        kwargs = dict(num_workers=8, pin_memory=True, collate_fn=collate_fn)
        generator = (torch.Generator().manual_seed(seed + max(self.current_frame, 0))
                     if seed is not None else None)
        if random:
            kwargs["sampler"] = RandomSampler(dataset, generator=generator)
        else:
            kwargs["sampler"] = SequentialSampler(dataset)
        kwargs = {**kwargs, **self.config.extra_dataloader_kwargs}
        if generator is not None:
            kwargs['generator'] = torch.Generator().manual_seed(seed + max(self.current_frame, 0))
        return DataLoader(dataset, **kwargs)

    def test_dataloader(self, scale: float = 1.0) -> Optional[DataLoader]:
        if dataset := self.get_test_dataset(scale):
            collate_fn = self._view_collate()
            kwargs = dict(num_workers=8, pin_memory=True, collate_fn=collate_fn)
            kwargs = {**kwargs, **self.config.extra_dataloader_kwargs}
            return DataLoader(dataset, **kwargs)

    def validation_dataloader(self, scale: float = 1.0) -> DataLoader:
        dataset = self.get_train_dataset(scale)
        self._configured_camera_split()
        validation_names = getattr(self.config, 'validation_camera_names', None)
        if validation_names is None:
            raise ValueError('validation_camera_names must be set for a validation loader')
        indices = self._camera_indices(dataset, validation_names)
        dataset = Subset(dataset, indices)
        kwargs = dict(num_workers=8, pin_memory=True, collate_fn=self._view_collate())
        kwargs = {**kwargs, **self.config.extra_dataloader_kwargs}
        kwargs.pop('shuffle', None)
        kwargs['sampler'] = SequentialSampler(dataset)
        return DataLoader(dataset, **kwargs)

    def val_dataloader(self, scale: float = 1.0) -> List[DataLoader]:
        datasets, eval_names = [], []
        eval_train = self.config.eval_train
        if self.current_frame == 0 and self.config.first_frame_eval_train is not None:
            eval_train = self.config.first_frame_eval_train
        if getattr(self.config, 'validation_camera_names', None) is not None:
            datasets.append(self.validation_dataloader(scale))
            eval_names.append('validation')
        elif eval_train:
            datasets.append(self.train_dataloader(scale, random=False))
            eval_names.append("train")
        if self.config.eval_test and (dataset := self.test_dataloader(scale)):
            datasets.append(dataset)
            eval_names.append("test")
        self.eval_names = eval_names
        return datasets
