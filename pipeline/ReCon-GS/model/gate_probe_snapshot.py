"""Immutable pre-update stage snapshots for independent gate/precision replay."""

from dataclasses import asdict
import copy
import hashlib
from pathlib import Path
import random

import numpy as np
import torch

from .storage_aware_binary import StorageAwareBinaryConfig, StorageAwareBinaryMotion


SNAPSHOT_VERSION = 'gate_joint_pre_update_v1'


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {name: cpu_copy(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    return copy.deepcopy(value)


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def tensor_fingerprint(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode('utf-8'))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def view_fingerprint(view):
    result = dict(camera=view.path.name, image_sha256=tensor_fingerprint(view.image),
                  image_shape=list(view.image.shape))
    for name in ('full_proj_transform', 'world_view_transform', 'camera_center'):
        value = getattr(view, name, None)
        if torch.is_tensor(value):
            result[name] = tensor_fingerprint(value)
    return result


def save_probe_snapshot(path, *, motion, frame, step, diagnostic_config, render_state,
                        data_config, module_config, renderer_config, background,
                        first_frame_gs_count, samples, view_plans, views,
                        training_camera, source_rng, gradient_components=None, purpose='training_probe'):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite stage snapshot: {path}')
    packet = dict(snapshot_version=SNAPSHOT_VERSION, frame=frame, step=step, purpose=purpose,
        context='online_after_backward_before_optimizer_update', motion=motion.capture(),
        parameter_gradients={name: parameter.grad for name, parameter in motion.named_parameters()},
        requires_grad={name: parameter.requires_grad for name, parameter in motion.named_parameters()},
        diagnostic_config=asdict(diagnostic_config), render_state=render_state,
        data_config=data_config, module_config=module_config, renderer_config=renderer_config,
        background=background, first_frame_gs_count=first_frame_gs_count,
        samples=samples, view_plans=view_plans, training_camera=training_camera,
        source_rng=source_rng, gradient_components=gradient_components,
        views=[view_fingerprint(view) for view in views],
        dataset_root_resolved=str(Path(data_config['root']).resolve()),
        semantics='Full frozen GS raw attributes, source motion/tree/optimizer/logits and gradients; '
            'external training images/cameras are fingerprinted, not copied. All local candidates '
            'use saved private camera plans. Replay never advances the main training RNG or optimizer.')
    packet = cpu_copy(packet)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(packet, stream)
    return path


def restore_probe_motion(packet, device):
    if packet.get('snapshot_version') != SNAPSHOT_VERSION:
        raise ValueError('Unsupported gate stage snapshot version')
    config = StorageAwareBinaryConfig(**packet['motion']['config'])
    motion = StorageAwareBinaryMotion(config)

    def move(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {name: move(item) for name, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    motion.restore(move(packet['motion']))
    for name, parameter in motion.named_parameters():
        parameter.requires_grad_(packet['requires_grad'][name])
        gradient = packet['parameter_gradients'][name]
        parameter.grad = None if gradient is None else gradient.to(parameter).clone()
    return motion


def verify_snapshot_views(packet, views):
    expected = {entry['camera']: entry for entry in packet['views']}
    actual = {view.path.name: view_fingerprint(view) for view in views}
    if actual != expected:
        raise ValueError('Replay images or camera transforms differ from the frozen snapshot dependencies')
