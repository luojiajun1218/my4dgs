"""测量已解码帧的纯渲染 FPS 和 GPU 播放器的完整播放 FPS。

复用 frame_stream_player 的 GPU 播放器，使用预加载的首帧 cam00 分别计时：
纯渲染只包含已解码状态的渲染；完整播放包含读取码包、解码、更新和渲染。
本文件负责速度统计及报告，不负责训练、画质评估或码流实现。
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.model.frame_stream_format import DecodedState, RAW_FIELDS, _validate_state
from scripts.frame_stream_player import GPUStream

def _check_gpu_exclusive():
    """Reject other compute processes; an unavailable GPU audit is reported."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=False, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        raise RuntimeError('Cannot verify exclusive GPU access with nvidia-smi') from error
    if result.returncode:
        raise RuntimeError(f'Cannot verify exclusive GPU access: {result.stderr.strip()}')
    for line in result.stdout.splitlines():
        pid = line.strip()
        if not pid.isdecimal():
            raise ValueError(f'Unexpected GPU process ID: {pid}')
        if int(pid) != os.getpid():
            raise RuntimeError(f'GPU is in use by process {pid}')


def _load_config(run):
    from omegaconf import OmegaConf
    from libgs.utils.config import to_structured
    from pipeline import Config
    saved = OmegaConf.to_container(OmegaConf.load(run / 'config.yaml'), resolve=True)
    config, extra = to_structured(OmegaConf.create(saved), Config)
    if extra:
        raise ValueError(f'Unrecognized run configuration: {extra}')
    return config


def _cam00(datamodule, frame, device):
    datamodule.current_frame = frame
    dataset = datamodule.get_test_dataset()
    if dataset is None:
        raise ValueError(f'Frame {frame} has no test dataset')
    indices = [i for i, item in enumerate(dataset.items) if Path(str(item.path)).name == 'cam00.png']
    if len(indices) != 1:
        raise ValueError(f'Frame {frame} needs exactly one cam00.png')
    return dataset[indices[0]].to(device), len(dataset)


def _make_module(config, device):
    """Create the ordinary renderer without creating a training optimizer."""
    from libgs.renderer import Renderer
    from pipeline.data import DataModule
    from pipeline.model.gaussian import GaussianModel
    from pipeline.module import Module

    config.data.shuffle = False
    datamodule = DataModule(config.data)
    datamodule.setup()
    module = Module(config.module, datamodule=datamodule)
    module.trainer = SimpleNamespace(device=device, global_step=config.module.rigid_tree.motion_steps)
    module.gaussians = GaussianModel(config.module.sh_degree)
    module.gaussians.active_sh_degree = config.module.sh_degree
    module.renderer = Renderer(config.module.renderer, module.gaussians)
    color = 1. if config.data.white_background else 0.
    module.register_buffer('background', torch.full((3,), color, device=device))
    module.frame_prepared = False
    module.motion_committed = False
    module.frame_finalized = True
    module.eval()
    return module, datamodule


def _load_reference(path, region_count=None):
    with np.load(path, allow_pickle=False) as packet:
        raw = {name: torch.from_numpy(packet[name].copy()) for name in RAW_FIELDS}
        state = DecodedState.from_reference(packet)
        if int(packet['region_count']) != len(packet['parents']):
            raise ValueError('Reference region count differs from stored tree')
        if region_count is not None and int(packet['region_count']) != region_count:
            raise ValueError('Reference region count differs from run config')
    _validate_state(state)
    return state, raw


def _capacity(stream, first_count, last_frame):
    capacity = first_count
    for frame in range(1, last_frame + 1):
        manifest = json.loads((stream / f'frame_{frame:03d}.json').read_text())
        births = manifest.get('birth_count')
        if type(births) is not int or births < 0:
            raise ValueError(f'Invalid birth count for frame {frame}')
        capacity += births
    return capacity


def _check_player_state(player):
    """Compare GPU renderer inputs with the player's decoded CPU reference."""
    actual = player.capture_raw()
    expected = player._portable_raw
    for name in RAW_FIELDS:
        if actual[name].shape != expected[name].shape or not bool(torch.isfinite(actual[name]).all()):
            raise ValueError(f'Invalid GPU {name} at frame {player.frame}')
        if actual[name].numel() and float((actual[name] - expected[name]).abs().max()) > 1e-5:
            raise ValueError(f'GPU {name} differs from decoded CPU state at frame {player.frame}')
    player.verify_render_inputs(expected, atol=1e-5)


@torch.no_grad()
def benchmark(run, last_frame, warmups, renders, rounds):
    _check_gpu_exclusive()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for FPS measurement')
    device = torch.device('cuda')
    config = _load_config(run)
    module, data = _make_module(config, device)
    if not 1 <= last_frame < data.num_frames:
        raise ValueError('Last frame must be a dynamic frame within the dataset')
    stream = run / 'frame_storage' / 'rigid_tree'
    state, raw = _load_reference(stream / 'reference_frame0.npz')
    if state.frame_index != 0:
        raise ValueError('Playback must start from frame 0')
    capacity = _capacity(stream, len(state.xyz), last_frame)
    view, _ = _cam00(data, 0, device)
    expected_size = config.data.expected_image_size
    if expected_size is not None and tuple(view.image.shape) != (3, expected_size[1], expected_size[0]):
        raise ValueError('Loaded camera image does not match the configured resolution')

    # Advance outside the timed region, then time only rendering each decoded frame.
    player = GPUStream(state, raw, capacity=capacity)
    render_rows = []
    for frame in range(last_frame + 1):
        if frame:
            player.advance_frame(stream, frame)
        _check_player_state(player)
        module.renderer.gaussians = player.view
        for _ in range(warmups):
            module.forward(view)['render'].clamp(0., 1.)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(renders):
            module.forward(view)['render'].clamp(0., 1.)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        render_rows.append(dict(frame=frame, seconds=seconds, fps=renders / seconds))
    module.renderer.gaussians = module.gaussians
    del player
    _check_gpu_exclusive()

    # GPUStream.advance_frame reads and decodes packets and updates GPU state.
    playback_rows = []
    for repeat in range(rounds):
        _check_gpu_exclusive()
        player = GPUStream(state, raw, capacity=capacity)
        module.renderer.gaussians = player.view
        for _ in range(warmups):
            module.forward(view)['render'].clamp(0., 1.)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for frame in range(1, last_frame + 1):
            module.renderer.gaussians = player.advance_frame(stream, frame)
            module.forward(view)['render'].clamp(0., 1.)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        _check_player_state(player)
        _check_gpu_exclusive()
        playback_rows.append(dict(round=repeat, seconds=seconds, fps=last_frame / seconds))
        module.renderer.gaussians = module.gaussians
        del player
    return dict(
        run=str(run), camera='frame 0 cam00', image_shape=list(view.image.shape), frames=last_frame + 1,
        render=dict(fps=renders * len(render_rows) / sum(row['seconds'] for row in render_rows),
                    renders_per_frame=renders, rows=render_rows,
                    scope='decoded GPU state; Module.forward plus clamp; excludes loading and decoding'),
        playback=dict(fps=last_frame * rounds / sum(row['seconds'] for row in playback_rows),
                      dynamic_frames_per_round=last_frame, rounds=playback_rows,
                      scope='packet reads, CPU decode and checks, GPU updates, Module.forward plus clamp; '
                            'excludes reference and camera loading'),
        gpu=torch.cuda.get_device_name(device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--last-frame', type=int, help='Last dynamic frame; defaults to the dataset last frame')
    parser.add_argument('--warmups', type=int, default=10)
    parser.add_argument('--renders', type=int, default=50)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if min(args.warmups, args.renders, args.rounds) < 1:
        parser.error('warmups, renders and rounds must be positive')
    run = args.run.resolve()
    if args.output.resolve().is_relative_to(run):
        raise ValueError('Write FPS results outside the source experiment directory')
    config = _load_config(run)
    last_frame = args.last_frame
    if last_frame is None:
        last_frame = int(config.data.extra_dataset_kwargs['num_frames']) - 1
    result = benchmark(run, last_frame, args.warmups, args.renders, args.rounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(f"Render: {result['render']['fps']:.2f} FPS; playback: {result['playback']['fps']:.2f} FPS")


if __name__ == '__main__':
    main()
