"""Load a fresh native checkpoint with my4dgs and verify held-out frame0 rendering."""

import argparse
import json
import os
from pathlib import Path
import re
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from libgs.data import load_scene
from libgs.metric import lpips, psnr, ssim
from libgs.renderer import Renderer, RendererConfig
from pipeline.model.gaussian import GaussianModel
from pipeline.module import GaussianConfig


def verify(config_path, checkpoint_path, native_log=None):
    config = yaml.safe_load(config_path.read_text())
    state = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    expected_steps = config['trainer']['num_init_steps']
    if state['current_frame'] != 0 or state['global_step'] != expected_steps:
        raise ValueError(f'Expected a completed {expected_steps}-step native first frame')
    model = GaussianModel(config['module']['sh_degree'])
    model.restore(state['gaussians'], GaussianConfig(**config['module']['gaussian']))
    renderer = Renderer(RendererConfig(**config['module'].get('renderer', {})), model)
    scene = load_scene(Path(config['data']['root']), resolution=1, split_train_test=True, num_frames=1)
    scene.test_dataset.setup()
    if len(scene.test_dataset) != 1 or scene.test_dataset.items[0].path.name != 'cam00.png':
        raise ValueError('Expected one held-out cam00 frame')
    view = scene.test_dataset[0].to('cuda')
    background = torch.ones(3, device='cuda') if config['data']['white_background'] else torch.zeros(3, device='cuda')
    with torch.no_grad():
        image = renderer(view, background)['render'].clamp(0, 1)
        metrics = dict(psnr_db=float(psnr(image, view.image).mean()),
                       ssim=float(ssim(image, view.image)),
                       lpips_vgg=float(lpips(image, view.image, net_type='vgg').mean()),
                       l1=float((image - view.image).abs().mean()))
    native_psnr = None
    if native_log:
        text = native_log.read_text(errors='replace')
        found = re.findall(rf'Evaluate test dataset at step {expected_steps}:\s*[^\n]*?PSNR:\s*([\d.]+)', text)
        if len(found) != 1:
            raise ValueError('Could not identify the final native held-out evaluation')
        native_psnr = float(found[0])
        if abs(metrics['psnr_db'] - native_psnr) > 0.00015:
            raise ValueError('my4dgs rendering differs from the native held-out evaluation')
    return dict(status='passed', checkpoint=str(checkpoint_path.resolve()),
                input_adapter=config['data']['root'], frame=0, camera='cam00.png',
                global_step=state['global_step'],
                image_size=list(view.image.shape[-2:][::-1]), gaussian_count=len(model.get_xyz),
                active_sh_degree=int(model.active_sh_degree), independent_my4dgs_render=metrics,
                native_logged_test_psnr_db=native_psnr,
                scope='Unpruned native first frame only; no dynamic-frame quality or valid FPS measurement.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--native-log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get('N3DV_CPU_THREADS', '4')))
    result = verify(args.config, args.checkpoint, args.native_log)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
