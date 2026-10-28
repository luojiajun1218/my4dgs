"""Evaluate a static checkpoint with the dynamic pipeline's exact camera setup."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
from torchvision.utils import save_image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from omegaconf import OmegaConf
from libgs.utils.config import to_structured
from pipeline.entry import Config
from pipeline.model.birth_quantization import reference_digest
from scripts import benchmark_rigid_residual_tree as base
from libgs.metric import psnr,ssim


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--render-config',type=Path,default=ROOT/'config/fs_units_300.yaml')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--opacity-threshold',type=float,default=.0075)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if any((parent/'config.yaml').is_file() for parent in args.output.resolve().parents):
        raise ValueError('Do not write evaluations into historical training runs')
    if not 0<=args.opacity_threshold<1:raise ValueError('Invalid opacity threshold')
    torch.set_num_threads(2)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint.get('current_frame')!=0:raise ValueError('A completed static frame0 is required')
    capture=checkpoint['gaussians']
    raw={name:capture[index].detach().float().contiguous() for name,index in base.CAPTURE_INDEX.items()}
    degree=int(math.isqrt(raw['sh_rest'].shape[1]+1)-1)
    if (degree+1)**2!=raw['sh_rest'].shape[1]+1 or int(capture[0])!=degree:
        raise ValueError('Checkpoint needs its complete trained SH degree')
    config,extra=to_structured(OmegaConf.load(args.render_config),Config)
    if extra:raise ValueError(extra)
    config.data.extra_dataset_kwargs['num_frames']=1
    config.module.sh_degree=degree
    module,dm=base._make_module(config,torch.device('cuda'))
    view,count=base._cam00(dm,0,torch.device('cuda'))
    if count!=1 or tuple(view.image.shape)!=(3,1008,1345):
        raise ValueError('Expected original1345x1008 cam00 split')
    attributes={k:v.to('cuda') for k,v in raw.items()}
    keep=attributes['opacity'].sigmoid().flatten()>=args.opacity_threshold
    if not keep.any():raise ValueError('Opacity filtering removed all points')
    rows=[]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for label,values in [('raw',attributes),('after_initial_opacity_prune',{k:v[keep] for k,v in attributes.items()})]:
        base._install_raw(module,values,torch.device('cuda'))
        image=module.forward(view)['render'].clamp(0,1)
        mse=(image-view.image).square().mean()
        rows.append(dict(variant=label,points=len(values['xyz']),project_PSNR=float(psnr(image,view.image)),
            global_RGB_PSNR=float(-10*mse.log10()),SSIM=float(ssim(image,view.image)),
            L1=float((image-view.image).abs().mean()),raw_sha=reference_digest(values)))
        save_image(image,args.output.with_name(args.output.stem+'_'+label+'.png'))
    result=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        frame=0,global_step=int(checkpoint['global_step']),SH_degree=degree,image_size=[1345,1008],
        opacity_threshold=args.opacity_threshold,rows=rows,
        scope='Same active dynamic Module.forward/cam00 and image size. Project metric averages per-channel dB; global RGB PSNR also shown. cam00 guides development model selection, so it is not an independent generalization estimate.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
