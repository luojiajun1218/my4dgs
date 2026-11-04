"""Render read-only photo-based figure evidence; no training or FPS timing.

Region masks use the native GS compositor, including all scene occluders.
Position interventions change this frame's surviving-old XYZ only; births and
all other current attributes stay fixed. These are frozen render diagnostics.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import benchmark_rigid_residual_tree as base
from scripts.plot_motion_sharing_scenes import load_scene, project, roi_points, selected_regions, region_color
from pipeline.model.residual_birth import RawView

FRAMES = (29, 149, 269)
WINDOWS = {
    'hand': (570, 625, 865, 825),
    'fire': (595, 810, 795, 970),
    'background': (310, 330, 550, 530),
}


def uint8(image):
    return image.detach().clamp(0, 1).mul(255).add(.5).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()


def quality(image, target):
    a, b = image.astype(np.float64)/255, target.astype(np.float64)/255
    return float(-10*np.log10(np.mean((a-b)**2)))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); run=args.run.resolve(); out=args.output.resolve()
    if out.exists() or out.is_relative_to(run):
        raise ValueError('Use a new evidence directory outside the read-only training run')
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    camera = next(c for c in json.loads((run/'cameras.json').read_text()) if c['img_name']=='cam00')
    config = base._load_config(run)
    module, dm = base._make_module(config, torch.device('cuda'))
    module.renderer.collect_contributions = False
    device = torch.device('cuda'); background=torch.zeros(3,device=device)
    records=[]; started=time.perf_counter()
    for frame in FRAMES:
        scene=load_scene(run,frame,camera)
        saved=torch.load(run/f'rigid-ckpt-{frame}-300.pth',map_location='cpu',weights_only=False)
        raw={k:saved['gaussians'][i].to(device) for k,i in base.CAPTURE_INDEX.items()}
        view, n=base._cam00(dm,frame,device); assert n==1
        target=uint8(view.image)
        assert np.array_equal(target,np.rint(scene['image']*255).astype(np.uint8))
        count=len(raw['xyz']); old=scene['old_count']

        def install(attributes):
            module.renderer.gaussians=RawView(attributes,config.module.sh_degree,module.gaussians)

        def render(colors=None):
            return module.renderer(view,background,override_color=colors)['render']

        install(raw)
        full=uint8(render())
        original=np.asarray(Image.open(run/f'image/frames/{frame:04d}/cam00_step300.png').convert('RGB'))
        if not np.array_equal(full,original):
            raise ValueError(f'Native full-render RGB8 differs from saved image at frame {frame}: '
                             f'{np.abs(full.astype(int)-original.astype(int)).max()} levels')
        Image.fromarray(full).save(out/f'{frame:03d}_full.png')
        Image.fromarray(target).save(out/f'{frame:03d}_gt.png')
        metrics={'full':quality(full,target)}
        base_xyz=torch.from_numpy(scene['previous_xyz']).to(device)
        shared=torch.from_numpy(scene['shared']).to(device)*scene['grid_world']
        residual=torch.from_numpy(scene['residual']).to(device)*scene['grid_world']
        for name,offset in [('shared_only',shared),('residual_only',residual),('no_old_position_update',torch.zeros_like(shared))]:
            changed=dict(raw); changed['xyz']=raw['xyz'].clone()
            changed['xyz'][:old]=base_xyz+offset
            install(changed); image=uint8(render())
            Image.fromarray(image).save(out/f'{frame:03d}_{name}.png')
            metrics[name]=quality(image,target)
            del changed,image
        install(raw)
        channels={}; colors=torch.zeros((count,3),device=device)
        for name,norm in [('total_amplitude',scene['total_norm']),('residual_amplitude',scene['residual_norm'])]:
            colors.zero_(); colors[:old,0]=torch.from_numpy(np.minimum(norm/60.,1.)).to(device)
            channels[name]=render(colors)[0].cpu().numpy().astype(np.float16)
        # Coarse/fine *candidate partition* visualizations on identical final
        # geometry. They are not separately trained fixed-depth baselines.
        paths=saved['rigid_tree_motion']['codec_previous_state']['spatial_paths'][:old]
        assert torch.equal(paths,saved['rigid_tree_motion']['tensors']['spatial_paths'])
        paths=paths.numpy()
        for label,owner in [('adaptive',scene['owner']),('coarse_candidates',paths[:,1]),('fine_candidates',paths[:,3])]:
            palette=np.array([region_color(int(node)) for node in owner],dtype=np.float32)
            colors.fill_(.4); colors[:old]=torch.from_numpy(palette).to(device)
            Image.fromarray(uint8(render(colors))).save(out/f'{frame:03d}_{label}_colors.png')
        # Exact per-region opacity contribution, with the rest of the scene
        # still present as occluders. Three independent masks per render.
        candidates={name:[int(n) for n in selected_regions(scene,roi,5)] for name,roi in WINDOWS.items()}
        chosen_nodes=sorted(set(n for nodes in candidates.values() for n in nodes))
        masks={}
        for start in range(0,len(chosen_nodes),3):
            group=chosen_nodes[start:start+3]; colors.zero_()
            for channel,node in enumerate(group):
                colors[:old,channel]=torch.from_numpy((scene['owner']==node).astype(np.float32)).to(device)
            result=render(colors)
            for channel,node in enumerate(group):
                masks[node]=result[channel].cpu().numpy().astype(np.float16)
        choices={}
        for name,roi in WINDOWS.items():
            x0,y0,x1,y1=roi
            choices[name]=max(candidates[name],key=lambda node:float(masks[node][y0:y1,x0:x1].astype(np.float32).sum()))
            channels[f'{name}_region_mask']=masks[choices[name]]
        # Two masks distinguish coarse/fine classes, with no invented spatial
        # boundary. Several real regions can belong to the same depth class.
        colors.zero_(); depths=scene['node_depth'][scene['owner']]
        colors[:old,0]=torch.from_numpy((depths<=2).astype(np.float32)).to(device)
        colors[:old,1]=torch.from_numpy((depths>=3).astype(np.float32)).to(device)
        result=render(colors)
        channels['coarse_class_mask']=result[0].cpu().numpy().astype(np.float16)
        channels['fine_class_mask']=result[1].cpu().numpy().astype(np.float16)
        # Exact dominant accumulated region contribution per pixel at f149.
        if frame==149:
            nodes=np.unique(scene['owner'])
            best=torch.zeros_like(result[0]); winner=torch.full_like(best,-1,dtype=torch.int32)
            for start in range(0,len(nodes),3):
                group=nodes[start:start+3];colors.zero_()
                for channel,node in enumerate(group):
                    colors[:old,channel]=torch.from_numpy((scene['owner']==node).astype(np.float32)).to(device)
                result=render(colors)[:len(group)]
                value,index=result.max(0); replace=value>best
                lookup=torch.as_tensor(group,dtype=torch.int32,device=device)
                winner[replace]=lookup[index[replace]];best=torch.maximum(best,value)
                if start%60==0:
                    print(json.dumps(dict(frame=frame,region_masks_completed=min(start+3,len(nodes)),region_masks_total=len(nodes))),flush=True)
            channels['dominant_owner']=winner.cpu().numpy()
            channels['dominant_owner_contribution']=best.cpu().numpy().astype(np.float16)
            del best,winner,value,index
        # Sparse real GS arrows for a single highlighted hand region.
        ids=roi_points(scene,WINDOWS['hand'])
        ids=ids[scene['owner'][ids]==choices['hand']]
        if len(ids):
            xy=scene['uv'][ids]; selected=[int(np.argmin(np.linalg.norm(xy-np.median(xy,axis=0),axis=1)))]
            while len(selected)<min(9,len(ids)):
                distance=np.min(np.linalg.norm(xy[:,None]-xy[selected][None],axis=2),axis=1)
                selected.append(int(distance.argmax()))
            ids=ids[selected]
        uv_before=project(scene['previous_xyz'][ids],camera)[0]
        uv_after=project(scene['previous_xyz'][ids]+scene['total'][ids]*scene['grid_world'],camera)[0]
        uv_shared=project(scene['previous_xyz'][ids]+scene['shared'][ids]*scene['grid_world'],camera)[0]
        uv_residual=project(scene['previous_xyz'][ids]+scene['residual'][ids]*scene['grid_world'],camera)[0]
        np.savez_compressed(out/f'{frame:03d}_maps.npz',**channels,
            arrow_ids=ids,arrow_start=uv_before,arrow_full=uv_after-uv_before,
            arrow_shared=uv_shared-uv_before,arrow_residual=uv_residual-uv_before)
        row=dict(frame=frame,original_RGB8_exact=True,metrics_RGB8=metrics,highlight_regions=choices,
            highlight_depths={k:int(scene['node_depth'][v]) for k,v in choices.items()},
            raw_sha=scene['raw_sha'],checkpoint_sha=scene['checkpoint_sha'])
        records.append(row);print(json.dumps(row),flush=True)
        (out/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
        del raw,saved,scene,view,colors,result,masks,channels,base_xyz,shared,residual
        gc.collect();torch.cuda.empty_cache()
    (out/'evidence.json').write_text(json.dumps(dict(run=str(run),frames=records,
        scope=__doc__,no_training=True,no_FPS_claim=True,
        region_contributions='Native sum of T*alpha per selected original encoded owner; other GS remain occluders. Coarse/fine are depth classes containing multiple regions.',
        frozen_interventions='Surviving-old XYZ only. Current birth poses, all opacities, scale, SH and rotation held fixed. Not retrained methods.',
        candidate_partitions='Saved reference spatial paths at depth2/depth4 on identical fitted geometry, native color rendering only; not trained fixed-granularity comparisons.'),indent=2)+'\n')
    print(json.dumps(dict(done=True,output=str(out),seconds=round(time.perf_counter()-started,1))),flush=True)


if __name__=='__main__': main()
