"""Fixed-attribute cam00 renders; never trains or selects a birth layout."""
from pathlib import Path
import json
import sys
import warnings

import torch
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from libgs.utils.config import to_structured
from libgs.renderer import Renderer
from libgs.metric import psnr,ssim
from libgs.model.merged_gaussian import MergedGaussianModel
from pipeline.entry import Config
from pipeline.data import DataModule
from scripts.probe_birth_revision import model,raw_capture

RUN=ROOT/'output/n3dv/flame_steak_1020/130k_multiview_select_ssim_180_80_40/20261024T095641_3b0b28'
OUTPUT=Path(__file__).with_name('fixed_attribute_diagnosis.json')


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        cfg,_=to_structured(OmegaConf.load(RUN/'config.yaml'),Config)
    if cfg.module.birth.refit_motion or cfg.module.opacity_update.joint_with_birth:
        raise ValueError('Fixed old attributes required for this diagnostic')
    data=DataModule(cfg.data)
    data.setup()
    background=torch.full((3,),float(cfg.data.white_background),device='cuda')
    rows=[]
    for frame in [1,9,29,59,119,179,239,299]:
        data.current_frame=frame
        dataset=data.get_test_dataset()
        if len(dataset)!=1 or dataset.items[0].path.name!='cam00.png':
            raise ValueError('Expected a single cam00 test view')
        camera=dataset[0].to('cuda')
        start=torch.load(RUN/f'rigid-ckpt-{frame}-180.pth',map_location='cuda',weights_only=False)
        end=torch.load(RUN/f'rigid-ckpt-{frame}-300.pth',map_location='cuda',weights_only=False)
        report=json.loads((RUN/f'frame_summary/frame_{frame:03d}.json').read_text())
        initial=raw_capture(start['gaussians'])
        final=raw_capture(end['gaussians'])
        k=report['birth_count']
        n=len(final['xyz'])-k
        newborn={name:value[n:] for name,value in final.items()}
        survivors={name:value[:n] for name,value in final.items()}
        old=model(initial)
        kept=model(survivors)
        models={'old_all':old,'old_survivors_only':kept,'old_all_plus_final_births':old,
                'final_survivors_plus_births':model(final)}
        if k:
            models['old_all_plus_final_births']=MergedGaussianModel([old,model(newborn)])
        metrics={}
        for name,gaussians in models.items():
            image=Renderer(cfg.module.renderer,gaussians)(camera,background)['render'].clamp(0,1)
            metrics[name]=dict(psnr=float(psnr(image,camera.image)),ssim=float(ssim(image,camera.image)))
        stages={r['step']:r['psnr'] for r in report['stage_image_metrics']}
        assert abs(metrics['old_all']['psnr']-stages[180])<1e-4
        assert abs(metrics['final_survivors_plus_births']['psnr']-stages[300])<1e-4
        p={name:score['psnr'] for name,score in metrics.items()}
        row=dict(frame=frame,births=k,retired=report['retired_count'],metrics=metrics,
            adding_final_births_to_all_old_db=p['old_all_plus_final_births']-p['old_all'],
            deleting_old_with_births_present_db=p['final_survivors_plus_births']-p['old_all_plus_final_births'],
            deleting_old_without_births_db=p['old_survivors_only']-p['old_all'],
            total_change_db=p['final_survivors_plus_births']-p['old_all'])
        rows.append(row)
        OUTPUT.write_text(json.dumps(dict(run=str(RUN),frames=rows,
            scope='Fixed-attribute render counterfactuals on eight saved cam00 frames. '
                  'Birth attributes were learned with retirement; this is not retraining, '
                  'a counterfactual sequence or an all-frame causal estimate.'),indent=2)+'\n')
        print(json.dumps({key:value for key,value in row.items() if key!='metrics'}),flush=True)
        del start,end,initial,final,newborn,survivors,old,kept,models,image,camera
    print('RESULT',OUTPUT,flush=True)


if __name__=='__main__':main()
