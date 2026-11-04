"""Projected decoded GS updates: full motion, regional share, GS residual.

Arrows are exact screen-coordinate differences times a single stated gain.
They are model-fitted updates, not ground-truth object motion or optical flow.
The full-motion panel is not a separately trained no-sharing baseline.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as effects
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.plot_motion_sharing_scenes import configure_style, load_scene, project


def sample_screen_grid(scene,starts,roi,spacing):
    ids=scene['visible']
    xy=starts[ids]
    keep=((xy[:,0]>roi[0]+20)&(xy[:,0]<roi[2]-20)&
          (xy[:,1]>roi[1]+20)&(xy[:,1]<roi[3]-20))
    ids=ids[keep]
    grid=np.floor((starts[ids]-np.array(roi[:2]))/spacing).astype(int)
    cell=grid[:,0]+1000*grid[:,1]
    distance=np.sum((starts[ids]-(np.array(roi[:2])+(grid+.5)*spacing))**2,1)
    order=np.lexsort((distance,cell));ordered_cell=cell[order]
    return ids[order[np.r_[True,ordered_cell[1:]!=ordered_cell[:-1]]]]


def plot_field(image,starts,screen_displacements,all_ids,arrow_ids,title,note,out,stem,roi,gain):
    fig=plt.figure(figsize=(9.6,9.6),facecolor='white')
    ax=fig.add_axes([.025,.105,.95,.80])
    ax.imshow(image)
    # Faint common origins keep point correspondence without making zero
    # residuals appear to be nonzero arrows.
    xy=starts[all_ids]
    ax.scatter(xy[:,0],xy[:,1],s=1.8,facecolor='white',edgecolor='#304151',
               linewidth=.15,alpha=.16,zorder=3)
    directions=screen_displacements[arrow_ids]
    nonzero=np.linalg.norm(directions,axis=1)>0
    # Preserve the actual projected vectors. No direction normalization,
    # 3-D-norm substitution, per-arrow scaling, or minimum arrow length.
    delta=directions*gain
    ids=arrow_ids[nonzero];delta=delta[nonzero]
    quiver=ax.quiver(starts[ids,0],starts[ids,1],delta[:,0],delta[:,1],
        angles='xy',scale_units='xy',scale=1,color='#00b8ff',
        width=.0008,headwidth=3.7,headlength=4.5,headaxislength=4.,
        minshaft=1.,minlength=0,pivot='tail',zorder=5)
    quiver.set_path_effects([effects.Stroke(linewidth=.35,foreground='#082c43'),effects.Normal()])
    ax.set_xlim(roi[0],roi[2]);ax.set_ylim(roi[3],roi[1]);ax.axis('off')
    fig.text(.045,.974,title,fontsize=21,va='top',color='#123149')
    fig.text(.045,.922,f'模型拟合位移 · Frame 148 → 149 · 投影位移统一放大 {gain:g} 倍',fontsize=10.5,color='#526575')
    fig.text(.045,.066,note,fontsize=10.5,color='#263f51')
    fig.text(.045,.033,'箭头 = 更新前后投影坐标之差 × 同一倍数；这不是物体运动真值。',fontsize=9.5,color='#607281')
    fig.savefig(out/f'{stem}.png',dpi=170,facecolor='white')
    fig.savefig(out/f'{stem}.svg',facecolor='white')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--spacing',type=int,default=18,help='Screen grid spacing in pixels; 56 was the initial sparse version.')
    p.add_argument('--gain',type=float,default=10.,help='Uniform multiplier of true projected displacement; 1 shows actual pixel displacement.')
    args=p.parse_args();run=args.run.resolve();out=args.output.resolve()
    if out.is_relative_to(run):raise ValueError('Historical outputs are read-only')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    configure_style();plt.rcParams['axes.titleweight']='normal'
    camera=next(c for c in json.loads((run/'cameras.json').read_text()) if c['img_name']=='cam00')
    scene=load_scene(run,149,camera)
    base=torch.from_numpy(scene['previous_xyz'])
    shared_xyz=base+torch.from_numpy(scene['shared']).float()*scene['region_grid_world']
    full_xyz=shared_xyz+torch.from_numpy(scene['residual']).float()*scene['grid_world']
    assert np.array_equal(full_xyz.numpy(),scene['current_xyz'])
    start=project(base.numpy(),camera)[0]
    after_shared=project(shared_xyz.numpy(),camera)[0]
    after_full=project(full_xyz.numpy(),camera)[0]
    full=after_full-start;shared=after_shared-start
    # The actual residual acts after the share. Translate its screen vector
    # to the common origin only for display. This makes full = share + residual
    # exact in the plotted 2-D coordinates despite perspective projection.
    residual=after_full-after_shared
    assert np.allclose(full,shared+residual,rtol=0,atol=1e-10)
    sampling_roi=(420,400,1050,1005);gain=args.gain;spacing=args.spacing
    if not np.isfinite(gain) or gain<=0:raise ValueError('Gain must be finite and positive')
    if spacing<6:raise ValueError('Grid spacing must be at least six pixels')
    ids=sample_screen_grid(scene,start,sampling_roi,spacing)
    representatives=[]
    for owner in np.unique(scene['owner'][ids]):
        members=ids[scene['owner'][ids]==owner]
        median=np.median(start[members],axis=0)
        representatives.append(int(members[np.argmin(np.linalg.norm(start[members]-median,axis=1))]))
    representatives=np.array(representatives,dtype=int)
    full_size=np.linalg.norm(scene['total'],axis=1)
    shared_size=np.linalg.norm(scene['shared'],axis=1)
    residual_size=np.linalg.norm(scene['residual'],axis=1)
    # Use actual screen vectors for all three panels. The former visualization
    # replaced each length with its 3-D norm, which was not a screen trajectory.
    endpoints=[]
    for directions,chosen in [(full,ids),(shared,representatives),(residual,ids)]:
        delta=directions[chosen]*gain
        endpoints.append(start[chosen]+delta)
    endpoints=np.concatenate(endpoints)
    roi=(min(sampling_roi[0],float(endpoints[:,0].min())-10),
         min(sampling_roi[1],float(endpoints[:,1].min())-10),
         max(sampling_roi[2],float(endpoints[:,0].max())+10),
         max(sampling_roi[3],float(endpoints[:,1].max())+10))
    image=np.asarray(Image.open(run/'image/frames/0148/cam00_step300.png').convert('RGB'))
    plot_field(image,start,full,ids,ids,'① GS 完整位移（共享 + 残差）',
        '同一已训练模型的总更新；不是另行训练的无共享消融。',out,'01_full_per_gs',roi,gain)
    plot_field(image,start,shared,ids,representatives,'② 区域共享位移的投影',
        '每个区域选一个实际成员，投影该成员的共享平移。',out,'02_regional_shared',roi,gain)
    plot_field(image,start,residual,ids,ids,'③ 逐 GS 残差的投影',
        '投影“共享后 → 完整更新后”的差；平移回共同起点比较。',out,'03_per_gs_residual',roi,gain)
    rows=[]
    for point in ids:
        rows.append(dict(gs_row=int(point),owner=int(scene['owner'][point]),x=float(start[point,0]),y=float(start[point,1]),
            full_dx=float(full[point,0]),full_dy=float(full[point,1]),
            shared_dx=float(shared[point,0]),shared_dy=float(shared[point,1]),
            residual_dx=float(residual[point,0]),residual_dy=float(residual[point,1]),
            full_magnitude_bins=float(full_size[point]),shared_magnitude_bins=float(shared_size[point]),
            residual_magnitude_bins=float(residual_size[point]),
            full_display_dx=float(full[point,0]*gain),full_display_dy=float(full[point,1]*gain),
            shared_display_dx=float(shared[point,0]*gain),shared_display_dy=float(shared[point,1]*gain),
            residual_display_dx=float(residual[point,0]*gain),residual_display_dy=float(residual[point,1]*gain),
            full_display_length_pixels=float(np.linalg.norm(full[point])*gain),
            shared_display_length_pixels=float(np.linalg.norm(shared[point])*gain),
            residual_display_length_pixels=float(np.linalg.norm(residual[point])*gain),
            shared_representative=bool(point in representatives),
            residual_zero_3D=bool(not np.any(scene['residual'][point]))))
    with (out/'arrows.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    summary=dict(run=str(run),frame=149,source_frame=148,gs_sample_count=len(ids),
        region_arrow_count=len(representatives),nonzero_GS_residual_count=int(np.any(scene['residual'][ids]!=0,axis=1).sum()),
        common_projected_displacement_gain=gain,roi=roi,grid_sampling_pixels=spacing,
        arrow_magnitude='Length of actual screen-coordinate displacement times a single uniform gain. No normalization, 3-D-norm substitution, per-arrow scaling, clipping, or minimum length.',
        actual_state_sha=scene['state_sha'],raw_sha=scene['raw_sha'],
        maximum_raw_projected_endpoint_decomposition_error=float(np.max(np.abs(full[ids]-shared[ids]-residual[ids]))),
        quantization_step_world_units=scene['grid_world'],
        display_scope='Projected model-fitted GS centre updates, not ground-truth object motion or optical flow. Full and share arrows use base positions; residual projected vectors are translated to the same base origins for comparison. The visualization filters approximate centre visibility, not exact per-GS alpha visibility.',
        scope=__doc__,
        residual_projection='project(base + shared + residual) - project(base + shared); displayed at base for correspondence.',
        sampling=f'One eligible Gaussian centre nearest each {spacing}px grid cell centre; depth/opacity-filtered centres; no motion-magnitude selection. All three maps use identical scale and origin set. Regional arrows are one actual member per represented region, not a normalized average.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # A preview places the three unchanged, separately saved maps side by side.
    pictures=[Image.open(out/f'{stem}.png').convert('RGB') for stem in ['01_full_per_gs','02_regional_shared','03_per_gs_residual']]
    width=2100;single=width//3;height=single
    preview=Image.new('RGB',(width,height),'white')
    for i,picture in enumerate(pictures):preview.paste(picture.resize((single,height),Image.Resampling.LANCZOS),(i*single,0))
    preview.save(out/'00_three_fields.png')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
