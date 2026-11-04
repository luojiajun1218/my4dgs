"""Analyse different face-adjacent 3-D regions in an existing fitted frame.

Neighbourhood is derived from the saved median split planes, not screen
proximity or motion similarity. Motion is the mean decoded full GS translation
within each region. This is fitted-field evidence, not independent motion GT.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.patheffects as effects
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.plot_motion_sharing_scenes import configure_style,load_scene,project

COLORS=['#098ee7','#ed8522','#079d79']


def spatial_bounds(parents,planes):
    lower=np.full((len(parents),3),-np.inf);upper=-lower.copy()
    for node in range(1,len(parents)):
        parent=int(parents[node]);assert parent<node
        assert node in range(8*parent+1,8*parent+9)
        lower[node]=lower[parent];upper[node]=upper[parent]
        octant=(node-1)%8
        for axis in range(3):
            if (octant>>axis)&1:lower[node,axis]=max(lower[node,axis],planes[parent,axis])
            else:upper[node,axis]=min(upper[node,axis],planes[parent,axis])
    return lower,upper


def adjacent_pairs(nodes,lower,upper):
    edges=[]
    for a in range(len(nodes)):
        for b in range(a+1,len(nodes)):
            lo_a,hi_a=lower[nodes[a]],upper[nodes[a]]
            lo_b,hi_b=lower[nodes[b]],upper[nodes[b]]
            overlap=np.minimum(hi_a,hi_b)-np.maximum(lo_a,lo_b)
            for axis in range(3):
                face_ab=(np.isfinite(hi_a[axis]) and np.isfinite(lo_b[axis]) and abs(hi_a[axis]-lo_b[axis])<1e-7)
                face_ba=(np.isfinite(hi_b[axis]) and np.isfinite(lo_a[axis]) and abs(hi_b[axis]-lo_a[axis])<1e-7)
                if (face_ab or face_ba) and np.all(np.delete(overlap,axis)>1e-7):
                    edges.append((a,b,axis));break
    return edges


def angles(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return np.degrees(np.arccos(np.clip(np.sum(a*b,axis=-1)/np.linalg.norm(a,axis=-1)/np.linalg.norm(b,axis=-1),-1,1)))


def statistics(vectors,edges):
    magnitude=np.linalg.norm(vectors,axis=1)
    threshold=float(np.quantile(magnitude,.75))
    moving=np.flatnonzero((magnitude>=threshold)&(magnitude>0))
    moving_set=set(moving.tolist())
    chosen=[(a,b,axis) for a,b,axis in edges if a in moving_set and b in moving_set]
    if not chosen:raise ValueError('No moving adjacent regions; do not manufacture an example')
    near_angles=[];random_angles=[];near_difference=[];random_difference=[]
    for a,b,_ in chosen:
        for anchor,partner in [(a,b),(b,a)]:
            others=moving[moving!=anchor]
            near_angles.append(float(angles(vectors[anchor],vectors[partner])))
            near_difference.append(float(np.linalg.norm(vectors[anchor]-vectors[partner])))
            # Exact expectation for a uniformly random non-self partner from
            # the SAME moving-region pool, with the SAME anchor weights.
            random_angles.append(float(angles(vectors[anchor],vectors[others]).mean()))
            random_difference.append(float(np.linalg.norm(vectors[anchor]-vectors[others],axis=1).mean()))
    return dict(moving_region_count=len(moving),moving_threshold_bins=threshold,
        adjacent_moving_pairs=len(chosen),adjacent_mean_angle_deg=float(np.mean(near_angles)),
        random_expected_mean_angle_deg=float(np.mean(random_angles)),
        adjacent_mean_vector_difference_bins=float(np.mean(near_difference)),
        random_expected_mean_vector_difference_bins=float(np.mean(random_difference)),
        random_scope='Exact uniform nonself-partner expectation within the same top-quartile moving-region pool; identical adjacency-degree anchor weights.')


def mean_region_vectors(scene,nodes):
    counts=np.array([np.sum(scene['owner']==node) for node in nodes])
    total=np.stack([scene['total'][scene['owner']==node].mean(0) for node in nodes])
    shared=np.stack([scene['shared'][np.flatnonzero(scene['owner']==node)[0]] for node in nodes])
    return counts,total,shared


def native_mask(winner,confidence,node):return (winner==node)&(confidence>=.12)


def select_example(nodes,depth,edges,winner,confidence):
    # A fixed image window on the torch-holding hand. Anchor and neighbours
    # depend ONLY on spatial adjacency, level and visible area, not motion.
    x0,y0,x1,y1=(630,650,720,855)
    local=winner[y0:y1,x0:x1]
    local_conf=confidence[y0:y1,x0:x1]
    local_area=np.array([np.sum((local==node)&(local_conf>=.12)) for node in nodes])
    global_area=np.array([native_mask(winner,confidence,node).sum() for node in nodes])
    graph={i:[] for i in range(len(nodes))}
    for a,b,_ in edges:graph[a].append(b);graph[b].append(a)
    for anchor in np.argsort(local_area,kind='stable')[::-1]:
        if local_area[anchor]<80 or depth[nodes[anchor]]!=4:continue
        neighbours=[b for b in graph[int(anchor)] if depth[nodes[b]]==4 and global_area[b]>=120]
        if len(neighbours)>=2:
            neighbours=sorted(neighbours,key=lambda b:(-int(global_area[b]),int(nodes[b])))[:2]
            return [int(anchor),*neighbours]
    raise ValueError('No three visible adjacent fine regions in the fixed hand window')


def arrow(ax,start,delta,color,width=2.2,head=15):
    patch=patches.FancyArrowPatch(start,start+delta,arrowstyle='-|>',mutation_scale=head,
        color=color,linewidth=width,zorder=9,shrinkA=0,shrinkB=0)
    patch.set_path_effects([effects.Stroke(linewidth=width+1.8,foreground='white'),effects.Normal()])
    ax.add_patch(patch)


def make_figure(scene,nodes,vectors,chosen,winner,confidence,stats,out):
    chosen_nodes=nodes[chosen];selected=vectors[chosen]
    fig=plt.figure(figsize=(15.7,8.6),facecolor='white')
    photo=fig.add_axes([.045,.19,.405,.61])
    vector_ax=fig.add_axes([.47,.19,.24,.61])
    bar=fig.add_axes([.77,.29,.19,.43])
    fig.text(.045,.961,'相邻区域的运动，相似到什么程度？',fontsize=23,color='#153248',va='top')
    fig.text(.045,.902,'握枪手附近的三个不同区域：A 与 B、C 在三维候选空间共面相邻。',fontsize=12,color='#526675')
    image=scene['reconstruction'].copy()
    positions=[]
    for node,color in zip(chosen_nodes,COLORS):
        mask=native_mask(winner,confidence,int(node))
        color_rgb=np.array(matplotlib.colors.to_rgb(color))
        image[mask]=image[mask]*.53+color_rgb*.47
        yy,xx=np.where(mask);positions.append(np.array([np.median(xx),np.median(yy)]))
    photo.imshow(image)
    # One common reference removes perspective-location differences when
    # comparing the direction of three different 3-D translation vectors.
    group=np.isin(scene['owner'],chosen_nodes)
    reference=np.median(scene['previous_xyz'][group],axis=0)
    origin=project(reference[None],scene['camera'])[0][0]
    endpoints=project(reference[None]+selected*scene['grid_world'],scene['camera'])[0]
    direction=endpoints-origin
    display=direction/np.linalg.norm(direction,axis=1)[:,None]*np.linalg.norm(selected,axis=1)[:,None]*.3
    for i,(node,color,position,delta) in enumerate(zip(chosen_nodes,COLORS,positions,display)):
        mask=native_mask(winner,confidence,int(node))
        photo.contour(mask.astype(float),levels=[.5],colors=[color],linewidths=1.2)
        arrow(photo,position,delta,color)
        photo.text(position[0]-12,position[1]+17,'ABC'[i],fontsize=14,color='white',ha='center',va='center',
            bbox=dict(facecolor=color,edgecolor='white',boxstyle='circle,pad=.22',linewidth=1.2),zorder=15)
    photo.set_xlim(590,820);photo.set_ylim(885,600);photo.axis('off')
    photo.set_title('三块相邻区域的可见部分',fontsize=15,pad=13)
    starts=np.array([[15.,95.],[15.,220.],[15.,345.]])
    for i,(start,delta,color) in enumerate(zip(starts,display,COLORS)):
        vector_ax.scatter(*start,s=25,facecolor='white',edgecolor=color,linewidth=1.2,zorder=12)
        arrow(vector_ax,start,delta,color,width=2.8,head=18)
        vector_ax.text(-1,start[1]+8,'ABC'[i],color=color,fontsize=17,ha='center',va='center')
    vector_ax.plot([15,15],[75,362],color='#c2cdd6',linewidth=1,linestyle='--',zorder=1)
    vector_ax.set_xlim(-12,125);vector_ax.set_ylim(390,0);vector_ax.set_aspect('equal');vector_ax.axis('off')
    vector_ax.set_title('三个区域的平均 GS 运动\n同一比例尺，大位移画长箭头',fontsize=13,pad=12)
    ab=float(angles(selected[0],selected[1]));ac=float(angles(selected[0],selected[2]))
    fig.text(.48,.125,f'A–B 三维方向夹角：{ab:.1f}°\nA–C 三维方向夹角：{ac:.1f}°',fontsize=11,color='#324e62')
    values=[stats['adjacent_mean_angle_deg'],stats['random_expected_mean_angle_deg']]
    bars=bar.bar(['相邻区域','随机配对'],values,color=['#3c718a','#b9c8d1'],width=.58)
    for b in bars:bar.text(b.get_x()+b.get_width()/2,b.get_height()+2,f'{b.get_height():.1f}°',ha='center',fontsize=15)
    bar.set_ylim(0,max(values)*1.3);bar.spines[['top','right']].set_visible(False)
    bar.set_ylabel('平均三维方向夹角（越小越相似）',fontsize=10)
    bar.set_title('本帧整体配对统计',fontsize=14,pad=20)
    bar.grid(axis='y',alpha=.18);bar.set_axisbelow(True)
    reduction=1-stats['adjacent_mean_vector_difference_bins']/stats['random_expected_mean_vector_difference_bins']
    fig.text(.755,.18,f'完整运动向量差也低 {reduction:.1%}',fontsize=11,color='#3c718a')
    fig.text(.045,.073,f"Frame 149：在至少含 16 个旧 GS 的区域中，取运动幅值前 25% 的 {stats['moving_region_count']} 个区域；含 {stats['adjacent_moving_pairs']} 对共面相邻区域，随机基准使用相同锚点。",fontsize=9,color='#526675')
    fig.text(.045,.039,'运动取区域内完整 GS 位移的均值，色块显示主导渲染贡献；箭头长度正比于三维位移。当前拟合结果并非独立真实运动验证。',fontsize=9,color='#637786')
    fig.savefig(out/'04_neighbor_regions.png',dpi=180,facecolor='white')
    fig.savefig(out/'04_neighbor_regions.svg',facecolor='white')
    plt.close(fig)
    return dict(nodes=chosen_nodes.tolist(),mean_motion_vectors_bins=selected.tolist(),
        AB_angle_deg=ab,AC_angle_deg=ac,common_reference_world=reference.tolist(),
        selection='Fixed torch-hand image window, maximum visible depth4 anchor with two visible face-adjacent depth4 neighbours; neighbours ranked by visible area. Motion values not used to select.',
        arrow_scope='Common camera reference for directions; lengths proportional to 3-D mean displacement, at 0.3 native pixels per quantization bin. Region colours show dominant native compositing contribution, not semantic masks.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--native-maps',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run=args.run.resolve();out=args.output.resolve()
    if out.is_relative_to(run):raise ValueError('Source run is read-only')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    configure_style();plt.rcParams['axes.titleweight']='normal'
    camera=next(c for c in json.loads((run/'cameras.json').read_text()) if c['img_name']=='cam00')
    scene=load_scene(run,149,camera)
    saved=torch.load(run/'rigid-ckpt-149-300.pth',map_location='cpu',weights_only=False)['rigid_tree_motion']
    state=saved['codec_previous_state']
    lower,upper=spatial_bounds(state['parents'].numpy(),state['spatial_centers'].numpy())
    ball=state.get('spatial_support_ball')
    supported=np.ones(scene['old_count'],bool) if ball is None else np.linalg.norm(scene['previous_xyz']-ball[:3].numpy(),axis=1)<=float(ball[3])
    point=scene['previous_xyz'][supported];owner=scene['owner'][supported]
    assert np.all(point>=lower[owner]-1e-6) and np.all(point<=upper[owner]+1e-6)
    nodes,pop=np.unique(scene['owner'],return_counts=True);nodes=nodes[pop>=16]
    counts,vectors,shared=mean_region_vectors(scene,nodes)
    edges=adjacent_pairs(nodes,lower,upper)
    stats=statistics(vectors,edges);shared_stats=statistics(shared,edges)
    with np.load(args.native_maps) as z:
        winner=z['dominant_owner'].copy();confidence=z['dominant_owner_contribution'].copy()
    native_evidence=json.loads((args.native_maps.parent/'evidence.json').read_text())
    native_row=next(row for row in native_evidence['frames'] if row['frame']==149)
    assert native_row['raw_sha']==scene['raw_sha'] and native_row['original_RGB8_exact']
    chosen=select_example(nodes,scene['node_depth'],edges,winner,confidence)
    for neighbour in chosen[1:]:assert any(set([chosen[0],neighbour])==set([a,b]) for a,b,_ in edges)
    example=make_figure(scene,nodes,vectors,chosen,winner,confidence,stats,out)
    with (out/'region_motion_vectors.csv').open('w',newline='') as f:
        columns=['region','old_GS_count','depth','mean_full_x_bins','mean_full_y_bins','mean_full_z_bins','shared_x_bins','shared_y_bins','shared_z_bins']
        writer=csv.DictWriter(f,fieldnames=columns,lineterminator='\n');writer.writeheader()
        for i,node in enumerate(nodes):
            row=dict(region=int(node),old_GS_count=int(counts[i]),depth=int(scene['node_depth'][node]))
            row.update({f'mean_full_{axis}_bins':float(vectors[i,j]) for j,axis in enumerate('xyz')})
            row.update({f'shared_{axis}_bins':int(shared[i,j]) for j,axis in enumerate('xyz')})
            writer.writerow(row)
    with (out/'neighbor_region_pairs.csv').open('w',newline='') as f:
        columns=['region_a','region_b','face_normal_axis','population_a','population_b','angle_deg','vector_difference_bins','both_moving_top25']
        writer=csv.DictWriter(f,fieldnames=columns,lineterminator='\n');writer.writeheader()
        threshold=stats['moving_threshold_bins']
        for a,b,axis in edges:
            angle=None if min(np.linalg.norm(vectors[a]),np.linalg.norm(vectors[b]))==0 else float(angles(vectors[a],vectors[b]))
            writer.writerow(dict(region_a=int(nodes[a]),region_b=int(nodes[b]),face_normal_axis='xyz'[axis],population_a=int(counts[a]),population_b=int(counts[b]),
                angle_deg=angle,vector_difference_bins=float(np.linalg.norm(vectors[a]-vectors[b])),
                both_moving_top25=bool(min(np.linalg.norm(vectors[a]),np.linalg.norm(vectors[b]))>=threshold)))
    summary=dict(run=str(run),frame=149,region_count=len(nodes),all_face_adjacent_pairs=len(edges),
        supported_position_routing_verified=True,actual_state_sha=scene['state_sha'],raw_sha=scene['raw_sha'],
        region_mean_full_motion_statistics=stats,decoded_shared_motion_statistics=shared_stats,example=example,
        scope=__doc__,adjacency='Candidate boxes derived from the saved median split planes; shared face with positive overlap on the other two axes. Full GS membership is validated for points inside the saved routing support ball; far-outlier nearest-leaf assignments are not geometric box membership.',
        caveat='The fitted model already has a spatial-sharing prior. These regional correlations do not independently validate real-world motion, gate optimality or SOTA.')
    (out/'neighbor_regions_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
