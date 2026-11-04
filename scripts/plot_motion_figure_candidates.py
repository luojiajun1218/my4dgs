"""Eight legible photo-based alternatives for selecting advisor-report figures."""
import argparse
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches, colors
import matplotlib.patheffects as effects
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import maximum_filter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.plot_motion_sharing_scenes import configure_style, region_color

BLUE='#1676ba'; ORANGE='#e78527'; INK='#163149'; GREEN='#099a82'
HAND=(560,610,875,835); FIRE=(590,805,805,980); BACK=(310,330,550,530)
FONTPATH='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def rgb(hex_color): return np.array(colors.to_rgb(hex_color))


def tint(image,mask,color,strength=.68):
    weight=np.clip(np.asarray(mask,dtype=np.float32),0,1)[...,None]*strength
    return image*(1-weight)+rgb(color)*weight


def panel(ax,image,title='',roi=None):
    ax.imshow(image)
    if roi:
        x0,y0,x1,y1=roi;ax.set_xlim(x0,x1);ax.set_ylim(y1,y0)
    ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_visible(False)
    if title:ax.set_title(title,pad=12,fontsize=15)


def heading(fig,letter,title,subtitle):
    fig.suptitle(f'{letter}   {title}',x=.045,y=.969,ha='left',fontsize=23)
    fig.text(.045,.885 if fig.get_figheight()<7 else .905,subtitle,fontsize=12,color='#4f6475')


def notes(fig,line): fig.text(.045,.035,line,fontsize=9.5,color='#5c6c79')


def save(fig,out,stem):
    fig.savefig(out/f'{stem}.png',dpi=170,facecolor='white')
    fig.savefig(out/f'{stem}.svg',facecolor='white')
    plt.close(fig)


def mark_box(ax,roi,color):
    x0,y0,x1,y1=roi
    box=patches.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=color,linewidth=2.5)
    box.set_path_effects([effects.Stroke(linewidth=4,foreground='white'),effects.Normal()])
    ax.add_patch(box)


def option_a(s,out):
    fig=plt.figure(figsize=(15.5,8.6))
    grid=fig.add_gridspec(2,2,left=.045,right=.97,top=.84,bottom=.11,width_ratios=[1.5,1],hspace=.23,wspace=.08)
    main=fig.add_subplot(grid[:,0]);hand=fig.add_subplot(grid[0,1]);back=fig.add_subplot(grid[1,1])
    image=tint(s['full'],s['maps']['background_region_mask'],ORANGE)
    image=tint(image,s['maps']['hand_region_mask'],BLUE)
    panel(main,image,'同一场景，单独高亮两块真实共享区域')
    mark_box(main,HAND,BLUE);mark_box(main,BACK,ORANGE)
    panel(hand,tint(s['full'],s['maps']['hand_region_mask'],BLUE),'蓝色：手部附近的一块共享区域',HAND)
    panel(back,tint(s['full'],s['maps']['background_region_mask'],ORANGE),'橙色：背景中的一块共享区域',BACK)
    heading(fig,'A','直接在照片上找到共享区域','只高亮两块区域，其他地方保留原来的场景外观。')
    notes(fig,'Frame 149，底图为当前重建。色块来自真实 GS 渲染权重，包含遮挡；高亮的是区域的像素贡献，不是语义分割。')
    save(fig,out,'A_highlight_two_regions')


def option_b(s,out):
    fig,axes=plt.subplots(2,3,figsize=(15.5,9.5))
    fig.subplots_adjust(left=.045,right=.97,top=.83,bottom=.10,wspace=.035,hspace=.17)
    for col,(key,title) in enumerate([('coarse_candidates_colors','统一用较粗的候选区域'),('fine_candidates_colors','统一用较细的候选区域'),('adaptive_colors','当前 gate 实际选择')]):
        overlay=.38*s['full']+.62*s[key]
        panel(axes[0,col],overlay,title);mark_box(axes[0,col],HAND,BLUE)
        panel(axes[1,col],overlay,'同一手部窗口放大',HAND)
    heading(fig,'B','粗划分、细划分、当前划分，摆在一起看','看区域块的大小与位置：背景和人物是否需要同样细的划分？')
    notes(fig,'同一已拟合场景的分区可视化。左/中使用真实候选层级第 2/4 层，仅改变颜色归属；未重新训练，不能用本图比较三者画质。')
    save(fig,out,'B_compare_partitions')


def region_overlay(s):
    owner=s['maps']['dominant_owner'].astype(int)
    weight=s['maps']['dominant_owner_contribution'].astype(float)
    valid=(owner>=0)&(weight>=.12)
    palette=np.zeros((*owner.shape,3))
    for node in np.unique(owner[valid]):palette[owner==node]=region_color(node)
    out=s['full'].copy();out[valid]=out[valid]*.48+palette[valid]*.52
    boundary=np.zeros(owner.shape,bool)
    boundary[:,1:]|=(owner[:,1:]!=owner[:,:-1])&valid[:,1:]&valid[:,:-1]
    boundary[1:,:]|=(owner[1:,:]!=owner[:-1,:])&valid[1:,:]&valid[:-1,:]
    out[boundary]=.93
    return out


def option_c(s,out):
    overlay=region_overlay(s)
    fig=plt.figure(figsize=(15.5,8.6))
    grid=fig.add_gridspec(2,2,left=.045,right=.97,top=.84,bottom=.11,width_ratios=[1.5,1],hspace=.23,wspace=.08)
    a=fig.add_subplot(grid[:,0]);b=fig.add_subplot(grid[0,1]);c=fig.add_subplot(grid[1,1])
    panel(a,overlay,'实际区域划分叠到当前重建上')
    mark_box(a,HAND,BLUE);mark_box(a,FIRE,ORANGE)
    panel(b,overlay,'手部：不同颜色是不同共享区域',HAND)
    panel(c,overlay,'火焰附近：看实际区域交界',FIRE)
    heading(fig,'C','保留照片细节，只把区域边界标出来','不用记区域编号：沿着白线看，哪些地方被分成了不同区域。')
    notes(fig,'Frame 149。逐像素计算各真实区域的 GS 渲染权重，取贡献最大的区域着色；白线是当前视角主导区域的投影交界，不是三维边界或语义分割。')
    save(fig,out,'C_regions_on_photo')


def draw_arrows(ax,starts,deltas,color,gain=40.):
    for start,delta in zip(starts,deltas):
        if np.linalg.norm(delta*gain)<.5:
            ax.scatter(*start,s=18,color=color,edgecolor='white',linewidth=.8,zorder=10)
            continue
        arrow=patches.FancyArrowPatch(start,start+gain*delta,arrowstyle='-|>',mutation_scale=17,
                                     linewidth=2.,color=color,zorder=10)
        arrow.set_path_effects([effects.Stroke(linewidth=4,foreground='white'),effects.Normal()])
        ax.add_patch(arrow)


def option_d(s,out):
    fig,axes=plt.subplots(1,3,figsize=(15.5,6.8))
    fig.subplots_adjust(left=.045,right=.97,top=.79,bottom=.16,wspace=.035)
    base=tint(s['full'],s['maps']['hand_region_mask'],BLUE,.28)
    starts=s['maps']['arrow_start']; full=s['maps']['arrow_full']; shared=s['maps']['arrow_shared']; residual=s['maps']['arrow_residual']
    for ax,title in zip(axes,['这块区域里，9 个点各自怎么动','其中，大家可以共用的部分','扣掉共享部分，还剩多少修正']):panel(ax,base,title,(550,600,910,875))
    draw_arrows(axes[0],starts,full,BLUE)
    draw_arrows(axes[1],starts[:1],shared[:1],BLUE)
    draw_arrows(axes[2],starts,residual,ORANGE)
    heading(fig,'D','同一区域里的运动，有多少能共用？','蓝色区域相同，三个窗口相同；仅更换箭头表示的实际运动分量。')
    notes(fig,'真实 GS 位置变化的屏幕投影，所有箭头长度统一放大 40 倍；不是像素光流。中图只画一个代表点的共享平移；点数为展示抽样。')
    save(fig,out,'D_motion_arrows')


def amplitude_photo(s,key):
    amount=np.asarray(s['maps'][key],dtype=np.float32)
    return tint(s['full'],np.minimum(amount,1.),'#e12727',.88)


def option_e(scenes,out):
    fig,axes=plt.subplots(2,2,figsize=(15.5,10.1))
    fig.subplots_adjust(left=.045,right=.97,top=.84,bottom=.10,wspace=.04,hspace=.15)
    for row,s in enumerate([scenes[1],scenes[2]]):
        for col,key in enumerate(['total_amplitude','residual_amplitude']):
            title=('共享前：完整逐点运动' if col==0 else '共享后：还需要逐点记录的修正') if row==0 else ''
            panel(axes[row,col],amplitude_photo(s,key),title)
            axes[row,col].text(.02,.96,f"Frame {s['frame']}",transform=axes[row,col].transAxes,va='top',color='white',
                bbox=dict(facecolor=INK,edgecolor='none',alpha=.9,pad=5))
    heading(fig,'E','共享前后，逐点运动主要留在什么地方？','红色越明显，运动幅值越大；左右使用同一张重建图作底图、使用同一幅值标尺。')
    notes(fig,'运动/残差幅值用真实 GS 渲染权重投到图像，统一以 60 个量化格归一化并截断；是拟合运动的分析，不是重建误差或真实光流。')
    save(fig,out,'E_residual_on_photo')


def crop_psnr(a,b,roi):
    x0,y0,x1,y1=roi
    return float(-10*np.log10(np.mean((a[y0:y1,x0:x1]-b[y0:y1,x0:x1])**2)))


def option_f(scenes,out):
    fig,axes=plt.subplots(2,4,figsize=(15.5,8.6))
    fig.subplots_adjust(left=.045,right=.97,top=.82,bottom=.14,wspace=.035,hspace=.24)
    specs=[(scenes[1],HAND,'手部 · Frame 149'),(scenes[2],FIRE,'火焰 · Frame 269')]
    for row,(s,roi,label) in enumerate(specs):
        for col,(key,title) in enumerate([('no_old_position_update','不更新旧点的位置'),('shared_only','只用共享运动'),('full','共享 + 逐点修正'),('gt','真实图像 GT')]):
            panel(axes[row,col],s[key],title if row==0 else '',roi)
            axes[row,col].text(.02,.96,label if col==0 else '',transform=axes[row,col].transAxes,va='top',fontsize=10,color='white',
                bbox=dict(facecolor=INK,edgecolor='none',alpha=.9,pad=4) if col==0 else None)
            if key!='gt':
                score=crop_psnr(s[key],s['gt'],roi)
                axes[row,col].text(.5,-.11,f'此窗口 PSNR：{score:.2f} dB',transform=axes[row,col].transAxes,ha='center',fontsize=11)
    heading(fig,'F','直接比较重建图像：共享和逐点修正各起什么作用？','三个版本来自同一个已训练快照，按当前帧位置分量做开关，然后实际渲染。')
    notes(fig,'冻结诊断，仅改旧点的本帧位置更新，其他属性与新点固定；窗口 PSNR 按 RGB8 整体 MSE 计算。未重新训练，不是完整方法训练消融。')
    save(fig,out,'F_frozen_render_comparison')


def benefit_overlay(s,scale):
    error_shared=np.mean((s['shared_only']-s['gt'])**2,axis=2)
    error_full=np.mean((s['full']-s['gt'])**2,axis=2)
    difference=error_shared-error_full
    amount=np.clip(np.abs(difference)/scale,0,1)*.82
    overlay=s['full']*(1-amount[...,None])
    positive=difference>=0
    overlay+=amount[...,None]*np.where(positive[...,None],rgb(BLUE),rgb(ORANGE))
    return np.clip(overlay,0,1)


def option_g(scenes,out):
    changes=[]
    for s in scenes:
        changes.append(np.abs(np.mean((s['shared_only']-s['gt'])**2,2)-np.mean((s['full']-s['gt'])**2,2)))
    scale=max(1e-7,float(np.quantile(np.stack(changes),.995)))
    fig,axes=plt.subplots(2,3,figsize=(15.5,8.6))
    fig.subplots_adjust(left=.045,right=.97,top=.82,bottom=.11,wspace=.035,hspace=.20)
    for row,(s,roi,label) in enumerate([(scenes[1],HAND,'手部 · Frame 149'),(scenes[2],FIRE,'火焰 · Frame 269')]):
        panel(axes[row,0],s['gt'],'真实图像' if row==0 else '',roi)
        panel(axes[row,1],benefit_overlay(s,scale),'蓝色：更好     橙色：更差' if row==0 else '',roi)
        panel(axes[row,2],s['full'],'最终重建' if row==0 else '',roi)
        axes[row,0].text(.02,.96,label,transform=axes[row,0].transAxes,va='top',fontsize=10,color='white',
            bbox=dict(facecolor=INK,edgecolor='none',alpha=.9,pad=4))
    heading(fig,'G','逐点修正补对了哪里，又改坏了哪里？','与“只用共享运动”的实际渲染相比，用颜色标出像素误差的变化。')
    notes(fig,'蓝/橙分别表示加逐点残差后 RGB8 平方误差下降/上升，三帧使用同一显示尺度；不把所有修正都预设为有益。其他属性与新点固定。')
    save(fig,out,'G_where_residual_helps')


def option_h(s,summary,out):
    fig=plt.figure(figsize=(15.5,8.6))
    grid=fig.add_gridspec(1,3,left=.045,right=.97,top=.81,bottom=.17,width_ratios=[1.2,1.2,.8],wspace=.13)
    a,b,c=[fig.add_subplot(grid[i]) for i in range(3)]
    roi=(535,430,900,975)
    panel(a,s['gt'],'真实图像 GT',roi);panel(b,s['full'],'当前重建',roi)
    vals=[summary['matched_no_share_motion_mean_KB'],summary['actual_motion_mean_KB']]
    bars=c.bar(['逐点记录\n完整运动','区域共享\n+ 逐点残差'],vals,color=[ORANGE,BLUE],width=.55)
    for bar in bars:c.text(bar.get_x()+bar.get_width()/2,bar.get_height()+3,f'{bar.get_height():.1f} KB',ha='center',fontsize=14)
    c.set_ylim(0,165);c.set_ylabel('实际运动包 / 帧',fontsize=11)
    c.set_title(f"运动包减少 {summary['motion_byte_reduction']:.1%}",fontsize=16,pad=20)
    c.spines[['top','right']].set_visible(False)
    c.grid(axis='y',alpha=.2);c.set_axisbelow(True)
    heading(fig,'H','同一量化运动场，编码空间能减少多少？','照片展示当前重建；右侧在完全相同的整数总运动上，仅改变共享/逐点表示。')
    notes(fig,'字节统计覆盖 299 次动态更新，同一量化精度、同一压缩器；38.5 KB 是运动包，完整动态更新为 48.96 KB/帧。首帧另计；非重新训练对照。')
    save(fig,out,'H_quality_and_storage')


CARDS=[
 ('A_highlight_two_regions','A','单个区域高亮','最容易指着照片讲：蓝色这块、橙色那块，各是一块真实共享区域。'),
 ('B_compare_partitions','B','粗 / 细 / 当前划分','适合讲粒度选择：在同一场景上，把候选粗划分、细划分和实际选择并排看。'),
 ('C_regions_on_photo','C','照片上的实际分区边界','适合讲最终分区长什么样：保留场景外观，用颜色和白线标出当前视角的主导区域。'),
 ('D_motion_arrows','D','同一区域的少量运动箭头','适合讲为什么能共用运动：同一块区域里的 9 个真实 GS，从总运动到共享部分，再到剩余修正。'),
 ('E_residual_on_photo','E','共享前后的照片热区','适合讲共享吸收了哪里的运动：背景照片不变，只看红色运动热区怎样变化。'),
 ('F_frozen_render_comparison','F','实际重建开关对照','最像直观消融：不更新旧点位置、只用共享、加逐点修正和 GT，直接比较手部/火焰。'),
 ('G_where_residual_helps','G','修正的收益与副作用','适合分析不足：蓝色更接近 GT，橙色反而变差。可以检查逐点修正是否都值得保存。'),
 ('H_quality_and_storage','H','真实图像与编码节省','适合报告总体价值：展示当前重建，同时说明同一整数运动场的运动包可以少存多少。'),
]


def contact_sheet(out):
    width=2000; cell_w=980; cell_h=585
    canvas=Image.new('RGB',(width,4*cell_h+120),'#eef2f6')
    draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(FONTPATH,32);title=ImageFont.truetype(FONTPATH,42)
    draw.text((30,20),'八种候选图：按 A—H 选你觉得好讲的',font=title,fill=INK)
    for i,(stem,letter,name,_) in enumerate(CARDS):
        x=20+(i%2)*1000;y=105+(i//2)*cell_h
        draw.rounded_rectangle((x,y,x+cell_w,y+cell_h-12),radius=16,fill='white')
        draw.text((x+20,y+10),f'{letter}   {name}',font=font,fill=INK)
        im=Image.open(out/f'{stem}.png').convert('RGB')
        im.thumbnail((cell_w-24,cell_h-74),Image.Resampling.LANCZOS)
        canvas.paste(im,(x+(cell_w-im.width)//2,y+65+(cell_h-76-im.height)//2))
    canvas.save(out/'00_choose_A_to_H.png',optimize=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();evidence=args.evidence.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    configure_style();plt.rcParams['axes.titleweight']='normal'
    manifest=json.loads((evidence/'evidence.json').read_text())
    summary=json.loads(args.summary.read_text());scenes=[]
    for frame in (29,149,269):
        s={'frame':frame}
        for key in ['full','gt','shared_only','residual_only','no_old_position_update','adaptive_colors','coarse_candidates_colors','fine_candidates_colors']:
            s[key]=np.asarray(Image.open(evidence/f'{frame:03d}_{key}.png').convert('RGB'))/255.
        with np.load(evidence/f'{frame:03d}_maps.npz') as z:s['maps']={k:z[k].copy() for k in z}
        scenes.append(s)
    option_a(scenes[1],out);option_b(scenes[1],out);option_c(scenes[1],out);option_d(scenes[1],out)
    option_e(scenes,out);option_f(scenes,out);option_g(scenes,out);option_h(scenes[1],summary,out)
    contact_sheet(out)
    sections=''
    for stem,letter,title,explanation in CARDS:
        sections+=f'<section id="{letter}"><h2>{letter} · {html.escape(title)}</h2><p class="talk">{html.escape(explanation)}</p><a href="{stem}.png"><img src="{stem}.png"></a><p><a href="{stem}.png">打开高清 PNG</a> · <a href="{stem}.svg">SVG</a></p></section>'
    nav=' · '.join(f'<a href="#{letter}">{letter} {html.escape(title)}</a>' for _,letter,title,_ in CARDS)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>选图：A—H 真实场景分析</title>
<style>body{{font:18px/1.8 system-ui,sans-serif;color:{INK};background:#eef2f6;max-width:1480px;margin:36px auto;padding:0 28px}}h1{{font-size:34px}}h2{{font-size:25px}}section{{background:white;padding:25px 32px;margin:28px 0;border-radius:15px}}img{{width:100%;height:auto}}a{{color:{BLUE}}}.talk{{font-size:19px}}nav{{position:sticky;top:0;background:#ffffffee;padding:12px;border-radius:10px;z-index:1;font-size:15px}}details{{background:white;padding:20px;border-radius:12px}}small{{color:#5c6c79}}</style>
<h1>八种分析图，选你觉得最容易讲的</h1><p>每张只讲一件事。先看总览，再点击 A—H 查看原图；无需记区域编号或层级数字。</p><nav>{nav}</nav>
<section><a href="00_choose_A_to_H.png"><img src="00_choose_A_to_H.png"></a></section>{sections}
<details><summary>数据来源、范围与复现说明</summary><p>A/C 使用原生 GS 合成器计算真实区域贡献，包含其他区域造成的遮挡。C 对每个像素选择贡献最大的旧点区域，色块边界是投影交界，不是三维分区面或语义分割。</p>
<p>B 的固定粗/细列只可视化保存的候选层级，并未训练相应基线；不能通过这些图声称画质更好或 gate 已被独立验证。D 是 GS 位移投影，不是像素光流；箭头统一放大 40 倍。</p>
<p>F/G 是冻结渲染诊断：只改变本帧存活旧点的位置，全部当前透明度、颜色、尺度、旋转与新点位置保持固定，不代表重新训练后的方法优劣。Frame 29 中不更新旧点位置的全图 RGB8 PSNR 反而高于完整更新，全部数据保留在 evidence.json；这些图不预设所有更新都有益。</p>
<p>H 在相同整数总运动、量化精度与压缩器下比较实际运动包，非重训消融；替代包独立使用原前一帧参考。动态完整包/共享首帧单列，避免把运动包当成全部存储。</p>
<p>三帧完整重建的 RGB8 像素均与历史保存图逐位一致。此次只有读取、冻结渲染和制图，没有训练或修改历史模型，也未测量 FPS。</p><p><a href="evidence.json">原生渲染诊断数值</a> · <a href="coding_summary.json">299 次编码对照</a></p><small>原始运行：{html.escape(manifest['run'])}。制图脚本 scripts/plot_motion_figure_candidates.py；原生证据脚本 scripts/render_motion_figure_candidates.py。</small></details></html>'''
    (out/'index.html').write_text(page)
    (out/'evidence.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    (out/'coding_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    roi_metrics=[]
    for s,roi,label in [(scenes[1],HAND,'hand'),(scenes[2],FIRE,'fire')]:
        roi_metrics.append(dict(frame=s['frame'],window=label,xyxy=roi,
            RGB8_PSNR={key:crop_psnr(s[key],s['gt'],roi) for key in ['no_old_position_update','shared_only','full']}))
    (out/'roi_metrics.json').write_text(json.dumps(roi_metrics,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),candidates=8,contact_sheet=str(out/'00_choose_A_to_H.png')),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
