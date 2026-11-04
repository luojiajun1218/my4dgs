"""Make scene-based, read-only diagnostics from a certified fitted GS stream.

These figures analyse a frozen fitted motion field. The point overlays are
depth-filtered projections of Gaussian centres, not pixel optical flow or an
occlusion-exact GS render. No model is trained or changed by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager, patches
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree, ConvexHull, QhullError
import matplotlib.patheffects as path_effects
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.model import rigid_tree_codec as codec
from pipeline.model.birth_quantization import reference_digest
from pipeline.model.spatial_gate import spatial_accumulate
from pipeline.model.stream_packet import read_stream_step

FRAMES = (29, 149, 269)
ROIS = (
    ('手部与喷枪', (570, 625, 865, 825), '#2474b8'),
    ('火焰与锅', (595, 810, 795, 970), '#dc7730'),
    ('背景窗帘', (310, 330, 550, 530), '#8b6bb1'),
)


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_style():
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({
        'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
        'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'medium',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'text.color': '#15273a', 'axes.labelcolor': '#15273a',
        'svg.fonttype': 'none', 'axes.unicode_minus': False,
    })


def project(xyz, camera):
    rotation = np.asarray(camera['rotation'])
    relative = (xyz - np.asarray(camera['position'])) @ rotation
    z = relative[:, 2]
    # Matches the rasterizer's ndc2Pix centre: ((ndc + 1) * size - 1) / 2.
    uv = relative[:, :2] / z[:, None]
    uv *= [camera['fx'], camera['fy']]
    uv += [(camera['width'] - 1) / 2, (camera['height'] - 1) / 2]
    return uv, z


def front_centres(uv, z, opacity, width, height):
    valid = ((z > .2) & np.isfinite(uv).all(1) & (opacity >= .03) &
             (uv[:, 0] >= 0) & (uv[:, 0] < width) &
             (uv[:, 1] >= 0) & (uv[:, 1] < height))
    ids = np.flatnonzero(valid)
    # One nearest qualifying centre per 2 x 2 screen cell. This is a point
    # visibility approximation, explicitly separate from alpha compositing.
    cell = (uv[ids, 1].astype(int) // 2) * ((width + 1) // 2) + uv[ids, 0].astype(int) // 2
    order = np.lexsort((z[ids], cell))
    ordered_cell = cell[order]
    first = np.r_[True, ordered_cell[1:] != ordered_cell[:-1]]
    return ids[order[first]]


def load_scene(run, frame, camera):
    checkpoint_path = run / f'rigid-ckpt-{frame}-300.pth'
    checkpoint_sha = file_sha(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    motion = checkpoint['rigid_tree_motion']
    previous = codec.DecodedState.restore(motion['frame_reference_state'])
    saved_current = codec.DecodedState.restore(motion['codec_previous_state'])
    stream = run / 'frame_storage/rigid_tree'
    step = read_stream_step(stream, frame, region_count=len(previous.parents),
                            depth=previous.spatial_paths.shape[1],
                            rest=checkpoint['gaussians'][3].shape[1])
    current = codec.decode_motion((stream / f'motion_{frame:03d}.bin').read_bytes(), previous)
    manifest = json.loads((stream / f'frame_{frame:03d}.json').read_text())
    assert current.digest() == saved_current.digest()
    assert current.digest().hex() == manifest['decoded_state_sha256']
    fields = dict(xyz=1, sh_dc=2, sh_rest=3, logscale=4, rotation=5, opacity=6)
    raw = {name: checkpoint['gaussians'][index] for name, index in fields.items()}
    assert reference_digest(raw) == manifest['raw_state_sha256']
    assert torch.equal(raw['xyz'], current.xyz)
    count = step.old_count
    owner = current.owner[:count].numpy()
    regional = spatial_accumulate(torch.from_numpy(step.regional_symbols), previous.parents).numpy()[owner]
    residual = np.zeros((count, 3), dtype=np.int64)
    residual[step.gs_rows] = step.gs_symbols
    total = regional.astype(np.int64) + residual
    for node in np.unique(owner):
        assert np.all(regional[owner == node] == regional[owner == node][0])
    parents = current.parents.numpy()
    node_depth = np.zeros(len(parents), dtype=int)
    for node in range(1, len(parents)):
        assert 0 <= parents[node] < node
        node_depth[node] = node_depth[parents[node]] + 1
    previous_xyz = previous.xyz[current.survivor_rows].numpy()
    xyz = current.xyz[:count].numpy()
    expected = (torch.from_numpy(previous_xyz) + torch.from_numpy(regional).float() * step.region_step +
                torch.from_numpy(residual).float() * step.gs_step)
    assert torch.equal(expected, current.xyz[:count])
    uv, z = project(xyz, camera)
    opacity = torch.sigmoid(raw['opacity'][:count]).flatten().numpy()
    visible = front_centres(uv, z, opacity, camera['width'], camera['height'])
    before_image = run / f'image/frames/{frame:04d}/cam00_gt.png'
    reconstruction_path = run / f'image/frames/{frame:04d}/cam00_step300.png'
    image = np.asarray(Image.open(before_image).convert('RGB')) / 255.
    reconstruction = np.asarray(Image.open(reconstruction_path).convert('RGB')) / 255.
    assert image.shape == (camera['height'], camera['width'], 3)
    old_count = len(total)
    norm = np.linalg.norm(total, axis=1)
    top_count = max(1, old_count // 4)
    moving = np.argpartition(norm, -top_count)[-top_count:]
    anchors = np.intersect1d(visible, moving)
    neighbours = cKDTree(previous_xyz).query(previous_xyz[anchors], k=2, workers=1)[1]
    nearest = np.where(neighbours[:, 0] != anchors, neighbours[:, 0], neighbours[:, 1])
    rng = np.random.default_rng(55291 + frame)
    random = rng.integers(0, count, len(anchors))
    random[random == anchors] = (random[random == anchors] + 1) % count
    near_diff = np.linalg.norm(total[anchors] - total[nearest], axis=1)
    random_diff = np.linalg.norm(total[anchors] - total[random], axis=1)
    assert file_sha(checkpoint_path) == checkpoint_sha
    return dict(frame=frame, image=image, reconstruction=reconstruction, uv=uv, z=z,
                visible=visible, total=total, shared=regional, residual=residual,
                owner=owner, node_depth=node_depth, previous_xyz=previous_xyz, camera=camera,
                old_count=count, grid_world=step.gs_step,
                region_grid_world=step.region_step, current_xyz=xyz,
                total_norm=norm, residual_norm=np.linalg.norm(residual, axis=1),
                anchors=anchors, near_diff=near_diff, random_diff=random_diff,
                near_ids=nearest, random_ids=random, checkpoint_sha=checkpoint_sha,
                state_sha=current.digest().hex(), raw_sha=manifest['raw_state_sha256'],
                image_sha=file_sha(before_image), reconstruction_sha=file_sha(reconstruction_path),
                rgb_debug=np.clip(raw['sh_dc'][:count].numpy().reshape(count, 3)*.2820947918+.5, 0, 1))


def finish_axes(ax, roi=None):
    if roi is not None:
        x0, y0, x1, y1 = roi
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def image_panel(ax, scene, roi=None, reconstruction=False):
    ax.imshow(scene['reconstruction' if reconstruction else 'image'])
    finish_axes(ax, roi)


def heat_panel(ax, scene, values, norm, roi=None, ids=None):
    image = scene['image']
    grey = image @ np.array([.2126, .7152, .0722])
    ax.imshow(.64 + .36 * grey, cmap='gray', vmin=0, vmax=1)
    ids = scene['visible'] if ids is None else ids
    if roi is not None:
        x0, y0, x1, y1 = roi
        xy = scene['uv'][ids]
        keep = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
        ids = ids[keep]
    # Marker area is fixed in screen pixels, independent of the quantity.
    marker_size = 1.6 if roi is None else 9.
    dots = ax.scatter(scene['uv'][ids, 0], scene['uv'][ids, 1], c=values[ids],
                      cmap='YlOrRd', norm=norm, s=marker_size, linewidths=0,
                      rasterized=True)
    finish_axes(ax, roi)
    return dots


def region_color(node):
    """Stable per-node colour, never a motion magnitude or semantic class."""
    return colors.hsv_to_rgb([(.61803398875 * int(node) + .07) % 1., .66, .88])


def roi_points(scene, roi):
    ids = scene['visible']
    x0, y0, x1, y1 = roi
    xy = scene['uv'][ids]
    return ids[(xy[:, 0] >= x0) & (xy[:, 0] <= x1) &
               (xy[:, 1] >= y0) & (xy[:, 1] <= y1)]


def selected_regions(scene, roi, count=3):
    ids = roi_points(scene, roi)
    nodes, counts = np.unique(scene['owner'][ids], return_counts=True)
    return nodes[np.argsort(counts, kind='stable')[::-1][:count]]


def region_panel(ax, scene, roi=None, depth=False):
    grey = scene['image'] @ np.array([.2126, .7152, .0722])
    ax.imshow(.45 + .55 * grey, cmap='gray', vmin=0, vmax=1)
    ids = scene['visible'] if roi is None else roi_points(scene, roi)
    owner = scene['owner'][ids]
    options = dict(s=1.8 if roi is None else 10., linewidths=0, rasterized=True)
    if depth:
        dots = ax.scatter(scene['uv'][ids, 0], scene['uv'][ids, 1],
            c=scene['node_depth'][owner], cmap='viridis',
            norm=colors.BoundaryNorm(np.arange(.5, 5.5), 256), **options)
    else:
        dots = ax.scatter(scene['uv'][ids, 0], scene['uv'][ids, 1],
            c=np.array([region_color(node) for node in owner]), **options)
    finish_axes(ax, roi)
    return dots


def region_annotations(ax, scene, roi, nodes, arrows=False, labels=True):
    """Outlines are projected member envelopes, not invented image segments."""
    visible = roi_points(scene, roi)
    x0, y0, x1, y1 = roi
    for rank, node in enumerate(nodes):
        ids = visible[scene['owner'][visible] == node]
        if len(ids) < 3:
            continue
        xy = scene['uv'][ids]
        color = region_color(node)
        try:
            hull = ConvexHull(xy)
            boundary = xy[np.r_[hull.vertices, hull.vertices[:1]]]
            line, = ax.plot(boundary[:, 0], boundary[:, 1], color=color,
                            linewidth=1.8, linestyle='--', alpha=.95)
            line.set_path_effects([path_effects.Stroke(linewidth=3., foreground='white', alpha=.9),
                                   path_effects.Normal()])
        except QhullError:
            pass
        if labels:
            # Matching colour/ID strips avoid collisions among overlapping
            # envelopes; a thin leader line points to a real member centre.
            median = np.median(xy, axis=0)
            pick = ids[np.argmin(np.linalg.norm(xy - median, axis=1))]
            label = f"R{int(node)} · D{scene['node_depth'][node]}"
            ax.annotate(label, xy=scene['uv'][pick],
                xytext=(.025, .945-rank*.105), textcoords='axes fraction',
                fontsize=9, va='top', color='#15273a',
                bbox=dict(facecolor='white', edgecolor=color, linewidth=1.5, alpha=.94, pad=3),
                arrowprops=dict(arrowstyle='-', color=color, linewidth=1.1))
        if arrows:
            # One representative GS per region, same 3-D shared translation.
            # Perspective is applied before multiplying the screen arrow by 8.
            target = np.median(xy, axis=0)
            pick = ids[np.argmin(np.linalg.norm(xy-target, axis=1))]
            xyz = scene['previous_xyz'][pick]
            start = project(xyz[None], scene['camera'])[0][0]
            end = project((xyz + scene['shared'][pick]*scene['grid_world'])[None], scene['camera'])[0][0]
            displayed_end = start + (end-start)*8
            if np.linalg.norm(displayed_end-start) >= .5:
                arrow = patches.FancyArrowPatch(start, displayed_end, arrowstyle='-|>',
                    mutation_scale=15, linewidth=2.2, color=color, zorder=12)
                arrow.set_path_effects([path_effects.Stroke(linewidth=4.5, foreground='white'),
                                        path_effects.Normal()])
                ax.add_patch(arrow)
            ax.scatter(*start, s=25, facecolor=color, edgecolor='white', linewidth=1., zorder=13)


def region_figure(scenes, out):
    fig, axes = plt.subplots(3, 3, figsize=(15.7, 12.2))
    fig.subplots_adjust(left=.055, right=.91, top=.885, bottom=.105, wspace=.025, hspace=.11)
    fig.suptitle('实际共享区域：gate 最终选中了哪些空间粒度？', x=.05, ha='left', y=.97, fontsize=22)
    fig.text(.05, .924, '直接读取每帧最终编码的区域归属：同一区域同色；同一区域的旧 GS 共享同一个三维平移向量。', fontsize=11)
    for row, scene in enumerate(scenes):
        image_panel(axes[row, 0], scene)
        region_panel(axes[row, 1], scene)
        dots = region_panel(axes[row, 2], scene, depth=True)
        if row == 0:
            for col, title in enumerate(['场景参考图像', '实际共享区域（颜色 = 区域编号）', '实际区域层级（深度越大，粒度越细）']):
                axes[row, col].set_title(title, pad=10, fontsize=12)
        for ax in axes[row]:
            ax.text(.025, .95, f"Frame {scene['frame']:03d}", transform=ax.transAxes,
                    color='white', va='top', fontsize=11,
                    bbox=dict(facecolor='#15273a', edgecolor='none', alpha=.82, pad=4))
        owners = np.unique(scene['owner'])
        axes[row, 1].text(.99, .04, f'全部旧 GS 共 {len(owners)} 个有效区域',
            transform=axes[row, 1].transAxes, ha='right', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='none', alpha=.95, pad=4))
        for ax in axes[row]:
            for _, roi, border in ROIS[:2]:
                x0, y0, x1, y1 = roi
                ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                                              edgecolor=border, linewidth=1.25))
    cax = fig.add_axes([.932, .28, .012, .46])
    cb = fig.colorbar(dots, cax=cax, ticks=[1, 2, 3, 4])
    cb.set_label('候选空间层级深度 D', labelpad=12)
    footer(fig, '彩框标出下一张局部分析图中的手部和火焰窗口；区域颜色由原始 owner 编号确定，跨图保持一致。',
           '这些是当前 gate 的实际选择，不是语义分割。显示旧 GS 中心投影，投影可以重叠；不能仅凭此图断言自适应选择优于固定粒度。')
    save_figure(fig, out, '00_shared_regions')


def save_figure(fig, out, stem):
    fig.savefig(out / f'{stem}.png', dpi=180, facecolor='white')
    fig.savefig(out / f'{stem}.svg', facecolor='white')
    plt.close(fig)


def footer(fig, line1, line2):
    fig.text(.05, .043, line1, fontsize=10, color='#526172')
    fig.text(.05, .018, line2, fontsize=9, color='#687585')


def global_figure(scenes, summary, out, norm):
    fig, axes = plt.subplots(3, 3, figsize=(15.7, 12.2))
    fig.subplots_adjust(left=.055, right=.91, top=.885, bottom=.105, wspace=.025, hspace=.11)
    fig.suptitle('共享运动是否减少了需要逐点记录的变化？', x=.05, ha='left', y=.97, fontsize=22)
    fig.text(.05, .924, '真实场景对照：同一已拟合运动场、同一量化网格；暖色越深，幅值越大。', fontsize=12)
    titles = ['场景参考图像', '无共享表示：完整逐点运动', '共享表示：剩余逐点残差']
    for row, scene in enumerate(scenes):
        image_panel(axes[row, 0], scene)
        heat_panel(axes[row, 1], scene, scene['total_norm'], norm)
        dots = heat_panel(axes[row, 2], scene, scene['residual_norm'], norm)
        if row == 0:
            for col, title in enumerate(titles):
                axes[row, col].set_title(title, pad=10)
        for ax in axes[row]:
            ax.text(.025, .95, f"Frame {scene['frame']:03d}", transform=ax.transAxes,
                    color='white', va='top', fontsize=11,
                    bbox=dict(facecolor='#15273a', edgecolor='none', alpha=.82, pad=4))
        a = np.any(scene['total'] != 0, axis=1).mean()
        b = np.any(scene['residual'] != 0, axis=1).mean()
        axes[row, 1].text(.99, .04, f'非零运动 {a:.1%}', transform=axes[row, 1].transAxes,
                          ha='right', fontsize=10, bbox=dict(facecolor='white', edgecolor='none', alpha=.94, pad=4))
        axes[row, 2].text(.99, .04, f'非零残差 {b:.1%}', transform=axes[row, 2].transAxes,
                          ha='right', fontsize=10, bbox=dict(facecolor='white', edgecolor='none', alpha=.94, pad=4))
    cax = fig.add_axes([.93, .28, .012, .48])
    cb = fig.colorbar(dots, cax=cax, extend='max')
    cb.set_label('运动幅值 / 量化格', labelpad=12)
    footer(fig,
           f"299 次帧更新的实际运动包：无共享 {summary['matched_no_share_motion_mean_KB']:.1f} KB → 共享 {summary['actual_motion_mean_KB']:.1f} KB / 帧，减少 {summary['motion_byte_reduction']:.1%}。",
           '色图是旧 GS 中心的深度筛选投影，非像素光流；排除本帧新生点。非零比例统计全部旧 GS。左图为 GT，右两图为拟合运动诊断。')
    save_figure(fig, out, '01_scene_residual')


def crop_figure(scene, out, norm):
    fig, axes = plt.subplots(3, 3, figsize=(15.7, 11.6))
    fig.subplots_adjust(left=.075, right=.91, top=.875, bottom=.11, wspace=.03, hspace=.19)
    fig.suptitle('局部对照：共享区域、共享运动与剩余残差', x=.05, ha='left', y=.967, fontsize=21)
    fig.text(.05, .922, f"Frame {scene['frame']:03d} · R = 真实区域编号，D = 候选层级深度；虚线圈出同一区域的投影点。", fontsize=12)
    for row, (name, roi, border) in enumerate(ROIS):
        image_panel(axes[row, 0], scene, roi)
        region_panel(axes[row, 1], scene, roi)
        dots = heat_panel(axes[row, 2], scene, scene['residual_norm'], norm, roi)
        selected = selected_regions(scene, roi)
        region_annotations(axes[row, 1], scene, roi, selected, arrows=True)
        region_annotations(axes[row, 2], scene, roi, selected)
        for col in range(3):
            if row == 0:
                axes[row, col].set_title(['局部真实图像', '真实共享区域 + 共享平移箭头', '对应区域中的剩余逐点残差'][col], pad=10, fontsize=12)
            for spine in axes[row, col].spines.values():
                spine.set_visible(True)
                spine.set_color(border)
                spine.set_linewidth(1.8)
        axes[row, 0].text(-.05, .5, name, rotation=90, transform=axes[row, 0].transAxes,
                          va='center', ha='right', fontsize=12, color=border)
    cax = fig.add_axes([.932, .29, .012, .44])
    cb = fig.colorbar(dots, cax=cax, extend='max')
    cb.set_label('运动幅值 / 量化格', labelpad=12)
    footer(fig, '每个窗口标注成员最多的 3 个可见区域；同色虚线和相同 R 编号对应同一批点。箭头为共享平移的屏幕投影，长度统一放大 8 倍。',
           '虚线为区域成员的投影包络，可相互重叠，不是二维分割边界。右图暖色越深，残差越大；R 颜色表示身份，不表示幅值。')
    save_figure(fig, out, '02_local_regions')


def locality_figure(scene, summary, out):
    fig, axes = plt.subplots(2, 3, figsize=(15.7, 10.1))
    fig.subplots_adjust(left=.055, right=.905, top=.86, bottom=.14, wspace=.025, hspace=.17)
    fig.suptitle('空间相邻的 GS，拟合运动是否更相似？', x=.05, ha='left', y=.967, fontsize=22)
    fig.text(.05, .912, '使用同一批运动较大的 GS 作锚点，仅更换比较对象：空间近邻 / 全场景随机 GS。', fontsize=12)
    ids = scene['anchors']
    near = np.full(scene['old_count'], np.nan)
    random = near.copy()
    near[ids] = scene['near_diff']
    random[ids] = scene['random_diff']
    vmax = float(np.quantile(np.r_[scene['near_diff'], scene['random_diff']], .95))
    norm = colors.Normalize(0, max(1., vmax))
    image_panel(axes[0, 0], scene)
    heat_panel(axes[0, 1], scene, near, norm, ids=ids)
    dots = heat_panel(axes[0, 2], scene, random, norm, ids=ids)
    roi = (560, 560, 880, 830)
    image_panel(axes[1, 0], scene, roi)
    heat_panel(axes[1, 1], scene, near, norm, roi, ids=ids)
    heat_panel(axes[1, 2], scene, random, norm, roi, ids=ids)
    for col, title in enumerate(['场景参考图像', '锚点与空间近邻的运动差异', '锚点与随机点的运动差异']):
        axes[0, col].set_title(title, pad=12)
    x0, y0, x1, y1 = roi
    for ax in axes[0]:
        ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor='#2375b8', linewidth=1.6))
    axes[1, 0].text(.02, .95, '同一窗口放大', transform=axes[1, 0].transAxes, color='white', va='top',
                    bbox=dict(facecolor='#15273a', edgecolor='none', alpha=.82, pad=4))
    cax = fig.add_axes([.93, .25, .012, .49])
    cb = fig.colorbar(dots, cax=cax, extend='max')
    cb.set_label('两点运动向量差的幅值 / 量化格', labelpad=12)
    footer(fig,
           f"全 299 次更新的配对抽样：近邻平均差异 {summary['nearest_difference_bins']:.1f} 格，随机配对 {summary['random_difference_bins']:.1f} 格，前者低 {summary['nearest_difference_reduction']:.1%}。",
           '锚点来自每帧位移幅值最大的 25% 旧 GS，排除自身配对。图中为 Frame 149 可见锚点；全序列统计每帧随机抽样 1200 锚点。')
    save_figure(fig, out, '03_neighbour_similarity')


def companion_metrics(rows, summary, out):
    fig, axes = plt.subplots(1, 3, figsize=(15.7, 5.6))
    fig.subplots_adjust(left=.065, right=.97, top=.77, bottom=.21, wspace=.4)
    fig.suptitle('配套统计：固定已拟合运动场的编码分析', x=.05, ha='left', y=.96, fontsize=21)
    bars = axes[0].bar(['无共享', '共享 + 残差'],
                       [summary['matched_no_share_motion_mean_KB'], summary['actual_motion_mean_KB']],
                       color=['#dd8c4f', '#287f9b'], width=.55)
    for bar in bars:
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+3, f'{bar.get_height():.1f}', ha='center', fontsize=13)
    axes[0].set_ylim(0, max(b.get_height() for b in bars)*1.2)
    axes[0].set_ylabel('实际运动包 KB / 帧')
    axes[0].set_title(f"平均减少 {summary['motion_byte_reduction']:.1%}")
    frames = [r['frame'] for r in rows]
    axes[1].plot(frames, [r['total_nonzero']/1e4 for r in rows], label='完整逐点运动', color='#dd8c4f')
    axes[1].plot(frames, [r['residual_nonzero']/1e4 for r in rows], label='逐点残差', color='#287f9b')
    axes[1].set(xlabel='帧索引', ylabel='非零向量 / 万 GS', title='全部旧 GS，统一量化网格')
    axes[1].legend(frameon=False, fontsize=9)
    axes[2].plot(frames, [r['random_difference_bins'] for r in rows], label='随机配对', color='#dd8c4f')
    axes[2].plot(frames, [r['nearest_difference_bins'] for r in rows], label='空间近邻', color='#287f9b')
    axes[2].set(xlabel='帧索引', ylabel='运动差异 / 量化格', title='同一批运动锚点的配对对比')
    axes[2].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#e8edf2')
        ax.set_axisbelow(True)
    footer(fig, '299 次动态更新；运动包包含结构及本帧新生点位姿，两组使用同一量化精度与生产压缩器。',
           '无共享包使用各帧原始前一帧参考独立重编码；不是重新训练消融，也不是新的连续替代码流。')
    save_figure(fig, out, '04_supporting_statistics')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, evidence, out = args.run.resolve(), args.evidence.resolve(), args.output.resolve()
    if out.is_relative_to(run):
        raise ValueError('Historical run is read-only')
    out.mkdir(parents=True, exist_ok=True)
    configure_style()
    torch.set_num_threads(2)
    summary = json.loads((evidence/'summary.json').read_text())
    assert summary['all299_state_hashes_verified'] and summary['all_integer_motion_fields_exact']
    assert summary['frames'] == 299 and summary['maximum_alternative_float32_difference_bins'] < .02
    with (evidence/'per_frame.csv').open() as f:
        rows = [{k: float(v) if k not in ('state_sha', 'alternative_packet_sha', 'quantized_motion_exact') else v
                 for k, v in row.items()} for row in csv.DictReader(f)]
    assert len(rows) == 299
    camera = next(c for c in json.loads((run/'cameras.json').read_text()) if c['img_name'] == 'cam00')
    scenes = [load_scene(run, frame, camera) for frame in FRAMES]
    vmax = float(np.quantile(np.concatenate([s['total_norm'][s['visible']] for s in scenes]), .95))
    norm = colors.Normalize(0, max(1., vmax))
    region_figure(scenes, out)
    global_figure(scenes, summary, out, norm)
    crop_figure(scenes[1], out, norm)
    locality_figure(scenes[1], summary, out)
    companion_metrics(rows, summary, out)
    # Useful for visual QA of camera conventions; never label this as a render.
    fig, ax = plt.subplots(figsize=(10, 7.5))
    image_panel(ax, scenes[1])
    s = scenes[1]; ids=s['visible']
    ax.scatter(s['uv'][ids, 0], s['uv'][ids, 1], c=s['rgb_debug'][ids], s=2, linewidths=0)
    fig.savefig(evidence/'projection_alignment_QA.png', dpi=140)
    plt.close(fig)
    (out/'per_frame.csv').write_bytes((evidence/'per_frame.csv').read_bytes())
    compact_summary = {k: v for k, v in summary.items() if k != 'input_sha256'}
    compact_summary['motion_display_max_bins'] = vmax
    compact_summary['region_overlay_scope'] = 'Actual final encoded owner IDs; stable per-node colours. Dashed outlines are convex envelopes of depth-filtered projected member centres within the crop, not image segmentation boundaries. Three most populous visible regions labelled per crop. Arrows use one actual member per labelled region, project its shared 3-D translation at the previous position, then multiply screen displacement by 8.'
    compact_summary['scene_projection_scope'] = __doc__
    compact_summary['scene_sources'] = [dict(frame=s['frame'], checkpoint_sha=s['checkpoint_sha'],
        state_sha=s['state_sha'], raw_sha=s['raw_sha'], image_sha=s['image_sha'],
        reconstruction_sha=s['reconstruction_sha'], shown_old_centres=len(s['visible']),
        shown_topquartile_anchors=len(s['anchors'])) for s in scenes]
    (out/'summary.json').write_text(json.dumps(compact_summary, indent=2, ensure_ascii=False)+'\n')
    region_rows = []
    for scene in scenes:
        for node in np.unique(scene['owner']):
            members = np.flatnonzero(scene['owner'] == node)
            shared = scene['shared'][members[0]]
            region_rows.append(dict(frame=scene['frame'], region=int(node), depth=int(scene['node_depth'][node]),
                old_members=len(members), shown_centres=int(np.sum(scene['owner'][scene['visible']] == node)),
                shared_x_bins=int(shared[0]), shared_y_bins=int(shared[1]), shared_z_bins=int(shared[2]),
                nonzero_residual_members=int(np.any(scene['residual'][members] != 0, axis=1).sum()),
                mean_residual_bins=float(scene['residual_norm'][members].mean())))
    with (out/'scene_regions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(region_rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(region_rows)
    cards = [
        ('00_shared_regions', '当前 gate 实际选出的共享区域与空间粒度',
         '直接从最终码流解码得到每个旧 GS 的 owner 编号。同编号同色；这些点共享同一个三维平移向量。第三列是相应候选区域的实际层级深度，深度越大、空间粒度越细。彩框连接后面的手部和火焰放大图。颜色是区域身份，并非语义类别。'),
        ('01_scene_residual', '共享前后，逐点需要记录的运动在哪里？',
         '三帧真实场景与对应 GS 中心投影。中列为完整逐点运动，右列为扣除区域共享运动后的残差。三行共用色标，深色代表更大的量化运动幅值。对比的是同一已拟合运动场的两种表示。'),
        ('02_local_regions', '放大手部、火焰与背景，观察共享的实际作用',
         '固定 Frame 149 的三个窗口。中列同色点属于实际相同的共享区域，标出 R 编号、D 深度和共享平移箭头；右列用相同颜色的虚线和编号对应这些区域，再查看区域内剩余残差。每个窗口标注成员最多的三个可见区域。虚线是区域成员的投影包络，不是二维语义分割边界；投影可重叠。共享平移箭头为真实成员的三维共享位移投影，长度统一放大 8 倍。'),
        ('03_neighbour_similarity', '相邻 GS 的拟合运动是否更相似？',
         '同一批位移幅值位于前 25% 的 GS 作锚点；中、右两列分别显示与三维空间最近邻、随机非自身 GS 的运动向量差。不是凭箭头示意，而是从实际码流中的完整运动向量计算。这里只验证当前模型中存在局部运动相关性，不等于独立真实运动的验证。'),
        ('04_supporting_statistics', '全序列配套数字',
         '299 次动态更新的成对统计。运动包字节数是生产压缩器输出的实际文件长度，包含结构与本帧新点位姿，未将独立压缩的子流大小相加。KB=1000 字节。'),
    ]
    body=''.join(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p><a href="{stem}.png"><img src="{stem}.png" alt="{html.escape(title)}"></a><p><a href="{stem}.png">PNG</a> · <a href="{stem}.svg">SVG</a></p></section>' for stem,title,caption in cards)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>运动共享：场景分析图</title>
<style>body{{font:17px/1.8 system-ui,sans-serif;background:#f3f5f8;color:#15273a;max-width:1400px;margin:45px auto;padding:0 30px}}h1{{font-size:32px}}h2{{font-size:23px}}section{{background:white;padding:26px 32px;margin:30px 0;border-radius:12px}}img{{width:100%;height:auto}}a{{color:#2474b8}}small{{color:#586675}}.scope{{background:#e7eef5;padding:22px;border-radius:12px}}</style>
<h1>运动共享：真实场景分析</h1>
<p>Flame Steak · cov150 完整 300 帧结果 · 首帧之后 299 次更新</p>
<p class="scope"><b>汇报时可说：</b>在这组拟合结果中，空间近邻的运动更相似；共享表示将逐点运动转为更稀疏、更小的残差；在相同量化运动场下，实际运动包从 {summary['matched_no_share_motion_mean_KB']:.2f} 降至 {summary['actual_motion_mean_KB']:.2f} KB/帧，减少 {summary['motion_byte_reduction']:.1%}。</p>
{body}
<section><h2>结论范围与复现</h2><p>这些是冻结已拟合场的场景诊断与编码消融，没有重新训练“关闭共享”或“固定粒度”的模型，因此不能据此单独声称 learnable gate 优于固定划分，也不能宣称超过其他方法的 SOTA。</p>
<p>点图在当前帧相机中投影旧 GS 中心，透明度阈值 0.03，每 2×2 像素格保留最近的合格中心。它不是 alpha 合成渲染、不是像素光流。局部窗口是图像窗口，可能包含不同深度的三维点；本帧新生点未进入旧点运动分析。</p>
<p>无共享编码将区域运动置零，将同一整数总运动放入逐点流，保留出生/删除与位姿负载，并使用相同压缩器。299 帧的整数运动均严格一致；浮点加法次序的最大差异为 {summary['maximum_alternative_float32_difference_bins']:.6f} 个量化格。因此这里只声称量化运动场一致，不声称替代连续流的像素逐位一致。</p>
<p>当前完整动态更新为 {summary['complete_dynamic_mean_KB']:.2f} KB/帧；前面的 {summary['actual_motion_mean_KB']:.2f} KB 仅指运动包。共享首帧单独保存，未摊入此动态均值。</p>
<p><a href="per_frame.csv">299 帧统计 CSV</a> · <a href="scene_regions.csv">样本帧的实际区域与共享位移</a> · <a href="summary.json">汇总与数据哈希</a></p>
<small>数据源：{html.escape(str(run))}<br>制图：scripts/plot_motion_sharing_scenes.py；证据提取：scripts/extract_motion_sharing_figures.py。样本帧 29 / 149 / 269（预先选取前、中、后时间点，未按图像收益重新挑选）。</small></section></html>'''
    (out/'index.html').write_text(page)
    print(json.dumps(dict(output=str(out),figures=[c[0] for c in cards],motion_display_max_bins=vmax,
        scene_points=[len(s['visible']) for s in scenes]), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
