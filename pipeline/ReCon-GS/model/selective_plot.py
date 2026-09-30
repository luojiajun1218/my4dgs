#!/usr/bin/env python3
"""Overlay region AABBs and sampled GS centers on existing output images."""
import itertools
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

try:
    from . import selective_diagnostics as summary_module
except ImportError:  # CLI loads this module without the dynamic pipeline package.
    import selective_diagnostics as summary_module
COLORS = {'KEEP': '#36d65d', 'SHARED': '#ff991f', 'SPLIT': '#319cff', 'KEEP_UNCERTAIN': '#b0b0b0'}


def selected_xyz(state):
    owner = state['owner']
    xyz = state['base_xyz']
    source = state.get('selected_sources')
    source = owner if source is None else source[owner]
    center = state['centers'][source]
    vector = state['rotation_vector'][source]
    angle = vector.norm(dim=-1, keepdim=True)
    qxyz = vector * (torch.sinc(angle / (2 * torch.pi)) / 2)
    local = xyz - center
    cross = torch.cross(qxyz, local, dim=-1)
    moved = local + 2 * (torch.cos(angle / 2) * cross + torch.cross(qxyz, cross, dim=-1))
    moved += center + state['translation'][source]
    keep = (state['states'][owner] == 0)
    if 'observation_count' not in state:
        keep |= state['evidence'][owner] != 2
    return torch.where(keep[:, None], xyz, moved).numpy()


def project(xyz, camera, size):
    # cameras.json stores camera-to-world rotation, using +z forward.
    camera_xyz = (xyz - np.asarray(camera['position'])) @ np.asarray(camera['rotation'])
    z = camera_xyz[:, 2]
    safe_z = np.where(z > 0, z, 1)
    xy = camera_xyz[:, :2] / safe_z[:, None]
    xy = xy * [camera['fx'], camera['fy']] + [camera['width'] / 2, camera['height'] / 2]
    xy *= np.asarray(size) / [camera['width'], camera['height']]
    return xy, z > 0


def draw_frame(packet, camera, image_path, out, frame, config, *, xyz=None,
               projection=None, output_stem=None, stage_label="selected cut", extra_gs=0, write_json=True):
    summary = summary_module.summarize(packet)
    summary.update(frame=frame, source_image=str(image_path), stage_label=stage_label,
                   extra_gs_outside_tree=extra_gs, camera=config['camera'],
                   projection='Current rendered-owner GS centers; no occlusion/depth test. '
                              'Region AABBs are spatial assignment regions, not object boxes.')
    image = np.asarray(Image.open(image_path).convert('RGB'))
    height, width = image.shape[:2]
    xyz = selected_xyz(packet['state']) if xyz is None else np.asarray(xyz)
    projector = projection if projection is not None else lambda points: project(points, camera, (width, height))
    output_stem = output_stem or f'frame_{frame:03d}'
    state = packet['state']
    states, evidence = state['states'].numpy(), state['evidence'].numpy()
    owner = state['owner'].numpy()
    terminals = summary['terminal_nodes']
    active = summary['active_nodes']
    depths = packet['depths']
    fig, axes = plt.subplots(2, 2, figsize=config['region_figsize'])
    for ax in axes.flat:
        ax.imshow(image)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis('off')
    axes[0, 0].set_title(f'Frame {frame} / {config["camera"]}: existing render')
    panels = [
        (axes[0, 1], 'KEEP green' if packet.get('mode') == 'learned_selective' else 'KEEP green / KEEP(insufficient evidence) gray', [n for n in terminals if states[n] == 0], None),
        (axes[1, 0], 'SHARED current owners (orange)', [n for n in terminals if states[n] == 1], config['shared_depth']),
        (axes[1, 1], 'SPLIT active parents (blue)', [n for n in active if states[n] == 2 and n not in terminals], config['split_depth']),
    ]
    draw_records = []
    for ax, title, candidates, depth in panels:
        filtered = [n for n in candidates if depth is None or depths[n] == depth]
        visible = []
        for node in filtered:
            points = xyz[packet['members'][node].numpy()]
            corners = np.array(list(itertools.product(*zip(points.min(0), points.max(0)))))
            xy, front = projector(corners)
            # Skip near-plane straddling boxes rather than draw misleading giant edges.
            if not front.all():
                continue
            lo, hi = xy.min(0), xy.max(0)
            if hi[0] < 0 or hi[1] < 0 or lo[0] > width or lo[1] > height:
                continue
            area = np.prod(np.minimum(hi, [width, height]) - np.maximum(lo, [0, 0]))
            visible.append((area, node, xy))
        visible.sort(key=lambda row: row[0], reverse=True)
        selected = visible[:config['max_boxes_per_panel']]
        for index, (_, node, xy) in enumerate(selected):
            label = 'KEEP_UNCERTAIN' if packet.get('mode') != 'learned_selective' and states[node] == 0 and evidence[node] == 0 else summary_module.NAMES[int(states[node])]
            color = COLORS[label]
            for a in range(8):
                for b in range(a + 1, 8):
                    if (a ^ b) in (1, 2, 4):
                        ax.plot(xy[[a, b], 0], xy[[a, b], 1], color=color, lw=config['line_width'], alpha=config['box_alpha'])
            if index < config['max_labels_per_panel']:
                anchor = np.clip(xy.mean(0), [0, 0], [width - 1, height - 1])
                ax.text(*anchor, f'n{node}/d{depths[node]}', fontsize=config['label_fontsize'], color=color, bbox=dict(facecolor='black', alpha=.5, pad=1))
        ax.set_title(f'{title}\ndepth={depth if depth is not None else "all"}, total={len(candidates)}, drawn={len(selected)}, not drawn={len(candidates)-len(selected)}', fontsize=config['panel_fontsize'])
        draw_records.append(dict(panel=title, depth=depth, total=len(candidates), drawn=len(selected), not_drawn=len(candidates)-len(selected), depth_filtered_out=len(candidates)-len(filtered), out_of_view_or_near_plane=len(filtered)-len(visible)))
    fig.suptitle(f'{stage_label} / projected region AABBs, not object boxes / no occlusion test / extra GS outside tree={extra_gs}', fontsize=config['panel_fontsize'])
    fig.tight_layout()
    fig.savefig(out / f'{output_stem}_regions.png', dpi=config['dpi'])
    plt.close(fig)

    summary['drawing'] = draw_records
    if not config.get('save_gs_projection', True):
        if write_json:
            (out / f'{output_stem}.json').write_text(json.dumps(summary, indent=2))
        return summary

    # Small centers reveal assignment without the large AABB overlap.
    xy, front = projector(xyz)
    inside = front & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    labels = np.where(states[owner] == 0, 'KEEP', 'SHARED')
    if packet.get('mode') != 'learned_selective':
        labels = np.where(evidence[owner] == 0, 'KEEP_UNCERTAIN', labels)
    samples = []
    kinds_available = ('KEEP', 'SHARED') if packet.get('mode') == 'learned_selective' else ('KEEP', 'SHARED', 'KEEP_UNCERTAIN')
    for label in kinds_available:
        indices = np.flatnonzero(inside & (labels == label))
        if len(indices) > config['max_points']:
            indices = indices[np.linspace(0, len(indices)-1, config['max_points'], dtype=int)]
        samples.append(indices)
    indices = np.concatenate(samples)
    fig, axes = plt.subplots(1, 3, figsize=config['gs_figsize'])
    panel_kinds = [('KEEP',), ('SHARED',), ('KEEP', 'SHARED')] if packet.get('mode') == 'learned_selective' else [('KEEP', 'KEEP_UNCERTAIN'), ('SHARED',), ('KEEP', 'SHARED', 'KEEP_UNCERTAIN')]
    for ax, kinds in zip(axes, panel_kinds):
        ax.imshow(image)
        for label in kinds:
            selected = indices[labels[indices] == label]
            ax.scatter(xy[selected, 0], xy[selected, 1], s=config['shared_point_size'] if label == 'SHARED' else config['point_size'], alpha=config['point_alpha'], color=COLORS[label], label=f'{label.replace("KEEP_UNCERTAIN", "KEEP(insufficient evidence)")}: GS={int((labels == label).sum())}, drawn={len(selected)}')
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis('off')
        ax.legend(loc='lower left', fontsize=config['legend_fontsize'], markerscale=8)
    fig.suptitle(f'Frame {frame}: {stage_label} GS centers / no occlusion test / extra GS outside tree={extra_gs}')
    fig.tight_layout()
    fig.savefig(out / f'{output_stem}_gs.png', dpi=config['dpi'])
    plt.close(fig)
    summary['drawing'] = draw_records
    summary['sampled_centers'] = dict(in_image=int(inside.sum()), drawn=len(indices), total=len(xyz))
    if write_json:
        (out / f'{output_stem}.json').write_text(json.dumps(summary, indent=2))
    return summary



def project_with_matrix(points, matrix, size):
    homogeneous = np.column_stack([points, np.ones(len(points))])
    clip = homogeneous @ matrix
    valid = clip[:, 3] > 0
    ndc = clip[:, :2] / np.where(valid[:, None], clip[:, 3:4], 1)
    return (ndc + 1) * np.asarray(size) / 2 - .5, valid
