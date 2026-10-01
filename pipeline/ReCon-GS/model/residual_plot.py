"""Final quantized-frame residual overlays; no renderer or extra sampling."""
import itertools
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw_residual_frame(motion, materialized_xyz, viewpoint, image_path, config):
    if hasattr(motion, 'gate_state_arrays'):
        return draw_gated_frame(motion, materialized_xyz, viewpoint, image_path, config)
    image = np.asarray(Image.open(image_path).convert('RGB'))
    height, width = image.shape[:2]
    xyz = materialized_xyz.detach().cpu().numpy()
    count = len(motion.base_xyz)
    old_xyz = xyz[:count]
    index = motion.index.detach().cpu().numpy()
    symbols = motion.symbol_tensors(integer=True)
    active = [symbols[f'region_{n}'].ne(0).any(-1).cpu().numpy() for n in range(3)]
    local = symbols['gs_local'].ne(0).any(-1).cpu().numpy()
    inherited = np.zeros(count, dtype=bool)
    matrix = viewpoint.full_proj_transform.detach().cpu().numpy()
    def project(points):
        homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=-1)
        clip = homogeneous @ matrix
        valid = clip[:, 3] > 0
        xy = ((clip[:, :2] / np.where(valid[:, None], clip[:, 3:4], 1) + 1) *
              [width, height] / 2 - .5)
        return xy, valid
    fig, axes = plt.subplots(2, 2, figsize=config['region_figsize'])
    for ax in axes.flat:
        ax.imshow(image)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis('off')
    corner_bits = np.asarray(list(itertools.product((0, 1), repeat=3)), dtype=bool)
    for level, ax in enumerate(axes.flat[:3]):
        inherited |= active[level][index[level]]
        lo = np.full((motion.counts[level], 3), np.inf)
        hi = np.full_like(lo, -np.inf)
        np.minimum.at(lo, index[level], old_xyz)
        np.maximum.at(hi, index[level], old_xyz)
        corners = np.where(corner_bits[None], hi[:, None], lo[:, None])
        xy, valid = project(corners.reshape(-1, 3))
        xy, valid = xy.reshape(-1, 8, 2), valid.reshape(-1, 8)
        inherited_node = np.zeros(motion.counts[level], dtype=bool)
        np.logical_or.at(inherited_node, index[level], inherited)
        descendants = local.copy()
        for deeper in range(level + 1, 3):
            descendants |= active[deeper][index[deeper]]
        descendant_node = np.zeros(motion.counts[level], dtype=bool)
        np.logical_or.at(descendant_node, index[level], descendants)
        visible = valid.all(-1) & (active[level] | inherited_node | descendant_node)
        lower, upper = xy.min(1), xy.max(1)
        visible &= (upper[:, 0] >= 0) & (upper[:, 1] >= 0) & (lower[:, 0] < width) & (lower[:, 1] < height)
        area = np.maximum(np.minimum(upper, [width, height]) - np.maximum(lower, [0, 0]), 0).prod(-1)
        candidates = np.flatnonzero(visible)
        candidates = candidates[np.argsort(-area[candidates], kind='stable')][:config['max_boxes_per_panel']]
        for rank, node in enumerate(candidates):
            color = '#ff991f' if active[level][node] else '#52d7db' if inherited_node[node] else '#319cff'
            for a in range(8):
                for b in range(a + 1, 8):
                    if (a ^ b) in (1, 2, 4):
                        ax.plot(xy[node, [a, b], 0], xy[node, [a, b], 1], color=color,
                                lw=config['line_width'], alpha=config['box_alpha'],
                                linestyle='-' if active[level][node] else '--')
            if rank < config['max_labels_per_panel']:
                anchor = np.clip(xy[node].mean(0), [0, 0], [width - 1, height - 1])
                ax.text(*anchor, f'L{level + 1}/n{node}', fontsize=config['label_fontsize'], color=color)
        ax.set_title(f'L{level + 1} residual active={active[level].mean():.1%}; '
                     f'orange=own, cyan=inherited, blue=active descendants', fontsize=config['panel_fontsize'])
    labels = np.full(len(xyz), 'new', dtype=object)
    labels[:count] = np.where(local, 'local', np.where(inherited, 'inherited', 'keep'))
    xy, valid = project(xyz)
    inside = valid & (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < width) & (xy[:, 1] < height)
    ax = axes.flat[3]
    colors = dict(keep='#36d65d', inherited='#ff991f', local='#ff5252', new='#b55bff')
    for label, color in colors.items():
        indices = np.flatnonzero(inside & (labels == label))
        if len(indices) > config['max_points']:
            indices = indices[np.linspace(0, len(indices) - 1, config['max_points'], dtype=int)]
        ax.scatter(xy[indices, 0], xy[indices, 1], s=config['point_size'], alpha=config['point_alpha'],
                   color=color, label=f'{label}: {int((labels == label).sum())} GS')
    ax.legend(loc='lower left', fontsize=config['legend_fontsize'], markerscale=8)
    ax.set_title('Quantized GS: local red / inherited orange / zero path green / new purple', fontsize=config['panel_fontsize'])
    fig.suptitle('Materialized quantized residuals; zero child inherits parent motion; no occlusion test', fontsize=config['panel_fontsize'])
    fig.tight_layout()
    path = Path(image_path)
    fig.savefig(path.with_name(path.stem + '_residuals.png'), dpi=config['dpi'])
    plt.close(fig)


def draw_gated_frame(motion, materialized_xyz, viewpoint, image_path, config):
    """Show the committed gate decisions on the already saved saved quantized image."""
    image = np.asarray(Image.open(image_path).convert('RGB'))
    height, width = image.shape[:2]
    xyz = materialized_xyz.detach().cpu().numpy()
    count = len(motion.base_xyz)
    old_xyz = xyz[:count]
    gates = motion.gate_state_arrays()

    def numpy(value):
        return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)

    index = numpy(motion.index)
    states = [numpy(value) for value in gates['states']]
    visited = [numpy(value) for value in gates['visited']]
    leaf_open = numpy(gates['gs_leaf_open'])
    terminal = numpy(gates['gs_terminal'])
    matrix = viewpoint.full_proj_transform.detach().cpu().numpy()

    def project(points):
        homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=-1)
        clip = homogeneous @ matrix
        valid = clip[:, 3] > 0
        xy = ((clip[:, :2] / np.where(valid[:, None], clip[:, 3:4], 1) + 1) *
              [width, height] / 2 - .5)
        return xy, valid

    fig, axes = plt.subplots(2, 2, figsize=config['region_figsize'])
    for ax in axes.flat:
        ax.imshow(image)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis('off')
    corner_bits = np.asarray(list(itertools.product((0, 1), repeat=3)), dtype=bool)
    palette = {0: ('KEEP', '#36d65d'), 1: ('SHARE', '#ff991f'), 2: ('SPLIT', '#ff5252')}

    for level, ax in enumerate(axes.flat[:3]):
        node_index = index[level]
        occupied = np.bincount(node_index, minlength=motion.counts[level]) > 0
        nodes = np.flatnonzero(visited[level] & occupied)
        if len(nodes):
            lo = np.full((motion.counts[level], 3), np.inf)
            hi = np.full_like(lo, -np.inf)
            np.minimum.at(lo, node_index, old_xyz)
            np.maximum.at(hi, node_index, old_xyz)
            corners = np.where(corner_bits[None], hi[nodes, None], lo[nodes, None])
            xy, valid = project(corners.reshape(-1, 3))
            xy, valid = xy.reshape(-1, 8, 2), valid.reshape(-1, 8)
            lower, upper = xy.min(1), xy.max(1)
            visible = valid.all(-1)
            visible &= (upper[:, 0] >= 0) & (upper[:, 1] >= 0)
            visible &= (lower[:, 0] < width) & (lower[:, 1] < height)
            area = np.maximum(np.minimum(upper, [width, height]) - np.maximum(lower, [0, 0]), 0).prod(-1)
            candidates = np.flatnonzero(visible)
            candidates = candidates[np.argsort(-area[candidates], kind='stable')][:config['max_boxes_per_panel']]
            for rank, row in enumerate(candidates):
                node = nodes[row]
                label, color = palette[int(states[level][node])]
                for a in range(8):
                    for b in range(a + 1, 8):
                        if (a ^ b) in (1, 2, 4):
                            ax.plot(xy[row, [a, b], 0], xy[row, [a, b], 1], color=color,
                                    lw=config['line_width'], alpha=config['box_alpha'])
                if rank < config['max_labels_per_panel']:
                    anchor = np.clip(xy[row].mean(0), [0, 0], [width - 1, height - 1])
                    ax.text(*anchor, f'L{level + 1}/n{node} {label}',
                            fontsize=config['label_fontsize'], color=color)
        totals = [int((states[level][nodes] == state).sum()) for state in range(3)]
        ax.set_title(f'L{level + 1} reached nodes: KEEP={totals[0]} SHARE={totals[1]} SPLIT={totals[2]}; '
                     'KEEP/SHARE close descendants', fontsize=config['panel_fontsize'])

    labels = np.full(len(xyz), 'NEW', dtype=object)
    labels[:count] = np.asarray(['KEEP', 'SHARE', 'SPLIT'], dtype=object)[terminal]
    labels[:count][leaf_open] = 'SPLIT'
    xy, valid = project(xyz)
    inside = valid & (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < width) & (xy[:, 1] < height)
    ax = axes.flat[3]
    colors = dict(KEEP=palette[0][1], SHARE=palette[1][1], SPLIT=palette[2][1], NEW='#b55bff')
    for label, color in colors.items():
        indices = np.flatnonzero(inside & (labels == label))
        if len(indices) > config['max_points']:
            indices = indices[np.linspace(0, len(indices) - 1, config['max_points'], dtype=int)]
        ax.scatter(xy[indices, 0], xy[indices, 1], s=config['point_size'], alpha=config['point_alpha'],
                   color=color, label=f'{label}: {int((labels == label).sum())} GS')
    ax.legend(loc='lower left', fontsize=config['legend_fontsize'], markerscale=8)
    ax.set_title('Quantized GS: terminal KEEP green / SHARE orange / SPLIT red / new purple',
                 fontsize=config['panel_fontsize'])
    fig.suptitle('Actual nested gates; closed descendants are not visited; no occlusion test',
                 fontsize=config['panel_fontsize'])
    fig.tight_layout()
    path = Path(image_path)
    fig.savefig(path.with_name(path.stem + '_residuals.png'), dpi=config['dpi'])
    plt.close(fig)
