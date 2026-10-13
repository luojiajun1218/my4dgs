"""Read exact final codec symbols and write completed-run region reports."""

import argparse
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name, directory in (('pipeline', ROOT / 'pipeline'),
                        ('pipeline.ReCon-GS', ROOT / 'pipeline' / 'ReCon-GS')):
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package

CODEC = importlib.import_module('pipeline.ReCon-GS.model.rigid_tree_codec')
DIAGNOSTICS = importlib.import_module('pipeline.ReCon-GS.model.online_motion_diagnostics')

RUNS = {
    'kmeans': 'output/fs_online_flat_kmeans_refine_first5_20261013/20261013T190219_cf861b',
    'gmm': 'output/fs_online_flat_gmm_refine_first5_20261013/20261013T190504_bff46c',
    'hdbscan': 'output/fs_online_flat_hdbscan_refine_first5_20261013/20261013T190751_b853aa',
    'greedy': 'output/fs_online_flat_greedy_refine_first5_20261013/20261013T191235_979a4c',
}


def read_symbol_state(payload, prior):
    """Capture the original decoder's integer reads without duplicating parsing."""
    captured = []
    original = CODEC._sparse_read

    def capture(reader, count, **kwargs):
        result = original(reader, count, **kwargs)
        captured.append(tuple(value.clone() for value in result))
        return result

    with patch.object(CODEC, '_sparse_read', side_effect=capture):
        state = CODEC.decode_motion(payload, prior)
    if len(captured) != 2:
        raise ValueError('Expected shared and GS sparse symbol sections')
    rt, rr = captured[0]
    zeros = torch.zeros(1, 3, dtype=rt.dtype)
    rt, rr = torch.cat((zeros, rt)), torch.cat((zeros, rr))
    gt, gr = captured[1]
    return state, rt, rr, gt, gr


def analyze(method, run):
    stream = run / 'frame_storage' / 'rigid_tree'
    with np.load(stream / 'reference_frame0.npz', allow_pickle=False) as reference:
        state = CODEC.DecodedState(
            torch.from_numpy(reference['xyz'].copy()),
            torch.from_numpy(reference['rotation'].copy()),
            torch.from_numpy(reference['owner'].copy()).long(),
            torch.from_numpy(reference['parents'].copy()).long(),
            int(reference['frame_index']))
    reports = []
    summaries = sorted((run / 'frame_summary').glob('frame_*.json'))
    if len(summaries) != 4:
        raise ValueError('This report expects four completed dynamic frames')
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text())
        frame = int(summary['frame'])
        state, rt, rr, gt, gr = read_symbol_state(
            (stream / f'motion_{frame:03d}.bin').read_bytes(), state)
        manifest = json.loads((stream / f'frame_{frame:03d}.json').read_text())
        assert state.digest().hex() == manifest['decoded_state_sha256']
        assert len(state.parents) == summary['region_count']
        survivors, total = len(gt), len(state.owner)
        births = total - survivors
        assert survivors == summary['old_count']
        assert total == summary['total_count']
        assert births == summary['birth_count']
        gt_active, gr_active = gt.ne(0).any(-1), gr.ne(0).any(-1)
        active = gt_active | gr_active
        shared_active = rt.ne(0).any(-1) | rr.ne(0).any(-1)
        final = summary['flat_region_diagnostics']['regions']
        diagnostics = {int(row['region_id']): row for row in final['regions']}
        quantized = DIAGNOSTICS.quantized_motion_counts(rt, rr, gt, gr)
        expected = summary['flat_region_diagnostics']['refinement']['after']['quantized_motion']
        assert quantized == expected, (method, frame, quantized, expected)
        rows = []
        for region in range(len(state.parents)):
            all_mask = state.owner == region
            survivor_mask = all_mask[:survivors]
            total_members = int(all_mask.sum())
            assert total_members == diagnostics[region]['gs_count']
            if not total_members:
                continue
            surviving_members = int(survivor_mask.sum())
            active_members = int((survivor_mask & active).sum())
            diagonal = diagnostics[region]
            # The production online model is flat: nonzero local shared
            # symbols coincide with a nonidentity global shared transform.
            assert region == 0 or int(state.parents[region]) == 0
            has_shared = bool(shared_active[region])
            has_reported_shared = (diagonal['shared_translation_norm'] > 0 or
                                   diagonal['shared_rotation_angle_rad'] > 0)
            assert has_shared == has_reported_shared
            rows.append(dict(
                region_id=region, total_gs=total_members,
                proportion_all_pct=100 * total_members / total,
                surviving_gs=surviving_members,
                newborn_gs=total_members - surviving_members,
                residual_nonzero_gs=active_members,
                residual_nonzero_survivor_pct=(100 * active_members / surviving_members
                                               if surviving_members else None),
                translation_nonzero_gs=int((survivor_mask & gt_active).sum()),
                rotation_nonzero_gs=int((survivor_mask & gr_active).sum()),
                shared_nonzero=has_shared,
                shared_translation_xyz=diagonal['shared_translation_xyz'],
                shared_translation_norm=diagonal['shared_translation_norm'],
                shared_rotation_axis=diagonal['shared_rotation_axis'],
                shared_rotation_degrees=diagonal['shared_rotation_angle_degrees']))
        assert sum(row['total_gs'] for row in rows) == total
        assert sum(row['surviving_gs'] for row in rows) == survivors
        assert sum(row['newborn_gs'] for row in rows) == births
        assert sum(row['residual_nonzero_gs'] for row in rows) == int(active.sum())
        assert sum(row['translation_nonzero_gs'] for row in rows) == int(gt_active.sum())
        assert sum(row['rotation_nonzero_gs'] for row in rows) == int(gr_active.sum())
        assert len(rows) == final['populated_region_count']
        root = next(row for row in rows if row['region_id'] == 0)
        reports.append(dict(
            frame=frame, total_gs=total, surviving_gs=survivors, newborn_gs=births,
            region_slots=len(state.parents), populated_regions=len(rows),
            motion_regions=len(rows) - 1, empty_slots=len(state.parents) - len(rows),
            residual_nonzero_gs=int(active.sum()),
            residual_nonzero_survivor_pct=100 * int(active.sum()) / survivors,
            translation_nonzero_gs=int(gt_active.sum()),
            rotation_nonzero_gs=int(gr_active.sum()),
            root=root, shared_nonzero_regions=sum(row['shared_nonzero'] for row in rows),
            shared_nonzero_gs=sum(row['total_gs'] for row in rows if row['shared_nonzero']),
            rows=rows))
    return dict(method=method, run=str(run), frames=reports)


def vector(values):
    return '[' + ','.join(f'{value:.7g}' for value in values) + ']'


def render(report):
    method = report['method']
    lines = [
        f'# {method} 联合细化实验：最终逐区域 GS 和整数残差统计',
        '',
        f'完成 run：{report["run"]}',
        '',
        '范围为已完成的动态帧 1–4，每帧最终 births 之后、实际码流解码状态。'
        '身份 root=区域0；只列非空区域，保留空槽数量单列。总GS和百分比包含新生点。',
        '',
        'root 的共享变换恒为 identity，但其中GS仍可有非零独立残差；'
        '因此root比例大不等于所有root点都静止。区域标签只表示共享归属。',
        '',
        '独立残差非零以最终 motion 包的 int16 平移/旋转符号任一分量不为0判断，'
        '不使用浮点噪声阈值。平移非零和旋转非零可能重叠，独立非零取并集。'
        '残差百分比分母是该区域存活GS；新生点没有上一帧独立转移，残差为N/A，'
        '不能把未计入活跃数的新生点解释成零残差。',
        '',
        '共享平移是 R*x+t 的世界 affine offset，xyz及范数单位为场景世界坐标；'
        '旋转给世界轴和角度（度），identity轴记[0,0,0]。共享非零由区域整数符号判断，'
        '不表示加入独立抵消项后所有成员最终位姿都改变。变换数值来自最终frame_summary，'
        '成员/残差来自相同实际码流。',
        '',
        '验证：逐帧codec checksum/上一帧digest由原解码器验证，最终decoded digest匹配manifest；'
        '各区域GS总和匹配最终总数，存活/新生分项匹配summary，整数非零总和与'
        'refinement.after.quantized_motion逐项一致。',
        '',
        '| 帧 | 全部GS | 存活GS | 新生GS | 非空区域(含root) | 非空运动区域 | 已分配槽位 | 空槽 | 独立非零GS | 存活非零% | 非零共享区域 | 非零共享覆盖GS |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for frame in report['frames']:
        lines.append(
            f'| {frame["frame"]} | {frame["total_gs"]} | {frame["surviving_gs"]} | '
            f'{frame["newborn_gs"]} | {frame["populated_regions"]} | '
            f'{frame["motion_regions"]} | {frame["region_slots"]} | {frame["empty_slots"]} | '
            f'{frame["residual_nonzero_gs"]} | {frame["residual_nonzero_survivor_pct"]:.4f} | '
            f'{frame["shared_nonzero_regions"]} | {frame["shared_nonzero_gs"]} |')
    for frame in report['frames']:
        lines += [
            '', f'## 帧 {frame["frame"]}', '',
            '| 区域id | 全部GS | 占全部GS% | 存活GS | 新生GS | 独立非零GS | 存活非零% | 平移非零GS | 旋转非零GS | 共享非零 | 共享平移xyz | 平移范数 | 旋转轴 | 旋转角° |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---:|',
        ]
        for row in frame['rows']:
            percentage = row['residual_nonzero_survivor_pct']
            percentage = 'N/A' if percentage is None else f'{percentage:.4f}'
            lines.append(
                f'| {row["region_id"]} | {row["total_gs"]} | {row["proportion_all_pct"]:.4f} | '
                f'{row["surviving_gs"]} | {row["newborn_gs"]} | {row["residual_nonzero_gs"]} | '
                f'{percentage} | {row["translation_nonzero_gs"]} | {row["rotation_nonzero_gs"]} | '
                f'{"是" if row["shared_nonzero"] else "否"} | '
                f'{vector(row["shared_translation_xyz"])} | {row["shared_translation_norm"]:.7g} | '
                f'{vector(row["shared_rotation_axis"])} | {row["shared_rotation_degrees"]:.7g} |')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Write NEW completed-run analysis docs')
    args = parser.parse_args()
    for method, relative in RUNS.items():
        report = analyze(method, ROOT / relative)
        path = ROOT / 'docs' / 'experiment' / f'20261013_online_flat_{method}_refine_region_details.md'
        if args.write:
            if path.exists():
                raise FileExistsError(path)
            path.write_text(render(report), encoding='utf-8')
        final = report['frames'][-1]
        representatives = final['rows'] if method in ('kmeans', 'gmm') else sorted(
            (row for row in final['rows'] if row['region_id']), key=lambda row: -row['total_gs'])[:5]
        print(json.dumps(dict(method=method, path=str(path),
            frames=[{key: value for key, value in frame.items() if key != 'rows'}
                    for frame in report['frames']],
            frame4_representatives=representatives), ensure_ascii=False))


if __name__ == '__main__':
    main()
