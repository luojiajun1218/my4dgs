import os
import sys
import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from absl import logging
from tqdm import tqdm

from libgs.data.utils.fetch import Fetcher
from libgs.pipeline import Trainer as BaseTrainer
from libgs.pipeline import TrainerConfig as BaseTrainerConfig
from ..data import PairedFetcher


@dataclass
class TrainerConfig(BaseTrainerConfig):
    cache_data_on_device: bool = True
    testing_steps: List[int] = field(default_factory=lambda: [5000])
    first_frame_testing_steps: Optional[List[int]] = None
    saving_ckpt_steps: List[int] = field(default_factory=lambda: [5000])
    saving_ckpt_every_n_frames: int = 1000
    saving_ckpt_frames: Optional[List[int]] = None
    num_init_steps: int = 15000
    num_incr_steps: int = 200
    selective_console_mode: str = 'legacy'  # compact保留实时进度条，每帧结束打印两行汇总。


class Trainer(BaseTrainer):
    def setup(self, module, ckpt_path=None):
        """Restore portable CPU checkpoints onto the current training device."""
        self.module, module.trainer = module, self
        self.init_logger()
        self.datamodule.setup(self.output_dir)
        self.module.to(self.device).setup()
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.global_step = ckpt['global_step']
            self.datamodule.on_load_checkpoint(ckpt)
            self.module.on_load_checkpoint(ckpt)

    @property
    def compact_console(self):
        return (self.config.selective_console_mode == 'compact'
                and self.module.config.motion_mode in {'selective', 'learned_selective', 'quantized_residual', 'sparse_hierarchical', 'flat_cluster', 'binary_refinement', 'storage_aware_binary', 'common_contrast_binary'})

    def routine_info(self, message, *args):
        if not self.compact_console:
            logging.info(message, *args)
            return
        # Preserve routine details in a file; warnings/errors keep their usual handlers.
        text = message % args if args else str(message)
        with (self.output_dir / 'training_details.log').open('a') as stream:
            stream.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {text}\n')

    def synchronize(self):
        if (self.module.config.motion_mode in {'selective', 'learned_selective', 'quantized_residual', 'sparse_hierarchical', 'flat_cluster', 'binary_refinement', 'storage_aware_binary', 'common_contrast_binary'} and torch.cuda.is_available()
                and str(self.device).startswith('cuda')):
            torch.cuda.synchronize(self.device)

    def reset_frame_accounting(self):
        self.frame_times = dict(training=0.0, evaluation=0.0, drawing=0.0, saving=0.0)
        self.frame_eval_metrics = {}
        self.frame_diagnostic_note = None
        self.frame_image_sizes = set()

    def record_io(self, category, started):
        if not hasattr(self, 'frame_times'):
            self.reset_frame_accounting()
        self.synchronize()
        self.frame_times[category] += time.perf_counter() - started

    def frame_storage(self):
        frame = self.current_frame
        groups = {
            'checkpoint': list(self.output_dir.glob(f'ckpt-{frame}-*.pth')),
            'ply': list((self.output_dir / 'point_cloud').glob(f'gaussians-{frame}-*.ply'))
                   + list((self.output_dir / 'point_cloud').glob(f'initial-gaussians-{frame}.ply')),
            'motion_debug_and_diagnostics': list((self.output_dir / 'selective_motion').glob(f'frame_{frame:03d}.*'))
                   + list((self.output_dir / 'selective_diagnostics').glob(f'frame_{frame:03d}_*.json'))
                   + list((self.output_dir / 'frame_summary').glob(f'evaluation_{frame:03d}_step*.json'))
                   + list((self.output_dir / 'frame_summary').glob(f'quantized_frame_{frame:03d}.json')),
            'sparse_hierarchical_exports': list(
                (self.output_dir / 'frame_storage' / 'sparse_hierarchical').glob(f'*_{frame:03d}.*')),
            'gate_mechanism_diagnostics': list(
                (self.output_dir / 'gate_diagnostics').glob(f'frame_{frame:03d}*')),
            'flat_cluster_exports': list(
                (self.output_dir / 'frame_storage' / 'flat_cluster').glob(f'*_{frame:03d}.*')),
            'binary_refinement_exports': list(
                (self.output_dir / 'frame_storage' / 'binary_refinement').glob(f'*_{frame:03d}.*')),
            'storage_aware_binary_exports': list(
                (self.output_dir / 'frame_storage' / 'storage_aware_binary').glob(f'*_{frame:03d}.*')),
            'common_contrast_binary_exports': list(
                (self.output_dir / 'frame_storage' / 'common_contrast_binary').glob(f'*_{frame:03d}.*')),
            'images': [],
        }
        image_dir = self.output_dir / 'image/frames' / f'{frame:04d}'
        # Scan only this frame, never recurse across historical output directories.
        if image_dir.exists():
            for path in image_dir.rglob('*'):
                if path.is_file():
                    category = 'motion_debug_and_diagnostics' if path.suffix == '.json' else 'images'
                    groups[category].append(path)
        details = {category: [dict(path=str(path.relative_to(self.output_dir)), bytes=path.stat().st_size)
                              for path in sorted(set(paths)) if path.is_file()]
                   for category, paths in groups.items()}
        totals = {category: sum(row['bytes'] for row in rows) for category, rows in details.items()}
        return dict(bytes_by_category=totals, total_bytes=sum(totals.values()), files=details,
                    meaning='Actual training/debug artifacts, not deployment encoded size or bitrate. '
                            'Excludes this summary JSON and shared logs/TensorBoard/global metadata.')

    def finish_quantized_frame_summary(self, wall_started):
        self.synchronize()
        started = time.perf_counter()
        stream = self.module.quant_report
        report = dict(frame=self.current_frame, stream=stream,
                      quantized_evaluation=self.frame_eval_metrics,
                      evaluation_image_sizes=sorted(getattr(self, 'frame_image_sizes', set())),
                      evaluation_fps_semantics='Views divided by summed synchronized forward-and-clamp seconds; '
                                               'excludes loading, quality metrics, image writes and diagnostics.',
                      debug_storage=self.frame_storage(), timing=dict(self.frame_times))
        path = self.output_dir / 'frame_summary' / f'frame_{self.current_frame:03d}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        self.record_io('saving', started)
        report['timing'] = dict(self.frame_times, wall=time.perf_counter() - wall_started)
        if stream is not None:
            stages = stream['stage_seconds']
            report['timing']['training_loop_including_state_update'] = report['timing']['training']
            report['timing']['training'] = max(
                0., report['timing']['training'] - stages.get('frame_prepare', 0.)
                - stages.get('finalize', 0.))
        report['timing_semantics'] = (
            'CUDA synchronized at measured boundaries. training excludes frame_prepare/finalize when present; '
            'training_loop_including_state_update includes them. evaluation includes validation, '
            'excluding measured image/figure writes; deletion diagnostics are in frame_prepare. '
            'saving includes the quantized detail and first frame summary writes. '
            'gate_diagnostics measures optional traces and isolated fits and is excluded from training. '
            'wall spans frame setup, training, evaluation, drawing and saves up to the first summary write. '
            'stream_export in stream.stage_seconds is a subset of state_update/finalize, including '
            'birth packaging, motion encoding and deployment-file writes; do not add it again to wall. '
            'direct_trial_codec and direct_trial_artifact_write are subsets of direct_acceptance. '
            'The final summary rewrite and console output are excluded; components need not sum to wall.')
        path.write_text(json.dumps(report, indent=2))
        self.routine_info('Quantized frame detail: %s', json.dumps(report, separators=(',', ':')))
        quality = ' | '.join(f'{name} final PSNR={row.get("psnr", float("nan")):.2f}'
                             for name, row in self.frame_eval_metrics.items())
        clocks = report['timing']
        if stream is None:
            logging.info('帧 %d | %s | 共享完整首帧参考；后续帧估计有效参数存储',
                         self.current_frame, quality)
            return
        activity = stream['activities']
        if self.module.config.motion_mode in {'binary_refinement', 'storage_aware_binary', 'common_contrast_binary'}:
            label = ('共同运动-对比树' if self.module.config.motion_mode == 'common_contrast_binary'
                     else '存储感知二叉精度' if self.module.config.motion_mode == 'storage_aware_binary'
                     else '二叉细化')
            logging.info(
                '帧 %d %s | %s | 纯训练=%.2fs 总计=%.2fs | '
                '导出=%.2fs | 必要文件=%.4f MB 参数=%.2f KiB | 新增=%d 删除=%d',
                self.current_frame, label, quality, clocks['training'], clocks['wall'],
                stream.get('stage_seconds', {}).get('stream_export', 0.),
                stream['inference_file_bytes'] / 1e6, stream['hard_storage_bytes'] / 1024,
                stream['new_count'], stream['deleted_count'])
            self.routine_info('二叉细化活动: %s', json.dumps(activity, ensure_ascii=False))
            return
        if self.module.config.motion_mode in {'sparse_hierarchical', 'flat_cluster'}:
            gs = activity['gs']
            label = '簇' if self.module.config.motion_mode == 'flat_cluster' else 'L'
            ratios = ' '.join(
                f'{label}{row["level"] + 1}:z开={row["z_on_relevant_regions"]}/'
                f'{row["denominator"]}' for row in activity['levels'])
            logging.info(
                '帧 %d 独立门控最终评估 | %s | 纯训练=%.2fs 本帧总计=%.2fs | '
                '必要参数=%.2f KiB（含门/区域/中心）密集训练参数=%.2f KiB | '
                '旧GS=%d m开=%d m关=%d local z开=%d | %s | 新增=%d 删除=%d',
                self.current_frame, quality, clocks['training'], clocks['wall'],
                stream['hard_storage_bytes'] / 1024,
                stream.get('dense_motion_training_parameter_bytes', 0.) / 1024,
                gs['denominator'], gs['m_open'], gs['m_closed'], gs['local_z_on'],
                ratios, stream['new_count'], stream['deleted_count'])
            logging.info('门控语义 | %s；m关闭精确保持上一帧位姿；门开启数不等于实际变化GS数。',
                         '单层共享运动和局部残差' if label == '簇' else '各层z独立')
            if stream.get('inference_file_bytes') is not None:
                logging.info('模型存储 | 必要文件实测=%.4f MB/动态帧（含文件头；不含共享首帧） | '
                             '格式=%s；不计图片、训练checkpoint或完整点云检查副本。',
                             stream['inference_file_bytes'] / 1e6,
                             stream['motion_export']['motion_export_format'])
            return

        def ratio(row):
            return '/'.join(f'{row["fractions"][name] * 100:.1f}'
                            for name in ('KEEP', 'SHARED', 'SPLIT'))

        gated = self.module.config.quantized_residual.enable_learned_gates
        level_kind = '到达节点' if gated else '固定层区域'
        ratios = ' '.join(
            f'L{row["level"] + 1}={ratio(row)}(n={row["denominator"]})'
            if row['denominator'] else f'L{row["level"] + 1}=未到达(n=0)'
            for row in activity['levels'])
        logging.info(
            '帧 %d 最终量化评估 | %s | 纯训练=%.2fs 评估=%.2fs 布局=%.2fs 状态更新=%.2fs 绘图=%.2fs 本帧总计=%.2fs | '
            '有效参数=%.2f KiB（模型估算，排除文件头等） 调试/训练文件=%.2f MiB（磁盘实测）\n'
            'KEEP/SHARED/SPLIT(%%；%s) %s | 旧GS(n=%d，不含新增)=%s',
            self.current_frame, quality, clocks['training'], clocks['evaluation'],
            stream['stage_seconds'].get('layout', 0.), stream['state_update_seconds'],
            clocks['drawing'], clocks['wall'], stream['hard_storage_bytes'] / 1024,
            report['debug_storage']['total_bytes'] / 2**20, level_kind, ratios,
            activity['gs']['denominator'], ratio(activity['gs']))
        tree = stream.get('tree_setup')
        if tree:
            logging.info(
                '树布局 | action=%s 间隔=%d帧 区域数=%s→%s 初始化=%s | '
                '拓扑/校验=%.4fs 参数初始化=%.4fs 重建合计=%.4fs（均已包含在布局耗时）',
                tree['action'], tree['interval'], tree['old_region_counts'], tree['region_counts'],
                tree['initialization'], tree['topology_seconds'], tree['initialization_seconds'],
                stream['stage_seconds'].get('tree_rebuild', 0.))
        birth = stream.get('birth_control')
        if birth is not None and birth.get('enabled', True):
            details = [
                f'梯度窗口={birth.get("stats_window_start", "?")}..{birth.get("stats_window_end", "?")} '
                f'实际累计={birth.get("stats_steps_observed", "?")}步',
                f'clone 候选={birth.get("clone_eligible", "?")} 实际={birth.get("clone_added", "?")} '
                f'预算阻止={birth.get("clone_blocked", "?")}',
                f'split 净增GS={birth.get("split_added", "?")} 细分母点拦截={birth.get("split_blocked", "?")}',
                f'候选池峰值={birth.get("candidate_pool_peak", "?")} '
                f'最终候选={birth.get("final_candidate_count", "?")} '
                f'最终保留={birth.get("final_retained", "?")} '
                f'候选上限={birth.get("max_new_gaussians", "?")} 候选触顶={birth.get("budget_touched", "?")} '
                f'最终保留上限={birth.get("max_retained_gaussians", "无限制")}',
                f'候选prune={birth.get("candidate_pruned", "?")} '
                f'opacity筛除={birth.get("opacity_pruned", "?")} '
                f'最终预算筛除={birth.get("retention_budget_pruned", 0)} '
                f'上帧补偿删除={birth.get("previous_compensation_deleted", "?")}',
            ]
            history = birth.get('motion_history')
            if birth.get('candidate_prune_reasons') is not None:
                reasons = birth['candidate_prune_reasons']
                details.append(
                    f'候选裁剪原因(可重叠)：低透明度={reasons["low_opacity"]} '
                    f'屏幕过大={reasons["screen_size"]} 世界过大={reasons["world_size"]} '
                    f'多原因={reasons["multiple"]}')
            if history:
                details.append(
                    f'历史对应(非实际非零运动)：原有运动GS={history.get("old_motion_rows", "?")} '
                    f'上一帧新生GS={history.get("newborn_rows", "?")} '
                    f'首帧参考GS={history.get("first_frame_rows", "?")} '
                    f'初始化衰减={history.get("initialization_decay", "?")}')
            protection = birth.get('deletion_protection')
            if protection:
                details.append(
                    f'上帧KEEP保护={protection.get("protected_keep_old_count", "?")} '
                    f'误删KEEP={protection.get("deleted_keep_count", "?")} '
                    f'可删候选={protection.get("eligible_count", "?")}')
            for key, label in (
                ('stage2_optimizer_lrs', '新增GS学习率'),
                ('deletion_damage', '删点影响'),
                ('point_origins', '点来源'),
            ):
                if key in birth:
                    details.append(f'{label}={json.dumps(birth[key], ensure_ascii=False, separators=(",", ":"))}')
            logging.info('帧 %d 补点 | %s', self.current_frame, ' | '.join(details))

    def finish_frame_summary(self, wall_started):
        if self.module.config.motion_mode in {'quantized_residual', 'sparse_hierarchical', 'flat_cluster', 'binary_refinement', 'storage_aware_binary', 'common_contrast_binary'}:
            self.finish_quantized_frame_summary(wall_started)
            return
        from .model.selective_diagnostics import summarize
        self.synchronize()
        save_started = time.perf_counter()
        selection = summarize(self.module.selective.capture()) if self.module.selective.members else None
        storage = self.frame_storage()
        report = dict(frame=self.current_frame, final_step=self.global_step,
                      evaluation=self.frame_eval_metrics, selection=selection,
                      evaluation_fps_semantics='Views divided by summed synchronized forward-and-clamp seconds; '
                                               'excludes loading, quality metrics, image writes and diagnostics.',
                      diagnostic_note=self.frame_diagnostic_note, storage=storage,
                      timing=dict(self.frame_times),
                      timing_semantics='CUDA synchronized at measured boundaries. Training is the training-step loop including data/selection, '
                                       'and evaluation includes all validation steps; both exclude measured drawing/saving. '
                                       'Drawing includes diagnostic figures and their JSON/file writes; saving covers other saves/report preparation. '
                                       'Wall is a whole-frame snapshot after the first report write, including setup/data/all evaluations/drawing/saving; '
                                       'it excludes the final report rewrite and console output and need not equal the component sum.')
        path = self.output_dir / 'frame_summary' / f'frame_{self.current_frame:03d}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        self.record_io('saving', save_started)
        report['timing'] = dict(self.frame_times, wall=time.perf_counter() - wall_started)
        path.write_text(json.dumps(report, indent=2))
        quality = ' | '.join(f'{name}: PSNR={row.get("psnr", float("nan")):.2f} SSIM={row.get("ssim", float("nan")):.4f}'
                             for name, row in self.frame_eval_metrics.items())
        if selection:
            def shares(key, names):
                row = selection[key]
                return '/'.join(f'{name}={row["fractions"][name]:.1%}' for name in names)
            active = shares('active_tree', ('KEEP', 'SHARED', 'SPLIT'))
            gs = shares('gs_weighted', ('KEEP', 'SHARED'))
            if selection.get('mode') == 'learned_selective':
                states = f'ACTIVE n={selection["active_tree"]["denominator"]} {active}; GS n={selection["gs_weighted"]["denominator"]} {gs}'
                policy = 'fixed real leaf cut' if selection.get('fixed_leaf_ablation') else 'fixed per-frame learned cut'
                motion = f'supported candidates={selection["observed_candidate_nodes"]}/{selection["candidate_nodes"]}; {policy}'
            else:
                unknown = selection['gs_weighted']['unknown'] / selection['gs_weighted']['denominator']
                internal = selection['comparison_coverage']['internal']
                motion = (f'SHARED depths={selection["terminal_regions"]["depth_by_state"]["SHARED"]}; '
                          f'internal blocked={internal["shared_blocked"]}/{internal["compared"]}')
                states = (f'ACTIVE n={selection["active_tree"]["denominator"]} {active}; '
                          f'GS n={selection["gs_weighted"]["denominator"]} {gs}, '
                          f'KEEP(insufficient evidence)={unknown:.1%}')
        else:
            states, motion = 'selective tree unavailable (first frame)', 'no motion cut'
        clocks = report['timing']
        files = storage['bytes_by_category']
        self.routine_info('Frame %d complete | %s\n%s | %s\n'
                     'seconds train=%.2f eval=%.2f draw=%.2f save=%.2f wall=%.2f | '
                     'training artifacts MiB ckpt=%.2f PLY=%.2f motion/debug=%.2f images=%.2f total=%.2f (not encoded bitrate)',
                     self.current_frame, quality, states, motion,
                     clocks['training'], clocks['evaluation'], clocks['drawing'], clocks['saving'], clocks['wall'],
                     files['checkpoint'] / 2**20, files['ply'] / 2**20,
                     files['motion_debug_and_diagnostics'] / 2**20, files['images'] / 2**20, storage['total_bytes'] / 2**20)
        if self.compact_console:
            if selection:
                fractions = selection['active_tree']['fractions']
                ratio = '/'.join(f'{100 * fractions[name]:.1f}' for name in ('KEEP', 'SHARED', 'SPLIT'))
                kept = selection['gs_weighted']['fractions']['KEEP']
                brief = f'区域 保留/共享/分裂={ratio}% | GS保留={kept:.1%}'
            else:
                brief = '首帧重建（无运动划分）'
            logging.info('帧 %d完成 | %s | 训练 %.1fs · 总计 %.1fs\n%s | 本帧文件 %.1f MiB',
                         self.current_frame, quality or '本帧未评估', clocks['training'], clocks['wall'],
                         brief, storage['total_bytes'] / 2**20)

    def progress_label(self, phase):
        """Status text only; it does not control the training stage."""
        names = {'setup': '准备', 'initial': '首帧训练', 'motion': '运动拟合',
                 'tree_probe': '建树探针', 'dense_fit': '运动拟合', 'gate_compress': '门压缩',
                 'soft_warmup': '预热', 'hard_joint': '联合选择', 'fixed_refine': '精修', 'fixed_leaf': '固定细区域运动',
                 'shared_warmup': '共享运动预热', 'joint_residual': '残差联合训练',
                 'gate_joint': '门控与运动联合学习', 'gate_refine': '固定门控精修',
                 'probe': '运动探测', 'shared_fit': '共享拟合', 'refine': '精修',
                 'incremental': '新增GS', 'saving': '保存', 'evaluation': '评估/绘图'}
        return f'帧 {self.current_frame}/{self.num_frames - 1} · {names.get(phase, phase)}'

    @property
    def num_frames(self) -> int:
        return self.datamodule.num_frames

    @property
    def current_frame(self) -> int:
        return self.datamodule.current_frame

    @property
    def is_first_frame(self) -> bool:
        return self.current_frame == 0  # range from 0

    @property
    def is_second_frame(self) -> bool:
        return self.current_frame == 1

    @property
    def is_last_frame(self) -> bool:
        return self.current_frame == self.num_frames - 1  # range from 0

    @property
    def num_steps(self) -> int:
        if self.current_frame == 0:
            return self.config.num_init_steps
        return self.config.num_incr_steps

    @property
    def is_first_step(self) -> bool:
        return self.global_step == 1  # range from 1

    @property
    def is_last_step(self) -> bool:
        return self.global_step == self.num_steps  # range from 1

    @property
    def testing_steps_for_frame(self) -> List[int]:
        if self.is_first_frame and self.config.first_frame_testing_steps is not None:
            return self.config.first_frame_testing_steps
        return self.config.testing_steps

    @property
    def ckpt_save_path(self) -> Path:
        filename = f"ckpt-{self.current_frame}-{self.global_step}.pth"
        return self.output_dir / filename

    def training_loop(self):
        next_frame = self.datamodule.current_frame + 1
        if self.global_step > 0:  # restored from a checkpoint
            next_frame = self.datamodule.current_frame
        for current_frame in range(next_frame, self.datamodule.num_frames):
            self.routine_info(f"\n==> Start learning frame {current_frame} ...")
            self.datamodule.current_frame = current_frame
            self._progress_bar = None
            try:
                self.frame_training_loop()
            finally:
                if self._progress_bar is not None:
                    self._progress_bar.close()
                    self._progress_bar = None

    def frame_training_loop(self):
        if self.global_step >= self.num_steps:
            label = '首帧' if self.is_first_frame else f'帧 {self.current_frame}'
            logging.info('%s已从检查点恢复，跳过训练（%d步，GS=%d）。',
                         label, self.global_step, len(self.module.gaussians.get_xyz))
            self.global_step = 0
            return
        self.synchronize()
        wall_started = time.perf_counter()
        self.reset_frame_accounting()
        train_dataloader = self.datamodule.train_dataloader()
        self.routine_info(f"Number of views in training dataset: {len(train_dataloader)}")
        cache = self.config.cache_data_on_device
        move_to_device = lambda ts: ts.to(self.device, True)
        fetcher = Fetcher(train_dataloader, cache, move_to_device)
        view_seed = self.datamodule.config.training_view_seed
        if view_seed is not None:
            fetcher = PairedFetcher(train_dataloader, cache, move_to_device,
                                    view_seed + self.current_frame)
        self.frame_training_views = []

        metrics_ema = {"loss": 0.0, "psnr": 0.0}

        def update_metric(name, value):
            metrics_ema[name] *= 0.6
            metrics_ema[name] += 0.4 * value

        interactive_progress = (
            sys.stderr.isatty()
            and os.environ.get("TERM") != "dumb"
        )
        progress_bar = (
            tqdm(
                total=self.num_steps,
                initial=self.global_step,
                desc=self.progress_label('setup'),
                dynamic_ncols=True,
                leave=not self.compact_console,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                           "{n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
            )
            if interactive_progress else None
        )
        self._progress_bar = progress_bar
        last_progress_step = self.global_step
        plain_log_interval = 100

        for self.global_step in range(self.global_step + 1, self.num_steps + 1):
            self.synchronize()
            train_started = time.perf_counter()
            io_before = self.frame_times['drawing'] + self.frame_times['saving']
            self.module.pre_training_step()

            self.module.timer.clock("data").start()
            train_view = fetcher.next()
            if view_seed is not None:
                self.frame_training_views.append(train_view.path.name)
            args = (train_view, self.global_step)
            self.module.timer.clock("data").stop()
            loss, metrics, render_pkg = self.module.training_step(*args)
            self.module.timer.clock("backward").start()
            loss.backward()
            if (self.module.config.motion_mode in {'quantized_residual', 'sparse_hierarchical', 'flat_cluster', 'binary_refinement', 'storage_aware_binary', 'common_contrast_binary'}
                    and self.module.config.quantized_diagnostics.gradient_steps is not None
                    and self.global_step in self.module.config.quantized_diagnostics.gradient_steps):
                self.module.record_quantized_gradients()
            self.module.timer.clock("backward").stop()

            diagnostic_seconds = self.module.record_gate_diagnostics(train_view)
            if diagnostic_seconds:
                self.frame_times['gate_diagnostics'] = (
                    self.frame_times.get('gate_diagnostics', 0.) + diagnostic_seconds)
                render_pkg['quantized_started'] += diagnostic_seconds

            self.module.timer.clock("progress updating").start()
            update_metric("loss", loss.item())
            update_metric("psnr", metrics["psnr"].item())
            show_progress = (
                self.is_last_step
                or (progress_bar is not None and self.global_step % 10 == 0)
                or (progress_bar is None
                    and self.global_step % plain_log_interval == 0)
            )
            if show_progress:
                metrics_text = (
                    f"train_ema_loss={metrics_ema['loss']:.4f} "
                    f"train_ema_psnr={metrics_ema['psnr']:.2f} gs={metrics['gs']}"
                )
                if progress_bar is not None:
                    phase = render_pkg.get('quantized_stage', render_pkg.get('selective_stage'))
                    if phase is None:
                        phase = ('initial' if self.is_first_frame else 'incremental'
                                 if self.global_step > self.module.deform.config.num_stage1_steps else 'motion')
                    progress_bar.set_description_str(self.progress_label(phase), refresh=False)
                    progress_bar.set_postfix_str(metrics_text, refresh=False)
                    progress_bar.update(self.global_step - last_progress_step)
                    last_progress_step = self.global_step
                else:
                    self.routine_info(
                        "Training frame %d: %d/%d (%.1f%%) %s",
                        self.current_frame, self.global_step, self.num_steps,
                        100 * self.global_step / self.num_steps,
                        metrics_text,
                    )
            self.module.timer.clock("progress updating").stop()

            gate_progress_bar = None
            gate_progress = None
            deform = self.module.deform
            gate_due = getattr(deform, "gate_due", None)
            if (not self.is_first_frame and render_pkg.get("delta_xyz") is not None
                    and callable(gate_due) and gate_due(self.global_step)):
                gate_level = deform.active_gate_count + 2
                if interactive_progress:
                    gate_progress_bar = tqdm(
                        total=4,
                        desc=f"Frame {self.current_frame} L{gate_level} gate",
                        position=1,
                        leave=False,
                        dynamic_ncols=True,
                        mininterval=0,
                    )
                else:
                    logging.info(
                        "Frame %d: opening L%d gate at step %d",
                        self.current_frame, gate_level, self.global_step,
                    )
                completed_gate_phases = 0

                def gate_progress(phase: str):
                    nonlocal completed_gate_phases
                    completed_gate_phases += 1
                    if gate_progress_bar is not None:
                        gate_progress_bar.set_postfix_str(phase, refresh=False)
                        gate_progress_bar.update(1)
                        if completed_gate_phases == gate_progress_bar.total:
                            gate_progress_bar.close()
                    else:
                        logging.info(
                            "Frame %d L%d gate: %d/4 %s",
                            self.current_frame, gate_level, completed_gate_phases, phase,
                        )

            try:
                self.module.post_training_step(render_pkg, gate_progress=gate_progress)
            finally:
                if gate_progress_bar is not None:
                    gate_progress_bar.close()

            self.synchronize()
            self.frame_times['training'] += max(0.0, time.perf_counter() - train_started -
                (self.frame_times['drawing'] + self.frame_times['saving'] - io_before) - diagnostic_seconds)
            saving_ckpt_steps = self.config.saving_ckpt_steps + [self.num_steps]
            if self.global_step in saving_ckpt_steps:
                if progress_bar is not None:
                    progress_bar.set_description_str(self.progress_label('saving'))
                save_started = time.perf_counter()
                self.save_checkpoint()
                self.record_io('saving', save_started)

            if self.global_step in self.testing_steps_for_frame + [self.num_steps]:
                if progress_bar is not None:
                    progress_bar.set_description_str(self.progress_label('evaluation'))
                self.validation_loop()
        if progress_bar is not None:
            progress_bar.close()
            self._progress_bar = None
        if self.module.config.motion_mode in {'selective', 'learned_selective', 'quantized_residual', 'sparse_hierarchical', 'flat_cluster', 'binary_refinement', 'storage_aware_binary', 'common_contrast_binary'}:
            if view_seed is not None:
                path = self.output_dir / 'frame_summary' / f'training_views_{self.current_frame:03d}.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(frame=self.current_frame,
                    seed=view_seed, views=self.frame_training_views), indent=2))
            self.finish_frame_summary(wall_started)
        self.global_step = 0  # !!!Important

    @torch.no_grad()
    def validation_loop(self):
        if not hasattr(self, 'frame_times'):
            self.reset_frame_accounting()
        self.synchronize()
        evaluation_started = time.perf_counter()
        io_before = self.frame_times['drawing'] + self.frame_times['saving']
        torch.cuda.empty_cache()

        loaders = self.datamodule.val_dataloader()
        if not isinstance(loaders, (list, tuple)):
            loaders = [loaders]

        results_list = []
        view_rows = []
        for loader_idx, loader in enumerate(loaders):
            self.routine_info(
                f"Number of views in testing dataset {loader_idx}: {len(loader)}"
            )
            results = []
            for idx, viewpoint in enumerate(loader):
                viewpoint = viewpoint.to(self.device)
                metrics = self.module.validation_step(viewpoint, idx, loader_idx)
                results.append(metrics)
                view_rows.append(dict(
                    split=self.datamodule.eval_names[loader_idx], camera=viewpoint.path.name,
                    **{key: float(value) for key, value in metrics.items()}))
            results_list.append(results)

        num_loaders = len(loaders)
        if num_loaders == 1:
            results_list = results_list[0]
        self.module.validation_end(results_list, num_loaders)
        if self.module.config.motion_mode in {'selective', 'learned_selective', 'quantized_residual', 'sparse_hierarchical', 'flat_cluster', 'binary_refinement', 'storage_aware_binary', 'common_contrast_binary'}:
            self.synchronize()
            started = time.perf_counter()
            path = self.output_dir / 'frame_summary' / f'evaluation_{self.current_frame:03d}_step{self.global_step}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(frame=self.current_frame, step=self.global_step,
                                            evaluation=self.frame_eval_metrics,
                                            views=view_rows), indent=2))
            self.record_io('saving', started)

        torch.cuda.empty_cache()
        self.synchronize()
        self.frame_times['evaluation'] += max(0.0, time.perf_counter() - evaluation_started -
            (self.frame_times['drawing'] + self.frame_times['saving'] - io_before))

    def checkpoint_frame_due(self):
        frames = self.config.saving_ckpt_frames
        due = (self.current_frame in frames if frames is not None else
               self.current_frame % self.config.saving_ckpt_every_n_frames == 0)
        return self.is_last_frame or due

    @torch.no_grad()
    def save_checkpoint(self):
        if self.checkpoint_frame_due():
            if self.compact_console:
                path = self.ckpt_save_path
                if path is None:
                    logging.warning('Skip checkpoint saving due to save_path is None')
                    return
                self.routine_info('Saving checkpoint to %s', path)
                ckpt = dict(global_step=self.global_step)
                self.module.on_save_checkpoint(ckpt)
                self.datamodule.on_save_checkpoint(ckpt)
                torch.save(ckpt, path)
            else:
                super().save_checkpoint()
            return
