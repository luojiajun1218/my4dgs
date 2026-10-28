"""Implementation of docs/frame_training_plan.md, selected explicitly in YAML."""
import math
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from .model.birth_quantization import RAW_FIELDS, RAW_ATTRIBUTES, model_raw, QuantizedNewGaussianView
from .model.frame_plan_birth import FramePlanBirthModel, nearest_old_references
from .model.early_replacement_model import _rotate
from .model.residual_birth import RawView
from .model.support_reincarnation_proposal import _native_responsibility, _sample
from .model.adaptive_birth_storage import newborn_storage_terms
from .model.rate_budget import RateBudgetState
from .model.residual_support_geometry import residual_support_seed,residual_support_opacity


def final_joint_scheduled(module, step):
    """The optional tail fits old motion with the complete newborn population."""
    if not getattr(module.config.birth, 'frame_training_plan', False):
        return False
    count = module.config.rigid_tree.spatial_gate.final_joint_refine_steps
    return bool(count and 301-count <= step <= 300)


def fitting_old_motion(module):
    # motion_committed is also the newborn-render/checkpoint readiness flag.
    # Keep it true after birth200, including while old motion learns again.
    return module.frame_prepared and (not module.motion_committed or (
        not module.frame_finalized and getattr(module, 'early_initialized', False) and
        final_joint_scheduled(module, module.motion.last_step)))


class FramePlanController:
    def _plan_setup_rate_budget(self):
        cfg=self.config.rigid_tree
        if not cfg.target_update_bytes:
            self.motion.rate_multiplier=1.
            return
        state=RateBudgetState.restore(getattr(self,'_rate_budget_state',None),cfg.max_rate_multiplier)
        if self.current_frame!=state.last_frame+1:
            raise ValueError('Frame budget state does not precede the current frame')
        self.motion.rate_multiplier=state.multiplier
        self._rate_budget_state=state.capture()

    def _plan_update_rate_budget(self):
        cfg=self.config.rigid_tree
        if not cfg.target_update_bytes:
            return
        state=RateBudgetState.restore(self._rate_budget_state,cfg.max_rate_multiplier)
        report=state.update(int(self.current_frame),int(self.frame_report['encoded_bytes']),
            cfg.target_update_bytes,cfg.rate_controller_gain,cfg.max_rate_multiplier)
        report['effective_rate_weight']=self.motion.effective_rate_loss_weight
        self.frame_report['rate_budget']=report
        self._rate_budget_state=state.capture()

    def frame_plan_training_views(self, step, count):
        # Existing per-frame cache loads training images once; it does not render them.
        cached = self.transaction_camera_views()
        dataset = list(cached.values())
        if count != 4 or len(dataset) < count:
            raise ValueError('Frame plan requires four distinct training views per step')
        # Stateless seeded sampling also reproduces exact camera sets on resume.
        seed = int(self.datamodule.config.training_view_seed or 0)
        generator = torch.Generator().manual_seed(seed+1000003*self.current_frame+int(step))
        indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
        self._plan_step_views = [dataset[i] for i in indices]
        names = [Path(str(view.path)).name for view in self._plan_step_views]
        if len(set(names)) != count or 'cam00.png' in names:
            raise ValueError('Frame plan sampled duplicate or held-out training cameras')
        return self._plan_step_views

    def _plan_record_phase_time(self):
        started = getattr(self,'_plan_phase_started',None)
        if started is None:
            return
        if torch.device(self.device).type == 'cuda':
            torch.cuda.synchronize(self.device)
        phase = self._plan_timed_phase
        times = self.birth_control.setdefault('phase_seconds',{})
        times[phase] = times.get(phase,0.)+time.perf_counter()-started
        self._plan_phase_started = None

    def _plan_prepare_rates(self, step):
        tail = final_joint_scheduled(self, step)
        phase = ('final_joint_refine' if tail else
                 'shared_gate' if step <= 100 else 'fixed_gate_residual' if step <= 150 else
                 'joint_gate_residual' if step <= 180 else 'fixed_gate_motion' if step <= 200 else
                 'newborn_old_alpha')
        if getattr(self,'_plan_timed_phase',None) != phase:
            self._plan_record_phase_time()
        if getattr(self,'_plan_phase_started',None) is None:
            self._plan_timed_phase = phase
            self._plan_phase_started = time.perf_counter()
        self._early_attach_opacity()
        if not self.motion_committed or tail:
            self.motion.prepare_plan_step(step)
        update = self.opacity_update
        update.delta.requires_grad_(step >= 101)
        if step <= 100:
            update.delta.grad = None
        self._early_prepare_rates_legacy(step)
        if isinstance(self.births, FramePlanBirthModel):
            self.births.set_rates(step, self.config.gaussian_stage2, self.config.birth.lr_final_ratio)

    @torch.no_grad()
    def _plan_post_step(self, packages, step):
        if step != self.early_shared_steps+1 or not 1 <= step <= 300:
            raise ValueError('Frame plan requires one shared Adam update per iteration')
        tail_start = 301-self.config.rigid_tree.spatial_gate.final_joint_refine_steps
        boundaries = (1,100,101,150,151,180,181,200,201,250,251,300)
        if step in boundaries or (tail_start <= 300 and step in (tail_start-1,tail_start)):
            parameters = {name: getattr(self.motion,name) for name in
                          ('gate_logits','region_translation','gs_translation','region_rotation','gs_rotation')}
            parameters['old_alpha'] = self.opacity_update.delta
            if isinstance(self.births,FramePlanBirthModel):
                for index,batch in enumerate(self.births.cohorts):
                    parameters.update({f'birth{index}_{name}': value for name,value in batch['parameters'].items()})
            self.birth_control.setdefault('frame_plan_gradients',[]).append(dict(step=step, parameters={
                name: dict(trainable=value.requires_grad, has_gradient=value.grad is not None,
                           gradient_norm=float(value.grad.norm()) if value.grad is not None else 0.,
                           adam_updates=int(self.motion.optimizer.state.get(value,{}).get('step',0)))
                for name,value in parameters.items()}))
        count = len(self.gaussians._xyz)
        for package in packages:
            observed = package['motion_contribution_filter'][:count]
            self.motion.mark_observed(observed)
            self.birth_control['motion_frustum_samples'] = torch.as_tensor(self.birth_control.get('motion_frustum_samples',0),device=self.device)+package['visibility_filter'][:count].sum()
            self.birth_control['motion_observed_samples'] = torch.as_tensor(self.birth_control.get('motion_observed_samples',0),device=self.device)+observed.sum()
        if fitting_old_motion(self):
            edits = self.motion.optimizer_step()
            self.birth_control['tree_edit_count'] = self.birth_control.get('tree_edit_count', 0)+len(edits)
        else:
            self.motion.optimizer.step()
            self.motion.optimizer.zero_grad(set_to_none=True)
        if step >= 101:
            self._early_sync_opacity_state()
            window = self.config.rigid_tree.motion_fit_window_steps
            # The old-alpha carrier keeps learning after old motion commits.
            # Its rate terms detach in proximal mode, so skipping these birth-
            # phase windows silently disables its nonzero-update byte price.
            opacity_proximal = (self.motion._motion_fit_window_due() if step <= 200 else
                window is None or (step-200) % window == 0 or step == 300)
            self.opacity_update.optimizer_step(self._rate_byte_price(step),
                proximal=opacity_proximal,
                lr_factor=self.motion._motion_lr_factor(min(step, 200))
                    if self.config.opacity_update.decay_with_motion else 1.,
                perform_step=False, external_state=self.motion.optimizer.state.get(self.opacity_update.delta, {}))
        self.early_shared_steps += 1
        if self.early_initialized:
            self.births.project_parameters()
            self.early_birth_updates += 1
        if step == 200:
            xyz, rotation = self.motion.decoded_survivor_pose()
            self.gaussians.apply_selected_motion(xyz.to(self.device), rotation.to(self.device))
            for parameter in self.motion.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            self.motion_committed = True
            self.births = FramePlanBirthModel(self.config.sh_degree, self.device,
                logit_opacity=getattr(self.config.birth,'residual_support_init',False))
            self.births.spatial_lr_scale = self.gaussians.spatial_lr_scale
            self.births.active_sh_degree = self.gaussians.active_sh_degree
            self.early_initialized = True
            self._plan_used_quota = 0
            self._plan_events = []
            self.birth_control['early_replacement'] = dict(mode='frame_training_plan', source_count=0,
                birth_steps=[200, 250], cleanup_steps=[200, 250, 300], population_screening=False,
                colour_initialization='responsibility-weighted training RGB to SH DC; higher SH zero',
                reference_distance='0.5 squared position/extent + 0.5 mean squared RGB DC; ties lowest old row',
                storage_loss=dict(units='surrogate bytes; not compressed bytes', enabled=True))
        if step == 300 and final_joint_scheduled(self, step):
            # Canonical old attributes must reflect the final quantized pose
            # before deletion/reanchoring and before the codec checks it.
            xyz, rotation = self.motion.decoded_survivor_pose()
            self.gaussians.apply_selected_motion(xyz.to(self.device), rotation.to(self.device))
        if step in (200, 250, 300):
            self._plan_cleanup(step)
            if step < 300:
                self._plan_birth(step)
            else:
                self._plan_finish()

    @torch.no_grad()
    def _plan_cleanup(self, step):
        # Snapshot quantized children before changing any references/old rows.
        reference = self._adaptive_reference()
        reference['opacity'] = self.opacity_update.canonical_logits().to(self.device)
        view = self.make_birth_view(reference=reference)
        newborn = ({name: view.raw_value(name).detach().clone() for name in RAW_FIELDS}
                   if view is not None else None)
        if newborn is not None:
            # Match the rotation materialized by the attribute/pose codecs.
            newborn['rotation'] = view.get_rotation.detach().clone()
            alive = newborn['opacity'].sigmoid().flatten() >= .01
            selected = alive.nonzero(as_tuple=True)[0]
            self.births.select_rows_inplace(selected, self.motion.optimizer)
            newborn = {k: v[selected] for k, v in newborn.items()}
            self.births.set_actual(newborn)
            new_removed = int((~alive).sum())
        else:
            new_removed = 0
        keep = (reference['opacity'].sigmoid().flatten() >= .01).nonzero(as_tuple=True)[0]
        if not len(keep):
            raise ValueError('Frame plan: all old GS have alpha < .01; no decodable birth anchor remains')
        remove = torch.ones(len(reference['xyz']), dtype=torch.bool, device=self.device)
        remove[keep] = False
        if remove.any():
            # Keep deleted static attributes indexed in the original previous
            # frame layout, allowing a post-deletion checkpoint to bootstrap.
            deleted_ids = self.motion.previous_rows[remove].cpu()
            attributes = {name: getattr(self.gaussians, attr).detach()[remove].cpu().clone()
                          for name, attr in zip(RAW_FIELDS[2:], RAW_ATTRIBUTES[2:])}
            if self.retired_reference is None:
                self.retired_reference = dict(rows=deleted_ids, attributes=attributes)
            else:
                self.retired_reference['rows'] = torch.cat((self.retired_reference['rows'], deleted_ids))
                for name in attributes:
                    self.retired_reference['attributes'][name] = torch.cat((
                        self.retired_reference['attributes'][name], attributes[name]))
            self.gaussians.prune_points(remove)
            self.motion.retain_survivors(keep)
        update = self.opacity_update
        update.commit(update.rows[keep])  # Keep original delta parameter/moments intact.
        update.delta.requires_grad_(step < 300)
        self.gaussians._opacity.copy_(update.canonical_logits().to(self.device))
        reference = {name: value[keep] for name, value in reference.items()}
        reanchor = {}
        if newborn is not None and len(newborn['xyz']):
            self.births.source_rows = nearest_old_references(newborn, reference, self.datamodule.cameras_extent)
            changed_view = self.make_birth_view(reference=reference)
            for name in RAW_FIELDS:
                error = (changed_view.raw_value(name)-newborn[name]).abs()
                reanchor[name] = float(error.max()) if error.numel() else 0.
                # Quaternion sign changes are handled by the quantizer.
                if name != 'rotation' and reanchor[name] > changed_view.steps[name]*.501+1e-6:
                    raise ValueError(f'Reanchored birth {name} exceeded half-bin reconstruction error')
        event = dict(step=step, removed_old=int(remove.sum()), removed_new=new_removed,
                     old_count=len(keep), new_count=len(self.births._xyz), reanchor_max_error=reanchor,
                     evidence_time='after optimizer update and cleanup; before birth')
        self._plan_events.append(event)
        self.birth_control['frame_plan_events'] = self._plan_events

    @torch.no_grad()
    def _plan_birth(self, step):
        reference = self._adaptive_reference()
        reference['opacity'] = self.opacity_update.canonical_logits().to(self.device)
        view = self.make_birth_view(reference=reference)
        raw = {name: value.clone() for name, value in reference.items()}
        if view is not None:
            for name in RAW_FIELDS:
                raw[name] = torch.cat((raw[name], view.raw_value(name).detach()))
        old = RawView(raw, self.config.sh_degree, self.gaussians)
        proposal = SimpleNamespace(config=self.config, gaussians=old, renderer=self.renderer, background=self.background)
        views = {Path(str(v.path)).name: v for v in self._plan_step_views}
        support_init=getattr(self.config.birth,'residual_support_init',False)
        evidence_args=dict(residual_floor=self.config.birth.residual.residual_floor,collect_colour_evidence=True)
        if support_init:evidence_args['collect_support_moments']=True
        scores, observed, _, report, cache = _native_responsibility(proposal, views,**evidence_args)
        eligible = ((scores > 0) & observed.ne(0)).nonzero(as_tuple=True)[0]
        limit = min(self.config.birth.max_new_gaussians, self.config.birth.residual.max_candidates)
        cap = min(limit-self._plan_used_quota,
                  max(0, self.config.birth.residual.max_gaussians-len(raw['xyz'])))
        if step == 200:
            cap = min(cap, limit//2)
        # A parent may have contribution in only one of the four views.
        rows = eligible[scores[eligible].argsort(descending=True, stable=True)[:max(0, cap)]]
        seed = {name: value[rows].clone() for name, value in raw.items()}
        support_report={}
        if support_init:
            geometry,valid_geometry,support_report=residual_support_seed(raw,rows,cache,
                self.config.birth.new_xyz_step_extent*self.datamodule.cameras_extent)
            seed.update(geometry)
            seed['opacity']=residual_support_opacity(seed['opacity'],rows,cache)
            support_report.update(alpha_semantics='minimum of parent alpha and native support-weighted residual RGB RMS prior; not measured opacity',
                initial_alpha_mean=float(seed['opacity'].sigmoid()[valid_geometry].mean()) if bool(valid_geometry.any()) else None)
            rows=rows[valid_geometry]
            seed={name:value[valid_geometry] for name,value in seed.items()}
        else:
            index = torch.arange(len(rows), device=self.device, dtype=raw['xyz'].dtype)+self._plan_used_quota
            direction = torch.stack((torch.sin(index*2.399), torch.cos(index*2.399), torch.sin(index*1.37)), 1)
            seed['xyz'] += _rotate(seed['rotation'], direction*seed['logscale'].exp()*self.config.birth.clone_sigma_scale)
        # Legacy alpha is copied; relocated support starts with a conservative
        # image-evidence amplitude prior, then learns through the same alpha carrier.
        rgb = seed['xyz'].new_zeros(len(rows), 3)
        mass = seed['xyz'].new_zeros(len(rows))
        for item in cache:
            camera = item['view']
            matrix = camera.world_view_transform.to(seed['xyz'])
            projected = seed['xyz']@matrix[:3, :3]+matrix[3, :3]
            z = projected[:, 2]
            grid = torch.stack((projected[:, 0]/z.clamp_min(1e-12)/math.tan(float(camera.fovx)*.5),
                                projected[:, 1]/z.clamp_min(1e-12)/math.tan(float(camera.fovy)*.5)), -1)
            valid = (z > 0) & (grid.abs() < 1).all(-1) & item['observed'][rows]
            weight = item['native_mass'][rows]*valid
            rgb += _sample(item['image'], grid)*weight[:, None]
            mass += weight
        supported = mass > 0
        rows = rows[supported]
        seed = {name: value[supported] for name, value in seed.items()}
        rgb = (rgb[supported]/mass[supported, None]).clamp(0, 1)
        seed['sh_dc'] = ((rgb-.5)/.28209479177387814)[:, None]
        seed['sh_rest'].zero_()
        refs = nearest_old_references(seed, reference, self.datamodule.cameras_extent)
        self.births.add(seed, rows, refs, step, self.motion.optimizer, self.config.gaussian_stage2)
        self._plan_used_quota += len(rows)
        self.birth_control['early_replacement']['source_count'] = self._plan_used_quota
        self.birth_control['proposal_count'] = self._plan_used_quota
        self._plan_events[-1].update(born=len(rows), used_quota=self._plan_used_quota,
            proposal_cameras=list(views), proposal=report, image_colour_supported=len(rows),
            error_before_birth=report['raw_old_RGB_SSE'],
            birth_jump_error_semantics='unfiltered RGB SSE before and after on identical proposal training cameras')
        if support_init:self._plan_events[-1]['support_initialization']=support_report
        # Only these same four training views are used for the post-birth jump diagnostic.
        after = dict(reference)
        quantized = self.make_birth_view(reference=reference)
        if quantized is not None:
            after = {name: torch.cat((after[name], quantized.raw_value(name).detach())) for name in RAW_FIELDS}
        previous = self.renderer.gaussians
        collection = self.renderer.collect_contributions
        try:
            self.renderer.gaussians = RawView(after, self.config.sh_degree, self.gaussians)
            self.renderer.collect_contributions = False
            error = sum(float((self.renderer(v, self.background)['render']-v.image).square().sum()) for v in views.values())
        finally:
            self.renderer.gaussians, self.renderer.collect_contributions = previous, collection
        self._plan_events[-1]['error_after_birth'] = error

    def _plan_storage_regularization(self, step, birth_view=None):
        view = self.make_birth_view(reference=self._adaptive_reference()) if birth_view is None else birth_view
        if view is None:
            zero = self.gaussians._xyz.new_zeros(())
            return zero, dict(birth_storage_loss=zero)
        # Each cohort's own age controls warmup, never the global step250 clock.
        ramps = []
        for batch in self.births.cohorts:
            age = step-batch['birth_step']
            cfg = self.config.birth
            ramp = min(max((age-cfg.storage_warmup_steps)/max(1, cfg.storage_ramp_steps), 0.), 1.)
            ramps.append(view.source.new_full((len(batch['parameters']['_xyz']),), ramp, dtype=view.get_xyz.dtype))
        view.storage_row_weights = torch.cat(ramps)
        terms = newborn_storage_terms(view, self.config.birth.opacity_threshold)
        price = self.config.birth.storage_loss_strength*self._rate_byte_price(step)
        loss = terms['surrogate_bytes']*price
        return loss, {**{f'birth_storage_{name}': value.detach() for name, value in terms.items()},
                      'birth_storage_loss': loss.detach()}

    def _plan_population_regularization(self, step):
        # Uniform representation pressure complements survival-priced bytes.
        # The same a/(a+.01) penalty applies to old/new decoded alpha.
        old = self.opacity_update.alphas().flatten()
        view = self.make_birth_view(reference=self._adaptive_reference()) if self.early_initialized else None
        new = view.get_opacity.flatten() if view is not None else old[:0]
        threshold = self.config.birth.opacity_threshold
        old_support = (old/(old+threshold)).sum()
        new_support = (new/(new+threshold)).sum()
        penalty = (old_support+new_support)*self.config.birth.population_loss_weight/max(1,len(old)+len(new))
        return penalty, dict(population_loss=penalty.detach(), population_old_support=old_support.detach(),
                             population_new_support=new_support.detach())

    @torch.no_grad()
    def _plan_finish(self):
        self._plan_record_phase_time()
        for name in ('motion_frustum_samples','motion_observed_samples'):
            self.birth_control[name] = int(self.birth_control.get(name,0))
        self.birth_control['phase_timing_scope'] = 'synchronized wall seconds; includes topology events; excludes checkpoint writes and final encoding'
        self.motion.record_final_diagnostics()
        tail_steps = self.config.rigid_tree.spatial_gate.final_joint_refine_steps
        self.birth_control['early_replacement'].update(global_iterations=300, motion_iterations=200+tail_steps,
            final_joint_refine_steps=tail_steps,
            newborn_iterations=self.early_birth_updates, extra_SGD_steps=0, codec_trials200=0,
            actual_birth_cohorts=len(self.births.cohorts),
            birth_updates_by_cohort=[300-batch['birth_step'] for batch in self.births.cohorts])
        self.finalize_frame()
        self._plan_update_rate_budget()
