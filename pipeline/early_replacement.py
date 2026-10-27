"""Single-frame ordered support transfer at70; one shared Adam, global<=300."""
from types import SimpleNamespace
from pathlib import Path
import math
import torch
from .model.birth_quantization import RAW_FIELDS,model_raw,LiveReplacementView
from .model.residual_birth import (BirthTransaction,RawView,CommitTrial,encode_trial,
    admit,selected_population,validation_errors)
from .model.support_reincarnation_proposal import score_support_responsibility


from .adaptive_birth import AdaptiveBirthController


class EarlyReplacementController(AdaptiveBirthController):
    def _early_attach_opacity(self):
        if self.opacity_update is None or self.opacity_update.delta is None:return
        optimizer=self.motion.optimizer
        groups={group.get('name'):group for group in optimizer.param_groups}
        delta=self.opacity_update.delta
        if 'early_old_opacity' not in groups:
            optimizer.add_param_group(dict(params=[delta],name='early_old_opacity',
                lr=self.opacity_update.config.learning_rate,betas=(.9,.999),eps=1e-8))
        else:groups['early_old_opacity']['params']=[delta]
        saved=self.opacity_update.optimizer.state.get(delta,{})
        if saved:optimizer.state[delta]=saved

    def _early_sync_opacity_state(self):
        update=self.opacity_update
        if update is None or update.delta is None or self.motion.optimizer is None:return
        state=self.motion.optimizer.state.get(update.delta)
        if state is not None:update.optimizer.state[update.delta]=state
        for group in self.motion.optimizer.param_groups:
            if group.get('name')=='early_old_opacity':update.optimizer.param_groups[0]['lr']=group['lr']

    def _early_prepare_rates(self,step):
        if getattr(self.config.birth,'frame_training_plan',False):
            return self._plan_prepare_rates(step)
        return self._early_prepare_rates_legacy(step)

    def _early_prepare_rates_legacy(self,step):
        self._early_attach_opacity()
        cfg=self.config.gaussian_stage2
        fraction=min(1.,max(0.,(step-71)/229))
        factor=self.config.birth.lr_final_ratio**fraction
        rates=dict(early_xyz=cfg.position_lr_init*self.gaussians.spatial_lr_scale,
            early_rotation=cfg.rotation_lr if self.config.birth.rotation_updates else 0.,early_scaling=cfg.scaling_lr,
            early_opacity=cfg.opacity_lr,early_f_dc=cfg.feature_lr,early_f_rest=cfg.feature_lr/20)
        for group in self.motion.optimizer.param_groups:
            name=group.get('name')
            if name in rates:group['lr']=rates[name]*factor
            if name=='early_old_opacity':
                group['lr']=self.opacity_update.config.learning_rate*(
                    self.motion._motion_lr_factor(min(step,200)) if self.config.opacity_update.decay_with_motion else 1.)

    @torch.no_grad()
    def _early_pin_sources(self,before_step=False):
        if self.config.birth.radiance_mode=='adaptive_residual_birth':return
        rows=self.early_pinned_rows
        if rows is None or not len(rows):return
        update=self.opacity_update
        update.delta[rows]=self.early_pinned_values
        if before_step and update.delta.grad is not None:update.delta.grad[rows]=0
        state=self.motion.optimizer.state.get(update.delta,{})
        for name in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
            if name in state:state[name][rows]=0

    @torch.no_grad()
    def _early_current_old(self,step):
        raw=model_raw(self.gaussians)
        feature_update=getattr(self,'feature_dc_update',None)
        if feature_update is not None:raw['sh_dc']=feature_update().detach()
        if not self.motion_committed:
            xyz,q=self.motion(step,training=False)
            raw['xyz'],raw['rotation']=xyz.detach(),q.detach()
            alpha=self.opacity_update.alphas().detach()
            raw['opacity']=self.opacity_update().detach()
        else:alpha=raw['opacity'].sigmoid()
        return raw,alpha

    @torch.no_grad()
    def _early_begin(self):
        if self.config.birth.radiance_mode=='adaptive_residual_birth':return self._adaptive_begin()
        from .model.early_replacement_model import EarlyReplacementModel
        if self.trainer.global_step!=70 or self.early_initialized:
            raise ValueError('The single early topology is born exactly once after gate70')
        views=self.transaction_camera_views()
        if len(views)!=20 or 'cam00.png' in views:raise ValueError('Early support transfer fits all20 training views only')
        raw,alpha=self._early_current_old(70)
        old=RawView(raw,self.config.sh_degree,self.gaussians,alpha)
        proxy=SimpleNamespace(config=self.config,gaussians=old,renderer=self.renderer,background=self.background)
        scores,observed,_,responsibility=score_support_responsibility(proxy,views)
        seen=sum((observed>>i)&1 for i in range(20))
        eligible=((scores>0)&(seen>=2)).nonzero(as_tuple=True)[0]
        quota=min(self.config.birth.residual.repair_candidates,self.config.birth.residual.max_candidates)
        rows=eligible[scores[eligible].argsort(descending=True,stable=True)[:quota]]
        seed={name:value[rows].clone() for name,value in raw.items()}
        learner=EarlyReplacementModel(self.config.sh_degree).install(seed,rows,alpha[rows])
        learner.spatial_lr_scale=self.gaussians.spatial_lr_scale
        learner.active_sh_degree=self.gaussians.active_sh_degree
        learner.bind_source_pose(raw['xyz'],raw['rotation'],detach=True)
        learner._opacity.requires_grad_(False)
        learner.attach_optimizer(self.motion.optimizer,self.config.gaussian_stage2)
        self.births=learner
        self.early_pinned_rows=self.opacity_update.rows[rows].clone()
        self.early_pinned_values=self.opacity_update.delta[self.early_pinned_rows].clone()
        self._early_pin_sources(before_step=True)
        n=len(rows);names=tuple(views)
        self.birth_transaction=BirthTransaction(seed,rows.clone(),rows.clone(),scores[rows].clone(),
            seed['xyz'].clone(),seed['rotation'].clone(),seed['logscale'].exp(),observed[rows].clone(),names,names,
            0,n,group_ids=torch.arange(n,device=self.device))
        self.early_initialized=True;self.early_birth_updates=0
        self.birth_cohort=0;self.birth_cohort_start=70;self.birth_end_step=300
        self.early_focus_weights={};identity=[]
        renderer=self.renderer;previous=renderer.gaussians;collect=renderer.collect_contributions
        colours=raw['xyz'].new_zeros(len(raw['xyz']),3);colours[rows]=1
        try:
            renderer.collect_contributions=False
            for name,view in views.items():
                renderer.gaussians=old;before=renderer(view,self.background)['render']
                weights=renderer(view,torch.zeros_like(self.background),override_color=colours)['render'][0]
                self.early_focus_weights[name]=weights.detach()
                renderer.gaussians=LiveReplacementView(old,learner)
                after=renderer(view,self.background)['render']
                same=torch.equal(before,after)
                identity.append(dict(camera=name,bitwise_equal=same,max_abs=float((after-before).abs().max())))
                if not same:raise ValueError('Ordered early source handoff changed its native initial image')
        finally:renderer.gaussians=previous;renderer.collect_contributions=collect
        self.birth_control.update(proposal_count=n,spatial_proposals=0,repair_proposals=n,
            source_mutated_before_acceptance=False,selection_camera_scope='all_standard_training_cameras',
            early_replacement=dict(birth_step=70,selected_source_rows=rows.cpu().tolist(),
                source_count=n,native_responsibility=responsibility,
                selected_residual_SSE=float(scores[rows].sum()),
                selected_fraction_total_SSE=float(scores[rows].sum())/max(responsibility['old_RGB_SSE'],1e-30),
                initial_native_identity=identity,ordered_original_row_layout=True,
                learnable_fields=['local_xyz','local_rotation','logscale','alpha_probability','DC','SH1'],
                immutable_material_bound='source70 full-sphere envelope',
                source_fallback_alpha='pinned70; excluded from subsequent gradient/momentum/prox shrink',
                model_count=1,shared_adam=True,extra_SGD_steps=0))
        self.birth_control['early_replacement']['alpha_training_phase']='source70 fixed through200; independent alpha learns201..300'

    @torch.no_grad()
    def _early_observations(self,packages):
        if not self.early_initialized or not len(self.births.parent_rows):return
        adaptive=self.config.birth.radiance_mode=='adaptive_residual_birth'
        if adaptive:
            rows=torch.arange(len(self.births.parent_rows),device=self.device)+len(self.gaussians._xyz)
        elif self.motion_committed:
            rows=torch.arange(len(self.births.parent_rows),device=self.device)+len(self.birth_transaction.fixed_keep)
        else:rows=self.births.parent_rows
        for item in packages:
            camera=item['birth_camera_index'];mask=item['motion_contribution_filter'][rows]
            self.birth_transaction.observations[mask]|=1<<camera
            visible=item['visibility_filter'][rows];radii=item['radii'][rows]
            self.births.max_radii2D[visible]=torch.maximum(self.births.max_radii2D[visible],radii[visible])

    @torch.no_grad()
    def _early_post_training_step(self,packages,step):
        if getattr(self.config.birth,'frame_training_plan',False):
            return self._plan_post_step(packages,step)
        if step!=self.early_shared_steps+1 or not 1<=step<=300:
            raise ValueError('Early replacement allows exactly one shared Adam step per global iteration')
        self._early_pin_sources(before_step=True)
        if not self.motion_committed:
            samples=[]
            for item in packages:
                frustum=item['visibility_filter'];observed=item['motion_contribution_filter']
                if self.config.birth.radiance_mode=='adaptive_residual_birth':
                    frustum=frustum[:len(self.gaussians._xyz)];observed=observed[:len(self.gaussians._xyz)]
                if len(observed)!=len(self.gaussians._xyz):raise ValueError('Ordered early evidence lost original row alignment')
                self.motion.mark_observed(observed);samples.append(torch.stack((frustum.sum(),observed.sum())))
            frustum,observed=torch.stack(samples).sum(0).tolist()
            self.birth_control['motion_frustum_samples']=self.birth_control.get('motion_frustum_samples',0)+frustum
            self.birth_control['motion_observed_samples']=self.birth_control.get('motion_observed_samples',0)+observed
            self._early_observations(packages)
            update=self.opacity_update
            if update.refit_support is not None and update.delta.grad is not None:update.delta.grad.mul_(update.refit_support)
            edits=self.motion.optimizer_step()  # The one step includes old-alpha and early groups.
            self._early_sync_opacity_state()
            update.optimizer_step(self._rate_byte_price(step),proximal=self.motion._motion_fit_window_due(),
                lr_factor=self.motion._motion_lr_factor(step) if update.config.decay_with_motion else 1.,
                perform_step=False,external_state=self.motion.optimizer.state.get(update.delta,{}))
            self._early_pin_sources()
            if edits:self.birth_control['tree_edit_count']=self.birth_control.get('tree_edit_count',0)+len(edits)
        else:
            self._early_observations(packages)
            self.motion.optimizer.step()
            if self.config.birth.joint_old_opacity:
                self._adaptive_post_motion_opacity(step,packages)
            self.motion.optimizer.zero_grad(set_to_none=True)
        if getattr(self, 'feature_dc_update', None) is not None:
            self._step_feature_dc(step)
        self.early_shared_steps+=1
        if self.early_initialized and len(self.births.parent_rows):
            self.early_birth_updates+=1;self.births.project_parameters()
        if step==70:self._early_begin()
        if step==200:self._early_select_population()
        if step==300:self._early_finish()

    @torch.no_grad()
    def _early_select_population(self):
        if self.config.birth.radiance_mode=='adaptive_residual_birth':return self._adaptive_freeze_motion()
        if not self.early_initialized or self.early_birth_updates not in (0,130):
            raise ValueError('Early population selection follows exactly130 shared birth iterations')
        xyz,q=self.motion(200,training=False)
        self.gaussians.apply_selected_motion(xyz,q)
        self.gaussians._opacity.copy_(self.opacity_update.canonical_logits().to(self.device))
        self.opacity_update.commit(torch.arange(len(xyz),device=self.device));self.opacity_update.finish()
        for parameter in self.motion.parameters():parameter.requires_grad_(False)
        self.motion.optimizer.zero_grad(set_to_none=True);self.motion_committed=True
        self.births.bind_source_pose(self.gaussians._xyz,self.gaussians._rotation,detach=True)
        tx=self.birth_transaction;views=list(self.transaction_camera_views().values())
        baseline=encode_trial(self,torch.zeros(len(self.births.parent_rows),device=self.device,dtype=torch.bool),
                              torch.zeros(len(xyz),device=self.device,dtype=torch.bool))
        self.gaussians._xyz.copy_(baseline.raw['xyz']);self.gaussians._rotation.copy_(baseline.raw['rotation'])
        self.births.bind_source_pose(self.gaussians._xyz,self.gaussians._rotation,detach=True)
        tx.old_render_reference=baseline.raw
        trial,report=admit(self,self.config.birth.residual,views,views,fitting_views=views, capture_prefixes=True,atomic_lineage=True)
        selected=selected_population(self,trial)
        self.births.select_rows_inplace(trial.candidates,self.motion.optimizer)
        self.births._opacity.requires_grad_(True)
        self.births.bind_source_pose(self.gaussians._xyz,self.gaussians._rotation,detach=True)
        self.births.bind_codec_alpha(True);self.births.source_rows=trial.sources.clone()
        self.birth_transaction=selected
        normalized=CommitTrial(trial.keep,torch.arange(len(trial.candidates),device=self.device),trial.sources,
            trial.raw,trial.complete_bytes,trial.motion_bytes,trial.birth_bytes,trial.opacity_bytes,trial.old_symbols)
        replay=encode_trial(self,torch.ones(len(self.births.parent_rows),device=self.device,dtype=torch.bool),selected.provisional_deleted)
        if not all(torch.equal(replay.raw[k],trial.raw[k]) for k in RAW_FIELDS):
            raise ValueError('Fixed200 population did not exactly replay its admitted codec scene')
        self.early_selected=dict(model=self.births.capture(),transaction=selected.capture(),trial=normalized.capture(),
            errors=validation_errors(self,views,trial.raw).tolist(),budget=report['complete_byte_budget'])
        self.birth_control['early_replacement'].update(selection200=report,selected_population=len(trial.candidates),
            fixed_population_from=200,SGD_200_codec_replay_exact=True)
        self._birth_admission_witnesses=None
        self.motion.record_final_diagnostics()
        directory=self.output_dir/'region_statistics';directory.mkdir(parents=True,exist_ok=True)
        import json
        (directory/f'frame_{self.current_frame:03d}.json').write_text(json.dumps(self.motion.flat_region_diagnostics,indent=2))
        if not len(trial.candidates):
            self.birth_end_step=200;self._early_finish()

    @torch.no_grad()
    def _early_finish(self):
        if self.config.birth.radiance_mode=='adaptive_residual_birth':return self._adaptive_finish()
        if self.trainer.global_step not in (200,300) or self.early_shared_steps!=self.trainer.global_step:
            raise ValueError('Early commit must remain inside its single global200/300 trajectory')
        views=list(self.transaction_camera_views().values());selected=self.early_selected
        current=encode_trial(self,torch.ones(len(self.births.parent_rows),device=self.device,dtype=torch.bool),
                             self.birth_transaction.provisional_deleted)
        errors=validation_errors(self,views,current.raw)
        accepted=bool(torch.isfinite(errors).all() and current.complete_bytes<=selected['budget'] and
                      errors.mean()<=errors.new_tensor(selected['errors']).mean()+1e-10)
        if not accepted:
            from .model.early_replacement_model import EarlyReplacementModel
            self.births=EarlyReplacementModel(self.config.sh_degree).restore(
                BirthTransaction.tensor_tree(selected['model'],self.device),self.motion.optimizer,self.config.gaussian_stage2)
            self.birth_transaction=BirthTransaction.restore(selected['transaction'],self.device)
            self.births.bind_source_pose(self.gaussians._xyz,self.gaussians._rotation,detach=True)
            current=CommitTrial.restore(selected['trial'],self.device)
            errors=errors.new_tensor(selected['errors'])
        self._commit_population_trial(current)
        self.birth_control['early_replacement'].update(accepted_final100=accepted,
            final_all20_mse=errors.tolist(),selected200_all20_mse=selected['errors'],final_codec_bytes=current.complete_bytes,
            global_iterations=self.early_shared_steps,motion_iterations=200,
            newborn_iterations=self.early_birth_updates,
            joint_overlap_iterations=130 if self.birth_control['early_replacement']['source_count'] else 0,
            material_local_geometry_iterations=self.early_birth_updates,
            new_alpha_iterations=self.trainer.global_step-200,
            final_population_iterations=self.trainer.global_step-200,extra_SGD_steps=0)
        recorder=getattr(self.trainer,'record_birth_comparison',None)
        if recorder is not None:recorder()
        self.finalize_frame()
