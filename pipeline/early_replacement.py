"""Shared optimizer helpers for adaptive births and the frame plan."""
import torch
from .model.birth_quantization import model_raw

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
        return self._adaptive_begin()

    @torch.no_grad()
    def _early_observations(self,packages):
        if not self.early_initialized or not len(self.births.parent_rows):return
        rows=torch.arange(len(self.births.parent_rows),device=self.device)+len(self.gaussians._xyz)
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
        if not self.motion_committed:
            samples=[]
            for item in packages:
                frustum=item['visibility_filter'];observed=item['motion_contribution_filter']
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
        return self._adaptive_freeze_motion()

    @torch.no_grad()
    def _early_finish(self):
        return self._adaptive_finish()
