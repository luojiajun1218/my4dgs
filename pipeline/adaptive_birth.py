"""Bounded-cost residual births sharing the existing 70/200/300 trajectory."""
from types import SimpleNamespace
import torch
from .model.adaptive_birth_model import AdaptiveBirthModel
from .model.birth_quantization import RAW_FIELDS, RAW_ATTRIBUTES
from .model.residual_birth import BirthTransaction, RawView, spatial_groups
from .model.support_reincarnation_proposal import score_support_responsibility, _native_responsibility
from .model.adaptive_birth_storage import newborn_storage_terms, storage_ramp
from .model.adaptive_birth_colour import conservative_colour_seed
from .model.early_replacement_model import _rotate


class AdaptiveBirthController:
    def _adaptive_view_indices(self):
        # Per-frame reproducible pair; never includes the held-out cam00.
        generator=torch.Generator(device='cpu').manual_seed(1729+int(self.current_frame))
        return torch.randperm(20,generator=generator)[:2].tolist()

    @torch.no_grad()
    def _adaptive_reference(self):
        raw={name:getattr(self.gaussians,attr).detach() for name,attr in zip(RAW_FIELDS,RAW_ATTRIBUTES)}
        xyz=getattr(self.gaussians,'rigid_xyz',None)
        rotation=getattr(self.gaussians,'rigid_rotation',None)
        raw['xyz']=(self.gaussians._xyz if xyz is None else xyz).detach()
        raw['rotation']=(self.gaussians._rotation if rotation is None else rotation).detach()
        raw['opacity']=self.opacity_update().detach()
        feature_update=getattr(self,'feature_dc_update',None)
        if feature_update is not None:raw['sh_dc']=feature_update().detach()
        return raw

    @torch.no_grad()
    def _adaptive_begin(self):
        if self.trainer.global_step!=70 or self.early_initialized:raise ValueError('Birth is once at70')
        views=self.transaction_camera_views()
        if len(views)!=20 or 'cam00.png' in views:raise ValueError('Training cameras only')
        raw,alpha=self._early_current_old(70)
        old=RawView(raw,self.config.sh_degree,self.gaussians,alpha)
        pair_indices=self._adaptive_view_indices()
        pair={name:view for i,(name,view) in enumerate(views.items()) if i in pair_indices}
        proposal=SimpleNamespace(config=self.config,gaussians=old,renderer=self.renderer,background=self.background)
        colour_enabled=self.config.birth.residual_colour_init
        if colour_enabled:
            scores,observed,_,responsibility,cache=_native_responsibility(proposal,pair,
                residual_floor=self.config.birth.residual.residual_floor,collect_colour_evidence=True)
        else:
            scores,observed,_,responsibility=score_support_responsibility(proposal,pair,
                residual_floor=self.config.birth.residual.residual_floor)
        seen=sum((observed>>i)&1 for i in range(2))
        eligible=((scores>0)&(seen>=2)).nonzero(as_tuple=True)[0]
        cap=min(self.config.birth.max_new_gaussians,self.config.birth.residual.max_candidates,
                max(0,self.config.birth.residual.max_gaussians-len(raw['xyz'])))
        regions=[];demands=[];masses=[]
        for group in spatial_groups(raw['xyz'][eligible],8):
            ranked=eligible[group];ranked=ranked[scores[ranked].argsort(descending=True,stable=True)]
            mass=scores[ranked].cumsum(0)
            demand=int(torch.searchsorted(mass,mass[-1]*.9))+1
            regions.append(ranked);demands.append(demand);masses.append(float(mass[-1]))
        quota=sum(demands)
        allocation=[0]*len(regions)
        # Allocate each point to the spatial region with the largest remaining
        # residual mass per already allocated point, subject to local demand.
        for _ in range(min(cap,quota)):
            choices=[i for i in range(len(regions)) if allocation[i]<demands[i]]
            chosen=max(choices,key=lambda i:masses[i]/(allocation[i]+1))
            allocation[chosen]+=1
        selected=[ranked[:count] for ranked,count in zip(regions,allocation)]
        rows=torch.cat(selected) if selected else eligible[:0]
        labels=torch.cat([torch.full((count,),i,device=self.device,dtype=torch.long) for i,count in enumerate(allocation)]) if allocation else eligible[:0]
        seed={k:v[rows].clone() for k,v in raw.items()}
        # Deterministic local offsets avoid coincident-depth ties and RNG drift.
        index=torch.arange(len(rows),device=self.device,dtype=seed['xyz'].dtype)
        direction=torch.stack((torch.sin(index*2.399),torch.cos(index*2.399),torch.sin(index*1.37)),1)
        local_offset=direction*seed['logscale'].exp()*.25
        seed['xyz']+=_rotate(seed['rotation'],local_offset)
        seed['logscale']+=torch.tensor(.7,device=self.device).log()
        initial=alpha[rows].clamp(.03,.1)
        seed['opacity']=torch.logit(initial)
        colour_report=(conservative_colour_seed(raw,seed,rows,initial,cache,
            gain=self.config.birth.residual_colour_gain,
            max_rgb_delta=self.config.birth.residual_colour_max_rgb_delta)
            if colour_enabled else dict(enabled=False))
        learner=AdaptiveBirthModel(self.config.sh_degree).install(seed,rows,initial)
        learner.spatial_lr_scale=self.gaussians.spatial_lr_scale
        learner.active_sh_degree=self.gaussians.active_sh_degree
        learner.bind_source_pose(raw['xyz'],raw['rotation'],detach=True)
        # install defines local zero at seed; bind changes anchor to parent.
        learner.local_xyz.copy_(local_offset)
        if not self.config.birth.rotation_updates:
            learner.local_rotvec.requires_grad_(False)
        learner.attach_optimizer(self.motion.optimizer,self.config.gaussian_stage2)
        self.births=learner;n=len(rows);names=tuple(views)
        self.birth_transaction=BirthTransaction(seed,rows.clone(),torch.full_like(rows,-1),scores[rows].clone(),
            seed['xyz'].clone(),seed['rotation'].clone(),seed['logscale'].exp(),torch.zeros_like(observed[rows]),
            names,names,n,0,group_ids=labels)
        self.early_initialized=True;self.early_birth_updates=0
        self.birth_cohort=0;self.birth_cohort_start=70;self.birth_end_step=300
        self.early_pinned_rows=None;self.early_pinned_values=None;self.early_focus_weights={}
        self.birth_control.update(proposal_count=n,spatial_proposals=n,repair_proposals=0,
            selection_camera_scope='fixed_seeded_pair_of_training_cameras',early_replacement=dict(
                mode='adaptive_residual_birth',source_count=n,birth_step=70,
                residual_mass_quota=quota,eligible_count=len(eligible),population_cap=cap,
                residual_floor=self.config.birth.residual.residual_floor,
                region_demands=demands,region_allocations=allocation,region_residual_mass=masses,
                source_rows=rows.cpu().tolist(),native_responsibility=responsibility,
                learnable_fields=['xyz']+(['rotation'] if self.config.birth.rotation_updates else [])+['scale','DC','SH','alpha'],
                source_preserved=True,parent_colour_envelope=False,quantized_training=True,
                proposal_render_calls=4,proposal_colour_backward_calls=6 if colour_enabled else 2,
                residual_colour_initialization=colour_report,
                storage_loss=dict(enabled=self.config.birth.storage_loss_enabled,
                    strength=self.config.birth.storage_loss_strength,
                    warmup_steps=self.config.birth.storage_warmup_steps,
                    ramp_steps=self.config.birth.storage_ramp_steps,
                    units='byte-scaled surrogate, not actual compressed bytes',
                    normalization='same rate_loss_weight/reference bytes as old motion'),
                selection_cameras=list(pair)))

    def _adaptive_storage_regularization(self,step,birth_view=None):
        if getattr(self.config.birth,'frame_training_plan',False):
            return self._plan_storage_regularization(step,birth_view)
        cfg=self.config.birth
        view=self.make_birth_view() if birth_view is None else birth_view
        if view is None:
            zero=self.births._opacity.sum()*0
            terms=dict(surrogate_bytes=zero,support_points=zero,base_bytes=zero,
                       pose_extra_bytes=zero,appearance_entropy_bytes=zero)
        else:terms=newborn_storage_terms(view,cfg.opacity_threshold)
        ramp=storage_ramp(step,cfg.storage_warmup_steps,cfg.storage_ramp_steps)
        price=cfg.storage_loss_strength*self.config.rigid_tree.rate_loss_weight/self.motion.rate_normalization_bytes
        penalty=terms['surrogate_bytes']*(price*ramp)
        metrics={f'birth_storage_{name}':value.detach() for name,value in terms.items()}
        metrics.update(birth_storage_loss=penalty.detach(),birth_storage_ramp=penalty.new_tensor(ramp))
        return penalty,metrics

    def _adaptive_population_regularization(self,step):
        if getattr(self.config.birth,'frame_training_plan',False):
            return self._plan_population_regularization(step)
        """Equal support pressure per observed point, independent of birth age.

        This is a representation penalty, not transmitted bytes. Unlike the
        opacity delta price it can retire a point whose previous alpha is high.
        New points share the birth warmup; unseen rows receive no pressure.
        """
        update=self.opacity_update
        if update.config.mode=='alpha':
            old=(update.alpha_reference+update.delta).clamp(0,1)[update.rows]
        else:old=update().sigmoid()
        new=self.births._opacity.clamp(0,1)
        threshold=self.config.birth.opacity_threshold
        old_mask=self.motion.observations.gt(0)
        new_mask=self.birth_transaction.observations.ne(0)
        old_support=(old.flatten()/(old.flatten()+threshold)*old_mask).sum()
        new_support=(new.flatten()/(new.flatten()+threshold)*new_mask).sum()
        ramp=storage_ramp(step,self.config.birth.storage_warmup_steps,self.config.birth.storage_ramp_steps)
        penalty=(old_support+new_support)*(self.config.birth.population_loss_weight*ramp/max(1,len(old)+len(new)))
        return penalty,dict(population_loss=penalty.detach(),population_old_support=old_support.detach(),
            population_new_support=new_support.detach(),population_ramp=penalty.new_tensor(ramp))

    @torch.no_grad()
    def _adaptive_post_motion_opacity(self,step,packages):
        """Postprocess the single shared Adam step with old alpha still live."""
        for package in packages:
            self.motion.mark_observed(package['motion_contribution_filter'][:len(self.gaussians._xyz)])
        self._early_sync_opacity_state()
        self.opacity_update.optimizer_step(self._rate_byte_price(step),
            proximal=False,perform_step=False,
            lr_factor=self.motion._motion_lr_factor(200) if self.config.opacity_update.decay_with_motion else 1.,
            external_state=self.motion.optimizer.state.get(self.opacity_update.delta,{}))

    @torch.no_grad()
    def _adaptive_freeze_motion(self):
        """Freeze the old scene at200 without rendering or making codec packets."""
        if not self.early_initialized or self.early_birth_updates not in (0,130):
            raise ValueError('Adaptive freeze requires the joint70..200 trajectory')
        xyz,q=self.motion.decoded_survivor_pose()
        xyz,q=xyz.to(self.device),q.to(self.device)
        self.gaussians.apply_selected_motion(xyz,q)
        self.gaussians._opacity.copy_(self.opacity_update.canonical_logits().to(self.device))
        self.opacity_update.commit(torch.arange(len(xyz),device=self.device))
        if self.config.birth.joint_old_opacity:self.opacity_update.delta.requires_grad_(True)
        else:self.opacity_update.finish()
        for parameter in self.motion.parameters():parameter.requires_grad_(False)
        self.motion.optimizer.zero_grad(set_to_none=True)
        self.motion_committed=True
        self.births.bind_source_pose(self.gaussians._xyz,self.gaussians._rotation,detach=True)
        self.birth_control['early_replacement'].update(
            motion_frozen_at=200,training_population=len(self.births.parent_rows),
            population_screening=False,codec_trials200=0,
            old_opacity_training_end=300 if self.config.birth.joint_old_opacity else 200)

    @torch.no_grad()
    def _adaptive_finish(self):
        """Prune observable low-alpha rows and write the actual frame once."""
        if self.trainer.global_step!=300 or self.early_shared_steps!=300:
            raise ValueError('Adaptive births always complete300 shared iterations')
        tx=self.birth_transaction
        if self.config.birth.joint_old_opacity:
            # Alpha is also the child's attribute predictor: materialize it
            # before quantizing/pruning children, matching the training view.
            self.gaussians._opacity.copy_(self.opacity_update.canonical_logits().to(self.device))
        if self.config.birth.population_loss_weight:
            _,metrics=self._adaptive_population_regularization(300)
            self.birth_control['early_replacement']['population_regularization']={
                name:float(value) for name,value in metrics.items()}
        if self.config.birth.storage_loss_enabled:
            _,metrics=self._adaptive_storage_regularization(300)
            self.birth_control['early_replacement']['storage_loss']['final_before_cleanup']={
                name:float(value) for name,value in metrics.items()}
        # Unseen points are preserved. Parents of surviving children remain as
        # codec predictors even if their own current alpha is negligible.
        visible=tx.observations.ne(0)
        alive=(self.births.get_opacity.flatten()>=self.config.birth.opacity_threshold)|~visible
        selected=alive.nonzero(as_tuple=True)[0]
        if len(selected)!=len(alive):
            self.births.select_rows_inplace(selected,self.motion.optimizer)
        for name in ('sources','replacements','scores','observations','seed_xyz','seed_rotation','seed_scale','group_ids'):
            value=getattr(tx,name,None)
            if isinstance(value,torch.Tensor) and len(value)==len(alive):setattr(tx,name,value[selected])
        tx.raw={name:value[selected] for name,value in tx.raw.items()}
        alpha=self.gaussians._opacity.sigmoid().flatten()
        deleted=(alpha<self.config.opacity_update.prune_threshold)&(self.motion.observations>0)
        deleted[tx.sources]=False
        if bool(deleted.all()):deleted[alpha.argmax()]=False
        keep=(~deleted).nonzero(as_tuple=True)[0]
        mapping=torch.full((len(deleted),),-1,device=self.device,dtype=torch.long)
        mapping[keep]=torch.arange(len(keep),device=self.device)
        sources=mapping[tx.sources]
        if bool((sources<0).any()):raise ValueError('Adaptive codec parent was retired')
        self._commit_opacity_and_retire(forced_remove=deleted)
        self.births.source_rows=sources
        tx.sources=sources
        tx.fixed_reference=None
        tx.old_render_reference=None
        self.motion.record_final_diagnostics()
        self.birth_control['committed_population_count']=len(selected)
        self.birth_control['early_replacement'].update(
            global_iterations=300,motion_iterations=200,newborn_iterations=self.early_birth_updates,
            joint_overlap_iterations=130 if len(alive) else 0,
            final_population_iterations=100,new_alpha_iterations=self.early_birth_updates,
            low_alpha_newborns_removed=int((~alive).sum()),population_screening=False,
            finish_render_calls=0,extra_SGD_steps=0)
        self.finalize_frame()
