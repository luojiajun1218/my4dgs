"""CPU numerical, lifecycle, diagnostics and recovery tests for the frame plan."""
import copy
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import torch

from pipeline.adaptive_birth import AdaptiveBirthController
from pipeline.frame_plan import (FramePlanController, effective_final_joint_refine_steps,
                                 fitting_old_motion, frame_plan_phase)
from pipeline.model.adaptive_birth_storage import newborn_storage_terms
from pipeline.model.birth_quantization import RAW_FIELDS, RAW_ATTRIBUTES, QuantizedNewGaussianView, model_raw, reference_digest
from pipeline.model.early_replacement_model import _logits
from pipeline.model.frame_plan_birth import FramePlanBirthModel, nearest_old_references
from pipeline.model.gaussian import BirthGaussianModel
from pipeline.model.sparse_logscale import LogScaleUpdateConfig, MotionSupportLogScaleUpdate
from pipeline.model.sparse_updates import QuantizedOpacityUpdate, OpacityUpdateConfig
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.module import Module, BirthConfig, GaussianConfig
from pipeline.trainer import Trainer
from scripts.replay_rigid_tree import decode_frame
from tests import test_spatial_motion as spatial_tests
from tests.fixture_birth_cpu import EvidenceRenderer, early_controller_fixture
from tests.test_codec import reference_raw



def motion_fixture():
    template = spatial_tests.JointSpatialMotionTests().model()
    cfg = copy.deepcopy(template.config)
    cfg.motion_steps = 200
    cfg.birth_steps = 100
    cfg.rotation_updates = False
    cfg.amplitude_refit_steps = 0
    cfg.proximal_gs_regularization = True
    cfg.differentiable_rate_support = True
    gate = cfg.spatial_gate
    gate.joint_image_gradients = gate.shared_image_recode = gate.shared_discovery = True
    gate.region_motion = 'relative'
    gate.image_relaxation = 'stochastic_ste'
    gate.frame_training_plan = True
    gate.freeze_step = 180
    cfg.validate()
    model = SpatialGatedMotion(cfg)
    model.setup(template.base_xyz, template.base_rotation, template.extent, frame_index=1)
    model.observations.fill_(1)
    return model


def controller_fixture(directory,budget=False,shared_colour=False):
    m, views = early_controller_fixture(directory)
    m.full_update_active = False
    m.config.birth.frame_training_plan = True
    m.config.birth.radiance_mode = 'adaptive_residual_birth'
    m.config.birth.joint_old_opacity = True
    m.config.birth.max_new_gaussians = 4
    m.config.birth.residual.max_candidates = 4
    m.config.birth.residual.max_gaussians = 100
    m.config.birth.storage_loss_enabled = True
    m.config.birth.storage_warmup_steps = 0
    m.config.birth.storage_ramp_steps = 1
    m.config.birth.population_loss_weight = .002
    m.config.photometric_loss = m.config.birth.photometric_loss = 'l1'
    m.config.lambda_dssim = .2
    motion = motion_fixture()
    cfg = motion.config
    m.config.rigid_tree = cfg
    m.motion = SpatialGatedMotion(cfg)
    m.motion.setup(m.gaussians._xyz, m.gaussians._rotation, 1., frame_index=1)
    m.config.opacity_update = OpacityUpdateConfig(enabled=True, mode='alpha', step=.01,
        learning_rate=.001, l1_weight=.05, proximal_regularization=True)
    m.opacity_update = QuantizedOpacityUpdate(m.config.opacity_update)
    m.opacity_update.setup(m.gaussians._opacity)
    if shared_colour:
        raise ValueError('Regional shared-colour training is retired')
    m.feature_dc_update = None
    m.birth_transaction = None
    m.births = FramePlanBirthModel(1)
    m.retired_reference = None
    m.datamodule.config = NS(training_view_seed=37)
    for view in views.values():
        view.world_view_transform = torch.eye(4)
        view.fovx = view.fovy = 1.5
    for cls in (AdaptiveBirthController, FramePlanController):
        for name, method in vars(cls).items():
            if callable(method):
                setattr(m, name, MethodType(method, m))
    for name in ('make_birth_view', 'training_step', '_step_feature_dc', '_opacity_rate_price'):
        setattr(m, name, MethodType(getattr(Module, name), m))
    m.prepare_frame = lambda: None
    m.pre_training_step = MethodType(Module.pre_training_step, m)
    m._early_attach_opacity()
    return m, views


class FramePlanMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_logged_phase_matches_region_repair_and_other_tail_modes(self):
        birth = NS(frame_training_plan=True, keep_incumbent_fixed_after_motion=False,
                   keep_incumbent_alpha_after_motion=False)
        gate = NS(final_joint_refine_steps=0)
        module = NS(full_update_active=True, config=NS(birth=birth,
                    full_update=NS(regional_repair=True), rigid_tree=NS(spatial_gate=gate)))
        trainer = NS(module=module)
        expected = [(1,'shared_gate'),(100,'shared_gate'),(101,'fixed_gate_residual'),
                    (150,'fixed_gate_residual'),(151,'joint_gate_residual'),
                    (180,'joint_gate_residual'),(181,'fixed_gate_motion'),
                    (200,'fixed_gate_motion'),(201,'regional_attribute_repair'),
                    (300,'regional_attribute_repair')]
        for step, phase in expected:
            self.assertEqual(frame_plan_phase(module,step),phase)
            self.assertEqual(Trainer._phase_name(trainer,step),phase)
        birth.keep_incumbent_fixed_after_motion=True
        self.assertEqual(Trainer._phase_name(trainer,201),'regional_attribute_repair')
        module.full_update_active=False
        self.assertEqual(Trainer._phase_name(trainer,201),'newborn_fixed_incumbent')
        birth.keep_incumbent_alpha_after_motion=True
        self.assertEqual(Trainer._phase_name(trainer,201),'newborn_old_alpha')
        birth.keep_incumbent_fixed_after_motion=False
        gate.final_joint_refine_steps=20
        self.assertEqual(Trainer._phase_name(trainer,280),'newborn_old_alpha')
        self.assertEqual(Trainer._phase_name(trainer,281),'final_joint_refine')
        self.assertEqual(Trainer._phase_name(trainer,300),'final_joint_refine')

    def test_boundaries_freeze_parameters_and_do_not_accumulate_frozen_adam(self):
        model = motion_fixture()
        expected = [(1,True,False),(100,True,False),(101,False,True),(150,False,True),
                    (151,True,True),(180,True,True),(181,False,True),(200,False,True)]
        for step, gate, residual in expected:
            model.prepare_plan_step(step)
            self.assertEqual(model.gate_logits.requires_grad, gate, step)
            self.assertEqual(model.gs_translation.requires_grad, residual, step)
            before = model.gs_translation.detach().clone()
            p, _ = model(step, training=True)
            (p.square().mean()+model.regularization_loss(step)).backward()
            if not gate: self.assertIsNone(model.gate_logits.grad)
            if not residual: self.assertIsNone(model.gs_translation.grad)
            model.optimizer_step()
            self.assertIsNone(model._amplitude_refit_support)
            if not residual:
                self.assertTrue(torch.equal(before,model.gs_translation))
                self.assertNotIn(model.gs_translation,model.optimizer.state)
        self.assertGreater(int(model.optimizer.state[model.gs_translation]['step']),0)

    def test_final_joint_schedule_and_partial_proximal_window(self):
        from pipeline.model.rigid_residual_tree import saved_config_with_default_motion_lr
        model = motion_fixture()
        cfg = model.config.spatial_gate
        for invalid in (-1, 51, True, 1.5):
            cfg.final_joint_refine_steps = invalid
            with self.assertRaises(ValueError): model.config.validate()
        cfg.final_joint_refine_steps = 20
        model.config.motion_fit_window_steps = 23
        model.config.validate()
        for step, active in ((200,True),(201,False),(280,False),(281,True),(300,True)):
            model.prepare_plan_step(step)
            self.assertEqual(model.region_translation.requires_grad,active)
            self.assertEqual(model.gs_translation.requires_grad,active)
            self.assertFalse(model.gate_logits.requires_grad)
            self.assertFalse(model.region_rotation.requires_grad)
        self.assertTrue(model._motion_fit_window_due())
        saved=copy.deepcopy(model.capture()['config'])
        del saved['spatial_gate']['final_joint_refine_steps']
        self.assertEqual(saved_config_with_default_motion_lr(saved)['spatial_gate']['final_joint_refine_steps'],0)

    def test_compact_reference_can_reuse_physical_candidate_layout_and_restore_without_source(self):
        from pipeline.trainer import Trainer
        from pipeline.model.spatial_gate import spatial_state_route
        with tempfile.TemporaryDirectory() as directory:
            source,_=controller_fixture(directory)
            checkpoint_harness(source)
            template=source.motion.frame_reference_state
            worker=NS(output_dir=Path(directory),module=source,datamodule=source.datamodule)
            Trainer._write_reference(worker,0,template)
            path=worker.decoder_reference_path
            cfg=copy.deepcopy(source.motion.config)
            cfg.spatial_gate.reference_layout=str(path)
            cfg.validate()
            # Changed populations and slightly refined geometry must not
            # move the precomputed split planes or the outlier dictionary.
            xyz=template.xyz[::2].clone()+.003
            rotation=template.rotation[::2].clone()
            model=SpatialGatedMotion(cfg)
            state=model.initialize_reference(xyz,rotation,source.motion.extent)
            self.assertTrue(torch.equal(state.xyz,xyz))
            self.assertTrue(torch.equal(state.rotation,rotation))
            self.assertTrue(torch.equal(state.spatial_centers,template.spatial_centers))
            self.assertTrue(torch.equal(state.parents,template.parents))
            self.assertTrue(torch.equal(state.spatial_paths,spatial_state_route(xyz,template)))
            for name in ('spatial_support_ball','spatial_leaf_centers','spatial_leaf_nodes'):
                a,b=getattr(state,name),getattr(template,name)
                self.assertTrue(a is None and b is None or torch.equal(a,b))
            bad=copy.deepcopy(cfg);bad.spatial_gate.max_depth+=1
            with self.assertRaisesRegex(ValueError,'same spatial depth'):
                SpatialGatedMotion(bad).initialize_reference(xyz,rotation,source.motion.extent)
            with self.assertRaisesRegex(ValueError,'extent'):
                SpatialGatedMotion(cfg).initialize_reference(xyz,rotation,source.motion.extent*2)
            model.setup(xyz,rotation,source.motion.extent,frame_index=1)
            saved=copy.deepcopy(model.capture())
            Path(path).unlink()
            restored=SpatialGatedMotion(copy.deepcopy(cfg));restored.restore(saved)
            for a,b in zip(model(101,training=False),restored(101,training=False)):
                torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_gate_reference_changes_preserve_quantized_motion_and_adam_history(self):
        model = motion_fixture()
        model.prepare_plan_step(151)
        with torch.no_grad():
            model.region_translation[1:,0] = torch.arange(1,len(model.parents))*.001
            model.gs_translation[:,0] = .0235
        p,_=model(151,training=True)
        p.sum().backward();model.optimizer_step()
        before=tuple(t.clone() for t in model._full_motion_symbols())
        state=model.optimizer.state[model.gs_translation]
        step=state['step'].clone();moment=state['exp_avg'].clone()
        with torch.no_grad():model.gate_logits[model.split_eligible] = 5.
        model._select_frontier(freeze=False)
        for a,b in zip(before,model._full_motion_symbols()):self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(step,state['step']))
        self.assertTrue(torch.equal(moment,state['exp_avg']))
        self.assertGreater(model.flat_region_diagnostics['reference_changes'][-1]['changed_gaussians'],0)
        model.last_step=180
        before=tuple(t.clone() for t in model._full_motion_symbols())
        with torch.no_grad():model.gate_logits.fill_(-5)
        model._select_frontier(freeze=True)
        for a,b in zip(before,model._full_motion_symbols()):self.assertTrue(torch.equal(a,b))

    def test_motion_checkpoint_resume_reopens_gate_without_support_lock(self):
        model=motion_fixture()
        for step in (100,101,150):
            p,_=model(step,training=True);p.square().mean().backward();model.optimizer_step()
        clone=SpatialGatedMotion(copy.deepcopy(model.config))
        clone.restore(copy.deepcopy(model.capture()))
        for instance in (model,clone):
            p,_=instance(151,training=True)
            (p.square().mean()+instance.regularization_loss(151)).backward()
            instance.optimizer_step()
        for name in ('region_translation','gs_translation','gate_logits'):
            torch.testing.assert_close(getattr(model,name),getattr(clone,name),rtol=0,atol=0)

    def test_closed_candidates_learn_current_motion_without_changing_rendered_field(self):
        model=motion_fixture()
        # Isolate L1 descent; discrete block bytes can jump when a region opens.
        model.config.rate_loss_weight=0.
        model.config.spatial_gate.refine_unselected_candidates=True
        model.prepare_plan_step(151)
        with torch.no_grad():
            model.region_translation.zero_()
            model.gate_logits.fill_(-5.)
        model._select_frontier(freeze=False)
        with torch.no_grad():
            model.gs_translation[:,0]=.02
        before=model(151,training=False)[0].detach().clone()
        used=model._stored_region_support()
        unused=model.node_counts.gt(0)&~used
        loss=model.loss_terms(151,diagnostics=False)['candidate_motion_loss']
        self.assertGreater(float(loss),0.)
        loss.backward()
        self.assertTrue(model.region_translation.grad[unused].ne(0).any())
        self.assertFalse(model.region_translation.grad[used].ne(0).any())
        self.assertIsNone(model.gs_translation.grad)
        self.assertIsNone(model.gate_logits.grad)
        model.optimizer.param_groups[0]['lr']=float(model.steps[0])*2
        model._mask_regional_fit_updates()
        model.optimizer.step()
        after=model(151,training=False)[0].detach()
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        fitted=model.loss_terms(151,diagnostics=False)['candidate_motion_loss']
        self.assertLess(float(fitted),float(loss))
        for step in (100,150,181):
            model.prepare_plan_step(step)
            self.assertEqual(float(model.loss_terms(step,diagnostics=False)['candidate_motion_loss']),0.)

    def test_unselected_candidates_receive_storage_gradients_without_l1(self):
        model=motion_fixture()
        model.config.spatial_gate.refine_unselected_candidates=True
        model.config.relative_motion_loss_weight=0.
        model.prepare_plan_step(151)
        with torch.no_grad():
            model.region_translation.zero_()
            model.gate_logits.fill_(-5.)
        model._select_frontier(freeze=False)
        with torch.no_grad():
            model.gs_translation[:,0]=model.gs_steps[0]
        used=model._stored_region_support()
        terms=model.loss_terms(151,diagnostics=False)
        self.assertGreater(float(terms['candidate_rate_bytes_surrogate']),0.)
        terms['candidate_motion_loss'].backward()
        unused=model.node_counts.gt(0)&~used
        self.assertTrue(model.region_translation.grad[unused].ne(0).any())
        self.assertFalse(model.region_translation.grad[used].ne(0).any())
        self.assertIsNone(model.gs_translation.grad)
        self.assertIsNone(model.gate_logits.grad)

    def test_fixed_gate_phase_can_prepare_unused_candidates_without_pose_changes(self):
        model=motion_fixture()
        model.config.spatial_gate.refine_unselected_candidates=True
        model.config.spatial_gate.candidate_refinement_start_step=101
        with torch.no_grad():
            model.region_translation.zero_();model.gate_logits.fill_(-5.)
        model._select_frontier(freeze=True)
        model.prepare_plan_step(101)
        with torch.no_grad():model.gs_translation[:,0]=.02
        before=model(101,training=False)[0].detach().clone()
        unused=model.node_counts.gt(0)&~model._stored_region_support()
        loss=model.loss_terms(101,diagnostics=False)['candidate_motion_loss']
        loss.backward();model._mask_regional_fit_updates()
        self.assertTrue(model._hard_gate_frozen)
        self.assertTrue(model.region_translation.grad[unused].ne(0).any())
        self.assertIsNone(model.gate_logits.grad)
        model.optimizer.step()
        torch.testing.assert_close(before,model(101,training=False)[0],rtol=0,atol=0)

    def test_gate_objective_restart_is_once_and_preserves_other_optimizer_states(self):
        model=motion_fixture()
        model.config.spatial_gate.reset_gate_optimizer_on_refine=True
        p,_=model(100,training=True)
        (p.square().mean()+model.regularization_loss(100)).backward();model.optimizer_step()
        self.assertIn(model.gate_logits,model.optimizer.state)
        region_state=model.optimizer.state[model.region_translation]
        before_logits=model.gate_logits.detach().clone()
        model.prepare_plan_step(150);model.prepare_plan_step(151)
        self.assertNotIn(model.gate_logits,model.optimizer.state)
        self.assertIs(model.optimizer.state[model.region_translation],region_state)
        torch.testing.assert_close(before_logits,model.gate_logits,rtol=0,atol=0)
        p,_=model(151,training=True)
        (p.square().mean()+model.regularization_loss(151)).backward();model.optimizer_step()
        new_state=model.optimizer.state[model.gate_logits]
        self.assertEqual(int(new_state['step']),1)
        model.prepare_plan_step(151)
        self.assertIs(model.optimizer.state[model.gate_logits],new_state)
        packet=copy.deepcopy(model.capture())
        restored=SpatialGatedMotion(copy.deepcopy(model.config));restored.restore(packet)
        restored.prepare_plan_step(152)
        self.assertEqual(int(restored.optimizer.state[restored.gate_logits]['step']),1)


class FramePlanBirthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def make_model(self):
        raw=reference_raw();cfg=BirthConfig(rotation_updates=True)
        args=GaussianConfig()
        dummy=torch.nn.Parameter(torch.tensor(0.))
        optimizer=torch.optim.Adam([dummy],lr=.001)
        model=FramePlanBirthModel(1)
        seed={k:v[[1,2]].clone() for k,v in raw.items()}
        seed['xyz']+=.002;seed['rotation'][:,1]=.1
        seed['opacity'][:]=torch.logit(torch.tensor(.4))
        model.add(seed,torch.tensor([1,1]),torch.tensor([1,1]),200,optimizer,args)
        return model,optimizer,raw,cfg,args

    def test_multiple_cohorts_keep_clean_adam_age_and_retain_survivor_state(self):
        model,opt,raw,cfg,args=self.make_model()
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            view=QuantizedNewGaussianView(model,cfg,2.,raw,model.source_rows)
            sum(view.raw_value(k).square().sum() for k in RAW_FIELDS).backward()
            opt.step()
        first=model.cohorts[0]['parameters']['_xyz']
        saved=opt.state[first]['exp_avg'][1:].clone()
        model.select_rows_inplace(torch.tensor([1]),opt)
        self.assertTrue(torch.equal(opt.state[model.cohorts[0]['parameters']['_xyz']]['exp_avg'],saved))
        model.add({k:v[[3]].clone() for k,v in raw.items()},torch.tensor([0]),torch.tensor([1]),250,opt,args)
        model.set_rates(251,args,.1)
        self.assertNotIn(model.cohorts[1]['parameters']['_xyz'],opt.state)
        group=next(g for g in opt.param_groups if g.get('name')=='early_plan_1__xyz')
        self.assertEqual(group['lr'],args.position_lr_init)
        opt.zero_grad(set_to_none=True)
        view=QuantizedNewGaussianView(model,cfg,2.,raw,model.source_rows)
        sum(view.raw_value(k).sum() for k in RAW_FIELDS).backward();opt.step()
        self.assertEqual(int(opt.state[model.cohorts[0]['parameters']['_xyz']]['step']),4)
        self.assertEqual(int(opt.state[model.cohorts[1]['parameters']['_xyz']]['step']),1)
        clone_opt=torch.optim.Adam([torch.nn.Parameter(torch.tensor(0.))],lr=.001)
        clone=FramePlanBirthModel(1).restore(model.capture(),clone_opt,args,'cpu')
        for instance,optim in ((model,opt),(clone,clone_opt)):
            optim.zero_grad(set_to_none=True)
            view=QuantizedNewGaussianView(instance,cfg,2.,raw,instance.source_rows)
            sum(view.raw_value(k).square().sum() for k in RAW_FIELDS).backward();optim.step()
        for name in RAW_FIELDS:torch.testing.assert_close(model_raw(model)[name],model_raw(clone)[name],rtol=0,atol=0)

    def test_all_six_attributes_storage_gradients_and_alpha_endpoint(self):
        model,opt,raw,cfg,args=self.make_model()
        view=QuantizedNewGaussianView(model,cfg,2.,raw,model.source_rows)
        newborn_storage_terms(view)['surrogate_bytes'].backward()
        for attr,param in model.cohorts[0]['parameters'].items():
            self.assertIsNotNone(param.grad,attr)
            self.assertTrue(torch.isfinite(param.grad).all(),attr)
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():model.cohorts[0]['parameters']['_opacity'].fill_(1.)
        view=QuantizedNewGaussianView(model,cfg,2.,raw,model.source_rows)
        self.assertTrue((view.get_opacity>.999).all())
        view.get_opacity.sum().backward()
        self.assertTrue((model.cohorts[0]['parameters']['_opacity'].grad>0).all())
        opt.step();model.project_parameters()
        self.assertTrue((model._opacity<1.).all())

    def test_reanchor_keeps_finite_quantized_logit_when_sigmoid_rounds_to_one(self):
        model,opt,raw,cfg,args=self.make_model()
        actual=model_raw(model)
        actual['opacity'].fill_(16.64)
        self.assertTrue((actual['opacity'].sigmoid()==1).all())
        model.set_actual(actual)
        self.assertTrue(torch.equal(model_raw(model)['opacity'],actual['opacity']))
        model.source_rows=torch.tensor([3,3])
        view=QuantizedNewGaussianView(model,cfg,2.,raw,model.source_rows)
        self.assertLess(float((view.raw_value('opacity')-actual['opacity']).abs().max()),.0051)
        snapshot={name:view.raw_value(name).detach().clone() for name in RAW_FIELDS}
        model.set_actual(snapshot)
        clone_opt=torch.optim.Adam([torch.nn.Parameter(torch.tensor(0.))])
        clone=FramePlanBirthModel(1).restore(model.capture(),clone_opt,args,'cpu')
        self.assertTrue(torch.equal(model_raw(clone)['opacity'],snapshot['opacity']))

    def test_reference_match_prefers_position_colour_and_ties_lowest_id(self):
        raw=reference_raw();seed={k:v[[1,1]].clone() for k,v in raw.items()}
        refs=nearest_old_references(seed,raw,2.)
        self.assertEqual(refs.tolist(),[1,1])
        raw['xyz'][2]=raw['xyz'][1];raw['sh_dc'][2]=raw['sh_dc'][1]
        self.assertEqual(nearest_old_references(seed,raw,2.).tolist(),[1,1])
        with self.assertRaisesRegex(ValueError,'zero surviving old'):
            nearest_old_references(seed,{k:v[:0] for k,v in raw.items()},2.)

    def test_sh3_birth_can_inherit_reference_rest_without_coding_a_cancellation(self):
        raw=reference_raw()
        raw['sh_rest']=torch.arange(len(raw['xyz'])*15*3,dtype=torch.float32).reshape(-1,15,3)/1000
        sources=torch.tensor([1,2])
        seed={name:value[sources].clone() for name,value in raw.items()}
        seed['xyz'][:,0]+=.001
        optimizer=torch.optim.Adam([torch.nn.Parameter(torch.tensor(0.))],lr=.001)
        model=FramePlanBirthModel(3)
        model.add(seed,sources,sources,200,optimizer,GaussianConfig())
        view=QuantizedNewGaussianView(model,BirthConfig(rotation_updates=True),2.,raw,sources)
        self.assertTrue(torch.equal(view.symbols['sh_rest'],torch.zeros_like(view.symbols['sh_rest'])))
        torch.testing.assert_close(view.raw_value('sh_rest'),raw['sh_rest'][sources],rtol=0,atol=0)

    def test_low_alpha_support_gets_time_to_fit_with_logit_learning_rates(self):
        raw=reference_raw();cfg=BirthConfig(rotation_updates=True,survival_aware_storage=True)
        args=GaussianConfig(opacity_lr=.06)
        seed={k:v[[1,2]].clone() for k,v in raw.items()}
        seed['opacity']=torch.logit(torch.full_like(seed['opacity'],.05))
        results={}
        for logit in (False,True):
            optimizer=torch.optim.Adam([torch.nn.Parameter(torch.tensor(0.))])
            model=FramePlanBirthModel(1,logit_opacity=logit)
            model.add(seed,torch.tensor([1,2]),torch.tensor([1,2]),200,optimizer,args)
            model.set_rates(201,args,.1)
            view=QuantizedNewGaussianView(model,cfg,2.,raw,model.source_rows)
            initial=view.get_opacity.detach().clone()
            # A positive finite alpha gradient is normal for a not-yet-fitted
            # seed; the old probability step erases it in one Adam update.
            view.get_opacity.sum().backward();optimizer.step();model.project_parameters()
            results[logit]=model._opacity.detach().clone()
            if logit:
                self.assertTrue((model._opacity>.01).all())
                self.assertTrue((model._opacity<initial).all())
                state=model.capture();restored=FramePlanBirthModel(1).restore(state,
                    torch.optim.Adam([torch.nn.Parameter(torch.tensor(0.))]),args,'cpu')
                self.assertTrue(restored.logit_opacity)
                torch.testing.assert_close(restored._opacity,model._opacity,rtol=0,atol=0)
                fitted=QuantizedNewGaussianView(model,cfg,2.,raw,model.source_rows)
                actual={name:fitted.raw_value(name).detach().clone() for name in RAW_FIELDS}
                model.set_actual(actual)
                again=QuantizedNewGaussianView(model,cfg,2.,raw,model.source_rows)
                for name in RAW_FIELDS:torch.testing.assert_close(again.raw_value(name),actual[name],rtol=0,atol=0)
                optimizer.zero_grad(set_to_none=True)
                loss=newborn_storage_terms(again)['surrogate_bytes'];loss.backward()
                self.assertTrue(model.cohorts[0]['parameters']['_opacity'].grad.isfinite().all())
        self.assertTrue((results[False]==0).all())

    def test_storage_alone_can_retire_unneeded_logit_births_in_both_lifetimes(self):
        raw=reference_raw();cfg=BirthConfig(rotation_updates=True,survival_aware_storage=True)
        seed={key:value[[1,2]].clone() for key,value in raw.items()}
        seed['opacity']=torch.logit(torch.tensor([[.05],[.285]]))
        args=GaussianConfig(opacity_lr=.06)
        for birth_step in (200,250):
            with self.subTest(birth=birth_step):
                optimizer=torch.optim.Adam([torch.nn.Parameter(torch.tensor(0.))])
                model=FramePlanBirthModel(1,logit_opacity=True)
                model.add(seed,torch.tensor([1,2]),torch.tensor([1,2]),birth_step,optimizer,args)
                for step in range(birth_step+1,301):
                    model.set_rates(step,args,.1);optimizer.zero_grad(set_to_none=True)
                    view=QuantizedNewGaussianView(model,cfg,2.,raw,model.source_rows)
                    newborn_storage_terms(view)['surrogate_bytes'].backward()
                    optimizer.step();model.project_parameters()
                    if step==birth_step+1:
                        self.assertTrue((model._opacity>.01).all(),'No one-step disappearance')
                final=QuantizedNewGaussianView(model,cfg,2.,raw,model.source_rows)
                self.assertTrue((final.get_opacity<.01).all(),'No image demand: byte price must permit actual exit')
                self.assertTrue((final.get_opacity>0).all())
                state=optimizer.state[model.cohorts[0]['parameters']['_opacity']]
                self.assertEqual(int(state['step']),300-birth_step)


class FramePlanControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_lifecycle_four_views_two_births_deletion_and_no_parent_protection(self):
        from pipeline.trainer import Trainer
        with tempfile.TemporaryDirectory() as directory:
            m,views=controller_fixture(directory)
            trainer=NS(module=m,config=NS(motion_views_per_step=4,birth_views_per_step=4))
            trainer._views_per_step=lambda step:4
            adam=m.motion.optimizer
            count=len(m.gaussians._xyz)
            cleanup=m._plan_cleanup
            @torch.no_grad()
            def force_cleanup(step):
                # Exercise retirement after Adam, including an unobserved
                # old reference and a first-cohort parent on cleanup250.
                row=0 if step==200 or not len(m.births.source_rows) else int(m.births.source_rows[0])
                original=m.opacity_update.rows[row]
                m.opacity_update.delta[original]=-m.opacity_update.alpha_reference[original]
                if step==250 and len(m.births._xyz):
                    m.births.cohorts[0]['parameters']['_opacity'][0]=0
                m.motion.observations[row]=0
                return cleanup(step)
            m._plan_cleanup=force_cleanup
            with patch.object(adam,'step',wraps=adam.step) as updates:
                for step in range(1,301):
                    m.trainer.global_step=step
                    m.pre_training_step()
                    Trainer._fit_views(trainer,NS(next=Mock(side_effect=AssertionError('No Fetcher in plan'))),step)
                    self.assertEqual(len({v.path for v in m._plan_step_views}),4)
                    if step==100:self.assertNotIn(m.motion.gs_translation,adam.state)
                    if step==200:
                        self.assertLessEqual(m._plan_used_quota,2)
                        batch=m.births.cohorts[0]['parameters']['_features_rest']
                        refs=m.births.source_rows[:len(batch)]
                        torch.testing.assert_close(batch,m._adaptive_reference()['sh_rest'][refs],rtol=0,atol=0)
                        self.assertEqual(len(m.gaussians._xyz),count-1)
                    if step==250:
                        self.assertLessEqual(m._plan_used_quota,4)
                        batch=m.births.cohorts[1]['parameters']['_features_rest']
                        refs=m.births.source_rows[-len(batch):]
                        torch.testing.assert_close(batch,m._adaptive_reference()['sh_rest'][refs],rtol=0,atol=0)
                self.assertEqual(updates.call_count,300)
            self.assertEqual([event['step'] for event in m._plan_events],[200,250,300])
            self.assertEqual(m.early_shared_steps,300)
            self.assertEqual(m._plan_used_quota,4)
            self.assertEqual(len(m.births.cohorts),2)
            self.assertEqual([b['birth_step'] for b in m.births.cohorts],[200,250])
            self.assertEqual(int(adam.state[m.births.cohorts[0]['parameters']['_xyz']]['step']),100)
            self.assertEqual(int(adam.state[m.births.cohorts[1]['parameters']['_xyz']]['step']),50)
            self.assertEqual(len(m.gaussians._xyz),count-3)
            self.assertEqual(m._plan_events[1]['removed_new'],1)
            self.assertTrue((m.births.source_rows<len(m.gaussians._xyz)).all())
            m.finalize_frame.assert_called_once()

    def test_old_opacity_is_fitted_on_every_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            m,_=controller_fixture(directory)
            for frame in (1,2,3,4,99):
                m.current_frame=frame
                m._plan_prepare_rates(151)
                self.assertTrue(m.opacity_update.delta.requires_grad)
            for name in ('tail_commit_validation', 'tail_commit_probe_trained_views',
                         'tail_commit_decouple_old_attributes',
                         'tail_commit_max_motion_only_streak',
                         'tail_commit_max_old_attribute_streak'):
                with self.assertRaises(TypeError):
                    BirthConfig(**{name: True})

    def test_frame_finalization_does_not_depend_on_frame_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            m, _ = controller_fixture(directory)
            m.birth_control['early_replacement'] = {}
            m.early_birth_updates = 0
            m.finalize_frame = Mock()
            for frame in (1, 2, 3, 4, 99):
                m.current_frame = frame
                m._plan_finish()
            self.assertEqual(m.finalize_frame.call_count, 5)

    def test_sampling_is_reproducible_without_touching_global_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            m,_=controller_fixture(directory)
            state=torch.random.get_rng_state()
            a=[v.path for v in m.frame_plan_training_views(200,4)]
            self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
            m.frame_plan_training_views(201,4)
            b=[v.path for v in m.frame_plan_training_views(200,4)]
            self.assertEqual(a,b)
            self.assertNotIn('cam00.png',a)

    def test_old_opacity_byte_pressure_survives_motion_commit_and_final_partial_window(self):
        from pipeline.model.legacy_opacity_codec import decode_opacity_delta
        # Real old-alpha carriers/Adam metric and encoded sparse rows: a weak
        # update must pay the same byte price after step200 and before export.
        for step,window,should_prune in ((219,20,False),(220,20,True),(300,30,True)):
            with self.subTest(step=step),tempfile.TemporaryDirectory() as directory:
                m,_=controller_fixture(directory)
                m.motion_committed=True;m.early_initialized=False;m.early_shared_steps=step-1
                m.config.rigid_tree.motion_fit_window_steps=window
                m.config.opacity_update.l1_weight=0  # Isolate the missing row-support cost.
                m._rate_byte_price=lambda current_step:.01
                m._plan_cleanup=Mock();m._plan_birth=Mock();m._plan_finish=Mock()
                update=m.opacity_update
                with torch.no_grad():
                    update.delta.zero_();update.delta[0]=.01;update.delta[1]=.3
                state=dict(step=torch.tensor(1.),exp_avg=torch.zeros_like(update.delta),
                    exp_avg_sq=torch.full_like(update.delta,1e-5))
                m.motion.optimizer.state[update.delta]=state
                before=update.alphas().detach().clone()
                m._plan_post_step([],step)
                self.assertEqual(int(update.symbols()[0]),0 if should_prune else 1)
                self.assertEqual(int(update.symbols()[1]),30)
                self.assertEqual(int(state['step']),1,'Proximal must not add an Adam update')
                if not should_prune:torch.testing.assert_close(update.alphas(),before)
                path=Path(directory)/'opacity.bin';update.export(path,1)
                decoded=decode_opacity_delta(path.read_bytes(),update.reference[update.rows],frame_index=1)
                self.assertEqual(0 in decoded.rows.tolist(),not should_prune)
                self.assertTrue(torch.equal(decoded.logits,update.canonical_logits()))
                if step==300:
                    m._plan_cleanup.assert_called_once_with(300);m._plan_finish.assert_called_once()

    def test_population_rate_exemption_does_not_disable_motion_price(self):
        with tempfile.TemporaryDirectory() as directory:
            m, _ = controller_fixture(directory)
            m.motion_committed = True
            m.early_initialized = False
            m.early_shared_steps = 219
            m.config.birth.population_rate_exempt = True
            m.config.rigid_tree.motion_fit_window_steps = 20
            m.config.opacity_update.l1_weight = 0
            m._rate_byte_price = lambda step: .01
            self.assertEqual(m._opacity_rate_price(220), 0.)
            self.assertEqual(m._rate_byte_price(220), .01)
            update = m.opacity_update
            with torch.no_grad():
                update.delta.zero_()
                update.delta[0] = .01
            m.motion.optimizer.state[update.delta] = dict(
                step=torch.tensor(1.), exp_avg=torch.zeros_like(update.delta),
                exp_avg_sq=torch.full_like(update.delta, 1e-5))
            m._plan_post_step([], 220)
            self.assertEqual(int(update.symbols()[0]), 1)
            m.config.birth.population_rate_exempt = False
            self.assertEqual(m._opacity_rate_price(220), .01)

def cpu_gaussian_setup(model, args):
    """Host allocator adapter for libgs' otherwise hard-coded CUDA setup."""
    model.xyz_gradient_accum = model._xyz.new_zeros(len(model._xyz),1)
    model.denom = model._xyz.new_zeros(len(model._xyz),1)
    names = dict(xyz='_xyz',f_dc='_features_dc',f_rest='_features_rest',
                 opacity='_opacity',scaling='_scaling',rotation='_rotation')
    model.optimizer = torch.optim.Adam([dict(name=k,params=[getattr(model,v)],lr=.001)
                                       for k,v in names.items()],lr=0.,eps=1e-15)


def checkpoint_harness(module):
    count = len(module.gaussians._xyz)
    module.reference_count = count
    module.initial_reference = dict(original_count=count, retained_count=count, removed_count=0, threshold=0.)
    module.frame_report = None
    module.stage_image_metrics = []
    module.birth_handoff_sources = None
    module._freeze_base = MethodType(Module._freeze_base,module)


class FramePlanPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_checkpoint_restores_when_reference_layout_is_verified_after_move(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(BirthGaussianModel, 'training_setup', cpu_gaussian_setup):
            model, _ = controller_fixture(directory)
            checkpoint_harness(model)
            old = '/previous/reference_frame0.npz'
            model.config.rigid_tree.spatial_gate.reference_layout = old
            checkpoint = dict(global_step=1, current_frame=1)
            Module.on_save_checkpoint(model, checkpoint)

            reference = Path(directory) / 'reference_frame0.npz'
            reference.write_bytes(b'verified reference')
            entry = dict(file=reference.name, original_path=old, original_resolved_path=old,
                         sha256=hashlib.sha256(reference.read_bytes()).hexdigest())
            (reference.parent / 'manifest.json').write_text(
                json.dumps({'files': {'reference_layout': entry}}))
            restored, _ = controller_fixture(directory)
            checkpoint_harness(restored)
            restored.config.rigid_tree.spatial_gate.reference_layout = str(reference)
            Module.on_load_checkpoint(restored, checkpoint)
            self.assertEqual(checkpoint['rigid_tree_config']['spatial_gate']['reference_layout'], old)
            self.assertEqual(restored.config.rigid_tree.spatial_gate.reference_layout, str(reference))

    def test_real_module_checkpoint_cleanup200_250_and_final_cpu_packets(self):
        for tail_steps in (0,20):
            with self.subTest(final_joint_refine_steps=tail_steps):
                self.check_real_module_checkpoint_and_packets(tail_steps)

    def check_real_module_checkpoint_and_packets(self,tail_steps):
        from pipeline.trainer import Trainer
        from pipeline.model.gaussian import BirthGaussianModel
        from scripts.replay_rigid_tree import decode_frame
        from pipeline.model.birth_quantization import reference_digest
        with tempfile.TemporaryDirectory() as directory, patch.object(BirthGaussianModel,'training_setup',cpu_gaussian_setup):
            m,_=controller_fixture(directory,budget=True)
            m.config.rigid_tree.spatial_gate.final_joint_refine_steps=tail_steps
            checkpoint_harness(m)
            initial=model_raw(m.gaussians)
            prior=copy.deepcopy(m.motion.frame_reference_state)
            worker=NS(output_dir=Path(directory),module=m,datamodule=m.datamodule)
            Trainer._write_reference(worker,0,prior)
            m.trainer.decoder_reference_path=worker.decoder_reference_path
            m.finalize_frame=MethodType(Module.finalize_frame,m)
            trainer=NS(module=m,config=NS(motion_views_per_step=4,birth_views_per_step=4),_views_per_step=lambda step:4)
            # Delete genuine alpha-zero original references at both events.
            cleanup=m._plan_cleanup
            @torch.no_grad()
            def retire(step):
                row=0 if step==200 or not len(m.births.source_rows) else int(m.births.source_rows[0])
                index=m.opacity_update.rows[row]
                m.opacity_update.delta[index]=-m.opacity_update.alpha_reference[index]
                return cleanup(step)
            m._plan_cleanup=retire
            clones={}
            for step in range(1,301):
                m.trainer.global_step=step;m.pre_training_step()
                if step==201:
                    with torch.no_grad():m.births.cohorts[0]['parameters']['_rotation'][:,1]=.123
                Trainer._fit_views(trainer,None,step)
                if step==280:
                    fixed_gate=m.motion.gate_logits.detach().clone()
                    motion_updates=int(m.motion.optimizer.state[m.motion.region_translation]['step'])
                if tail_steps and step==281:
                    from pipeline.frame_plan import fitting_old_motion
                    self.assertTrue(m.motion_committed,'Births must remain in render/capture')
                    self.assertTrue(fitting_old_motion(m))
                    image=m.forward(next(iter(m.transaction_camera_views().values())),training=False)
                    self.assertEqual(len(image['visibility_filter']),len(m.gaussians._xyz)+len(m.births._xyz))
                    row=m.birth_control['frame_plan_gradients'][-1]['parameters']
                    for name in ('region_translation','gs_translation'):
                        self.assertTrue(row[name]['trainable'])
                        self.assertGreater(row[name]['gradient_norm'],0.)
                    self.assertFalse(row['gate_logits']['has_gradient'])
                if step in (200,250,280,281):
                    packet=dict(global_step=step,current_frame=1)
                    Module.on_save_checkpoint(m,packet)
                    # Real save/load serialization also removes tensor aliasing.
                    import io
                    stream=io.BytesIO();torch.save(packet,stream);stream.seek(0)
                    packet=torch.load(stream,weights_only=False)
                    clone,_=controller_fixture(directory,budget=True)
                    clone.config.rigid_tree.spatial_gate.final_joint_refine_steps=tail_steps
                    checkpoint_harness(clone)
                    Module.on_load_checkpoint(clone,packet)
                    self.assertEqual(clone._plan_used_quota,m._plan_used_quota)
                    self.assertEqual([e['step'] for e in clone._plan_events],[e['step'] for e in m._plan_events])
                    self.assertEqual(clone.early_shared_steps,step)
                    if tail_steps and step==281:
                        from pipeline.frame_plan import fitting_old_motion
                        self.assertTrue(fitting_old_motion(clone))
                        self.assertTrue(clone.motion_committed)
                        self.assertEqual(len(clone.births._xyz),len(m.births._xyz))
                        self.assertTrue(clone.motion.region_translation.requires_grad)
                    clone.trainer.global_step=step+1;clone.pre_training_step()
                    if step==200:
                        with torch.no_grad():clone.births.cohorts[0]['parameters']['_rotation'][:,1]=.123
                    clone_trainer=NS(module=clone,config=trainer.config,_views_per_step=lambda step:4)
                    Trainer._fit_views(clone_trainer,None,step+1)
                    clones[step+1]=clone
                if step in clones:
                    clone=clones[step]
                    for name in RAW_FIELDS:
                        torch.testing.assert_close(model_raw(m.births)[name],model_raw(clone.births)[name],rtol=0,atol=0)
                    torch.testing.assert_close(m.opacity_update.delta,clone.opacity_update.delta,rtol=0,atol=0)
                    for name in ('region_translation','gs_translation','gate_logits'):
                        torch.testing.assert_close(getattr(m.motion,name),getattr(clone.motion,name),rtol=0,atol=0)
                    self.assertEqual([e['step'] for e in clone._plan_events],[e['step'] for e in m._plan_events])
            self.assertTrue(m.frame_finalized)
            torch.testing.assert_close(fixed_gate,m.motion.gate_logits,rtol=0,atol=0)
            self.assertEqual(int(m.motion.optimizer.state[m.motion.region_translation]['step']),motion_updates+tail_steps)
            self.assertEqual(m.birth_control['early_replacement']['motion_iterations'],200+tail_steps)
            self.assertEqual(m.early_shared_steps,300)
            self.assertEqual(m.early_birth_updates,100)
            self.assertEqual([e['step'] for e in m._plan_events],[200,250,300])
            stream_dir=Path(directory)/'frame_storage'/'rigid_tree'
            state,decoded=decode_frame(stream_dir,1,prior,initial)
            actual=model_raw(m.gaussians)
            for name in RAW_FIELDS:torch.testing.assert_close(decoded[name],actual[name],rtol=0,atol=0)
            self.assertEqual(state.digest(),m.motion.codec_previous_state.digest())
            self.assertEqual(reference_digest(decoded),m.frame_report['materialized_sha256'])
            if torch.cuda.is_available():
                from pipeline.model.gpu_stream import GPUStream
                from pipeline.model.stream_packet import read_stream_step
                player=GPUStream(prior,initial,capacity=100)
                encoded=read_stream_step(stream_dir,1,region_count=len(prior.parents),
                    depth=prior.spatial_paths.shape[1],rest=3)
                player.advance(encoded)
                gpu=player.capture_raw()
                self.assertEqual(reference_digest(gpu),reference_digest(decoded))
                player.verify_render_inputs(decoded)
            import json
            manifest=json.loads((stream_dir/'frame_001.json').read_text())
            self.assertEqual(manifest['birth_cohorts'],2)
            self.assertEqual(manifest['birth_events'],[200,250])
            self.assertIsNone(manifest.get('feature_dc_update'))
            self.assertEqual(m.frame_report['retired_count'],3)
            self.assertEqual(m.frame_report['birth_count'],4)
            checkpoint={"current_frame":1,"global_step":300}
            Module.on_save_checkpoint(m,checkpoint)
            resumed,_=controller_fixture(directory,budget=True)
            resumed.config.rigid_tree.spatial_gate.final_joint_refine_steps=tail_steps
            checkpoint_harness(resumed)
            Module.on_load_checkpoint(resumed,copy.deepcopy(checkpoint))
            self.assertEqual(m.frame_report['complete_dynamic_update_bytes'],sum(
                p.stat().st_size for p in stream_dir.iterdir() if p.suffix in ('.bin','.json')))
            # Previous newborns now become ordinary old rows; new frame has
            # fresh quota/event/optimizer state and zero current point residuals.
            m.current_frame=2;m.frame_prepared=False;m.frame_finalized=False;m.motion_committed=False
            Module.prepare_frame(m)
            self.assertEqual(m._plan_used_quota,0)
            self.assertEqual(m._plan_events,[])
            self.assertFalse(m.motion.gs_translation.requires_grad)
            self.assertTrue((m.motion.gs_translation==0).all())
            self.assertEqual(len(m.motion.optimizer.state),0)
            self.assertEqual(m.motion.effective_rate_loss_weight, m.config.rigid_tree.rate_loss_weight)


class NoisyRenderer(EvidenceRenderer):
    """Exercise restoration even if a renderer consumes RNG and stores evidence."""
    def __init__(self, model):
        super().__init__(model)
        self.last_contributions = None

    def __call__(self, *args, **kwargs):
        random.random()
        np.random.random()
        torch.rand(())
        self.last_contributions = object()
        return super().__call__(*args, **kwargs)


class FramePlanDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_rng_equal(self, a, b):
        self.assertTrue(torch.equal(a['torch'], b['torch']))
        self.assertEqual(a['python'], b['python'])
        self.assertEqual(a['numpy'][0], b['numpy'][0])
        np.testing.assert_array_equal(a['numpy'][1], b['numpy'][1])
        self.assertEqual(a['numpy'][2:], b['numpy'][2:])

    def run_frame(self, directory, diagnostic, legacy_reference=False, fixed=False):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        m, _ = controller_fixture(directory)
        m.config.birth.keep_incumbent_fixed_after_motion = fixed
        m.config.rigid_tree.spatial_gate.final_joint_refine_steps = 20
        checkpoint_harness(m)
        m.renderer = NoisyRenderer(m.gaussians)
        m.finalize_frame = MethodType(Module.finalize_frame, m)
        view = NS(path='cam00.png', image=torch.full((3, 8, 8), .55))
        trainer = NS(module=m, datamodule=m.datamodule, output_dir=Path(directory),
                     current_frame=1, global_step=0, device=torch.device('cpu'),
                     config=NS(stage_image_camera='cam00.png' if diagnostic else None,
                               motion_views_per_step=4, birth_views_per_step=4),
                     _stage_viewpoint=view, _frame_diagnostic_seconds=0.)
        for name in ('_capture_rng', '_record_stage_image', 'record_birth_comparison',
                     'snapshot_frame_plan_reference', 'record_frame_plan_stage',
                     'record_frame_plan_decoded'):
            setattr(trainer, name, MethodType(getattr(Trainer, name), trainer))
        trainer._restore_rng = Trainer._restore_rng
        trainer._views_per_step = lambda step: 4
        m.trainer = trainer
        Trainer._write_reference(trainer, 0, m.motion.frame_reference_state)
        trainer.snapshot_frame_plan_reference()

        # Verify each actual boundary render restores all RNGs and temporary
        # renderer/cache state, including when the renderer itself mutates them.
        record = trainer._record_stage_image
        def check_record(*args, **kwargs):
            before_rng = trainer._capture_rng()
            renderer_model = m.renderer.gaussians
            contribution = m.renderer.last_contributions
            cache = getattr(m.gaussians, '_render_cache', None)
            contents = dict(cache) if cache is not None else None
            attrs = {name: value.detach().clone() for name, value in model_raw(m.gaussians).items()}
            result = record(*args, **kwargs)
            if kwargs.get('old_pose') is not None:
                # At the current pose, the raw-view counterfactual must match
                # the normal old-only rendering of the same appearance.
                step = args[0]
                record(step, view, label='__pose_normal', old_only=True)
                normal = m.stage_image_metrics[-1]['mse']
                pose = (m.gaussians._xyz.detach(), m.gaussians._rotation.detach())
                record(step, view, label='__pose_override', old_only=True, old_pose=pose)
                overridden = m.stage_image_metrics[-1]['mse']
                self.assertAlmostEqual(normal, overridden, delta=1.e-7)
                m.stage_image_metrics[:] = [r for r in m.stage_image_metrics
                    if not r['phase'].startswith('__pose_')]
            self.assert_rng_equal(before_rng, trainer._capture_rng())
            self.assertIs(m.renderer.gaussians, renderer_model)
            self.assertIs(m.renderer.last_contributions, contribution)
            self.assertIs(getattr(m.gaussians, '_render_cache', None), cache)
            if cache is not None:
                self.assertEqual(cache.keys(), contents.keys())
                for key in cache:
                    self.assertIs(cache[key], contents[key])
            for name in RAW_FIELDS:
                torch.testing.assert_close(model_raw(m.gaussians)[name], attrs[name], rtol=0, atol=0)
            return result
        trainer._record_stage_image = check_record

        for step in range(1, 301):
            trainer.global_step = step
            m.pre_training_step()
            if step in (200, 250, 300) and (not fixed or step == 200):
                # Force a low-alpha old row before the normal optimizer update.
                row = m.opacity_update.rows[0]
                with torch.no_grad():
                    m.opacity_update.delta[row] = -m.opacity_update.alpha_reference[row]
            Trainer._fit_views(trainer, None, step, defer_metrics=True)
            if diagnostic and step == 250:
                packet = dict(global_step=step, current_frame=1)
                Module.on_save_checkpoint(m, packet)
                clone, _ = controller_fixture(directory)
                clone.config.birth.keep_incumbent_fixed_after_motion = fixed
                clone.config.rigid_tree.spatial_gate.final_joint_refine_steps = 20
                checkpoint_harness(clone)
                Module.on_load_checkpoint(clone, packet)
                for name in RAW_FIELDS:
                    torch.testing.assert_close(clone._plan_diagnostic_reference[name],
                        m._plan_diagnostic_reference[name], rtol=0, atol=0)
                if legacy_reference:
                    # A checkpoint produced before diagnostic snapshots must
                    # still be able to replay its prior stream at finalization.
                    m._plan_diagnostic_reference = None

        files = {item.name: item.read_bytes() for item in
                 (Path(directory) / 'frame_storage' / 'rigid_tree').glob('*.bin')}
        return (m.stage_image_metrics, reference_digest(model_raw(m.gaussians)),
                m.motion.codec_previous_state.digest(), files, trainer._capture_rng(),
                m.birth_control['frame_plan_events'])

    def test_boundaries_decoding_and_training_are_unchanged(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second, \
                tempfile.TemporaryDirectory() as legacy, \
                patch.object(BirthGaussianModel, 'training_setup', cpu_gaussian_setup):
            observed = self.run_frame(first, True)
            baseline = self.run_frame(second, False)
            resumed_legacy = self.run_frame(legacy, True, legacy_reference=True)
        records, digest, state, files, rng, events = observed
        self.assertEqual((digest, state, files), baseline[1:4])
        self.assert_rng_equal(rng, baseline[4])
        self.assertEqual((digest, state, files), resumed_legacy[1:4])
        self.assert_rng_equal(rng, resumed_legacy[4])
        self.assertTrue(resumed_legacy[0][-1]['decoded_raw_matches_manifest'])
        expected = [(step, f'plan_{phase}') for step in (200, 250)
                    for phase in ('pre_cleanup', 'post_cleanup', 'pre_birth', 'post_birth')]
        expected += [(280, phase) for phase in ('plan_pre_tail', 'plan_pre_tail_old_only')]
        expected += [(300, phase) for phase in
                     ('plan_pre_cleanup', 'plan_pre_cleanup_old_only',
                      'plan_pre_cleanup_old_pre_tail_pose', 'plan_post_cleanup',
                      'birth_off', 'birth_on', 'plan_decoded')]
        self.assertEqual([(row['step'], row['phase']) for row in records], expected)
        self.assertTrue(all(not row['participates_in_training_or_selection'] for row in records))
        self.assertTrue(all(row['camera'] == 'cam00.png' for row in records))
        for step in (200, 250, 300):
            before = next(r for r in records if r['step'] == step and r['phase'] == 'plan_pre_cleanup')
            after = next(r for r in records if r['step'] == step and r['phase'] == 'plan_post_cleanup')
            event = next(e for e in events if e['step'] == step)
            self.assertGreater(event['removed_old'], 0)
            self.assertEqual(before['population_old_count'] - after['population_old_count'], event['removed_old'])
            self.assertEqual(before['population_birth_count'] - after['population_birth_count'], event['removed_new'])
        off, on, decoded = records[-3:]
        self.assertTrue(off['old_only'])
        self.assertFalse(on['old_only'])
        self.assertEqual(off['population_birth_count'], on['population_birth_count'])
        self.assertGreater(on['population_birth_count'], 0)
        self.assertEqual(decoded['representation'], 'decoded_stream')
        self.assertTrue(decoded['decoded_state_matches_manifest'])
        self.assertTrue(decoded['decoded_raw_matches_manifest'])
        self.assertAlmostEqual(on['mse'], decoded['mse'], delta=1.e-6)
        prior = next(r for r in records if r['phase'] == 'plan_pre_tail_old_only')
        frozen_pose = next(r for r in records if r['phase'] == 'plan_pre_cleanup_old_pre_tail_pose')
        self.assertEqual(prior['population_old_count'], frozen_pose['population_old_count'])
        self.assertEqual(frozen_pose['pose'], 'pre_tail')
        self.assertTrue(frozen_pose['old_only'])

    def test_fixed_mode_has_no_tail_and_preserves_old_only_image(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(BirthGaussianModel, 'training_setup', cpu_gaussian_setup):
            records, _, _, _, _, events = self.run_frame(directory, True, fixed=True)
        self.assertFalse(any('tail' in row['phase'] for row in records))
        before_birth = next(r for r in records
                            if r['step'] == 200 and r['phase'] == 'plan_post_cleanup')
        old_at_end = next(r for r in records if r['phase'] == 'plan_pre_cleanup_old_only')
        self.assertEqual(before_birth['population_old_count'], old_at_end['population_old_count'])
        self.assertEqual(before_birth['mse'], old_at_end['mse'])
        self.assertEqual([event['removed_old'] for event in events[1:]], [0, 0])


class FixedIncumbentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_state_equal(self, before, after):
        self.assertEqual(before.keys(), after.keys())
        for key, value in before.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, after[key], rtol=0, atol=0)
            else:
                self.assertEqual(value, after[key])

    def setup_controller(self, directory, fixed):
        torch.manual_seed(94)
        m, _ = controller_fixture(directory)
        m.config.birth.keep_incumbent_fixed_after_motion = fixed
        m.config.rigid_tree.spatial_gate.final_joint_refine_steps = 30
        m.config.logscale_update = LogScaleUpdateConfig(enabled=True)
        m.logscale_update = MotionSupportLogScaleUpdate(m.config.logscale_update)
        checkpoint_harness(m)
        m.finalize_frame = MethodType(Module.finalize_frame, m)
        trainer = NS(module=m, datamodule=m.datamodule, output_dir=Path(directory),
                     global_step=0, config=NS(stage_image_camera=None,
                         motion_views_per_step=4, birth_views_per_step=4),
                     _views_per_step=lambda step: 4)
        m.trainer = trainer
        prior = copy.deepcopy(m.motion.frame_reference_state)
        initial = {k: v.detach().cpu().clone() for k, v in model_raw(m.gaussians).items()}
        Trainer._write_reference(trainer, 0, prior)
        return m, trainer, prior, initial

    def step(self, m, trainer, step):
        trainer.global_step = step
        m.pre_training_step()
        Trainer._fit_views(trainer, None, step, defer_metrics=True)

    def test_configuration_validation(self):
        with self.assertRaisesRegex(ValueError, 'requires the frame training plan'):
            BirthConfig(keep_incumbent_fixed_after_motion=True).validate(200)
        with self.assertRaisesRegex(ValueError, 'must be boolean'):
            BirthConfig(frame_training_plan=True, keep_incumbent_fixed_after_motion=1).validate(200)
        BirthConfig(frame_training_plan=True, keep_incumbent_fixed_after_motion=True).validate(200)
        self.assertFalse(BirthConfig().keep_incumbent_fixed_after_motion)

    def test_setup_uses_effective_tail_for_optional_logscale(self):
        class PastConfigurationChecks(Exception):
            pass
        with tempfile.TemporaryDirectory() as directory:
            for fixed in (False, True):
                m, _, _, _ = self.setup_controller(directory, fixed)
                m.datamodule.config.white_background = False
                m.config.effective_motion_observations = True
                def after_checks(*args, **kwargs):
                    raise PastConfigurationChecks()
                m.register_buffer = after_checks
                if fixed:
                    with self.assertRaises(PastConfigurationChecks):
                        Module.setup(m)
                else:
                    with self.assertRaisesRegex(ValueError, 'Log-scale updates require'):
                        Module.setup(m)

    def test_freezes_200_state_and_adam_but_trains_births_and_decodes(self):
        with tempfile.TemporaryDirectory() as baseline_dir, tempfile.TemporaryDirectory() as fixed_dir, \
                tempfile.TemporaryDirectory() as resumed_dir, \
                patch.object(BirthGaussianModel, 'training_setup', cpu_gaussian_setup):
            baseline, baseline_trainer, _, _ = self.setup_controller(baseline_dir, False)
            fixed, trainer, prior, initial = self.setup_controller(fixed_dir, True)
            for step in range(1, 201):
                self.step(baseline, baseline_trainer, step)
                self.step(fixed, trainer, step)
            # A configured tail changes nothing in the first 200 updates.
            for name in RAW_FIELDS:
                torch.testing.assert_close(model_raw(fixed.gaussians)[name],
                    model_raw(baseline.gaussians)[name], rtol=0, atol=0)
                torch.testing.assert_close(model_raw(fixed.births)[name],
                    model_raw(baseline.births)[name], rtol=0, atol=0)
            self.assertTrue(baseline.logscale_update.ready)
            self.assertFalse(fixed.logscale_update.ready)
            self.assertEqual(effective_final_joint_refine_steps(fixed), 0)
            legacy_packet = dict(global_step=200, current_frame=1)
            Module.on_save_checkpoint(baseline, legacy_packet)
            del legacy_packet['rigid_tree_module_config']['birth']['keep_incumbent_fixed_after_motion']
            legacy, _, _, _ = self.setup_controller(resumed_dir, False)
            Module.on_load_checkpoint(legacy, copy.deepcopy(legacy_packet))
            self.assertFalse(legacy.config.birth.keep_incumbent_fixed_after_motion)

            old = {key: value.detach().clone() for key, value in model_raw(fixed.gaussians).items()}
            old_count = len(old['xyz'])
            parameters = {name: getattr(fixed.motion, name) for name in
                ('gate_logits', 'region_translation', 'gs_translation', 'region_rotation', 'gs_rotation')}
            parameters['old_alpha'] = fixed.opacity_update.delta
            values = {name: value.detach().clone() for name, value in parameters.items()}
            states = {name: copy.deepcopy(fixed.motion.optimizer.state.get(value, {}))
                      for name, value in parameters.items()}
            alpha = fixed.opacity_update.canonical_logits().clone()
            resume = None

            for step in range(201, 301):
                self.step(fixed, trainer, step)
                self.assertFalse(fitting_old_motion(fixed))
                self.assertFalse(fixed.logscale_update.ready)
                self.assertFalse(any(group.get('name') == 'early_old_logscale'
                                     for group in fixed.motion.optimizer.param_groups))
                if step in (201, 250, 270, 271, 299, 300):
                    for name, value in parameters.items():
                        self.assertFalse(value.requires_grad, name)
                        self.assertIsNone(value.grad, name)
                        torch.testing.assert_close(value, values[name], rtol=0, atol=0)
                        self.assert_state_equal(states[name], fixed.motion.optimizer.state.get(value, {}))
                    torch.testing.assert_close(fixed.opacity_update.canonical_logits(), alpha, rtol=0, atol=0)
                    current = model_raw(fixed.gaussians)
                    for name in RAW_FIELDS:
                        torch.testing.assert_close(current[name][:old_count], old[name], rtol=0, atol=0)
                if step == 250:
                    packet = dict(global_step=step, current_frame=1)
                    Module.on_save_checkpoint(fixed, packet)
                    resume, resume_trainer, _, _ = self.setup_controller(fixed_dir, True)
                    Module.on_load_checkpoint(resume, copy.deepcopy(packet))
                    self.step(resume, resume_trainer, 251)
                if step == 251:
                    for name in RAW_FIELDS:
                        torch.testing.assert_close(model_raw(fixed.births)[name],
                            model_raw(resume.births)[name], rtol=0, atol=0)
            events = fixed.birth_control['frame_plan_events']
            self.assertEqual([event['removed_old'] for event in events[1:]], [0, 0])
            gradient_rows = {row['step']: row['parameters']
                             for row in fixed.birth_control['frame_plan_gradients']}
            for step in (201, 250, 251, 300):
                self.assertFalse(gradient_rows[step]['old_alpha']['has_gradient'])
            self.assertGreater(gradient_rows[201]['birth0__opacity']['gradient_norm'], 0.)
            self.assertEqual(gradient_rows[300]['birth0__opacity']['adam_updates'], 99)
            self.assertEqual(gradient_rows[300]['birth1__opacity']['adam_updates'], 49)
            self.assertGreater(fixed.frame_report['birth_count'], 0)
            directory = Path(fixed_dir) / 'frame_storage' / 'rigid_tree'
            decoded, raw = decode_frame(directory, 1, prior, initial)
            self.assertEqual(decoded.digest(), fixed.motion.codec_previous_state.digest())
            self.assertEqual(reference_digest(raw), fixed.frame_report['materialized_sha256'])
            manifest = json.loads((directory / 'frame_001.json').read_text())
            accounting = manifest['training_step_accounting']
            self.assertEqual(accounting['motion_optimizer_steps'], 200)
            self.assertEqual(accounting['old_opacity'], 100)
            self.assertEqual(accounting['configured_final_joint_refine_steps'], 30)
            self.assertTrue(accounting['incumbent_fixed_after_motion'])
            self.assertEqual(fixed.birth_control['early_replacement']['final_joint_refine_steps'], 0)

            # A real frame-boundary checkpoint must restart the next frame
            # from the same decoded motion and retain the fixed policy.
            checkpoint = dict(global_step=300, current_frame=1)
            Module.on_save_checkpoint(fixed, checkpoint)
            self.assertTrue(checkpoint['rigid_tree_module_config']['birth']
                            ['keep_incumbent_fixed_after_motion'])
            resumed, resumed_trainer, _, _ = self.setup_controller(resumed_dir, True)
            Module.on_load_checkpoint(resumed, copy.deepcopy(checkpoint))
            previous = {key: value.detach().cpu().clone()
                        for key, value in model_raw(fixed.gaussians).items()}
            previous_state = copy.deepcopy(fixed.motion.codec_previous_state)
            for instance in (fixed, resumed):
                Module.begin_next_frame(instance)
                instance.current_frame = 2
                Module.prepare_frame(instance)
                torch.testing.assert_close(instance.motion.base_xyz.cpu(), previous_state.xyz,
                                           rtol=0, atol=0)
                torch.testing.assert_close(instance.motion.base_rotation.cpu(), previous_state.rotation,
                                           rtol=0, atol=0)
            for step in range(1, 301):
                self.step(fixed, trainer, step)
                self.step(resumed, resumed_trainer, step)
                if step in (200, 250, 299, 300):
                    for name in RAW_FIELDS:
                        torch.testing.assert_close(model_raw(fixed.gaussians)[name],
                            model_raw(resumed.gaussians)[name], rtol=0, atol=0)
                    for name in ('region_translation', 'gs_translation', 'gate_logits'):
                        torch.testing.assert_close(getattr(fixed.motion, name),
                            getattr(resumed.motion, name), rtol=0, atol=0)
                    torch.testing.assert_close(fixed.opacity_update.delta,
                                               resumed.opacity_update.delta, rtol=0, atol=0)
            second_dir = Path(resumed_dir) / 'frame_storage' / 'rigid_tree'
            fixed_packets = {p.name: p.read_bytes() for p in directory.glob('*_002.bin')}
            resumed_packets = {p.name: p.read_bytes() for p in second_dir.glob('*_002.bin')}
            self.assertEqual(fixed_packets, resumed_packets)
            decoded2, raw2 = decode_frame(second_dir, 2, previous_state, previous)
            self.assertEqual(decoded2.digest(), resumed.motion.codec_previous_state.digest())
            self.assertEqual(reference_digest(raw2), resumed.frame_report['materialized_sha256'])


class SurvivalStorageTests(unittest.TestCase):
    def make(self,alpha):
        reference=reference_raw();reference['opacity'].fill_(torch.logit(torch.tensor(.8)))
        raw={k:v[:1].clone() for k,v in reference.items()}
        raw['opacity'].fill_(_logits(torch.tensor(alpha)))
        model=FramePlanBirthModel(1)
        optimizer=torch.optim.Adam([torch.nn.Parameter(torch.zeros(()))])
        model.add(raw,torch.tensor([0]),torch.tensor([0]),200,optimizer,GaussianConfig())
        cfg=BirthConfig(frame_training_plan=True,storage_loss_enabled=True,
            radiance_mode='adaptive_residual_birth',rotation_updates=True,new_opacity_step=.01)
        return model,reference,cfg

    def test_alive_forward_cost_is_unchanged_but_opacity_gradient_can_retire(self):
        model,reference,cfg=self.make(.2)
        old=QuantizedNewGaussianView(model,cfg,1.,reference,model.source_rows)
        before=newborn_storage_terms(old)['surrogate_bytes']
        before.backward()
        self.assertLess(float(model.cohorts[0]['parameters']['_opacity'].grad),0.)
        model.optimizer.zero_grad(set_to_none=True)
        cfg.survival_aware_storage=True
        view=QuantizedNewGaussianView(model,cfg,1.,reference,model.source_rows)
        after=newborn_storage_terms(view)['surrogate_bytes']
        torch.testing.assert_close(after,before,rtol=0,atol=0)
        after.backward()
        self.assertGreater(float(model.cohorts[0]['parameters']['_opacity'].grad),0.)

    def test_zero_attribute_delta_still_pays_but_deleted_point_does_not(self):
        model,reference,cfg=self.make(.8);cfg.survival_aware_storage=True
        alive=QuantizedNewGaussianView(model,cfg,1.,reference,model.source_rows)
        self.assertTrue(all(not value.detach().any() for value in alive.symbols.values()))
        self.assertGreater(float(newborn_storage_terms(alive)['surrogate_bytes']),0.)
        model,reference,cfg=self.make(.005);cfg.survival_aware_storage=True
        gone=QuantizedNewGaussianView(model,cfg,1.,reference,model.source_rows)
        self.assertLess(float(gone.get_opacity),.01)
        terms=newborn_storage_terms(gone)
        self.assertEqual(float(terms['surrogate_bytes']),0.)
        self.assertEqual(float(terms['support_points']),0.)

    def test_opaque_newborn_still_has_a_finite_exit_gradient(self):
        model,reference,cfg=self.make(1.);cfg.survival_aware_storage=True
        view=QuantizedNewGaussianView(model,cfg,1.,reference,model.source_rows)
        newborn_storage_terms(view)['surrogate_bytes'].backward()
        gradient=model.cohorts[0]['parameters']['_opacity'].grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient),0.)

    def test_nonfinite_seed_is_rejected_before_training(self):
        model,reference,cfg=self.make(.2)
        raw={key:value[:1].clone() for key,value in reference.items()}
        raw['opacity'].fill_(math.inf)
        with self.assertRaisesRegex(ValueError,'finite raw'):
            model.add(raw,torch.tensor([0]),torch.tensor([0]),250,model.optimizer,GaussianConfig())


if __name__ == "__main__":
    unittest.main()
