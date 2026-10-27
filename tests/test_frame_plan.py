"""CPU numerical/lifecycle tests for the explicit 300-step frame plan."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from unittest.mock import Mock, patch
import torch
from pipeline.frame_plan import FramePlanController
from pipeline.adaptive_birth import AdaptiveBirthController
from pipeline.module import Module, BirthConfig, GaussianConfig
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.frame_plan_birth import FramePlanBirthModel, nearest_old_references
from pipeline.model.birth_quantization import RAW_FIELDS, RAW_ATTRIBUTES, QuantizedNewGaussianView, model_raw
from pipeline.model.adaptive_birth_storage import newborn_storage_terms
from pipeline.model.sparse_updates import QuantizedOpacityUpdate, OpacityUpdateConfig
from tests import test_spatial_motion as spatial_tests
from tests.test_codec import reference_raw
from tests.test_reincarnation import early_controller_fixture


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


def controller_fixture(directory):
    m, views = early_controller_fixture(directory)
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
    for name in ('make_birth_view', 'training_step'):
        setattr(m, name, MethodType(getattr(Module, name), m))
    m.prepare_frame = lambda: None
    m.pre_training_step = MethodType(Module.pre_training_step, m)
    m._early_attach_opacity()
    return m, views


class FramePlanMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

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
                        self.assertTrue((m.births.cohorts[0]['parameters']['_features_rest']==0).all())
                        self.assertEqual(len(m.gaussians._xyz),count-1)
                    if step==250:
                        self.assertLessEqual(m._plan_used_quota,4)
                        self.assertTrue((m.births.cohorts[1]['parameters']['_features_rest']==0).all())
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

    def test_real_module_checkpoint_cleanup200_250_and_final_cpu_packets(self):
        from pipeline.trainer import Trainer
        from pipeline.model.gaussian import BirthGaussianModel
        from scripts.replay_rigid_tree import decode_frame
        from pipeline.model.birth_quantization import reference_digest
        with tempfile.TemporaryDirectory() as directory, patch.object(BirthGaussianModel,'training_setup',cpu_gaussian_setup):
            m,_=controller_fixture(directory)
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
                if step in (200,250):
                    packet=dict(global_step=step,current_frame=1)
                    Module.on_save_checkpoint(m,packet)
                    # Real save/load serialization also removes tensor aliasing.
                    import io
                    stream=io.BytesIO();torch.save(packet,stream);stream.seek(0)
                    packet=torch.load(stream,weights_only=False)
                    clone,_=controller_fixture(directory)
                    checkpoint_harness(clone)
                    Module.on_load_checkpoint(clone,packet)
                    self.assertEqual(clone._plan_used_quota,m._plan_used_quota)
                    self.assertEqual([e['step'] for e in clone._plan_events],[e['step'] for e in m._plan_events])
                    self.assertEqual(clone.early_shared_steps,step)
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
            self.assertTrue(m.frame_finalized)
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
            self.assertEqual(m.frame_report['retired_count'],3)
            self.assertEqual(m.frame_report['birth_count'],4)
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


if __name__=='__main__':unittest.main()
