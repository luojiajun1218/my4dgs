"""CPU checks for adaptive births and clone checkpoint recovery."""
import copy
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pipeline.model.adaptive_birth_model import AdaptiveBirthModel
from pipeline.model.birth_quantization import QuantizedNewGaussianView, RAW_ATTRIBUTES, RAW_FIELDS, model_raw
from pipeline.model.gaussian import BirthGaussianModel
from pipeline.module import BirthConfig, Module
from tests.fixture_birth_cpu import early_controller_fixture, early_model_fixture, optimizer
from tests.test_frame_plan import checkpoint_harness, cpu_gaussian_setup


class AdaptiveBirthTests(unittest.TestCase):
    def test_inherited_rotation_has_no_pose_symbols_or_storage_charge_and_stays_frozen_after_restore(self):
        from pipeline.model.adaptive_birth_storage import newborn_storage_terms
        old,raw=early_model_fixture()
        model=AdaptiveBirthModel(1).install(old._seed_raw,old.parent_rows)
        model.local_rotvec.requires_grad_(False)
        cfg=SimpleNamespace(new_xyz_step_extent=1e-5,new_rotation_step=.001,
            new_logscale_step=.001,new_opacity_step=.01,new_sh_step=.001,rotation_updates=False)
        view=QuantizedNewGaussianView(model,cfg,2.,raw,model.parent_rows)
        self.assertTrue(torch.equal(view.raw_value('rotation'),raw['rotation'][model.parent_rows]))
        terms=newborn_storage_terms(view)
        support=model._opacity.flatten()/(model._opacity.flatten()+.01)
        source_bytes=torch.ones_like(support)
        for bits in (7,14,21,28):source_bytes+=(view.source>=(1<<bits)).to(support.dtype)
        attribute_bytes=2*sum(view.symbols[name][0].numel()
                              for name in ('logscale','opacity','sh_dc','sh_rest'))
        expected=(support*(source_bytes+3+attribute_bytes)).sum()
        torch.testing.assert_close(terms['base_bytes'],expected,rtol=0,atol=0)
        terms['surrogate_bytes'].backward()
        self.assertIsNone(model.local_rotvec.grad)
        clone=AdaptiveBirthModel(1).restore(model.capture())
        self.assertFalse(clone.local_rotvec.requires_grad)

    def test_bulk_raw_matches_per_field_symbols_and_gradients(self):
        from unittest.mock import patch
        from pipeline.model.birth_quantization import model_raw
        old, raw = early_model_fixture()
        model = AdaptiveBirthModel(1).install(old._seed_raw, old.parent_rows)
        with torch.no_grad():
            model.local_xyz.add_(.013)
            model.local_rotvec.add_(.017)
            model._opacity.fill_(.3)
        cfg = SimpleNamespace(new_xyz_step_extent=1e-5, new_rotation_step=.001,
            new_logscale_step=.001, new_opacity_step=.01, new_sh_step=.001,
            rotation_updates=True)
        results = []
        for legacy in (True, False):
            for _, attr in model.GROUPS:
                getattr(model, attr).grad = None
            # Joint steps70..200 pass differentiable shared-motion poses.
            source_xyz = raw['xyz'].detach().clone().requires_grad_(True)
            source_q = raw['rotation'].detach().clone().requires_grad_(True)
            model.bind_source_pose(source_xyz, source_q, detach=False)
            with patch.object(model, 'birth_parameter_values',
                              None if legacy else model.birth_parameter_values), \
                    patch.object(model, 'raw', wraps=model.raw) as calls:
                view = QuantizedNewGaussianView(model, cfg, 2., raw, model.parent_rows)
                self.assertEqual(calls.call_count, 6 if legacy else 1)
                values = {k: view.raw_value(k).detach().clone() for k in RAW_FIELDS}
                sum(view.raw_value(k).square().sum() for k in RAW_FIELDS).backward()
                grads = {a: getattr(model, a).grad.clone() for _, a in model.GROUPS}
                grads.update(source_xyz=source_xyz.grad.clone(), source_q=source_q.grad.clone())
                snapshot = model_raw(model)
                self.assertEqual(calls.call_count, 12 if legacy else 2)
                results.append((values, grads, snapshot))
        for before, after in zip(*results):
            for name in before:
                torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)

    def test_unbounded_material_and_all_fields_receive_quantized_gradients(self):
        old,raw=early_model_fixture()
        model=AdaptiveBirthModel(1).install(old._seed_raw,old.parent_rows)
        shared,_=optimizer(model)
        with torch.no_grad():model._features_dc.add_(10);model._features_rest.add_(3)
        cfg=SimpleNamespace(new_xyz_step_extent=1e-5,new_rotation_step=.001,new_logscale_step=.001,new_opacity_step=.01,new_sh_step=.001,rotation_updates=True)
        view=QuantizedNewGaussianView(model,cfg,2.,raw,model.parent_rows)
        self.assertGreater(float(view.raw_value('sh_dc').min()),9.)
        loss=sum(view.raw_value(k).sum() for k in RAW_FIELDS)
        loss.backward()
        for _,attr in model.GROUPS:
            grad=getattr(model,attr).grad
            self.assertIsNotNone(grad,attr)
            self.assertTrue(torch.isfinite(grad).all(),attr)
            self.assertGreater(float(grad.abs().sum()),0,attr)
        for name in RAW_FIELDS:
            expected=view.predictor[name]+view.symbols[name]*view.steps[name]
            torch.testing.assert_close(view.raw_value(name),expected,rtol=0,atol=0)
        snapshot=model.capture();clone=AdaptiveBirthModel(1).restore(snapshot)
        self.assertIsInstance(model.subset(torch.tensor([1])),AdaptiveBirthModel)
        for name in RAW_FIELDS:torch.testing.assert_close(clone.raw()[name],model.raw()[name])
        self.assertTrue(clone._opacity.requires_grad)


def controller_fixture(directory,empty=False,raw_transform=None):
    from types import MethodType
    from unittest.mock import patch
    from tests.fixture_birth_cpu import early_controller_fixture
    from pipeline.adaptive_birth import AdaptiveBirthController
    m,views=early_controller_fixture(directory)
    m.config.birth.radiance_mode='adaptive_residual_birth'
    m.config.birth.max_new_gaussians=4
    for name,value in vars(AdaptiveBirthController).items():
        if callable(value):setattr(m,name,MethodType(value,m))
    m.trainer.global_step=70
    if raw_transform is not None:
        current_old=m._early_current_old
        def transformed_old(step):
            raw,alpha=current_old(step)
            return raw_transform(raw),alpha
        m._early_current_old=transformed_old
    n=len(m.gaussians._xyz)
    score=torch.zeros(n) if empty else torch.arange(n,dtype=torch.float32)
    observed=torch.full((n,),3,dtype=torch.long)
    with patch('pipeline.adaptive_birth.score_support_responsibility',return_value=(score,observed,observed,{})) as scoring:
        m._early_begin()
        pair_names={view.path for i,view in enumerate(views.values()) if i in m._adaptive_view_indices()}
        assert set(scoring.call_args.args[1])==pair_names
        assert len(pair_names)==2 and 'cam00.png' not in pair_names
    return m,views


class AdaptiveControllerTests(unittest.TestCase):
    def test_seed_offsets_follow_parent_axes_and_rigid_rotation_after_restore(self):
        import math
        import tempfile
        from unittest.mock import patch
        snapshots=[]
        for transformed in (False,True):
            def source_frame(raw):
                raw={name:value.clone() for name,value in raw.items()}
                # Nonunit quaternions exercise the model's normalized rotation
                # convention. Parent local x/y are world y/-x at seed time.
                q=([.5,.5,-.5,.5] if transformed else [math.sqrt(.5),0,0,math.sqrt(.5)])
                raw['rotation'][:]=raw['rotation'].new_tensor(q)*2
                if transformed:
                    x,y,z=raw['xyz'].unbind(-1)
                    raw['xyz']=torch.stack((x,-z,y),-1)+raw['xyz'].new_tensor([.7,-.3,.2])
                return raw
            # Fix proposal grouping so this checks coordinate conversion, not
            # the axis-aligned candidate-region construction.
            with tempfile.TemporaryDirectory() as directory,patch(
                    'pipeline.adaptive_birth.spatial_groups',
                    side_effect=lambda xyz,_:[torch.arange(len(xyz))] if len(xyz) else []):
                m,_=controller_fixture(directory,raw_transform=source_frame)
                raw,_=m._early_current_old(70)
                rows=m.birth_transaction.sources
                index=torch.arange(len(rows),dtype=raw['xyz'].dtype)
                direction=torch.stack((torch.sin(index*2.399),torch.cos(index*2.399),torch.sin(index*1.37)),1)
                expected_local=direction*raw['logscale'][rows].exp()*.25
                torch.testing.assert_close(m.births.local_xyz,expected_local,rtol=0,atol=0)
                # Independent matrix oracle: Rz(90), then Rx(90) if requested.
                x,y,z=expected_local.unbind(-1)
                expected_world=torch.stack((-y,-z,x) if transformed else (-y,x,z),-1)+raw['xyz'][rows]
                torch.testing.assert_close(m.birth_transaction.seed_xyz,expected_world)
                torch.testing.assert_close(m.births.get_xyz,expected_world)
                restored=AdaptiveBirthModel(1).restore(m.births.capture())
                torch.testing.assert_close(restored.local_xyz,expected_local,rtol=0,atol=0)
                torch.testing.assert_close(restored.get_xyz,expected_world)
                snapshots.append((rows.clone(),m.births.get_xyz.detach().clone()))
        torch.testing.assert_close(snapshots[0][0],snapshots[1][0],rtol=0,atol=0)
        x,y,z=snapshots[0][1].unbind(-1)
        expected=torch.stack((x,-z,y),-1)+x.new_tensor([.7,-.3,.2])
        torch.testing.assert_close(snapshots[1][1],expected)

    def test_predictor_reference_skips_discarded_opacity_graph(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            m, _ = controller_fixture(directory)
            expected = m.opacity_update().detach()
            forward = m.opacity_update.forward
            modes = []
            def capture_mode():
                modes.append(torch.is_grad_enabled())
                return forward()
            with patch.object(m.opacity_update, 'forward', side_effect=capture_mode):
                reference = m._adaptive_reference()
            self.assertEqual(modes, [False])
            self.assertTrue(all(not value.requires_grad for value in reference.values()))
            torch.testing.assert_close(reference['opacity'], expected, rtol=0, atol=0)

    def test200_no_codec_no_screen_and300_no_validation_or_rollback(self):
        import tempfile
        from unittest.mock import patch
        for empty in (False,True):
            with self.subTest(empty=empty),tempfile.TemporaryDirectory() as directory:
                m,_=controller_fixture(directory,empty)
                n=len(m.births.parent_rows)
                m.trainer.global_step=200;m.early_birth_updates=130 if n else 0
                with patch('pipeline.model.residual_birth.encode_trial',side_effect=AssertionError('trial')):
                    m._early_select_population()
                    self.assertTrue(m.motion_committed)
                    self.assertEqual(len(m.births.parent_rows),n)
                    self.assertEqual(m.birth_end_step,300)
                    m.finalize_frame.assert_not_called()
                    self.assertIsNone(m.early_selected)
                    m.trainer.global_step=300;m.early_shared_steps=300
                    m.early_birth_updates=230 if n else 0
                    m._early_finish()
                    m.finalize_frame.assert_called_once()
                    self.assertEqual(m.birth_control['early_replacement']['global_iterations'],300)

    def test_all300_iterations_share_one_adam_including_empty_birth(self):
        import tempfile
        from types import MethodType,SimpleNamespace
        from unittest.mock import patch
        import pipeline.trainer as T
        for empty in (False,True):
            with self.subTest(empty=empty),tempfile.TemporaryDirectory() as directory:
                m,views=controller_fixture(directory,empty)
                # Begin at70 with the fixture's newborn installation complete.
                m.early_shared_steps=70
                shared=m.motion.optimizer
                def training(view,step,regularization_weight=1.):
                    package=m.forward(view,training=True)
                    loss=(package['render']-view.image).square().mean()
                    if len(m.births.parent_rows):
                        loss=loss+sum(m.births.raw()[name].square().mean() for name in RAW_FIELDS)*1e-4
                    return loss,dict(image_loss=loss.detach(),psnr=loss.detach()*0,gs=len(m.gaussians._xyz)),package
                m.training_step=training
                trainer=SimpleNamespace(module=m,config=SimpleNamespace(motion_views_per_step=2))
                trainer._views_per_step=MethodType(T.Trainer._views_per_step,trainer)
                fetcher=SimpleNamespace(next=lambda:None)
                with patch.object(shared,'step',wraps=shared.step) as steps:
                    for step in range(71,301):
                        m.trainer.global_step=step;m._early_prepare_rates(step)
                        T.Trainer._fit_views(trainer,fetcher,step,defer_metrics=True)
                        self.assertIs(m.motion.optimizer,shared)
                        self.assertEqual(m.birth_end_step,300)
                    self.assertEqual(steps.call_count,230)
                self.assertEqual(m.early_shared_steps,300)
                self.assertEqual(m.early_birth_updates,0 if empty else 230)
                m.finalize_frame.assert_called_once()

    def test_nonempty_codec_parent_guard_and_alpha_cleanup(self):
        import tempfile
        from unittest.mock import patch
        from pipeline.model.residual_birth import encode_trial
        from pipeline.model import rigid_tree_codec as codec
        with tempfile.TemporaryDirectory() as directory:
            m,_=controller_fixture(directory)
            m.trainer.global_step=200;m.early_birth_updates=130;m._early_select_population()
            n=len(m.gaussians._xyz);mask=torch.ones(len(m.births.parent_rows),dtype=torch.bool)
            parents=m.birth_transaction.sources.clone()
            deletion=torch.zeros(n,dtype=torch.bool);deletion[parents[0]]=True
            with self.assertRaisesRegex(ValueError,'parent was deleted'):encode_trial(m,mask,deletion)
            deletion.zero_()
            packets=[];original=codec.encode_motion
            def capture(*args,**kwargs):
                packet=original(*args,**kwargs);packets.append(packet);return packet
            with patch.object(codec,'encode_motion',side_effect=capture):trial=encode_trial(m,mask,deletion)
            decoded=codec.decode_motion(packets[-1],m.motion.frame_reference_state)
            for name,value in [('xyz',decoded.xyz),('rotation',decoded.rotation)]:
                torch.testing.assert_close(value,trial.raw[name],rtol=0,atol=0)
            # Observable low-alpha children die; unseen children stay, and the
            # surviving child's parent cannot be retired even at alpha zero.
            with torch.no_grad():
                m.births._opacity.fill_(.001)
                m.birth_transaction.observations.fill_(1)
                m.birth_transaction.observations[-1]=0
                m.gaussians._opacity[parents[-1]]=-100
            m.trainer.global_step=300;m.early_shared_steps=300;m.early_birth_updates=230
            m._early_finish()
            self.assertEqual(len(m.births.parent_rows),1)
            self.assertGreaterEqual(int(m.births.source_rows[0]),0)
            self.assertEqual(m.birth_control['early_replacement']['low_alpha_newborns_removed'],len(mask)-1)
            # The checkpoint transaction retains all aligned arrays after prune.
            from pipeline.model.residual_birth import BirthTransaction
            clone=BirthTransaction.restore(m.birth_transaction.capture(),'cpu')
            self.assertEqual(len(clone.sources),1)
            self.assertEqual(len(clone.seed_xyz),1)

class AdaptiveFinalizationTests(unittest.TestCase):
    def test_real_single_frame_output_and_next_frame_reference_after_restore(self):
        import tempfile
        from types import MethodType
        from unittest.mock import patch
        import pipeline.module as M
        from pipeline.model import rigid_tree_codec as codec
        from pipeline.model.residual_birth import BirthTransaction
        from pipeline.model.birth_quantization import model_raw
        with tempfile.TemporaryDirectory() as directory:
            m,_=controller_fixture(directory)
            # Restore the exact newborn optimizer and transaction state at200.
            m.trainer.global_step=200;m.early_birth_updates=130;m._early_select_population()
            raw_before=model_raw(m.births)
            captured=m.births.capture();tx=m.birth_transaction.capture()
            m.births=AdaptiveBirthModel(1).restore(captured,m.motion.optimizer,m.config.gaussian_stage2)
            m.birth_transaction=BirthTransaction.restore(tx,'cpu')
            for name in RAW_FIELDS:torch.testing.assert_close(model_raw(m.births)[name],raw_before[name],rtol=0,atol=0)
            m.trainer.global_step=300;m.early_shared_steps=300;m.early_birth_updates=230
            m.initial_reference={};m.stage_image_metrics=[]
            m.trainer.decoder_reference_path=m.output_dir/'reference.npz'
            m.trainer.decoder_reference_path.write_bytes(b'fixture')
            for name in ('finalize_frame','_freeze_base'):
                setattr(m,name,MethodType(getattr(M.Module,name),m))
            prior=m.motion.frame_reference_state
            # Base optimizer recreation is CUDA-only in libgs; the codec and
            # materialized state under test remain real CPU implementations.
            with patch.object(m.gaussians,'training_setup',return_value=None),patch.object(codec,'encode_motion',wraps=codec.encode_motion) as encode:
                m._early_finish()
                self.assertEqual(encode.call_count,1)
            path=m.output_dir/'frame_storage'/'rigid_tree'/'motion_001.bin'
            decoded=codec.decode_motion(path.read_bytes(),prior)
            self.assertTrue(m.frame_finalized)
            self.assertGreater(m.frame_report['birth_count'],0)
            torch.testing.assert_close(m.gaussians._xyz,decoded.xyz,rtol=0,atol=0)
            torch.testing.assert_close(m.gaussians._rotation,decoded.rotation,rtol=0,atol=0)
            self.assertEqual(m.motion.codec_previous_state.digest(),decoded.digest())
            from pipeline.model.rigid_tree_birth_codec import decode_birth_attributes,ATTRIBUTES
            old_count=m.frame_report['old_count']
            base={name:value[:old_count] for name,value in model_raw(m.gaussians).items()}
            sources=m.motion.pending_birth_sources
            packet=(m.output_dir/'frame_storage'/'rigid_tree'/'birth_001.bin').read_bytes()
            attrs=decode_birth_attributes(packet,base,decoded.xyz[old_count:],decoded.rotation[old_count:],sources)
            for name in ATTRIBUTES:
                torch.testing.assert_close(model_raw(m.gaussians)[name][old_count:],attrs[name],rtol=0,atol=0)
            xyz=m.gaussians._xyz.detach();q=m.gaussians._rotation.detach()
            m.motion.setup(xyz,q,m.datamodule.cameras_extent,frame_index=2,
                           previous_rows=torch.arange(len(xyz)))
            self.assertEqual(m.motion.frame_reference_state.digest(),decoded.digest())


def clone_fixture(directory):
    module, _ = early_controller_fixture(directory)
    module.config.birth = BirthConfig(mechanism='clone')
    module.config.rigid_tree = module.motion.config
    module.config.opacity_update = module.opacity_update.config
    module.motion.optimizer.param_groups[:] = [group for group in module.motion.optimizer.param_groups
                                               if group.get('name') != 'early_old_opacity']
    module.births = BirthGaussianModel(1)
    rows = torch.tensor([1, 3])
    for attr in RAW_ATTRIBUTES:
        value = getattr(module.gaussians, attr).detach()[rows].clone()
        setattr(module.births, attr, torch.nn.Parameter(value))
    module.births.active_sh_degree = 1
    module.births.spatial_lr_scale = 1.
    module.births.max_radii2D = torch.zeros(len(rows))
    module.births.source_rows = rows.clone()
    cpu_gaussian_setup(module.births, module.config.gaussian_stage2)
    module.motion_committed = True
    module.retired_reference = None
    module.trainer.global_step = 250
    checkpoint_harness(module)
    return module


def update_births(module):
    module.births.optimizer.zero_grad(set_to_none=True)
    loss = sum(getattr(module.births, attr).square().sum() for attr in RAW_ATTRIBUTES)
    loss.backward()
    module.births.optimizer.step()


class CloneResumeTests(unittest.TestCase):
    def test_midframe_clone_checkpoint_restores_lineage_and_adam(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second, \
                patch.object(BirthGaussianModel, 'training_setup', cpu_gaussian_setup):
            module = clone_fixture(first)
            update_births(module)
            packet = dict(current_frame=1, global_step=250)
            Module.on_save_checkpoint(module, packet)
            resumed = clone_fixture(second)
            Module.on_load_checkpoint(resumed, copy.deepcopy(packet))
            self.assertTrue(torch.equal(module.births.source_rows, resumed.births.source_rows))
            for name, value in model_raw(module.births).items():
                torch.testing.assert_close(value, model_raw(resumed.births)[name], rtol=0, atol=0)
            update_births(module)
            update_births(resumed)
            for name, value in model_raw(module.births).items():
                torch.testing.assert_close(value, model_raw(resumed.births)[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
