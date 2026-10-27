"""Numerical checks of byte feedback and physical newborn survival pricing."""
import copy
import math
import unittest
import torch
from pipeline.model.rate_budget import RateBudgetState
from pipeline.model.frame_plan_birth import FramePlanBirthModel
from pipeline.model.birth_quantization import QuantizedNewGaussianView
from pipeline.model.adaptive_birth_storage import newborn_storage_terms
from pipeline.model.early_replacement_model import _logits
from pipeline.module import BirthConfig,GaussianConfig
from tests.test_codec import reference_raw
from tests.test_frame_plan import motion_fixture


class RateBudgetTests(unittest.TestCase):
    def test_previous_bytes_change_next_weight_and_update_is_idempotent(self):
        state=RateBudgetState()
        report=state.update(1,75000,50000,.25,64.)
        self.assertEqual(report['used_multiplier'],1.)
        self.assertGreater(report['next_multiplier'],1.)
        saved=state.capture()
        self.assertEqual(report,state.update(1,75000,50000,.25,64.))
        self.assertEqual(saved,state.capture())
        restored=RateBudgetState.restore(saved,64.)
        second=restored.update(2,25000,50000,.25,64.)
        self.assertEqual(second['used_multiplier'],report['next_multiplier'])
        self.assertLess(second['next_multiplier'],second['used_multiplier'])
        self.assertEqual(restored.total_bytes,100000)
        self.assertEqual(second['mean_complete_bytes'],50000)
        self.assertEqual(state.frames,1)
        with self.assertRaises(ValueError):state.update(0,75000,50000,.25,64.)
        with self.assertRaises(ValueError):state.update(3,75000,50000,.25,64.)
        with self.assertRaises(ValueError):state.update(1,75001,50000,.25,64.)

    def test_multiplier_bounds_and_invalid_history(self):
        state=RateBudgetState()
        state.update(1,10**9,50000,10.,4.)
        self.assertAlmostEqual(state.multiplier,4.)
        state.update(2,1,50000,10.,4.)
        self.assertEqual(state.multiplier,1.)
        capped=RateBudgetState()
        capped.update(1,10**9,50000,10.,100.)
        RateBudgetState.restore(capped.capture(),100.)
        capped.update(2,10**9,50000,10.,100.)
        RateBudgetState.restore(capped.capture(),100.)
        for field,value in (('frames',-1),('last_frame',9),('log_multiplier',math.inf)):
            bad=state.capture();bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):RateBudgetState.restore(bad,4.)

    def test_motion_multiplier_is_checkpointed_and_prices_loss_without_config_mutation(self):
        model=motion_fixture();model.prepare_plan_step(151)
        with torch.no_grad():model.gs_translation[:,0]=.01
        base=model.loss_terms(151,diagnostics=False)['rate_loss'].detach()
        model.rate_multiplier=3.
        changed=model.loss_terms(151,diagnostics=False)['rate_loss'].detach()
        torch.testing.assert_close(changed,base*3)
        original=model.config.rate_loss_weight
        restored=type(model)(copy.deepcopy(model.config));restored.restore(copy.deepcopy(model.capture()))
        self.assertEqual(restored.rate_multiplier,3.)
        self.assertEqual(restored.config.rate_loss_weight,original)


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


if __name__=='__main__':unittest.main()
