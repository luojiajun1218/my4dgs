"""Capture a genuine frame-plan step-150 checkpoint and probe image-only gate gradients."""
import gc
import json
from pathlib import Path
import torch
from pipeline.trainer import Trainer
from pipeline.module import motion_photometric_loss

class ProbeFinished(Exception):
    pass

original_fit = Trainer._fit_views

def probe_fit(self, fetcher, step, *, defer_metrics=False):
    if step == 151:
        module = self.module
        view = module.frame_plan_training_views(step, 1)[0]
        module.motion.optimizer.zero_grad(set_to_none=True)
        enabled = bool(getattr(module.motion, 'refine_image_gate_ste', False))
        module.motion.refine_image_gate_ste = False
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(module.device)
        hard = module.forward(view, training=True)['render']
        baseline_loss = motion_photometric_loss(hard, view.image,
                                      module.config.lambda_dssim, 1., metric='l1')
        gate = module.motion.gate_logits
        baseline_gradient = torch.autograd.grad(baseline_loss, gate, allow_unused=True)[0]
        baseline_peak = torch.cuda.max_memory_allocated(module.device) if torch.cuda.is_available() else None
        baseline_value = float(baseline_loss.detach())
        hard_snapshot = hard.detach().clone()
        del hard, baseline_loss
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.synchronize(module.device)
        module.motion.refine_image_gate_ste = enabled
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(module.device)
        package = module.forward(view, training=True)
        loss = motion_photometric_loss(package['render'], view.image,
                                      module.config.lambda_dssim, 1., metric='l1')
        gradient = torch.autograd.grad(loss, gate, allow_unused=True)[0]
        ste_peak = torch.cuda.max_memory_allocated(module.device) if torch.cuda.is_available() else None
        eligible = module.motion.split_eligible
        paths = module.motion.spatial_paths
        matches = paths.eq(module.motion.owner[:, None])
        if not bool(matches.sum(1).eq(1).all()):
            raise ValueError('Gate gradient diagnostic lost unique hard owner')
        depth = matches.long().argmax(1)
        reaches = torch.arange(paths.shape[1], device=paths.device)[None, :] <= depth[:, None]
        reachable = torch.zeros_like(eligible)
        reachable[paths[reaches]] = True
        reachable &= eligible
        unreachable = eligible & ~reachable
        terms = module.motion.loss_terms(step, diagnostics=False)
        def gate_grad(term, retain):
            if not term.requires_grad: return None
            return torch.autograd.grad(term, gate, allow_unused=True, retain_graph=retain)[0]
        rate_gradient = gate_grad(terms['rate_loss'], True)
        relative_gradient = gate_grad(terms['relative_motion_loss'], False)
        def norm(value, mask=None):
            if value is None: return None
            return float((value if mask is None else value[mask]).norm())
        def cosine(a, b, mask):
            if a is None or b is None: return None
            a,b=a[mask],b[mask]
            denominator=a.norm()*b.norm()
            return None if float(denominator)==0 else float((a*b).sum()/denominator)
        difference = (hard_snapshot-package['render'].detach()).abs()
        provenance = self.output_dir/'gate_fork_provenance.json'
        checkpoint = (json.loads(provenance.read_text())['checkpoint'] if provenance.exists() else
                      str(self.output_dir / f'rigid-ckpt-{self.current_frame}-150.pth'))
        result = dict(frame=int(self.current_frame), step=step,
            camera=Path(str(view.path)).name, objective='photometric L1+DSSIM only; no rate or motion regularization',
            loss=float(loss.detach()), gate_requires_grad=bool(gate.requires_grad),
            gate_gradient_none=gradient is None,
            gate_gradient_norm=None if gradient is None else float(gradient.norm()),
            eligible_gate_gradient_norm=None if gradient is None else float(gradient[eligible].norm()),
            invalid_gate_gradient_max=None if gradient is None else float(gradient[~eligible].abs().max()),
            hard_forward_equal=torch.equal(hard_snapshot,package['render'].detach()),
            hard_forward_max_abs=float(difference.max()), baseline_image_loss=baseline_value,
            baseline_gate_gradient_none=baseline_gradient is None, ste_enabled=enabled,
            baseline_peak_allocated_bytes=baseline_peak, ste_peak_allocated_bytes=ste_peak,
            image_gradient_reachable_norm=norm(gradient, reachable), image_gradient_unreachable_norm=norm(gradient, unreachable),
            rate_gradient_norm=norm(rate_gradient), rate_gradient_reachable_norm=norm(rate_gradient, reachable),
            relative_motion_gradient_norm=norm(relative_gradient),
            image_rate_cosine_all_eligible=cosine(gradient, rate_gradient, eligible),
            image_rate_cosine_reachable=cosine(gradient, rate_gradient, reachable),
            reachable_eligible_gate_count=int(reachable.sum()), unreachable_eligible_gate_count=int(unreachable.sum()),
            eligible_gate_count=int(eligible.sum()), candidate_nodes=int(gate.numel()), checkpoint_150=checkpoint)
        (self.output_dir/'gate_image_gradient_151.json').write_text(json.dumps(result,indent=2))
        print('GATE_IMAGE_ONLY', json.dumps(result), flush=True)
        raise ProbeFinished()
    result = original_fit(self, fetcher, step, defer_metrics=defer_metrics)
    if step == 150:
        self.save_checkpoint()
        print('GATE_FORK_150', self.ckpt_save_path, flush=True)
    return result

Trainer._fit_views = probe_fit
import main as project_main
if __name__ == '__main__':
    try:
        project_main.app.run(project_main.main)
    except ProbeFinished:
        print('Gate image-only probe complete.', flush=True)
