"""Training-only counterfactual evidence for existing sparse motion rows.

The probes reuse each current training camera. They change only the old-point
residual amplitude; newborns follow their existing live source coordinates.
No gate, score, or camera observation is transmitted to the decoder.
"""
import torch


OBJECTIVE = 'conditional_motion_rd_v1'
PROBE_ALPHAS = (.25, .75)


def evidence_window(config):
    window = config.motion_fit_window_steps or 20
    return config.motion_steps - 2 * window, config.motion_steps - window


class MotionEvidence:
    def __init__(self, like):
        self.total = like.new_zeros(len(like))
        self.seen = torch.zeros(len(like), dtype=torch.bool, device=like.device)
        self.samples = 0
        self.done = False

    @torch.no_grad()
    def add(self, gradient, quantized_translation, alpha):
        if self.done:
            raise ValueError('Cannot change frozen motion evidence')
        score = -(gradient * quantized_translation).sum(-1) / alpha
        self.total.add_(score)
        self.seen |= gradient.ne(0).any(-1)
        self.samples += 1

    def retain(self, rows):
        self.total = self.total[rows].clone()
        self.seen = self.seen[rows].clone()

    def capture(self):
        return dict(objective=OBJECTIVE, total=self.total.detach().clone(),
                    seen=self.seen.clone(), samples=self.samples, done=self.done)

    @classmethod
    def restore(cls, packet, like):
        if (not isinstance(packet, dict) or packet.get('objective') != OBJECTIVE or
                set(packet) != {'objective', 'total', 'seen', 'samples', 'done'}):
            raise ValueError('Checkpoint lacks the conditional motion objective')
        total, seen = packet['total'], packet['seen']
        samples, done = packet['samples'], packet['done']
        if (not torch.is_tensor(total) or total.shape != (len(like),) or
                not total.is_floating_point() or not bool(torch.isfinite(total).all()) or
                not torch.is_tensor(seen) or seen.shape != total.shape or seen.dtype != torch.bool or
                isinstance(samples, bool) or not isinstance(samples, int) or samples < 0 or
                not isinstance(done, bool) or (done and not samples)):
            raise ValueError('Invalid conditional motion evidence state')
        result = cls(like)
        result.total = total.to(like).clone()
        result.seen = seen.to(device=like.device).clone()
        result.samples, result.done = samples, done
        return result


def probe_motion_evidence(module, viewpoint, step, loss_function):
    motion = module.motion
    if not motion.config.counterfactual_residual_support or module.motion_committed:
        return
    start, freeze = evidence_window(motion.config)
    if not start <= step < freeze:
        return
    evidence = motion._motion_evidence
    if evidence is None or evidence.done or not motion.amplitude_refit_active:
        raise ValueError('Motion probes require unfrozen evidence during amplitude fitting')
    symbols = (motion.gs_translation.detach() / motion.gs_steps[:3]).round()
    translation = symbols * motion.gs_steps[:3]
    try:
        for alpha in PROBE_ALPHAS:
            motion._counterfactual_probe_scale = alpha
            # training=False disables contribution collection and training-only
            # bookkeeping; quantized pose arithmetic still carries autograd.
            image = module.forward(viewpoint, training=False)['render']
            gradient, = torch.autograd.grad(loss_function(image), motion.gs_translation)
            evidence.add(gradient.detach(), translation, alpha)
    finally:
        motion._counterfactual_probe_scale = None
        # A probe also temporarily rebinds newborn source coordinates. Restore
        # the physical pose before any subsequent attribute or storage update.
        if getattr(module, 'early_initialized', False):
            with torch.no_grad():
                xyz, rotation = motion(step, training=False)
                module.births.bind_source_pose(xyz, rotation, detach=True)


@torch.no_grad()
def finalize_motion_evidence(motion, step):
    if not motion.config.counterfactual_residual_support:
        return
    evidence = motion._motion_evidence
    start, freeze = evidence_window(motion.config)
    if step < freeze or evidence.done:
        return
    if not motion.amplitude_refit_active or evidence.samples < freeze - start:
        raise ValueError('Insufficient training evidence before residual selection')
    if not bool(torch.isfinite(evidence.total).all()):
        raise ValueError('Counterfactual motion evidence must be finite')

    # Calibrate the per-row byte-context proxy to the production packet's
    # measured marginal length. Metadata and shared motion are held fixed.
    motion._measure_codec_rate(step, force=True)
    rate = motion._codec_rate
    rt, rr, gt, gr = [v.detach().cpu() for v in motion._values(hard=True)]
    full_bytes = rate.last_packet_bytes
    zero_bytes = len(rate.packet(rt, rr, torch.zeros_like(gt), torch.zeros_like(gr),
                                 motion.opened.detach().cpu()))
    t, r = gt.to(motion.gs_translation), gr.to(motion.gs_rotation)
    st, sr = t.ne(0).any(-1).to(t), r.ne(0).any(-1).to(r)
    row_bytes = rate.costs(t, r, st, sr, 1)
    marginal_bytes = max(0, full_bytes - zero_bytes)
    row_bytes *= marginal_bytes / row_bytes.sum().clamp_min(1.)
    utility = evidence.total / evidence.samples
    ramp = min(max((step-motion.config.rate_warmup_steps) /
                   motion.config.rate_ramp_steps, 0.), 1.)
    byte_price = motion.config.rate_loss_weight * ramp / motion.rate_normalization_bytes
    active = st.bool() | sr.bool()
    prune = active & evidence.seen & (utility < byte_price * row_bytes)

    for name in ('gs_translation', 'gs_rotation'):
        parameter = getattr(motion, name)
        parameter[prune] = 0
        motion._amplitude_refit_support[name][prune] = False
        state = motion.optimizer.state.get(parameter, {})
        for key in ('exp_avg', 'exp_avg_sq'):
            if key in state:
                state[key][prune] = 0
    evidence.done = True
    motion.flat_region_diagnostics['conditional_motion_support'] = dict(
        objective=OBJECTIVE, probe_window=[start, freeze], probe_alphas=list(PROBE_ALPHAS),
        probe_samples=evidence.samples, candidate_rows=int(active.sum()),
        disabled_rows=int(prune.sum()), selected_rows=int((active & ~prune).sum()),
        measured_candidate_motion_bytes=full_bytes,
        measured_zero_residual_motion_bytes=zero_bytes,
        selected_residual_byte_proxy=float(row_bytes[active & ~prune].sum()),
        byte_price=float(byte_price), extra_render_calls=evidence.samples)
