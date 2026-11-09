# Shared DC quantization step: 0.01 vs 0.0025

2026-11-07. Hypothesis: a 0.01 regional shared DC quantization step suppresses some useful shared updates. The only algorithmic configuration change is `module.feature_dc_update.step`; learned motion gate, 1,000-point DC/log-scale budget, 200 motion steps, 100 attribute steps, train cameras, optimization parameters and v3 point codec remain fixed. Both configurations use shared DC followed by point DC/log-scale residuals and decoded state propagation. No test camera was used for selecting or tuning the step.

No runtime source was modified. `RegionalFeatureDCUpdate` uses the same float32 step for forward quantization and symbol/rate computations; SDR1 embeds its own step in the inner opacity packet header. Zero update, rounding boundaries and metadata/bit-exact decoded features were tested for both steps; two-frame Sear and Flame Salmon smoke tests and independent full-sequence replays passed. Code hashes match the existing Sear 0.01 run. New experiment files: `output/experiments/shared_dc_quant_step_20261107/`.

| Scene / frames | Shared step | Test PSNR dB | Test SSIM | Train cam06 PSNR dB | Full dynamic bytes/frame | Shared bytes/frame | Active shared owners/frame | Seconds/frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sear / 201–230 | 0.01 | 33.48775 | 0.9644335 | 34.87301 | 34,918.23 | 127.23 | 9.83 | 16.718 |
| Sear / 201–230 | 0.0025 | 33.53508 | 0.9644406 | 34.81154 | 35,081.17 | 176.67 | 26.67 | 16.662 |
| Flame Salmon / 1–30 | 0.01 | 30.19082 | 0.9296532 | 33.10919 | 51,743.53 | 139.50 | 14.30 | 18.026 |
| Flame Salmon / 1–30 | 0.0025 | 30.23220 | 0.9297299 | 33.11396 | 51,374.00 | 178.40 | 26.27 | 18.005 |

Smaller step minus original: Sear +0.04733 dB with +162.93 B/frame; 21 frames improve, 9 decline (range -0.0954 to +0.1547 dB). Flame +0.04138 dB with -369.53 B/frame; 25 improve, 5 decline (range -0.0290 to +0.1002 dB). The total-byte difference includes the effect of separate encoded/decoded trajectories; it is not the shared packet cost alone. Sear train cam06 decreased 0.0615 dB, so the test change should not be generalized without more runs.

Sear runs share the exact decoded frame-200 state and reference and use the same code hashes; each then recurs on its own decoded state. Sear original 0.01 data: `output/experiments/shared_dc_sparse_scale_20261106/sear_verified_30frames.json`. New Sear data: `output/experiments/shared_dc_quant_step_20261107/sear_step0025_full30_verified.json`. Both Flame runs start from the same first-frame reference with identical SHA-256 `06100ff01fa41ef2e8c756dc0e2b32568309313fddee73912977a444a65ea3ca`; result files are `flame_step001_full30_verified.json` and `flame_step0025_full30_verified.json` in the new experiment directory. Per-frame bytes, selected and nonzero point union, active owners and timings are recorded there.

The 30-frame values exclude the shared first-frame reference (Sear 11,299,285 B; Flame 11,563,100 B). Sear also requires its common dynamic prefix through frame 200. Flame 0.0025 averages 51,374 B/frame over this 30-frame segment and 19 frames exceed 50,000 B; the project target is a whole-sequence mean, which this short segment cannot determine.

These two short segments show that reducing the quantization step increases active shared-owner counts and coincides with a small test PSNR gain under the measured setup. They do not establish that finer quantization or sharing alone caused the PSNR gain: later states, owner sets and point updates follow different continuous trajectories. One reported training camera is not the train-camera average; cam00 has been examined during prior development. This is not a full N3DV or MeetRoom result and does not establish a general improvement or gate effectiveness.
