# Sear Steak: 1,000 vs 5,000 point update budget

Date: 2026-11-07. Existing 200-step motion/opacity and hard motion gate; shared DC plus error-selected sparse per-point DC/log-scale updates, v3 stream, seed 0. All 20 training cameras used for selection/training; cam00 is the test view. Train PSNR below is cam06, not an average across training cameras. The test view has previously been examined during development. No test scores were used to select settings for this experiment.

Early frames 31–60 use the same historical frame-30 decoded predecessor; late frames 201–230 use the same historical frame-200 decoded predecessor. Each branch then uses its own actual decoded previous frame. Two-frame smoke checks and independent decoding from the first reference through the end of each run passed. The 1,000- and 5,000-point runs have the same checked code hashes and differ in point budget. The early zero-point run uses the same checked code and disables the extra attribute stage. The older late zero-point run has a shared verified prefix, but historical code identity is not fully verifiable; interpret its comparison accordingly. Files and code hash records are in `output/experiments/sear_budget5000_20261107/`.

| Segment | Point budget | Test PSNR dB | Test SSIM | Train cam06 PSNR dB | Dynamic B/frame | Total 30-frame dynamic B | s/frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| 31–60 | 0 | 34.2274 | 0.965649 | 35.2708 | 31,935.13 | 958,054 | 12.594 |
| 31–60 | 1,000 | 34.2164 | 0.965915 | 35.3830 | 35,572.53 | 1,067,176 | 16.625 |
| 31–60 | 5,000 | 34.2105 | 0.966216 | 35.3946 | 41,823.07 | 1,254,692 | 16.629 |
| 201–230 | 0 | 32.6601 | 0.963783 | 33.7620 | 31,004.53 | 930,136 | 12.578 |
| 201–230 | 1,000 | 33.4878 | 0.964433 | 34.8730 | 34,918.23 | 1,047,547 | 16.718 |
| 201–230 | 5,000 | 33.5302 | 0.964910 | 34.9233 | 42,667.47 | 1,280,024 | 16.669 |

The 5,000 vs 1,000 test-PSNR per-frame differences in 31–60 averaged -0.00594 dB (14 positive, 16 negative; range -0.0891 to +0.0703 dB); in 201–230 averaged +0.04244 dB (24 positive, 6 negative; range -0.1024 to +0.1363 dB). Byte cost increased by 6,250.53 and 7,749.23 B/frame respectively. Mean test SSIM rose by 0.000301 and 0.000476, respectively. These are small observed differences, and no claim of generalization across scenes or whole sequences is justified.

For 5,000-point runs the mean selected/nonzero transmitted union point counts are 4,994.8/1,978.47 in frames 31–60 and 4,997.87/2,168.77 in frames 201–230. The actual complete dynamic byte count includes all stored per-frame packages and metadata; the 11,299,285-byte shared reference is not included. The preserved dynamic prefix 1–30 is 1,097,306 B and prefix 1–200 is 6,529,789 B; add the reference and required prefix to obtain a decodable stream. Three early 5,000-point frames exceeded 50,000 B; none did in the late 30 frames. The 50 KB criterion is a sequence average target, not a per-frame hard limit.

The early extension rule was recorded before reviewing test scores: the 5,000-point late train cam06 mean had to exceed the previous 1,000-point mean and average real dynamic bytes be at most 55,000 B/frame. Both held. Per-frame metrics, per-package bytes, selected/nonzero counts and forward/decoded checks: `output/experiments/sear_budget5000_20261107/early30_verified.json` and `late30_verified.json`. The earlier late 1,000/zero data are `output/experiments/shared_dc_sparse_scale_20261106/sear_verified_30frames.json`. Raw logs, configs and distinct outputs remain in the experiment directory.

This evaluates two 30-frame Sear segments only, not 300 complete frames or all N3DV. Cam06 is a single training view. Separate decoded trajectories and previously examined cam00 limit causal and generalization claims. Results do not establish that changing the learned gate caused the observed differences.
