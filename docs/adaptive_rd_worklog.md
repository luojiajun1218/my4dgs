# Adaptive motion tree and rate-distortion work log

## Agreed scope

- No attribute gate. Translation and rotation are always represented together.
- Parent common motion plus one independent difference per binary split.
- Parent-restricted two-means; spatial and gradient-assisted variants are allowed.
- All current leaves compete regardless of depth. No forced keep percentage.
- No multi-level lookahead special case. Greedy growth can miss deeper-only gains.
- Use quantized motion for rendering; include assignment/structure/model overhead.
- Preserve existing modes, data links, checkpoints, and historical results.

## Reference protocol (2026-09-19)

100 total frames, indices 0–99. Shared static frame 0 checkpoint:
`/home/gamer-02/Documents/DeepLearning/ReCon-GS/output/flame_steak_6frame/20260919T182703_ccad30/ckpt-0-15000.pth`.
Primary means exclude shared frame 0 and cover all 99 dynamic frames. Original
ReCon-GS source is loaded unchanged by `scripts/run_controlled.py`, using its
DyNeRF model/training parameters (100 motion + 100 incremental steps). Resolution
-1 means approximately 1600px width. Cam00 is the test view; training views only
may drive clustering, candidate trials, and decisions. Dataset shuffle is disabled;
frame RNG and sampler RNG are explicitly controlled for all compared runs.
Sampler paths, per-view floating-point metrics, source hashes, motion storage, and
synchronized whole-frame wall times are recorded. Existing unsynchronized FPS is
not used as a performance claim. Extra candidate optimization must be reported and
any success against the 200-step reference distinguished from compute-matched
controls. Early short sequences are exploratory tuning, not independent evidence
of generalization; the complete 100-frame comparison is mandatory.

The initial launch with num_workers=0 failed in the installed libgs image-size
warning function before any training update. Reverted to original num_workers=8.
Failed run remains at `output/flame_steak_recongs_100_controlled_20260919/20260919T204637_eb6b6f`.
Active reference run: `output/flame_steak_recongs_100_controlled_20260919/20260919T204703_b833f8`.

Reference completed successfully at 21:10 on 2026-09-19. All 99 dynamic frames
present: test mean PSNR **32.36370647 dB**, SSIM **0.9581258014**, L1
0.01770789458. Mean motion npz 1,590,447 bytes, incremental PLY 20,884 bytes,
combined 1,611,331 bytes/frame. Synchronized whole-frame time including evaluation
and storage 13.9590 s/frame. These npz bytes are a reference storage measurement,
not an optimized entropy-coded version of upstream motion.

## Compression implementation sources inspected

- QUEEN official `utils/compress_utils.py`:
  https://github.com/NVlabs/queen/blob/main/utils/compress_utils.py
  Reads probability mass over quantization bins, -log2 probabilities, and
  torchac encode/decode with explicit metadata. Its model and attribute gates
  are not being transplanted into this project.
- HAC official `utils/entropy_models.py` and `train.py`:
  https://github.com/YihangChen-ee/HAC/blob/main/utils/entropy_models.py
  https://github.com/YihangChen-ee/HAC/blob/main/train.py
  CDF differences at x +/- Q/2, entropy regularization, and actual codec checks.
- RD4DGS official `utils/entropy_model.py`:
  https://github.com/HyeongminLEE/RD4DGS/blob/main/utils/entropy_model.py
  Explicit integer CDFs, range coding, and side information required by decoder.

These are conceptual/API references, not copied implementations. New code should
keep its probability model explicit, test numerical/codec round trips, and report
both estimated bits and actual full-container bytes rather than conflating them.

## First implementation

`BinaryMotion` is an independent `hierarchy_mode=binary`; old modes stay available.
Root groups are sampled spatial anchors with nearest membership. Subsequent
two-means is restricted to each current leaf and uses per-parent normalized xyz
plus optionally averaged visible translation gradients. Membership is hard, not
overlapping, in this first version. Child labels are explicitly packed and stored;
the decoder never needs image gradients. Each binary split adds one 6D coefficient
(translation + rotation-vector difference), with opposite count-weighted child
contributions. The summed rotation-vector field is converted to a unit quaternion;
this is not claimed to be composition of shared rigid-body SE(3) transforms.

CDF-bin Laplace probabilities yield estimated motion bits. The Laplace scale is
fit per depth/channel by mean absolute quantized symbol magnitude, with detached
statistics; this is an explicit fitted entropy model, not a neural context model.
Quantized motion is rendered using straight-through gradients. Actual coefficients
are arithmetic-coded using existing torchac. Exact integer CDFs, tree metadata,
root membership, child labels and quantization steps are in the resulting npz;
all bytes are counted. Rare huge alphabets use a disclosed lossless raw fallback.
The estimated rate includes compressed structure and conservative CDF/header
costs but is not equated to final container size. Export immediately decodes and
compares motion/rotation to the quantized renderer input. This validates the motion
packet for a given GS ordering, not a standalone complete video playback codec.

Growth trials optimize a no-split branch and a proposed split batch from the same
snapshot, with identical training cameras and step counts. Separate training
cameras compare photometric loss plus lambda * bits / fixed base-GS count. A whole
batch is accepted or rejected; no independent per-node gain is claimed. Rejected
regions can be reconsidered later, but the deeper-only-gain limitation remains.
Probe backward calls are additional work (default maximum 40/frame). The chosen
branch retains its short optimization. A compute-matched reference is necessary
before attributing a quality gain exclusively to the representation/search.

The motion regularizer acts on the reconstructed 6D field, so allocating zero
coefficients does not dilute it. Baseline's original parameter regularizer differs;
the coefficient is exposed for ablation. Static-frame and incremental-GS stages
remain unchanged. Initial exploration uses a 5-dynamic-frame prefix only after
the required 100-total-frame reference is complete.

## Short exploratory trials

1. `flame_binary_rd_v1_g05_q1e4_short_20260919/20260919T211028_711032`:
   xyz+gradient clustering weight .5, lambda_rate=1e-4, q_xyz=1e-4,
   q_rotation=2e-4, root size 72, motion regularizer .1. Completed 5 dynamic frames:
   PSNR 32.0491058 (reference same prefix 32.0924286, delta -0.0433228 dB),
   SSIM .9598902 (delta +.00033054), 398,383 combined bytes/frame. All splits
   rejected: metadata+motion rate outweighed small four-step trial gains.
   Regular 200-step camera paths exactly match the reference. 40 additional
   trial backward calls/frame. This is a failed quality target, not a success.
2. `flame_binary_rd_v1_g05_r1e6_short_20260919/20260919T211222_37a981`:
   changes only lambda_rate to 1e-6. First frame has accepted and rejected rounds,
   reaching 5,670 splits and multiple depths. Completed: PSNR 32.1029640
   (+0.0105354 dB), SSIM .9599853 (+.00042565), combined 421,819 bytes/frame,
   mean 20.895 s/frame, regular camera streams identical. Frames 2–5 rejected
   all further proposals after per-frame rebuild. Tiny prefix gain is insufficient
   to conclude success on 100 frames.
3. `flame_binary_rd_v1_lr1e3_short_20260919/20260919T211430_ceab79`:
   changes learning rate from .0005 to .001 at lambda_rate=1e-6. Completed:
   PSNR 32.0834133 (-0.0090153 dB), SSIM .9596149 (+.00005529). Higher LR
   did not improve the prefix; do not select it as best.
4. `flame_binary_rd_v1_root36_short_20260919`: doubles initial root density
   (root_size=36 vs 72), otherwise trial 2 settings. Completed at
   `20260919T211659_76669a`: PSNR 32.1302029 (+.0377743 dB), SSIM .9600703
   (+.00051064), 462,562 combined bytes/frame. Best prefix so far, not yet
   full-sequence evidence.
5. `flame_binary_rd_v1_batch256_short_20260919/20260919T211931_18a9de`:
   root_size=72 and candidate batch reduced to 256 (previous cap 2048 was larger
   than the initial eligible set, producing coarse all-or-nothing decisions).
   Completed: PSNR 32.0783657 (-.0140629 dB), SSIM .9599573 (+.00039763),
   419,747 combined bytes/frame. Some later-frame splits accepted; quality does
   not surpass the best prefix. Timings use pre-fix dense absent-path evaluation.
6. `flame_binary_rd_allviews_root36_b512_short_20260919/20260919T212256_4227ef`:
   root36, batch512, all-training-view decision, lambda1e-6, gradient weight .5,
   sparse-path performance fix. PSNR 32.1137825 (+.0213539 dB), SSIM .9600477
   (+.00048808), 475,183 combined bytes/frame, 18.580 s/frame. The previous
   root36/batch2048 prefix remains better in PSNR.
7. `flame_binary_rd_spatial_root36_short_20260919`: trial 6 with gradient feature
   weight zero, to determine whether gradient-assisted clustering helps. Completed
   at `20260919T212603_956dc4`: PSNR 32.1324188 (+.0399902 dB), SSIM .9600385
   (+.00047884), 470,575 combined bytes/frame, 18.377 s/frame. Spatial-only is
   competitive; these tiny prefix differences do not establish which is better.

## Full sequence validation

`flame_binary_rd_root36_full100_20260919/20260919T213044_389a06` launched with
trial 4's gradient-assisted root36 configuration, using the tested sparse-path
implementation. The slightly better spatial-only prefix is retained as an
alternative rather than treating a .0022 dB exploratory difference as decisive.
No changes to the running experiment's configuration or source will be made.

**Completed all 99 dynamic frames, exit 0, 21:58 on 2026-09-19.** PSNR
**32.471332762 dB** versus 32.363706473, difference **+.107626289 dB**;
SSIM **.958978134** versus .958125801 (+.000852333). Wins in 78/99 frames.
Regular 200-step camera paths are exactly identical. Whole-frame time 16.5989 s
versus 13.9590 s; 40 additional probe backward calls/frame. Actual motion+
incremental storage 456,040 versus 1,611,331 bytes/frame (reference format caveat
above applies). Maximum round-trip errors: xyz 7.45e-9, rotation 3.73e-9.
Mean accepted split count 271.13/frame, so this result alone does not isolate
the effect of adaptive refinement from roots, regularization, and extra training.
Report, full-precision metrics and unbiased example selection live in the run's
`controlled_report/`; final checkpoint is `ckpt-99-200.pth`.

Interim audit after 32 dynamic frames: PSNR 32.2659127, identical-frame reference
delta +.0735726 dB, SSIM delta +.00054459; 449,415 bytes/frame and 16.467 s/frame.
This is an incomplete prefix, not the final 100-frame result. Later frames began
accepting additional splits after many early rejections.

While that frozen run proceeds, an optional `RowScaledAdam` helper is prepared
but NOT integrated into it. It scales the final Adam update for the difference
coefficient rows (not gradients, whose scale Adam would largely cancel).
Deterministic CPU tests verify ordinary-Adam equivalence at multiplier 1,
specified suffix-step scaling, closure handling, and state restoration. The main
numerical test entrypoint includes these checks. Its value for image quality
remains untested; the rationale is avoiding overly large newborn-detail steps.
After the full run finished, the helper was integrated behind
`binary_detail_lr_scale=1.0` (unchanged default update). Loading older ordinary
Adam checkpoints retains compatibility. Scale .2 is the next single-variable
short experiment; no retroactive changes to the completed run's frozen snapshot.

8. `flame_binary_rd_detail02_short_20260919/20260919T215950_f2b049` completed:
   only difference-update multiplier .2 versus trial 4/full validation. PSNR
   32.1279312 (+.0355026 vs reference prefix), SSIM .9601198 (+.00056016),
   466,925 bytes/frame, 16.611 s/frame. More splits (mean 1391) did not beat
   trial 4's prefix PSNR; not selected merely for producing more hierarchy.
9. `flame_binary_rd_root18_short_20260919`: root size 18, ordinary detail step
   multiplier 1, otherwise the validated root36 recipe. A final root-density
   quality check before the stronger 240-backward reference. Completed at
   `20260919T220239_0a4aee`: PSNR 32.1255219 (+.0330933), SSIM .9600996
   (+.00053990), 523,458 bytes/frame. Higher storage without better prefix PSNR;
   retain root36 as the best fully validated recipe.

## Stronger backward-count reference

`flame_recongs_240backward_full100_20260919/20260919T220421_2f05e2` runs unchanged
upstream source with 140 motion + 100 incremental steps (240 total), versus our
100 + 100 regular plus 40 branch-trial backward calls. All of its 140 motion
updates contribute to the retained model; ours retains only 20 of the 40 trial
updates. This matches backward counts, NOT all forward/search/I/O work or wall
time. Main step streams necessarily differ in length; each phase still visits
the same training cameras uniformly over complete 20-view cycles. All reference
parameters other than motion/total step counts remain unchanged.

An optional `binary_decision_all_views` now measures RD distortion over all cached
TRAINING views instead of a disjoint subset. This is fitting the full training
objective, not held-out validation. It is explicitly logged; cam00 remains excluded.
Both modes are integration-tested with matched branch budgets. The original
disjoint-subset mode remains the default for reproducibility of the first trials.

Performance fix after launching trial 5: absent paths originally gathered
coefficient row zero with zero weights, creating severe CUDA scatter-backward
contention in sparse trees. Field evaluation now gathers/scatters only points
with an active path. Outputs and gradients are checked against the dense formula.
This is a mathematically equivalent implementation optimization; trial 5's frozen
source snapshot still contains the slower version, and its timings must not be
used as timings of the revised implementation.
