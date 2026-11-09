# Sear full sequence: shared DC step 0.0025 vs 0.01

Common first-frame reference, seed, 20 training views and code. Only shared DC quantization step changes. Each run uses its own real decoded predecessor. Test cam00 is for evaluation; it has been inspected during development. Dynamic statistics cover frames 1–299; full 300-timepoint metrics include evaluated frame 0. Bytes include full dynamic packages and manifests; first reference is separately included in total. Independent replay and forward checks passed. Separate trajectories and GPU nondeterminism preclude attributing all differences to the changed step. No Coffee training was started.

```json
{
  "scope": "frame 0 reference + frames 1\u2013299",
  "new": {
    "bytes": 34812.52173913043,
    "shared_bytes": 146.9264214046823,
    "shared_scalars": 28.49163879598662,
    "eligible_regions": 128.77926421404683,
    "active_regions": 18.70903010033445,
    "selected": 999.0602006688963,
    "active_union": 436.4046822742475,
    "seconds": 16.659787408448445,
    "selection_seconds": 0.22467673662855261,
    "test_psnr": 33.999942562651874,
    "test_ssim": 0.9656634037709954,
    "train_psnr": 35.231788762835755,
    "train_ssim": 0.9670024482302841,
    "dynamic_total_bytes": 10408944,
    "over_50000": 0
  },
  "old": {
    "frames": 299,
    "test_psnr": 33.97548664054743,
    "test_ssim": 0.9656121166653457,
    "train_cam06_psnr": 35.23283631187618,
    "train_cam06_ssim": 0.9669933297163666,
    "dynamic_bytes": 34862.3143812709,
    "training_seconds": 16.724217542504384,
    "selection_seconds": 0.2247747633494063,
    "nonzero_union": 417.10033444816054,
    "over_50000_bytes": 0
  },
  "delta_test_psnr": 0.024455922104443195,
  "wins": 202,
  "losses": 97,
  "delta_min": -0.2665672302246094,
  "delta_max": 0.3928871154785156,
  "new_total_bytes_including_reference": 21708229,
  "old_total_bytes_including_reference": 21723117,
  "reference": {
    "frame": 0,
    "scene": "sear",
    "reference": "/home/gamer-02/Documents/DeepLearning/my4dgs/output/experiments/sear_shared_dc_step0025_full_20261107/runs/sear_step0025_full/20261107T050748_7d9042/frame_storage/rigid_tree/reference_frame0.npz",
    "reference_bytes": 11299285,
    "reference_state_sha256": "3da505c71b228f9f03bbd429a812f2d1571972a18b2069ecd4b79882195e2a9e",
    "reference_raw_sha256": "e27068e07aa6050141141bf17686514b2c4c1c05525213623731ba1bd9eac14d",
    "evaluation": {
      "train": {
        "camera": "cam06.png",
        "psnr": 36.034507751464844,
        "ssim": 0.9714687466621399
      },
      "test": {
        "camera": "cam00.png",
        "psnr": 34.19684600830078,
        "ssim": 0.9673956036567688
      }
    },
    "scope": "fixed reference; test only evaluated, not used for selection"
  },
  "test_psnr_300_new": 34.0005989074707,
  "test_ssim_300_new": 0.9656691777706147,
  "test_psnr_300_old": 33.97622450510661,
  "test_ssim_300_old": 0.9656180616219838,
  "segments": [
    {
      "first": 1,
      "last": 50,
      "psnr_new": 34.32300857543945,
      "psnr_old": 34.323473358154295,
      "bytes_new": 39326.78,
      "bytes_old": 39268.76
    },
    {
      "first": 51,
      "last": 100,
      "psnr_new": 34.15106605529785,
      "psnr_old": 34.12289825439453,
      "bytes_new": 35569.54,
      "bytes_old": 35682.46
    },
    {
      "first": 101,
      "last": 150,
      "psnr_new": 34.04957374572754,
      "psnr_old": 34.021954193115235,
      "bytes_new": 34868.22,
      "bytes_old": 34874.24
    },
    {
      "first": 151,
      "last": 200,
      "psnr_new": 34.11074508666992,
      "psnr_old": 34.041761322021486,
      "bytes_new": 33791.3,
      "bytes_old": 34111.12
    },
    {
      "first": 201,
      "last": 250,
      "psnr_new": 33.55201820373535,
      "psnr_old": 33.51413444519043,
      "bytes_new": 33647.96,
      "bytes_old": 33558.6
    },
    {
      "first": 251,
      "last": 299,
      "psnr_new": 33.80943352835519,
      "psnr_old": 33.825702589385365,
      "bytes_new": 31607.224489795917,
      "bytes_old": 31613.73469387755
    }
  ]
}
```

Per-frame results and provenance are in `output/experiments/sear_shared_dc_step0025_full_20261107/`; previous run is `output/experiments/n3dv_shared_dc_scale_full_20261107/sear_full_verified.json`. This is one scene, not full N3DV or MeetRoom.
