# Flame Salmon full sequence: shared DC step 0.0025

Frame 0 is the verified five-level reference; 1–299 are actual encoded dynamic frames. The fixed method uses 200 motion steps, 100 regional DC plus up to 1,000 point DC/log-scale steps, and real v3 bitstreams. All 18 training cameras are used for selection and training; cam00 is test, and cam06 is the reported training sample. Each frame uses its own prior decoded state; independent replay from frame 0, per-frame manifest digests, packet step, camera selection and forward-vs-decode checks passed. No test-camera tuning. Reference bytes are excluded from per-dynamic-frame average and included in the total.

```json
{
  "scope": "frame0 reference and 299 dynamic frames 1\u2013299",
  "reference": {
    "frame": 0,
    "scene": "flame",
    "reference": "/home/gamer-02/Documents/DeepLearning/my4dgs/output/experiments/flame_salmon_shared_dc_step0025_full_20261107/runs/flame_step0025_full/20261107T063551_5f041b/frame_storage/rigid_tree/reference_frame0.npz",
    "reference_bytes": 11563100,
    "reference_state_sha256": "eefa9c436e2692910a876b12cc2a660ad8d11c113ac46a02ca7862391d3b995d",
    "reference_raw_sha256": "630b1cd2b76162d9fe31c803d94d5ca87d25b1667374b2248c990cf7910c8922",
    "evaluation": {
      "train": {
        "camera": "cam06.png",
        "psnr": 33.43490982055664,
        "ssim": 0.950647234916687
      },
      "test": {
        "camera": "cam00.png",
        "psnr": 30.26346206665039,
        "ssim": 0.9295274615287781
      }
    },
    "scope": "fixed reference; test only evaluated, not used for selection"
  },
  "summary_299": {
    "bytes": 45757.38127090301,
    "shared_bytes": 157.71571906354515,
    "shared_scalars": 31.903010033444815,
    "eligible_regions": 149.12040133779263,
    "active_regions": 19.53846153846154,
    "selected": 999.8160535117057,
    "active_union": 578.9397993311037,
    "seconds": 17.952807906549815,
    "selection_seconds": 0.25619026522594424,
    "test_psnr": 30.16452990885961,
    "test_ssim": 0.929437079158515,
    "train_psnr": 32.90071153162315,
    "train_ssim": 0.946995875109797,
    "dynamic_total_bytes": 13681457,
    "over_50000": 27
  },
  "test_psnr_300": 30.164859682718912,
  "test_ssim_300": 0.9294373804330825,
  "total_bytes_including_reference": 25244557,
  "max_forward_error": 0.0,
  "decoded_sha256": "33ffd84c498047cef5860144f34941a2333f711fabc26fc0af4b9bd00dfc6a41",
  "raw_sha256": "f7b6f5e25ee135a65cc5af5c4352c89f89aa7f7f65b4117f8c6863fa7e57d73f",
  "segments": [
    {
      "first": 1,
      "last": 50,
      "test_psnr": 30.22523452758789,
      "test_ssim": 0.9297980070114136,
      "dynamic_bytes": 48798.82,
      "seconds": 17.974322816249913
    },
    {
      "first": 51,
      "last": 100,
      "test_psnr": 30.20002269744873,
      "test_ssim": 0.9295640432834625,
      "dynamic_bytes": 44177.46,
      "seconds": 17.92346563115716
    },
    {
      "first": 101,
      "last": 150,
      "test_psnr": 30.17836528778076,
      "test_ssim": 0.9294580554962159,
      "dynamic_bytes": 45693.96,
      "seconds": 17.915503479614856
    },
    {
      "first": 151,
      "last": 200,
      "test_psnr": 30.048689041137695,
      "test_ssim": 0.9291167294979096,
      "dynamic_bytes": 44749.66,
      "seconds": 17.94454957189271
    },
    {
      "first": 201,
      "last": 250,
      "test_psnr": 30.123912658691406,
      "test_ssim": 0.9293303740024567,
      "dynamic_bytes": 45549.5,
      "seconds": 17.980524643259123
    },
    {
      "first": 251,
      "last": 299,
      "test_psnr": 30.211902696259166,
      "test_ssim": 0.9293535959963896,
      "dynamic_bytes": 45571.163265306124,
      "seconds": 17.97900524387155
    }
  ]
}
```

Data, configs, packet files, individual metrics and hashes: `output/experiments/flame_salmon_shared_dc_step0025_full_20261107/`. Historic `config/best_flame_salmon.yaml` differs in predecessor, gate/attribute training and codec configuration, so it is not a strict matched baseline. The earlier 30-frame step 0.01 and 0.0025 experiments start from the same reference but are separate short continuous trajectories and do not establish a full-length comparison. This result is one scene, not complete N3DV or MeetRoom.
