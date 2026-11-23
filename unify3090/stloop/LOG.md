# FID research loop

Start: 2026-10-10 07:17. First parent: `unify-wclamp01-selfflow3to6-100k-spark` (FID 126.9 from the trained weights at step 100,000).
Every run starts from that checkpoint with the config change {"selfflow": {"teacher": "ema"}}.
Each run gets the same compute: 20,000 steps at step cost 1, fewer steps for a costlier step. FID from the averaged weights. Lower is better.

| time | run | change | FID | reference | margin | verdict | next |
|---|---|---|---|---|---|---|---|
| 07:37 | stloop-g0-control | none (control) | 122.52 | | | reference | |
| 07:59 | stloop-g0-st_z6 | {"selfteach": {"layer": 6}} (10,000 steps) | 158.87 | 122.52 | 1.50 | fail: dropped | st_z6_out_only |
| 08:21 | stloop-g0-st_z6_out_only | {"selfteach": {"layer": 6, "weight_noised": 0.0}} (12,000 steps) | 116.08 | 122.52 | 1.50 | PASS: kept, new reference | None |
| 08:45 | stloop-g1-st_z6_tenth | {"selfteach": {"layer": 6, "weight_out": 0.1, "weight_noised": 0.1}} (10,000 steps) | 127.11 | 116.08 | 1.50 | fail: dropped | None |
| 09:05 | stloop-g1-st_z3 | {"selfteach": {"layer": 3}} (12,500 steps) | 116.55 | 116.08 | 1.50 | fail: dropped | None |
| 09:27 | stloop-g1-st_z2 | {"selfteach": {"layer": 2}} (15,500 steps) | 120.63 | 116.08 | 1.50 | fail: dropped | None |
| 09:48 | stloop-g1-st_out_w2 | {"selfteach": {"weight_out": 2.0}} (11,000 steps) | 119.19 | 116.08 | 1.50 | fail: dropped | st_out_half |
| 10:09 | stloop-g1-st_out_half | {"selfteach": {"weight_out": 0.5}} (11,000 steps) | 117.40 | 116.08 | 1.50 | fail: dropped | None |
| 10:28 | stloop-g1-st_mild_crop | {"selfteach": {"crop_min": 0.5, "crop_max": 1.0}} (11,000 steps) | 118.81 | 116.08 | 1.50 | fail: dropped | st_small_crop |
| 10:47 | stloop-g1-st_small_crop | {"selfteach": {"crop_min": 0.1, "crop_max": 0.5}} (11,000 steps) | 116.23 | 116.08 | 1.50 | fail: dropped | None |
| 11:08 | stloop-g1-st_no_selfflow_teacher | {"selfflow": {"teacher": false}} (12,000 steps) | 119.51 | 116.08 | 1.50 | fail: dropped | None |
| 16:20 | stloop-g1-st_flow_w2 | {"selfflow": {"weight": 2.0}} (11,000 steps) | 115.70 | 116.08 | 1.50 | fail: dropped | st_flow_w4 |
