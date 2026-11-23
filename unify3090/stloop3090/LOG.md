# FID research loop

Start: 2026-10-10 07:33. First parent: `unify-wclamp01-selfflow3to6-100k-spark` (FID nan from the trained weights at step 100,000).
Every run starts from that checkpoint with the config change {"selfflow": {"teacher": "ema"}}.
Each run gets the same compute: 16,000 steps at step cost 1, fewer steps for a costlier step. FID from the averaged weights. Lower is better.

| time | run | change | FID | reference | margin | verdict | next |
|---|---|---|---|---|---|---|---|
| 07:52 | stloop90-g0-control | none (control) | 124.57 | | | reference | |
| 08:10 | stloop90-g0-st_z3 | {"selfteach": {"layer": 3}} (10,000 steps) | 123.09 | 124.57 | 1.50 | fail: dropped | st_z3_out_only |
| 08:28 | stloop90-g0-st_z3_out_only | {"selfteach": {"layer": 3, "weight_noised": 0.0}} (11,500 steps) | 116.91 | 124.57 | 1.50 | PASS: kept, new reference | None |
| 08:43 | stloop90-g1-st_z8 | {"selfteach": {"layer": 8}} (7,000 steps) | 118.80 | 116.91 | 1.50 | fail: dropped | st_z8_out_only |
| 09:01 | stloop90-g1-st_z8_out_only | {"selfteach": {"layer": 8, "weight_noised": 0.0}} (8,500 steps) | 117.18 | 116.91 | 1.50 | fail: dropped | None |
| 09:19 | stloop90-g1-st_z1 | {"selfteach": {"layer": 1}} (13,500 steps) | 120.78 | 116.91 | 1.50 | fail: dropped | None |
| 09:35 | stloop90-g1-st_z6_out | {"selfteach": {"layer": 6}} (8,500 steps) | 119.17 | 116.91 | 1.50 | fail: dropped | None |
| 09:53 | stloop90-g1-st_out_w2 | {"selfteach": {"weight_out": 2.0}} (11,500 steps) | 118.73 | 116.91 | 1.50 | fail: dropped | st_out_half |
| 10:10 | stloop90-g1-st_out_half | {"selfteach": {"weight_out": 0.5}} (11,500 steps) | 117.78 | 116.91 | 1.50 | fail: dropped | None |

Plan exhausted at 10-10 10:10: the loop waits for new nodes in plan.json.

| time | run | change | FID | reference | margin | verdict | next |
|---|---|---|---|---|---|---|---|
| 11:06 | stloop90-g1-st_z4_out | {"selfteach": {"layer": 4}} (10,500 steps) | 117.84 | 116.91 | 1.50 | fail: dropped | None |
| 11:24 | stloop90-g1-st_wide_crop | {"selfteach": {"crop_min": 0.1, "crop_max": 1.0}} (11,500 steps) | 117.37 | 116.91 | 1.50 | fail: dropped | None |
| 11:42 | stloop90-g1-st_whole_image | {"selfteach": {"crop_min": 1.0, "crop_max": 1.0, "overhang": 0.0}} (11,500 steps) | 117.39 | 116.91 | 1.50 | fail: dropped | None |
| 12:01 | stloop90-g1-st_judge_ema | {"selfteach": {"judge": "ema"}} (11,500 steps) | 117.41 | 116.91 | 1.50 | fail: dropped | None |
| 12:18 | stloop90-g1-st_z3z6 | {"selfteach": {"layers": [3, 6]}} (8,500 steps) | 118.67 | 116.91 | 1.50 | fail: dropped | None |
| 12:37 | stloop90-g1-st_no_selfflow_teacher | {"selfflow": {"teacher": false}} (13,000 steps) | 121.69 | 116.91 | 1.50 | fail: dropped | None |
| 12:55 | stloop90-g1-st_small_crop | {"selfteach": {"crop_min": 0.1, "crop_max": 0.5}} (11,500 steps) | 115.95 | 116.91 | 1.50 | fail: dropped | None |
| 13:13 | stloop90-g1-st_inside_crop | {"selfteach": {"overhang": 0.0}} (11,500 steps) | 117.19 | 116.91 | 1.50 | fail: dropped | None |
| 13:31 | stloop90-g1-st_judge_t07 | {"selfteach": {"judge_t": 0.7}} (11,500 steps) | 134.76 | 116.91 | 1.50 | fail: dropped | st_judge_t09 |
| 13:49 | stloop90-g1-st_judge_t09 | {"selfteach": {"judge_t": 0.9}} (11,500 steps) | 128.12 | 116.91 | 1.50 | fail: dropped | None |
| 14:07 | stloop90-g1-st_flow_w2 | {"selfflow": {"weight": 2.0}} (11,500 steps) | 115.39 | 116.91 | 1.50 | PASS: kept, new reference | None |
| 14:24 | stloop90-g2-st_share40 | {"selfflow": {"noisy_share": 0.4}} (11,500 steps) | 115.35 | 115.39 | 1.50 | fail: dropped | None |

Plan exhausted at 10-10 14:24: the loop waits for new nodes in plan.json.

| time | run | change | FID | reference | margin | verdict | next |
|---|---|---|---|---|---|---|---|
| 15:12 | stloop90-g2-st_flow_w4 | {"selfflow": {"weight": 4.0}} (11,500 steps) | 114.38 | 115.39 | 1.50 | fail: dropped | st_flow_w3 |
| 15:31 | stloop90-g2-st_flow_w3 | {"selfflow": {"weight": 3.0}} (11,500 steps) | 114.94 | 115.39 | 1.50 | fail: dropped | None |
| 15:46 | stloop90-g2-st_small_crop_b | {"selfteach": {"crop_min": 0.1, "crop_max": 0.5}} (11,500 steps) | 114.43 | 115.39 | 1.50 | fail: dropped | None |
| 16:04 | stloop90-g2-st_small_crop_share40 | {"selfteach": {"crop_min": 0.1, "crop_max": 0.5}, "selfflow": {"noisy_share": 0.4}} (11,500 steps) | 114.85 | 115.39 | 1.50 | fail: dropped | None |
| 16:21 | stloop90-g2-st_share60 | {"selfflow": {"noisy_share": 0.6}} (11,500 steps) | 116.06 | 115.39 | 1.50 | fail: dropped | None |
| 16:39 | stloop90-g2-st_teach_noisy | {"selfflow": {"teach_on": "noisy"}} (11,500 steps) | 115.80 | 115.39 | 1.50 | fail: dropped | None |
| 16:57 | stloop90-g2-st_teacher_slow | {"selfflow": {"teacher_decay": 0.9999}} (11,500 steps) | 115.86 | 115.39 | 1.50 | fail: dropped | None |
| 17:15 | stloop90-g2-st_ctx_half | {"selfflow": {"ctx_weight": 0.5}} (11,500 steps) | 115.46 | 115.39 | 1.50 | fail: dropped | None |
| 17:32 | stloop90-g2-st_out_w075 | {"selfteach": {"weight_out": 0.75}} (11,500 steps) | 115.88 | 115.39 | 1.50 | fail: dropped | None |
| 17:50 | stloop90-g2-st_w4_small_crop | {"selfflow": {"weight": 4.0}, "selfteach": {"crop_min": 0.1, "crop_max": 0.5}} (11,500 steps) | 114.29 | 115.39 | 1.50 | fail: dropped | None |
| 18:08 | stloop90-g2-st_tiny_crop_b | {"selfteach": {"crop_min": 0.05, "crop_max": 0.3}} (11,500 steps) | 117.24 | 115.39 | 1.50 | fail: dropped | None |
| 18:26 | stloop90-g2-st_crop_02_05 | {"selfteach": {"crop_min": 0.2, "crop_max": 0.5}} (11,500 steps) | 113.50 | 115.39 | 1.50 | PASS: kept, new reference | None |
| 18:44 | stloop90-g3-st_flow_w6 | {"selfflow": {"weight": 6.0}} (11,500 steps) | 113.09 | 113.50 | 1.50 | fail: dropped | None |
| 19:01 | stloop90-g3-st_teacher_fast | {"selfflow": {"teacher_decay": 0.99}} (11,500 steps) | 113.84 | 113.50 | 1.50 | fail: dropped | None |
| 19:20 | stloop90-g3-st_crop_03_05 | {"selfteach": {"crop_min": 0.3, "crop_max": 0.5}} (11,500 steps) | 114.02 | 113.50 | 1.50 | fail: dropped | None |
| 19:38 | stloop90-g3-st_crop_02_04 | {"selfteach": {"crop_min": 0.2, "crop_max": 0.4}} (11,500 steps) | 113.51 | 113.50 | 1.50 | fail: dropped | None |
| 19:56 | stloop90-g3-st_crop_02_07 | {"selfteach": {"crop_min": 0.2, "crop_max": 0.7}} (11,500 steps) | 114.63 | 113.50 | 1.50 | fail: dropped | None |
| 20:13 | stloop90-g3-st_crop_fixed_035 | {"selfteach": {"crop_min": 0.35, "crop_max": 0.35}} (11,500 steps) | 113.52 | 113.50 | 1.50 | fail: dropped | None |
| 20:22 | stloop90-g3-st_flow_w4_c | {"selfflow": {"weight": 4.0}} (11,500 steps) | FAILED TO RUN | 113.50 | 1.50 | fail: dropped | None |
