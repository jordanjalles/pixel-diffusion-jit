# FID research loop

Start: 2026-10-09 23:00. First parent: `unify-wclamp01-selfflow3to6-100k-spark` (FID 126.9 from the trained weights at step 100,000).
Each run: 16,000 steps from the parent, FID from the averaged weights. Lower is better.

| time | run | change | FID | reference | margin | verdict | next |
|---|---|---|---|---|---|---|---|
| 23:17 | fidloop-g0-control | none (control) | 126.42 | | | reference | |
| 23:34 | fidloop-g0-ema_teacher | {"selfflow": {"teacher": "ema"}} | 124.39 | 126.42 | 1.50 | PASS: kept, new reference | ema_teacher_slow |
| 23:51 | fidloop-g1-ema_teacher_slow | {"selfflow": {"teacher_decay": 0.9999}} | 124.70 | 124.39 | 1.50 | fail: dropped | noisy_half |
| 00:07 | fidloop-g1-noisy_half | {"selfflow": {"noisy_share": 0.5}} | 124.78 | 124.39 | 1.50 | fail: dropped | noisy_eighth |
| 00:26 | fidloop-g1-noisy_eighth | {"selfflow": {"noisy_share": 0.125}} | 124.63 | 124.39 | 1.50 | fail: dropped | None |
| 00:43 | fidloop-g1-teach_weight_2 | {"selfflow": {"weight": 2.0}} | 124.64 | 124.39 | 1.50 | fail: dropped | teach_weight_half |
| 01:04 | fidloop-g1-teach_weight_half | {"selfflow": {"weight": 0.5}} | 124.31 | 124.39 | 1.50 | fail: dropped | None |
| 01:22 | fidloop-g1-block_layout | {"selfflow": {"noisy_layout": "block"}} | 125.92 | 124.39 | 1.50 | fail: dropped | teach_noisy_only |
| 01:40 | fidloop-g1-teach_noisy_only | {"selfflow": {"teach_on": "noisy"}} | 124.43 | 124.39 | 1.50 | fail: dropped | None |
| 01:59 | fidloop-g1-pair_early | {"selfflow": {"pairs": [[2, 6]]}} | 131.04 | 124.39 | 1.50 | fail: dropped | pair_last |
| 02:17 | fidloop-g1-pair_last | {"selfflow": {"pairs": [[4, 8]]}} | 128.88 | 124.39 | 1.50 | fail: dropped | None |

Plan exhausted at 10-10 02:17: the loop waits for new nodes in plan.json.

| time | run | change | FID | reference | margin | verdict | next |
|---|---|---|---|---|---|---|---|
| 02:38 | fidloop-g1-weight_logflip | {"loss_weight": "logflip"} | 127.09 | 124.39 | 1.50 | fail: dropped | None |
| 02:56 | fidloop-g1-sample_ema_slow | {"ema": 0.9995} | 124.41 | 124.39 | 1.50 | fail: dropped | sample_ema_fast |
| 03:15 | fidloop-g1-sample_ema_fast | {"ema": 0.998} | 124.32 | 124.39 | 1.50 | fail: dropped | None |
| 03:33 | fidloop-g1-readouts_half | {"hparams": {"pyramid_weight": 0.5}} | 124.50 | 124.39 | 1.50 | fail: dropped | readouts_double |
| 03:50 | fidloop-g1-readouts_double | {"hparams": {"pyramid_weight": 2.0}} | 124.23 | 124.39 | 1.50 | fail: dropped | None |
| 04:06 | fidloop-g1-no_teaching | {"selfflow": {"teacher": false}} | 126.48 | 124.39 | 1.50 | fail: dropped | None |
| 04:24 | fidloop-g1-ema_teacher_fast | {"selfflow": {"teacher_decay": 0.99}} | 124.45 | 124.39 | 1.50 | fail: dropped | None |
| 04:43 | fidloop-g1-teach_half_noisy | {"selfflow": {"weight": 0.5, "teach_on": "noisy"}} | 124.21 | 124.39 | 1.50 | fail: dropped | None |

Plan exhausted at 10-10 04:43: the loop waits for new nodes in plan.json.

| time | run | change | FID | reference | margin | verdict | next |
|---|---|---|---|---|---|---|---|
| 05:07 | fidloop-g1-out_align_half | {"selfflow": {"out_weight": 0.5}} | 124.73 | 124.39 | 1.50 | fail: dropped | out_align_tenth |
| 05:26 | fidloop-g1-out_align_tenth | {"selfflow": {"out_weight": 0.1}} | 124.59 | 124.39 | 1.50 | fail: dropped | None |
| 05:45 | fidloop-g1-no_mixed_noise | {"selfflow": {"noisy_share": 0.0}} | 125.44 | 124.39 | 1.50 | fail: dropped | None |
| 06:04 | fidloop-g1-ema_clean_teacher | {"selfflow": {"teacher_input": "clean"}} | 125.95 | 124.39 | 1.50 | fail: dropped | None |

Plan exhausted at 10-10 06:04: the loop waits for new nodes in plan.json.

| time | run | change | FID | reference | margin | verdict | next |
|---|---|---|---|---|---|---|---|
| 06:58 | fidloop-g1-teach_weight_quarter | {"selfflow": {"weight": 0.25}} | 124.96 | 124.39 | 1.50 | fail: dropped | None |
