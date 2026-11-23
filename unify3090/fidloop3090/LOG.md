# FID research loop

Start: 2026-10-10 00:51. First parent: `unify-wclamp01-selfflow3to6-100k-spark` (FID nan from the trained weights at step 100,000).
Each run: 16,000 steps from the parent, FID from the averaged weights. Lower is better.

| time | run | change | FID | reference | margin | verdict | next |
|---|---|---|---|---|---|---|---|
| 01:09 | fidloop90-g0-control | none (control) | 126.78 | | | reference | |
| 01:29 | fidloop90-g0-teacher_clean | {"selfflow": {"teacher_input": "clean"}} | 127.67 | 126.78 | 1.50 | fail: dropped | pairs_three |
| 01:49 | fidloop90-g0-pairs_three | {"selfflow": {"pairs": [[2, 5], [3, 6], [4, 7]]}} | 135.65 | 126.78 | 1.50 | fail: dropped | pair_deep_target |
| 02:12 | fidloop90-g0-pair_deep_target | {"selfflow": {"pairs": [[3, 8]], "to": 8}} | 127.54 | 126.78 | 1.50 | fail: dropped | ctx_half |
| 02:32 | fidloop90-g0-ctx_half | {"selfflow": {"ctx_weight": 0.5}} | 126.78 | 126.78 | 1.50 | fail: dropped | ctx_double |
| 02:52 | fidloop90-g0-ctx_double | {"selfflow": {"ctx_weight": 2.0}} | 126.67 | 126.78 | 1.50 | fail: dropped | None |
| 03:12 | fidloop90-g0-mask_student | {"selfflow": {"noisy_t": "mask"}} | 132.14 | 126.78 | 1.50 | fail: dropped | pairs_three |
| 03:32 | fidloop90-g0-pairs_three | {"selfflow": {"pairs": [[2, 5], [3, 6], [4, 7]]}} | 136.42 | 126.78 | 1.50 | fail: dropped | pair_deep_target |
| 03:53 | fidloop90-g0-pair_deep_target | {"selfflow": {"pairs": [[3, 8]], "to": 8}} | 127.73 | 126.78 | 1.50 | fail: dropped | ctx_half |
| 04:13 | fidloop90-g0-ctx_half | {"selfflow": {"ctx_weight": 0.5}} | 126.83 | 126.78 | 1.50 | fail: dropped | ctx_double |
| 04:33 | fidloop90-g0-ctx_double | {"selfflow": {"ctx_weight": 2.0}} | 127.05 | 126.78 | 1.50 | fail: dropped | None |
| 04:50 | fidloop90-g0-ema_teacher | {"selfflow": {"teacher": "ema"}} | 124.57 | 126.78 | 1.50 | PASS: kept, new reference | mask_student |
| 05:07 | fidloop90-g1-mask_student | {"selfflow": {"noisy_t": "mask"}} | 128.54 | 124.57 | 1.50 | fail: dropped | None |
| 05:26 | fidloop90-g1-no_selfflow | {"selfflow": null} | 126.58 | 124.57 | 1.50 | fail: dropped | None |
| 05:45 | fidloop90-g1-teach_weight_half | {"selfflow": {"weight": 0.5}} | 124.92 | 124.57 | 1.50 | fail: dropped | None |
| 06:04 | fidloop90-g1-out_align_1 | {"selfflow": {"out_weight": 1.0}} | 125.10 | 124.57 | 1.50 | fail: dropped | None |

Plan exhausted at 10-10 06:04: the loop waits for new nodes in plan.json.

| time | run | change | FID | reference | margin | verdict | next |
|---|---|---|---|---|---|---|---|
| 06:58 | fidloop90-g1-readouts_double | {"hparams": {"pyramid_weight": 2.0}} | 124.62 | 124.57 | 1.50 | fail: dropped | None |
