# FID research loop, round 3

Start: 2026-10-10 16:14. Parent: `unify-wclamp01-selfflow3to6-selfteach-200k-spark` (FID 114.19 from the trained weights, 114.54 from the averaged weights, at step 200,000).
Every run starts from that checkpoint with its own recipe. Each run gets the same compute: 15,000 steps at step cost 1, fewer steps for a costlier step. FID from the averaged weights. Lower is better.

| time | run | change | FID | reference | margin | verdict | next |
|---|---|---|---|---|---|---|---|
