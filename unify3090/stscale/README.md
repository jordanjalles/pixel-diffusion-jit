# Self-teaching scaling sweep (built 2026-10-21)

Hypothesis: self-teaching gives larger FID gains as the network grows, because its own frozen copy is a stronger judge.
This sweep scales learner and judge together. It cannot show that the stronger judge causes a gain.

## Files

- `../stscale.py`: the driver (plan, smoke, run, report). Its docstring lists every fixed setting.
- `../job_stscale/`: the job code, a copy of `job_selfteach/` plus cfg `pair` (search for `cfg["pair"]` in `main.py`).
- `../stscale_report.py`: the report. It runs in the PyTorch container and writes `<data-root>/outputs/stscale/`.
- `../stscale.service`: the user unit for the sweep. It is not linked or started.
- `queue.log`: the driver's log when it runs as the service. `STOP`: when this file exists, the driver stops before the next run.

## Design

Measured in the smoke runs of 2026-10-21 (trainable parameters from params.json; steps per second in the clean
window that ends at step 400, Spark GPU alone):

| scale | width | heads (size 36) | trainable parameters | steps/s base | st3 | st6 |
|---|---|---|---|---|---|---|
| 1m | 72 | 2 | 901,272 | 46 | 32 | 22 |
| 2m | 108 | 3 | 1,914,948 | 30 | 37 | 37 |
| 5m | 180 | 5 | 5,069,820 | 32 | 23 | 25 |
| 10m | 252 | 7 | 9,728,052 | 25 | 25 | 22 |
| 20m | 360 | 10 | 19,534,200 | 29 | 17 | 15 |
| 50m | 576 | 16 | 49,294,176 | 17 | 11 | 8.9 |

From 2m to 10m the step time hardly depends on the variant: the loader keeps up (data wait under 1 ms), so a fixed
per-step overhead sets the floor there. Training time of the whole sweep is about 70 GPU hours, plus about 15 hours
for 216 FID scores of 10,000 samples. At width 72 and 108 the eval and sampling forward falls back to eager
(Triton "PassManager::run failed"); the train step compiles at every width.

- 8 blocks at every scale. Readouts: R32 after block 3 and R32 after block 6 (key `R32b6`) at every scale.
  The usual R64 readout needs width 192 or more, so the three small scales cannot have it.
- Variants: `base` (no self-teaching), `st3` and `st6` (judge at block 3 or 6).
- Self-teaching recipe of the latest runs: output term only (weight 1), crops 0.2 to 0.5 of the side, overhang 0.5,
  the same crop of output and clean image, judge = a frozen copy synced from the trained weights at every step,
  target detached, gradient through the judge into the output, normalized mse (`teach_distance`).
- Fixed: ImageNet 128 px (Spark copy), batch 64, 100,000 steps, AdamW lr 1e-4 then 1e-5 from 87,500, EMA 0.999,
  broad15 t, clamp01 weight, readout weight 1.
- Seeds 0, 1, 2. Within a scale and seed, the three variants share the init, the data order and every random number
  of the train step (t, noise, rope scale, crops). See "Pairing" below.
- Scores: EMA FID, KID and fd_ae at 25k, 50k, 75k and 100k. 10,000 samples of seeds 0..9999, 16 Euler steps,
  the same cached real reference. The primary result is the FID at 100k. The best checkpoint is never used.
- Cost: `train_seconds` (training only) and the wall time of each run (`stscale_status.json`) are reported apart.

## Pairing (cfg pair)

- The job draws every random number of the train step before the step, from its own generator (seed 7300 + seed).
  The compiled step then has no random operation. A variant with extra passes cannot move the noise stream.
- The rope scale of the model pass is read from a tensor that is filled before each step. The judge passes get
  their own drawn scale. CHECK pair rope compares the patched function with the notebook's.
- `pairing.jsonl` in each run folder holds fingerprints of the init, the batches and the drawn numbers.
  A variant reads its baseline's file at `/ref` and stops at the first fingerprint that differs (CHECK pairing).
- GPU kernels are not bit-exact between runs. Two baseline smoke runs differed in the 5th digit of a loss.
  Pairing covers the inputs, not the arithmetic.
- A resumed run is not paired after its resume step: the loader and the generator restart. The driver does not
  resume runs. It stops if a run folder exists without a good end.

## Checks that stop a run

- CHECK pair config: the variant's config equals its baseline's, except `selfteach` and `pair`.
- CHECK pairing at steps 0, 1, 2 and every 1,000 steps: init, images, t, noise, crops and rope scales are equal.
- CHECK self-teaching gradient at step 0: the judge gets no gradient, every model block gets gradient through the
  judge, and an output equal to the clean image gives a term of 0 (the same crop of both).
- CHECK pairing treatment at step 1: the weights after the first update differ from the baseline's.
- CHECK self-teaching at step 100: the term is in the loss, and the judge equals the trained weights after the sync.

## What train-viz shows

- `st_out`: the self-teaching term in the loss (st3 and st6 runs).
- `st_eval_b3`, `st_eval_b6`: the same distance measured on fixed test images in every run, the baseline too.
