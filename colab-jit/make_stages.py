"""Main-run configs from the tuning results. python3 make_stages.py --tune RUN_DIR --shape NAME --units-a U --units-b U
       [--rate UNITS_PER_H] [--batch-b B] [--dry]
Stage A: 128 px from scratch, tuned batch / lr / shape, constant lr. cfg "next" = stageB: the watchdog starts B from A's
final checkpoint. Stage B: 256 px, same lr until 85% of its steps, then 1e-5 (lr_switch).
Steps = units / rate * 3600 s * (1 - overhead) / (ms per step). overhead covers compile, data download, evals, previews,
FID and the shipping at each job end (one job per ~11.6 h VM).
"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_cfg import base  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tune", required=True)
ap.add_argument("--shape", required=True, help="a run name of the shape phase, e.g. 8x800")
ap.add_argument("--units-a", type=float, required=True)
ap.add_argument("--units-b", type=float, required=True)
ap.add_argument("--rate", type=float, default=6.73, help="A100 units per hour (measured 2026-11-07)")
ap.add_argument("--batch-b", type=int, default=0, help="256 px batch (default: the stage-A batch if it fit in the res256 phase)")
ap.add_argument("--overhead", type=float, default=0.12)
ap.add_argument("--dry", action="store_true")
a = ap.parse_args()

summ = json.loads((Path(a.tune) / "tune_summary.json").read_text())
ph = summ["phases"]
picked = {}
for name in ("batch", "lr"):
    best = summ["picked"][name]["best"]
    picked.update(next(r for r in ph[name] if r["run"] == best)["set"])
shape_run = next(r for r in ph["shape"] if r["run"] == a.shape)
shape = {k: v for k, v in shape_run["set"].items() if k.startswith("hparams.") and k.split(".")[1] in
         ("num_blocks", "emb_dim", "n_heads", "head_size")}
batch, lr = int(picked["hparams.CACHE_BATCH_SIZE"]), float(picked["hparams.lr"])
ms_a = shape_run["ms_step"]
r256 = {r["set"]["hparams.CACHE_BATCH_SIZE"]: r for r in ph["res256"] if not r["oom"] and r["ms_step"]}
batch_b = a.batch_b or (batch if batch in r256 else max(r256))
# res256 ran with the stage-A shape of the tuning job (8x800); scale its step time by the shape's 128 px ratio
ms_ref128 = next(r for r in ph["shape"] if r["run"] == "8x800")["ms_step"]
ms_b = r256[batch_b]["ms_step"] * (ms_a / ms_ref128)


def steps_for(units, ms):
    s = units / a.rate * 3600 * (1 - a.overhead) * 1000 / ms
    return int(s // 10000 * 10000)


steps_a, steps_b = steps_for(a.units_a, ms_a), steps_for(a.units_b, ms_b)
ck = lambda ms: max(1000, int(1800 * 1000 / ms // 1000 * 1000))       # noqa: E731  a checkpoint about every 30 min


def setk(c, k, v):
    *path, last = k.split(".")
    for p in path:
        c = c.setdefault(p, {})
    c[last] = v


A = base(128, batch, lr, steps_a)
for k, v in shape.items():
    setk(A, k, v)
A.update(ckpt_every=ck(ms_a), wall_s=39600, next="stageB", next_units=a.units_b,
         gen_eval=dict(A["gen_eval"], every=max(50000, steps_a // 8 // 10000 * 10000)),
         preview_every=max(10000, steps_a // 40 // 10000 * 10000), ckpt_keep={"last": 2, "every": steps_a})
B = base(256, batch_b, lr, steps_a + steps_b)
for k, v in shape.items():
    setk(B, k, v)
B.update(ckpt_every=ck(ms_b), wall_s=39600, lr_switch={"step": steps_a + int(0.85 * steps_b) // 1000 * 1000, "lr": 1e-5},
         gen_eval=dict(B["gen_eval"], every=max(20000, steps_b // 6 // 10000 * 10000)),
         preview_every=max(10000, steps_b // 40 // 10000 * 10000), z_snapshot={"block": 6, "n": 64},
         ckpt_keep={"last": 2, "every": 10 ** 12})
for c, steps in ((A, steps_a), (B, steps_a + steps_b)):
    assert steps % c["log_every"] == 0 and c["ckpt_every"] % c["log_every"] == 0 and c["preview_every"] % c["log_every"] == 0

table = {
    "shape": a.shape, "batch A": batch, "lr": lr, "ms/step A": round(ms_a), "steps A": steps_a,
    "images A (M)": round(steps_a * batch / 1e6, 1), "epochs A": round(steps_a * batch / 1281167, 1),
    "batch B": batch_b, "ms/step B": round(ms_b), "steps B": steps_b, "images B (M)": round(steps_b * batch_b / 1e6, 1),
    "epochs B": round(steps_b * batch_b / 1281167, 1), "lr 1e-5 from step": B["lr_switch"]["step"],
    "hours A": round(a.units_a / a.rate, 1), "hours B": round(a.units_b / a.rate, 1),
    "jobs A": math.ceil(a.units_a / a.rate / 11.0), "jobs B": math.ceil(a.units_b / a.rate / 11.0),
    "ckpt every A/B (steps)": f"{A['ckpt_every']}/{B['ckpt_every']}",
}
print(json.dumps(table, indent=1))
if not a.dry:
    (HERE / "configs" / "stageA.json").write_text(json.dumps(A, indent=1) + "\n")
    (HERE / "configs" / "stageB.json").write_text(json.dumps(B, indent=1) + "\n")
    (HERE / "configs" / "stages_table.json").write_text(json.dumps(table, indent=1) + "\n")
    print("wrote configs/stageA.json, configs/stageB.json")
