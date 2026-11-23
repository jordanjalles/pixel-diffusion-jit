"""Experiment configs of the Colab JiT series (2026-11-07). Recipe = best known from the Spark/3090 runs:
100M (8 x 800, head 40), R32/R64 readouts, clamp01 weight, broad15 noise, self-flow 3 -> 6 (EMA teacher, weight 2),
XM transport with 3 seeds, EMA 0.999, class-conditional (10% label dropout). NO self-teaching judge (user 2026-11-07:
not worth it, caused the black/white contrast).

python make_cfg.py NAME [--res 128|256] [--batch B] [--lr X] [--steps N] [--smoke] [key=json ...]  -> configs/NAME.json
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def base(res=128, batch=64, lr=1e-4, steps=400000):
    return {
        "steps": steps, "log_every": 100, "eval_every": 2000, "preview_every": 10000, "ckpt_every": 10000,
        "compile": True, "yield_gpu": False, "loader_workers": 6, "disk_lock": False, "draft_decode": True,
        "keep_outputs": True, "host": "colab-a100", "ship": True, "ship_gap": 75,
        "hf_repo": f"benjamin-paine/imagenet-1k-{res}x{res}", "dataset": f"hf{res}", "dataset_dir": "/datasets/hf/data",
        "dataset_size": res,
        "hparams": {"SHARDS_IN_RAM": 1, "ALLOW_PARTIAL_IMAGENET": True, "IMAGE_SIZE": res, "CACHE_BATCH_SIZE": batch,
                    "pyramid_readouts": [[3, 4], [6, 8]], "pyramid_weight": 1.0,
                    "emb_dim": 800, "n_heads": 20, "head_size": 40, "num_blocks": 8, "lr": lr},
        "run_type": "pixel_pyramid", "pyramid": True, "schedule": "broad15", "loss_weight": "clamp01",
        "selfflow": {"from": 3, "to": 6, "weight": 2.0, "teacher": "ema"},
        "xm_ot": {"seeds": 3},
        "cond": {"classes": 1000, "drop": 0.1, "preview_cfg": 3.0},
        "ema": 0.999,
        "z_snapshot": {"block": 6, "n": 128},
        "guard": {"step": 4000, "max_test": 5.0},
        "gen_eval": {"every": 50000, "n": 10000, "steps": 16, "sampler": "euler", "weights": "ema", "cfg": 3.0},
        "ckpt_keep": {"last": 2, "every": 100000},
        "seed": 44,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=400000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("set", nargs="*", help="key=json, dotted keys go into dicts (hparams.lr=2e-4)")
    a = ap.parse_intermixed_args()
    c = base(a.res, a.batch, a.lr, a.steps)
    if a.smoke:
        c.update(log_every=50, eval_every=100, preview_every=200, ckpt_every=200, guard={"step": 100, "max_test": 5.0},
                 gen_eval={"every": 400, "n": 500, "steps": 4, "sampler": "euler", "weights": "ema", "cfg": 3.0},
                 z_snapshot={"block": 6, "n": 16}, ckpt_keep={"last": 2, "every": 10 ** 9})
    for kv in a.set:
        k, v = kv.split("=", 1)
        d = c
        *path, last = k.split(".")
        for p in path:
            d = d.setdefault(p, {})
        d[last] = json.loads(v)
    (HERE / "configs").mkdir(exist_ok=True)
    out = HERE / "configs" / f"{a.name}.json"
    out.write_text(json.dumps(c, indent=1) + "\n")
    print(out)
