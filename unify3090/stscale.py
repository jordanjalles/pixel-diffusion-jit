"""Self-teaching scaling sweep on the Spark (user request, 2026-10-21).

Hypothesis: self-teaching gives larger FID gains as the network grows (its own frozen copy is a stronger judge).
6 scales x 3 variants x 3 paired seeds = 54 runs of the pixel pyramid model, code job_stscale/ (= job_selfteach/ + cfg pair).

    python3 stscale.py plan                    the 54 runs, their configs and their state
    python3 stscale.py smoke [SCALE ...]       short paired triples (base, st3, st6) to check the code; folders stscale-smoke-*
    python3 stscale.py run                     the sweep, one run at a time (user service stscale.service); touch stscale/STOP to pause
    python3 stscale.py report                  stscale_report.py in the container -> <data-root>/outputs/stscale/

Fixed in every run: 8 blocks, head size 36 (heads = width / 36), ImageNet 128 px (Spark copy), batch 64, 100,000 steps,
lr 1e-4 then 1e-5 from 87,500, EMA 0.999, broad15 t with the clamp01 weight, pyramid readouts R32 after block 3 and R32
after block 6 (key R32b6). The usual R64 readout needs width >= 192 (the skip layout of a token holds 3 x k x k values with
k = isqrt(width / 3) and R64 needs k >= 8), so 1M / 2M / 5M could not have it: every scale gets the same two R32 readouts.
Variants: base (no self-teaching), st3 / st6 = selfteach judged at block 3 / 6 with the recipe of the latest self-teaching
runs: output term only (weight 1), crops 0.2-0.5 of the side, judge = a frozen copy synced at every step, normalized mse.
Pairing: within a scale and seed the three runs share the init (seed), the data order (DATA_SEED) and every random number of
the train step (cfg pair: own generator); base runs first and its pairing.jsonl is the reference of st3 / st6 (CHECK pairing).
Scores: EMA FID / KID / fd_ae at 25k / 50k / 75k / 100k, 10,000 samples of seeds 0..9999, 16 Euler steps, the same reference.

STSCALE_SERIES=sf (user, 2026-10-23, after the first sweep: self-teaching alone lost at every scale): the same sweep with
self-flow in every run (from 3 to 6, weight 1, EMA teacher, as the 200k run of 2026-10-10), so base = self-flow only and
st3 / st6 = self-flow + self-teaching. Names unify-wclamp01-sfscale-*, state sfscale/, report <data-root>/outputs/sfscale/,
train-viz experiment sfscale, seeds 0 and 1. Service sfscale.service.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path("<data-root>/outputs/training-runs")
SERIES = os.environ.get("STSCALE_SERIES", "st")
assert SERIES in ("st", "sf"), SERIES
PREFIX = {"st": "stscale", "sf": "sfscale"}[SERIES]
STATE_DIR = HERE / PREFIX
OUT = Path("<data-root>/outputs") / PREFIX
STEPS = 100000
SEEDS = (0, 1, 2) if SERIES == "st" else (0, 1)
SELFFLOW = {"from": 3, "to": 6, "weight": 1.0, "teacher": "ema"} if SERIES == "sf" else None
# target parameters -> width (head size 36). Counts are measured by the job (params.json): about 1.1 / 2.1 / 5.2 / 9.8 / 19.6 / 49.4 M
SCALES = {"1m": 72, "2m": 108, "5m": 180, "10m": 252, "20m": 360, "50m": 576}
VARIANTS = {"base": None, "st3": 3, "st6": 6}
SELFTEACH = {"weight_out": 1.0, "weight_noised": 0.0, "crop_min": 0.2, "crop_max": 0.5}
log = lambda *a: print(time.strftime("%m-%d %H:%M:%S"), *a, flush=True)


def run_name(scale, variant, seed, tag=""):
    if tag:
        return f"unify-wclamp01-{PREFIX}-{tag}-{scale}-{variant}-seed{seed}-spark"
    return f"unify-wclamp01-{PREFIX}-{scale}-{variant}-seed{seed}-{STEPS // 1000}k-spark"


def build_cfg(scale, variant, seed, steps=STEPS, smoke=False):
    width = SCALES[scale]
    base = json.loads((HERE / "scaleup.json").read_text())["config"]
    cfg = dict(base, run_type="pixel_pyramid", pyramid=True, keep_outputs=True, steps=steps, eval_every=2000, preview_every=10000,
               ckpt_every=25000, loader_workers=6, host="spark", yield_gpu=False,
               hparams={"SHARDS_IN_RAM": 1, "pyramid_readouts": [[3, 4], [6, 4]], "pyramid_weight": 1.0,
                        "emb_dim": width, "n_heads": width // 36, "head_size": 36, "num_blocks": 8, "lr": 1e-4},
               schedule="broad15", loss_weight="clamp01", lr_switch={"step": int(0.875 * steps), "lr": 1e-5},
               guard={"step": 4000, "max_test": 5.0}, ema=0.999, seed=seed,
               gen_eval={"every": 25000, "n": 10000, "steps": 16, "sampler": "euler", "weights": "ema"},
               pair={"every": 1000})
    if SELFFLOW:
        cfg["selfflow"] = dict(SELFFLOW)
    if VARIANTS[variant]:
        cfg["selfteach"] = dict(SELFTEACH, layer=VARIANTS[variant])
        cfg["pair"]["ref"] = "/ref/pairing.jsonl"
    if smoke:
        cfg.update(eval_every=max(100, steps // 2), preview_every=steps, ckpt_every=steps, guard={"step": 10 ** 9, "max_test": 5.0},
                   gen_eval={"every": steps, "n": 500, "steps": 16, "sampler": "euler", "weights": "ema"}, pair=dict(cfg["pair"], every=100))
        # no lr switch in a smoke run: its CHECK compares one-step weight changes, which are not steady after a few hundred
        # steps (the 50M st3 smoke at step 350: ratio 0.037 for a rate ratio 0.1, while the 1M runs passed)
        cfg.pop("lr_switch")
    return cfg


def description(scale, variant, seed, cfg):
    w, h = cfg["hparams"]["emb_dim"], cfg["hparams"]["n_heads"]
    st = cfg.get("selfteach")
    others = [run_name(scale, v, seed) for v in VARIANTS if v != variant]
    what = "baseline without self-teaching" if not st else f"self-teaching judged at block {st['layer']}"
    if SELFFLOW:
        what = "self-flow only (baseline)" if not st else f"self-flow + self-teaching judged at block {st['layer']}"
    return "\n".join([
        f"Self-teaching {'on self-flow ' if SELFFLOW else ''}scaling sweep: {what} at the {scale.upper()} scale, seed {seed}",
        f"compare: {others[0]} and {others[1]} (same init, data and noise; paired by cfg pair)",
        f"model: 8 blocks x width {w} ({h} heads of 36), readouts R32 after blocks 3 and 6 (R64 needs width >= 192); trainable count in params.json",
        f"train: broad15 t, clamp01 weight, lr 1e-4 then 1e-5 from {cfg.get('lr_switch', {}).get('step', 0):,}, {cfg['steps']:,} steps, batch 64, EMA 0.999"
        + (", self-flow 3 -> 6 weight 1 with an EMA teacher" if SELFFLOW else ""),
        ("self-teaching: none (st_eval_b3 / st_eval_b6 measure the judge distance without training it)" if not st else
         f"self-teaching: output crop pulled to the clean crop in z after block {st['layer']} of a frozen copy synced every step, weight 1, crops 0.2-0.5, no noised term"),
        "scores: EMA FID / KID / fd_ae at 25k / 50k / 75k / 100k, 10,000 samples of seeds 0..9999, 16 Euler steps"]) + "\n"


def docker(*args, check=True):
    r = subprocess.run(["docker", "--context", "rootless", *args], capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"docker {args[0]}: {r.stderr[-1500:]}")
    return r


def finished(out):
    """The status the sweep wrote when the run's container ended (None: never started, or still running)."""
    f = out / "stscale_status.json"
    return json.loads(f.read_text()) if f.exists() else None


def launch(name, cfg, ref=None, desc=""):
    """Start one container, wait for it to end, return its status dict (also written to the run folder)."""
    out = RUNS / name
    if (out / "metrics.jsonl").exists():
        raise RuntimeError(f"{out} already has a run: move it away first")
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiment-config.json").write_text(json.dumps(cfg, indent=1))
    if desc:
        (out / "DESCRIPTION.md").write_text(desc)
    container = "unify_" + name
    docker("rm", "-f", container, check=False)
    mounts = ["-v", f"{HERE / 'job_stscale'}:/code:ro", "-v", "<data-root>/data/imagenet/_staging:/datasets/imagenet/_source/data:ro",
              "-v", f"{out}:/output", "-v", "<data-root>/caches/gen_eval:/gen_eval", "-v", "<data-root>/outputs/imagenet-selfteach/scale:/scale:ro"]
    if ref is not None:
        mounts += ["-v", f"{ref}:/ref:ro"]
    t0 = time.time()
    docker("run", "-d", "--name", container, "--device", "nvidia.com/gpu=all", "--shm-size=16g", "--cap-drop", "ALL",
           "--security-opt", "no-new-privileges:true", "--pids-limit", "4096", "--network", "none", *mounts,
           "-e", "OUTPUT_DIR=/output", "-e", "EXPERIMENT_CONFIG=/output/experiment-config.json",
           "-e", "TZ=America/Los_Angeles", "-e", "PYTHONDONTWRITEBYTECODE=1",
           "nvcr.io/nvidia/pytorch:26.01-py3", "python", "/code/main.py")
    log(f"started {name}" + (f" (pair reference {ref.name})" if ref else ""))
    return wait(name, t0)


def wait(name, t0):
    """Wait for the run's container to end; write and return its status (stscale_status.json in the run folder)."""
    out, container = RUNS / name, "unify_" + name
    while docker("inspect", "-f", "{{.State.Running}}", container).stdout.strip() == "true":
        time.sleep(30)
    code = int(docker("inspect", "-f", "{{.State.ExitCode}}", container).stdout.strip())
    logs = docker("logs", container, check=False)
    text = logs.stdout + logs.stderr
    (out / "container.log").write_text(text)
    done = [l for l in text.splitlines() if l.startswith("DONE ")]
    status = {"exit": code, "done": done[-1] if done else None, "guard": bool(done and "guard" in done[-1]),
              "wall_seconds": time.time() - t0, "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
              "checks_failed": [l for l in text.splitlines() if "AssertionError" in l][-3:]}
    status["ok"] = code == 0 and bool(done) and not status["guard"]
    (out / "stscale_status.json").write_text(json.dumps(status, indent=1))
    docker("rm", container, check=False)
    log(f"ended {name}: {'ok' if status['ok'] else 'FAILED'} | exit {code} | {status['done']} | {status['wall_seconds'] / 3600:.2f} h")
    return status


def plan():
    """Seed-major, then scales small to large, then base / st3 / st6: one complete seed of the curve comes first."""
    return [(sc, v, sd) for sd in SEEDS for sc in SCALES for v in VARIANTS]


EXP_ADD = Path("<data-root>/workspaces/train-viz/scripts/experiment_add.py")


def exp_add(*args):
    """train-viz experiment registry: the whole sweep is one experiment, id stscale."""
    r = subprocess.run(["python3", str(EXP_ADD), *args], capture_output=True, text=True)
    if r.returncode:
        log(f"experiment_add failed (the sweep goes on): {r.stderr.strip()[-300:]}")


def register(sc, v, sd):
    # the UI compares against one baseline run: 1m base seed 0; the other base runs are controls (paired gains: report.md)
    first = (sc, v, sd) == (next(iter(SCALES)), "base", SEEDS[0])
    role = "treatment" if VARIANTS[v] else "baseline" if first else "control"
    exp_add("--id", PREFIX, "--run", f"shared/{run_name(sc, v, sd)}", "--label", f"{sc} {v} seed {sd}", "--role", role)


def cmd_register():
    """Create the experiment with every run of the plan: started runs as runs, the others as planned runs."""
    if SELFFLOW:
        title, question = "Self-teaching on self-flow, scaling sweep (1M to 50M)", (
            "With self-flow (3 -> 6, EMA teacher) in every run, does adding self-teaching (judge at block 3 or 6) lower the EMA FID "
            f"at 100k against the paired self-flow-only run, and does that gain grow from 1M to 50M? Paired gains per scale: {OUT}/report.md")
    else:
        title, question = "Self-teaching scaling sweep (1M to 50M)", (
            "Does the EMA FID gain of self-teaching (judge at block 3 or 6) over the paired baseline grow with the "
            f"trainable parameter count, 1M to 50M? Paired gains per scale: {OUT}/report.md")
    exp_add("--id", PREFIX, "--title", title, "--status", "running", "--question", question,
            *[x for sc, v, sd in plan() for x in ("--planned", run_name(sc, v, sd))])
    for sc, v, sd in plan():
        if (RUNS / run_name(sc, v, sd)).exists():
            register(sc, v, sd)


def cmd_plan():
    for sc, v, sd in plan():
        name = run_name(sc, v, sd)
        st = finished(RUNS / name)
        print(f"{name:55s} width {SCALES[sc]:4d} heads {SCALES[sc] // 36:2d} | "
              + ("not started" if st is None and not (RUNS / name).exists() else "running or broken" if st is None else ("ok" if st["ok"] else "FAILED")))


def cmd_smoke(scales, steps=300):
    """Paired triples of short runs (pairing CHECKs, treatment CHECKs, steady ms per step, a tiny FID). Names stscale-smoke-*."""
    stamp = time.strftime("%m%d%H%M")
    for sc in scales:
        ref = None
        for v in VARIANTS:
            name = run_name(sc, v, 0, tag=f"smoke{stamp}")
            status = launch(name, build_cfg(sc, v, 0, steps=steps, smoke=True), ref=ref, desc=f"Smoke test of the {PREFIX} sweep ({sc}, {v}, {steps} steps): not a result\n")
            if not status["ok"]:
                sys.exit(f"smoke {name} failed: see {RUNS / name / 'container.log'}")
            if v == "base":
                ref = RUNS / name


def cmd_run():
    STATE_DIR.mkdir(exist_ok=True)
    for sc, v, sd in plan():
        if (STATE_DIR / "STOP").exists():
            log(f"STOP file found: the sweep pauses (remove {PREFIX}/STOP and restart the service to go on)")
            return
        name = run_name(sc, v, sd)
        st = finished(RUNS / name)
        if st is not None and st["ok"]:
            continue
        if st is None and docker("inspect", "-f", "{{.State.Running}}", "unify_" + name, check=False).stdout.strip() == "true":
            log(f"{name} is still running (the sweep was restarted): waiting for it")
            st = wait(name, time.time())
            if not st["ok"] and not st["guard"]:
                sys.exit(f"{name} failed: the sweep stops. Log: {RUNS / name / 'container.log'}")
            cmd_report()
            continue
        if st is not None or (RUNS / name / "metrics.jsonl").exists():
            # an interrupted or failed run: a restart would not be paired (the loader and the drawn numbers restart). Stop.
            sys.exit(f"{name} exists but did not end ok: inspect it, move the folder away, then restart the sweep")
        ref = None
        if VARIANTS[v]:
            ref = RUNS / run_name(sc, "base", sd)
            bst = finished(ref)
            assert bst is not None and bst["ok"], f"the baseline {ref.name} has not ended ok"
        register(sc, v, sd)
        status = launch(name, build_cfg(sc, v, sd), ref=ref, desc=description(sc, v, sd, build_cfg(sc, v, sd)))
        if not status["ok"] and not status["guard"]:
            sys.exit(f"{name} failed (a CHECK or a crash): the sweep stops. Log: {RUNS / name / 'container.log'}")
        cmd_report()
    log("sweep complete")


def cmd_report():
    OUT.mkdir(parents=True, exist_ok=True)
    r = docker("run", "--rm", "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true", "--network", "none", "--cpus", "2",
               "-v", f"{HERE / 'stscale_report.py'}:/report.py:ro", "-v", f"{RUNS}:/runs:ro", "-v", f"{OUT}:/out",
               "-e", "MPLCONFIGDIR=/tmp", "-e", f"STSCALE_SERIES={SERIES}", "nvcr.io/nvidia/pytorch:26.01-py3", "python", "/report.py", check=False)
    print(r.stdout[-3000:], r.stderr[-1500:], flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if what == "plan":
        cmd_plan()
    elif what == "smoke":
        args = sys.argv[2:]
        n_steps = int(args.pop(0)) if args and args[0].isdigit() else 300
        cmd_smoke(args or ["1m"], steps=n_steps)
    elif what == "run":
        cmd_run()
    elif what == "report":
        cmd_report()
    elif what == "register":
        cmd_register()
    else:
        sys.exit(__doc__)
