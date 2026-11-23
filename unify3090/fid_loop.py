"""FID research loop on the Spark: short experiments from the best checkpoint, chosen by a fixed decision tree.

    python3 fid_loop.py            (host python; user service fid-loop; `touch fidloop/STOP` ends it after the current run)

User, 2026-10-09: take the model with the lowest FID and push the FID down in 15-minute experiments. Every experiment
names its two followers before it runs: the next change if it fails, the change to put on top if it succeeds.

One experiment = one container. It continues the parent checkpoint for CYCLE steps (learning rate 1e-4 for HOT steps,
then 1e-5) with one change of the job config, and scores FID from the averaged weights (10,000 samples, 16 Euler steps).
User, 2026-10-09 22:45: every experiment starts from the same baseline checkpoint on the Spark (never from a random init,
never from another experiment). The control runs once: the same cycle without a change. An experiment PASSES if its FID is
lower than the reference by more than MIN_MARGIN. A change that passes stays in the config of the later experiments, and its
FID is the new reference. A change that fails is dropped. No extra seed runs.

Two loops run at the same time, one per GPU, each with its own plan and its own control:
    python3 fid_loop.py                    Spark, folder fidloop/ (service fid-loop)
    python3 fid_loop.py --backend 3090     3090 runner, folder fidloop3090/ (service fid-loop-3090)
A 3090 experiment reads the same baseline checkpoint from the runner dataset "baselines" (/datasets/baselines, copied there
by the user's PC agent on 2026-10-09), trains on the imagenet128 copy, and is scored on the Spark from its collected checkpoint.
When a plan has no next node, the loop takes the first node of plan.json that has not run, and waits for new nodes if none is left.

Files in fidloop/: plan.json (the tree; nodes can be added while the loop runs), state.json, LOG.md (the journal).
Runs: <data-root>/outputs/training-runs/fidloop-g<generation>-<node>.
"""
import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_p = argparse.ArgumentParser()
_p.add_argument("--backend", default="spark", choices=("spark", "3090"))
# Round 2 (user, 2026-10-10: self-teaching in scope, 20-minute cycles, equal compute): --work names another folder with its
# own plan.json / state.json / LOG.md, --tag the prefix of the run names. plan.json then also holds
#   "base_change": the config change of every run of the round (the kept winners of round 1)
#   "base_steps":  steps of a run at step cost 1 (the control), chosen so that one cycle takes 20 minutes
#   node "cost":   step time of the node's config relative to the control. The node trains base_steps / cost steps,
#                  so every node gets the same compute. The learning rate drops to 1e-5 after 62.5% of the steps.
_p.add_argument("--work", default="")
_p.add_argument("--tag", default="")
ARGS, _ = _p.parse_known_args()
BACKEND = ARGS.backend
WORK = HERE / (ARGS.work or ("fidloop" if BACKEND == "spark" else "fidloop3090"))
RUNS3090, RAW3090 = Path("<data-root>/outputs/3090-experiments/runs"), Path("<data-root>/outputs/3090-experiments/raw")
CLIENT = "<data-root>/outputs/3090-experiments/runner-updates/resume-2026-10-07/spark_client.py"
TAG = ARGS.tag or ("fidloop" if BACKEND == "spark" else "fidloop90")
# the baseline checkpoints on the 3090 (runner dataset "baselines"), by their run on the Spark
BASELINE_FILES = {"unify-wclamp01-selfflow3to6-selfteach-200k-spark": "selfflow3to6_selfteach_seed44_step200000.pt",
                  "unify-wclamp01-selfflow3to6-100k-spark": "selfflow3to6_seed44_step100000.pt",
                  "unify-wclamp01-control-100k-spark": "control_seed44_step100000.pt"}
RUNS = Path("<data-root>/outputs/training-runs")
PLAN, STATE, LOG = WORK / "plan.json", WORK / "state.json", WORK / "LOG.md"
CYCLE, HOT = 16000, 10000              # steps per experiment; steps at the learning rate 1e-4 before 1e-5
MIN_MARGIN = 1.5                       # FID. Fixed: no noise measurement with a second seed (user: no extra seeds)
TIMEOUT_MIN = 45
DOCKER = ["docker", "--context", "rootless"]


def log(*a):
    print(time.strftime("%m-%d %H:%M"), *a, flush=True)


def journal(line):
    with open(LOG, "a") as f:
        f.write(line + "\n")


def containers():
    return subprocess.run(DOCKER + ["ps", "--format", "{{.Names}}"], capture_output=True, text=True).stdout.split()


def records(run):
    f = RUNS / run / "metrics.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []


def merge(base, change):
    out = copy.deepcopy(base)
    for k, v in change.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        elif v is None:
            out.pop(k, None)
        else:
            out[k] = copy.deepcopy(v)
    return out


def pick_parent():
    """The finished 100,000-step run of the retest on the Spark with the lowest FID at its last step."""
    best = None
    # the two methods of the seed test (user: the baseline is the checkpoint on the Spark of the method that wins it)
    for d in (RUNS / "unify-wclamp01-control-100k-spark", RUNS / "unify-wclamp01-selfflow3to6-100k-spark"):
        cfg_file = d / "experiment-config.json"
        if not cfg_file.exists():
            continue
        cfg = json.loads(cfg_file.read_text())
        ckpt = d / f"checkpoint_final_step_{cfg['steps']:06d}.pt"
        last = [r for r in records(d.name) if "fid" in r and r["step"] == cfg["steps"]]
        if cfg.get("hparams", {}).get("num_blocks") != 8 or not ckpt.exists() or not last:
            continue
        log(f"candidate {d.name}: FID {last[-1]['fid']:.1f}")
        if best is None or last[-1]["fid"] < best[1]:
            best = (d.name, last[-1]["fid"], cfg["steps"])
    assert best, "no finished run with a final checkpoint and a FID"
    return best


def client(*args):
    try:
        r = subprocess.run([sys.executable, CLIENT, *args], capture_output=True, text=True, timeout=600)
        return json.loads(r.stdout)
    except Exception as e:
        return {"error": repr(e)[:300]}


def run_3090(name, parent, cfg_change, seed, test, compare):
    """One cycle on the 3090 from the baseline checkpoint, scored on the Spark. Returns the record with fid_ema, or None."""
    base = json.loads((RUNS / parent["run"] / "experiment-config.json").read_text())
    for key in ("guard", "wait_for_ckpt", "gen_eval", "keep_outputs", "resume_weights_only"):
        base.pop(key, None)
    cfg = merge(base, cfg_change)
    if "selfflow" in cfg and "from" not in cfg["selfflow"]:
        cfg["selfflow"] = merge({"from": 3, "to": 6, "weight": 1.0}, cfg["selfflow"])
    steps = parent["step"] + CYCLE
    ckpt_in = "/datasets/baselines/" + BASELINE_FILES[parent["run"]]
    cfg.update(steps=steps, resume_ckpt=ckpt_in, resume_lr=1e-4, lr_switch={"step": parent["step"] + HOT, "lr": 1e-5}, seed=seed,
               ckpt_every=10**9, preview_every=10**9, eval_every=4000, ema=0.999, dataset="imagenet128", host="3090-wsl",
               yield_gpu=True, loader_workers=4, fidloop={"parent": parent["run"], "change": cfg_change})
    cfg_file = WORK / f"config_{name}.json"
    cfg_file.write_text(json.dumps(cfg, indent=1))
    job = client("submit", str(HERE / "job"), "--name", name, "--entrypoint", "main.py", "--config", str(cfg_file),
                 "--dataset", "imagenet128", "--dataset", "baselines")
    if "id" not in job:
        log("submit failed:", job)
        return None
    run = f"{name}--{job['id']}"
    t0, status = time.time(), "queued"
    while time.time() - t0 < 6 * 3600:
        status = client("status", job["id"]).get("status", status)
        d = RUNS3090 / run
        if d.is_dir() and not (d / "DESCRIPTION.md").exists():
            h, sf = cfg["hparams"], cfg.get("selfflow")
            (d / "DESCRIPTION.md").write_text("\n".join([
                test, "compare: " + compare,
                f"model: {h['num_blocks']} blocks x width {h['emb_dim']} ({h['n_heads']} heads), 19.6M parameters, continued from {parent['run']} at step {parent['step']:,}",
                f"train: {CYCLE:,} steps, lr 1e-4 for {HOT:,} steps then 1e-5, batch 64, EMA 0.999. FID from the averaged weights, scored on the Spark",
                f"weight: {cfg.get('loss_weight')}",
                "teaching: " + (json.dumps(sf) if sf else "none") + (" | self-teaching: " + json.dumps(cfg["selfteach"]) if cfg.get("selfteach") else ""),
                "data: imagenet128"]) + "\n")
        if status in ("done", "failed", "cancelled", "interrupted"):
            break
        time.sleep(30)
    text = client("logs", job["id"]).get("text", "")
    for line in text.splitlines():
        if line.startswith(("CHECK", "RESUME")):
            log("   ", line[:260])
    if status != "done" or f"RESUME from {ckpt_in} at step {parent['step']:,}" not in text:
        log(f"{name}: job {job['id']} ended as {status} or did NOT continue from the baseline checkpoint. Last log lines:", text[-400:].replace("\n", " | "))
        return None
    ckpt = RAW3090 / job["id"] / "published" / f"checkpoint_final_step_{steps:06d}.pt"
    t1 = time.time()
    while not Path(str(ckpt) + ".verified.json").exists() and time.time() - t1 < 3600:
        time.sleep(20)
    if not Path(str(ckpt) + ".verified.json").exists():
        log(f"{name}: the final checkpoint did not arrive on the Spark")
        return None
    r = subprocess.run([sys.executable, str(HERE / "gen_eval_spark.py"), str(RUNS3090 / run), "--ckpt-dir", str(ckpt.parent), "--steps", str(steps),
                        "--weights", "ema", "--wait"], capture_output=True, text=True)
    log("   ", (r.stdout + r.stderr).strip()[-300:].replace("\n", " | "))
    f = RUNS3090 / run / "gen_metrics.jsonl"
    scored = [r_ for r_ in map(json.loads, f.read_text().splitlines()) if "fid_ema" in r_ and r_["step"] == steps] if f.exists() else []
    if not scored:
        log(f"{name}: no FID from the Spark")
        return None
    return dict(scored[-1], minutes=(time.time() - t0) / 60, run=run)


def run_experiment(name, parent, cfg_change, seed, test, compare):
    """Train one cycle from the parent's final checkpoint. Returns the last record (with fid_ema) or None."""
    if BACKEND == "3090":
        return run_3090(name, parent, cfg_change, seed, test, compare)
    out = RUNS / name
    out.mkdir(exist_ok=True)
    base = json.loads((RUNS / parent["run"] / "experiment-config.json").read_text())
    for key in ("guard", "resume_weights_only", "wait_for_ckpt"):
        base.pop(key, None)
    cfg = merge(base, cfg_change)
    if "selfflow" in cfg and "from" not in cfg["selfflow"]:
        cfg["selfflow"] = merge({"from": 3, "to": 6, "weight": 1.0}, cfg["selfflow"])
    steps = parent["step"] + CYCLE
    cfg.update(steps=steps, resume_ckpt=f"/parent/checkpoint_final_step_{parent['step']:06d}.pt", resume_lr=1e-4,
               lr_switch={"step": parent["step"] + HOT, "lr": 1e-5}, seed=seed, ckpt_every=10**9, preview_every=10**9, eval_every=4000,
               ema=0.999, gen_eval={"every": 10**9, "n": 10000, "steps": 16, "sampler": "euler", "weights": "ema"},
               fidloop={"parent": parent["run"], "change": cfg_change})
    (out / "experiment-config.json").write_text(json.dumps(cfg, indent=1))
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    h = cfg["hparams"]
    sf = cfg.get("selfflow")
    (out / "DESCRIPTION.md").write_text("\n".join([
        test, "compare: " + compare,
        f"model: {h['num_blocks']} blocks x width {h['emb_dim']} ({h['n_heads']} heads), 19.6M parameters, continued from {parent['run']} at step {parent['step']:,}",
        f"train: {CYCLE:,} steps, lr 1e-4 for {HOT:,} steps then 1e-5, batch 64, EMA 0.999. FID from the averaged weights at the end",
        f"weight: {cfg.get('loss_weight')}",
        "teaching: " + (json.dumps(sf) if sf else "none") + (" | self-teaching: " + json.dumps(cfg["selfteach"]) if cfg.get("selfteach") else "")]) + "\n")
    container = "unify_" + name
    subprocess.run(DOCKER + ["rm", "-f", container], capture_output=True)
    cmd = DOCKER + ["run", "-d", "--name", container, "--device", "nvidia.com/gpu=all", "--shm-size=16g", "--cap-drop", "ALL",
                    "--security-opt", "no-new-privileges:true", "--pids-limit", "4096", "--network", "none",
                    "-v", f"{HERE / 'job'}:/code:ro", "-v", "<data-root>/data/imagenet/_staging:/datasets/imagenet/_source/data:ro",
                    "-v", f"{out}:/output", "-v", f"{RUNS / parent['run']}:/parent:ro",
                    "-e", "OUTPUT_DIR=/output", "-e", "EXPERIMENT_CONFIG=/output/experiment-config.json",
                    "-v", "<data-root>/caches/gen_eval:/gen_eval", "-v", "<data-root>/outputs/imagenet-selfteach/scale:/scale:ro",
                    "-e", "TZ=America/Los_Angeles", "-e", "PYTHONDONTWRITEBYTECODE=1",
                    "nvcr.io/nvidia/pytorch:26.01-py3", "python", "/code/main.py"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        log("docker run failed:", r.stderr[-300:])
        return None
    t0 = time.time()
    while container in containers() and time.time() - t0 < TIMEOUT_MIN * 60:
        tail = subprocess.run(DOCKER + ["logs", "--tail", "5", container], capture_output=True, text=True)
        if "DONE " in tail.stdout + tail.stderr:
            break
        time.sleep(20)
    full = subprocess.run(DOCKER + ["logs", container], capture_output=True, text=True)
    text = full.stdout + full.stderr
    (out / "log.txt").write_text(text)
    for line in text.splitlines():
        if line.startswith(("CHECK", "RESUME", "GEN step")):
            log("   ", line[:260])
    subprocess.run(DOCKER + ["stop", "-t", "5", container], capture_output=True)
    last = [r_ for r_ in records(name) if "fid_ema" in r_ and r_["step"] == steps]
    if f"RESUME from /parent/checkpoint_final_step_{parent['step']:06d}.pt at step {parent['step']:,}" not in text:
        log(f"{name}: the run did NOT continue from the baseline checkpoint: its result is not used")
        return None
    if not last or not (out / f"checkpoint_final_step_{steps:06d}.pt").exists():
        log(f"{name}: no FID or no final checkpoint ({(time.time() - t0) / 60:.0f} min). Last log lines:", text[-400:].replace("\n", " | "))
        return None
    rec = dict(last[-1], minutes=(time.time() - t0) / 60)
    return rec


def set_cycle(plan, cost=1.0):
    """Steps of the next run from the plan's compute budget. Without "base_steps" in the plan: the fixed cycle of round 1."""
    global CYCLE, HOT
    if plan.get("base_steps"):
        CYCLE = max(500, int(round(plan["base_steps"] / float(cost) / 500.0)) * 500)
        HOT = int(round(CYCLE * 0.625 / 500.0)) * 500
    return CYCLE


def save(state):
    STATE.write_text(json.dumps(state, indent=1))


def main():
    WORK.mkdir(exist_ok=True)
    # the Spark queue of full runs goes first. The 3090 loop waits for the dataset "baselines" and an empty runner queue.
    # The runner worker reads its config only at a restart, and the user's PC agent restarts it once the runner queue has
    # been empty at one of its checks (every 10 minutes). So: start only after 15 minutes with an empty queue.
    idle_since = None
    while BACKEND == "3090":
        ready = "baselines" in client("info").get("datasets", [])
        jobs = client("list")
        busy = [j for j in jobs if isinstance(j, dict) and j.get("status") in ("queued", "running")] if isinstance(jobs, list) else ["?"]
        idle_since = (idle_since or time.time()) if (ready and not busy) else None
        if (idle_since and time.time() - idle_since > 15 * 60) or STATE.exists():
            break
        if (WORK / "STOP").exists():
            return
        time.sleep(120)
    while BACKEND == "spark":
        done = (HERE / "queue.done").read_text().splitlines()
        todo = [l.strip() for l in (HERE / "queue.txt").read_text().splitlines() if l.strip() and not l.startswith("#") and l.strip() not in done]
        busy = [c for c in containers() if c.startswith(("unify_", "gen_"))]
        if not todo and not busy:
            break
        if (WORK / "STOP").exists():
            return
        time.sleep(120)
    state = json.loads(STATE.read_text()) if STATE.exists() else None
    if state is None:
        if BACKEND == "3090" and (HERE / "fidloop" / "state.json").exists():
            # the same baseline as the Spark loop
            first = json.loads((HERE / "fidloop" / "state.json").read_text())["parent"]
            run, fid, step = first["run"], float("nan"), first["step"]
        else:
            run, fid, step = pick_parent()
        plan0 = json.loads(PLAN.read_text())
        state = {"parent": {"run": run, "step": step}, "generation": 0, "base_change": plan0.get("base_change", {}), "ref_fid": None,
                 "margin": MIN_MARGIN, "node": plan0.get("start"), "history": []}
        journal(f"# FID research loop\n\nStart: {time.strftime('%Y-%m-%d %H:%M')}. First parent: `{run}` (FID {fid:.1f} from the trained weights at step {step:,}).\n"
                + (f"Every run starts from that checkpoint with the config change {json.dumps(plan0['base_change'])}.\n" if plan0.get("base_change") else "")
                + (f"Each run gets the same compute: {plan0['base_steps']:,} steps at step cost 1, fewer steps for a costlier step. " if plan0.get("base_steps")
                   else f"Each run: {CYCLE:,} steps from the parent. ")
                + "FID from the averaged weights. Lower is better.\n\n"
                "| time | run | change | FID | reference | margin | verdict | next |\n|---|---|---|---|---|---|---|---|")
        save(state)
    while not (WORK / "STOP").exists():
        gen, parent = state["generation"], state["parent"]
        seed = 100                       # one seed for every experiment: same data order and noise as the control
        if state["ref_fid"] is None:
            name = f"{TAG}-g{gen}-control"
            set_cycle(json.loads(PLAN.read_text()))
            log("control of generation", gen, "from", parent["run"])
            rec = run_experiment(name, parent, state["base_change"], seed, f"Control of generation {gen} of the FID loop: one more cycle of the parent with no change",
                                 f"the experiments named {TAG}-g*")
            if rec is None and BACKEND == "3090" and state.get("control_tries", 0) < 8:
                # most likely the runner worker does not know the dataset "baselines" yet: leave the queue empty and try again
                state["control_tries"] = state.get("control_tries", 0) + 1
                save(state)
                log("the control did not run: next try in 20 minutes, try", state["control_tries"])
                time.sleep(20 * 60)
                continue
            if rec is None:
                journal(f"| {time.strftime('%H:%M')} | {name} | none | FAILED TO RUN | | | loop stopped | |")
                log("the control did not run: stop")
                return
            state["ref_fid"] = rec["fid_ema"]
            state["history"].append({"run": name, "fid_ema": rec["fid_ema"], "kind": "control", "minutes": rec["minutes"]})
            journal(f"| {time.strftime('%H:%M')} | {name} | none (control) | {rec['fid_ema']:.2f} | | | reference | |")
            save(state)
            continue
        plan = json.loads(PLAN.read_text())
        node_id = state["node"]
        if node_id in {h_.get("node") for h_ in state["history"]}:
            node_id = None                  # a node runs once
        if not node_id or node_id not in plan["nodes"]:
            # no follower named: the first node of the plan that has not run. None left: wait for new nodes in plan.json
            ran = {h_.get("node") for h_ in state["history"]}
            fresh = [n for n, v in plan["nodes"].items() if n not in ran and not v.get("follower_only")]
            if not fresh:
                if not state.get("idle_noted"):
                    journal(f"\nPlan exhausted at {time.strftime('%m-%d %H:%M')}: the loop waits for new nodes in plan.json.\n")
                    log("plan exhausted: waiting for new nodes")
                    state["idle_noted"] = True
                    save(state)
                time.sleep(120)
                continue
            if state.pop("idle_noted", None):
                journal("| time | run | change | FID | reference | margin | verdict | next |\n|---|---|---|---|---|---|---|---|")
            node_id = state["node"] = fresh[0]
        node = plan["nodes"][node_id]
        name = f"{TAG}-g{gen}-{node_id}"
        set_cycle(plan, node.get("cost", 1.0))
        log("experiment", name, "|", f"{CYCLE:,} steps (cost {node.get('cost', 1.0)}) |", node["why"])
        rec = run_experiment(name, parent, merge(state["base_change"], node["change"]), seed, node["why"],
                             f"{TAG}-g0-control (same baseline checkpoint, same seed, same machine, no change)")
        ok = rec is not None and rec["fid_ema"] < state["ref_fid"] - state["margin"]
        nxt = node.get("pass") if ok else node.get("fail")
        fid_text = f"{rec['fid_ema']:.2f}" if rec else "FAILED TO RUN"
        journal(f"| {time.strftime('%H:%M')} | {name} | {json.dumps(node['change'])}{f' ({CYCLE:,} steps)' if plan.get('base_steps') else ''} | {fid_text} | {state['ref_fid']:.2f} | {state['margin']:.2f} | "
                f"{'PASS: kept, new reference' if ok else 'fail: dropped'} | {nxt} |")
        state["history"].append({"run": name, "node": node_id, "fid_ema": rec and rec["fid_ema"], "ref": state["ref_fid"], "pass": ok,
                                 "minutes": rec and rec["minutes"], "steps": CYCLE})
        if ok:
            # the baseline checkpoint stays; the change stays in the config and its FID is the reference to beat
            state.update(generation=gen + 1, base_change=merge(state["base_change"], node["change"]), ref_fid=rec["fid_ema"])
        state["node"] = nxt
        save(state)
        log(name, "->", fid_text, "against", round(state["ref_fid"] or 0, 2) if not ok else "the control", "| pass" if ok else "| fail", "| next:", nxt)


if __name__ == "__main__":
    main()
