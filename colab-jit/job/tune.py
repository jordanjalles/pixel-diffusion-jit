"""A100 hardware tuning (2026-11-07): short main.py runs on one Colab VM, one data download.

cfg "tune" = {"phases": [{"name": ..., "seconds": S, "runs": [{"name": ..., "set": {dotted.key: value}}]}]}
Each run = main.py with the base config (this config without "tune") plus its "set" values, own folder OUT/tune/<name>,
stopped after S seconds of wall time (SIGTERM; no final checkpoint is needed). Every run appends rows to OUT/tune.jsonl
(run, phase, step, images, seconds, test loss, ms/step, img/s, peak GPU memory) so train-viz / the runner see progress.
A phase may name "pick": the best run of an earlier phase gives its values to later runs (lowest test loss at the
end, at equal images when "equal": "images"). Results: OUT/tune_summary.json.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("OUTPUT_DIR", "/output"))
cfg = json.loads(Path(os.environ["EXPERIMENT_CONFIG"]).read_text())
TUNE = cfg.pop("tune")
COMPILE_S = float(TUNE.get("compile_s", 1200))     # longest wait for the first training row
t0 = time.time()
summary = {"phases": {}, "picked": {}}


def setk(c, k, v):
    *path, last = k.split(".")
    for p in path:
        c = c.setdefault(p, {})
    c[last] = v


def rows(folder):
    f = folder / "metrics.jsonl"
    out = []
    if f.exists():
        for line in f.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def run(phase, r, seconds, extra):
    folder = OUT / "tune" / r["name"]
    folder.mkdir(parents=True, exist_ok=True)
    c = json.loads(json.dumps(cfg))
    for k, v in {**extra, **r.get("set", {})}.items():
        setk(c, k, v)
    c.update(gen_eval={}, preview_every=10 ** 9, ckpt_every=10 ** 9, ckpt_keep={})
    sys.path.insert(0, str(CODE))
    import data_setup
    data_setup.ensure(c["hf_repo"], c["dataset_dir"], lambda m: print(f"[tune] {m}", flush=True))
    c.pop("z_snapshot", None)
    (folder / "experiment-config.json").write_text(json.dumps(c, indent=1))
    env = dict(os.environ, OUTPUT_DIR=str(folder), EXPERIMENT_CONFIG=str(folder / "experiment-config.json"))
    print(f"[tune {time.time() - t0:6.0f}s] {phase}/{r['name']}: {json.dumps({**extra, **r.get('set', {})})}", flush=True)
    start = time.time()
    first = None            # time of the first logged training row: compiling ends there (12 slow vCPUs on Colab)
    with open(folder / "log.txt", "w") as log:
        p = subprocess.Popen([sys.executable, str(CODE / "main.py")], cwd=str(folder), env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        sent = 0
        while p.poll() is None:
            time.sleep(10)
            if first is None and any("sps" in x for x in rows(folder)):
                first = time.time()
            over = (first is not None and time.time() - first > seconds) or (first is None and time.time() - start > COMPILE_S)
            if over and not sent:
                os.killpg(p.pid, signal.SIGTERM)
                sent = time.time()
            elif sent and time.time() - sent > 30:
                os.killpg(p.pid, signal.SIGKILL)
    wall = time.time() - start
    compile_s = (first - start) if first else None
    # no model files stay behind: the runner's final collection moves every .pt in one request (2 GiB failed)
    for pt in folder.rglob("*.pt"):
        pt.unlink(missing_ok=True)
    rs = rows(folder)
    train = [x for x in rs if "sps" in x]
    test = [x for x in rs if "test_clean" in x]
    batch = int(c["hparams"]["CACHE_BATCH_SIZE"])
    steady = train[len(train) // 3:] or train
    res = {"phase": phase, "run": r["name"], "set": {**extra, **r.get("set", {})}, "wall_s": wall, "compile_s": compile_s, "exit": p.returncode,
           "steps": train[-1]["step"] if train else 0, "batch": batch,
           "images": (train[-1]["step"] if train else 0) * batch,
           "ms_step": 1000 / (sum(x["sps"] for x in steady) / len(steady)) if steady else None,
           "img_s": sum(x["sps"] for x in steady) / len(steady) * batch if steady else None,
           "gpu_mem_gib": max((x.get("gpu_mem_gib", 0) for x in train), default=None),
           "test_curve": [(x["step"] * batch, x["test_clean"]) for x in test],
           "oom": "OutOfMemoryError" in (folder / "log.txt").read_text()[-20000:]}
    res["test_end"] = test[-1]["test_clean"] if test else None
    with open(OUT / "tune.jsonl", "a") as f:
        f.write(json.dumps({k: v for k, v in res.items() if k != "test_curve"}) + "\n")
    print(f"[tune {time.time() - t0:6.0f}s]   -> {res['steps']} steps, {res['ms_step'] or 0:.0f} ms/step, {res['img_s'] or 0:.0f} img/s, "
          f"{res['gpu_mem_gib'] or 0:.1f} GiB, test {res['test_end']}, oom {res['oom']}", flush=True)
    return res


def at_images(curve, n):
    """test loss at n images (linear interpolation; None when the run did not get there)"""
    for (a, la), (b, lb) in zip(curve, curve[1:]):
        if a <= n <= b:
            return la + (lb - la) * (n - a) / max(b - a, 1)
    return curve[-1][1] if curve and curve[-1][0] == n else None


if TUNE.get("sync_probe_gib"):
    # a dummy checkpoint of the main run's size: the runner's checkpoint_sync log then shows how long a copy to the Spark takes
    import torch
    _p = OUT / "checkpoint_step_000001.pt"
    torch.save({"probe": torch.randn(int(TUNE["sync_probe_gib"] * 2 ** 28))}, _p.with_name(_p.name + ".tmp"))
    os.replace(_p.with_name(_p.name + ".tmp"), _p)
    print(f"[tune] sync probe {_p.name}: {_p.stat().st_size / 2 ** 30:.2f} GiB", flush=True)

picked = {}
for ph in TUNE["phases"]:
    results = [run(ph["name"], r, ph["seconds"], dict(picked)) for r in ph["runs"]]
    summary["phases"][ph["name"]] = results
    if ph.get("pick"):
        ok = [x for x in results if x["test_curve"] and not x["oom"]]
        if ph.get("pick_by") == "throughput":
            # the smallest batch with at least 90% of the best images per second (a larger batch then buys little)
            top = max(x["img_s"] for x in ok)
            score = {x["run"]: (0 if x["img_s"] >= 0.9 * top else 1, x["batch"]) for x in ok}
        elif ph.get("equal") == "images":
            n = min(x["test_curve"][-1][0] for x in ok)
            score = {x["run"]: at_images(x["test_curve"], n) for x in ok}
        else:
            score = {x["run"]: x["test_end"] for x in ok}
        best = min(score, key=score.get)
        picked.update(next(x for x in results if x["run"] == best)["set"])
        summary["picked"][ph["name"]] = {"best": best, "scores": score}
        print(f"[tune] {ph['name']}: best {best} | {json.dumps(score)} | now fixed: {json.dumps(picked)}", flush=True)
    (OUT / "tune_summary.json").write_text(json.dumps(summary, indent=1))
print(f"[tune] DONE in {(time.time() - t0) / 60:.0f} min", flush=True)
