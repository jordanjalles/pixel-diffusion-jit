"""Multi-scale attention separation series on the 3090 (user 2026-10-15: "use 3090 to run 20M param 150k step experiments",
"Go ahead and start the experiments on 3090 while I'm gone"). One job at a time, in this order, then EMA FID scores.

    python3 msa_series.py            (systemd user unit msa-series.service; log: msa_series.log via the journal)

Runs (seed 2, 20M: 8 blocks x 360, the seed 2 control config, lr 1e-4 then 1e-5 for the last 12.5%):
  msa spatial pairs x shared t, spatial x staggered t, long-range x shared, long-range x staggered   (150k steps)
  control 150k (dense only, equal steps)    control 450k (dense only, about equal compute: an msa step is three forwards)
A run whose folder exists with status done is skipped (restart-safe). A job the device interrupts is submitted again
with --resume-from (AUTO-RESUME in job/main.py). A job that fails stops the series.
Scores (Spark, gen_eval_spark.py, EMA weights, 10k samples, 16 Euler steps): 50k / 100k / 150k; the 450k control also 300k / 450k.
"""
import json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FT_FROM = {}                                      # fine-tune run name -> submit args of its start checkpoint
CUT_JOB, CUT_STEP = "47e65e63b7ce4f558cb715566c5a16a1", 400000     # the 450k control job, cancelled after this checkpoint
CUT_MARK = HERE / "msa_series_cut.txt"           # written when the series has cancelled it
CLIENT = "<data-root>/outputs/3090-experiments/runner-updates/resume-2026-10-07"
R = Path("<data-root>/outputs/3090-experiments")
BASE = json.loads((HERE / "config_unify-wclamp01-control-seed2-inet128-100k.json").read_text())
log = lambda *a: print(time.strftime("%m-%d %H:%M"), *a, flush=True)
MODEL = "model: 8 blocks x width 360 (10 heads), 19.6M parameters, readouts R32 after block 3 and R64 after block 6"


def series():
    out = []
    T_TXT = {"shared": "one t per image for all three forwards",
             "staggered": "staggered t per group: dense cleaner than split cleaner than sparse at every patch (split t = t_D * u, sparse t = min split t of the group * v)",
             "mixed": "mixed t (Self-Flow, 3 levels): t1 >= t2 >= t3 from broad15, dense t1 everywhere; on one mask of 16 tokens split t2 and sparse t3, t1 elsewhere",
             "above": "dense t from broad15 as the control; one noisier level per image from broad15 at or above that noise, on one mask of 16 tokens of the split student, dense t elsewhere"}

    def msa_run(pairing, tsched):
        cfg = dict(BASE, steps=150000, lr_switch={"step": 131250, "lr": 1e-5}, msa={"pairing": pairing, "timesteps": tsched, "weight": 1.0})
        pair_txt = "neighbor pairs (domino tiling)" if pairing == "spatial" else "long-range pairs (4 rows and 4 columns apart)"
        out.append((f"unify-wclamp01-msa-{pairing}-{tsched}-seed2-inet128-150k", cfg, [50000, 100000, 150000], "\n".join([
            f"Multi-scale attention separation: dense + split 42/22 + sparse (32 single, 16 pairs) forwards of the same weights, {pair_txt}, {T_TXT[tsched]}",
            "compare: unify-wclamp01-control-seed2-inet128-150k (equal steps) and -450k (about equal compute); the other msa runs",
            MODEL,
            "train: clamp01 weight, lr 1e-4 then 1e-5 from step 131,250, 150,000 steps, batch 64, EMA 0.999 (for sampling only, no EMA teacher)",
            "loss: flow loss of each forward (weight 1) + 1.0 x (dense 6 > split 3, dense 6 > sparse 3, split 6 > sparse 3), normalized mse, targets detached",
            "see: CHECK msa at start and at step 100; charts MSA flow loss per forward, alignment terms, mean t per forward; EMA FID at 50k/100k/150k"]) + "\n"))

    def split_only():
        # user 2026-10-16: "Evaluate the 35-65 split attention different noise without the solo/pair patch forward"
        cfg = dict(BASE, steps=150000, lr_switch={"step": 131250, "lr": 1e-5}, msa={"pairing": "spatial", "timesteps": "above", "weight": 1.0, "sparse": False})
        out.append(("unify-wclamp01-msa-splitonly-above-seed2-inet128-150k", cfg, [50000, 100000, 150000], "\n".join([
            f"Split attention only: dense + split 42/22 forwards of the same weights (no single/pair forward), {T_TXT['above']}",
            "compare: unify-wclamp01-control-seed2-inet128-150k (equal steps, FID 130.1) and -450k (wall clock); the msa runs with a sparse forward",
            MODEL,
            "train: clamp01 weight, lr 1e-4 then 1e-5 from step 131,250, 150,000 steps, batch 64, EMA 0.999 (for sampling only, no EMA teacher)",
            "loss: flow loss of dense and split (weight 1, with readouts) + 1.0 x (dense 6 > split 3), normalized mse, target detached",
            "see: CHECK msa above / no sparse forward at step 100; charts MSA flow loss, alignment D6>S3, mean t; EMA FID at 50k/100k/150k"]) + "\n"))

    def control(steps):
        cfg = dict(BASE, steps=steps, lr_switch={"step": steps * 7 // 8, "lr": 1e-5})
        out.append((f"unify-wclamp01-control-seed2-inet128-{steps // 1000}k", cfg,
                     [50000, 100000, 150000] + ([300000, 400000, 425000, 450000] if steps == 450000 else []), "\n".join([
            f"Dense-only control of the msa series, {'equal steps' if steps == 150000 else 'about equal compute (an msa step is three forwards)'}",
            "compare: the unify-wclamp01-msa-*-seed2-inet128-150k runs",
            MODEL,
            f"train: clamp01 weight, lr 1e-4 then 1e-5 from step {steps * 7 // 8:,}, {steps:,} steps, batch 64, EMA 0.999",
            "loss: flow loss of the dense forward only (the same as unify-wclamp01-control-seed2-inet128-100k)",
            "see: EMA FID at 50k/100k/150k" + (" and 300k/450k" if steps == 450000 else "")]) + "\n"))

    # 2026-10-15 19:20: long x staggered taken out. "staggered" (students noisier at every patch, no equal-noise patches) is not
    # the user's spec. The user chose "mixed" (option A) instead: spatial and long, after the 150k control, before the 450k control
    for pairing, tsched in (("spatial", "shared"), ("spatial", "staggered"), ("long", "shared")):
        msa_run(pairing, tsched)
    control(150000)
    msa_run("spatial", "mixed")
    msa_run("long", "mixed")
    split_only()                                    # user 2026-10-16: in front of the 450k control
    control(450000)

    # user 2026-10-16 13:10: "Rerun the variation experiments as fine-tunes from the best 20 million param model. We should see
    # if they just hurt initial convergence not later on speed of FID improvement" + "just use the best resume checkpoint whatever
    # you can on a 3090 right now". User 13:58: "skip the repeated control ... use the 450k run 100th k checkpoint".
    # User 14:20: "start the experiments from the 450k controls 400th K checkpoint". A finished job removes its rolling resume
    # file, so the series cancels the 450k control job (CUT) right after its 400k checkpoint, resumes the control from that job
    # to 450k, and starts each fine-tune from the same file (submit --resume-from CUT job -> /resume/resume_latest.pt).
    # Control: the 450k control at 400k/425k/450k (same lr: 1e-5 from 393,750).
    def ft_run(tag, msa, what):
        cfg = dict(BASE, steps=450000, lr_switch={"step": 393750, "lr": 1e-5}, resume_ckpt="/resume/resume_latest.pt", msa=msa)
        name = f"unify-wclamp01-ft450c400k-{tag}-seed2-inet128-450k"
        FT_FROM[name] = ["--resume-from", CUT_JOB]
        out.append((name, cfg, [425000, 450000], "\n".join([
            f"Fine-tune test: {what}. Does the variation only slow early convergence, or also the later FID gain per step?",
            "compare: unify-wclamp01-control-seed2-inet128-450k at 400k/425k/450k (the same checkpoint and lr, plain flow training)",
            MODEL + ", continued from the 450k control at step 400,000 (optimizer and EMA restored)",
            "train: clamp01 weight, lr 1e-5 (as the control at these steps), to step 450,000, batch 64, EMA 0.999",
            "falsifier: FID gain from 400k to 450k equal to the control's = the variation hurts only early convergence",
            "see: RESUME line at step 400,000, CHECK msa lines; EMA FID at 425k/450k"]) + "\n"))

    for pairing, tsched in (("spatial", "shared"), ("spatial", "staggered"), ("long", "shared"), ("spatial", "mixed"), ("long", "mixed")):
        ft_run(f"msa-{pairing}-{tsched}", {"pairing": pairing, "timesteps": tsched, "weight": 1.0}, f"msa {pairing} pairs, {T_TXT[tsched]}")
    ft_run("msa-splitonly-above", {"pairing": "spatial", "timesteps": "above", "weight": 1.0, "sparse": False},
           f"split attention only, {T_TXT['above']}")
    return out


def client(*args):
    # a lost runner connection (2026-10-16 09:07: "Connection timed out during banner exchange") is retried for an hour
    for attempt in range(60):
        try:
            r = subprocess.run(["python3", "spark_client.py", *args], cwd=CLIENT, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            r = None
        if r is not None and r.returncode == 0:
            return json.loads(r.stdout)
        err = "timeout" if r is None else r.stderr[-300:].strip()
        if attempt == 59 or (r is not None and "Runner unavailable" not in r.stderr and "timed out" not in r.stderr and "Broken pipe" not in r.stderr):
            raise RuntimeError(err)
        log(f"runner not reachable ({err.splitlines()[-1] if err else ''}), retry in 60 s")
        time.sleep(60)


def find_interrupted(name):
    # the newest interrupted job of this name: the run continues from it (--resume-from)
    best = None
    for d in (R / "runs").glob(f"{name}--*"):
        if (client("status", d.name.split("--")[-1])["status"] == "interrupted" or d.name.endswith(CUT_JOB) and CUT_MARK.exists()) and (best is None or d.stat().st_mtime > best.stat().st_mtime):
            best = d
    return best.name.split("--")[-1] if best else None


def ckpt_step_max(job):
    steps = [int(f.name.split("_")[-1][:-3]) for f in (R / "raw" / job / "published").glob("checkpoint_step_*.pt")]
    return max(steps, default=0)


def find_done(name, steps):
    for d in (R / "runs").glob(f"{name}--*"):
        st = client("status", d.name.split("--")[-1])
        if st["status"] == "done" or (R / "raw" / st["id"] / "published" / f"checkpoint_final_step_{steps:06d}.pt").exists():
            return st["id"]
    return None


def find_live(name):
    # a job of this name still queued or running (the service was restarted while it trained): wait for it, do not submit again
    for d in (R / "runs").glob(f"{name}--*"):
        st = client("status", d.name.split("--")[-1])
        if st["status"] in ("queued", "running", "starting", "pending"):
            return st["id"]
    return None


for name, cfg, score_steps, desc in series():
    job = find_done(name, cfg["steps"])
    if job:
        log(f"{name}: done already ({job})")
    else:
        cfg_file = HERE / f"config_{name}.json"
        cfg_file.write_text(json.dumps(cfg, indent=1))
        resume = []
        live = find_live(name)
        pred = None if live else find_interrupted(name)
        if pred:
            resume = ["--resume-from", pred]
        elif not live:
            resume = FT_FROM.get(name, [])
        for attempt in range(3):
            if live:
                job, live = live, None
                log(f"{name}--{job}: already on the device, waiting for it")
            else:
                st = client("submit", str(HERE / "job"), "--name", name, "--entrypoint", "main.py", "--config", str(cfg_file),
                            "--dataset", "imagenet128", *resume)
                job = st["id"]
                log(f"submitted {name}--{job} ({st['status']}){' resume from ' + resume[1] if resume else ''}")
            for _ in range(60):
                d = R / "runs" / f"{name}--{job}"
                if d.is_dir():
                    (d / "DESCRIPTION.md").write_text(desc)
                    if resume:              # train-viz shows the predecessor's records up to the resume step
                        pred_run = next((R / "runs").glob(f"*--{resume[1]}")).name
                        (d / "BRANCHED_FROM.json").write_text(json.dumps({"run": pred_run, "step": ckpt_step_max(resume[1])}) + "\n")
                    break
                time.sleep(10)
            while (st := client("status", job))["status"] in ("queued", "running", "starting", "pending"):
                if job == CUT_JOB and not CUT_MARK.exists() and f"checkpoint step {CUT_STEP:,} published" in client("logs", job)["text"]:
                    client("cancel", job)
                    CUT_MARK.write_text(job + "\n")
                    while client("status", job)["status"] not in ("cancelled", "canceled", "interrupted", "done", "failed"):
                        time.sleep(30)      # --resume-from takes only an ended job
                    log(f"{name}--{job}: cancelled after its {CUT_STEP:,} checkpoint (the fine-tunes start from its resume file)")
                time.sleep(120)
            log(f"{name}--{job}: {st['status']} (exit {st.get('exit_code')})")
            # 2026-10-16 15:41: a job that wrote its final checkpoint and then lost the machine shows "interrupted"; it is done
            if st["status"] == "done" or (R / "raw" / job / "published" / f"checkpoint_final_step_{cfg['steps']:06d}.pt").exists():
                break
            if st["status"] != "interrupted" and not (job == CUT_JOB and CUT_MARK.exists()):
                sys.exit(f"{name} {st['status']}: series stopped. Logs: spark_client.py logs {job}")
            resume = ["--resume-from", job]
        else:
            sys.exit(f"{name}: interrupted 3 times, series stopped")
    run = R / "runs" / f"{name}--{job}"
    gm = run / "gen_metrics.jsonl"
    scored = {json.loads(l)["step"] for l in gm.read_text().splitlines() if '"fid_ema"' in l} if gm.exists() else set()
    if set(score_steps) <= scored:
        log(f"{name}: scored already")
        continue
    score_steps = [s for s in score_steps if s not in scored]
    time.sleep(180)                                    # the collector copies the final checkpoint
    # a resumed run: each step's checkpoint is in the published folder of whichever job of the run wrote it (scores go to the last folder)
    by_dir = {}
    for step in score_steps:
        dirs = [R / "raw" / d.name.split("--")[-1] / "published" for d in (R / "runs").glob(f"{name}--*")]
        have = [p for p in dirs if (p / f"checkpoint_step_{step:06d}.pt").exists()] or [R / "raw" / job / "published"]
        by_dir.setdefault(have[0], []).append(step)
    for ckpt_dir, steps in by_dir.items():
        r = subprocess.run([sys.executable, str(HERE / "gen_eval_spark.py"), str(run), "--ckpt-dir", str(ckpt_dir),
                            "--steps", *map(str, steps), "--weights", "ema", "--wait"], cwd=HERE)
        log(f"{name}: scores {steps} from {ckpt_dir.parent.name} {'written' if r.returncode == 0 else 'FAILED (exit %d)' % r.returncode} (gen_metrics.jsonl)")
log("series done")
