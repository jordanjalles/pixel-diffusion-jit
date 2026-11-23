"""Score the transport runs of the 3090 on the Spark: FID at 100,000 steps (raw and EMA) and at 25k / 50k / 75k (EMA).

    python3 score_ot.py NAME--JOB [NAME--JOB ...]

A folder whose job is already done is scored at once (the baseline, for its 25k / 50k / 75k checkpoints).
The scores go to the run's gen_metrics.jsonl.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLIENT = "<data-root>/outputs/3090-experiments/runner-updates/resume-2026-10-07"
ROOT = Path("<data-root>/outputs/3090-experiments")
for folder in sys.argv[1:]:
    job = folder.split("--")[-1]
    waited = False
    while True:
        out = subprocess.run(["python3", "spark_client.py", "status", job], cwd=CLIENT, capture_output=True, text=True).stdout
        status = json.loads(out).get("status") if out.strip().startswith("{") else "?"
        if status not in ("queued", "running", "?"):
            break
        waited = True
        time.sleep(60)
    print(time.strftime("%H:%M"), folder, "ended with status", status, flush=True)
    if status != "done":
        continue
    if waited:
        time.sleep(120)                                                   # the collection of the checkpoint
    for steps, weights in ((["100000"], "raw"), (["100000"], "ema"), (["25000", "50000", "75000"], "ema")):
        r = subprocess.run(["python3", "gen_eval_spark.py", str(ROOT / "runs" / folder), "--ckpt-dir", str(ROOT / "raw" / job / "published"),
                            "--steps", *steps, "--weights", weights, "--wait"], cwd=HERE, capture_output=True, text=True)
        print(time.strftime("%H:%M"), (r.stdout + r.stderr)[-2500:], flush=True)
