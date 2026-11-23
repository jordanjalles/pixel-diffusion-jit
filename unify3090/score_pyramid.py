"""Score the shared un-embed readout runs of the 3090 on the Spark when each one ends (FID, raw and EMA weights).

    python3 score_pyramid.py NAME--JOB [NAME--JOB ...]

For each run folder name in <data-root>/outputs/3090-experiments/runs: wait until the runner job is done, then run
gen_eval_spark.py on its step 100,000 checkpoint. The scores go to the run's gen_metrics.jsonl.
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
    while True:
        out = subprocess.run(["python3", "spark_client.py", "status", job], cwd=CLIENT, capture_output=True, text=True).stdout
        status = json.loads(out).get("status") if out.strip().startswith("{") else "?"
        if status not in ("queued", "running", "?"):
            break
        time.sleep(60)
    print(time.strftime("%H:%M"), folder, "ended with status", status, flush=True)
    if status != "done":
        continue
    time.sleep(120)                                                       # the collection of the checkpoint
    r = subprocess.run(["python3", "gen_eval_spark.py", str(ROOT / "runs" / folder), "--ckpt-dir", str(ROOT / "raw" / job / "published"),
                        "--steps", "100000", "--weights", "both", "--wait"], cwd=HERE, capture_output=True, text=True)
    print(time.strftime("%H:%M"), (r.stdout + r.stderr)[-1500:], flush=True)
