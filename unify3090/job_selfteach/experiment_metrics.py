"""Copy with experiment code. No credentials or network required."""
import json, os
from pathlib import Path

OUT = Path(os.environ.get('OUTPUT_DIR','/output'))

def log_metrics(step, **metrics):
    with (OUT/'metrics.jsonl').open('a') as f:
        f.write(json.dumps({'step':step,**metrics},allow_nan=False)+'\n'); f.flush()

def publish(path):
    """Hand off an immutable artifact; never reuse this path after publishing."""
    path=Path(path)
    target=OUT/'published'/path.name
    target.parent.mkdir(exist_ok=True)
    if target.exists(): raise FileExistsError(target)
    with path.open('rb') as f: os.fsync(f.fileno())
    path.rename(target)
    return target
