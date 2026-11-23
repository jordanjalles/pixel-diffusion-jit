"""Colab entry for the JiT-style pixel flow model (main.py = the Spark/3090 job, with the Colab patches of 2026-11-07).

Steps, in order:
1. Data: download the Hugging Face copy named in cfg "hf_repo" (train + validation parquet shards) unless the folder
   cfg "dataset_dir" already holds them (Spark smoke test: mounted read-only). /datasets/hf -> the download folder.
2. Scores: copy the uploaded Inception files into a writable folder (gen_eval caches its reference statistics there).
3. Resume: the runner passes the newest synced checkpoint in RESUME_CKPT (attempt 2 and later). It becomes
   OUTPUT_DIR/resume_latest.pt, which main.py's AUTO-RESUME takes when its saved config equals this config.
4. Run main.py in this process. Exit 0 only after "DONE"; then write the runner's done marker OUTPUT_DIR/DONE.
"""
import json
import os
import runpy
import shutil
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("OUTPUT_DIR", "/output"))
OUT.mkdir(parents=True, exist_ok=True)
cfg = json.loads(Path(os.environ["EXPERIMENT_CONFIG"]).read_text())
DATA = Path(os.environ.get("DATA_DIR", "/data"))
t0 = time.time()
os.environ.setdefault("JOB_T0", str(t0))          # main.py cfg "wall_s" counts from here
# 2026-11-07: with 8 loader workers + parallel compile on the 12 vCPUs, the runner's status/snapshot calls into the VM
# timed out (180 s each; three in a row count as a lost VM). Training and its workers run at lower CPU priority, and
# compile uses at most 6 threads, so the runner's calls always get CPU time.
os.nice(10)
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "6")


def log(msg):
    print(f"[entry {time.time() - t0:6.0f}s] {msg}", flush=True)


# 1. data
sys.path.insert(0, str(CODE))
import data_setup  # noqa: E402

ds = data_setup.ensure(cfg["hf_repo"], cfg["dataset_dir"], log)
if Path(ds) != Path(cfg["dataset_dir"]):              # Spark smoke test: read-only mount, mixed copy elsewhere
    cfg["dataset_dir"] = str(ds)
    Path("/tmp/experiment-config.json").write_text(json.dumps(cfg))
    os.environ["EXPERIMENT_CONFIG"] = "/tmp/experiment-config.json"

# 2. generative scores: Inception weights + code (uploaded), AE for fd_ae (uploaded); the folder must be writable
gen_src = Path(os.environ.get("GEN_EVAL_SRC", DATA / "gen_eval"))
gen_dir = Path(os.environ.get("GEN_EVAL_DIR", "/content/gen_eval"))
if gen_src != gen_dir:
    gen_dir.mkdir(parents=True, exist_ok=True)
    # only the network files: reference statistics of the Spark's own ImageNet copy (ref_*.pt) must not be used here,
    # the reference of this dataset is computed from its validation split on first use
    for f in [*gen_src.glob("pytorch_fid_inception.py"), *gen_src.glob("pt_inception-*.pth")]:
        if f.is_file() and not (gen_dir / f.name).exists():
            shutil.copyfile(f, gen_dir / f.name)
INC = "pt_inception-2015-12-05-6726825d.pth"
if not (gen_dir / INC).exists():
    # 2026-11-07: the runner's upload is one HTTP request per file and a 95 MB file broke it, so the VM fetches the
    # public pytorch-fid weights itself and checks the SHA-256 prefix that the file name carries
    import hashlib
    import urllib.request
    gen_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/{INC}"
    urllib.request.urlretrieve(url, gen_dir / (INC + ".part"))
    digest = hashlib.sha256((gen_dir / (INC + ".part")).read_bytes()).hexdigest()
    assert digest.startswith("6726825d"), ("Inception weights do not match", digest)
    os.replace(gen_dir / (INC + ".part"), gen_dir / INC)
    log(f"gen_eval: downloaded {INC} (sha256 {digest[:12]})")
os.environ["GEN_EVAL_DIR"] = str(gen_dir)
os.environ.setdefault("GEN_EVAL_SCALE_DIR", str(DATA / "scale"))
log(f"gen_eval: {sorted(p.name for p in gen_dir.glob('*'))} | AE folder {os.environ['GEN_EVAL_SCALE_DIR']}")

# 3a. start checkpoint of a stage (cfg resume_ckpt "DATA:init/x.pt" = an uploaded input). resume_ckpt is a resume key,
# so main.py's AUTO-RESUME config comparison ignores the rewrite.
if str(cfg.get("resume_ckpt", "")).startswith("DATA:"):
    cfg["resume_ckpt"] = str(DATA / cfg["resume_ckpt"][5:])
    assert Path(cfg["resume_ckpt"]).exists(), cfg["resume_ckpt"]
    Path("/tmp/experiment-config.json").write_text(json.dumps(cfg))
    os.environ["EXPERIMENT_CONFIG"] = "/tmp/experiment-config.json"
    log(f"stage start checkpoint: {cfg['resume_ckpt']}")

# 3b. resume
rc = os.environ.get("RESUME_CKPT")
if rc and Path(rc).exists():
    import torch
    new_step = int(torch.load(rc, map_location="cpu", weights_only=False, mmap=True)["step"])
    old = OUT / "resume_latest.pt"
    old_step = int(torch.load(old, map_location="cpu", weights_only=False, mmap=True)["step"]) if old.exists() else -1
    if new_step > old_step:
        shutil.copyfile(rc, OUT / "resume_latest.pt.tmp")
        os.replace(OUT / "resume_latest.pt.tmp", old)
    log(f"resume: runner checkpoint {rc} at step {new_step:,} (output folder had {old_step:,}) -> resume_latest.pt")
else:
    log("resume: none from the runner (first attempt)")

# 3c. start checkpoint given as uploaded parts (cfg resume_ckpt "DATA:init/<folder>/" with part_NNN.ckpt + manifest.json)
import ckship  # noqa: E402

if cfg.get("resume_ckpt") and Path(cfg["resume_ckpt"]).is_dir():
    joined = ckship.join(cfg["resume_ckpt"], Path("/content/init_joined.pt" if Path("/content").exists() else "/tmp/init_joined.pt"))
    log(f"stage start checkpoint joined from parts: {joined} ({joined.stat().st_size / 2**30:.2f} GiB, sha256 ok)")
    cfg["resume_ckpt"] = str(joined)
    Path("/tmp/experiment-config.json").write_text(json.dumps(cfg))
    os.environ["EXPERIMENT_CONFIG"] = "/tmp/experiment-config.json"

# 4. train
(OUT / "DONE").unlink(missing_ok=True)
shipper = None
if cfg.get("ship"):
    local = Path(os.environ.get("CKPT_LOCAL", "/content/ckpt_local" if Path("/content").exists() else "/tmp/ckpt_local"))
    shipper = ckship.Shipper(OUT, local, gap=float(cfg.get("ship_gap", 120)),
                             keep_every=int((cfg.get("ckpt_keep") or {}).get("every", 0)), log=log)
    shipper.start()
    log(f"ckship: checkpoints leave OUT as {ckship.PART // 2**20} MiB parts, one every {shipper.gap:.0f}s, local copies in {local}")
sys.argv = [str(CODE / "main.py")]
os.environ.setdefault("ML_DATASETS", "/datasets")
script = "tune.py" if cfg.get("tune") else "main.py"
try:
    runpy.run_path(str(CODE / script), run_name="__main__")
except SystemExit as exc:
    if exc.code not in (None, 0):
        raise
if script == "tune.py":
    assert (OUT / "tune_summary.json").exists(), "tune.py ended without tune_summary.json"
    (OUT / "DONE").write_text("tune_summary.json\n")
elif shipper is not None and (OUT / "PAUSED").exists():
    # stopped by cfg wall_s before the VM limit: ship the last checkpoint, no DONE; the watchdog chains the next job
    shipper.finish()
    log(f"PAUSED at step {(OUT / 'PAUSED').read_text().strip()}: checkpoint shipped, no DONE marker")
elif shipper is not None:
    shipper.finish()
    final = sorted(p for p in (OUT / "ckparts").glob("checkpoint_final_step_*") if ckship.complete(p))
    assert final, "main.py ended without a shipped final checkpoint"
    (OUT / "DONE").write_text(f"ckparts/{final[-1].name}\n")
else:
    final = sorted(OUT.glob("checkpoint_final_step_*.pt"))
    assert final, "main.py ended without a final checkpoint"
    (OUT / "DONE").write_text(f"{final[-1].name}\n")
log(f"DONE marker written ({(OUT / 'DONE').read_text().strip()})")
# loader workers and compile workers kept the Spark smoke container alive ~1 min after DONE: on Colab that time is paid
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
