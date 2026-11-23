"""Hugging Face ImageNet copies for the Colab job: download once per VM, link the shard folder to the config's path."""
import os
import time
from pathlib import Path

SPLITS = ("train", "validation")


def shards(ds_dir):
    ds_dir = Path(ds_dir)
    return {s: sorted(ds_dir.glob(f"{s}-*-of-*.parquet")) for s in SPLITS} if ds_dir.exists() else {}


def ensure(repo, ds_dir, log=print):
    """ds_dir holds the train + validation shards of repo afterwards (download unless they are already there)."""
    ds_dir = Path(ds_dir)
    have = shards(ds_dir)
    if not all(have.get(s) for s in SPLITS):
        from huggingface_hub import snapshot_download
        local = Path(os.environ.get("HF_LOCAL", "/content/hf")) / repo.replace("/", "__")
        log(f"downloading {repo} (train + validation) to {local}")
        t0 = time.time()
        for attempt in range(5):
            try:
                snapshot_download(repo, repo_type="dataset", local_dir=str(local),
                                  allow_patterns=["data/train-*", "data/validation-*"], max_workers=8)
                break
            except Exception as exc:             # a transient network error: try again, then give up loudly
                log(f"download attempt {attempt + 1} failed: {exc}")
                if attempt == 4:
                    raise
                time.sleep(30 * (attempt + 1))
        ds_dir.parent.mkdir(parents=True, exist_ok=True)
        if ds_dir.is_symlink() or ds_dir.exists():
            ds_dir.unlink()
        ds_dir.symlink_to(local / "data")
        have = shards(ds_dir)
        log(f"download of {repo} took {time.time() - t0:.0f}s")
    size = sum(p.stat().st_size for s in SPLITS for p in have[s])
    log(f"data: {len(have['train'])} train + {len(have['validation'])} validation shards, {size / 2**30:.1f} GiB in {ds_dir}")
    assert have["train"] and have["validation"], have
    return mix(ds_dir, log)


def mix(ds_dir, log=print, seed=0):
    """The HF copies are sorted by class (2026-11-09: shard 0 = classes 0-76, ..., shard 12 = 923-999), and the loader
    builds each batch from one shard, so a batch held ~77 of the 1000 classes. Every image goes to a random output shard
    (fixed seed: every job of a run gets the same shards); the loader shuffles inside a shard. One input shard in RAM at
    a time. Returns the folder to train from: ds_dir itself when it is a symlink (relinked), else the mixed folder."""
    import numpy as np
    import pyarrow.parquet as pq
    ds_dir = Path(ds_dir)
    if (ds_dir.resolve() / "MIXED").exists():          # already relinked by an earlier call on this VM
        return ds_dir
    src = shards(ds_dir)
    out =Path(os.environ.get("MIX_DIR", "/content/hf_mixed" if Path("/content").exists() else "/tmp/hf_mixed")) / ds_dir.resolve().parent.name
    done = out / "MIXED"
    if not done.exists():
        t0 = time.time()
        out.mkdir(parents=True, exist_ok=True)
        n = len(src["train"])
        names = [p.name for p in src["train"]]
        rng = np.random.default_rng(seed)
        writers = [None] * n
        for p in src["train"]:
            t = pq.read_table(p)
            dest = rng.integers(0, n, size=t.num_rows)
            for k in range(n):
                part = t.take(np.flatnonzero(dest == k))
                if writers[k] is None:
                    writers[k] = pq.ParquetWriter(out / (names[k] + ".tmp"), part.schema)
                writers[k].write_table(part)
            del t
        for k, w in enumerate(writers):
            w.close()
            os.replace(out / (names[k] + ".tmp"), out / names[k])
        for p in src["validation"]:
            if not (out / p.name).exists():
                (out / p.name).symlink_to(p.resolve())
        labels = pq.read_table(out / names[0], columns=["label"]).column("label").to_numpy()
        rows = sum(pq.ParquetFile(out / x).metadata.num_rows for x in names)
        assert rows == sum(pq.ParquetFile(p).metadata.num_rows for p in src["train"]), "mixed shards lost rows"
        assert len(np.unique(labels)) > 990, ("mixed shard 0 does not hold all classes", len(np.unique(labels)))
        done.write_text(f"{rows} rows, seed {seed}\n")
        log(f"CHECK data mix: {rows:,} train images into {n} shards in {time.time() - t0:.0f}s | "
            f"shard 0 holds {len(np.unique(labels))} classes (sorted copy: ~77)")
    if ds_dir.is_symlink():
        ds_dir.unlink()
        ds_dir.symlink_to(out)
        return ds_dir
    return out
