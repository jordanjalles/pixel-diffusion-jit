"""Checkpoint transport for Colab (2026-11-07), inside the job, without runner changes.

Measured limits of the runner's transfer path (one HTTP request per file, base64 JSON):
  Colab -> Spark: 1 GiB arrives, 2 GiB fails (server error 500, and the runner then counts the VM as lost)
  Spark -> Colab: 48 MiB uploads, 95 MiB fails
The runner's live snapshots skip .pt files and carry everything else. So:

ship (on the VM): a thread moves each new OUT/checkpoint_*.pt out of OUT (into LOCAL), splits it into parts of PART bytes
  and releases one part every GAP seconds into OUT/ckparts/<stem>/part_NNN.ckpt; manifest.json (size, sha256, parts) comes
  last. Older checkpoints' parts are removed from OUT once a newer set is complete (the Spark keeps its copies), except
  milestones. The OUT folder never holds a large .pt file, so the final collection stays small.
join (anywhere): parts + manifest -> one .pt, sha256 checked.
split (on the Spark): a .pt or a parts folder -> upload parts of UP_PART bytes for job inputs.
"""
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

PART = 256 * 2 ** 20
UP_PART = 32 * 2 ** 20


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def write_parts(src, folder, part, gap=0.0, stop=None):
    """Split src into folder/part_NNN.ckpt (temporary name, then rename), manifest.json last."""
    folder.mkdir(parents=True, exist_ok=True)
    size, digest = src.stat().st_size, sha256(src)
    n = 0
    with open(src, "rb") as f:
        while True:
            block = f.read(part)
            if not block:
                break
            tmp = folder / f"part_{n:03d}.ckpt.tmp"
            tmp.write_bytes(block)
            os.replace(tmp, folder / f"part_{n:03d}.ckpt")
            n += 1
            if gap:              # also at the end of the job: each live snapshot must carry at most ~one part
                time.sleep(gap)
    man = {"file": src.name, "size": size, "sha256": digest, "parts": n, "part_bytes": part}
    tmp = folder / "manifest.json.tmp"
    tmp.write_text(json.dumps(man, indent=1))
    os.replace(tmp, folder / "manifest.json")
    return man


def complete(folder):
    """True when folder holds a manifest and every part it names, with the right total size."""
    m = folder / "manifest.json"
    if not m.exists():
        return False
    man = json.loads(m.read_text())
    parts = [folder / f"part_{i:03d}.ckpt" for i in range(man["parts"])]
    return all(p.exists() for p in parts) and sum(p.stat().st_size for p in parts) == man["size"]


def join(folder, dest):
    folder, dest = Path(folder), Path(dest)
    man = json.loads((folder / "manifest.json").read_text())
    tmp = dest.with_name(dest.name + ".tmp")
    with open(tmp, "wb") as out:
        for i in range(man["parts"]):
            with open(folder / f"part_{i:03d}.ckpt", "rb") as f:
                shutil.copyfileobj(f, out, 1 << 22)
    got = sha256(tmp)
    assert got == man["sha256"] and tmp.stat().st_size == man["size"], ("joined checkpoint does not match", folder, got)
    os.replace(tmp, dest)
    return dest


class Shipper(threading.Thread):
    def __init__(self, out, local, gap, keep_every=0, log=print):
        super().__init__(daemon=True)
        self.out, self.local, self.gap, self.keep_every, self.log = Path(out), Path(local), gap, keep_every, log
        self.local.mkdir(parents=True, exist_ok=True)
        self.stop, self.busy, self.shipped, self.error = threading.Event(), threading.Event(), [], None

    def step_of(self, name):
        return int(Path(name).stem.split("_")[-1])

    def pending(self):
        return sorted(p for p in self.out.glob("checkpoint_*.pt") if p.is_file())

    def run(self):
        try:
            while not self.stop.is_set() or self.pending():
                todo = self.pending()
                if not todo:
                    time.sleep(5)
                    continue
                src = todo[0]
                # main.py writes .tmp and renames, so a checkpoint_*.pt name is a complete file
                self.busy.set()
                moved = self.local / src.name
                shutil.move(src, moved)         # a rename on one file system, else copy + delete (Spark smoke: bind mount)
                t0 = time.time()
                man = write_parts(moved, self.out / "ckparts" / moved.stem, PART, gap=self.gap, stop=self.stop)
                self.shipped.append(moved.stem)
                self.log(f"ckship: {moved.name} -> ckparts/{moved.stem}/ ({man['parts']} parts, {man['size'] / 2**30:.2f} GiB, "
                         f"{time.time() - t0:.0f}s)")
                self.prune(moved.stem)
                self.busy.clear()
        except Exception as exc:               # a failed shipper must not pass silently: the job checks .error at the end
            self.error = repr(exc)
            self.log(f"ckship: FAILED {exc!r}")

    def prune(self, newest):
        """Remove older part sets from OUT (and local copies) once `newest` is complete; keep milestones and finals."""
        for folder in sorted((self.out / "ckparts").iterdir()):
            if folder.name == newest or "final" in folder.name:
                continue
            step = self.step_of(folder.name)
            milestone = self.keep_every and step % self.keep_every == 0
            # give the snapshot that carries the newest set's last part time to run before older parts leave OUT
            if not milestone:
                shutil.rmtree(folder, ignore_errors=True)
            (self.local / (folder.name + ".pt")).unlink(missing_ok=True)

    def finish(self, timeout=3600):
        """Ship what is left, then wait until two more snapshot periods have passed (parts collected live)."""
        self.stop.set()
        self.join(timeout)
        if self.error:
            raise RuntimeError("checkpoint shipping failed: " + self.error)
        time.sleep(2 * self.gap)


if __name__ == "__main__":
    # python ckship.py split SRC(.pt or parts folder) DEST_FOLDER  -> upload parts (32 MiB) + manifest.json
    # python ckship.py join PARTS_FOLDER DEST.pt
    cmd, a, b = sys.argv[1:4]
    if cmd == "join":
        print(join(a, b))
    elif cmd == "split":
        src = Path(a)
        if src.is_dir():
            src = join(src, Path(b).with_suffix(".joined.pt"))
        print(write_parts(src, Path(b), UP_PART))
        if src.name.endswith(".joined.pt"):
            src.unlink()
