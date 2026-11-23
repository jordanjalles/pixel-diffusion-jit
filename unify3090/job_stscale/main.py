"""ImageNet-Unify on the 3090 experiment runner: one training run of the XM series.

The model, data pipeline, train step, split loss and sampler are the user's notebook cells
(nb_code.json = the code cells of ImageNet-Unify.ipynb at the commit named in the config),
executed in notebook order. This file only adds: the XM train step for xm_k > 1, the step
loop with metric logging for train-viz, sample previews, checkpoints, and yielding the GPU
when another workload needs it.

Config (EXPERIMENT_CONFIG): xm_k, xm_n, steps, log_every, eval_every, preview_every,
ckpt_every, compile, yield_gpu, gpu_step_ms (optional), env.
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("OUTPUT_DIR", "/output"))
cfg = json.loads(Path(os.environ["EXPERIMENT_CONFIG"]).read_text()) if os.environ.get("EXPERIMENT_CONFIG") else {}
os.environ.update({
    "ML_DATASETS": "/datasets", "MPLCONFIGDIR": "/tmp/mpl", "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor",
    "TRITON_CACHE_DIR": "/tmp/triton", "XDG_CACHE_HOME": "/tmp/cache", "PYTHONUNBUFFERED": "1",
})
os.environ.update({k: str(v) for k, v in cfg.get("env", {}).items()})
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.chdir(OUT)
sys.path.insert(0, str(CODE))

import matplotlib
matplotlib.use("Agg")
import torch
import torch._dynamo
from experiment_metrics import log_metrics, publish

# cfg["resume_ckpt"]: continue a run from one of its own checkpoints that holds the optimizer state
# (a final checkpoint). Model, optimizer and step count come from the file; metrics.jsonl goes on.
# With cfg["wait_for_ckpt"] the job first waits for that file: it can be started while the run that
# writes the file is still training (the Spark has no job queue).
RESUME_CKPT = cfg.get("resume_ckpt", "")
# AUTO-RESUME (2026-10-13): checkpoint() keeps a rolling resume file resume_latest.pt in the output folder (not published,
# so it stays on the device). A job that starts again continues from it: first the file of this output folder (the same
# folder restarted), then /resume/resume_latest.pt (a job submitted again with --resume-from of its interrupted
# predecessor). The file counts only if its saved config equals this config in all keys but the resume keys and its step
# is below the total: a file of another experiment is skipped. The progress of this run goes before cfg["resume_ckpt"]
# (the start checkpoint of the experiment, e.g. the dense base of a MoE run), which is used when no such file is found.
RESUME_KEYS = ("resume_ckpt", "resume_copy", "resume_every", "resume_weights_only", "resume_lr", "wait_for_ckpt")
AUTO_RESUME = ""
for _cand in (OUT / "resume_latest.pt", Path("/resume/resume_latest.pt")):
    if not _cand.exists():
        continue
    try:
        _head = torch.load(_cand, map_location="cpu", weights_only=False)
        _step, _cfg = int(_head["step"]), _head.get("config") or {}
        del _head
    except Exception as _exc:
        print(f"AUTO-RESUME: {_cand} skipped, it cannot be read ({_exc})", flush=True)
        continue
    # keys with a leading "_" are runtime marks the job adds to its cfg (z_snapshot "_thumbs_done"), not settings
    # (2026-10-15: they made every saved config differ, so a restart went back to cfg resume_ckpt)
    _plain = lambda v: {a: _plain(b) for a, b in v.items() if not str(a).startswith("_")} if isinstance(v, dict) else v
    # cfg seed: the job adds hparams DATA_SEED = seed + 1000 below (after this check), so the saved config has it
    _now = dict(cfg, hparams=dict(cfg.get("hparams") or {}, DATA_SEED=int(cfg["seed"]) + 1000)) if cfg.get("seed") is not None else cfg
    _diff = sorted(k for k in set(_cfg) | set(_now) if k not in RESUME_KEYS and _plain(_cfg.get(k)) != _plain(_now.get(k)))
    if _diff or _step >= int(cfg.get("steps", 30000)):
        print(f"AUTO-RESUME: {_cand} skipped (step {_step:,}, config keys that differ: {_diff})", flush=True)
        continue
    RESUME_CKPT = AUTO_RESUME = str(_cand)
    print(f"AUTO-RESUME from {_cand} at step {_step}", flush=True)
    break
else:
    print("AUTO-RESUME: none found, starting at step 0" + (f" (cfg resume_ckpt {RESUME_CKPT} is used)" if RESUME_CKPT else ""), flush=True)
TEACHER ={"net": None, "decay": 0.999}       # self-flow "teacher": "ema" (set by make_teach_train_step)
if RESUME_CKPT and cfg.get("wait_for_ckpt"):
    print(f"waiting for {RESUME_CKPT}", flush=True)
    while not (Path(RESUME_CKPT).exists() and time.time() - Path(RESUME_CKPT).stat().st_mtime > 60):
        time.sleep(30)
    print("checkpoint is there and no longer changing", flush=True)
if cfg.get("keep_outputs"):
    # On the Spark there is no collector: OUTPUT_DIR is the train-viz run folder itself, so
    # previews and checkpoints stay where they are written, and the config goes next to them.
    publish = lambda path: path
    (OUT / "config.json").write_text(json.dumps(cfg, indent=1))

torch._dynamo.config.suppress_errors = True      # a graph that cannot compile runs uncompiled
XM_K, XM_N = int(cfg.get("xm_k", 1)), int(cfg.get("xm_n", 1))
STEPS = int(cfg.get("steps", 30000))
LOG_EVERY, EVAL_EVERY = int(cfg.get("log_every", 100)), int(cfg.get("eval_every", 1000))
PREVIEW_EVERY, CKPT_EVERY = int(cfg.get("preview_every", 5000)), int(cfg.get("ckpt_every", 10000))
assert 1 <= XM_N <= XM_K
print("config:", json.dumps(cfg), flush=True)

# ---------------------------------------------------------------- the notebook's cells, in order
cells = json.loads((CODE / "nb_code.json").read_text())["cells"]
# The cells run in their own module, registered under its name. torch.compile looks a
# function's globals up through sys.modules[its __name__]; a bare dict named "__main__"
# pointed it at this file, where the notebook's names do not exist (first smoke run:
# "module '__main__' has no attribute 'F'", and the step silently ran uncompiled).
import types
nb_module = types.ModuleType("unify_nb")
sys.modules["unify_nb"] = nb_module
g = nb_module.__dict__


# Loader settings of the job config. They are the same edits the user makes in the notebook:
#   loader_workers: N   ->  "num_workers=2," becomes "num_workers=N," in the two loader cells
#   draft_decode: true  ->  ImageNetFullImage asks the JPEG decoder to downscale while decoding
LOADER_WORKERS = int(cfg.get("loader_workers", 2))
DRAFT_LINE = '        image.draft("RGB", (self.image_size, self.image_size))\n'
#   disk_lock: true     ->  only one loader worker reads a shard from the disk at a time. The
#                           dataset disk reads one stream at 154 MiB/s and several streams
#                           at 9 to 32 MiB/s in total (disk_bench, 2026-10-05).
LOCK_OLD = ('                    tables.append(\n                        parquet.read(\n                            columns=["image", "label"],\n'
            '                            use_threads=False,\n                        )\n                    )\n')
LOCK_NEW = ('                    # one worker reads the disk at a time\n                    with DISK_LOCK:\n'
            '                        tables.append(\n                            parquet.read(\n'
            '                                columns=["image", "label"],\n                                use_threads=False,\n'
            '                            )\n                        )\n')
LOCK_DEF = ('import multiprocessing\n\n# Shared by all loader workers (they are forked after this line runs).\n'
            'DISK_LOCK = multiprocessing.get_context("fork").Lock()\n\n\n')
SUBS = {
    "class BufferedImageNet": [("class BufferedImageNet(IterableDataset):\n", LOCK_DEF + "class BufferedImageNet(IterableDataset):\n"),
                               (LOCK_OLD, LOCK_NEW)] if cfg.get("disk_lock") else [],
    "train_ds = BufferedImageNet(": [("    num_workers=2,\n", f"    num_workers={LOADER_WORKERS},\n")],
    "# Validation image loader": [("    num_workers=2,\n", f"    num_workers={LOADER_WORKERS},\n")],
    "#imagenet full image": [("    def __call__(self, image):\n", "    def __call__(self, image):\n" + DRAFT_LINE)] if cfg.get("draft_decode") else [],
}


RUN_SEED = cfg.get("seed")          # None = the notebook's seeds
if RUN_SEED is not None:
    RUN_SEED = int(RUN_SEED)
    cfg.setdefault("hparams", {})["DATA_SEED"] = RUN_SEED + 1000


def run_cell(prefix, after=""):
    hits = [c for c in cells if c.lstrip().startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    src = hits[0]
    for old, new in SUBS.get(prefix, []):
        assert src.count(old) == 1, (prefix, old, src.count(old))
        src = src.replace(old, new)
    hits = [src]
    t0 = time.time()
    exec(compile(hits[0] + "\n" + after, f"<cell {prefix[:40]!r}>", "exec"), g)
    print(f"ran cell {prefix[:48]!r} ({time.time() - t0:.1f}s)", flush=True)


# first cell: hyperparameters. Only the dataset location differs from the notebook.
# cfg["hparams"] overrides values of that cell (used by the local dry run and by later series).
if cfg.get("corrector"):
    cfg.setdefault("hparams", {})["pyramid_readouts"] = []      # cfg corrector: no low-resolution readouts (see CORR below)
overrides = "".join(f"{k} = {v!r}\n" for k, v in cfg.get("hparams", {}).items())
# cfg["loop"]: the looped sparse middle with XSA and deep supervision. Its cells are the blocks of
# loop_cells.py (a copy of train-viz/notebook/unify_loop_cells.py, the code of /unify_loop.html),
# put where that page tells the user to put them.
LOOP = bool(cfg.get("loop"))
# cfg["pyramid"]: Pyramid-JiT style readouts (low-resolution x-prediction heads after earlier blocks,
# each with its own loss). Cells: pyramid_cells.py, a copy of train-viz/notebook/unify_pyramid_cells.py.
PYR = bool(cfg.get("pyramid"))
assert not (LOOP and PYR)


def cell_block(n, file="loop_cells.py"):
    import re
    text = (CODE / file).read_text()
    body = re.search(rf"^# === BLOCK {n}[: ].*?===\n(.*?)(?=^# === BLOCK |\Z)", text, re.M | re.S).group(1)
    exec(compile(body, f"<{file} block {n}>", "exec"), g)
    print(f"ran {file} block {n}", flush=True)


loop_block = cell_block
pyr_block = lambda n: cell_block(n, "pyramid_cells.py")
# cfg["dataset"] = "imagenet128" (2026-10-09): the copy with every image squeezed to 128 x 128 (make_imagenet128.py),
# same shards, rows and labels. The 3090 loader waits for the disk on the 143 GiB source; the copy has 13.8 GiB.
DATASET = cfg.get("dataset", "imagenet")
IMAGENET_DIR = "/datasets/imagenet/_source/data" if DATASET == "imagenet" else f"/datasets/{DATASET}"
assert DATASET == "imagenet" or int(cfg.get("hparams", {}).get("IMAGE_SIZE", 128)) == 128, "the small copy holds 128 x 128 images"
print(f"dataset: {DATASET} at {IMAGENET_DIR}", flush=True)
run_cell("from pathlib import Path\nimport os", after=f'IMAGENET_ROOT = Path("{IMAGENET_DIR}")\n')
if LOOP:
    loop_block(1)                                            # settings: n_loops, sparse_per_dense, use_xsa, EVAL_LOOPS
if PYR:
    pyr_block(1)                                             # settings: pyramid_readouts, pyramid_weight
exec(overrides, g)
for prefix in ("#imagenet full image", "#shard finder", "class BufferedImageNet", "import signal", "#Find ImageNet shards",
               "train_ds = BufferedImageNet(", "# Validation image loader", "#convenient globals and checks",
               "import torch\nfrom torch import nn", "# patch embed / un-embed", "#qk-norm multi-head attention", "class SwiGLU",
               "#pre-post norm adaLN zero", "class TimestepEmbedder", "#define core transformer model", "#define unified model",
               "# get split losses (definitions only)", "# samplers and seed plots (definitions only)",
               "# --------------------Def Init Models"):
    if LOOP and prefix == "# --------------------Def Init Models":
        loop_block(3)                                        # init_models builds LoopedImageTransformer
        continue
    if PYR and prefix == "# --------------------Def Init Models":
        pyr_block(3)                                         # init_models builds PyramidImageTransformer
        continue
    run_cell(prefix)
    if PYR and prefix == "#define unified model":
        pyr_block(2)                                         # ReadoutHead, PyramidImageTransformer
    if LOOP and prefix == "#define unified model":
        loop_block(2)                                        # the looped model classes
    if LOOP and prefix == "# get split losses (definitions only)":
        g.setdefault("eval_logs", {})
        loop_block(4)                                        # use_loops, is_sparse_step, check_looped_model
init = [c for c in cells if ") = init_models()" in c]
assert len(init) == 1
init_src = init[0]
if RUN_SEED is not None:
    # cfg["seed"]: another init and another noise stream. The data order follows through DATA_SEED (hparams).
    assert init_src.count("SEED = 44\n") == 1
    init_src = init_src.replace("SEED = 44\n", f"SEED = {RUN_SEED}\n")
exec(compile(init_src, "<cell init models>", "exec"), g)
if RUN_SEED is not None:
    assert g["SEED"] == RUN_SEED and g["DATA_SEED"] == RUN_SEED + 1000, (g["SEED"], g["DATA_SEED"])
    print(f"CHECK seed: init and noise seed {g['SEED']} (notebook: 44), data seed {g['DATA_SEED']} (notebook: 42)", flush=True)
run_cell("################## Training Step Methods")
if LOOP:
    loop_block(5)                                            # check_looped_model(): prints the token counts, asserts the invariants
    loop_block(6)                                            # make_looped_train_step
if PYR:
    pyr_block(4)                                             # readout_losses, make_pyramid_train_step
model, optimizer, device = g["model"], g["optimizer"], g["device"]
# cfg["pair"] = {"ref": "/ref/pairing.jsonl", "every": 1000} (self-teaching scaling sweep, 2026-10-21). Variants of one scale
# and seed must see the same init, data, t, noise, rope scales and crops. Every random number of the train step is drawn
# BEFORE the step from PAIR_GEN (own generator, seed 7300 + seed), in a fixed order, also in the baseline (which does not
# use the crops). The compiled step then has no random op, so a variant with more passes cannot shift the noise stream.
#   rope: the model's rope_tables reads its training log scale from ROPE_LOG (filled before every step) instead of
#   drawing it; an explicit log_scale argument (the judge passes) wins. Eval mode is unchanged (scale 1).
#   pairing.jsonl: fingerprints of the init, the batches and the drawn numbers every "every" steps (+ steps 0, 1, 2).
#   "ref": the baseline's file: the run stops at the first fingerprint that differs (see CHECK pairing).
PAIR = cfg.get("pair") or {}
if PAIR:
    import math
    import types as _types
    assert RUN_SEED is not None and cfg.get("schedule") == "broad15", "pair: needs a seed and the broad15 step"
    PAIR_GEN = torch.Generator(device=device).manual_seed(7300 + RUN_SEED)
    ROPE_LOG = torch.zeros((), device=device)
    _rope_m = getattr(model, "_orig_mod", model)
    _rope_core = _rope_m.core_transformer

    def _paired_rope_tables(self, patch_rows, patch_cols, log_scale=None):
        # the notebook's rope_tables with the random log scale replaced by ROPE_LOG / log_scale
        dev_ = self.inv_freq.device
        row_pos = (2 * torch.arange(patch_rows, device=dev_, dtype=torch.float32) / max(patch_rows - 1, 1) - float(patch_rows > 1)).repeat_interleave(patch_cols)
        col_pos = (2 * torch.arange(patch_cols, device=dev_, dtype=torch.float32) / max(patch_cols - 1, 1) - float(patch_cols > 1)).repeat(patch_rows)
        scale = 1.0
        if self.training:
            scale = (ROPE_LOG if log_scale is None else log_scale).exp()
        row_angles = 2 * math.pi * scale * row_pos[:, None] * self.inv_freq[None, :]
        col_angles = 2 * math.pi * scale * col_pos[:, None] * self.inv_freq[None, :]
        return row_angles.cos(), row_angles.sin(), col_angles.cos(), col_angles.sin()

    # CHECK: the patched function equals the notebook's in eval mode (scale 1) and in training mode for the same drawn
    # scale (the notebook's draw is replayed from a saved state of the global stream, which is restored afterwards)
    with torch.no_grad(), torch.random.fork_rng(devices=[device]):
        _was_training = _rope_core.training
        _rope_core.eval()
        _ref_eval = _rope_core.rope_tables(8, 8)
        _rope_core.train()
        _cuda_state = torch.cuda.get_rng_state(device)
        _ref_train = _rope_core.rope_tables(8, 8)
        torch.cuda.set_rng_state(_cuda_state, device)
        ROPE_LOG.copy_(torch.empty((), device=device).uniform_(-math.log(2), math.log(2)))
        _rope_core.rope_tables = _types.MethodType(_paired_rope_tables, _rope_core)
        _new_train = _rope_core.rope_tables(8, 8)
        _rope_core.eval()
        _new_eval = _rope_core.rope_tables(8, 8)
        _rope_core.train(_was_training)
        ROPE_LOG.zero_()
        _rope_gap = max(float((a_ - b_).abs().max()) for a_, b_ in zip(_ref_train + _ref_eval, _new_train + _new_eval))
    print(f"CHECK pair rope: patched rope_tables against the notebook's, eval and training mode: largest gap {_rope_gap:.2e}", flush=True)
    assert _rope_gap < 1e-5, "paired rope_tables differs from the notebook's"
    PAIR_EVERY = int(PAIR.get("every", 1000))

    def pair_draw(images):
        """The step's random numbers, in a fixed order. The rope log scale of the model pass goes into ROPE_LOG."""
        B_ = images.shape[0]
        out_ = {"u": torch.randn(B_, generator=PAIR_GEN, device=device),
                "noise": torch.randn(tuple(images.shape), generator=PAIR_GEN, device=device),
                "crop": torch.rand(B_, 3, generator=PAIR_GEN, device=device),
                "st_log": torch.empty((), device=device).uniform_(-math.log(2), math.log(2), generator=PAIR_GEN)}
        ROPE_LOG.copy_(torch.empty((), device=device).uniform_(-math.log(2), math.log(2), generator=PAIR_GEN))
        if cfg.get("selfflow"):
            # self-flow series (2026-10-23): the two t levels and the noisier-patch draw of the self-flow step. Drawn after
            # the numbers above, so a run without self-flow keeps the stream of the first sweep
            T_ = (images.shape[-2] // _rope_m.patch_size) * (images.shape[-1] // _rope_m.patch_size)
            out_["u2"] = torch.randn(B_, 2, generator=PAIR_GEN, device=device)
            out_["mask"] = torch.rand(B_, T_, generator=PAIR_GEN, device=device)
        return out_

    PAIR_REF = {}            # (step, kind) -> fingerprint of the baseline
    if PAIR.get("ref"):
        for _l in Path(PAIR["ref"]).read_text().splitlines():
            if _l.strip():
                _r = json.loads(_l)
                PAIR_REF[(int(_r["step"]), _r["kind"])] = _r
        assert (0, "init") in PAIR_REF and (1, "after") in PAIR_REF, f"pair ref {PAIR['ref']}: no fingerprint of the init and of step 1"
        # the reference must be the same run without self-teaching: every config key but these must be equal
        _ref_cfg = json.loads((Path(PAIR["ref"]).parent / "experiment-config.json").read_text())
        _skip = ("selfteach", "pair")
        # hparams DATA_SEED: the job adds it to its own cfg (seed + 1000); the reference file on disk may not have it
        _norm = lambda c_: dict(c_, hparams={k_: v_ for k_, v_ in (c_.get("hparams") or {}).items() if k_ != "DATA_SEED"})
        _a, _b = _norm(_ref_cfg), _norm(cfg)
        _cfg_diff = sorted(k_ for k_ in set(_a) | set(_b) if k_ not in _skip and not k_.startswith("_") and _a.get(k_) != _b.get(k_))
        print(f"CHECK pair config: reference config differs only in {list(_skip)}: {not _cfg_diff} (other keys that differ: {_cfg_diff})", flush=True)
        assert not _cfg_diff and not _ref_cfg.get("selfteach"), ("the pair reference is not this run's baseline", _cfg_diff)
    PAIR_STATE = {"resumed": False, "checked": 0}
    print(f"pair: own generator seed {7300 + RUN_SEED}, rope scale from ROPE_LOG, fingerprints every {PAIR_EVERY} steps"
          + (f", reference {PAIR['ref']} ({len(PAIR_REF)} fingerprints)" if PAIR_REF else ", no reference (this run is the baseline)"), flush=True)
# cfg["cond"] = {"classes": 1000, "drop": 0.1} (user, 2026-10-13): class-conditional model. One learned vector per class
# is added to the time embedding, so every block gets it through its scale / shift / gate. Index "classes" means "no
# class": 10% of the train images get it (label dropout), and every pass outside the train step uses it (test losses,
# previews). The generative scores sample with the class seed % classes, with classifier-free guidance when
# gen_eval "cfg" is not 1: x = x_noclass + cfg * (x_class - x_noclass).
COND = cfg.get("cond") or {}
COND_CTX = {"y": None}
if COND:
    COND_N, COND_DROP = int(COND.get("classes", 1000)), float(COND.get("drop", 0.1))
    assert not cfg.get("moe") and not cfg.get("xm_ot") and not cfg.get("distill2"), "cond is written for the plain train step"

    class CondTimeEmbedder(torch.nn.Module):
        def __init__(self, inner, n, dim, gen):
            super().__init__()
            self.inner, self.n = inner, n
            self.class_emb = torch.nn.Embedding(n + 1, dim)
            with torch.no_grad():       # own generator: the noise stream of the run stays that of the baseline
                self.class_emb.weight.copy_(torch.randn(n + 1, dim, generator=gen) * 0.02)

        def forward(self, time):
            out = self.inner(time)
            y = COND_CTX["y"]
            if y is None:
                return out + self.class_emb.weight[self.n].to(out.dtype)
            if y.shape[0] != out.shape[0]:          # one time per token, flattened (B * T,)
                assert out.shape[0] % y.shape[0] == 0, (out.shape, y.shape)
                y = y.repeat_interleave(out.shape[0] // y.shape[0])
            return out + self.class_emb(y).to(out.dtype)

    _cond_raw = getattr(model, "_orig_mod", model)
    with torch.no_grad():
        _cond_dim = _cond_raw.timestep_embedder(torch.zeros(2, device=device)).shape[-1]
    _cond_raw.timestep_embedder = CondTimeEmbedder(_cond_raw.timestep_embedder, COND_N, _cond_dim,
                                                   torch.Generator().manual_seed(7101 + int(cfg.get("seed", 0) or 0))).to(device)
    COND_EMB = _cond_raw.timestep_embedder.class_emb
    # the class of a sample seed: a fixed random order of the classes (seed 0..11 of the previews are 12 unrelated classes,
    # not the first 12 of the list). cfg cond "preview_cfg": guidance scale of the preview samples (user, 2026-10-13:
    # "for conditional runs the reconstructions should have CFG with fixed random classes"). Default 2.
    COND_PERM = torch.randperm(COND_N, generator=torch.Generator().manual_seed(7104))
    COND_PREVIEW_CFG = float(COND.get("preview_cfg", 2.0))
    COND_GEN = torch.Generator().manual_seed(7102 + int(cfg.get("seed", 0) or 0))
    assert len(optimizer.param_groups) == 1
    import inspect
    _cond_keys = set(inspect.signature(type(optimizer).__init__).parameters)
    # nothing is trained yet: a new optimizer of the same kind and settings over all parameters
    optimizer = type(optimizer)(model.parameters(), **{k_: v_ for k_, v_ in optimizer.defaults.items() if k_ in _cond_keys})
    print(f"cond: {COND_N} classes + no class, label dropout {COND_DROP:g}, class vector of {_cond_dim} values added to the time embedding | "
          f"{sum(p_.numel() for p_ in model.parameters()):,} parameters", flush=True)


@torch.inference_mode()
def cond_seeds(seeds, steps, scale=1.0, use_class=True):
    """Euler sampler of the class-conditional model. Class of a sample: seed % classes. scale 1: no guidance."""
    size, n = g["IMAGE_SIZE"], len(seeds)
    m = getattr(model, "_orig_mod", model)
    x = torch.stack([torch.randn((m.un_embed.c, size, size), generator=torch.Generator(device=device).manual_seed(int(s_)), device=device)
                     for s_ in seeds])
    y = torch.tensor([int(COND_PERM[int(s_) % COND_N]) for s_ in seeds], device=device, dtype=torch.long)
    ts = g["make_timesteps"](steps, "linear", 1.0)
    was_training = model.training
    model.eval()
    try:
        for t, t_next in zip(ts[:-1], ts[1:]):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                COND_CTX["y"] = y if use_class else None
                final = m(x, t.expand(n)).float()
                if use_class and scale != 1.0:
                    COND_CTX["y"] = None
                    none = m(x, t.expand(n)).float()
                    COND_STAT["diff"] += float((final - none).abs().mean()); COND_STAT["n"] += 1
                    final = none + scale * (final - none)
            x = x + (t_next - t) * (final - x) / (1 - t)
    finally:
        COND_CTX["y"] = None
        model.train(was_training)
    return x.clamp(0, 1)


COND_STAT = {"diff": 0.0, "n": 0}
COND_PROBE = {}


@torch.inference_mode()
def cond_probe():
    """mse to the clean image of 4 fixed test batches at noise level 0.95: true class, no class, the class of another image.
    (First version, 2026-10-13: 1 batch at noise level 0.7. There the image itself shows the class, and the three errors
    differed by 0.00001: the run stopped itself at step 10,000 on noise.)"""
    m = getattr(model, "_orig_mod", model)
    if not COND_PROBE:
        _it = iter(g["test_dataloader"])
        _four = [next(_it) for _ in range(4)]
        images, labels = torch.cat([b_[0] for b_ in _four]), torch.cat([b_[1] for b_ in _four])
        gen = torch.Generator().manual_seed(7103)
        COND_PROBE.update(x=images.to(device), y=labels.to(device).long(), noise=torch.randn(images.shape, generator=gen).to(device))
        assert int(COND_PROBE["y"].min()) >= 0 and int(COND_PROBE["y"].max()) < COND_N and COND_PROBE["y"].unique().numel() > 1, COND_PROBE["y"]
    x, y, noise = COND_PROBE["x"].float(), COND_PROBE["y"], COND_PROBE["noise"]
    t = torch.full((x.shape[0],), 0.05, device=device)
    noised = 0.05 * x + 0.95 * noise
    was_training = model.training
    model.eval()
    out = {}
    try:
        for name, labels in (("true", y), ("none", None), ("wrong", y.roll(1))):
            COND_CTX["y"] = labels
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                pred = m(noised.to(COND_PROBE["x"].dtype), t)
            pred = (pred[-1] if isinstance(pred, (list, tuple)) else pred).float()
            out["cond_err_" + name] = float((pred - x).square().mean())
    finally:
        COND_CTX["y"] = None
        model.train(was_training)
    out["cond_emb_norm"] = float(COND_EMB.weight[:COND_N].float().norm(dim=1).mean())
    return out
# cfg["corrector"] = {"split": 4, "k": 1} (user, 2026-10-13): iterative correction training, adapted from "Correction
# Training for Diffusion Models" (sunovivid.github.io/correction-training). Needs the pyramid code path with an EMPTY
# readout list (hparams pyramid_readouts = []). Blocks 1 .. split are the base denoiser D, the other blocks the corrector C.
#   D: embed -> norm_in -> blocks 1 .. split -> base_norm_out -> base_un_embed -> sigmoid     = x0, the clean-image prediction
#   C: corr_embed -> corr_norm_in -> blocks split + 1 .. -> norm_out -> un_embed -> sigmoid   = the corrected prediction
# The ONLY data input of C is the detached image x0 (and the time, through C's own timestep embedder). No hidden state,
# no residual stream, no skip of D reaches C: the un-embed skip of C reads C's own token stream, which starts at corr_embed.
# Train: model(x, t, all_readouts=True) = [x0, C(stop_grad(x0))]. The pyramid train step gives both the same weighted
# loss against the clean image, at equal weight: total = L_corr ("clean", the final output) + L_base (readout "base").
# Exactly one correction pass per train step. Sampling: model(x, t) = C applied corr_k times to x0 with t fixed
# (corr_k 0 = D alone). "k" is the count of the previews and of the test loss (default 1, the trained setting).
# gen_eval "corr_k" sets it for a score. One sampler step costs split + k * (blocks - split) block evaluations.
CORR = cfg.get("corrector") or {}
if CORR:
    CORR_SPLIT, CORR_K = int(CORR["split"]), int(CORR.get("k", 1))
    _corr_raw = getattr(model, "_orig_mod", model)
    CORR_BLOCKS = len(_corr_raw.core_transformer.blocks)
    assert PYR and len(_corr_raw.readout_heads) == 0 and 1 <= CORR_SPLIT < CORR_BLOCKS, "corrector: pyramid path with hparams pyramid_readouts = []"
    assert not any(cfg.get(k_) for k_ in ("cond", "moe", "xm_ot", "distill2", "selfflow", "distill", "selfteach", "z_snapshot", "pyramid_shared")), "corrector: plain train step only"

    class CorrectorTransformer(g["PyramidImageTransformer"]):
        def base(self, images, time, rope, rows, cols):
            core = self.core_transformer
            x = core.norm_in(self.embed(images))
            time_emb = self.timestep_embedder(time)
            for i in range(self.corr_split):
                x = core.blocks[i](x, rope, time_emb)
            return torch.sigmoid(self.base_un_embed(self.base_norm_out(x), rows, cols))

        def correct(self, pred, time, rope, rows, cols):
            # everything C knows of the input comes through `pred`, an image
            core = self.core_transformer
            x = self.corr_norm_in(self.corr_embed(pred))
            time_emb = self.corr_timestep_embedder(time)
            for i in range(self.corr_split, len(core.blocks)):
                x = core.blocks[i](x, rope, time_emb)
            return torch.sigmoid(self.un_embed(core.norm_out(x), rows, cols))

        def corr_chain(self, images, time, k):
            """[x0, C(x0), C(C(x0)), ...]: k corrections with the time fixed."""
            rows, cols = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size
            rope = self.core_transformer.rope_tables(rows, cols)
            chain = [self.base(images, time, rope, rows, cols)]
            for _ in range(int(k)):
                chain.append(self.correct(chain[-1].detach(), time, rope, rows, cols))
            return chain

        def forward(self, images, time, all_readouts=False):
            assert images.shape[-2] % self.patch_size == 0 and images.shape[-1] % self.patch_size == 0
            if all_readouts:
                rows, cols = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size
                rope = self.core_transformer.rope_tables(rows, cols)
                x0 = self.base(images, time, rope, rows, cols)
                # corr_fixed_base: the no-leak CHECK only (C gets this image in place of D's prediction)
                given = x0 if self.corr_fixed_base is None else self.corr_fixed_base
                return [x0, self.correct(given.detach(), time, rope, rows, cols)]
            k = self.corr_k
            if self.corr_range is not None:
                # gen_eval "corr_range": [lo, hi] (noise levels, user 2026-10-13): corr_k corrections only at sampler steps whose
                # noise level is inside the range, corr_k_out corrections at the others. The sampler gives one time per batch.
                inside = self.corr_range[0] <= 1.0 - float(time.flatten()[0]) <= self.corr_range[1]
                k = self.corr_k if inside else self.corr_k_out
                self.corr_calls[0 if inside else 1] += 1
            return self.corr_chain(images, time, k)[-1]

    with torch.random.fork_rng(devices=[0] if torch.cuda.is_available() else []):
        # own seed for the new modules: the random streams of the run are not moved by their creation
        torch.manual_seed(7200 + int(cfg.get("seed", 0) or 0))
        _d, _ps = int(g["emb_dim"]), int(g["patch_size"])
        _corr_raw.base_norm_out = torch.nn.RMSNorm(_d)
        _corr_raw.base_un_embed = g["LinSkipUnEmbed"](_ps, 3, _d)
        _corr_raw.corr_embed = g["LinSkipPatchEmbed"](_ps, 3, _d)
        _corr_raw.corr_norm_in = torch.nn.RMSNorm(_d)
        _corr_raw.corr_timestep_embedder = g["TimestepEmbedder"](_d)
    _corr_raw.corr_split, _corr_raw.corr_k, _corr_raw.corr_fixed_base = CORR_SPLIT, CORR_K, None
    _corr_raw.corr_range, _corr_raw.corr_k_out, _corr_raw.corr_calls = None, CORR_K, [0, 0]
    _corr_raw.__class__ = CorrectorTransformer
    _corr_raw.to(device)
    assert len(optimizer.param_groups) == 1
    import inspect
    _corr_keys = set(inspect.signature(type(optimizer).__init__).parameters)
    # nothing is trained yet: a new optimizer of the same kind and settings over all parameters
    optimizer = type(optimizer)(model.parameters(), **{k_: v_ for k_, v_ in optimizer.defaults.items() if k_ in _corr_keys})

    def corr_parts():
        m, core = _corr_raw, _corr_raw.core_transformer
        d_mods = [m.embed, core.norm_in, m.base_norm_out, m.base_un_embed, m.timestep_embedder] + [core.blocks[i] for i in range(CORR_SPLIT)]
        c_mods = [m.corr_embed, m.corr_norm_in, core.norm_out, m.un_embed, m.corr_timestep_embedder] + [core.blocks[i] for i in range(CORR_SPLIT, CORR_BLOCKS)]
        return [p_ for mod in d_mods for p_ in mod.parameters()], [p_ for mod in c_mods for p_ in mod.parameters()]

    def corr_check():
        """CHECK at the start: every parameter is in D or in C; the corrector loss gives D no gradient and the base loss
        gives C none; a change of D's hidden state that leaves the image given to C unchanged does not change C's output."""
        m = _corr_raw
        d_par, c_par = corr_parts()
        ids = [id(p_) for p_ in d_par + c_par]
        assert len(ids) == len(set(ids)) == len(list(m.parameters())), "a parameter is in both parts or in none"
        size = int(g["IMAGE_SIZE"])
        gen = torch.Generator().manual_seed(7201)
        images = torch.rand(4, 3, size, size, generator=gen).to(device)
        t = torch.tensor([0.1, 0.4, 0.6, 0.9], device=device)
        noised = t[:, None, None, None] * images + (1 - t)[:, None, None, None] * torch.randn(images.shape, generator=gen).to(device)
        was = m.training
        m.eval()
        gsum = lambda ps: sum(float(p_.grad.abs().sum()) for p_ in ps if p_.grad is not None)
        try:
            grads = {}
            for name, index in (("corr", 1), ("base", 0)):
                m.zero_grad(set_to_none=True)
                (m(noised, t, all_readouts=True)[index].float() - images).square().mean().backward()
                grads[name] = (gsum(d_par), gsum(c_par))
            m.zero_grad(set_to_none=True)
            with torch.no_grad():
                a = m(noised, t, all_readouts=True)
                hook = m.core_transformer.blocks[CORR_SPLIT - 1].register_forward_hook(lambda mod, args, out: out + 0.5 * torch.randn_like(out))
                m.corr_fixed_base = a[0]
                try:
                    b = m(noised, t, all_readouts=True)
                finally:
                    hook.remove()
                    m.corr_fixed_base = None
                moved_d, moved_c = float((b[0] - a[0]).abs().max()), float((b[1] - a[1]).abs().max())
                k_out = [float((m.corr_chain(noised, t, k_)[-1] - a[0]).abs().mean()) for k_ in (0, 1, 2)]
        finally:
            m.train(was)
        n_d, n_c = sum(p_.numel() for p_ in d_par), sum(p_.numel() for p_ in c_par)
        print(f"CHECK corrector: split after block {CORR_SPLIT} of {CORR_BLOCKS}, k = {CORR_K} | parameters: base denoiser {n_d:,} + corrector {n_c:,} = {n_d + n_c:,} | "
              f"gradient of the corrector loss: into D {grads['corr'][0]:.3g}, into C {grads['corr'][1]:.3g} | of the base loss: into D {grads['base'][0]:.3g}, into C {grads['base'][1]:.3g} | "
              f"no-leak test: D's hidden state changed (its prediction moved by {moved_d:.3g}), C's output moved by {moved_c:.3g} | "
              f"mean |output - x0| with k = 0 / 1 / 2: {k_out[0]:.3g} / {k_out[1]:.3g} / {k_out[2]:.3g}", flush=True)
        assert grads["corr"][0] == 0.0 and grads["corr"][1] > 0.0, "the corrector loss reaches the base denoiser"
        assert grads["base"][1] == 0.0 and grads["base"][0] > 0.0, "the base loss reaches the corrector"
        assert moved_d > 1e-6 and moved_c < 1e-6, "the corrector sees more of the base denoiser than its image prediction"
        assert k_out[0] < 1e-6 and k_out[1] > 0.0 and k_out[2] > 0.0, "corr_k does not set the number of corrections"

    corr_check()

    @torch.no_grad()
    def corr_eval(n_batches=4):
        """Held-out x-MSE to the clean image after k = 0 .. 8 corrections (k = 0: the base denoiser), on test images.
        corr_mse_k<k>: the noise levels of training. corr_mse_n<level>_k<k>: one fixed noise level, k in 0 1 2 4 8."""
        m = _corr_raw
        was = model.training
        model.eval()
        rng = torch.Generator(device=device).manual_seed(0)
        levels, keep = (0.9, 0.7, 0.5, 0.3), (0, 1, 2, 4, 8)
        sums, count = {name: [0.0] * 9 for name in ("mix",) + levels}, 0
        try:
            for images in g["get_eval_batches"]("test", n_batches):
                images = images.to(device, non_blocking=True).float()
                B = images.shape[0]
                noise = torch.randn(images.shape, device=device, generator=rng)
                for name in sums:
                    t = g["sample_train_t"](B, generator=rng) if name == "mix" else torch.full((B,), 1.0 - name, device=device)
                    t_img = t[:, None, None, None]
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        chain = m.corr_chain(t_img * images + (1.0 - t_img) * noise, t, 8)
                    for k_, pred in enumerate(chain):
                        sums[name][k_] += float((pred.float() - images).square().mean()) * B
                count += B
        finally:
            model.train(was)
        out = {f"corr_mse_k{k_}": v_ / count for k_, v_ in enumerate(sums["mix"])}
        for level in levels:
            out.update({f"corr_mse_n{round(100 * level)}_k{k_}": sums[level][k_] / count for k_ in keep})
        return out

    def _corr_grid(rows_of_images, path):
        from PIL import Image
        n_rows, n_cols = len(rows_of_images), len(rows_of_images[0])
        h, w = rows_of_images[0][0].shape[-2:]
        grid = torch.ones(3, n_rows * h + (n_rows - 1) * 4, n_cols * w + (n_cols - 1) * 4)
        for r, row in enumerate(rows_of_images):
            for c_, image in enumerate(row):
                grid[:, r * (h + 4):r * (h + 4) + h, c_ * (w + 4):c_ * (w + 4) + w] = image.float().clamp(0, 1).cpu()
        Image.fromarray((grid.permute(1, 2, 0) * 255).round().byte().numpy()).save(path)
        publish(path)

    @torch.no_grad()
    def corr_picture(step, levels=(0.9, 0.6, 0.3), n=6):
        """corr_<step>.png: the corrections of n fixed test images at fixed noise levels. One row per image and level
        (levels from top to bottom). Columns: clean, noised input, D's prediction, then C applied 1, 2, 4, 8 times."""
        m = _corr_raw
        was = model.training
        model.eval()
        try:
            images = g["get_eval_batches"]("test", 1)[0][:n].to(device).float()
            noise = torch.randn(images.shape, generator=torch.Generator().manual_seed(7202)).to(device)
            rows = []
            for level in levels:
                t = torch.full((images.shape[0],), 1.0 - level, device=device)
                noised = (1.0 - level) * images + level * noise
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    chain = m.corr_chain(noised, t, 8)
                for i in range(images.shape[0]):
                    rows.append([images[i], noised[i]] + [chain[k_][i] for k_ in (0, 1, 2, 4, 8)])
        finally:
            model.train(was)
        _corr_grid(rows, OUT / f"corr_{step:06d}.png")
        print(f"corrector picture step {step:,}: corr_{step:06d}.png | rows: {n} test images at noise levels {levels} | columns: clean, noised, D, C x 1, 2, 4, 8", flush=True)

    def corr_k_grid(step, seeds=range(8), sample_steps=20, ks=(0, 1, 2, 4, 8), path=None):
        """corrk_<step>.png: samples of the same seeds, one row per number of corrections k (averaged weights when there are any)."""
        m = _corr_raw
        params = [p_ for p_ in model.parameters()]
        with torch.no_grad():
            saved = [p_.detach().clone() for p_ in params]
            if ema_params is not None:
                for p_, e_ in zip(params, ema_params):
                    p_.copy_(e_)
        rows = []
        try:
            for k_ in ks:
                m.corr_k = k_
                rows.append(list(denoise_seeds(list(seeds), steps=sample_steps).float().cpu()))
        finally:
            m.corr_k = CORR_K
            with torch.no_grad():
                for p_, s_ in zip(params, saved):
                    p_.copy_(s_)
        _corr_grid(rows, path or OUT / f"corrk_{step:06d}.png")
        print(f"corrector samples step {step:,}: corrk_{step:06d}.png | rows: k = {list(ks)} corrections per sampler step | columns: seeds {list(seeds)} | "
              f"{'averaged' if ema_params is not None else 'trained'} weights, {sample_steps} euler steps", flush=True)

    print(f"corrector: base denoiser = blocks 1..{CORR_SPLIT}, corrector = blocks {CORR_SPLIT + 1}..{CORR_BLOCKS} with its own patch embedding and timestep embedder | "
          f"previews and test loss with k = {CORR_K} | {sum(p_.numel() for p_ in model.parameters()):,} parameters", flush=True)


# cfg["moe"] = {"experts": E, "at": N, "route": "choice" | "bucket", "blocks": [2, 4, 6, 8]} (user, 2026-10-12): sparse
# upcycling (Komatsuzaki et al., arXiv 2212.05055). The MLP (SwiGLU) of those blocks holds E experts, 1 expert per token.
# Until step N only expert 0 is used: the model is the dense model. At step N every expert becomes an exact copy of
# expert 0 (weights, optimizer moments, averaged weights), then the routing starts. The output directly after the copy
# equals the dense output (CHECK moe conversion).
#   "choice": Expert Choice. A linear router scores every token for every expert. The experts take turns (random
#             order) and each takes its N / E best tokens of the batch that are still free: every token has exactly
#             1 expert and every expert has the same load. The combine weight is p / p.detach(): value 1, with a
#             gradient for the router.
#   "bucket": no router. The expert is the noise bucket of the image: E buckets of equal probability under the
#             training noise distribution t = sigmoid(1.5 * randn).
MOE = cfg.get("moe") or {}
MOE_CTX = {"t": None}
MOE_LAYERS = []
if MOE:
    import torch.nn.functional as F_moe
    MOE_E, MOE_AT, MOE_ROUTE = int(MOE["experts"]), int(MOE["at"]), MOE.get("route", "choice")
    MOE_BLOCKS = [int(b_) for b_ in MOE.get("blocks", [2, 4, 6, 8])]
    assert MOE_ROUTE in ("choice", "bucket") and MOE_E >= 2 and cfg.get("schedule") == "broad15"

    class MoESwiGLU(torch.nn.Module):
        def __init__(self, dense, n_experts, route, gen):
            super().__init__()
            self.E, self.route, self.active, self.collect, self.stats = n_experts, route, False, False, None
            for name in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(dense, name)
                setattr(self, name + "_w", torch.nn.Parameter(lin.weight.detach().clone()[None].repeat(n_experts, 1, 1)))
                setattr(self, name + "_b", torch.nn.Parameter(lin.bias.detach().clone()[None].repeat(n_experts, 1)))
            if route == "choice":
                d = dense.gate_proj.in_features
                self.router = torch.nn.Linear(d, n_experts)
                with torch.no_grad():       # own generator: the noise stream of the run stays that of the dense baseline
                    self.router.weight.copy_(torch.randn(n_experts, d, generator=gen) * 0.02)
                    self.router.bias.zero_()

        def mlp(self, x, sel=None):
            # x: (G, n, d). One set of weights per row of G: all experts (sel None) or the experts sel (G,)
            gw, gb, uw, ub, dw, db = (self.gate_proj_w, self.gate_proj_b, self.up_proj_w, self.up_proj_b, self.down_proj_w, self.down_proj_b)
            if sel is not None:
                gw, gb, uw, ub, dw, db = gw[sel], gb[sel], uw[sel], ub[sel], dw[sel], db[sel]
            h = F_moe.silu(torch.baddbmm(gb[:, None], x, gw.transpose(1, 2))) * torch.baddbmm(ub[:, None], x, uw.transpose(1, 2))
            return torch.baddbmm(db[:, None], h, dw.transpose(1, 2))

        def forward(self, x):
            B, T, d = x.shape
            if not self.active:
                h = F_moe.silu(F_moe.linear(x, self.gate_proj_w[0], self.gate_proj_b[0])) * F_moe.linear(x, self.up_proj_w[0], self.up_proj_b[0])
                return F_moe.linear(h, self.down_proj_w[0], self.down_proj_b[0])
            t = MOE_CTX["t"]
            t_tok = (t.float().reshape(B, 1).expand(B, T) if t.ndim == 1 else t.float().reshape(B, T))
            if self.route == "bucket":
                t_img = t_tok.mean(1).clamp(1e-6, 1.0 - 1e-6)
                u = 0.5 * (1.0 + torch.erf(torch.logit(t_img) / (1.5 * 2 ** 0.5)))        # the share of the training inputs with a lower t
                idx = (u * self.E).long().clamp(0, self.E - 1)
                if self.collect:
                    self.stats = (idx.detach(),)
                return self.mlp(x, idx)
            N = B * T
            n = N // self.E
            assert N % self.E == 0, (N, self.E)
            flat = x.reshape(N, d)
            prob = torch.softmax(self.router(flat).float(), dim=-1)                       # (N, E)
            with torch.no_grad():
                score = prob.detach().clone()
                assign = torch.zeros(N, device=x.device, dtype=torch.long)
                turn = torch.randperm(self.E, device=x.device)
                for i in range(self.E):
                    e = turn[i:i + 1]
                    top = score.index_select(1, e).squeeze(1).topk(n).indices
                    assign[top] = e
                    score[top] = -1.0                                                     # taken: below every probability
                order = assign.argsort(stable=True)                                       # the tokens of expert 0, then 1, ...
            p_own = prob.gather(1, assign[:, None]).squeeze(1)[order].reshape(self.E, n, 1)
            out = self.mlp(flat[order].reshape(self.E, n, d))
            out = out * (p_own / p_own.detach()).to(out.dtype)
            if self.collect:
                noise = (1.0 - t_tok).reshape(N)[order].reshape(self.E, n).mean(1)
                self.stats = (noise.detach(), p_own.detach().mean())
            return torch.zeros(N, d, device=x.device, dtype=out.dtype).index_copy(0, order, out.reshape(N, d)).reshape(B, T, d)

    _moe_raw = getattr(model, "_orig_mod", model)
    _moe_gen = torch.Generator().manual_seed(7001 + int(cfg.get("seed", 0) or 0))
    for _i, _blk in enumerate(_moe_raw.core_transformer.blocks):
        if _i + 1 in MOE_BLOCKS:
            _blk.SwiGLU = MoESwiGLU(_blk.SwiGLU, MOE_E, MOE_ROUTE, _moe_gen).to(device)
            MOE_LAYERS.append(_blk.SwiGLU)
    assert len(MOE_LAYERS) == len(MOE_BLOCKS) and len(optimizer.param_groups) == 1
    import inspect
    _moe_keys = set(inspect.signature(type(optimizer).__init__).parameters)
    # nothing is trained yet: a new optimizer of the same kind and settings over the new parameters
    optimizer = type(optimizer)(model.parameters(), **{k_: v_ for k_, v_ in optimizer.defaults.items() if k_ in _moe_keys})

    def _moe_time_hook(module, args, kwargs):
        MOE_CTX["t"] = args[1] if len(args) > 1 else kwargs.get("time", kwargs.get("t"))
    _moe_raw.register_forward_pre_hook(_moe_time_hook, with_kwargs=True)
    print(f"moe: {MOE_E} experts in blocks {MOE_BLOCKS}, route {MOE_ROUTE}, dense until step {MOE_AT:,} | "
          f"{sum(p_.numel() for p_ in model.parameters()):,} parameters", flush=True)


def moe_set_active(on):
    for layer in MOE_LAYERS:
        layer.active = bool(on)


def moe_convert(at_step):
    """Every expert becomes a copy of expert 0: weights, optimizer moments, averaged weights. Then the routing starts."""
    raw = getattr(model, "_orig_mod", model)
    gen = torch.Generator().manual_seed(99)
    probe_x = torch.randn(64, 3, g["IMAGE_SIZE"], g["IMAGE_SIZE"], generator=gen).to(device) * 0.5 + 0.5
    probe_t = torch.linspace(0.01, 0.99, 64, device=device)
    def run():
        was_training = raw.training
        raw.eval()                                   # the train mode of the model is random: two passes differ
        with torch.no_grad():
            out = raw(probe_x, probe_t)
        raw.train(was_training)
        return (out[-1] if isinstance(out, (list, tuple)) else out).float()
    before = run()
    repeat = (run() - before).abs().max().item()          # the same weights twice: the size of pure rounding or randomness
    all_params = list(model.parameters())
    n_state = 0
    with torch.no_grad():
        for layer in MOE_LAYERS:
            for name in ("gate_proj", "up_proj", "down_proj"):
                for q in (getattr(layer, name + "_w"), getattr(layer, name + "_b")):
                    q[1:] = q[:1]
                    state = optimizer.state.get(q, {})
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in state:
                            state[key][1:] = state[key][:1]
                            n_state += 1
                    if ema_params is not None:
                        k = next(i for i, other in enumerate(all_params) if other is q)
                        ema_params[k][1:] = ema_params[k][:1]
                        if ema2_params is not None:
                            ema2_params[k][1:] = ema2_params[k][:1]
    moe_set_active(True)
    after = run()
    gap = (after - before).abs().max().item()
    print(f"CHECK moe conversion at step {at_step:,}: {len(MOE_LAYERS)} MLP blocks x {MOE_E} experts copied from the dense MLP | optimizer moments copied: "
          f"{n_state} tensors | averaged weights copied: {ema_params is not None} | largest output change on 64 probe inputs {gap:.2e} (two dense passes differ by {repeat:.2e})", flush=True)
    # the outputs are images in 0..1, from the model in eval mode: only float32 rounding (bmm against linear) may differ
    assert gap < 1e-4 and repeat < 1e-6 and n_state == 12 * len(MOE_LAYERS), (gap, repeat, n_state)


def moe_stats(images):
    """Expert spread (how far the experts have moved apart) and, for the router, the noise level of the tokens of each expert."""
    raw = getattr(model, "_orig_mod", model)
    out = {}
    with torch.no_grad():
        w = torch.stack([layer.gate_proj_w.detach().float() for layer in MOE_LAYERS])                # (L, E, h, d)
        center = w.mean(1, keepdim=True)
        out["moe_spread"] = ((w - center).flatten(2).norm(dim=2) / center.flatten(2).norm(dim=2)).mean().item()
        if MOE_LAYERS[0].active:
            for layer in MOE_LAYERS:
                layer.collect = True
            t = torch.sigmoid(torch.randn(images.shape[0], device=device) * 1.5)
            t_img = t[:, None, None, None]
            raw(t_img * images.to(device) + (1.0 - t_img) * torch.randn_like(images, device=device), t)
            for layer in MOE_LAYERS:
                layer.collect = False
            last = MOE_LAYERS[-1].stats
            if MOE_ROUTE == "choice":
                out["moe_noise_low"], out["moe_noise_high"] = last[0].min().item(), last[0].max().item()
                out["moe_router_p"] = sum(layer.stats[1].item() for layer in MOE_LAYERS) / len(MOE_LAYERS)
            else:
                out["moe_buckets_used"] = float(last[0].unique().numel())
    return out
# cfg["pyramid_shared"] = "pixels" (user, 2026-10-10): the readouts use the ONE final un-embed, no readout head of their
# own (no parameters). Every token goes through norm_out + un-embed as at the end of the model, then the image is
# average-pooled to the resolution of the readout. cfg["pyramid_ladder"] = [[block, pixels], ...] names the resolutions
# (down to 1 px); without it they follow pyramid_readouts. The blocks must be those of hparams pyramid_readouts.
PYR_SHARED = cfg.get("pyramid_shared") if PYR else None
if PYR_SHARED:
    assert PYR_SHARED == "pixels", PYR_SHARED

    class SharedReadout(torch.nn.Module):
        def __init__(self, owner, res):
            super().__init__()
            self.owner, self.res = (owner,), int(res)                  # a tuple: the owner is not registered as a submodule

        def forward(self, x, patch_rows, patch_cols):
            m = self.owner[0]
            image = torch.sigmoid(m.un_embed(m.core_transformer.norm_out(x), patch_rows, patch_cols))
            f = image.shape[-1] // self.res
            return torch.nn.functional.avg_pool2d(image, f) if f > 1 else image

    _m = getattr(model, "_orig_mod", model)
    _ladder = cfg.get("pyramid_ladder") or [[b_, g["IMAGE_SIZE"] // g["patch_size"] * q] for b_, q in g["pyramid_readouts"]]
    assert [b_ for b_, _ in _ladder] == [b_ for b_, _ in g["pyramid_readouts"]], (_ladder, g["pyramid_readouts"])
    assert all(g["IMAGE_SIZE"] % r_ == 0 and r_ < g["IMAGE_SIZE"] for _, r_ in _ladder), _ladder
    _m.readout_heads = torch.nn.ModuleDict({str(b_): SharedReadout(_m, r_) for b_, r_ in _ladder})
    assert sum(p_.numel() for p_ in _m.readout_heads.parameters()) == 0
    with torch.no_grad():
        _shapes = [tuple(o_.shape[-2:]) for o_ in _m(torch.rand(2, 3, g["IMAGE_SIZE"], g["IMAGE_SIZE"], device=device),
                                                     torch.rand(2, device=device), all_readouts=True)]
    assert _shapes == [(r_, r_) for _, r_ in _ladder] + [(g["IMAGE_SIZE"],) * 2], _shapes
    if cfg.get("pyramid_grad_mix") is not None:
        # the measure: with the mix, the gradient of the final loss in block 1 is (1 - mix)^(number of readouts) of the plain one
        _mix, _was = float(cfg["pyramid_grad_mix"]), _m.training
        _m.train()
        _xi, _ti = torch.rand(2, 3, g["IMAGE_SIZE"], g["IMAGE_SIZE"], device=device), torch.rand(2, device=device)
        _p1 = next(_m.core_transformer.blocks[0].parameters())
        _norms = []
        for _v in (None, _mix):
            _m.grad_mix = _v
            _outs = _m(_xi, _ti, all_readouts=True)
            _norms.append([float(torch.autograd.grad(_o.float().pow(2).mean(), _p1, retain_graph=True)[0].norm()) for _o in (_outs[-1], _outs[0])])
        _m.train(_was)
        _want = (1 - _mix) ** len(_ladder)
        _got, _got_r = _norms[1][0] / _norms[0][0], _norms[1][1] / _norms[0][1]
        assert abs(_got / _want - 1) < 0.05 and abs(_got_r / _mix - 1) < 0.05, (_got, _want, _got_r)
        print(f"CHECK gradient mix {_mix}: gradient in block 1 with the mix / without: final loss {_got:.5f} (rule {_want:.5f}) | "
              f"first readout {_got_r:.3f} (rule {_mix})", flush=True)
    print(f"CHECK shared readouts: the readouts use the final un-embed and pixel averaging, 0 readout parameters | "
          f"output sizes {[s_[0] for s_ in _shapes]} px | model parameters {sum(p_.numel() for p_ in _m.parameters()):,}", flush=True)
if PYR:
    # readout name by resolution, low to high: R32, R64 at 128 px
    PYR_NAMES = [f"R{g['IMAGE_SIZE'] // g['patch_size'] * q}" for _, q in g["pyramid_readouts"]]
    # two readouts of the same size (scaling sweep: R32 after blocks 3 and 6, because R64 needs width >= 192): the later
    # ones get their block in the name (R32, R32b6), so their losses do not overwrite each other in the records
    PYR_NAMES = [n_ if PYR_NAMES.index(n_) == i_ else f"{n_}b{g['pyramid_readouts'][i_][0]}" for i_, n_ in enumerate(PYR_NAMES)]
    assert len(set(PYR_NAMES)) == len(PYR_NAMES), PYR_NAMES
    if PYR_SHARED:
        PYR_NAMES = [f"R{r_}" for _, r_ in _ladder]
    if CORR:
        PYR_NAMES = ["base"]          # the one early output: the base denoiser's full-resolution prediction
    print(f"pyramid readouts: " + ", ".join(f"{name} after block {b}" for name, (b, _) in zip(PYR_NAMES, g["pyramid_readouts"]))
          + f" | weight {g['pyramid_weight']} each | heads: {sum(p.numel() for p in model.readout_heads.parameters()):,} parameters", flush=True)
assert str(device) == "cuda" or cfg.get("allow_cpu"), device
gpu_sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)


# ---------------------------------------------------------------- XM train step (xm_k > 1)
def make_train_step_xm(K, N):
    """The notebook's train step with XM: K noise candidates per image at the image's one t,
    train on the N candidates with the lowest loss (as in the latent flow notebook)."""
    t_std, t_mean = float(g["t_logit_std"]), float(g["t_logit_mean"])

    def train_step(images, optimizer):
        optimizer.zero_grad(set_to_none=True)
        images = images.to(device, non_blocking=True)
        B = images.shape[0]

        u = torch.randn(B, device=images.device, dtype=torch.float32) * t_std + t_mean
        t = torch.sigmoid(u).repeat(K)                 # same t across an image's K candidates
        x = images.repeat(K, 1, 1, 1)
        noise = torch.randn_like(x)
        t_img = t[:, None, None, None]
        noised_images = t_img * x + (1.0 - t_img) * noise

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x_pred = model(noised_images, t)
            denom = (1.0 - t_img).clamp_min(0.05)
            v_pred = (x_pred - noised_images) / denom
            v_target = (x - noised_images) / denom
            per_candidate = (v_pred.float() - v_target.float()).square().mean((1, 2, 3)).reshape(K, B)
            loss = per_candidate.topk(N, dim=0, largest=False).values.mean()

        loss.backward()
        optimizer.step()
        return {"loss": loss.detach(), "loss_all": per_candidate.detach().mean(), "forward_batch": x.shape[0]}

    return train_step


assert not (LOOP and XM_K > 1), "the looped train step has no XM"
# cfg["schedule"] = "broad15" (user, 2026-10-07): one broad logit-normal time schedule and a direct
# x-prediction loss with a smooth weight, for the train step and for the split losses alike:
#   t = sigmoid(1.5 * randn)          weight = log1p(1 / t^2) / log(2)
#   loss = mean((x_pred - images)^2 * weight)
# It replaces the notebook's logit-normal(t_logit_mean, t_logit_std) and its v-loss. The x-MSE at
# fixed noise levels (x_mse_t01..09) does not depend on it and stays comparable with older runs.
SCHED = cfg.get("schedule", "")
assert SCHED in ("", "broad15"), SCHED
if SCHED:
    import math
    import torch.nn.functional as F_

    def broad_t(B, generator=None):
        u = torch.randn(B, device=device, dtype=torch.float32, generator=generator) * 1.5
        return torch.sigmoid(u)

    # cfg["t_shift"] = alpha (user, 2026-10-09): dimension-based timestep shift of Esser et al. (2024), train step only.
    #   alpha = sqrt(C * H * W / 4096)      ("auto"; 3.46 for 3 x 128 x 128)      1 = no shift
    #   noise' = alpha * noise / (1 + (alpha - 1) * noise)
    # `noise` is the NOISE LEVEL, 0 = clean and 1 = pure noise. In this code t is the reverse (t = 1 clean:
    # noised = t * image + (1 - t) * noise), so the shift acts on 1 - t and alpha > 1 moves the draws towards
    # pure noise, as in the paper. The drawn distribution and the loss weight function do not change: the
    # shifted t goes into the noised input, the model and the weight. Evals and the sampler do not shift.
    # cfg["t_dist"] = {"mean": m, "std": s} (user, 2026-10-13): the train inputs get t = sigmoid(s * randn + m) in place of
    # broad15 (m = 0, s = 1.5). t = 1 is the clean image, so m > 0 means less noise. Train step of the plain model only.
    # The noise recipe of JiT (Li & He 2025: pixels in -1..1, noise scale image_size / 256 = 0.5 at 128 px, logit-normal
    # mean -0.8, std 0.8) is, for our unit noise on pixels in 0..1, exactly {"mean": 0.586, "std": 0.8}: a noise scale k on
    # the linear path equals a move of logit(t) by -ln(k), and 0.5 on -1..1 is k = 0.25 on 0..1: -0.8 + ln 4 = 0.586.
    # cfg["noise_mean_std"] = v (user, 2026-10-13): at train time each noise sample gets one more constant, v * randn, on
    # all its pixels. The mean of a noise sample then has the std sqrt(1 / pixels + v^2) (0.0045 without it), so the mean
    # of the input says less about the brightness of the image. Sampling is unchanged.
    T_DIST = cfg.get("t_dist") or {}
    T_MEAN, T_STD = float(T_DIST.get("mean", 0.0)), float(T_DIST.get("std", 1.5))
    # t_dist "pure_share": p, "pure_min": m (2026-10-13): a share p of the train inputs gets a noise level uniform in
    # m..1 (default 0.9..1) in place of the logit-normal draw. Reason: with the JiT recipe only 0.025% of the inputs have
    # a noise level above 0.9, and the model's prediction from pure noise, where every sampler starts, was far too dark
    # (brightness 0.13 against 0.44 of real images).
    T_PURE, T_PURE_MIN = float(T_DIST.get("pure_share", 0.0)), float(T_DIST.get("pure_min", 0.9))
    assert 0.0 <= T_PURE < 1.0 and 0.0 < T_PURE_MIN < 1.0
    # t_dist "kind": "uniform" (user, 2026-10-14): t = rand, every noise level equally often (mean and std are not used).
    # The 100M self-flow run tests it against the JiT recipe as a continuation: the self-flow step draws its two t with
    # draw_t as well (t_dist works there; pure_share and noise_mean_std do not).
    T_UNIFORM = T_DIST.get("kind", "logit_normal") == "uniform"
    assert T_DIST.get("kind", "logit_normal") in ("logit_normal", "uniform")

    def draw_t(*shape, device=None, generator=None):
        # generator only when given: torch.rand(..., generator=None) in a compiled step fails once the batch size
        # changes and dynamo recompiles with a variable size (both schedule tests crashed at step 670,000, 2026-10-15)
        kw = {"generator": generator} if generator is not None else {}
        if T_UNIFORM:
            return torch.rand(*shape, device=device, dtype=torch.float32, **kw)
        return torch.sigmoid(torch.randn(*shape, device=device, dtype=torch.float32, **kw) * T_STD + T_MEAN)

    NOISE_MEAN_STD = float(cfg.get("noise_mean_std", 0) or 0)
    assert not (NOISE_MEAN_STD and (cfg.get("xm_ot") or cfg.get("selfflow") or cfg.get("distill"))), "noise_mean_std: plain train step only"
    assert not (T_DIST and (cfg.get("distill") or (cfg.get("xm_ot") and not cfg.get("selfflow")))), "t_dist: plain or self-flow train step only"
    assert not (T_PURE and cfg.get("selfflow")), "t_dist pure_share: plain train step only"
    T_SHIFT = cfg.get("t_shift", 1.0)
    T_SHIFT = math.sqrt(3 * int(g["IMAGE_SIZE"]) ** 2 / 4096) if T_SHIFT == "auto" else float(T_SHIFT)
    assert T_SHIFT > 0, T_SHIFT

    def shift_t(t):
        if T_SHIFT == 1.0:
            return t
        level = 1.0 - t
        return 1.0 - T_SHIFT * level / (1.0 + (T_SHIFT - 1.0) * level)

    # cfg["loss_weight"] (user, 2026-10-09): the weight on (x_pred - x)^2. The time schedule does not change.
    #   "log"     = log1p(1 / t^2) / log(2), the smooth weight of broad15 until now (default: old runs resume unchanged)
    #   "clamp01" = 1 / max(noise level, 0.1)^2 = min(1 / noise level^2, 100): the velocity MSE with a hard clamp.
    #   "logflip" = log1p(1 / noise level^2) / log(2): the smooth weight with the direction of "clamp01". Test of
    #               whether "log" had the direction of t backwards (it weights nearly pure noise most).
    # The noise level is 1 - t in this code (t = 1 clean), so "clamp01" weights nearly clean inputs most.
    LOSS_WEIGHT = cfg.get("loss_weight", "log")
    assert LOSS_WEIGHT in ("log", "clamp01", "logflip"), LOSS_WEIGHT

    def broad_weight(t):
        if LOSS_WEIGHT == "clamp01":
            return (1.0 - t).clamp_min(0.1).square().reciprocal()
        if LOSS_WEIGHT == "logflip":
            return torch.log1p(1.0 / (1.0 - t).square()) / math.log(2.0)
        # Smooth low-t weighting for direct x-prediction.
        return torch.log1p(1.0 / t.square()) / math.log(2.0)

    # CHECK: the weight function is the configured one, at a nearly clean, a middle and a nearly pure-noise input
    _w = broad_weight(torch.tensor([0.99, 0.5, 0.02])).tolist()
    _want = ([100.0, 4.0, 1.0 / 0.98 ** 2] if LOSS_WEIGHT == "clamp01" else
             [math.log1p(1 / v ** 2) / math.log(2.0) for v in ((0.01, 0.5, 0.98) if LOSS_WEIGHT == "logflip" else (0.99, 0.5, 0.02))])
    assert all(abs(a_ - b_) < 1e-3 * b_ for a_, b_ in zip(_w, _want)), (_w, _want)
    print(f"CHECK loss weight {LOSS_WEIGHT}: weight at noise level 0.01 / 0.5 / 0.98 = {_w[0]:.2f} / {_w[1]:.2f} / {_w[2]:.2f}", flush=True)

    def broad_losses(outputs, images, noised_images, t_img):
        # one loss per output; a low-resolution readout is scored against the clean image resized to its size
        weight = broad_weight(t_img)
        losses = []
        for x_pred in outputs:
            if x_pred.shape[-2:] == images.shape[-2:]:
                target = images
            elif x_pred.shape[-1] < 16:
                # 1 to 8 px: the mean of each cell. The antialiased resize fails on CUDA at these scale factors (2026-10-11)
                target = F_.avg_pool2d(images, images.shape[-1] // x_pred.shape[-1])
            else:
                target = F_.interpolate(images, size=x_pred.shape[-2:], mode="bilinear", antialias=True)
            losses.append(((x_pred.float() - target.float()).square() * weight).mean())
        return losses

    def broad_split_loss(images, t, noise):
        # per image, for get_split_loss: the same loss as the train step
        t_img = t[:, None, None, None]
        noised_images = t_img * images + (1.0 - t_img) * noise
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x_pred = model(noised_images, t)
        return (x_pred.float() - images.float()).square().mean((1, 2, 3)) * broad_weight(t)

    def make_broad_train_step(pyramid):
        weight_of_readouts = float(g.get("pyramid_weight", 1.0))
        # cfg["xm_ot"] = {"seeds": 2} (user, 2026-10-11): XM with transport. The step draws 2 noise samples and uses the
        # same 2 for every image. Every image runs with both (2 x B rows, with gradient). Half of the images are then
        # assigned to seed 1 and half to seed 2 at the lowest total weighted loss of the final output (for 2 seeds: sort
        # by the loss difference). Only the B chosen rows enter the loss. Plain XM would let every image take its best seed.
        XMOT = int((cfg.get("xm_ot") or {}).get("seeds", 0))
        # 3 and 4 seeds (2026-10-12): the B images go to B slots, slot j belongs to seed j % seeds (64 images, 3 seeds:
        # 22 + 21 + 21). The Hungarian algorithm (scipy, on the CPU) gives the assignment with the lowest total loss.
        assert XMOT in (0, 2, 3, 4), XMOT
        # "assign": "random" (2026-10-12) = control: the same shared seeds and the same balanced split, but by chance.
        XM_RANDOM = (cfg.get("xm_ot") or {}).get("assign", "loss") == "random"
        # "assign": "best" (user, 2026-10-12) = normal XM, the comparison for the transport version: every image has its
        # own K noise samples and keeps the one with the lowest loss. No sharing and no balance.
        XM_BEST = (cfg.get("xm_ot") or {}).get("assign", "loss") == "best"
        # cfg["distill2"] = {"weight": 1.0, "second": "target" | "train", "min_noise": 0.0} (user, 2026-10-12): continuous
        # 2-step distillation. Every step: the 1-step output x_a of the noised input (time t) is taken half of the way with
        # the Euler rule, x_mid = (input + x_a) / 2 at time (1 + t) / 2, and the model runs again from there: x_b, the
        # 2-step output. x_b (no gradient) is a second target of the 1-step output, beside the clean image.
        #   "second": "train"  the second pass has a gradient and its own loss against the clean image (the model learns
        #                      from an input it built itself). Default "target": the second pass is a target only.
        #   "min_noise"        only the images with a noise level above this value get the second target.
        #   "link"             what ties the 1-step output to the 2-step output (user, 2026-10-12 late). "target" (default):
        #                      the 2-step output is a target of the 1-step output. "mutual": the two attract each other
        #                      (gradient on both; needs "second": "train"). "none": no tie, weight 0.
        #   The input of the second pass is detached: the first pass gets no gradient through it.
        #   "second_weight"    "mid" (default): the clean-image loss of the second pass has the loss weight of its own time
        #                      (1 + t) / 2. "first": the loss weight of the first time t (2026-10-13: with "mid" the second
        #                      pass had 4 times the loss of the first pass and the FID at 4 sampler steps got worse).
        D2 = cfg.get("distill2") or {}
        D2_W, D2_TRAIN, D2_MIN = float(D2.get("weight", 1.0)), D2.get("second", "target") == "train", float(D2.get("min_noise", 0.0))
        D2_LINK = D2.get("link", "target")
        D2_W2 = D2.get("second_weight", "mid")
        assert D2_W2 in ("mid", "first")
        assert D2.get("second", "target") in ("target", "train") and not (D2 and XMOT) and D2_LINK in ("target", "mutual", "none")
        assert D2_LINK != "mutual" or D2_TRAIN, 'link "mutual" needs "second": "train"'
        if D2_LINK == "none":
            D2_W = 0.0
        # "groups": true (user, 2026-10-11: transport inside groups of the batch): the batch is cut into groups of K
        # images. Each group has its own K noise samples, and the K images of a group share them out 1-1 at the lowest
        # total loss. Every step then uses B different noise samples (not K), each of them once.
        XM_GROUPS = bool((cfg.get("xm_ot") or {}).get("groups", False))
        if XM_GROUPS:
            from scipy.optimize import linear_sum_assignment as xm_assign
        # cfg["noise_ot"] = True (2026-10-12): minibatch transport on the inputs. Each step draws B noise samples and pairs them
        # 1-1 with the B images at the lowest total squared pixel distance (Hungarian algorithm). No extra forward pass.
        NOISE_OT = bool(cfg.get("noise_ot"))
        # cfg["noise_ot"] = {"pool": 16}: the distance is taken between the images and the noise after an average pool
        # by that factor (16 at 128 px: 8 x 8 x 3 = 192 numbers), so the pairing follows the coarse structure only.
        NX_POOL = int(cfg["noise_ot"].get("pool", 1)) if isinstance(cfg.get("noise_ot"), dict) else 1
        assert not (NOISE_OT and XMOT)
        if NOISE_OT:
            from scipy.optimize import linear_sum_assignment as xm_assign
        assert (cfg.get("xm_ot") or {}).get("assign", "loss") in ("loss", "random", "best") and not (XM_BEST and XM_GROUPS)
        if XMOT > 2:
            from scipy.optimize import linear_sum_assignment as xm_assign
        # cfg["selfteach"] (2026-10-15): the self-teaching term of the self-flow step, on this step's noised input and final output
        st = make_selfteach() if SELFTEACH else None
        if PAIR:
            assert not (T_UNIFORM or T_PURE or XMOT or NOISE_MEAN_STD or NOISE_OT or D2), "pair: the plain broad15 step only"

        def train_step(images, optimizer, pair=None):
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device, non_blocking=True)
            B = images.shape[0]

            if pair is not None:
                # cfg pair: t and noise drawn before the step from the run's own generator (PAIR_GEN), the same in every variant
                t = shift_t(torch.sigmoid(pair["u"] * T_STD + T_MEAN))
            else:
                t = shift_t(draw_t(B, device=images.device))
            if T_PURE:
                pure_level = T_PURE_MIN + (1.0 - T_PURE_MIN) * torch.rand(B, device=images.device, dtype=torch.float32)
                t = torch.where(torch.rand(B, device=images.device) < T_PURE, 1.0 - pure_level, t)
            t_img = t[:, None, None, None]
            xm_all = None

            if XMOT:
                if XM_BEST:
                    seeds_per_image = torch.randn(XMOT, *images.shape, device=images.device, dtype=images.dtype)
                    noised_all = ((t_img * images)[None] + (1.0 - t_img)[None] * seeds_per_image).flatten(0, 1)
                elif XM_GROUPS:
                    # a last group with fewer than K images (the short batch at the end of a data pass) still has K noise samples
                    n_groups = -(-B // XMOT)
                    group_noise = torch.randn(n_groups * XMOT, *images.shape[1:], device=images.device, dtype=images.dtype)   # noise g * K + k = sample k of group g
                    group_of = torch.arange(B, device=images.device) // XMOT
                    seeds_per_image = group_noise[group_of[None] * XMOT + torch.arange(XMOT, device=images.device)[:, None]]    # (K, B, ...)
                    noised_all = ((t_img * images)[None] + (1.0 - t_img)[None] * seeds_per_image).flatten(0, 1)
                else:
                    seeds = torch.randn(XMOT, *images.shape[1:], device=images.device, dtype=images.dtype)
                    noised_all = ((t_img * images)[None] + (1.0 - t_img)[None] * seeds[:, None]).flatten(0, 1)   # row k * B + i
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outs_all = model(noised_all, t.repeat(XMOT), all_readouts=True) if pyramid else [model(noised_all, t.repeat(XMOT))]
                with torch.no_grad():
                    err_all = (outs_all[-1].detach().float().reshape(XMOT, B, -1) - images.float().flatten(1)[None]).square().mean(-1)   # (2, B)
                    cost = err_all * broad_weight(t)[None]
                    cost_true = cost
                    if XM_RANDOM:
                        cost = torch.rand_like(cost)
                    if XM_BEST:
                        pick = cost.argmin(0)
                    elif XM_GROUPS:
                        # (image, noise) cost, 1e9 outside the group of the image: the assignment is 1-1 inside every group
                        full = torch.full((B, n_groups * XMOT), 1e9, device=images.device)
                        full[torch.arange(B, device=images.device)[None].expand(XMOT, B), group_of[None] * XMOT + torch.arange(XMOT, device=images.device)[:, None]] = cost
                        _, col = xm_assign(full.cpu().numpy())
                        col = torch.as_tensor(col, device=images.device)
                        assert bool((col // XMOT == group_of).all())
                        pick = col % XMOT
                    elif XMOT == 2:
                        order = (cost[0] - cost[1]).argsort()           # first half: the images that gain most from seed 1
                        pick = torch.zeros(B, device=images.device, dtype=torch.long)
                        pick[order[B // 2:]] = 1
                    else:
                        slot_seed = torch.arange(B, device=images.device) % XMOT
                        _, slot = xm_assign(cost.t()[:, slot_seed].cpu().numpy())      # (image, slot)
                        pick = torch.as_tensor(slot, device=images.device) % XMOT
                    sel = pick * B + torch.arange(B, device=images.device)
                    xm_all = cost_true.mean()                                # weighted loss of a random assignment
                outputs = [o[sel] for o in outs_all]
                noised_images = noised_all[sel]
            else:
                noise = pair["noise"].to(images.dtype) if pair is not None else torch.randn_like(images)
                if NOISE_MEAN_STD:
                    noise = noise + NOISE_MEAN_STD * torch.randn(B, 1, 1, 1, device=images.device, dtype=images.dtype)
                noise_dc = noise.float().mean((1, 2, 3)).std()
                if NOISE_OT:
                    with torch.no_grad():
                        nx_a, nx_b = (images.float(), noise.float()) if NX_POOL == 1 else (
                            F_.avg_pool2d(images.float(), NX_POOL), F_.avg_pool2d(noise.float(), NX_POOL) * NX_POOL)       # noise back to std 1
                        dist = torch.cdist(nx_a.flatten(1), nx_b.flatten(1)).square() / nx_a[0].numel()   # (image, noise)
                        _, col = xm_assign(dist.cpu().numpy())
                        col = torch.as_tensor(col, device=images.device)
                        nx = (dist.gather(1, col[:, None]).mean(), dist.diagonal().mean(), (col != torch.arange(B, device=images.device)).float().mean())
                    noise = noise[col]
                noised_images = t_img * images + (1.0 - t_img) * noise

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if not XMOT:
                    outputs = model(noised_images, t, all_readouts=True) if pyramid else [model(noised_images, t)]
                losses = broad_losses(outputs, images, noised_images, t_img)
                loss_final = losses[-1]
                loss = loss_final
                if D2:
                    t_mid = 0.5 * (1.0 + t)
                    x_mid = (0.5 * (noised_images.float() + outputs[-1].detach().float())).to(noised_images.dtype)
                    with torch.set_grad_enabled(D2_TRAIN):
                        outs_b = model(x_mid, t_mid, all_readouts=True) if pyramid else [model(x_mid, t_mid)]
                    x_b = outs_b[-1].detach().float()
                    d2_on = ((1.0 - t) >= D2_MIN).float()
                    d2_gap = (outputs[-1].float() - (outs_b[-1].float() if D2_LINK == "mutual" else x_b)).square().mean((1, 2, 3))   # per image
                    d2_loss = (d2_gap * broad_weight(t) * d2_on).mean()
                    loss = loss + D2_W * d2_loss
                    if D2_TRAIN:
                        losses_b = broad_losses(outs_b, images, x_mid, (t if D2_W2 == "first" else t_mid)[:, None, None, None])
                        loss = loss + losses_b[-1]
                        for readout_loss in losses_b[:-1]:
                            loss = loss + weight_of_readouts * readout_loss
                for readout_loss in losses[:-1]:
                    loss = loss + weight_of_readouts * readout_loss
                if st is not None:
                    st_loss, st_out, st_noised = st["terms"](images, noised_images, t, outputs[-1], pair=pair)
                    loss = loss + st_loss

            loss.backward()
            optimizer.step()
            # per image, for the train loss by noise level: the noise level that built the input, and the plain x-MSE
            out = {"loss": loss.detach(), "loss_final": loss_final.detach(), "noise_level": 1.0 - t,
                   "x_err": (outputs[-1].detach().float() - images).square().mean((1, 2, 3))}
            if pyramid:
                out["readout_losses"] = torch.stack(losses[:-1]).detach()
            if CORR:
                # plain x-MSE of the base denoiser and of the corrector on the train batch (the losses are weighted)
                out["corr_base_xmse"] = (outputs[0].detach().float() - images).square().mean()
                out["corr_final_xmse"] = (outputs[-1].detach().float() - images).square().mean()
            if NOISE_MEAN_STD:
                out["noise_dc_std"] = noise_dc.detach()
            if NOISE_OT:
                out["nx_dist"], out["nx_dist_fixed"], out["nx_moved"] = nx
            if D2 and D2_TRAIN:
                out["d2_loss_b"] = losses_b[-1].detach()      # weighted clean-image loss of the second pass
            if D2:
                # d2_gap: mse between the 1-step and the 2-step output. d2_err_b: mse of the 2-step output to the clean image
                out["d2_loss"], out["d2_gap"], out["d2_share"] = d2_loss.detach(), d2_gap.detach().mean(), d2_on.mean()
                out["d2_err_b"] = (x_b - images.float()).square().mean()
                out["d2_err_a"] = (outputs[-1].detach().float() - images.float()).square().mean()
            if XMOT:
                out["xm_loss_all"], out["x_err_all"], out["xm_share1"] = xm_all, err_all.mean(0), (pick == 1).float().mean()
            if st is not None:
                out["st_out"], out["st_noised"] = st_out.detach(), st_noised.detach()
            return out

        return train_step

    g["sample_train_t"], g["x_pred_v_loss"], g["readout_losses"] = broad_t, broad_split_loss, broad_losses
    assert XM_K == 1 and not LOOP, "the broad schedule is wired for the plain and the pyramid model"

    # ---- hidden-state teaching (user, 2026-10-07). Both need the pyramid model and this schedule. ----
    # cfg["distill"] = {"from": 6, "to": 12, "weight": 1.0}: layer distillation in one forward pass:
    #     + weight * mse(h after block `from`, (h after block `to`).detach())
    # cfg["selfflow"] = {"from": 6, "to": 12, "weight": 1.0, "noisy_share": 0.25}: self-flow teaching.
    #     Two noise levels are drawn per image. The teacher (the same weights, no gradient) sees the whole image
    #     at the lower noise level. The student sees 75% of the patches at that level and 25% at the higher one,
    #     with the time of every patch given per token. Loss: the student's flow loss (per-patch weight), plus
    #     weight * mse(student h after block `from`, teacher h after block `to`).
    DISTILL, SELFFLOW = cfg.get("distill") or {}, cfg.get("selfflow") or {}
    assert not (DISTILL and SELFFLOW) and (PYR or not (DISTILL or SELFFLOW))
    # cfg["msa"] (user, 2026-10-15): multi-scale attention separation, three forwards per step (job/msa.py). It counts as
    # teaching: "weight" is the alignment weight, test_teach the same-input distance of block 3 to block 6
    MSA = cfg.get("msa") or {}
    assert not (MSA and (DISTILL or SELFFLOW or cfg.get("xm_ot") or cfg.get("wreg"))) and (PYR or not MSA)      # selfteach: yes (2026-10-15)
    TEACH = DISTILL or SELFFLOW or MSA
    # cfg["distill"]["pairs"] = [[student, target], ...] (user, 2026-10-08): several pairs in one forward pass, the
    # loss is the sum over the pairs. Block 0 is the patch embedding (after norm_in), block num_blocks the last block.
    # [[0, 6], [6, 12], [12, 16]] is the chain over the whole depth; [[n, n + 1] for every n] the layer-by-layer form.
    PAIRS = [(int(s), int(t)) for s, t in (TEACH.get("pairs") or ([[TEACH["from"], TEACH["to"]]] if TEACH and not MSA else [[3, 6]] if MSA else []))]
    assert all(0 <= s < t <= int(g["num_blocks"]) for s, t in PAIRS) , PAIRS
    PAIR_TAPS = tuple(sorted({b for pair in PAIRS for b in pair}))

    def block_forward(block, x, rope, time_emb):
        # The notebook's Block.forward, also for a time embedding per token: (B, C) or (B, T, C).
        ssg = block.scale_shift_gate(time_emb)
        if ssg.ndim == 2:
            ssg = ssg[:, None, :]
        scale_a, scale_m, shift_a, shift_m, gate_a, gate_m = ssg.chunk(6, dim=-1)
        x = x + gate_a * block.post_norm1(block.MHA(block.pre_norm1(x) * (1 + scale_a) + shift_a, rope))
        x = x + gate_m * block.post_norm2(block.SwiGLU(block.pre_norm2(x) * (1 + scale_m) + shift_m))
        return x

    def forward_taps(images, time, taps=(), upto=None, rope=None, net=None):
        """PyramidImageTransformer.forward with the hidden states after the blocks in `taps` (1-based).
        time: (B,) or, per patch token, (B, T). upto: stop after that block and return only the taps.
        Returns (list of output images [readouts..., final], {block: hidden state})."""
        m = net if net is not None else getattr(model, "_orig_mod", model)
        core = m.core_transformer
        MOE_CTX["t"] = time
        rows, cols = images.shape[-2] // m.patch_size, images.shape[-1] // m.patch_size
        x = core.norm_in(m.embed(images))
        if time.ndim == 2:
            time_emb = m.timestep_embedder(time.reshape(-1)).reshape(time.shape[0], time.shape[1], -1)
        else:
            time_emb = m.timestep_embedder(time)
        if rope is None:
            rope = core.rope_tables(rows, cols)
        outputs, hidden = [], {}
        if 0 in taps:
            hidden[0] = x
        for i, block in enumerate(core.blocks):
            x = block_forward(block, x, rope, time_emb)
            if i + 1 in taps:
                hidden[i + 1] = x
            if upto is not None and i + 1 >= upto:
                return outputs, hidden
            if str(i + 1) in m.readout_heads:
                outputs.append(m.readout_heads[str(i + 1)](x, rows, cols))
        outputs.append(torch.sigmoid(m.un_embed(core.norm_out(x), rows, cols)))
        return outputs, hidden

    def weight_maps(t_tok, rows, cols, outputs):
        # the loss weight of every pixel of every output, from the time of its patch
        w = broad_weight(t_tok).reshape(-1, 1, rows, cols)
        return [w.repeat_interleave(o.shape[-2] // rows, 2).repeat_interleave(o.shape[-1] // cols, 3) for o in outputs]

    def teach_distance(student, target, mask=None):
        # mse between the hidden states. cfg teach "normalize" (default true): every token is first scaled to
        # RMS 1, so only the direction is compared (= 2 - 2 cos). The raw mse of the first run (2026-10-08) ran
        # away within 500 steps: block `from` grows to reach block `to`, which then grows too (teach loss 0.3 -> 1e7).
        s, t = student.float(), target.float()
        if TEACH.get("normalize", True):
            s = s / s.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            t = t / t.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        if mask is None:
            return F_.mse_loss(s, t)
        # only the tokens of `mask` (B, T) count
        per_token = (s - t).square().mean(-1)
        return (per_token * mask).sum() / mask.sum().clamp_min(1.0)

    # cfg["selfteach"] = {"layer": 6, "weight_out": 1.0, "weight_noised": 1.0, "crop_min": 0.2, "crop_max": 0.9}
    # (user, 2026-10-10): self-teaching as in the AE notebook, with the model as its own judge. Needs cfg selfflow.
    #   One random crop per image (the AE notebook's crop: scale 0.2 to 0.9 of the side, up to half of it off the image,
    #   black outside) is taken from the clean image, from the noised image (lower noise level, whole image) and from the
    #   model's final output image, and scaled back to the full size.
    #   z = the residual stream after block `layer` (8 = the last block, before the un-embed).
    #   target    = z of the clean crop at t = 1, frozen copy of the weights, no gradient
    #   out term  = distance(z of the output crop at t = 1, frozen copy, gradient through the image into the model, target)
    #   noised term = distance(z of the noised crop at its t, target). "noised_net": "live" (default): through the trained
    #                 weights, so the term has a gradient. "frozen": through the frozen copy (the noised image does not
    #                 depend on a weight, so this term then trains nothing and is only logged).
    #   The frozen copy takes the trained weights at the start of every step (the AE notebook's Ef).
    # 2026-10-15 (user: "self-teaching (frozen first 3 layers as perceptual judge)" on the best run of the msa series):
    # also in the plain pyramid step (broad15, no teaching) and in the msa step, through make_selfteach below. There the
    # noised image is the step's own noised input at its t (msa: the dense forward's input), the output its final output
    # (msa: the dense forward's). The judge is the last-step weights only ("ema" needs the self-flow averaged teacher).
    SELFTEACH = cfg.get("selfteach") or {}
    assert not SELFTEACH or PYR, "selfteach is wired into the pyramid steps (self-flow, plain broad15, msa)"
    assert not SELFTEACH or SELFFLOW or SELFTEACH.get("judge", "live") == "live", 'selfteach judge "ema" needs selfflow teacher "ema"'

    def st_crop_grid(batch, size, crop_min, crop_max, dev, overhang=0.5):
        fraction = torch.empty(batch, 1, device=dev, dtype=torch.float32).uniform_(crop_min, crop_max)
        half_span = fraction * torch.ones(2, device=dev)
        center = (torch.rand(batch, 2, device=dev) * 2 - 1) * (1.0 - half_span * (1 - 2 * overhang))
        axis = torch.linspace(-1, 1, size, device=dev)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        return torch.stack((xx, yy), dim=-1)[None] * half_span[:, None, None, :] + center[:, None, None, :]

    def st_crop_grid_from(u, size, crop_min, crop_max, overhang=0.5):
        # st_crop_grid with its random numbers given: u (B, 3) uniform in [0, 1): column 0 the crop size, 1-2 the center
        fraction = crop_min + (crop_max - crop_min) * u[:, :1].float()
        half_span = fraction * torch.ones(2, device=u.device)
        center = (u[:, 1:3].float() * 2 - 1) * (1.0 - half_span * (1 - 2 * overhang))
        axis = torch.linspace(-1, 1, size, device=u.device)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        return torch.stack((xx, yy), dim=-1)[None] * half_span[:, None, None, :] + center[:, None, None, :]

    def st_crop_scale(images, grid):
        with torch.autocast(device_type="cuda", enabled=False):
            crops = F_.grid_sample(images.float(), grid, mode="bilinear", padding_mode="border", align_corners=True)
        inside = (grid.abs() <= 1 + 1.0 / (images.shape[-1] - 1)).all(-1)[:, None]
        return crops * inside.to(crops.dtype)

    def make_selfteach():
        """The self-teaching term for any of the three steps. Makes the frozen copy once (TEACHER["st_frozen"]).
        Returns st with st["terms"](images, noised, t_noised, output, rope=None) -> (weighted sum, st_out, st_noised):
        syncs the frozen copy from st["pairs"] (frozen, source), then the two terms of the comment above.
        noised: the noised whole image, t_noised (B,) its t. output: the final output image. The self-flow step points
        st["pairs"] at its averaged teacher for judge "ema"."""
        import copy as _copy
        layers = tuple(sorted(int(l_) for l_ in (SELFTEACH.get("layers") or [SELFTEACH.get("layer", 6)])))
        layer, overhang = layers[-1], float(SELFTEACH.get("overhang", 0.5))
        judge_t = float(SELFTEACH.get("judge_t", 1.0))
        w_out, w_noised = float(SELFTEACH.get("weight_out", 1.0)), float(SELFTEACH.get("weight_noised", 1.0))
        live = SELFTEACH.get("noised_net", "live") == "live"
        crop_min, crop_max = float(SELFTEACH.get("crop_min", 0.2)), float(SELFTEACH.get("crop_max", 0.9))
        assert 0.0 < judge_t <= 1.0 and 1 <= layers[0] and layer <= int(g["num_blocks"]) and SELFTEACH.get("noised_net", "live") in ("live", "frozen")
        assert SELFTEACH.get("judge", "live") in ("live", "ema") and 0.0 <= overhang <= 0.5
        src = getattr(model, "_orig_mod", model)
        frozen = _copy.deepcopy(src).requires_grad_(False)
        st = {"layers": layers, "frozen": frozen, "w_out": w_out, "w_noised": w_noised,
              "pairs": tuple(zip(frozen.parameters(), src.parameters())) + tuple(zip(frozen.buffers(), src.buffers()))}
        TEACHER["st_frozen"], TEACHER["st"] = frozen, st

        def dist(z_student, z_goal, rows=slice(None)):
            return sum(teach_distance(z_student[l_][rows], z_goal[l_]) for l_ in layers) / len(layers)

        def sync():
            with torch.no_grad():
                for frozen_tensor, live_tensor in st["pairs"]:
                    frozen_tensor.copy_(live_tensor)

        def terms(images, noised, t_noised, output, rope=None, pair=None):
            B = images.shape[0]
            st_out = st_noised = torch.zeros((), device=images.device)
            sync()
            m = getattr(model, "_orig_mod", model)
            rows_, cols_ = images.shape[-2] // m.patch_size, images.shape[-1] // m.patch_size
            if pair is not None:
                # cfg pair (scaling sweep, 2026-10-21): the crops and the rope scale of the judge passes come from the
                # step's pre-drawn numbers (pair["crop"] (B, 3) uniform, pair["st_log"] the log rope scale), so this term
                # draws nothing from the global random stream and the paired baseline keeps the same noise stream
                assert judge_t == 1.0, "pair: judge_t < 1 would draw judge noise in the step"
                if rope is None:                                                       # the self-flow step passes its own table
                    rope = m.core_transformer.rope_tables(rows_, cols_, log_scale=pair["st_log"])
                grid = st_crop_grid_from(pair["crop"], images.shape[-1], crop_min, crop_max, overhang=overhang)
            else:
                if rope is None:                                                       # one rope table (one random scale) for the passes
                    rope = m.core_transformer.rope_tables(rows_, cols_)
                grid = st_crop_grid(B, images.shape[-1], crop_min, crop_max, images.device, overhang=overhang)
            crops = st_crop_scale(torch.cat([images, output.float(), noised.float()]), grid.repeat(3, 1, 1, 1))
            crop_clean, crop_out, crop_noised = crops.chunk(3)
            one = torch.ones(2 * B, device=images.device, dtype=torch.float32) * judge_t
            if judge_t < 1.0:
                judge_noise = torch.randn_like(crop_clean) * (1.0 - judge_t)
                crop_clean, crop_out = judge_t * crop_clean + judge_noise, judge_t * crop_out + judge_noise
            # clean and output crops in one pass of the frozen copy. Only the output half carries a gradient
            if w_out > 0:
                z_two = forward_taps(torch.cat([crop_clean.detach(), crop_out]), one, taps=layers, upto=layer, rope=rope, net=frozen)[1]
                z_target = {l_: z_two[l_][:B].detach() for l_ in layers}
                st_out = dist(z_two, z_target, slice(B, None))
            else:
                with torch.no_grad():
                    z_target = forward_taps(crop_clean, one[:B], taps=layers, upto=layer, rope=rope, net=frozen)[1]
            if w_noised > 0:
                z_noised = forward_taps(crop_noised.detach(), t_noised, taps=layers, upto=layer, rope=rope, net=None if live else frozen)[1]
                st_noised = dist(z_noised, z_target)
            return w_out * st_out + w_noised * st_noised, st_out, st_noised

        st["terms"], st["sync"] = terms, sync
        return st

    def make_teach_train_step():
        weight_of_readouts = float(g.get("pyramid_weight", 1.0))
        frm, to = int(TEACH["from"]), int(TEACH["to"])
        st_judge_ema = SELFTEACH.get("judge", "live") == "ema"
        st = make_selfteach() if SELFTEACH else None
        w_teach = float(TEACH.get("weight", 1.0))
        noisy_share = float(SELFFLOW.get("noisy_share", 0.25)) if SELFFLOW else 0.0
        # selfflow "teacher": false = mixed noise only: the student input of self-flow (a share of the patches at a
        # higher noise level, per-token time) without the teacher pass and without the teaching term. One forward pass.
        use_teacher = bool(SELFFLOW.get("teacher", True)) if SELFFLOW else False
        # Options for the research loop (2026-10-09). None adds a parameter to the model.
        #   "teacher": "ema"          the teacher pass uses averaged weights (decay "teacher_decay", default 0.999), as in JEPA
        #   "noisy_t": "mask"         the noisier patches of the student are pure noise (t = 0): a masked student
        #   "teacher_input": "clean"  the teacher sees the clean image (t = 1), not the image at the lower noise level
        #   "ctx_weight": w           weight of the denoising loss on the cleaner (context) patches of the student. 0 = they are
        #                             trained only through what they give the noisier patches and through the teaching term
        #   "pairs": [[s, t], ...]    several student <- target block pairs
        #   "noisy_layout": "block"   the noisier patches form one rectangle per image (wraps at the borders), as the
        #                             target blocks of I-JEPA. Default "random": spread over the image
        #   "teach_on": "noisy"       the teaching term counts only the noisier patches (JEPA: predict the masked part)
        #   "out_weight": w           output alignment of the two views: + w * weighted mse(student final image, teacher final
        #                             image). The teacher then runs through all blocks (a longer step). 0 = off
        out_weight = float(SELFFLOW.get("out_weight", 0.0))
        block_layout = SELFFLOW.get("noisy_layout", "random") == "block"
        teach_noisy_only = SELFFLOW.get("teach_on", "all") == "noisy"
        assert SELFFLOW.get("noisy_layout", "random") in ("random", "block") and SELFFLOW.get("teach_on", "all") in ("all", "noisy")
        mask_student = SELFFLOW.get("noisy_t") == "mask"
        teacher_clean = SELFFLOW.get("teacher_input") == "clean"
        ctx_weight = float(SELFFLOW.get("ctx_weight", 1.0))
        assert SELFFLOW.get("noisy_t", "lower") in ("lower", "mask") and SELFFLOW.get("teacher_input", "lower") in ("lower", "clean")
        # cfg["wreg"] = {"block": 6, "weight": 0.3} (user, 2026-10-11): the family of 4 Wasserstein regularizers of the AE
        # notebook (ImageNet-AE-selfteach: sliced, basis, channel-sliced, token-sliced; each returns its loss squared) on
        # the residual stream after that block of the student pass, against a fresh N(0, 1) prior. weight = each of the 4.
        WREG = cfg.get("wreg")
        wreg_block, wreg_w = (int(WREG.get("block", 6)), float(WREG.get("weight", 0.3))) if WREG else (None, 0.0)

        def wreg_sorted_loss(source, target, dim, bulk_sigma=1.0):
            # squared error between sorted values, weighted to the bulk of the prior (softmax of -0.5 (prior / sigma)^2)
            squared_error = (source - target).square()
            weights = torch.softmax(-0.5 * (target / bulk_sigma).square(), dim=dim)
            return (weights * squared_error).sum(dim=dim).mean()

        def wreg_sliced(z, prior, n_slices):
            # z, prior: (N, D). Random unit directions in D, sorted over the N samples
            slices = F_.normalize(torch.randn(z.shape[1], n_slices, device=z.device, dtype=z.dtype), dim=0)
            return wreg_sorted_loss(torch.sort(z @ slices, dim=0).values, torch.sort(prior @ slices, dim=0).values, 0).square()

        def wreg_family(z):
            """z: (B, T, C) float32. Returns the 4 terms: swd (flat), bwd (basis, across images), c_swd, t_swd."""
            B_, T_, C_ = z.shape
            prior = torch.randn_like(z)
            swd = wreg_sliced(z.flatten(1), prior.flatten(1), 1024)
            bwd = wreg_sorted_loss(torch.sort(z, dim=0).values, torch.sort(prior, dim=0).values, 0).square()
            c_swd = wreg_sliced(z.reshape(B_ * T_, C_), prior.reshape(B_ * T_, C_), 32)
            t_swd = wreg_sliced(z.permute(0, 2, 1).reshape(B_ * C_, T_), prior.permute(0, 2, 1).reshape(B_ * C_, T_), 32)
            return torch.stack([swd, bwd, c_swd, t_swd])

        student_taps = tuple(sorted({s_ for s_, _ in PAIRS} | ({wreg_block} if WREG else set())))
        target_taps = tuple(sorted({t_ for _, t_ in PAIRS}))
        teacher_net = None
        if SELFFLOW.get("teacher") == "ema":
            import copy
            teacher_net = copy.deepcopy(getattr(model, "_orig_mod", model)).requires_grad_(False)
            TEACHER["net"], TEACHER["decay"] = teacher_net, float(SELFFLOW.get("teacher_decay", 0.999))
        if st_judge_ema:
            assert teacher_net is not None, 'selfteach judge "ema" needs selfflow teacher "ema"'
            st["pairs"] = tuple(zip(st["frozen"].parameters(), teacher_net.parameters())) + tuple(zip(st["frozen"].buffers(), teacher_net.buffers()))

        # cfg["xm_ot"] with self-flow (2026-10-12): XM transport as a choice of the noise. A pass without gradient scores
        # the student input of every image with each of the K shared noise samples. The Hungarian algorithm gives each
        # image one sample (balanced, lowest total weighted final loss). The step then runs as usual with that noise.
        XMT = int((cfg.get("xm_ot") or {}).get("seeds", 0))
        assert not XMT or (SELFFLOW and ctx_weight == 1.0 and (cfg.get("xm_ot") or {}).get("assign", "loss") in ("loss", "best"))
        XMT_BEST = (cfg.get("xm_ot") or {}).get("assign", "loss") == "best"
        # "score": "readout" = the pass without gradient stops after the block of the first readout (block 3) and scores
        # the candidates by the loss of that readout (R32). Default "final": all blocks, loss of the final output.
        XMT_READOUT = (cfg.get("xm_ot") or {}).get("score", "final") == "readout"
        # "min_noise": 0.5 = only the images with a mean noise level above 0.5 count in the assignment. The cost of the
        # other images is set to 0, so they fill the free places of the seeds without a choice.
        XMT_MIN_NOISE = float((cfg.get("xm_ot") or {}).get("min_noise", 0.0))
        assert (cfg.get("xm_ot") or {}).get("score", "final") in ("final", "readout")
        if XMT:
            from scipy.optimize import linear_sum_assignment as xmt_assign

        if PAIR:
            # cfg pair with self-flow (2026-10-23): noise, the two t levels, the noisier patches, the rope scale (ROPE_LOG) and
            # the self-teaching crops come from the numbers drawn before the step (pair_draw)
            assert SELFFLOW and not (XMT or block_layout or WREG or T_UNIFORM or DISTILL), "pair: self-flow with random patches only"

        def train_step(images, optimizer, pair=None):
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device, non_blocking=True)
            B = images.shape[0]
            m = getattr(model, "_orig_mod", model)
            p = m.patch_size
            rows, cols = images.shape[-2] // p, images.shape[-1] // p
            T = rows * cols
            noise = pair["noise"] if pair is not None else torch.randn_like(images)

            if SELFFLOW:
                if pair is not None:
                    t_two = torch.sigmoid(pair["u2"] * T_STD + T_MEAN)                    # = draw_t(B, 2) from the drawn normals
                else:
                    t_two = draw_t(B, 2, device=images.device)                            # broad15 without cfg t_dist
                t_clean, t_noisy = t_two.max(1).values, t_two.min(1).values          # t = 1 is the clean image
                if mask_student:
                    t_noisy = torch.zeros_like(t_noisy)
                t_teach = torch.ones_like(t_clean) if teacher_clean else t_clean
                k = int(round(noisy_share * T))
                if block_layout:
                    bh = max(1, min(rows, int(round((noisy_share * rows * rows) ** 0.5))))
                    bw = max(1, min(cols, int(round(k / bh))))
                    r0 = torch.randint(0, rows, (B, 1, 1), device=images.device)
                    c0 = torch.randint(0, cols, (B, 1, 1), device=images.device)
                    rr = torch.arange(rows, device=images.device)[None, :, None]
                    cc = torch.arange(cols, device=images.device)[None, None, :]
                    noisy = ((((rr - r0) % rows) < bh) & (((cc - c0) % cols) < bw)).reshape(B, T)
                else:
                    mask_u = pair["mask"] if pair is not None else torch.rand(B, T, device=images.device)
                    noisy = mask_u.argsort(1).argsort(1) < k                                 # exactly k patches per image
                t_tok = torch.where(noisy, t_noisy[:, None], t_clean[:, None])               # (B, T), row-major patches
                t_map = t_tok.reshape(B, 1, rows, cols).repeat_interleave(p, 2).repeat_interleave(p, 3)
                if XMT:
                    with torch.no_grad():
                        if XMT_BEST:
                            xm_seeds = torch.randn(XMT, *images.shape, device=images.device, dtype=images.dtype)       # own samples per image
                            xm_cand = ((t_map * images)[None] + (1.0 - t_map)[None] * xm_seeds).flatten(0, 1)
                        else:
                            xm_seeds = torch.randn(XMT, *images.shape[1:], device=images.device, dtype=images.dtype)
                            xm_cand = ((t_map * images)[None] + (1.0 - t_map)[None] * xm_seeds[:, None]).flatten(0, 1)      # row k * B + i
                        xm_goal = images
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            if XMT_READOUT:
                                xm_blk = min(int(b_) for b_ in m.readout_heads)
                                xm_out = m.readout_heads[str(xm_blk)](forward_taps(xm_cand, t_tok.repeat(XMT, 1), taps=(xm_blk,), upto=xm_blk)[1][xm_blk], rows, cols)
                                xm_goal = F_.interpolate(images, size=xm_out.shape[-2:], mode="bilinear", antialias=True)
                            else:
                                xm_out = forward_taps(xm_cand, t_tok.repeat(XMT, 1))[0][-1]
                        xm_w = weight_maps(t_tok, rows, cols, [xm_goal])[0]
                        xm_cost = ((xm_out.float().reshape(XMT, B, *xm_goal.shape[1:]) - xm_goal.float()[None]).square() * xm_w[None]).mean((2, 3, 4))
                        xm_all, xm_true = xm_cost.mean(), xm_cost
                        xm_scored = (1.0 - t_tok.mean(1)) >= XMT_MIN_NOISE                  # (B,)
                        if XMT_MIN_NOISE > 0:
                            xm_cost = xm_cost * xm_scored[None]
                        xm_slot_seed = torch.arange(B, device=images.device) % XMT
                        if XMT_BEST:
                            xm_pick = xm_cost.argmin(0)
                            noise = xm_seeds[xm_pick, torch.arange(B, device=images.device)]
                        else:
                            _, xm_slot = xmt_assign(xm_cost.t()[:, xm_slot_seed].cpu().numpy())      # (image, slot)
                            xm_pick = torch.as_tensor(xm_slot, device=images.device) % XMT
                            noise = xm_seeds[xm_pick]
                        xm_cost = xm_true
                noised_student = t_map * images + (1.0 - t_map) * noise
                t_img = t_teach[:, None, None, None]
                noised_teacher = t_img * images + (1.0 - t_img) * noise               # the same noise, the lower level everywhere
                rope = m.core_transformer.rope_tables(rows, cols)                      # one table for both passes
                if use_teacher:
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        teacher_out, target = forward_taps(noised_teacher, t_teach, taps=target_taps, rope=rope, net=teacher_net,
                                                           upto=None if out_weight > 0 else target_taps[-1])
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs, hidden = forward_taps(noised_student, t_tok, taps=student_taps, rope=rope)
            else:
                t = draw_t(B, device=images.device)
                t_tok = t[:, None].expand(B, T)
                t_img = t[:, None, None, None]
                noised = t_img * images + (1.0 - t_img) * noise
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs, hidden = forward_taps(noised, t, taps=PAIR_TAPS)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                losses = []
                w_maps = weight_maps(t_tok, rows, cols, outputs)
                if SELFFLOW and ctx_weight != 1.0:
                    # role weight per patch, scaled so that the mean weight stays 1
                    role = torch.where(noisy, 1.0, ctx_weight).reshape(B, 1, rows, cols)
                    role = role / role.mean()
                    w_maps = [w * role.repeat_interleave(w.shape[-2] // rows, 2).repeat_interleave(w.shape[-1] // cols, 3) for w in w_maps]
                for x_pred, w in zip(outputs, w_maps):
                    goal = images if x_pred.shape[-2:] == images.shape[-2:] else F_.interpolate(
                        images, size=x_pred.shape[-2:], mode="bilinear", antialias=True)
                    losses.append(((x_pred.float() - goal.float()).square() * w).mean())
                if SELFFLOW:
                    teach = (sum(teach_distance(hidden[s_], target[t_], noisy.float() if teach_noisy_only else None)
                                 for s_, t_ in PAIRS) if use_teacher
                             else torch.zeros((), device=images.device))
                else:
                    teach = sum(teach_distance(hidden[s], hidden[t_].detach()) for s, t_ in PAIRS)
                loss_final = losses[-1]
                loss = loss_final + w_teach * teach
                st_out = st_noised = torch.zeros((), device=images.device)
                wreg_terms = None
                if WREG:
                    with torch.autocast(device_type="cuda", enabled=False):
                        wreg_z = hidden[wreg_block].float()
                        wreg_terms = wreg_family(wreg_z)
                    loss = loss + wreg_w * wreg_terms.sum()
                if SELFTEACH:
                    t_c = t_clean[:, None, None, None]
                    st_loss, st_out, st_noised = st["terms"](images, t_c * images + (1.0 - t_c) * noise, t_clean, outputs[-1], rope, pair=pair)
                    loss = loss + st_loss
                out_align = torch.zeros((), device=images.device)
                if SELFFLOW and use_teacher and out_weight > 0:
                    out_align = ((outputs[-1].float() - teacher_out[-1].float()).square() * w_maps[-1]).mean()
                    loss = loss + out_weight * out_align
                for readout_loss in losses[:-1]:
                    loss = loss + weight_of_readouts * readout_loss

            loss.backward()
            optimizer.step()
            out = {"loss": loss.detach(), "loss_final": loss_final.detach(), "readout_losses": torch.stack(losses[:-1]).detach(),
                   "teach": teach.detach()}
            if wreg_terms is not None:
                for name_, term_ in zip(("wreg_swd", "wreg_bwd", "wreg_c_swd", "wreg_t_swd"), wreg_terms.detach()):
                    out[name_] = term_
                out["wreg_z_std"], out["wreg_z_mean"] = wreg_z.detach().std(), wreg_z.detach().mean()
            if XMT:
                out["xm_loss_all"], out["xm_share1"] = xm_all, (xm_pick == 1).float().mean()
                out["xm_scored"] = xm_scored.float().mean()
                if XMT_READOUT:
                    out["xm_score_pick"] = xm_cost.gather(0, xm_pick[None]).mean()
            if SELFFLOW:
                out["noisy_share"] = noisy.float().mean()
                out["st_out"], out["st_noised"] = st_out.detach(), st_noised.detach()
                out["out_align"] = out_align.detach()
                # share of noisier patches whose right neighbor is noisier too: about noisy_share when spread, high for a block
                grid = noisy.reshape(B, rows, cols)
                out["noisy_neighbor"] = (grid & grid.roll(-1, 2)).float().sum() / grid.float().sum().clamp_min(1.0)
                out["t_gap"] = (t_clean - t_noisy).mean()
                out["t_teacher"] = t_teach.mean()
                out["t_student_noisy"] = t_noisy.mean()
                out["t_noisy_low"] = (t_noisy < 0.1).float().mean()                    # share of noisier patches above noise level 0.9
            return out

        return train_step

    @torch.no_grad()
    def teach_eval(n_batches=20):
        """test_teach: mse(h after block `from`, h after block `to`) on the fixed test images, t and noise, one
        forward pass per batch. For self-flow this is the same-input distance, not its two-input train loss."""
        was = model.training
        model.eval()
        rng = torch.Generator(device=device).manual_seed(0)
        total, count = [0.0] * len(PAIRS), 0
        try:
            for images in g["get_eval_batches"]("test", n_batches):
                images = images.to(device, non_blocking=True)
                t = broad_t(images.shape[0], generator=rng)
                noise = torch.randn(images.shape, device=device, generator=rng)
                t_img = t[:, None, None, None]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs, hidden = forward_taps(t_img * images + (1.0 - t_img) * noise, t, taps=PAIR_TAPS)
                for i, (s, t_) in enumerate(PAIRS):
                    total[i] += teach_distance(hidden[s], hidden[t_]).item() * images.shape[0]
                count += images.shape[0]
        finally:
            model.train(was)
        out = {"test_teach": sum(total) / count}        # the sum over the pairs, as in the train loss
        if len(PAIRS) > 1:
            out.update({f"teach_layer_{s}": v / count for (s, t_), v in zip(PAIRS, total)})      # per student block
        return out

    if TEACH:
        # CHECK at start: the job's copy of the forward pass must give the model's own output.
        with torch.no_grad():
            _x = torch.rand(2, 3, g["IMAGE_SIZE"], g["IMAGE_SIZE"], device=device)
            _t = torch.tensor([0.3, 0.8], device=device)
            _was = model.training
            model.eval()
            _own = model(_x, _t, all_readouts=True)
            _mine, _ = forward_taps(_x, _t)
            _T = (g["IMAGE_SIZE"] // g["patch_size"]) ** 2
            _per_token, _ = forward_taps(_x, _t[:, None].expand(2, _T))
            model.train(_was)
            _err = max((a - b).abs().max().item() for a, b in zip(_own, _mine))
            _err_tok = max((a - b).abs().max().item() for a, b in zip(_own, _per_token))
        if DISTILL:
            # CHECK at start, per pair: the gradient of its term reaches the student side (the block `student`, or the
            # patch embedding for 0) and no block after it, so the target is detached and the pair is wired as planned.
            _m = getattr(model, "_orig_mod", model)
            _blocks = _m.core_transformer.blocks
            def _gsum(mod):
                return sum(p.grad.abs().sum().item() for p in mod.parameters() if p.grad is not None)
            for _s, _t2 in PAIRS:
                _m.zero_grad(set_to_none=True)
                _, _h = forward_taps(_x, _t, taps=(_s, _t2), upto=_t2)
                teach_distance(_h[_s], _h[_t2].detach()).backward()
                _student = _gsum(_m.embed) if _s == 0 else _gsum(_blocks[_s - 1])
                _after = sum(_gsum(b) for b in _blocks[_s:])
                assert _student > 0 and _after == 0, f"pair {_s} <- {_t2}: gradient at the student {_student:.3g}, in later blocks {_after:.3g}"
            _m.zero_grad(set_to_none=True)
            print(f"CHECK distillation pairs (student <- target): {' '.join(f'{s}<-{t2}' for s, t2 in PAIRS)} | "
                  f"{len(PAIRS)} of {len(PAIRS)} terms send gradient to their student and to no later block", flush=True)
        print(f"teaching: {'self-flow' if SELFFLOW else 'layer distillation'} | student block {PAIRS[0][0]} -> target block {PAIRS[-1][1]} | {len(PAIRS)} pair(s) | "
              f"weight {TEACH.get('weight', 1.0)} | forward copy against the model: max difference {_err:.2e}, with per-token time {_err_tok:.2e}", flush=True)
        assert _err < 1e-4 and _err_tok < 1e-3, "the job's forward pass differs from the model's"
    print("schedule broad15: t = sigmoid(1.5 * randn), loss = mean((x_pred - x)^2 * weight), weight = " + {"clamp01": "1 / max(1 - t, 0.1)^2", "logflip": "log1p(1 / (1 - t)^2) / log 2", "log": "log1p(1 / t^2) / log 2"}[LOSS_WEIGHT] + ", in the train step and the split losses", flush=True)
assert not (PYR and XM_K > 1), "the pyramid train step has no XM"
assert SCHED and not TEACH or cfg.get("t_shift", 1.0) == 1.0, "t_shift is wired for the broad15 train step without teaching"
if SCHED and T_SHIFT != 1.0:
    print(f"timestep shift: alpha {T_SHIFT:.4f} on the noise level 1 - t, train step only", flush=True)
if not SCHED:
    TEACH, SELFFLOW, DISTILL, MSA = {}, {}, {}, {}
    assert not (cfg.get("distill") or cfg.get("selfflow") or cfg.get("selfteach")), "distill / selfflow / selfteach need schedule broad15"
if MSA:
    import msa as msa_mod
    _msa_step, _msa_check = msa_mod.make({"model": model, "device": device, "cfg": cfg, "apply_2d_rope": g["apply_2d_rope"], "weight_maps": weight_maps,
                                          "teach_distance": teach_distance, "pyramid_weight": g["pyramid_weight"],
                                          "selfteach": make_selfteach() if SELFTEACH else None})
    _msa_check()
step_fn = (_msa_step if MSA else make_teach_train_step() if TEACH else make_broad_train_step(PYR) if SCHED else g["make_looped_train_step"]() if LOOP else g["make_pyramid_train_step"]() if PYR
           else g["make_train_step"]() if XM_K == 1 else make_train_step_xm(XM_K, XM_N))
torch.set_float32_matmul_precision("high")
train_step = torch.compile(step_fn) if cfg.get("compile", True) else step_fn
# cfg["ot_noise"] = {"every": 4, "weight": 16.0} (user, 2026-10-11): every 4th step is a transport step at 100% noise.
# One forward pass of B pure-noise inputs (t = 0). The Hungarian algorithm pairs the B outputs 1-1 with the B images of
# the batch at the lowest total squared distance. The loss of that same pass, against the paired images, times `weight`
# (16 = the clamp01 weight at 25% noise; clamp01 itself gives 1 at 100% noise). The other steps are normal steps.
OT = cfg.get("ot_noise") or {}
ot_acc = {}
if OT or cfg.get("xm_ot") or cfg.get("noise_ot") or cfg.get("distill2"):
    assert SCHED and PYR and (not TEACH or bool(SELFFLOW and cfg.get("xm_ot") and not OT)) and not (OT and cfg.get("xm_ot")), "ot_noise / xm_ot are wired for the plain pyramid model"
if OT:
    from scipy.optimize import linear_sum_assignment
    OT_EVERY, OT_W = int(OT.get("every", 4)), float(OT.get("weight", 16.0))
    assert OT_EVERY >= 2 and OT_W > 0

    def grad_norm():
        return torch.sqrt(sum(p_.grad.float().square().sum() for p_ in model.parameters() if p_.grad is not None))

    def ot_step(images, optimizer):
        optimizer.zero_grad(set_to_none=True)
        images = images.to(device, non_blocking=True)
        B = images.shape[0]
        noise = torch.randn_like(images)
        t = torch.zeros(B, device=images.device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(noise, t, all_readouts=True)
        with torch.no_grad():
            final = outputs[-1].detach().float()
            cost = torch.cdist(final.flatten(1)[None], images.float().flatten(1)[None])[0].square() / final[0].numel()   # (output, image)
            _, cols_ = linear_sum_assignment(cost.cpu().numpy())
            assert sorted(cols_.tolist()) == list(range(B)), "the pairing is not 1-1"
            perm = torch.as_tensor(cols_, device=images.device)
            rows_ = torch.arange(B, device=images.device)
        paired = images[perm]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            losses = broad_losses(outputs, paired, noise, t[:, None, None, None])
            loss = losses[-1]
            for readout_loss in losses[:-1]:
                loss = loss + float(g.get("pyramid_weight", 1.0)) * readout_loss
        (OT_W * loss).backward()
        gn = grad_norm()
        optimizer.step()
        return {"ot_loss": cost[rows_, perm].mean(), "ot_loss_fixed": cost.diagonal().mean(), "ot_loss_final": losses[-1].detach(),
                "ot_moved": (perm != rows_).float().mean(), "ot_out_std": final.std(0).mean(), "ot_grad_norm": gn}
get_split_loss, denoise_seeds = g["get_split_loss"], g["denoise_seeds"]


# ---------------------------------------------------------------- evals, previews, checkpoints
@torch.no_grad()
def x_mse_by_t(ts=(0.1, 0.3, 0.5, 0.7, 0.9), n_batches=4):
    """Plain MSE of the predicted clean image on fixed test images, at fixed noise levels."""
    was = model.training
    model.eval()
    rng = torch.Generator(device=device).manual_seed(1)
    out = {}
    try:
        for t_value in ts:
            errs = []
            for images in g["get_eval_batches"]("test", 20)[:n_batches]:
                images = images.to(device)
                t = torch.full((images.shape[0],), t_value, device=device)
                noise = torch.randn(images.shape, device=device, generator=rng)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    x_pred = model(t[:, None, None, None] * images + (1 - t[:, None, None, None]) * noise, t)
                errs.append((x_pred.float() - images).square().mean().item())
            out[f"x_mse_t{int(t_value * 10):02d}"] = sum(errs) / len(errs)
    finally:
        model.train(was)
    return out


@torch.no_grad()
def loop_eval(max_loops=6, n_batches=20, probe_batches=4):
    """Looped model only. On the fixed test images, t and noise of get_split_loss("test"):

      loopq_dense_N / loopq_sparse_N   eval loss when the model exits after N loops (N = 1..max_loops),
                                       with all tokens / with the sparse routing of a sparse train step
      upd_K    RMS(h_{K+1} - h_K) / RMS(h_K), h_K = the loop state after loop K (dense)
      cos_K    mean cosine similarity of h_K and h_{K+1} per token (dense)
      xsa_N    RMS(part of the attention output that XSA removes) / RMS(attention output), mean of the
               middle blocks, in loop N (dense)

    One pass with all exits per mode: exit N of a longer pass is the output of an N-loop pass."""
    import torch.nn.functional as F
    core = model.core_transformer
    middle = list(range(core.shallow_blocks, core.middle_end))
    was = model.training
    model.eval()
    cpu_rng, cuda_rng = torch.random.get_rng_state(), (torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
    states, removed = [], []

    def state_hook(module, args, out):
        states.append(out.float())

    def xsa_hook(module, args):
        # the attention of this call again, to measure what XSA takes out of it
        x, rope = args
        B, T, _ = x.shape
        nh, hs = module.n_heads, module.head_size
        q, k, v = (part.view(B, T, nh, hs).permute(0, 2, 1, 3) for part in module.QKV(x).chunk(3, dim=-1))
        q, k = g["apply_2d_rope"](module.q_norm(q), module.k_norm(k), rope=rope)
        o = F.scaled_dot_product_attention(q, k, v).float()
        v_hat = F.normalize(v.float(), dim=-1)
        part = (o * v_hat).sum(-1, keepdim=True) * v_hat
        removed.append((part.square().mean() / o.square().mean()).sqrt().item())

    has_xsa = type(core.blocks[middle[0]].MHA).__name__ == "XSAMHA"
    sums = {"dense": torch.zeros(max_loops, device=device), "sparse": torch.zeros(max_loops, device=device)}
    upd, cos, xsa, count = torch.zeros(max_loops - 1), torch.zeros(max_loops - 1), torch.zeros(max_loops), 0
    rng = torch.Generator(device=device).manual_seed(0)
    try:
        for bi, images in enumerate(g["get_eval_batches"]("test", n_batches)):
            images = images.to(device, non_blocking=True)
            t = g["sample_train_t"](images.shape[0], generator=rng)
            noise = torch.randn(images.shape, device=device, generator=rng)
            t_img = t[:, None, None, None]
            noised = t_img * images + (1.0 - t_img) * noise
            denom = (1.0 - t_img).clamp_min(0.05)
            v_target = (images - noised) / denom
            count += images.shape[0]
            for mode in ("dense", "sparse"):
                probe = mode == "dense" and bi < probe_batches
                hooks = []
                if probe:
                    states.clear(), removed.clear()
                    hooks.append(core.blocks[middle[-1]].register_forward_hook(state_hook))
                    if has_xsa:
                        hooks += [core.blocks[i].MHA.register_forward_pre_hook(xsa_hook) for i in middle]
                torch.manual_seed(1000 + bi)                      # the sparse token set of this batch: the same at every eval
                try:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        exits = model(noised, t, n_loops=max_loops, all_exits=True, sparse=(mode == "sparse"))
                finally:
                    for hook in hooks:
                        hook.remove()
                for n, x_pred in enumerate(exits):
                    sums[mode][n] += (((x_pred.float() - noised) / denom) - v_target).square().mean((1, 2, 3)).sum()
                if probe:
                    assert len(states) == max_loops and len(removed) == (max_loops * len(middle) if has_xsa else 0)
                    for k in range(max_loops - 1):
                        a, b = states[k], states[k + 1]
                        upd[k] += ((b - a).square().mean().sqrt() / a.square().mean().sqrt()).item() / probe_batches
                        cos[k] += F.cosine_similarity(a, b, dim=-1).mean().item() / probe_batches
                    for n in range(max_loops if has_xsa else 0):
                        xsa[n] += statistics.fmean(removed[n * len(middle):(n + 1) * len(middle)]) / probe_batches
    finally:
        model.train(was)
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng)
    rec = {}
    for mode in sums:
        rec.update({f"loopq_{mode}_{n + 1}": (sums[mode][n] / count).item() for n in range(max_loops)})
    rec.update({f"upd_{k + 1}": upd[k].item() for k in range(max_loops - 1)})
    rec.update({f"cos_{k + 1}": cos[k].item() for k in range(max_loops - 1)})
    if has_xsa:
        rec.update({f"xsa_{n + 1}": xsa[n].item() for n in range(max_loops)})
    return rec


@torch.no_grad()
def pyramid_eval(n_batches=20):
    """Pyramid model only: the loss of every readout head on the fixed test images, t and noise of
    get_split_loss("test"). test_R32 etc.; test_final_check is the final head through the same path."""
    was = model.training
    model.eval()
    rng = torch.Generator(device=device).manual_seed(0)
    sums, count = None, 0
    try:
        for images in g["get_eval_batches"]("test", n_batches):
            images = images.to(device, non_blocking=True)
            B = images.shape[0]
            t = g["sample_train_t"](B, generator=rng)
            noise = torch.randn(images.shape, device=device, generator=rng)
            t_img = t[:, None, None, None]
            noised = t_img * images + (1.0 - t_img) * noise
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(noised, t, all_readouts=True)
                losses = torch.stack([l.float() for l in g["readout_losses"](outputs, images, noised, t_img)])
            sums = losses * B if sums is None else sums + losses * B
            count += B
    finally:
        model.train(was)
    values = (sums / count).tolist()
    return {**{f"test_{n}": v for n, v in zip(PYR_NAMES, values)}, "test_final_check": values[-1]}


def st_judge_eval(n_batches=4, crop_min=0.2, crop_max=0.5):
    """cfg pair runs (scaling sweep, baseline included): the self-teaching distance measured, not trained. Fixed test
    images, t and noise (own generator), one-pass output in eval mode, the same fixed crop of output and clean image,
    z after blocks 3 and 6 of the current weights at t = 1 (= the judge of a self-teaching step), normalized mse.
    Keys st_eval_b3 / st_eval_b6: the same numbers in every variant, so the baseline shows what the term would see."""
    was = model.training
    model.eval()
    raw_ = getattr(model, "_orig_mod", model)
    rng = torch.Generator(device=device).manual_seed(5)
    sums, count = {3: 0.0, 6: 0.0}, 0
    try:
        with torch.no_grad():
            for images in g["get_eval_batches"]("test", n_batches):
                images = images.to(device, non_blocking=True).float()
                B = images.shape[0]
                t = g["sample_train_t"](B, generator=rng)
                noise = torch.randn(images.shape, device=device, generator=rng)
                t_img = t[:, None, None, None]
                grid = st_crop_grid_from(torch.rand(B, 3, device=device, generator=rng), images.shape[-1], crop_min, crop_max)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(t_img * images + (1.0 - t_img) * noise, t, all_readouts=True)[-1]
                    crops = st_crop_scale(torch.cat([images, out.float()]), grid.repeat(2, 1, 1, 1))
                    z = forward_taps(crops, torch.ones(2 * B, device=device), taps=(3, 6), upto=6, net=raw_)[1]
                for l_ in (3, 6):
                    sums[l_] += float(teach_distance(z[l_][B:], z[l_][:B])) * B
                count += B
    finally:
        model.train(was)
    return {f"st_eval_b{l_}": sums[l_] / count for l_ in (3, 6)}


# cfg["gen_eval"] = {"every": N, "n": 10000, "steps": 16, "sampler": "euler"}: generative scores
# (FID, KID, fd_ae: see gen_eval.py) of n samples from fixed seeds against n real validation
# images, every N steps and at the last step. The sampler is the notebook's denoise_seeds.
GEN = cfg.get("gen_eval") or {}


def gen_scores(step):
    """cfg gen_eval "weights": "raw" (default), "ema" (the averaged weights, keys fid_ema ...) or "both"."""
    which = GEN.get("weights", "raw")
    out = {}
    if CORR and GEN.get("corr_grid"):
        # gen_eval "corr_grid": [k, ...] (user, 2026-10-13): no scores. One grid of samples beside cfg gen_eval_out: one row
        # per number of corrections k, the same seeds in every row, the sampler steps of the scores, averaged weights.
        ks = [int(k_) for k_ in GEN["corr_grid"]]
        name = f"corrgrid_step{step:06d}.png"
        corr_k_grid(step, seeds=range(8), sample_steps=int(GEN.get("steps", 16)), ks=ks, path=Path(cfg.get("gen_eval_out") or OUT / "metrics.jsonl").parent / name)
        print(f"GEN step {step:,} | corrector grid saved: {name} | rows k = {ks}", flush=True)
        return {"gen_corr_grid": name, "gen_corr_ks": json.dumps(ks), "gen_sampler_steps": float(GEN.get("steps", 16))}
    if CORR:
        _corr_raw.corr_k = int(GEN.get("corr_k", CORR_K))        # corrections per sampler step of this score
        if GEN.get("corr_range"):
            _corr_raw.corr_range = [float(v_) for v_ in GEN["corr_range"]]
            _corr_raw.corr_k_out, _corr_raw.corr_calls = int(GEN.get("corr_k_out", CORR_K)), [0, 0]
    if which in ("raw", "both") or ema_params is None:
        out.update(gen_scores_now(step))
    if which in ("ema", "both") and ema_params is not None:
        params = [p for p in model.parameters()]
        with torch.no_grad():
            kept = [p.detach().clone() for p in params]
            for p, e in zip(params, ema_params):
                p.copy_(e)
        try:
            scores = gen_scores_now(step, "averaged weights")
        finally:
            with torch.no_grad():
                for p, k in zip(params, kept):
                    p.copy_(k)
        out.update({k + "_ema": v for k, v in scores.items() if not k.startswith("gen_")})
        out.update({k: v for k, v in scores.items() if k.startswith("gen_")})
    if CORR:
        k_ = _corr_raw.corr_k
        _corr_raw.corr_k = CORR_K
        per_step = CORR_SPLIT + k_ * (CORR_BLOCKS - CORR_SPLIT)
        out.update(gen_corr_k=float(k_), gen_block_evals=float(per_step * int(GEN.get("steps", 16))))
        if _corr_raw.corr_range is not None:
            n_in, n_out = _corr_raw.corr_calls
            assert 0 < n_in and 0 < n_out, ("the range holds all sampler steps or none", _corr_raw.corr_range, n_in, n_out)
            k_mean = (k_ * n_in + _corr_raw.corr_k_out * n_out) / (n_in + n_out)
            out.update(gen_corr_range=json.dumps(_corr_raw.corr_range), gen_corr_k_out=float(_corr_raw.corr_k_out), gen_corr_share_in=n_in / (n_in + n_out),
                       gen_block_evals=float((CORR_SPLIT + k_mean * (CORR_BLOCKS - CORR_SPLIT)) * int(GEN.get("steps", 16))))
            print(f"GEN corrector range: k = {k_} at noise levels {_corr_raw.corr_range}, k = {_corr_raw.corr_k_out} outside | {100 * n_in / (n_in + n_out):.0f}% of the "
                  f"sampler steps inside | {out['gen_block_evals']:.0f} block evaluations per sample", flush=True)
            _corr_raw.corr_range = None
        print(f"GEN corrector: k = {k_} corrections per sampler step | {per_step} block evaluations per step x {int(GEN.get('steps', 16))} steps = "
              f"{per_step * int(GEN.get('steps', 16))} (the plain {CORR_BLOCKS}-block model: {CORR_BLOCKS * int(GEN.get('steps', 16))})", flush=True)
    return out


# cfg gen_eval "ig" = {"source": "R32" | "R64" | "blur32" | "blur64", "scale": w, "t_max": 0.7} (2026-10-11): sampling
# only, no training. Internal Guidance (arXiv 2512.24176): the prediction of the clean image is pushed away from a weaker
# prediction, x = weak + w * (final - weak), while t <= t_max (t = 1 is the clean image). "R32": the weak prediction is
# the readout R32 of the model, upsampled. "blur32": the final prediction itself, averaged to 32 px and upsampled: the
# control that separates "the readout knows less" from a plain sharpening of the final prediction.
@torch.inference_mode()
def guided_seeds(seeds, steps, ig):
    F = torch.nn.functional
    size, n = g["IMAGE_SIZE"], len(seeds)
    m = getattr(model, "_orig_mod", model)
    source, w, t_max = ig["source"], float(ig["scale"]), float(ig.get("t_max", 0.7))
    x = torch.stack([torch.randn((m.un_embed.c, size, size), generator=torch.Generator(device=device).manual_seed(int(s_)), device=device)
                     for s_ in seeds])
    ts = g["make_timesteps"](steps, "linear", 1.0)
    was_training = model.training
    model.eval()
    try:
        for t, t_next in zip(ts[:-1], ts[1:]):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outs = [o_.float() for o_ in m(x, t.expand(n), all_readouts=True)]
            final = outs[-1]
            if float(t) <= t_max:
                if source.startswith("blur"):
                    weak = F.avg_pool2d(final, size // int(source[4:]))
                else:
                    weak = outs[PYR_NAMES.index(source)]
                weak = F.interpolate(weak, size=(size, size), mode="bilinear", align_corners=False)
                final = weak + w * (final - weak)
                IG_STAT["out"] += float(((final < 0) | (final > 1)).float().mean())
                IG_STAT["diff"] += float((outs[-1] - weak).abs().mean())
                IG_STAT["n"] += 1
            x = x + (t_next - t) * (final - x) / (1 - t)
    finally:
        model.train(was_training)
    return x.clamp(0, 1)


IG_STAT = {"out": 0.0, "diff": 0.0, "n": 0}

# cfg gen_eval "fix" = {"noise_dc": a, "mean_cap": s, "from_step": 2} (user, 2026-10-13): sampling tests against samples
# that go black or white. The model reads the brightness of the image from the mean of the starting noise, too strongly.
#   noise_dc   the mean of each starting noise sample is multiplied by a (1 = unchanged, 0 = every sample has mean 0)
#   mean_cap   from sampler step "from_step" on, the predictions of a batch are shifted so that the spread of their image
#              means (std over the batch) is at most s (real images: 0.137). Each image moves toward the batch mean by
#              the same factor, then the prediction is clamped to 0..1. No change when the spread is below s.
# {} = the plain Euler sampler through this code path (its FID must equal the normal sampler's).
FIX_STAT = {"means": [], "dark": 0.0, "light": 0.0, "n": 0, "shrink": 0.0, "shrunk": 0}


@torch.inference_mode()
def fix_seeds(seeds, steps, fix):
    size, n = g["IMAGE_SIZE"], len(seeds)
    m = getattr(model, "_orig_mod", model)
    a, cap, first = float(fix.get("noise_dc", 1.0)), float(fix.get("mean_cap", 0) or 0), int(fix.get("from_step", 2))
    x = torch.stack([torch.randn((m.un_embed.c, size, size), generator=torch.Generator(device=device).manual_seed(int(s_)), device=device)
                     for s_ in seeds])
    if a != 1.0:
        x = x - (1.0 - a) * x.mean((1, 2, 3), keepdim=True)
    ts = g["make_timesteps"](steps, "linear", 1.0)
    if fix.get("t_scale"):
        # "t_scale": k: the grid is linear in the t of a model with noise scale k: t = t_k / (t_k + (1 - t_k) * k).
        # k < 1 puts more of the steps at low noise (for runs trained with cfg t_dist toward less noise)
        k_scale = float(fix["t_scale"])
        ts = ts / (ts + (1.0 - ts) * k_scale)
    was_training = model.training
    model.eval()
    try:
        for k_, (t, t_next) in enumerate(zip(ts[:-1], ts[1:])):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                final = m(x, t.expand(n)).float()
            if cap and k_ + 1 >= first:
                means = final.mean((1, 2, 3), keepdim=True)
                spread = means.std(unbiased=False)
                if spread > cap:
                    final = (final + (cap / spread - 1.0) * (means - means.mean())).clamp(0, 1)
                    FIX_STAT["shrink"] += float(cap / spread); FIX_STAT["shrunk"] += 1
            x = x + (t_next - t) * (final - x) / (1 - t)
    finally:
        model.train(was_training)
    x = x.clamp(0, 1)
    FIX_STAT["means"].append(x.mean((1, 2, 3)).cpu())
    FIX_STAT["dark"] += float((x < 0.02).float().mean()) * n; FIX_STAT["light"] += float((x > 0.98).float().mean()) * n; FIX_STAT["n"] += n
    return x


def gen_scores_now(step, label="trained weights"):
    import gen_eval
    n, sample_steps, sampler = int(GEN.get("n", 10000)), int(GEN.get("steps", 16)), GEN.get("sampler", "euler")
    if GEN.get("ig"):
        ig = GEN["ig"]
        assert PYR and sampler == "euler" and (ig["source"].startswith("blur") or ig["source"] in PYR_NAMES), (ig, PYR_NAMES)
        IG_STAT.update(out=0.0, diff=0.0, n=0)

        def real_batches_ig():
            for images, _ in g["test_dataloader"]:
                yield images.to(device, non_blocking=True)

        tag = f"val{g['IMAGE_SIZE']}px_draft{int(bool(cfg.get('draft_decode')))}"
        scores, _ = gen_eval.score(lambda seeds: guided_seeds(seeds, sample_steps, ig), real_batches_ig, n, tag)
        k = max(IG_STAT["n"], 1)
        assert IG_STAT["n"] > 0 and IG_STAT["diff"] / k > 1e-4, ("the guidance changed nothing", IG_STAT)
        print(f"GEN step {step:,} ({label}) | guidance {ig['source']} scale {ig['scale']} t_max {ig.get('t_max', 0.7)} | {n} samples, "
              f"{sample_steps} euler steps | FID {scores['fid']:.2f} | KID x1000 {scores['kid']:.2f} | fd_ae_z {scores['fd_ae_z']:.2f} | "
              f"mean |final - weak| {IG_STAT['diff'] / k:.4f} | share of guided pixels outside 0..1 {100 * IG_STAT['out'] / k:.2f}%", flush=True)
        return {"fid": scores["fid"], "kid": scores["kid"], "fd_ae_z": scores["fd_ae_z"], "fd_ae_pool": scores["fd_ae_pool"],
                "gen_n": float(n), "gen_sampler_steps": float(sample_steps), "gen_ig_source": ig["source"], "gen_ig_scale": float(ig["scale"]),
                "gen_ig_t_max": float(ig.get("t_max", 0.7)), "gen_ig_outside": IG_STAT["out"] / k, "gen_ig_diff": IG_STAT["diff"] / k}

    def real_batches():
        for images, _ in g["test_dataloader"]:
            yield images.to(device, non_blocking=True)

    tag = f"val{g['IMAGE_SIZE']}px_draft{int(bool(cfg.get('draft_decode')))}"
    if GEN.get("fix") is not None:
        fix = GEN["fix"]
        assert sampler == "euler" and not COND
        FIX_STAT.update(means=[], dark=0.0, light=0.0, n=0, shrink=0.0, shrunk=0)
        if fix.get("grid_only"):
            # "grid": N, "grid_only": true: no scores. The first scoring batch (seeds 0..249, so the cap sees the same batch
            # as in the scoring) is sampled once and its first N images are saved as one grid beside cfg gen_eval_out.
            from PIL import Image
            n_grid = int(fix.get("grid", 48))
            images = fix_seeds(list(range(250)), sample_steps, fix)[:n_grid].float().cpu()
            _, _, h, w = images.shape
            rows = (n_grid + 7) // 8
            grid = torch.ones(3, rows * h + (rows - 1) * 4, 8 * w + 7 * 4)
            for i, image in enumerate(images):
                r, c = divmod(i, 8)
                grid[:, r * (h + 4):r * (h + 4) + h, c * (w + 4):c * (w + 4) + w] = image
            name = "plain" if not (fix.get("mean_cap") or float(fix.get("noise_dc", 1.0)) != 1.0) else \
                "_".join(([f"dc{fix['noise_dc']:g}".replace(".", "")] if float(fix.get("noise_dc", 1.0)) != 1.0 else [])
                         + ([f"cap{fix['mean_cap']:g}".replace(".", "")] if fix.get("mean_cap") else []))
            path = Path(cfg.get("gen_eval_out") or OUT / "metrics.jsonl").parent / f"fixgrid_step{step:06d}_{name}.png"
            Image.fromarray((grid.permute(1, 2, 0).clamp(0, 1) * 255).round().byte().numpy()).save(path)
            means = torch.cat(FIX_STAT["means"])
            print(f"GEN step {step:,} ({label}) | fix {json.dumps(fix)} | grid of {n_grid} samples saved: {path.name} | 250 samples: spread between images "
                  f"{means.std(unbiased=False).item():.3f} | black {100 * FIX_STAT['dark'] / FIX_STAT['n']:.1f}% | white {100 * FIX_STAT['light'] / FIX_STAT['n']:.1f}%", flush=True)
            return {"gen_fix": json.dumps(fix), "gen_grid": path.name, "gen_n": 250.0, "gen_sampler_steps": float(sample_steps)}
        scores, floor = gen_eval.score(lambda seeds: fix_seeds(seeds, sample_steps, fix), real_batches, n, tag)
        means = torch.cat(FIX_STAT["means"])
        stat = {"gen_between": means.std(unbiased=False).item(), "gen_bright": means.mean().item(), "gen_dark": FIX_STAT["dark"] / FIX_STAT["n"],
                "gen_light": FIX_STAT["light"] / FIX_STAT["n"], "gen_cap_factor": FIX_STAT["shrink"] / max(FIX_STAT["shrunk"], 1), "gen_cap_steps": float(FIX_STAT["shrunk"])}
        print(f"GEN step {step:,} ({label}) | fix {json.dumps(fix)} | {n} samples, {sample_steps} euler steps | FID {scores['fid']:.2f} | KID x1000 {scores['kid']:.2f} | "
              f"fd_ae_z {scores['fd_ae_z']:.2f} | spread between images {stat['gen_between']:.3f} | brightness {stat['gen_bright']:.3f} | black {100 * stat['gen_dark']:.1f}% | "
              f"white {100 * stat['gen_light']:.1f}% | cap used in {FIX_STAT['shrunk']} batch steps, mean factor {stat['gen_cap_factor']:.2f}", flush=True)
        return {"fid": scores["fid"], "kid": scores["kid"], "fd_ae_z": scores["fd_ae_z"], "fd_ae_pool": scores["fd_ae_pool"], "gen_n": float(n),
                "gen_sampler_steps": float(sample_steps), "gen_fix": json.dumps(fix), **stat}
    if COND:
        scale, use_class = float(GEN.get("cfg", 1.0)), bool(GEN.get("class", True))
        assert sampler == "euler"
        if GEN.get("grid"):
            # gen_eval "grid": N (2026-10-15): no scores. Seeds 0..N-1 (class of a seed: COND_PERM[seed % classes]), one PNG
            # per sample in condgrid_step<step>_cfg<scale>/ beside gen_eval_out, and classes.json (seed -> class index)
            from PIL import Image
            n_grid = int(GEN["grid"])
            images = cond_seeds(list(range(n_grid)), sample_steps, scale, use_class).float().cpu()
            folder = Path(cfg.get("gen_eval_out") or OUT / "metrics.jsonl").parent / f"condgrid_step{step:06d}_cfg{scale:g}"
            folder.mkdir(exist_ok=True)
            for i, image in enumerate(images):
                Image.fromarray((image.permute(1, 2, 0) * 255).round().byte().numpy()).save(folder / f"{i:03d}.png")
            classes = [int(COND_PERM[i % COND_N]) if use_class else -1 for i in range(n_grid)]
            (folder / "classes.json").write_text(json.dumps(classes))
            print(f"GEN step {step:,} ({label}) | grid of {n_grid} samples, guidance scale {scale:g}, {sample_steps} euler steps: {folder.name}", flush=True)
            return {"gen_grid": folder.name, "gen_n": float(n_grid), "gen_cfg": scale, "gen_class": float(use_class), "gen_sampler_steps": float(sample_steps)}
        COND_STAT.update(diff=0.0, n=0)
        scores, floor = gen_eval.score(lambda seeds: cond_seeds(seeds, sample_steps, scale, use_class), real_batches, n, tag)
        assert not (use_class and scale != 1.0) or COND_STAT["diff"] / max(COND_STAT["n"], 1) > 1e-5, ("the class changed nothing", COND_STAT)
        print(f"GEN step {step:,} ({label}) | {'class of the seed' if use_class else 'no class'}, guidance scale {scale:g} | {n} samples, {sample_steps} euler steps | "
              f"FID {scores['fid']:.2f} | KID x1000 {scores['kid']:.2f} | fd_ae_z {scores['fd_ae_z']:.2f} | fd_ae_pool {scores['fd_ae_pool']:.3f} | "
              f"{scores['gen_seconds']:.0f}s | mean |class - no class| {COND_STAT['diff'] / max(COND_STAT['n'], 1):.4f}", flush=True)
        return {"fid": scores["fid"], "kid": scores["kid"], "fd_ae_z": scores["fd_ae_z"], "fd_ae_pool": scores["fd_ae_pool"],
                "gen_n": float(n), "gen_sampler_steps": float(sample_steps), "gen_cfg": scale, "gen_class": float(use_class)}
    scores, floor = gen_eval.score(lambda seeds: denoise_seeds(seeds, steps=sample_steps, sampler=sampler), real_batches, n, tag)
    print(f"GEN step {step:,} ({label}) | {n} samples, {sample_steps} {sampler} steps | FID {scores['fid']:.2f} | KID x1000 {scores['kid']:.2f} | "
          f"fd_ae_z {scores['fd_ae_z']:.2f} | fd_ae_pool {scores['fd_ae_pool']:.3f} | {scores['gen_seconds']:.0f}s | real against real: "
          f"FID {floor['fid']:.2f}, KID {floor['kid']:.2f}, fd_ae_z {floor['fd_ae_z']:.2f}, fd_ae_pool {floor['fd_ae_pool']:.3f}", flush=True)
    return {"fid": scores["fid"], "kid": scores["kid"], "fd_ae_z": scores["fd_ae_z"], "fd_ae_pool": scores["fd_ae_pool"],
            "gen_n": float(n), "gen_sampler_steps": float(sample_steps)}


def evaluate(step):
    t0 = time.time()
    rec = {"test_clean": get_split_loss("test").item(), "train_clean_eval": get_split_loss("train").item()}
    if LOOP:
        rec.update(loop_eval())
        trained = model.core_transformer.n_loops
        same = rec[f"loopq_dense_{trained}"]
        print(f"LOOPS step {step:,} | dense " + " ".join(f"{n}:{rec[f'loopq_dense_{n}']:.5f}" for n in range(1, 7))
              + " | sparse " + " ".join(f"{n}:{rec[f'loopq_sparse_{n}']:.5f}" for n in range(1, 7))
              + " | update " + " ".join(f"{rec[f'upd_{k}']:.3f}" for k in range(1, 6))
              + " | cos " + " ".join(f"{rec[f'cos_{k}']:.4f}" for k in range(1, 6))
              + (" | xsa " + " ".join(f"{rec[f'xsa_{n}']:.3f}" for n in range(1, 7)) if "xsa_1" in rec else ""), flush=True)
        # the dense loss at the trained loop count is the test loss, computed another way: they must agree
        assert abs(same - rec["test_clean"]) <= 0.01 * rec["test_clean"], (same, rec["test_clean"])
    if PYR:
        rec.update(pyramid_eval())
        if TEACH:
            rec.update(teach_eval())
        print(f"READOUTS step {step:,} | test loss " + " ".join(f"{n}:{rec['test_' + n]:.5f}" for n in PYR_NAMES)
              + f" | final head {rec['test_final_check']:.5f}", flush=True)
        # the final head's loss here is the test loss, computed through the readout path: they must agree
        assert abs(rec["test_final_check"] - rec["test_clean"]) <= 0.01 * rec["test_clean"], (rec["test_final_check"], rec["test_clean"])
        del rec["test_final_check"]
    if PAIR and PYR:
        rec.update(st_judge_eval())
        print(f"JUDGE step {step:,} | self-teaching distance of one-pass outputs (not trained here): block 3 {rec['st_eval_b3']:.5f} | block 6 {rec['st_eval_b6']:.5f}", flush=True)
    rec.update(x_mse_by_t())
    rec["test_x_mse"] = statistics.fmean(v for k, v in rec.items() if k.startswith("x_mse_t"))
    if CORR:
        rec.update(corr_eval())
        print(f"CORRECTOR step {step:,} | test loss base {rec['test_base']:.5f}, corrector {rec['test_clean']:.5f} | test x-mse after k corrections: "
              + " ".join(f"{k_}:{rec[f'corr_mse_k{k_}']:.5f}" for k_ in range(9)) + " | at noise 0.9: "
              + " ".join(f"{k_}:{rec[f'corr_mse_n90_k{k_}']:.5f}" for k_ in (0, 1, 2, 4, 8)), flush=True)
        if step == EVAL_EVERY:
            # CHECK at the first eval: the two outputs are different predictions with different losses
            assert rec["test_base"] != rec["test_clean"] and rec["corr_mse_k0"] != rec["corr_mse_k1"] != rec["corr_mse_k2"], "base denoiser and corrector give the same output"
    # not logged here: train-viz wants one record per step, so the caller merges this into the step's record
    print(f"EVAL step {step:,} | test-loss {rec['test_clean']:.5f} | train-loss {rec['train_clean_eval']:.5f} | "
          f"test x-mse {rec['test_x_mse']:.5f} ({time.time() - t0:.1f}s)", flush=True)
    return rec


def image_detail(images):
    """Sharpness of a batch of images in [0, 1]: mean |Laplacian| (real 128 px images: about 0.093),
    fine = RMS of the difference to a 3 x 3 box blur, contrast = std of the pixel values."""
    import torch.nn.functional as F_
    x = images.float()
    lap = (4 * x[..., 1:-1, 1:-1] - x[..., :-2, 1:-1] - x[..., 2:, 1:-1] - x[..., 1:-1, :-2] - x[..., 1:-1, 2:]).abs().mean().item()
    blur = F_.avg_pool2d(x, 3, stride=1)
    fine = (x[..., 1:-1, 1:-1] - blur).square().mean().sqrt().item()
    # bright: mean pixel value. dark / light: share of the pixels below 0.02 / above 0.98 (samples that collapse to black or white)
    return {"sharp": lap, "fine": fine, "contrast": x.std().item(), "bright": x.mean().item(),
            "dark": (x < 0.02).float().mean().item(), "light": (x > 0.98).float().mean().item()}


# cfg["ema"] = decay (for example 0.999): keep an exponential moving average of the weights beside the
# trained ones and, at every preview, sample the same seeds from it too. Training never reads it.
EMA_DECAY = float(cfg.get("ema", 0) or 0)
ema_params = None
# cfg["ema_preview"] = decay (default 0.9998 when cfg ema is on; 0 = off) (user, 2026-10-13): a second, slower average of
# the weights, for pictures only. At every preview the fixed seeds are sampled from it too: emaprev_<step>.png, the
# same grid as recon_<step>.png, for the second "samples over training" view of train-viz. Nothing else reads it and
# it is not in the checkpoints. It starts with a short warm-up: decay = min(decay, (1 + n) / (10 + n)).
EMA2_DECAY = float(cfg.get("ema_preview", 0.9998 if EMA_DECAY else 0) or 0)
ema2_params, ema2_n = None, 0


def teacher_update():
    net = TEACHER["net"]
    if net is not None:
        with torch.no_grad():
            torch._foreach_lerp_([p for p in net.parameters()], [p.detach() for p in getattr(model, "_orig_mod", model).parameters()],
                                 1.0 - TEACHER["decay"])


def teacher_distance():
    """Relative distance between the averaged teacher and the trained weights (0 = the teacher is a plain copy)."""
    live = [p.detach().float() for p in getattr(model, "_orig_mod", model).parameters()]
    avg = [p.detach().float() for p in TEACHER["net"].parameters()]
    return (sum((a - b).square().sum() for a, b in zip(avg, live)).sqrt() / sum(b.square().sum() for b in live).sqrt()).item()


def ema_update():
    global ema_params, ema2_params, ema2_n
    params = [p for p in model.parameters()]
    with torch.no_grad():
        if ema_params is None:
            ema_params = [p.detach().clone() for p in params]
        else:
            torch._foreach_lerp_(ema_params, [p.detach() for p in params], 1.0 - EMA_DECAY)
        if EMA2_DECAY:
            if ema2_params is None:
                ema2_params = [p.detach().clone() for p in params]
            else:
                ema2_n += 1
                torch._foreach_lerp_(ema2_params, [p.detach() for p in params], 1.0 - min(EMA2_DECAY, (1 + ema2_n) / (10 + ema2_n)))


# cfg["jitter_probe"] = {"every": N} (needs cfg ema): which weights carry the swing of the samples?
# At every N-th preview, with w = the trained weights and e = their moving average:
#   dev      ||w - e|| / ||w|| per weight group: how far the group is from its average right now
#   change   ||w - w at the last probe|| / ||w||: how far it moved since then
#   raw_with_avg   sharpness of the samples when ONLY this group is set to its average
#   avg_with_raw   sharpness when all weights are the average and ONLY this group is the trained one
# If setting one group to its average brings the sharpness to the averaged model's value, that group
# carries the swing. One line per probe in /output/jitter.jsonl.
JIT = cfg.get("jitter_probe") or {}
_jit = {"last": None}


# cfg["grad_probe"] = [steps] (user, 2026-10-10: "measure issues with lower layers being trained more"). Measure only: no
# balancing. On one extra batch (own random generator, no optimizer step), per block:
#   the gradient norm of every loss term (each readout at its weight, the final loss),
#   the share of the final loss in the sum of these norms,
#   the cosine between the gradient of the final loss and the summed gradient of the readouts (negative = they conflict).
# grad_probe_gap steps later: the relative weight change per step of every block.
GRAD_PROBE = set(int(s_) for s_ in (cfg.get("grad_probe") or ()))
GRAD_GAP = int(cfg.get("grad_probe_gap", 100))
grad_snap = {}


def probe_groups():
    m = getattr(model, "_orig_mod", model)
    groups = [(f"block {i + 1}", list(b_.parameters())) for i, b_ in enumerate(m.core_transformer.blocks)]
    return groups + [("un-embed", list(m.un_embed.parameters()) + list(m.core_transformer.norm_out.parameters()))]


def grad_probe(images):
    import json as _json
    assert PYR and SCHED and not TEACH, "the gradient probe is written for the pyramid step without teaching"
    m = getattr(model, "_orig_mod", model)
    gen = torch.Generator(device=device).manual_seed(1234 + step)
    images = images.to(device)
    B = images.shape[0]
    t = shift_t(torch.sigmoid(torch.randn(B, device=device, dtype=torch.float32, generator=gen) * 1.5))
    noise = torch.randn(images.shape, device=device, dtype=images.dtype, generator=gen)
    t_img = t[:, None, None, None]
    noised = t_img * images + (1.0 - t_img) * noise
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=str(device) == "cuda"):
        losses = broad_losses(m(noised, t, all_readouts=True), images, noised, t_img)
    names = list(PYR_NAMES) + ["final"]
    weights = [float(g.get("pyramid_weight", 1.0))] * len(PYR_NAMES) + [1.0]
    groups = probe_groups()
    flat = [p_ for _, ps in groups for p_ in ps]
    per_loss = []
    for loss_k, w_k in zip(losses, weights):
        grads = torch.autograd.grad(w_k * loss_k.float(), flat, retain_graph=True, allow_unused=True)
        per_loss.append(grads)
    # regroup: per group, one flat vector per loss
    table, offset = [], 0
    for name, ps in groups:
        vecs = []
        for grads in per_loss:
            part = grads[offset:offset + len(ps)]
            vecs.append(torch.cat([(g_.float().reshape(-1) if g_ is not None else torch.zeros(p_.numel(), device=device)) for g_, p_ in zip(part, ps)]))
        offset += len(ps)
        norms = [float(v_.norm()) for v_ in vecs]
        final, readouts = vecs[-1], sum(vecs[:-1])
        cos = float(torch.nn.functional.cosine_similarity(final, readouts, dim=0)) if float(readouts.norm()) > 0 else None
        share = norms[-1] / max(sum(norms), 1e-30)
        table.append({"group": name, "norms": dict(zip(names, norms)), "final_share": share, "cos_final_readouts": cos})
        print(f"GRADPROBE step {step} | {name:8s} | final share {100 * share:5.1f}% | cos(final, readouts) "
              + (f"{cos:+.3f}" if cos is not None else "  none") + " | grad norm: " + " ".join(f"{n_} {v_:.2e}" for n_, v_ in zip(names, norms)), flush=True)
    grad_snap[step + GRAD_GAP] = [torch.cat([p_.detach().float().reshape(-1) for p_ in ps]).clone() for _, ps in groups]
    with open(os.path.join(os.environ.get("OUTPUT_DIR", "."), "grad_probe.jsonl"), "a") as f_:
        f_.write(_json.dumps({"step": step, "losses": dict(zip(names, [float(l_) for l_ in losses])), "groups": table}) + "\n")
    # the probe must see every readout in the blocks before it and none after it
    first = table[0]["norms"]
    assert all(v_ > 0 for v_ in first.values()), ("block 1 has no gradient from a loss", first)
    assert table[len(groups) - 2]["cos_final_readouts"] is None, "the last block gets a readout gradient"


def grad_probe_update():
    old = grad_snap.pop(step)
    change = []
    for (name, ps), before in zip(probe_groups(), old):
        now = torch.cat([p_.detach().float().reshape(-1) for p_ in ps])
        change.append((name, float((now - before).norm() / before.norm().clamp_min(1e-30)) / GRAD_GAP))
    print(f"GRADPROBE step {step} | relative weight change per step over the last {GRAD_GAP} steps: " + " | ".join(f"{n_} {v_:.2e}" for n_, v_ in change), flush=True)


def weight_group(name):
    import re
    m = re.match(r"core_transformer\.blocks\.(\d+)\.(.*)", name)
    if m:
        rest = m.group(2)
        kind = ("attn" if rest.startswith("MHA.") else "mlp" if rest.startswith("SwiGLU.") else "gain" if rest.startswith("post_norm")
                else "gate" if rest.startswith("scale_shift_gate") else "prenorm")
        return f"block{int(m.group(1)):02d}", "all " + kind
    for prefix, group in (("embed.", "embed"), ("un_embed.", "unembed"), ("timestep_embedder.", "time embed"), ("readout_heads.", "readout heads")):
        if name.startswith(prefix):
            return group, None
    return "core norms", None


@torch.no_grad()
def jitter_probe(step, sharp_raw, sharp_avg, seeds=range(12), sample_steps=20):
    t0 = time.time()
    names = [n for n, _ in model.named_parameters()]
    params = [p for _, p in model.named_parameters()]
    raw = [p.detach().clone() for p in params]
    members = {}
    for i, name in enumerate(names):
        for group in weight_group(name):
            if group:
                members.setdefault(group, []).append(i)

    def sharp_with(base, group, other):
        for p, b in zip(params, base):
            p.copy_(b)
        for i in members[group]:
            params[i].copy_(other[i])
        return image_detail(denoise_seeds(list(seeds), steps=sample_steps).float().cpu())["sharp"]

    out = {}
    try:
        for group, idx in sorted(members.items()):
            norm = sum(raw[i].float().square().sum() for i in idx).sqrt()
            dev = sum((raw[i] - ema_params[i]).float().square().sum() for i in idx).sqrt() / norm
            change = (sum((raw[i] - _jit["last"][i]).float().square().sum() for i in idx).sqrt() / norm).item() if _jit["last"] else None
            entry = {"dev": dev.item(), "change": change, "n": sum(raw[i].numel() for i in idx)}
            if group != "readout heads":            # the samples do not use them
                entry["raw_with_avg"] = sharp_with(raw, group, ema_params)
                entry["avg_with_raw"] = sharp_with(ema_params, group, raw)
            out[group] = entry
    finally:
        for p, r in zip(params, raw):
            p.copy_(r)
    _jit["last"] = raw
    with open(OUT / "jitter.jsonl", "a") as f:
        f.write(json.dumps({"step": step, "sharp_raw": sharp_raw, "sharp_avg": sharp_avg, "groups": out}) + "\n")
    print(f"jitter probe step {step:,}: {len(out)} weight groups, sharpness trained {sharp_raw:.4f} / averaged {sharp_avg:.4f} ({time.time() - t0:.0f}s)", flush=True)


# cfg["jitter_diff"] = true: which weights move WITH the swing of the samples?
# A copy of all weights is kept at every preview, with the sharpness of that preview's samples.
# At the end the copies are split into the sharper half and the blurrier half, and per weight tensor:
#   moves     RMS distance of the copies from their mean / ||w||       (how much the tensor jitters at all)
#   step      mean distance between consecutive copies / ||w||          (how fast)
#   split     ||mean of the sharp copies - mean of the blurry copies|| / ||w||
#   aligned   split, divided by what a random split of the same copies gives (mean of 30 random
#             splits). 1 = this tensor's jitter has nothing to do with sharpness; well above 1 = it moves with it.
# Written to /output/jitter_diff.json (one row per tensor).
JIT_DIFF = bool(cfg.get("jitter_diff"))
_snaps = []


@torch.no_grad()
def jitter_snapshot(sharp):
    _snaps.append(([p.detach().clone() for p in model.parameters()], float(sharp)))


@torch.no_grad()
def jitter_diff_report():
    import random
    n = len(_snaps)
    names = [name for name, _ in model.named_parameters()]
    order = sorted(range(n), key=lambda i: _snaps[i][1])
    blurry, sharp = set(order[:n // 2]), set(order[n - n // 2:])
    rng = random.Random(0)
    randoms = []
    for _ in range(30):
        idx = list(range(n))
        rng.shuffle(idx)
        randoms.append((set(idx[:n // 2]), set(idx[n - n // 2:])))

    def split_norm(stack, a, b):
        return (stack[sorted(a)].mean(0) - stack[sorted(b)].mean(0)).norm().item()

    rows = []
    for k, name in enumerate(names):
        stack = torch.stack([_snaps[i][0][k].float().flatten() for i in range(n)])        # (n, size)
        w = stack.mean(0).norm().item() + 1e-12
        # Training also moves the weights steadily in one direction. Take that straight-line drift out
        # first, or a chance link between sharpness and time would count as "moves with sharpness".
        ic = torch.arange(n, device=stack.device, dtype=torch.float32) - (n - 1) / 2
        slope = (ic[:, None] * stack).sum(0) / ic.square().sum()
        drift = slope.norm().item() * (n - 1) / w
        stack = stack - ic[:, None] * slope
        moves = (stack - stack.mean(0)).square().sum(1).mean().sqrt().item()
        step_change = (stack[1:] - stack[:-1]).norm(dim=1).mean().item()
        split = split_norm(stack, sharp, blurry)
        null = sum(split_norm(stack, a, b) for a, b in randoms) / len(randoms)
        rows.append({"name": name, "group": [g for g in weight_group(name) if g], "n": stack.shape[1], "norm": w, "drift": drift, "moves": moves / w,
                     "step": step_change / w, "split": split / w, "aligned": split / (null + 1e-12)})
    out = {"steps": None, "sharpness": [s for _, s in _snaps], "sharp_half": sorted(sharp), "blurry_half": sorted(blurry), "tensors": rows}
    (OUT / "jitter_diff.json").write_text(json.dumps(out))
    top = sorted(rows, key=lambda r: -r["aligned"])[:8]
    print(f"jitter diff: {n} weight copies, sharpness {min(out['sharpness']):.4f} to {max(out['sharpness']):.4f} | most aligned with sharpness: "
          + "; ".join(f"{r['name']} x{r['aligned']:.2f}" for r in top), flush=True)


Z_SNAP = cfg.get("z_snapshot")          # {"block": 6, "n": 128}: tokens after that block, for the Constellation view of train-viz
_z_images = []


def z_snapshot(step):
    """The tokens after block Z_SNAP["block"] of fixed clean test images (t = 1) as z_<step>.pt [N, T, C] in float16,
    and once their thumbnails as z_images.pt (uint8). The format of the AE runs: train-viz draws their PCA."""
    m = getattr(model, "_orig_mod", model)
    block, n = int(Z_SNAP.get("block", 6)), int(Z_SNAP.get("n", 128))
    if not _z_images:
        got = 0
        for images, _ in g["test_dataloader"]:
            _z_images.append(images[:n - got].clone())
            got += len(_z_images[-1])
            if got >= n:
                break
    kept = []
    hook = m.core_transformer.blocks[block - 1].register_forward_hook(lambda mod, inp, out: kept.append(out.detach().float().cpu()))
    was = m.training
    m.eval()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device) == "cuda"):
            for images in _z_images:
                images = images.to(device).float()
                m(images, torch.ones(len(images), device=device))
    finally:
        hook.remove()
        m.train(was)
    z = torch.cat(kept)
    assert z.dim() == 3 and len(z) == n and len(kept) == len(_z_images), (z.shape, len(kept))
    torch.save(z.half().contiguous(), OUT / f"z_{step:06d}.pt")
    publish(OUT / f"z_{step:06d}.pt")                 # as the previews: the runner of the 3090 hands over published files only
    if not Z_SNAP.get("_thumbs_done") and not (OUT / "z_images.pt").exists() and not (OUT / "published" / "z_images.pt").exists():
        thumbs = torch.nn.functional.interpolate(torch.cat(_z_images).float(), size=(64, 64), mode="area")
        torch.save((thumbs * 255).round().clamp(0, 255).to(torch.uint8).contiguous(), OUT / "z_images.pt")
        publish(OUT / "z_images.pt")
    Z_SNAP["_thumbs_done"] = True
    flat = z.flatten(1).double()
    lam = torch.linalg.svdvals(flat - flat.mean(0)).square()
    rank = (lam.sum().square() / lam.square().sum()).item()
    print(f"z snapshot step {step:,}: block {block}, {tuple(z.shape)}, std over images {flat.std(0).mean().item():.4f}, "
          f"participation rank {rank:.1f} of {n - 1}", flush=True)
    return {"z_std_mean": flat.std(0).mean().item(), "z_pr_rank": rank}


def preview_samples(seeds, sample_steps):
    """The preview samples of the current weights. Class-conditional runs: the fixed class of each seed, with guidance."""
    if COND:
        return cond_seeds(list(seeds), sample_steps, COND_PREVIEW_CFG).float().cpu()
    return denoise_seeds(list(seeds), steps=sample_steps).float().cpu()


def preview(step, seeds=range(12), sample_steps=20, ema=False):
    """Samples from fixed seeds as recon_<step>.png for train-viz. Returns their detail statistics
    for the step's record (prev_sharp, ...). ema=True: the same from the averaged weights (ema_prev_*)."""
    from PIL import Image
    if ema:
        params = [p for p in model.parameters()]
        with torch.no_grad():
            saved = [p.detach().clone() for p in params]
            for p, e in zip(params, ema2_params if ema == "slow" else ema_params):
                p.copy_(e)
        try:
            images = preview_samples(seeds, sample_steps)
        finally:
            with torch.no_grad():
                for p, s in zip(params, saved):
                    p.copy_(s)
        stats = image_detail(images)
        if ema == "slow":
            n, _, h, w = images.shape
            rows = (n + 5) // 6
            grid = torch.ones(3, rows * h + (rows - 1) * 4, 6 * w + 20)
            for i, image in enumerate(images):
                r, c = divmod(i, 6)
                grid[:, r * (h + 4):r * (h + 4) + h, c * (w + 4):c * (w + 4) + w] = image
            path = OUT / f"emaprev_{step:06d}.png"
            Image.fromarray((grid.permute(1, 2, 0).clamp(0, 1) * 255).round().byte().numpy()).save(path)
            publish(path)
            print(f"preview step {step:,}, slow average (decay {EMA2_DECAY:g}): {path.name} | sharpness {stats['sharp']:.4f}, contrast {stats['contrast']:.3f}, "
                  f"brightness {stats['bright']:.3f}", flush=True)
            return {**{"ema2_prev_" + k: v for k, v in stats.items()}, "ema_preview_decay": EMA2_DECAY}
        if cfg.get("ema_images") and not EMA2_DECAY:
            n, _, h, w = images.shape
            grid = torch.ones(3, 2 * h + 4, 6 * w + 20)
            for i, image in enumerate(images[:12]):
                r, c = divmod(i, 6)
                grid[:, r * (h + 4):r * (h + 4) + h, c * (w + 4):c * (w + 4) + w] = image
            Image.fromarray((grid.permute(1, 2, 0).clamp(0, 1) * 255).round().byte().numpy()).save(OUT / f"emaprev_{step:06d}.png")
        return {"ema_prev_" + k: v for k, v in stats.items()}
    images = preview_samples(seeds, sample_steps)
    samples, extra = images, {}
    if COND and step <= PREVIEW_EVERY:
        print(f"preview: class-conditional, guidance scale {COND_PREVIEW_CFG:g}, classes of the 12 seeds {[int(COND_PERM[int(s_) % COND_N]) for s_ in seeds]}", flush=True)
    if OT:
        # rows 3 and 4 of the grid: the output of ONE pass from fixed pure noise (t = 0), what the transport step trains
        m_ = getattr(model, "_orig_mod", model)
        was_ = m_.training
        m_.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device) == "cuda"):
            z_ = torch.randn(len(images), *images.shape[1:], generator=torch.Generator().manual_seed(1234)).to(device)
            one_ = m_(z_, torch.zeros(len(z_), device=device)).float().cpu()
        m_.train(was_)
        extra = {"ot_prev_std": one_.std(0).mean().item()}
        images = torch.cat([images, one_])
    n, _, h, w = images.shape
    cols = 6
    rows = (n + cols - 1) // cols
    grid = torch.ones(3, rows * h + (rows - 1) * 4, cols * w + (cols - 1) * 4)
    for i, image in enumerate(images):
        r, c = divmod(i, cols)
        grid[:, r * (h + 4):r * (h + 4) + h, c * (w + 4):c * (w + 4) + w] = image
    path = OUT / f"recon_{step:06d}.png"
    Image.fromarray((grid.permute(1, 2, 0).clamp(0, 1) * 255).round().byte().numpy()).save(path)
    publish(path)
    images = samples
    n = len(images)
    chroma = (images - images.mean(1, keepdim=True)).abs().mean().item()
    stats = image_detail(images)
    print(f"preview step {step:,}: {n} samples, {sample_steps} euler steps, colour {chroma:.3f}, sharpness {stats['sharp']:.4f}", flush=True)
    return {**{"prev_" + k: v for k, v in stats.items()}, **extra}


# cfg["contrast_probe"] = {"every": 50, "from": N} (user, 2026-10-13): where does the swing of brightness and contrast of
# the samples come from? From step N on, every "every" steps: the fixed seeds are sampled from the trained weights, and
# the probe keeps (1) brightness, contrast and the share of black / white pixels of the samples, (2) mean and RMS of the
# output of every module, at the first sampler step (pure noise in: the same input at every snapshot) and over all
# sampler steps, (3) a copy of all weights. At the last step it writes contrast_probe.json and prints, for contrast
# and for brightness: the activations and the weight tensors that move most with it, the largest differences between
# the highest and the lowest snapshot, and a swap test: the samples of the highest snapshot with ONE weight group
# taken from the lowest one (how much of the gap does that group close?).
CPROBE = cfg.get("contrast_probe") or {}
_cp = []


def _cp_sample(seeds, sample_steps, record=None):
    """Samples of the fixed seeds. record: dict that gets {module name: [first mean, first rms, all mean, all rms]}."""
    m = getattr(model, "_orig_mod", model)
    acc, calls, hooks = {}, {"n": 0}, []
    if record is not None:
        def count(module, args):
            calls["n"] += 1
        hooks.append(m.register_forward_pre_hook(count))

        def make(name):
            def hook(module, args, out):
                outs = out if isinstance(out, (tuple, list)) else (out,)
                for i_, o_ in enumerate(outs):
                    if not torch.is_tensor(o_) or not o_.is_floating_point():
                        continue
                    key = name if len(outs) == 1 else f"{name}#{i_}"
                    o_ = o_.detach().float()
                    mean, rms = o_.mean().item(), o_.square().mean().sqrt().item()
                    a_ = acc.setdefault(key, [mean, rms, 0.0, 0.0, 0])
                    if calls["n"] == 1:
                        a_[0], a_[1] = mean, rms
                    a_[2] += mean; a_[3] += rms; a_[4] += 1
            return hook
        for name, module in m.named_modules():
            if name:
                hooks.append(module.register_forward_hook(make(name)))
    try:
        images = denoise_seeds(list(seeds), steps=sample_steps).float().cpu()
    finally:
        for h_ in hooks:
            h_.remove()
    if record is not None:
        record.update({k_: [v_[0], v_[1], v_[2] / v_[4], v_[3] / v_[4]] for k_, v_ in acc.items()})
    return images


@torch.no_grad()
def contrast_snapshot(step, seeds=range(12), sample_steps=20):
    from PIL import Image
    acts = {}
    images = _cp_sample(seeds, sample_steps, acts)
    stats = image_detail(images)
    per_image = [[im.mean().item(), im.std().item()] for im in images]
    n, _, h, w = images.shape
    rows = (n + 5) // 6
    grid = torch.ones(3, rows * h + (rows - 1) * 4, 6 * w + 20)
    for i, image in enumerate(images):
        r, c = divmod(i, 6)
        grid[:, r * (h + 4):r * (h + 4) + h, c * (w + 4):c * (w + 4) + w] = image
    Image.fromarray((grid.permute(1, 2, 0).clamp(0, 1) * 255).round().byte().numpy()).save(OUT / f"cprobe_{step:06d}.png")
    weights = [p_.detach().clone() for p_ in model.parameters()]
    _cp.append({"step": step, "stats": stats, "per_image": per_image, "acts": acts, "weights": weights})
    print(f"contrast probe step {step:,}: brightness {stats['bright']:.3f} | contrast {stats['contrast']:.3f} | black {100 * stats['dark']:.1f}% | "
          f"white {100 * stats['light']:.1f}% | sharpness {stats['sharp']:.4f} | {len(acts)} activations", flush=True)


@torch.no_grad()
def contrast_report(seeds=range(12), sample_steps=20):
    n = len(_cp)
    names = [name for name, _ in model.named_parameters()]
    params = [p_ for _, p_ in model.named_parameters()]
    ic = torch.arange(n, dtype=torch.float64) - (n - 1) / 2

    def detrend(v):                       # training drifts steadily: take the straight line out before any comparison
        v = torch.as_tensor(v, dtype=torch.float64)
        return v - v.mean() - ic * (ic * (v - v.mean())).sum() / ic.square().sum()

    def corr(a, b):
        a, b = detrend(a), detrend(b)
        return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))

    out = {"steps": [c_["step"] for c_ in _cp], "samples": {k_: [c_["stats"][k_] for c_ in _cp] for k_ in _cp[0]["stats"]},
           "per_image": [c_["per_image"] for c_ in _cp], "targets": {}}
    act_keys = sorted(set.intersection(*(set(c_["acts"]) for c_ in _cp)))
    kinds = ("first step mean", "first step rms", "all steps mean", "all steps rms")
    for target in ("contrast", "bright"):
        y = out["samples"][target]
        hi, lo = max(range(n), key=lambda i: y[i]), min(range(n), key=lambda i: y[i])
        acts = []
        for key in act_keys:
            for j_, kind in enumerate(kinds):
                series = [c_["acts"][key][j_] for c_ in _cp]
                scale = sum(abs(v_) for v_ in series) / n + 1e-12
                acts.append({"module": key, "stat": kind, "corr": corr(series, y), "swing": float(detrend(series).std()) / scale,
                             "high": series[hi], "low": series[lo], "rel_diff": (series[hi] - series[lo]) / scale})
        weights = []
        for k_, name in enumerate(names):
            stack = torch.stack([c_["weights"][k_].float().flatten() for c_ in _cp]).double().cpu()
            norm = stack.mean(0).norm().item() + 1e-12
            means = stack.mean(1)
            rmss = stack.square().mean(1).sqrt()
            # direction test: project every copy on (highest - lowest); a tensor that carries the swing has a projection that follows y
            d = stack[hi] - stack[lo]
            proj = (stack @ d) / (d.norm() + 1e-30)
            weights.append({"name": name, "n": stack.shape[1], "hi_lo_dist": d.norm().item() / norm,
                            "step_dist": (stack[1:] - stack[:-1]).norm(dim=1).mean().item() / norm,
                            "corr_mean": corr(means, y), "corr_rms": corr(rmss, y), "corr_proj": corr(proj, y),
                            "mean_high": means[hi].item(), "mean_low": means[lo].item(), "rms_high": rmss[hi].item(), "rms_low": rmss[lo].item()})
        # swap test: all weights of the highest snapshot, ONE group from the lowest
        members = {}
        for i, name in enumerate(names):
            for group in weight_group(name):
                if group and group != "readout heads":
                    members.setdefault(group, []).append(i)
        kept = [p_.detach().clone() for p_ in params]
        swaps = {}
        try:
            for group, idx in sorted(members.items()):
                for p_, w_ in zip(params, _cp[hi]["weights"]):
                    p_.copy_(w_)
                for i in idx:
                    params[i].copy_(_cp[lo]["weights"][i])
                value = image_detail(_cp_sample(seeds, sample_steps))[target]
                swaps[group] = {"value": value, "gap_closed": (y[hi] - value) / (y[hi] - y[lo] + 1e-12)}
        finally:
            for p_, k2 in zip(params, kept):
                p_.copy_(k2)
        out["targets"][target] = {"high_step": _cp[hi]["step"], "low_step": _cp[lo]["step"], "high": y[hi], "low": y[lo],
                                  "activations": acts, "weights": weights, "swap": swaps}
        print(f"CONTRAST PROBE {target}: {n} snapshots, {min(y):.3f} to {max(y):.3f} (highest at step {_cp[hi]['step']:,}, lowest at step {_cp[lo]['step']:,})", flush=True)
        for r_ in sorted(acts, key=lambda r_: -abs(r_["corr"]))[:14]:
            print(f"  activation corr {r_['corr']:+.2f} | {r_['module']} [{r_['stat']}] | high {r_['high']:.4g}, low {r_['low']:.4g} | swing {100 * r_['swing']:.1f}%", flush=True)
        for r_ in sorted(acts, key=lambda r_: -abs(r_["rel_diff"]))[:10]:
            print(f"  activation high-low {100 * r_['rel_diff']:+.0f}% | {r_['module']} [{r_['stat']}] | high {r_['high']:.4g}, low {r_['low']:.4g} | corr {r_['corr']:+.2f}", flush=True)
        for r_ in sorted(weights, key=lambda r_: -max(abs(r_["corr_mean"]), abs(r_["corr_rms"])))[:10]:
            print(f"  weight corr mean {r_['corr_mean']:+.2f} rms {r_['corr_rms']:+.2f} | {r_['name']} ({r_['n']:,}) | mean high {r_['mean_high']:.4g}, low {r_['mean_low']:.4g} | "
                  f"distance high-low {100 * r_['hi_lo_dist']:.2f}% of the norm", flush=True)
        for r_ in sorted(weights, key=lambda r_: -r_["hi_lo_dist"])[:8]:
            print(f"  weight distance high-low {100 * r_['hi_lo_dist']:.2f}% of the norm (one probe interval: {100 * r_['step_dist']:.2f}%) | {r_['name']} ({r_['n']:,})", flush=True)
        for group, r_ in sorted(swaps.items(), key=lambda kv: -abs(kv[1]["gap_closed"])):
            print(f"  swap {group}: {target} {r_['value']:.3f} | closes {100 * r_['gap_closed']:+.0f}% of the gap", flush=True)
    (OUT / "contrast_probe.json").write_text(json.dumps(out))
    print("contrast probe: contrast_probe.json written", flush=True)


def checkpoint(step, final=False):
    path = OUT / (f"checkpoint_final_step_{step:06d}.pt" if final else f"checkpoint_step_{step:06d}.pt")
    state = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
    # "ema": the averaged weights (cfg ema), in the order of model.parameters(), so samples and scores can use them later
    torch.save({"model": state, "optimizer": optimizer.state_dict(), "step": step, "config": cfg,
                "ema": [e.detach().cpu() for e in ema_params] if ema_params is not None else None, "ema_decay": EMA_DECAY or None}, path)
    if final and cfg.get("resume_copy"):
        # cfg["resume_copy"]: the 3090 runner deletes a published file after the Spark has collected it. A later job
        # (submit --resume-from) can continue only from a file that stays in the output folder: weights and averaged
        # weights, no optimizer state (resume it with resume_weights_only).
        # 2026-10-13: the copy keeps the optimizer state (user: "save optimizer state w/ checkpoints"; the MoE runs start
        # from one shared dense base and the upcycling paper keeps the optimizer state)
        import shutil
        shutil.copyfile(path, OUT / "resume_final.pt")
        print(f"resume copy kept in the output folder (resume_final.pt, {(OUT / 'resume_final.pt').stat().st_size / 2**20:.0f} MiB)", flush=True)
    # The rolling resume file (see AUTO-RESUME at the top): the same content, written to a temporary name and then renamed,
    # so a stop during the write leaves the earlier file whole. A finished run has nothing to resume: its file is removed.
    if final:
        (OUT / "resume_latest.pt").unlink(missing_ok=True)
    else:
        import shutil
        shutil.copyfile(path, OUT / "resume_latest.pt.tmp")
        os.replace(OUT / "resume_latest.pt.tmp", OUT / "resume_latest.pt")
    publish(path)
    print(f"checkpoint step {step:,} published ({path.name})", flush=True)


# ---------------------------------------------------------------- yielding the GPU
class GpuYield:
    """Pause training while another workload is using the GPU.

    The job cannot be signalled from outside, so it watches its own step time. Every 10th
    step is timed on the GPU (synchronised). The first timed steps after warm-up give the
    normal step time. If recent steps take much longer, something else is computing on the
    GPU: the job stops stepping, sleeps, and probes again with a few steps until the step
    time is back to normal. Weights stay in memory, so nothing is lost."""
    def __init__(self):
        self.on = bool(cfg.get("yield_gpu", True))
        self.base = cfg.get("gpu_step_ms")
        self.samples, self.recent, self.paused_s, self.pauses = [], [], 0.0, 0

    def timed(self, step):
        return step % 10 == 0

    def add(self, step, ms):
        if step < 300:                                   # compile and warm-up
            return
        if self.base is None:
            self.samples.append(ms)
            if len(self.samples) >= 30:
                self.base = statistics.median(self.samples)
                print(f"gpu-yield: normal step time {self.base:.1f} ms (median of 30 timed steps)", flush=True)
            return
        self.recent = (self.recent + [ms])[-12:]

    def busy(self):
        return self.on and self.base is not None and len(self.recent) >= 12 and statistics.median(self.recent) > 1.6 * self.base

    def wait(self, step, probe):
        """Called when busy(): sleep, then run `probe` (a few timed training steps) until the GPU is free."""
        self.pauses += 1
        t0 = time.time()
        print(f"gpu-yield: PAUSED at step {step:,}: steps take {statistics.median(self.recent):.0f} ms, normal {self.base:.0f} ms", flush=True)
        while True:
            time.sleep(float(cfg.get("yield_sleep_s", 90)))
            ms = probe()
            if statistics.median(ms) < 1.3 * self.base:
                break
            print(f"gpu-yield: still busy ({statistics.median(ms):.0f} ms per step), waiting", flush=True)
        self.paused_s += time.time() - t0
        self.recent = []
        print(f"gpu-yield: RESUMED after {(time.time() - t0) / 60:.1f} min (paused {self.paused_s / 60:.1f} min in total)", flush=True)


# ---------------------------------------------------------------- the loop
print(f"run: xm_k={XM_K} xm_n={XM_N} | steps={STEPS:,} | batch {g['CACHE_BATCH_SIZE']} | lr {g['lr']} | "
      f"blocks {g['num_blocks']} emb {g['emb_dim']} keep_ratio {g['keep_ratio']} | compile {cfg.get('compile', True)} | "
      f"{sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
model.train()
yielder = GpuYield()
# cfg["traj_probe"] = {"ckpts": [paths], "n": 96, "steps": 20} (user, 2026-10-13: "the 1 step prediction doesn't blacken /
# whiten pixels as much as multi-step does?"). No training. For the trained and the averaged weights of each checkpoint:
#   one pass:  the prediction from noised REAL test images at noise levels 1.0 .. 0.1 (what training sees)
#   sampler:   the prediction and the state at every Euler step from fixed seeds (what sampling sees)
# Measures of a set of images: brightness, spread between images (std of the image means), contrast inside an image,
# share of pixels below 0.02 / above 0.98 after the clamp to 0..1, and the share the model put OUTSIDE 0..1.
# Also per seed: the mean of the starting noise against the brightness of the final image (slope and correlation).
if cfg.get("traj_probe"):
    TP = cfg["traj_probe"]
    tp_n, tp_steps = int(TP.get("n", 96)), int(TP.get("steps", 20))
    m_ = getattr(model, "_orig_mod", model)

    def tp_meas(x):
        c = x.clamp(0, 1)
        means = c.mean((1, 2, 3))
        return {"bright": c.mean().item(), "between": means.std(unbiased=False).item(), "within": c.std((1, 2, 3)).mean().item(),
                "dark": (c < 0.02).float().mean().item(), "light": (c > 0.98).float().mean().item(),
                "below0": (x < 0).float().mean().item(), "above1": (x > 1).float().mean().item(), "raw_std": x.std().item()}

    def tp_line(tag, r):
        return (f"  {tag:34s} bright {r['bright']:.3f} | between {r['between']:.3f} | within {r['within']:.3f} | black {100 * r['dark']:5.1f}% | white {100 * r['light']:5.1f}% | "
                f"below 0: {100 * r['below0']:5.1f}% | above 1: {100 * r['above1']:5.1f}%")

    @torch.inference_mode()
    def tp_run(label):
        size = g["IMAGE_SIZE"]
        out = {}
        real = []
        for images, _ in g["test_dataloader"]:
            real.append(images)
            if sum(len(r_) for r_ in real) >= tp_n:
                break
        real = torch.cat(real)[:tp_n].to(device).float()
        gen = torch.Generator(device=device).manual_seed(4242)
        noise = torch.randn(real.shape, generator=gen, device=device)
        out["real"] = tp_meas(real)
        print(f"TRAJ {label}: {tp_n} images", flush=True)
        print(tp_line("real test images", out["real"]), flush=True)
        for level in (1.0, 0.95, 0.9, 0.7, 0.5, 0.3, 0.1):
            t = torch.full((tp_n,), 1.0 - level, device=device)
            preds = []
            for i in range(0, tp_n, 48):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    preds.append(m_(((1.0 - level) * real[i:i + 48] + level * noise[i:i + 48]), t[i:i + 48]).float())
            pred = torch.cat(preds)
            r = tp_meas(pred)
            r["mse"] = (pred - real).square().mean().item()
            r["mean_err"] = (pred.clamp(0, 1).mean((1, 2, 3)) - real.mean((1, 2, 3))).abs().mean().item()
            out[f"one_pass_{level}"] = r
            print(tp_line(f"one pass, real at noise level {level:g}", r) + f" | error of the image mean {r['mean_err']:.3f}", flush=True)
        seeds = list(range(tp_n))
        x = torch.stack([torch.randn((m_.un_embed.c, size, size), generator=torch.Generator(device=device).manual_seed(int(s_)), device=device) for s_ in seeds])
        noise_mean = x.mean((1, 2, 3))
        ts = g["make_timesteps"](tp_steps, "linear", 1.0)
        first_mean = None
        for k_, (t, t_next) in enumerate(zip(ts[:-1], ts[1:])):
            preds = []
            for i in range(0, tp_n, 48):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    preds.append(m_(x[i:i + 48], t.expand(len(x[i:i + 48]))).float())
            pred = torch.cat(preds)
            if first_mean is None:
                first_mean = pred.clamp(0, 1).mean((1, 2, 3))
            r = tp_meas(pred)
            out[f"sampler_{k_ + 1}"] = r
            if k_ + 1 in (1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, tp_steps):
                print(tp_line(f"sampler step {k_ + 1} of {tp_steps} (noise {1 - float(t):.2f})", r), flush=True)
            x = x + (t_next - t) * (pred - x) / (1 - t)
        r = tp_meas(x)
        out["final"] = r
        print(tp_line("final samples", r), flush=True)
        final_mean = x.clamp(0, 1).mean((1, 2, 3))

        def fit(a, b):
            a_, b_ = a - a.mean(), b - b.mean()
            return (a_ * b_).sum().item() / (a_.square().sum().item() + 1e-30), (a_ * b_).sum().item() / (a_.norm().item() * b_.norm().item() + 1e-30)
        s1, c1 = fit(noise_mean, final_mean)
        s2, c2 = fit(first_mean, final_mean)
        s3, c3 = fit(noise_mean, first_mean)
        out["coupling"] = {"noise_to_final": [s1, c1], "first_to_final": [s2, c2], "noise_to_first": [s3, c3], "noise_mean_std": noise_mean.std().item()}
        print(f"  per seed: mean of the starting noise (std {noise_mean.std().item():.4f}) -> brightness of the final image: slope {s1:.1f}, correlation {c1:+.2f} | "
              f"-> brightness of the first prediction: slope {s3:.1f}, correlation {c3:+.2f} | first prediction -> final image: slope {s2:.2f}, correlation {c2:+.2f}", flush=True)
        return out

    model.eval()
    tp_out = {}
    for path in TP["ckpts"]:
        saved = torch.load(path, map_location=device, weights_only=False)
        m_.load_state_dict(saved["model"])
        tp_out[f"{Path(path).name} trained"] = tp_run(f"{Path(path).name}, trained weights")
        if saved.get("ema") is not None:
            with torch.no_grad():
                for p_, e_ in zip(model.parameters(), saved["ema"]):
                    p_.copy_(e_.to(device))
            tp_out[f"{Path(path).name} averaged"] = tp_run(f"{Path(path).name}, averaged weights")
        del saved
    (OUT / "traj_probe.json").write_text(json.dumps(tp_out))
    print("traj probe: done", flush=True)
    raise SystemExit(0)
# cfg["gen_eval_only"] = [checkpoint paths]: score existing checkpoints of this run and stop.
# One record per checkpoint goes to cfg["gen_eval_out"] (default: this run's metrics.jsonl).
if cfg.get("gen_eval_only"):
    target = Path(cfg.get("gen_eval_out") or OUT / "metrics.jsonl")
    for path in cfg["gen_eval_only"]:
        saved = torch.load(path, map_location=device, weights_only=False)
        getattr(model, "_orig_mod", model).load_state_dict(saved["model"])
        if MOE:
            moe_set_active(int(saved["step"]) > MOE_AT)        # a checkpoint scored after the fact. The checkpoint of step "at" is saved before the conversion: dense
        ema_params = [e.to(device) for e in saved["ema"]] if saved.get("ema") is not None else None
        if cfg.get("z_only"):
            # snapshots after the fact (z_snapshot of a finished run): no samples, no FID. The files go beside the records
            import shutil
            rec = {"step": int(saved["step"]), **z_snapshot(int(saved["step"]))}
            for f_ in list(OUT.glob("z_*.pt")) + list((OUT / "published").glob("z_*.pt")):
                shutil.copy(f_, target.parent / f_.name)
            with open(target, "a") as f:
                f.write(json.dumps(rec) + "\n")
            del saved
            continue
        for ig_ in (GEN.get("ig_list") or [None]):
            if ig_:
                GEN["ig"] = ig_
            rec = {"step": int(saved["step"]), **gen_scores(int(saved["step"]))}
            with open(target, "a") as f:
                f.write(json.dumps(rec) + "\n")
        del saved
    print("gen eval only: done", flush=True)
    sys.stdout.flush()
    os._exit(0)          # the persistent loader workers keep a normal exit waiting

def moe_adapt_dense(saved, weights_only):
    """A MoE run that resumes from a DENSE checkpoint (user, 2026-10-13: all upcycling runs start from one pretrained dense
    run). Returns `saved` in the layout of this model: the dense MLP of every MoE block repeated over the experts (weights,
    averaged weights, optimizer moments). The routers keep their new init. A checkpoint that is MoE already comes back as is."""
    import re
    raw = getattr(model, "_orig_mod", model)
    dense, cur_state = saved["model"], raw.state_dict()
    pat = re.compile(r"(.*\.SwiGLU)\.(gate_proj|up_proj|down_proj)_(w|b)$")

    def dense_name(key):                 # -> (name in the dense checkpoint or None for a router, stacked over the experts?)
        m_ = pat.match(key)
        if m_ and key not in dense:
            return f"{m_.group(1)}.{m_.group(2)}.{'weight' if m_.group(3) == 'w' else 'bias'}", True
        if ".SwiGLU.router." in key and key not in dense:
            return None, False
        return key, False

    if not any(dense_name(k_)[1] for k_ in cur_state):
        return saved                     # the checkpoint has the experts already

    def stack(v):
        return v[None].repeat(MOE_E, *([1] * v.dim()))

    new_model, used, n_stacked = {}, set(), 0
    for key, value in cur_state.items():
        name, stacked = dense_name(key)
        if name is None:
            new_model[key] = value.detach().clone()
            continue
        assert name in dense, f"the dense checkpoint has no {name} (wanted for {key})"
        new_model[key] = stack(dense[name]) if stacked else dense[name]
        assert new_model[key].shape == value.shape, (key, tuple(new_model[key].shape), tuple(value.shape))
        used.add(name)
        n_stacked += stacked
    assert used == set(dense), f"keys of the dense checkpoint without a place in this model: {sorted(set(dense) - used)[:8]}"
    cur_params = [(n_, p_) for n_, p_ in raw.named_parameters()]
    position = {k_: i_ for i_, k_ in enumerate(dense)}
    dense_order = sorted({dense_name(n_)[0] for n_, _ in cur_params} - {None}, key=position.__getitem__)      # = parameters() of the dense model
    index_of = {n_: i_ for i_, n_ in enumerate(dense_order)}
    out = dict(saved, model=new_model)
    if saved.get("ema") is not None:
        assert len(saved["ema"]) == len(dense_order), (len(saved["ema"]), len(dense_order))
        assert all(e_.shape == dense[n_].shape for e_, n_ in zip(saved["ema"], dense_order)), "the averaged weights are not in the order of the dense parameters"
        ema = []
        for n_, p_ in cur_params:
            name, stacked = dense_name(n_)
            ema.append(p_.detach().clone() if name is None else stack(saved["ema"][index_of[name]]) if stacked else saved["ema"][index_of[name]])
        out["ema"] = ema
    moments = False
    if saved.get("optimizer") is not None and not weights_only:
        opt = saved["optimizer"]
        assert len(opt["param_groups"]) == 1 and len(opt["param_groups"][0]["params"]) == len(dense_order), "the optimizer of the checkpoint is not over the dense parameters"
        ids = opt["param_groups"][0]["params"]
        state = {}
        for j_, (n_, p_) in enumerate(cur_params):
            name, stacked = dense_name(n_)
            st = None if name is None else opt["state"].get(ids[index_of[name]])
            if st is None:
                continue                 # a router (or a parameter without moments): Adam starts its state at the first step
            state[j_] = {k_: (stack(v_) if stacked and torch.is_tensor(v_) and v_.shape == dense[name].shape else v_) for k_, v_ in st.items()}
        out["optimizer"] = {"state": state, "param_groups": [dict(opt["param_groups"][0], params=list(range(len(cur_params))))]}
        moments = True
    # proof on the model itself: load, then compare every expert of every MoE tensor with the dense tensor
    raw.load_state_dict(new_model)
    diff = max((p_.detach()[e_] - dense[dense_name(n_)[0]].to(p_.device)).abs().max().item()
               for n_, p_ in cur_params if dense_name(n_)[1] for e_ in range(MOE_E))
    assert diff == 0.0, diff
    print(f"CHECK moe resume: dense checkpoint of step {int(saved['step']):,} loaded into {MOE_E} experts | {len(new_model)} weight tensors, {n_stacked} stacked | "
          f"optimizer moments restored: {'yes' if moments else 'no'} | averaged weights: {'yes' if saved.get('ema') is not None else 'no'} | "
          f"largest difference of an expert to the dense MLP {diff:.1e} | the routing is off until the conversion: the test loss of the RESUME line "
          f"must equal the test loss of the dense run at this step", flush=True)
    return out


step, seconds_before = 0, 0.0
if RESUME_CKPT:
    saved = torch.load(RESUME_CKPT, map_location=device, weights_only=False)
    # cfg["resume_weights_only"]: continue from a checkpoint that has no optimizer state (every checkpoint
    # but the final one). The optimizer then starts new, with the learning rate of this config.
    weights_only = bool(cfg.get("resume_weights_only")) and not AUTO_RESUME      # the rolling resume file holds the optimizer state
    if AUTO_RESUME and (Path(AUTO_RESUME).parent / "metrics.jsonl").exists():
        # metrics.jsonl goes on without duplicates: keep the records up to the resume step (the same folder: the records
        # after that step are logged again; /resume: the history of the interrupted job comes into this job's file)
        _keep = []
        for _line in (Path(AUTO_RESUME).parent / "metrics.jsonl").read_text().splitlines():
            try:
                if json.loads(_line)["step"] <= int(saved["step"]):
                    _keep.append(_line + "\n")
            except (ValueError, KeyError, TypeError):
                pass                     # a line cut off by the stop
        (OUT / "metrics.jsonl.tmp").write_text("".join(_keep))
        os.replace(OUT / "metrics.jsonl.tmp", OUT / "metrics.jsonl")
    assert saved["optimizer"] is not None or weights_only, "this checkpoint has no optimizer state: resume from a final checkpoint"
    if MOE:
        saved = moe_adapt_dense(saved, weights_only)      # a dense checkpoint: its MLPs repeated over the experts
    getattr(model, "_orig_mod", model).load_state_dict(saved["model"])
    if not weights_only:
        optimizer.load_state_dict(saved["optimizer"])
    if EMA_DECAY and saved.get("ema") is not None:
        ema_params = [e.to(device) for e in saved["ema"]]          # the averaged weights go on from the checkpoint
        if EMA2_DECAY:
            ema2_params, ema2_n = [e.clone() for e in ema_params], 10 ** 9      # the slow average is not saved: it starts from this one
    if TEACHER["net"] is not None:
        # the averaged teacher starts from the averaged weights of the checkpoint (or from its weights)
        TEACHER["net"].load_state_dict(saved["model"])
        if saved.get("ema") is not None:
            with torch.no_grad():
                for p_, e_ in zip(TEACHER["net"].parameters(), saved["ema"]):
                    p_.copy_(e_.to(device))
    if cfg.get("resume_lr"):
        # a probe may go on with another learning rate; the restored optimizer state (moments) is kept
        for group in optimizer.param_groups:
            group["lr"] = float(cfg["resume_lr"])
    step = int(saved["step"])
    old = [json.loads(l) for l in (OUT / "metrics.jsonl").read_text().splitlines()] if (OUT / "metrics.jsonl").exists() else []
    seconds_before = max((r.get("train_seconds", 0.0) for r in old if r["step"] <= step), default=0.0)
    check = evaluate(step)
    logged = next((r["test_clean"] for r in reversed(old) if r["step"] == step and "test_clean" in r), None)
    print(f"RESUME from {RESUME_CKPT} at step {step:,} of {STEPS:,} | optimizer {'NEW' if weights_only else 'restored'} (lr {float(optimizer.param_groups[0]['lr']):g}) | "
          f"test loss now {check['test_clean']:.5f}, logged at that step {logged if logged is None else round(logged, 5)}", flush=True)
    # the restored weights must give the test loss the run logged at this step
    assert logged is None or abs(check["test_clean"] - logged) <= 0.01 * logged, (check["test_clean"], logged)
    del saved
else:
    log_metrics(0, **evaluate(0))
acc, acc_all, acc_n = None, None, 0
first_step = step + 1

# cfg["lr_switch"] = {"step": N, "lr": X}: the learning rate changes once, after N steps. The rate is held
# in a tensor that is filled in place, so the compiled step reads the new value (a plain float can stay
# baked into the compiled code). The change is checked on the weights: see CHECK lr switch below.
LR_SWITCH = cfg.get("lr_switch") or {}
if LR_SWITCH:
    lr_now = torch.tensor(float(optimizer.param_groups[0]["lr"]), device=device)
    for group in optimizer.param_groups:
        group["lr"] = lr_now
    if step >= int(LR_SWITCH["step"]):
        lr_now.fill_(float(LR_SWITCH["lr"]))
    lr_probe_param = max(model.parameters(), key=lambda p: p.numel())
    lr_probe = {}
    print(f"lr switch: {float(lr_now):g} now, {float(LR_SWITCH['lr']):g} from step {int(LR_SWITCH['step']):,}", flush=True)
t_run = t_window = time.time()
data_wait_s = 0.0
loader = iter(g["train_dataloader"])


cond_acc = {"none": 0, "n": 0}

if PAIR:
    def _dsum(x_):
        return float(x_.double().sum())

    def _wsum(x_):
        # order-sensitive: row i of the batch counts i + 1 times
        x_ = x_.double().reshape(x_.shape[0], -1)
        return float((x_.sum(1) * torch.arange(1, x_.shape[0] + 1, device=x_.device, dtype=torch.float64)).sum())

    def pair_fingerprint(at, images, pr, after=False):
        """One line of pairing.jsonl; with a reference, the CHECK against the baseline's line of the same step and kind.
        kind init (step 0): the weights before training. batch (steps 1, 2, every PAIR_EVERY): the images and the drawn
        numbers of that step. after (step 1): the weights after the first update."""
        raw_ = getattr(model, "_orig_mod", model)
        if images is None:
            with torch.no_grad():
                rec_ = {"step": at, "kind": "after" if after else "init",
                        "params": sum(_dsum(p_.detach()) for p_ in raw_.parameters()),
                        "params_abs": sum(float(p_.detach().double().abs().sum()) for p_ in raw_.parameters())}
        else:
            rec_ = {"step": at, "kind": "batch", "img": _dsum(images), "img_w": _wsum(images), "u": _wsum(pr["u"]),
                    "noise": _dsum(pr["noise"]), "noise_w": _wsum(pr["noise"]), "crop": _wsum(pr["crop"]),
                    "st_log": float(pr["st_log"]), "rope": float(ROPE_LOG)}
            if "u2" in pr:
                rec_.update(u2=_wsum(pr["u2"]), mask=_wsum(pr["mask"]))
        rec_["resumed"] = PAIR_STATE["resumed"]
        with open(OUT / "pairing.jsonl", "a") as f_:
            f_.write(json.dumps(rec_) + "\n")
        ref_ = PAIR_REF.get((at, rec_["kind"]))
        if ref_ is None or PAIR_STATE["resumed"]:
            return
        keys_ = [k_ for k_ in rec_ if k_ not in ("step", "kind", "resumed")]
        differ_ = [k_ for k_ in keys_ if abs(rec_[k_] - ref_[k_]) > 1e-12 * max(1.0, abs(ref_[k_]))]
        PAIR_STATE["checked"] += 1
        if rec_["kind"] == "after":
            # the treatment: with self-teaching the first update differs from the baseline's (same init, batch, noise)
            gap_ = abs(rec_["params_abs"] - ref_["params_abs"])
            print(f"CHECK pairing treatment at step 1: weights after the first update differ from the baseline's: {bool(differ_)} "
                  f"(sum of |w| {rec_['params_abs']:.9e} vs {ref_['params_abs']:.9e}, gap {gap_:.3e}) | self-teaching on: {bool(SELFTEACH)}", flush=True)
            # GPU kernels are not bit-exact between runs (two baseline smoke runs differed in the 5th digit), so equal
            # weights cannot be required of a run without self-teaching; the gradient path is proven by CHECK self-teaching
            # gradient (step 0) and CHECK self-teaching (step 100). Here: a self-teaching run must not equal its baseline.
            assert bool(differ_) or not SELFTEACH, "self-teaching did not change the first update"
            return
        if at <= 2 or PAIR_STATE["checked"] % 10 == 0 or differ_:
            print(f"CHECK pairing at step {at:,} ({rec_['kind']}): {len(keys_) - len(differ_)} of {len(keys_)} fingerprints equal to the baseline's"
                  + (f" | DIFFERENT: {differ_}" if differ_ else ""), flush=True)
        assert not differ_, ("the variant left the baseline's init / data / noise stream", at, rec_["kind"], differ_)

    if step > 0:
        # a resumed run restarts the loader and PAIR_GEN: from here it is no longer paired with its baseline
        PAIR_STATE["resumed"] = True
        print(f"pair: WARNING resumed at step {step:,}: the data order and the drawn numbers restart, so the run is not paired after this step", flush=True)
    else:
        pair_fingerprint(0, None, None)
    # trainable parameters (the frozen judge copy and the averaged weights are not counted)
    _raw_m = getattr(model, "_orig_mod", model)
    _n_train = sum(p_.numel() for p_ in _raw_m.parameters() if p_.requires_grad)
    _judge = TEACHER.get("st_frozen")
    _n_judge = sum(p_.numel() for p_ in _judge.parameters()) if _judge is not None else 0
    _h = cfg.get("hparams") or {}
    (OUT / "params.json").write_text(json.dumps({
        "trainable": _n_train, "all_model": sum(p_.numel() for p_ in _raw_m.parameters()), "judge_frozen_not_counted": _n_judge,
        "judge_requires_grad": sum(p_.numel() for p_ in _judge.parameters() if p_.requires_grad) if _judge is not None else 0,
        "blocks": int(_h.get("num_blocks", 8)), "width": int(_h.get("emb_dim", 256)), "heads": int(_h.get("n_heads", 8))}, indent=1))
    print(f"pair: {_n_train:,} trainable parameters (frozen judge copy {_n_judge:,}, not counted) -> params.json", flush=True)
    assert _judge is None or not any(p_.requires_grad for p_ in _judge.parameters()), "the judge copy has trainable parameters"

    if SELFTEACH and step == 0:
        # CHECK self-teaching gradient (eager, own generator, nothing of the run's streams or weights changes):
        # (1) the judge parameters get no gradient, (2) the output term sends gradient through the judge into the model,
        # (3) crops: an output equal to the clean image gives an output term of 0 (the same crop of both, the same pass)
        _st_c = TEACHER["st"]
        _gp = torch.Generator(device=device).manual_seed(7399)
        _S = int(g["IMAGE_SIZE"])
        _x = torch.rand(4, 3, _S, _S, generator=_gp, device=device)
        _tt = torch.full((4,), 0.5, device=device)
        _noisy = 0.5 * _x + 0.5 * torch.randn(_x.shape, generator=_gp, device=device)
        _pr = {"crop": torch.rand(4, 3, generator=_gp, device=device), "st_log": torch.zeros((), device=device)}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _out = _raw_m(_noisy, _tt, all_readouts=True)[-1]
            _, _st_o, _ = _st_c["terms"](_x, _noisy, _tt, _out, pair=_pr)
        _st_o.backward()
        _judge_grad = sum(1 for p_ in _judge.parameters() if p_.grad is not None)
        _blk_g = [float(sum(p_.grad.float().square().sum() for p_ in b_.parameters() if p_.grad is not None)) ** 0.5 for b_ in _raw_m.core_transformer.blocks]
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, _st_same, _ = _st_c["terms"](_x, _noisy, _tt, _x, pair=_pr)
        _st_o = _st_o.detach()
        print(f"CHECK self-teaching gradient: output term {float(_st_o):.5f} | judge tensors with a gradient {_judge_grad} (want 0) | "
              f"gradient norm per block of the model {' '.join(f'{v_:.1e}' for v_ in _blk_g)} (want > 0) | "
              f"output = clean image gives {float(_st_same):.2e} (want 0: same crop of both)", flush=True)
        assert _judge_grad == 0 and min(_blk_g) > 0 and float(_st_same) < 1e-6 and float(_st_o) > 0, "self-teaching gradient path is wrong"


def one_step(timed=False):
    global loader, step, acc, acc_all, acc_n, data_wait_s
    is_ot = False
    _t_data = time.perf_counter()
    try:
        images, labels_ = next(loader)
    except StopIteration:
        loader = iter(g["train_dataloader"])
        images, labels_ = next(loader)
    if COND:
        labels_ = labels_.long().clone()
        labels_[torch.rand(labels_.shape[0], generator=COND_GEN) < COND_DROP] = COND_N
        COND_CTX["y"] = labels_.to(device, non_blocking=True)
        cond_acc["none"] += int((labels_ == COND_N).sum()); cond_acc["n"] += labels_.shape[0]
    data_wait_s += time.perf_counter() - _t_data         # time the loop waits for the loader: 0 when the loader keeps up
    if MOE:
        MOE_CTX["images"] = images
        if step == MOE_AT:
            moe_convert(step)
            # the step is slower from here: the normal step time is measured again. Without this, the run with
            # 32 experts read its own slower step as another workload and paused for good (2026-10-13)
            yielder.base, yielder.samples, yielder.recent = cfg.get("gpu_step_ms"), [], []
            print("gpu-yield: normal step time is measured again after the MoE conversion", flush=True)
    if timed:
        gpu_sync(); t0 = time.perf_counter()
    if LOOP:
        sparse = bool(g["is_sparse_step"](step))                 # S S S S D: `step` counts the steps done so far
        metrics = train_step(images, optimizer, sparse)
        kind = "sparse" if sparse else "dense"
        loop_acc["exits"] = metrics["exit_losses"] if loop_acc["exits"] is None else loop_acc["exits"] + metrics["exit_losses"]
        loop_acc[kind] = metrics["loss_final"] if loop_acc[kind] is None else loop_acc[kind] + metrics["loss_final"]
        loop_acc["n_" + kind] += 1
    else:
        if LR_SWITCH:
            at = int(LR_SWITCH["step"])
            if step == at:
                lr_now.fill_(float(LR_SWITCH["lr"]))
            if step in (at - 1, at):
                lr_before = lr_probe_param.detach().clone()
        if step in GRAD_PROBE:
            grad_probe(images)
        if step in grad_snap:
            grad_probe_update()
        is_ot = bool(OT) and step % OT_EVERY == OT_EVERY - 1
        if is_ot:
            for k_, v_ in ot_step(images, optimizer).items():
                ot_acc[k_] = ot_acc.get(k_, 0.0) + v_
            ot_acc["n"] = ot_acc.get("n", 0) + 1
            metrics = {}
        elif PAIR:
            _pair = pair_draw(images)
            if step + 1 in (1, 2) or (step + 1) % PAIR_EVERY == 0:
                pair_fingerprint(step + 1, images, _pair)
            metrics = train_step(images, optimizer, _pair)
            if step + 1 == 1:
                pair_fingerprint(1, None, None, after=True)
        else:
            metrics = train_step(images, optimizer)
            if OT and step % (8 * OT_EVERY) == OT_EVERY - 2:
                # the gradient norm of a normal step (the gradients stay until the next step), beside that of the transport steps
                ot_acc["grad_norm_normal"] = ot_acc.get("grad_norm_normal", 0.0) + grad_norm()
                ot_acc["n_normal"] = ot_acc.get("n_normal", 0) + 1
        if "x_err_all" in metrics:
            _lv, _ea = metrics["noise_level"], metrics.pop("x_err_all")
            _ix = (_lv * NOISE_BINS).long().clamp_(0, NOISE_BINS - 1)
            bin_all["sum"].index_add_(0, _ix, _ea)
            bin_all["n"].index_add_(0, _ix, torch.ones_like(_ea))
        if "x_err" in metrics:
            # train loss by noise level: 10 equal bins of the noise level (bin 0 = nearly clean, bin 9 = nearly pure noise)
            level, err = metrics.pop("noise_level"), metrics.pop("x_err")
            idx = (level * NOISE_BINS).long().clamp_(0, NOISE_BINS - 1)
            bin_acc["sum"].index_add_(0, idx, err)
            bin_acc["n"].index_add_(0, idx, torch.ones_like(err))
            bin_acc["level"] += level.sum()
        if LR_SWITCH and step in (at - 1, at):
            # CHECK lr switch: the weight change of the first step at the new rate, against the last step at the old one
            lr_probe[step] = ((lr_probe_param.detach() - lr_before).float().norm().item(), float(optimizer.param_groups[0]["lr"]))
            if step == at and at - 1 in lr_probe:
                (d_old, lr_old), (d_new, lr_new) = lr_probe[at - 1], lr_probe[at]
                print(f"CHECK lr switch at step {at:,}: optimizer lr {lr_old:g} -> {lr_new:g} | weight change of one step {d_old:.3e} -> {d_new:.3e} "
                      f"| ratio of changes {d_new / d_old:.3f}, ratio of rates {lr_new / lr_old:.3f}", flush=True)
                assert abs(lr_new - float(LR_SWITCH["lr"])) < 1e-12 and 0.5 < (d_new / d_old) / (lr_new / lr_old) < 2.0, "the learning rate change did not take effect"
    ms = None
    if timed:
        gpu_sync(); ms = (time.perf_counter() - t0) * 1000
    step += 1
    if EMA_DECAY:
        ema_update()
        teacher_update()
    if is_ot:
        return ms          # a transport step: its numbers are in ot_acc, not in the train loss of the normal steps
    if PYR:
        # `clean` stays the final head's loss, the number the plain runs log; the sum and the readouts go beside it
        pyr_acc["total"] = metrics["loss"] if pyr_acc["total"] is None else pyr_acc["total"] + metrics["loss"]
        for key in ("teach", "noisy_share", "t_gap", "t_teacher", "t_student_noisy", "t_noisy_low", "noisy_neighbor", "out_align", "st_out", "st_noised",
                    "wreg_swd", "wreg_bwd", "wreg_c_swd", "wreg_t_swd", "wreg_z_std", "wreg_z_mean", "xm_loss_all", "xm_share1", "nx_dist", "nx_dist_fixed", "nx_moved", "xm_score_pick", "xm_scored", "d2_loss", "d2_gap", "d2_share", "d2_err_a", "d2_err_b", "d2_loss_b", "noise_dc_std", "corr_base_xmse", "corr_final_xmse",
                    "msa_flow_split", "msa_flow_sparse", "msa_final_split", "msa_final_sparse", "msa_d6s3", "msa_d6p3", "msa_s6p3",
                    "msa_t_dense", "msa_t_split", "msa_t_sparse", "msa_order_bad", "msa_pair_dist", "msa_split_share", "msa_noisy_share", "msa_mix_bad"):
            if key in metrics:
                pyr_acc[key] = metrics[key] if pyr_acc.get(key) is None else pyr_acc[key] + metrics[key]
        pyr_acc["readouts"] = metrics["readout_losses"] if pyr_acc["readouts"] is None else pyr_acc["readouts"] + metrics["readout_losses"]
        metrics = dict(metrics, loss=metrics["loss_final"])
    acc = metrics["loss"] if acc is None else acc + metrics["loss"]
    if "loss_all" in metrics:
        acc_all = metrics["loss_all"] if acc_all is None else acc_all + metrics["loss_all"]
    acc_n += 1
    if step == first_step:
        fb = int(metrics.get("forward_batch", images.shape[0]))
        print(f"step {step:,}: forward batch {fb} = {XM_K} x {images.shape[0]} images", flush=True)
        assert fb == XM_K * images.shape[0]
        if cfg.get("compile", True):
            # a failed compile falls back to uncompiled code without stopping the job: say which it is
            from torch._dynamo.utils import counters
            frames = dict(counters["frames"])
            # the 3090 smoke run compiled with {'total': 17, 'ok': 16}; a fallback gives ok = 0 for the step's frames
            compiled = frames.get("ok", 0) >= 0.8 * frames.get("total", 0) > 0
            print(f"torch.compile: frames {frames} -> train step is {'compiled' if compiled else 'NOT COMPILED (running slow uncompiled code)'}", flush=True)
    if LOOP and step == 5 and cfg.get("compile", True):
        # step 5 was the first dense step: the step function has now compiled for both kinds
        from torch._dynamo.utils import counters
        frames = dict(counters["frames"])
        ok = frames.get("ok", 0) >= 0.8 * frames.get("total", 0) > 0
        print(f"torch.compile after the first dense step: frames {frames} -> {'both step kinds compiled' if ok else 'NOT COMPILED'}", flush=True)
    return ms


t_other = 0.0        # seconds spent in evals, previews and checkpoints (not training time)
LOOP_ACC_EMPTY = {"exits": None, "sparse": None, "dense": None, "n_sparse": 0, "n_dense": 0}
loop_acc = dict(LOOP_ACC_EMPTY)     # looped model: sums of the per-loop losses since the last log
pyr_acc = {"total": None, "readouts": None}     # pyramid model: sums since the last log
guard_state = {}
NOISE_BINS = 10
bin_acc = {"sum": torch.zeros(NOISE_BINS, device=device), "n": torch.zeros(NOISE_BINS, device=device), "level": torch.zeros((), device=device)}
bin_all = {"sum": torch.zeros(NOISE_BINS, device=device), "n": torch.zeros(NOISE_BINS, device=device)}     # xm_ot: all candidates
if PYR:
    pyr_head_init = [p.detach().clone() for p in model.readout_heads.parameters()]


def periodic():
    """Everything that happens on a schedule, after each step. One metrics record per step."""
    global acc, acc_all, acc_n, t_window, t_other, data_wait_s
    COND_CTX["y"] = None                      # outside the train step every pass is without a class
    if step % LOG_EVERY:
        return
    now = time.time()
    rec = {"clean": (acc / acc_n).item(), "sps": acc_n / (now - t_window),
           "lr": float(optimizer.param_groups[0]["lr"]),       # read from the optimizer at every log

           "train_seconds": now - t_run - yielder.paused_s - t_other + seconds_before}
    rec["data_wait_ms"] = 1000 * data_wait_s / acc_n          # mean wait for the loader per step in this window
    if COND and step % EVAL_EVERY == 0:
        rec.update(cond_probe())
        rec["cond_gain"] = 1.0 - rec["cond_err_true"] / rec["cond_err_none"]
        if step in (EVAL_EVERY, 10 * EVAL_EVERY):
            print(f"CHECK cond at step {step:,}: mse at noise level 0.95 with the true class {rec['cond_err_true']:.5f} | no class {rec['cond_err_none']:.5f} | "
                  f"class of another image {rec['cond_err_wrong']:.5f} | norm of a class vector {rec['cond_emb_norm']:.4f} | "
                  f"{100 * cond_acc['none'] / max(cond_acc['n'], 1):.1f}% of the train images had no class", flush=True)
            assert rec["cond_err_true"] != rec["cond_err_none"] and rec["cond_err_true"] != rec["cond_err_wrong"], "the class does not reach the model"
            assert abs(cond_acc["none"] / max(cond_acc["n"], 1) - COND_DROP) < 0.02, "label dropout is not as configured"
            if step == 10 * EVAL_EVERY:
                assert rec["cond_err_true"] < rec["cond_err_wrong"], "the model does not use the class"
    if MOE and step % EVAL_EVERY == 0:
        rec.update(moe_stats(MOE_CTX["images"]))
        if step == MOE_AT + EVAL_EVERY:
            print(f"CHECK moe at step {step:,}: " + " | ".join(f"{k_} {v_:.4f}" for k_, v_ in rec.items() if k_.startswith("moe_")), flush=True)
            assert rec["moe_spread"] > 0, "the experts are still identical: the routing is not on"
    data_wait_s = 0.0
    if ot_acc.get("n"):
        # transport steps of this window: paired loss against the loss of the fixed pairing (output i with image i)
        for k_ in ("ot_loss", "ot_loss_fixed", "ot_loss_final", "ot_moved", "ot_out_std", "ot_grad_norm"):
            rec[k_] = float(ot_acc[k_]) / ot_acc["n"]
        rec["ot_gain"] = 1.0 - rec["ot_loss"] / rec["ot_loss_fixed"]
        if ot_acc.get("n_normal"):
            rec["grad_norm_normal"] = float(ot_acc["grad_norm_normal"]) / ot_acc["n_normal"]
            rec["ot_grad_ratio"] = rec["ot_grad_norm"] / rec["grad_norm_normal"]
        if step == first_step - 1 + LOG_EVERY:
            print(f"CHECK transport at step {step}: {ot_acc['n']} transport steps of {LOG_EVERY} (1 in {OT_EVERY}) | paired loss {rec['ot_loss']:.5f}, "
                  f"fixed pairing {rec['ot_loss_fixed']:.5f}, gain {100 * rec['ot_gain']:.2f}% | {100 * rec['ot_moved']:.0f}% of the outputs changed partner | "
                  f"weight {OT_W} | gradient norm {rec['ot_grad_norm']:.3f}, normal step {rec.get('grad_norm_normal', float('nan')):.3f} | "
                  f"std of the outputs over the batch {rec['ot_out_std']:.4f}", flush=True)
            assert ot_acc["n"] == LOG_EVERY // OT_EVERY and rec["ot_loss"] <= rec["ot_loss_fixed"] and rec["ot_moved"] > 0.5
            assert abs(rec["ot_loss"] - rec["ot_loss_final"]) < 0.02 * rec["ot_loss"], (rec["ot_loss"], rec["ot_loss_final"])
        ot_acc.clear()
    if acc_all is not None:
        rec["loss_all"] = (acc_all / acc_n).item()
        rec["xm_gap"] = rec["loss_all"] - rec["clean"]        # how much lower the chosen candidates are
    if PYR:
        readouts = (pyr_acc["readouts"] / acc_n).tolist()
        rec.update({n: v for n, v in zip(PYR_NAMES, readouts)})
        rec["total"] = (pyr_acc["total"] / acc_n).item()
        if pyr_acc.get("xm_loss_all") is not None:
            # with the readout score, both sides are the R32 loss of the candidates (not on the scale of the final loss)
            _pick = (pyr_acc["xm_score_pick"] / acc_n).item() if pyr_acc.get("xm_score_pick") is not None else rec["clean"]
            rec["xm_gain"] = 1.0 - _pick / (pyr_acc["xm_loss_all"] / acc_n).item()
        for key in ("teach", "noisy_share", "t_gap", "t_teacher", "t_student_noisy", "t_noisy_low", "noisy_neighbor", "out_align", "st_out", "st_noised",
                    "wreg_swd", "wreg_bwd", "wreg_c_swd", "wreg_t_swd", "wreg_z_std", "wreg_z_mean", "xm_loss_all", "xm_share1", "nx_dist", "nx_dist_fixed", "nx_moved", "xm_score_pick", "xm_scored", "d2_loss", "d2_gap", "d2_share", "d2_err_a", "d2_err_b", "d2_loss_b", "noise_dc_std", "corr_base_xmse", "corr_final_xmse",
                    "msa_flow_split", "msa_flow_sparse", "msa_final_split", "msa_final_sparse", "msa_d6s3", "msa_d6p3", "msa_s6p3",
                    "msa_t_dense", "msa_t_split", "msa_t_sparse", "msa_order_bad", "msa_pair_dist", "msa_split_share", "msa_noisy_share", "msa_mix_bad"):
            if pyr_acc.get(key) is not None:
                rec[key] = (pyr_acc[key] / acc_n).item()
        if step == first_step - 1 + LOG_EVERY:
            # CHECK: the readouts are in the loss (total = final + weight * readouts) and their heads train
            want = rec["clean"] + g["pyramid_weight"] * sum(readouts) + (float(TEACH.get("weight", 1.0)) * rec["teach"] if TEACH else 0.0)
            if cfg.get("distill2"):                               # the second target is in the total. With "second": "train" the losses of the second pass are too (not logged apart)
                want = rec["total"] if cfg["distill2"].get("second") == "train" else want + (0.0 if cfg["distill2"].get("link") == "none" else float(cfg["distill2"].get("weight", 1.0))) * rec["d2_loss"]
            if SELFFLOW:
                want += float(SELFFLOW.get("out_weight", 0.0)) * rec["out_align"]
            if MSA:
                # the flow losses of the two students are in the total too. CHECK of the step's own numbers: group sizes,
                # pair distance, and the noise order (staggered: dense cleaner than split cleaner than sparse; shared: equal t)
                want += rec["msa_flow_split"] + rec["msa_flow_sparse"]
                _stag = MSA.get("timesteps", "shared") == "staggered"
                print(f"CHECK msa at step {step}: split share {rec['msa_split_share']:.4f} (want {42 / 64:.4f}) | pair distance {rec['msa_pair_dist']:.3f} | "
                      f"mean t dense / split / sparse {rec['msa_t_dense']:.3f} / {rec['msa_t_split']:.3f} / {rec['msa_t_sparse']:.3f} | order violations {rec['msa_order_bad']:.4f} | "
                      f"flow split {rec['msa_flow_split']:.5f} sparse {rec['msa_flow_sparse']:.5f} | align D6>S3 {rec['msa_d6s3']:.4f} D6>P3 {rec['msa_d6p3']:.4f} S6>P3 {rec['msa_s6p3']:.4f}", flush=True)
                assert abs(rec["msa_split_share"] - 42 / 64) < 1e-6 and rec["msa_order_bad"] == 0.0
                assert abs(rec["msa_pair_dist"] - (1.0 if MSA.get("pairing", "spatial") == "spatial" else 32 ** 0.5)) < 1e-3
                if MSA.get("timesteps") == "mixed":
                    # one mask of 16 of 64 tokens: split and sparse noisier there, the teacher's t elsewhere
                    print(f"CHECK msa mixed: tokens at the higher levels {rec['msa_noisy_share']:.4f} (want 0.2500) | unmasked tokens off the teacher's t {rec['msa_mix_bad']:.4f} (want 0)", flush=True)
                    assert abs(rec["msa_noisy_share"] - 0.25) < 1e-6 and rec["msa_mix_bad"] == 0.0, "mixed t not applied"
                    assert rec["msa_t_dense"] - rec["msa_t_split"] > 0.01 and rec["msa_t_split"] - rec["msa_t_sparse"] > 0.01, "mixed t not applied"
                elif _stag:
                    assert rec["msa_t_dense"] - rec["msa_t_split"] > 0.1 and rec["msa_t_split"] - rec["msa_t_sparse"] > 0.05, "staggered t not applied"
                else:
                    assert abs(rec["msa_t_dense"] - rec["msa_t_split"]) < 1e-6 and abs(rec["msa_t_dense"] - rec["msa_t_sparse"]) < 1e-6, "shared t not applied"
            _ST = cfg.get("selfteach") or {}
            if _ST:
                want += float(_ST.get("weight_out", 1.0)) * rec["st_out"] + float(_ST.get("weight_noised", 1.0)) * rec["st_noised"]
                _fz = TEACHER["st_frozen"]
                _gap = max(float((a.float() - b.detach().float()).abs().max()) for a, b in zip(_fz.parameters(), getattr(model, "_orig_mod", model).parameters()))
                # the frozen copy holds the weights of the start of the last step: one optimizer step behind, not older and not equal
                _ema_judge = _ST.get("judge", "live") == "ema"
                print(f"CHECK self-teaching at step {step}: z after block {_ST.get('layers') or int(_ST.get('layer', 6))} | overhang {float(_ST.get('overhang', 0.5))} | "
                      f"judge {'averaged' if _ema_judge else 'last-step'} weights at t {float(_ST.get('judge_t', 1.0))} | output term {rec['st_out']:.5f} x {float(_ST.get('weight_out', 1.0))} | "
                      f"noised term {rec['st_noised']:.5f} x {float(_ST.get('weight_noised', 1.0))} ({_ST.get('noised_net', 'live')} weights) | "
                      f"frozen copy: largest weight gap to the trained weights {_gap:.2e}", flush=True)
                assert (rec["st_out"] > 0) == (float(_ST.get("weight_out", 1.0)) > 0) and (rec["st_noised"] > 0) == (float(_ST.get("weight_noised", 1.0)) > 0), (rec["st_out"], rec["st_noised"])
                # a judge with averaged weights lags the trained weights by many steps: a larger gap than one optimizer step
                assert (_gap > 1e-4 if _ema_judge and step >= 500 else True) and 0 < _gap < (1.0 if _ema_judge else 1e-2) and float(_ST.get("weight_out", 1.0)) + float(_ST.get("weight_noised", 1.0)) > 0, _gap
                # CHECK (2026-10-15): the step's own sync, run here once more, makes the frozen copy equal to its source in every
                # parameter and buffer (by name): the judge is the whole model of this step, not a part of it or another model
                _st = TEACHER["st"]
                _st["sync"]()
                _src = TEACHER["net"] if _ema_judge else getattr(model, "_orig_mod", model)
                _fsd, _ssd = _fz.state_dict(), _src.state_dict()
                _sync_gap = max((float((_fsd[k_].float() - v_.float()).abs().max()) for k_, v_ in _ssd.items() if v_.numel()), default=0.0)
                _n_src = len(list(_src.parameters())) + len(list(_src.buffers()))
                print(f"CHECK self-teaching sync at step {step}: {len(_st['pairs'])} of {_n_src} weight and buffer tensors paired | "
                      f"largest gap after the sync {_sync_gap:.2e} | step {'plain' if not TEACH else 'msa' if MSA else 'self-flow'} | "
                      f"output term {rec['st_out']:.5f} > 0: {rec['st_out'] > 0}", flush=True)
                assert _fsd.keys() == _ssd.keys() and len(_st["pairs"]) == _n_src and _sync_gap == 0.0, ("frozen copy differs after the sync", _sync_gap)
                assert rec["st_out"] > 0 or float(_ST.get("weight_out", 1.0)) == 0, ("no output term", rec["st_out"])
            if TEACH:
                # CHECK: the teaching term is in the loss; for self-flow, exactly the planned share of patches is at the higher noise level
                share = f" | patches at the higher noise level {100 * rec['noisy_share']:.1f}% | mean t gap {rec['t_gap']:.3f}" if SELFFLOW else ""
                print(f"CHECK teaching at step {step}: teach loss {rec['teach']:.5f} x weight {TEACH.get('weight', 1.0)}{share}", flush=True)
                # with a teacher the teaching term is in the loss; mixed noise only (selfflow teacher: false) has none
                assert (rec["teach"] > 0) == bool(not SELFFLOW or SELFFLOW.get("teacher", True)), rec["teach"]
                if SELFFLOW:
                    _T = (g["IMAGE_SIZE"] // g["patch_size"]) ** 2
                    _blk = SELFFLOW.get("noisy_layout") == "block"
                    assert abs(rec["noisy_share"] - round(float(SELFFLOW.get("noisy_share", 0.25)) * _T) / _T) < (0.1 if _blk else 1e-6) and rec["t_gap"] > 0.05
                    # CHECK of the options: each one is read from a value that the step itself computed
                    opts = (f"teacher input mean t {rec['t_teacher']:.3f} | noisier patches mean t {rec['t_student_noisy']:.3f} | "
                            f"context loss weight {SELFFLOW.get('ctx_weight', 1.0)} | pairs {PAIRS}")
                    opts += (f" | layout {SELFFLOW.get('noisy_layout', 'random')}: noisier right neighbor {rec['noisy_neighbor']:.3f}"
                             f" | teaching on {SELFFLOW.get('teach_on', 'all')} patches")
                    # a rectangle of side n has (n - 1) / n of its patches with a noisier right neighbor. Spread patches: about the share
                    assert (rec["noisy_neighbor"] > rec["noisy_share"] + 0.15) == _blk, (rec["noisy_neighbor"], rec["noisy_share"])
                    opts += f" | output alignment {rec['out_align']:.5f} x weight {float(SELFFLOW.get('out_weight', 0.0))}"
                    assert (rec["out_align"] > 0) == (float(SELFFLOW.get("out_weight", 0.0)) > 0 and SELFFLOW.get("teacher", True) is not False), rec["out_align"]
                    assert (rec["t_teacher"] > 0.999) == (SELFFLOW.get("teacher_input") == "clean"), rec["t_teacher"]
                    assert (rec["t_student_noisy"] < 1e-6) == (SELFFLOW.get("noisy_t") == "mask"), rec["t_student_noisy"]
                    if TEACHER["net"] is not None:
                        _d = teacher_distance()
                        opts += f" | averaged teacher (decay {TEACHER['decay']}): distance to the trained weights {_d:.4f}"
                        assert _d > 0, "the averaged teacher equals the trained weights"
                    print(f"CHECK self-flow options at step {step}: {opts}", flush=True)
                    if not SELFFLOW.get("teacher_input") and not SELFFLOW.get("noisy_t"):
                        # CHECK t_dist in the self-flow step: the logged means of the two drawn t (higher = teacher,
                        # lower = noisier patches) and the share of noisier patches above noise 0.9 match the setting
                        _two = draw_t(1_000_000, 2, generator=torch.Generator().manual_seed(0))
                        _b15 = torch.sigmoid(torch.randn(1_000_000, 2, generator=torch.Generator().manual_seed(0)) * 1.5)
                        _st = lambda x: (x.max(1).values.mean().item(), x.min(1).values.mean().item(), (x.min(1).values < 0.1).float().mean().item())
                        _want, _base = _st(_two), _st(_b15)
                        _got = (rec["t_teacher"], rec["t_student_noisy"], rec["t_noisy_low"])
                        print(f"CHECK t distribution in the self-flow step ({'uniform' if T_UNIFORM else f'sigmoid({T_STD:g} * randn + {T_MEAN:g})'}): "
                              f"mean higher t / mean lower t / share of lower t below 0.1 = {_got[0]:.3f} / {_got[1]:.3f} / {_got[2]:.3f} | "
                              f"the setting gives {_want[0]:.3f} / {_want[1]:.3f} / {_want[2]:.3f} | broad15 gives {_base[0]:.3f} / {_base[1]:.3f} / {_base[2]:.3f}", flush=True)
                        assert all(abs(a - b) < tol for a, b, tol in zip(_got, _want, (0.015, 0.015, 0.025))), "the t distribution of the self-flow step is not the setting"
                        assert not T_DIST or max(abs(a - b) for a, b in zip(_want, _base)) > 0.04, "the t_dist setting is too close to broad15 for the check"
            if cfg.get("wreg"):
                _wr = [rec[k_] for k_ in ("wreg_swd", "wreg_bwd", "wreg_c_swd", "wreg_t_swd")]
                want += float(cfg["wreg"].get("weight", 0.3)) * sum(_wr)
                print(f"CHECK wasserstein at step {step}: block {int(cfg['wreg'].get('block', 6))} | swd {_wr[0]:.4f} bwd {_wr[1]:.4f} c_swd {_wr[2]:.4f} "
                      f"t_swd {_wr[3]:.4f}, each x weight {float(cfg['wreg'].get('weight', 0.3))} | z mean {rec['wreg_z_mean']:+.3f} std {rec['wreg_z_std']:.3f}", flush=True)
                assert all(v > 0 for v in _wr), _wr
            if cfg.get("distill2"):
                print(f"CHECK 2-step distillation at step {step}: mse 1-step to 2-step output {rec['d2_gap']:.5f} | weighted term {rec['d2_loss']:.5f} x {float(cfg['distill2'].get('weight', 1.0)):g} "
                      f"(final loss {rec['clean']:.5f}) | mse to the clean image: 1-step {rec['d2_err_a']:.5f}, 2-step {rec['d2_err_b']:.5f} | {100 * rec['d2_share']:.0f}% of the images get the second target", flush=True)
                print(f"CHECK 2-step distillation settings: second pass {cfg['distill2'].get('second', 'target')}, link {cfg['distill2'].get('link', 'target')}, "
                      f"loss weight of the second pass {cfg['distill2'].get('second_weight', 'mid')} | weighted loss: first pass {rec['clean']:.5f}, second pass {rec.get('d2_loss_b') or 0:.5f}", flush=True)
                assert rec["d2_gap"] > 0 and rec["d2_loss"] > 0 and abs(rec["d2_err_b"] - rec["d2_err_a"]) > 0
            if cfg.get("noise_ot"):
                print(f"CHECK noise transport at step {step}: mean squared distance image to paired noise {rec['nx_dist']:.5f}, to the unpaired noise "
                      f"{rec['nx_dist_fixed']:.5f} (gain {100 * (1 - rec['nx_dist'] / rec['nx_dist_fixed']):.3f}%) | {100 * rec['nx_moved']:.0f}% of the noise samples changed image", flush=True)
                assert rec["nx_dist"] < rec["nx_dist_fixed"] and rec["nx_moved"] > 0.5
            if cfg.get("xm_ot") and cfg["xm_ot"].get("score") == "readout":
                print(f"CHECK xm transport by readout at step {step}: weighted R32 loss of the chosen rows {rec['xm_score_pick']:.5f}, of all candidates {rec['xm_loss_all']:.5f} "
                      f"(gain {100 * (1 - rec['xm_score_pick'] / rec['xm_loss_all']):.2f}%) | share of images on seed 2: {rec['xm_share1']:.3f}", flush=True)
                assert rec["xm_score_pick"] < rec["xm_loss_all"]
            if cfg.get("xm_ot") and cfg["xm_ot"].get("min_noise"):
                print(f"CHECK xm transport noise limit at step {step}: {100 * rec['xm_scored']:.1f}% of the images count in the assignment (noise level above {cfg['xm_ot']['min_noise']})", flush=True)
                assert 0.2 < rec["xm_scored"] < 0.8, rec["xm_scored"]
            if cfg.get("xm_ot") and cfg["xm_ot"].get("score") == "readout":
                pass
            elif cfg.get("xm_ot"):
                print(f"CHECK xm transport at step {step}: weighted final loss of the chosen rows {rec['clean']:.5f}, of all candidates {rec['xm_loss_all']:.5f} "
                      f"(gain {100 * (1 - rec['clean'] / rec['xm_loss_all']):.2f}%) | share of images on seed 2: {rec['xm_share1']:.3f}", flush=True)
                _k, _b = int(cfg["xm_ot"]["seeds"]), int(g["CACHE_BATCH_SIZE"])
                if cfg["xm_ot"].get("assign") == "random":
                    assert abs(1 - rec["clean"] / rec["xm_loss_all"]) < 0.015, "random assignment must save no loss"
                if cfg["xm_ot"].get("assign") == "best":
                    assert rec["clean"] < rec["xm_loss_all"] and 0.5 / _k < rec["xm_share1"] < 1.5 / _k, rec["xm_share1"]
                else:
                    assert (rec["clean"] < rec["xm_loss_all"] or cfg["xm_ot"].get("assign") == "random") and abs(rec["xm_share1"] - sum(j % _k == 1 for j in range(_b)) / _b) < 1e-6, rec["xm_share1"]
            moved = sum(not torch.equal(a, p.detach()) for a, p in zip(pyr_head_init, model.readout_heads.parameters()))
            print(f"CHECK pyramid at step {step}: final {rec['clean']:.5f} | " + " ".join(f"{n} {v:.5f}" for n, v in zip(PYR_NAMES, readouts))
                  + f" | total {rec['total']:.5f}, rule gives {want:.5f} | {moved} of {len(pyr_head_init)} head weight tensors moved", flush=True)
            assert abs(want - rec["total"]) < 1e-3 * want and moved == len(pyr_head_init)
        # every entry is a sum since the last log (2026-10-15: a list of names here missed the msa_* keys and t_noisy_low,
        # so those logged a running sum since the start / LOG_EVERY)
        pyr_acc.update(dict.fromkeys(pyr_acc))
    if bin_acc["n"].sum() > 0:
        # train loss by noise level (plain x-MSE of the final head, mean of this window) and the mean noise level drawn
        counts = bin_acc["n"].tolist()
        rec.update({f"train_x_mse_n{b}": v for b, (v, c) in enumerate(zip((bin_acc["sum"] / bin_acc["n"].clamp_min(1)).tolist(), counts)) if c})
        rec["noise_mean"] = bin_acc["level"].item() / sum(counts)
        if bin_all["n"].sum() > 0:
            # xm_ot: by noise level, how much lower the x-MSE of the chosen rows is than that of all candidates
            _all = (bin_all["sum"] / bin_all["n"].clamp_min(1)).tolist()
            rec.update({f"xm_gain_n{b}": 1.0 - rec[f"train_x_mse_n{b}"] / _all[b] for b, c in enumerate(bin_all["n"].tolist()) if c and f"train_x_mse_n{b}" in rec})
            bin_all["sum"].zero_(); bin_all["n"].zero_()
        if step == LOG_EVERY:
            # CHECK: the noise levels that built the inputs have the mean that the shift gives (0.5 without a shift)
            _level = 1.0 - draw_t(1_000_000, generator=torch.Generator().manual_seed(0))
            if T_PURE:
                _g = torch.Generator().manual_seed(1)
                _level = torch.where(torch.rand(1_000_000, generator=_g) < T_PURE, T_PURE_MIN + (1.0 - T_PURE_MIN) * torch.rand(1_000_000, generator=_g), _level)
                print(f"CHECK pure-noise share: {100 * T_PURE:g}% of the inputs get a noise level in {T_PURE_MIN:g}..1 | share of the train inputs in the top noise bin "
                      f"{100 * counts[-1] / sum(counts):.1f}% (without it {100 * float(((1.0 - torch.sigmoid(torch.randn(1_000_000, generator=torch.Generator().manual_seed(0)) * T_STD + T_MEAN)) >= 0.9).float().mean()):.2f}%)", flush=True)
                assert counts[-1] / sum(counts) > 0.5 * T_PURE, "the pure-noise share is not applied"
            _want = (T_SHIFT * _level / (1.0 + (T_SHIFT - 1.0) * _level)).mean().item()
            if T_DIST:
                print(f"CHECK t distribution: t = sigmoid({T_STD:g} * randn + {T_MEAN:g}) | mean noise level of the train inputs {rec['noise_mean']:.4f}, "
                      f"the setting gives {_want:.4f}, broad15 gives 0.5000", flush=True)
                assert abs(rec["noise_mean"] - 0.5) > 0.03 or abs(T_MEAN) < 0.1, "the t distribution is still broad15"
            if NOISE_MEAN_STD:
                _px = 3 * int(g["IMAGE_SIZE"]) ** 2
                _dc = (1.0 / _px + NOISE_MEAN_STD ** 2) ** 0.5
                print(f"CHECK noise mean: std of the mean of a noise sample {rec['noise_dc_std']:.4f} | the setting gives {_dc:.4f}, plain noise gives {(1.0 / _px) ** 0.5:.4f}", flush=True)
                assert 0.6 * _dc < rec["noise_dc_std"] < 1.5 * _dc and rec["noise_dc_std"] > 1.5 * (1.0 / _px) ** 0.5, "the noise mean spread is not applied"
            print(f"CHECK timestep shift at step {step}: alpha {T_SHIFT:.4f} | mean noise level of the {int(sum(counts)):,} train inputs "
                  f"{rec['noise_mean']:.4f}, the shift gives {_want:.4f}, no shift gives {_level.mean().item():.4f} | "
                  f"share of inputs per noise bin 0..9: {' '.join(f'{100 * c / sum(counts):.0f}%' for c in counts)}", flush=True)
            assert abs(rec["noise_mean"] - _want) < 0.02 and (T_SHIFT == 1.0 or abs(rec["noise_mean"] - _level.mean().item()) > 0.03), "the timestep shift is not applied as configured"
        for v in bin_acc.values():
            v.zero_()
    if LOOP:
        # per-loop supervision: L1..Ln = train loss of the exit after each loop, mean of this window
        exits = (loop_acc["exits"] / acc_n).tolist()
        rec.update({f"L{k + 1}": v for k, v in enumerate(exits)})
        if len(exits) > 1:
            rec[f"L_avg{''.join(str(k + 1) for k in range(len(exits) - 1))}"] = statistics.fmean(exits[:-1])
        for kind in ("sparse", "dense"):
            if loop_acc["n_" + kind]:
                rec[f"L{len(exits)}_{kind}"] = (loop_acc[kind] / loop_acc["n_" + kind]).item()
        if step == LOG_EVERY:
            # CHECK: both step kinds ran, 4 sparse to 1 dense, and the total follows the rule
            ns, nd = loop_acc["n_sparse"], loop_acc["n_dense"]
            want = 0.5 * statistics.fmean(exits[:-1]) + 0.5 * exits[-1] if len(exits) > 1 else exits[0]
            print(f"CHECK loop at step {step}: {ns} sparse and {nd} dense steps | L1..L{len(exits)} "
                  f"{' '.join(f'{v:.5f}' for v in exits)} | total {rec['clean']:.5f}, rule gives {want:.5f}", flush=True)
            assert ns == g["sparse_per_dense"] * nd and nd > 0 and abs(want - rec["clean"]) < 1e-3 * want
        loop_acc.update(LOOP_ACC_EMPTY)
    if step % (LOG_EVERY * 5) == 0:
        extra = f" | all candidates {rec['loss_all']:.5f}" if "loss_all" in rec else ""
        print(f"step {step:,} | {1000 / rec['sps']:.0f} ms/step (loader wait {rec['data_wait_ms']:.0f}) | {rec['sps'] * g['CACHE_BATCH_SIZE']:.0f} img/s | loss {rec['clean']:.5f}{extra} | "
              f"{int(rec['train_seconds']) // 60} min", flush=True)
    t0 = time.time()
    if step % EVAL_EVERY == 0 or step == STEPS:
        rec.update(evaluate(step))
    if GEN and (step % int(GEN["every"]) == 0 or step == STEPS):
        try:
            rec.update(gen_scores(step))
        except Exception as e:                       # a score must never stop the training
            print("gen eval failed (training continues):", repr(e), flush=True)
    if CPROBE and step >= int(CPROBE.get("from", 0)) and (step % int(CPROBE.get("every", 50)) == 0 or step == STEPS):
        contrast_snapshot(step)
        if step == STEPS:
            contrast_report()
    if step % PREVIEW_EVERY == 0 or step == STEPS:
        rec.update(preview(step) or {})
        if CORR:
            try:
                corr_picture(step)
                if step == STEPS:
                    corr_k_grid(step)
            except Exception as e:                   # a picture must never stop the training
                print("corrector picture failed (training continues):", repr(e), flush=True)
        if Z_SNAP:
            try:
                rec.update(z_snapshot(step))
            except Exception as e:                   # a picture must never stop the training
                print("z snapshot failed (training continues):", repr(e), flush=True)
        if JIT_DIFF:
            jitter_snapshot(rec["prev_sharp"])
            if step == STEPS:
                jitter_diff_report()
        if EMA_DECAY and ema_params is not None:
            rec.update(preview(step, ema=True))
            if EMA2_DECAY and ema2_params is not None:
                try:
                    rec.update(preview(step, ema="slow"))
                except Exception as e:               # a picture must never stop the training
                    print("slow-average preview failed (training continues):", repr(e), flush=True)
            if JIT and step % int(JIT.get("every", PREVIEW_EVERY)) == 0:
                jitter_probe(step, rec["prev_sharp"], rec["ema_prev_sharp"])
    log_metrics(step, **rec)
    # GUARD: end a run that has diverged or does not learn, so that a queue goes on to the next run.
    # (2026-10-08: a diverged run held the Spark for two hours.) cfg["guard"] = {"step": N, "max_test": X}:
    # from step N on, the test loss must be at most X.
    bad = None
    if rec["clean"] != rec["clean"] or abs(rec["clean"]) == float("inf"):
        bad = f"the train loss is {rec['clean']}"
    if "teach" in rec:
        guard_state.setdefault("teach0", rec["teach"])
        if rec["teach"] != rec["teach"] or rec["teach"] > max(4.0, 100 * guard_state["teach0"]):
            bad = f"the teaching loss ran away: {rec['teach']:.4g} (first value {guard_state['teach0']:.4g})"
    guard = cfg.get("guard") or {}
    if guard and "test_clean" in rec and step >= int(guard["step"]) and rec["test_clean"] > float(guard["max_test"]):
        bad = f"test loss {rec['test_clean']:.5f} at step {step:,} is above the limit {guard['max_test']} (set for step {guard['step']:,} on)"
    if bad:
        print(f"GUARD: stopping the run at step {step:,}: {bad}", flush=True)
        print(f"DONE {step:,} steps (stopped by the guard)", flush=True)
        sys.stdout.flush()
        os._exit(3)
    if step == STEPS:
        if cfg.get("save_final", True):
            checkpoint(step, final=True)
    elif step % CKPT_EVERY == 0:
        checkpoint(step)
    t_other += time.time() - t0
    acc, acc_all, acc_n, t_window = None, None, 0, time.time()


def probe():
    """A few timed training steps, with their scheduled work, to see if the GPU is free again."""
    out = []
    for _ in range(20):
        if step >= STEPS:
            break
        out.append(one_step(timed=True))
        periodic()
    return out or [0.0]


assert EVAL_EVERY % LOG_EVERY == 0 and PREVIEW_EVERY % LOG_EVERY == 0 and CKPT_EVERY % LOG_EVERY == 0 and STEPS % LOG_EVERY == 0
while step < STEPS:
    ms = one_step(timed=yielder.timed(step + 1))
    if ms is not None:
        yielder.add(step, ms)
    periodic()
    if yielder.busy() and step < STEPS:
        yielder.wait(step, probe)
        t_window = time.time()

total = time.time() - t_run
print(f"DONE {step:,} steps | training {(total - yielder.paused_s - t_other) / 60:.1f} min | evals and saves {t_other / 60:.1f} min | "
      f"paused {yielder.paused_s / 60:.1f} min in {yielder.pauses} pauses", flush=True)
