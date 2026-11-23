"""Generative scores of an image model against real validation images: FID, KID and fd_ae.

  fid          Frechet Inception Distance: Frechet distance between Gaussians fitted to the
               2,048 pool features of the FID Inception network (weights and network code of
               pytorch-fid, the port of the original TF implementation). Images go in as
               [0, 1] floats and are resized to 299 px inside the network, as pytorch-fid does.
  kid          Kernel Inception Distance on the same features: unbiased MMD^2 with the kernel
               (x.y / d + 1)^3, mean over 100 random subsets of 1,000 images. Reported x 1000.
  fd_ae_z      the Frechet distance again, in the latent space of the user's 128 px autoencoder
               (ae128_180k): the flattened z, 2,304 numbers per image.
  fd_ae_pool   the same on a summary of z: mean and std over its 48 tokens, 96 numbers.

The scores depend on the number of images and on the sampler, so both are part of the result:
compare only numbers made with the same n, sampler and step count. With n = 10,000 the FID is
higher than the usual 50,000-image FID of the same model.

Needs (mounted): /gen_eval/pt_inception-2015-12-05-6726825d.pth, /gen_eval/pytorch_fid_inception.py,
/scale/ae128_180k/ckpt_step_180000.pt, /scale/u64at100k_128/generated_u64at100k_128.py.
Features of the real images are computed once per (n, image size, tag) and kept in /gen_eval.
"""
import importlib.util
import os
import re
import time
from pathlib import Path

import torch

GEN_DIR = Path(os.environ.get("GEN_EVAL_DIR", "/gen_eval"))
SCALE_DIR = Path(os.environ.get("GEN_EVAL_SCALE_DIR", "/scale"))
_nets = {}


def _inception(device):
    if "inception" not in _nets:
        spec = importlib.util.spec_from_file_location("pytorch_fid_inception", GEN_DIR / "pytorch_fid_inception.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        weights = GEN_DIR / "pt_inception-2015-12-05-6726825d.pth"
        mod.load_state_dict_from_url = lambda *a, **k: torch.load(weights, map_location="cpu")      # no network in the job
        net = mod.InceptionV3([mod.InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=True, normalize_input=True, use_fid_inception=True)
        _nets["inception"] = net.to(device).eval().requires_grad_(False)
    return _nets["inception"]


def _ae_encoder(device):
    if "ae" not in _nets:
        ae = {"__name__": "ae_nb"}
        cells = re.split(r"^# %% cell \d+\n", (SCALE_DIR / "u64at100k_128" / "generated_u64at100k_128.py").read_text(), flags=re.M)
        os.environ.setdefault("ML_CHECKPOINTS", "/tmp/ckpt")
        os.environ.setdefault("ML_DATASETS", "/datasets")
        exec(cells[1], ae)
        first = next(i for i, c in enumerate(cells) if "def apply_2d_rope" in c)
        last = next(i for i, c in enumerate(cells) if "class Encoder" in c)
        for c in cells[first:last + 1]:
            exec(c, ae)
        enc = ae["Encoder"](16, 3, ae["ae_emb_dim"], ae["emb_dim"], ae["n_ae_latent_tokens"], ae["n_latent_tokens"], ae["num_blocks"],
                            ae["n_heads"], ae["head_size"])
        state = torch.load(SCALE_DIR / "ae128_180k" / "ckpt_step_180000.pt", map_location="cpu", weights_only=False)["encoder"]
        res = enc.load_state_dict(state, strict=False)
        assert not res.unexpected_keys and not [k for k in res.missing_keys if k in dict(enc.named_parameters())], res
        _nets["ae"] = enc.to(device).eval().requires_grad_(False)
    return _nets["ae"]


@torch.no_grad()
def features(images):
    """images: (B, 3, H, W) in [0, 1] on the GPU. Returns the three feature sets on the CPU."""
    images = images.float().clamp(0, 1)
    inc = _inception(images.device)(images)[0].flatten(1)
    if images.shape[-1] != 128:      # Colab 256 px stage (2026-11-07): the AE is a 128 px model, fd_ae is scored at 128 px
        images = torch.nn.functional.interpolate(images, size=(128, 128), mode="bilinear", antialias=True)
    if not (SCALE_DIR / "ae128_180k" / "ckpt_step_180000.pt").exists():
        # Colab (2026-11-07): the 96 MB AE cannot be uploaded (one HTTP request per file). Stand-in so the keys stay:
        # "ae_z" = 8 x 8 average-pooled pixels (192 values), "ae_pool" = channel mean and std. Not the AE latent space.
        z = torch.nn.functional.adaptive_avg_pool2d(images, 8).flatten(2).transpose(1, 2)        # (B, 64, 3)
        return {"inception": inc.cpu(), "ae_z": z.flatten(1).cpu(), "ae_pool": torch.cat((z.mean(1), z.std(1)), dim=1).cpu()}
    z = _ae_encoder(images.device)(images).float()
    return {"inception": inc.cpu(), "ae_z": z.flatten(1).cpu(), "ae_pool": torch.cat((z.mean(1), z.std(1)), dim=1).cpu()}


def collect(batches, n):
    """Features of the first n images of an iterator of image batches."""
    out, count = {}, 0
    for images in batches:
        f = features(images[:n - count])
        for k, v in f.items():
            out.setdefault(k, []).append(v)
        count += len(images[:n - count])
        if count >= n:
            break
    assert count == n, f"only {count} of {n} images"
    return {k: torch.cat(v) for k, v in out.items()}


def frechet(a, b):
    a, b = a.double(), b.double()
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca, cb = torch.cov(a.T), torch.cov(b.T)
    w, v = torch.linalg.eigh(ca)
    root_a = (v * w.clamp_min(0).sqrt()) @ v.T
    mid = torch.linalg.eigvalsh(root_a @ cb @ root_a).clamp_min(0).sqrt().sum()
    return float((mu_a - mu_b).square().sum() + torch.trace(ca) + torch.trace(cb) - 2 * mid)


def kid(a, b, subsets=100, size=1000, seed=0):
    a, b = a.double(), b.double()
    d = a.shape[1]
    size = min(size, len(a), len(b))
    gen = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(subsets):
        x = a[torch.randperm(len(a), generator=gen)[:size]]
        y = b[torch.randperm(len(b), generator=gen)[:size]]
        kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
        m = size
        total += float((kxx.sum() - kxx.diag().sum()) / (m * (m - 1)) + (kyy.sum() - kyy.diag().sum()) / (m * (m - 1)) - 2 * kxy.mean())
    return total / subsets


def reference(real_batches, n, tag):
    """Features of n real images: from the file when it exists, else computed from real_batches() and saved."""
    path = GEN_DIR / f"ref_{tag}_n{n}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu")
    t0 = time.time()
    # 2n images: the second half is the "real against real" noise floor
    feats = collect(real_batches(), 2 * n)
    ref = {k: v[:n].half() if k != "inception" else v[:n] for k, v in feats.items()}
    floor = {"fid": frechet(feats["inception"][n:], feats["inception"][:n]), "kid": 1000 * kid(feats["inception"][n:], feats["inception"][:n]),
             "fd_ae_z": frechet(feats["ae_z"][n:], feats["ae_z"][:n]), "fd_ae_pool": frechet(feats["ae_pool"][n:], feats["ae_pool"][:n])}
    ref["floor"] = floor
    try:
        tmp = path.with_name(path.name + f".{os.getpid()}.part")
        torch.save(ref, tmp)
        tmp.rename(path)
    except OSError as e:
        print("gen eval: reference not saved (read-only mount?):", e, flush=True)
    print(f"gen eval: reference of {n} real images ({tag}) in {time.time() - t0:.0f}s | real against real: "
          + " ".join(f"{k} {v:.3f}" for k, v in floor.items()), flush=True)
    return ref


def score(sample, real_batches, n, tag, batch=250):
    """sample(seeds) -> images in [0, 1] on the GPU. Returns the scores and the real-against-real floor."""
    ref = reference(real_batches, n, tag)
    t0 = time.time()
    gen = collect((sample(list(range(i, min(i + batch, n)))) for i in range(0, n, batch)), n)
    out = {"fid": frechet(gen["inception"], ref["inception"]), "kid": 1000 * kid(gen["inception"], ref["inception"]),
           "fd_ae_z": frechet(gen["ae_z"], ref["ae_z"].float()), "fd_ae_pool": frechet(gen["ae_pool"], ref["ae_pool"].float()),
           "gen_seconds": time.time() - t0}
    return out, ref["floor"]
