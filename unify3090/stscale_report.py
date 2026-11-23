"""Report of the self-teaching scaling sweep (run by `stscale.py report` in the PyTorch container: numpy + matplotlib).

Reads /runs/unify-wclamp01-stscale-<scale>-<variant>-seed<k>-100k-spark/ (metrics.jsonl, gen_metrics.jsonl, params.json,
stscale_status.json) and writes /out: results.json, report.md, fid_vs_params.png, gain_abs.png, gain_pct.png.

Primary result: EMA FID at the final checkpoint (step 100,000), never the best checkpoint of a run.
gain = FID_baseline - FID_selfteach, per seed (paired), then mean and std over the seeds. Positive = self-teaching helped.
percentage gain = 100 * gain / FID_baseline, per seed. std = sample std (ddof 1) over the seeds that have both runs.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS, OUT = Path("/runs"), Path("/out")
SCALES = ["1m", "2m", "5m", "10m", "20m", "50m"]
# STSCALE_SERIES=sf: the self-flow series (self-flow in every run; base = self-flow only), folders unify-wclamp01-sfscale-*
SF = os.environ.get("STSCALE_SERIES", "st") == "sf"
PREFIX = "sfscale" if SF else "stscale"
VARIANTS = ({"base": "self-flow only", "st3": "self-flow + self-teaching, judge block 3", "st6": "self-flow + self-teaching, judge block 6"} if SF
            else {"base": "no self-teaching", "st3": "self-teaching, judge block 3", "st6": "self-teaching, judge block 6"})
SEEDS = [0, 1] if SF else [0, 1, 2]
# STSCALE_TAG / STSCALE_FINAL: report on smoke folders (unify-wclamp01-<prefix>-<tag>-...-spark) to test this script
TAG = os.environ.get("STSCALE_TAG", "")
FINAL = int(os.environ.get("STSCALE_FINAL", 100000))
STEPS = [FINAL * k // 4 for k in (1, 2, 3, 4)]
KEY = "fid_ema"
# black/white high contrast (user, 2026-10-24): pixel std and the share of pixels below 0.02 / above 0.98.
# Real values: 2,400 validation images at the job's 128 px resize.
LOOK = ("contrast", "dark", "light")
LOOK_REAL = {"contrast": 0.270, "dark": 0.031, "light": 0.027}


def name(sc, v, sd):
    if TAG:
        return f"unify-wclamp01-{PREFIX}-{TAG}-{sc}-{v}-seed{sd}-spark"
    return f"unify-wclamp01-{PREFIX}-{sc}-{v}-seed{sd}-100k-spark"


def records(d):
    rows = []
    for f in ("metrics.jsonl", "gen_metrics.jsonl"):
        p = d / f
        if p.exists():
            for line in p.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def run_info(sc, v, sd):
    d = RUNS / name(sc, v, sd)
    if not d.is_dir():
        return None
    rows = records(d)
    status = json.loads((d / "stscale_status.json").read_text()) if (d / "stscale_status.json").exists() else None
    params = json.loads((d / "params.json").read_text()) if (d / "params.json").exists() else None
    fid = {int(r["step"]): r[KEY] for r in rows if KEY in r and not r.get("gen_fix")}
    kid = {int(r["step"]): r.get("kid_ema") for r in rows if "kid_ema" in r}
    secs = {int(r["step"]): r["train_seconds"] for r in rows if "train_seconds" in r}
    # contrast of the 12 fixed-seed previews from the EMA weights at the final step (image_detail in main.py)
    look = next((r for r in rows if int(r.get("step", -1)) == FINAL and "ema_prev_contrast" in r), None)
    look = look and {k: look[f"ema_prev_{k}"] for k in LOOK}
    pairing = (d / "pairing.jsonl").exists() and not any(json.loads(l).get("resumed") for l in (d / "pairing.jsonl").read_text().splitlines() if l.strip())
    return {"fid": fid, "kid": kid, "train_seconds": secs.get(FINAL) or (max(secs.values()) if secs else None),
            "train_steps": max(secs) if secs else 0, "ok": bool(status and status["ok"]), "wall_seconds": status and status["wall_seconds"],
            "trainable": params and params["trainable"], "width": params and params["width"], "paired": pairing, "look": look}


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return float(np.mean(xs)), (float(np.std(xs, ddof=1)) if len(xs) > 1 else None), len(xs)


info = {(sc, v, sd): run_info(sc, v, sd) for sc in SCALES for v in VARIANTS for sd in SEEDS}
res = {"primary": f"{KEY} at step {FINAL:,}", "scales": {}}
for sc in SCALES:
    s = res["scales"][sc] = {"trainable": None, "variants": {}, "gains": {}}
    counts = [i["trainable"] for (a, _, _), i in info.items() if a == sc and i and i["trainable"]]
    s["trainable"] = counts[0] if counts else None
    assert len(set(counts)) <= 1, (sc, counts)          # the three variants have the same trainable model
    for v in VARIANTS:
        per_seed = {sd: info[(sc, v, sd)] for sd in SEEDS if info[(sc, v, sd)]}
        entry = s["variants"][v] = {"steps": {}}
        for step in STEPS:
            m, sdv, n = stats([i["fid"].get(step) for i in per_seed.values()])
            entry["steps"][step] = {"mean": m, "std": sdv, "n": n}
        m, sdv, n = stats([i["train_seconds"] / max(i["train_steps"], 1) * 1000 if i["train_seconds"] else None for i in per_seed.values()])
        entry["ms_per_step"] = {"mean": m, "std": sdv, "n": n}
        entry["gpu_hours_total"] = sum((i["wall_seconds"] or 0) for i in per_seed.values()) / 3600
        entry["per_seed_final"] = {sd: i["fid"].get(FINAL) for sd, i in per_seed.items()}
    for v in ("st3", "st6"):
        g = {}
        for step in STEPS:
            pairs = [(info[(sc, "base", sd)], info[(sc, v, sd)]) for sd in SEEDS]
            pairs = [(b["fid"].get(step), t["fid"].get(step)) for b, t in pairs if b and t]
            pairs = [(b, t) for b, t in pairs if b is not None and t is not None]
            ma, sa, n = stats([b - t for b, t in pairs])
            mp, sp, _ = stats([100 * (b - t) / b for b, t in pairs])
            g[step] = {"abs_mean": ma, "abs_std": sa, "pct_mean": mp, "pct_std": sp, "n_pairs": n}
        s["gains"][v] = g
(OUT / "results.json").write_text(json.dumps(res, indent=1))


def fmt(m, s, nd=2):
    if m is None:
        return "-"
    return f"{m:.{nd}f}" + (f" ± {s:.{nd}f}" if s is not None else "")


lines = ["# Self-teaching on self-flow, scaling sweep" if SF else "# Self-teaching scaling sweep", "",
         *(["Every run has self-flow (3 -> 6, weight 1, EMA teacher). base = self-flow only; st3 / st6 add self-teaching.", ""] if SF else []),
         f"Primary result: EMA FID at the final checkpoint (step {FINAL:,}); mean ± std over the seeds that have finished.",
         "Gain = FID_baseline - FID_selfteach, paired by seed. Positive gain = self-teaching helped.", "",
         "| scale | trainable params | FID base | FID st3 | FID st6 | gain st3 | gain st6 | gain st3 % | gain st6 % | seeds |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for sc in SCALES:
    s = res["scales"][sc]
    f = {v: s["variants"][v]["steps"][FINAL] for v in VARIANTS}
    g3, g6 = s["gains"]["st3"][FINAL], s["gains"]["st6"][FINAL]
    lines.append(f"| {sc} | {s['trainable']:,} | " if s["trainable"] else f"| {sc} | - | ")
    lines[-1] += " | ".join([fmt(f["base"]["mean"], f["base"]["std"]), fmt(f["st3"]["mean"], f["st3"]["std"]), fmt(f["st6"]["mean"], f["st6"]["std"]),
                             fmt(g3["abs_mean"], g3["abs_std"]), fmt(g6["abs_mean"], g6["abs_std"]),
                             fmt(g3["pct_mean"], g3["pct_std"], 1), fmt(g6["pct_mean"], g6["pct_std"], 1),
                             f"{f['base']['n']}/{f['st3']['n']}/{f['st6']['n']}"]) + " |"
lines += ["", "## Gain at matched steps (EMA FID, paired by seed)", "", "| scale | variant | " + " | ".join(f"{s // 1000}k" if s >= 1000 else f"{s}" for s in STEPS) + " |",
          "|---|---|" + "---|" * len(STEPS)]
for sc in SCALES:
    for v in ("st3", "st6"):
        g = res["scales"][sc]["gains"][v]
        lines.append(f"| {sc} | {v} | " + " | ".join(fmt(g[s]["abs_mean"], g[s]["abs_std"]) for s in STEPS) + " |")
lines += ["", "## Compute cost (training only: train_seconds excludes evals, previews and FID scoring)", "",
          "| scale | ms/step base | ms/step st3 | ms/step st6 | st3 / base | st6 / base | GPU hours of the runs (wall, with scoring) |", "|---|---|---|---|---|---|---|"]
for sc in SCALES:
    vv = res["scales"][sc]["variants"]
    ms = {v: vv[v]["ms_per_step"]["mean"] for v in VARIANTS}
    ratio = lambda v: f"{ms[v] / ms['base']:.2f}" if ms[v] and ms["base"] else "-"
    lines.append(f"| {sc} | {fmt(ms['base'], None, 1)} | {fmt(ms['st3'], None, 1)} | {fmt(ms['st6'], None, 1)} | {ratio('st3')} | {ratio('st6')} | "
                 f"{sum(vv[v]['gpu_hours_total'] for v in VARIANTS):.1f} |")
lines += ["", f"## Black/white contrast of the samples (12 fixed seeds, EMA weights, step {FINAL:,}; mean over the seeds)", "",
          f"Real images: pixel std {LOOK_REAL['contrast']:.3f}, black (< 0.02) {100 * LOOK_REAL['dark']:.1f} %, white (> 0.98) {100 * LOOK_REAL['light']:.1f} %. "
          "Sets of 12 real images span std 0.24-0.30 and black 0.4-8.5 % (5-95 %).", "",
          "| scale | std base | std st3 | std st6 | black % base | black % st3 | black % st6 | white % base | white % st3 | white % st6 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
for sc in SCALES:
    cell = {}
    for v in VARIANTS:
        ls = [info[(sc, v, sd)]["look"] for sd in SEEDS if info[(sc, v, sd)] and info[(sc, v, sd)]["look"]]
        cell[v] = {k: (float(np.mean([l[k] for l in ls])) if ls else None) for k in LOOK}
        res["scales"][sc]["variants"][v]["look"] = cell[v]
    f3 = lambda x: "-" if x is None else f"{x:.3f}"
    pc = lambda x: "-" if x is None else f"{100 * x:.1f}"
    lines.append(f"| {sc} | " + " | ".join([f3(cell[v]["contrast"]) for v in VARIANTS] + [pc(cell[v]["dark"]) for v in VARIANTS]
                                            + [pc(cell[v]["light"]) for v in VARIANTS]) + " |")
(OUT / "results.json").write_text(json.dumps(res, indent=1))
unpaired = [name(*k) for k, i in info.items() if i and not i["paired"]]
failed = [name(*k) for k, i in info.items() if i and not i["ok"]]
lines += ["", f"Runs not paired after a resume: {unpaired or 'none'}. Runs not ended ok (running, failed or stopped by the guard): {failed or 'none'}.", "",
          "Reading: gains that grow with the parameter count support the scaling hunch. The sweep scales learner and judge",
          "together, so it does not show that a stronger judge causes the gain."]
(OUT / "report.md").write_text("\n".join(lines) + "\n")

# plots: x = measured trainable parameters (log), error bars = std over seeds
xs = {sc: res["scales"][sc]["trainable"] for sc in SCALES}
colors = {"base": "#555555", "st3": "#1f77b4", "st6": "#d62728"}


def series(fn):
    pts = [(xs[sc], *fn(sc)) for sc in SCALES if xs[sc]]
    pts = [p for p in pts if p[1] is not None]
    return [p[0] for p in pts], [p[1] for p in pts], [p[2] or 0 for p in pts]


fig, ax = plt.subplots(figsize=(7, 4.5))
for v, label in VARIANTS.items():
    x, y, e = series(lambda sc: (res["scales"][sc]["variants"][v]["steps"][FINAL]["mean"], res["scales"][sc]["variants"][v]["steps"][FINAL]["std"]))
    if x:
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, color=colors[v], label=label)
ax.set_xscale("log"); ax.set_xlabel("trainable parameters (frozen judge copy not counted)"); ax.set_ylabel(f"EMA FID at step {FINAL:,} (lower is better)")
ax.set_title("FID against model size"); ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "fid_vs_params.png", dpi=130); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, k, ylabel in zip(axes, LOOK, ("pixel std", "share of black pixels (< 0.02)", "share of white pixels (> 0.98)")):
    for v, label in VARIANTS.items():
        pts = [(xs[sc], res["scales"][sc]["variants"][v]["look"][k]) for sc in SCALES if xs[sc] and res["scales"][sc]["variants"][v]["look"][k] is not None]
        if pts:
            ax.plot(*zip(*pts), marker="o", color=colors[v], label=label)
    ax.axhline(LOOK_REAL[k], color="black", lw=0.8, ls=":", label="real images")
    ax.set_xscale("log"); ax.set_xlabel("trainable parameters"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.suptitle(f"Black/white contrast of the samples (12 fixed seeds, EMA weights, step {FINAL:,})")
fig.tight_layout(); fig.savefig(OUT / "contrast_vs_params.png", dpi=130); plt.close(fig)

for kind, ylabel, fname in (("abs", "gain = FID_base - FID_selfteach", "gain_abs.png"), ("pct", "gain in % of FID_base", "gain_pct.png")):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, v in zip(axes, ("st3", "st6")):
        x, y, e = series(lambda sc: (res["scales"][sc]["gains"][v][FINAL][kind + "_mean"], res["scales"][sc]["gains"][v][FINAL][kind + "_std"]))
        if x:
            ax.errorbar(x, y, yerr=e, marker="o", capsize=3, color=colors[v])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xscale("log"); ax.set_xlabel("trainable parameters"); ax.set_title(VARIANTS[v] + " (positive = helped)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=130); plt.close(fig)
print("\n".join(lines))
