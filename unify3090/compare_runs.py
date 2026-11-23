"""Side-by-side numbers of the XM series runs, to see whether XM changed anything (host python).

    python3 compare_runs.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path("<data-root>/outputs/3090-experiments/runs")
latest = {r["xm_k"]: r for r in json.loads((HERE / "series.json").read_text())["runs"]}
data = {}
for k, r in sorted(latest.items()):
    f = RUNS / f"{r['name']}--{r['job']}" / "metrics.jsonl"
    data[k] = {x["step"]: x for x in map(json.loads, f.read_text().splitlines())} if f.exists() else {}
    print(f"xm{k}: job {r['job'][:8]} | {len(data[k])} records | last step {max(data[k], default=0)} | config xm_k={r['config']['xm_k']}")

ks = [k for k in data if data[k]]
print("\ntest loss (one noise draw, same images / t / noise in every run)")
print("step   " + "".join(f"{'xm' + str(k):>11}" for k in ks) + "   | spread between runs")
for s in (0, 1000, 2000, 3000, 5000, 10000, 15000, 20000, 25000, 30000):
    vals = {k: data[k][s]["test_clean"] for k in ks if s in data[k] and "test_clean" in data[k][s]}
    if vals:
        sp = (max(vals.values()) - min(vals.values())) / min(vals.values()) if len(vals) > 1 else 0
        print(f"{s:<7}" + "".join(f"{vals[k]:>11.5f}" if k in vals else f"{'':>11}" for k in ks) + f"   | {100 * sp:5.2f}%")

print("\ntrain records at the same steps (clean = loss of the chosen draw; loss_all = mean over all K draws)")
for s in (100, 200, 500, 1000, 5000, 10000, 20000, 30000):
    parts = []
    for k in ks:
        r = data[k].get(s)
        if r:
            parts.append(f"xm{k}: clean {r['clean']:.5f}" + (f" all {r['loss_all']:.5f} gap {100 * r['xm_gap'] / r['loss_all']:.1f}%" if "loss_all" in r else ""))
    print(f"{s:<6} " + " | ".join(parts))

if len(ks) >= 2:
    a, b = ks[-2], ks[-1]
    common = sorted(set(data[a]) & set(data[b]) - {0})
    same = sum(data[a][s]["clean"] == data[b][s]["clean"] for s in common)
    print(f"\nxm{a} vs xm{b}: {same} of {len(common)} train records are bit-identical")
    for k in ks:
        gaps = [data[k][s]["xm_gap"] / data[k][s]["loss_all"] for s in sorted(data[k]) if "xm_gap" in data[k][s]]
        if gaps:
            print(f"xm{k}: chosen draw is on average {100 * sum(gaps) / len(gaps):.2f}% below the mean of its {k} draws (min {100 * min(gaps):.2f}%, max {100 * max(gaps):.2f}%)")
