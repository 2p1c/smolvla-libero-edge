"""Turn results/*/eval_info.json into a markdown table, and optionally plot the n_action_steps sweep.

    python scripts/collect_results.py                       # table of every eval
    python scripts/collect_results.py --plot-sweep sweep_official_
"""

import argparse
import json
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load_rows(prefix=""):
    rows = []
    for info_path in sorted(RESULTS.glob(f"{prefix}*/eval_info.json")):
        run = info_path.parent
        info = json.loads(info_path.read_text())
        meta_path = run / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        overall = info["overall"]
        rows.append({
            "tag": run.name,
            "policy": meta.get("policy", "?"),
            "suite": meta.get("suite", "?"),
            "n_action_steps": meta.get("n_action_steps"),
            "num_steps": meta.get("num_steps"),
            "episodes": overall["n_episodes"],
            "success": overall["pc_success"],
            "ci95": overall.get("pc_success_ci95"),
            "sec_per_episode": overall.get("eval_ep_s"),
        })
    return rows


def fmt(value, spec="{}", default="default"):
    return default if value is None else spec.format(value)


def print_table(rows):
    print("| tag | policy | suite | n_action_steps | num_steps | episodes | success % | s/episode |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        success = f"{r['success']:.1f}"
        if r["ci95"]:
            success += f" ({r['ci95'][0]:.0f}–{r['ci95'][1]:.0f})"
        print(
            f"| {r['tag']} | {r['policy']} | {r['suite']} | {fmt(r['n_action_steps'])} | "
            f"{fmt(r['num_steps'])} | {r['episodes']} | {success} | {fmt(r['sec_per_episode'], '{:.1f}', '-')} |"
        )


def common_episode_success(tags):
    """Success over the episodes every run shares: the first k episodes of each task, k = the smallest
    per-task count. Same seed and eval batch means episode i starts from the same scene in every run."""
    per_task = {}
    for tag in tags:
        info = json.loads((RESULTS / tag / "eval_info.json").read_text())
        per_task[tag] = {t["task_id"]: t["metrics"]["successes"] for t in info["per_task"]}
    k = min(len(s) for runs in per_task.values() for s in runs.values())
    out = {}
    for tag, runs in per_task.items():
        hits = [bool(x) for s in runs.values() for x in s[:k]]
        out[tag] = (sum(hits), len(hits))
    return out, k


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (centre - half), 100 * (centre + half)


def plot_sweep(rows, prefix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r["n_action_steps"] is not None]
    if not rows:
        raise SystemExit(f"no sweep results under results/{prefix}*")
    counts, k = common_episode_success([r["tag"] for r in rows])
    points = sorted((r["n_action_steps"], *counts[r["tag"]]) for r in rows)
    xs = [x for x, _, _ in points]
    ys = [100 * hit / n for _, hit, n in points]
    lo, hi = zip(*(wilson(hit, n) for _, hit, n in points))
    ink, muted, series = "#0b0b0b", "#52514e", "#2a78d6"
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.fill_between(xs, lo, hi, color=series, alpha=0.15, linewidth=0, label="95% CI (Wilson)")
    ax.plot(xs, ys, color=series, linewidth=2, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=2)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color=ink)
    ax.set_xscale("log")
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.minorticks_off()
    ax.set_xlabel("n_action_steps (actions executed per inference)", color=ink)
    ax.set_ylabel("success rate (%)", color=ink)
    ax.set_ylim(0, 100)
    ax.tick_params(colors=muted)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"{rows[0]['suite']}: same {points[0][2]} episodes per point ({k} per task)", fontsize=10, color=ink)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    out = RESULTS / f"{prefix.rstrip('_')}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")
    for (x, hit, n), l, h in zip(points, lo, hi):
        print(f"n={x}: {hit}/{n} = {100 * hit / n:.0f}% (95% CI {l:.0f}-{h:.0f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="", help="only include result dirs starting with this")
    parser.add_argument("--plot-sweep", metavar="PREFIX", help="plot success vs n_action_steps for these dirs")
    args = parser.parse_args()

    prefix = args.plot_sweep or args.prefix
    rows = load_rows(prefix)
    print_table(rows)
    if args.plot_sweep:
        plot_sweep(rows, args.plot_sweep)


if __name__ == "__main__":
    main()
