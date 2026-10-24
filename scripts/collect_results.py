"""Turn results/*/eval_info.json into a markdown table, and optionally plot the n_action_steps sweep.

    python scripts/collect_results.py                       # table of every eval
    python scripts/collect_results.py --plot-sweep sweep_official_
"""

import argparse
import json
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


def plot_sweep(rows, prefix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = sorted((r["n_action_steps"], r["success"]) for r in rows if r["n_action_steps"] is not None)
    if not points:
        raise SystemExit(f"no sweep results under results/{prefix}*")
    xs, ys = zip(*points)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(xs, ys, marker="o")
    ax.set_xscale("log")
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_xlabel("n_action_steps (actions executed per inference)")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"{rows[0]['suite']}, {rows[0]['episodes']} episodes per point")
    ax.grid(alpha=0.3)
    out = RESULTS / f"{prefix.rstrip('_')}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


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
