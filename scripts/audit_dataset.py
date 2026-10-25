"""Audit a LeRobot v3 dataset before training on it: composition, episode lengths, idle actions,
gripper behaviour, action saturation, anomalies, and whether normalization stats match the checkpoint.

Reads only the parquet tables (no video decoding), so it runs on CPU in about a minute.

    python scripts/audit_dataset.py
    python scripts/audit_dataset.py --policy outputs/smolvla_libero_ft/checkpoints/last/pretrained_model

Writes results/data_audit/{audit.json, report.md, *.png, spatial_episodes.json}.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "data_audit"
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
# OpenVLA's no-op definition: no translation/rotation command and the gripper command unchanged.
IDLE_EPS = 1e-4
# Inspection showed translation/rotation commands are clipped to +-15/16 by the teleop device.
SATURATION = 0.9375


def dataset_dir(repo_id):
    from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE

    return Path(snapshot_download(repo_id, repo_type="dataset", cache_dir=HF_LEROBOT_HUB_CACHE,
                                  allow_patterns=["meta/*", "meta/**", "data/**"]))


def task_to_suite(task_names):
    """LIBERO task strings are unique per suite, so the suite is recovered by matching the language."""
    from libero.libero import benchmark

    lookup = {}
    for suite in SUITES:
        bench = benchmark.get_benchmark_dict()[suite]()
        for i in range(bench.n_tasks):
            lookup.setdefault(bench.get_task(i).language, []).append(suite)
    suites = {}
    for name in task_names:
        matches = lookup.get(name, [])
        if len(matches) != 1:
            raise SystemExit(f"task {name!r} matches suites {matches}; cannot attribute it")
        suites[name] = matches[0]
    return suites


def per_episode(frames):
    """One row per episode. `frames` is sorted by (episode_index, frame_index)."""
    rows = []
    for ep, g in frames.groupby("episode_index", sort=True):
        a = np.stack(g["action"].to_numpy())
        move = np.abs(a[:, :6]).max(axis=1)
        grip = a[:, 6]
        grip_changed = np.r_[False, grip[1:] != grip[:-1]]
        idle = (move < IDLE_EPS) & ~grip_changed
        leading_idle = int(np.argmin(idle)) if not idle.all() else len(idle)
        rows.append({
            "episode_index": int(ep),
            "task_index": int(g["task_index"].iloc[0]),
            "length": len(g),
            "idle_frames": int(idle.sum()),
            "leading_idle": leading_idle,
            "gripper_switches": int(grip_changed.sum()),
            "gripper_closed_frac": float((grip > 0).mean()),
            "saturated_frac": float((np.abs(a[:, :3]) >= SATURATION - 1e-6).any(axis=1).mean()),
            "has_nan": bool(np.isnan(a).any() or np.isnan(np.stack(g["observation.state"].to_numpy())).any()),
        })
    return pd.DataFrame(rows)


def flag_anomalies(eps, max_steps):
    """Episodes worth a human look. Length outliers are judged within each task (tasks differ a lot)."""
    flags = []
    for task, g in eps.groupby("task_index"):
        q1, q3 = g["length"].quantile([0.25, 0.75])
        hi = q3 + 3 * (q3 - q1)
        for r in g[g["length"] > hi].itertuples():
            flags.append({"episode_index": r.episode_index, "reason": f"length {r.length} > task Q3+3IQR ({hi:.0f})"})
    # Some tasks (push a plate, open a drawer) never grasp, so "no gripper switch" is only suspicious
    # when the task's other demos do grasp.
    task_grasps = eps.groupby("task_index")["gripper_switches"].median() > 0
    for r in eps.itertuples():
        if r.has_nan:
            flags.append({"episode_index": r.episode_index, "reason": "NaN in action or state"})
        if r.gripper_switches == 0 and task_grasps[r.task_index]:
            flags.append({"episode_index": r.episode_index, "reason": "no grasp, unlike the task's other demos"})
        if r.length > max_steps.get(r.suite, 10**9):
            flags.append({"episode_index": r.episode_index,
                          "reason": f"demo longer than eval limit ({r.length} > {max_steps[r.suite]})"})
    return flags


def compare_stats(frames, meta_stats, policy):
    """Recompute mean/std from raw frames and compare with meta/stats.json and the checkpoint's normalizer."""
    from huggingface_hub import snapshot_download as snap

    path = Path(policy) if Path(policy).exists() else Path(snap(policy))
    ckpt = load_file(next(path.glob("policy_preprocessor_step_*_normalizer_processor.safetensors")))
    out = {}
    for key in ["action", "observation.state"]:
        x = np.stack(frames[key].to_numpy()).astype(np.float64)
        ours = {"mean": x.mean(0), "std": x.std(0)}
        row = {}
        for stat in ["mean", "std"]:
            meta = np.asarray(meta_stats[key][stat])
            ck = ckpt[f"{key}.{stat}"].numpy()
            # Differences in units of the data's std, so dims with different scales are comparable.
            scale = ours["std"] + 1e-8
            row[stat] = {
                "recomputed": np.round(ours[stat], 4).tolist(),
                "max_diff_vs_meta_in_std": float(np.max(np.abs(ours[stat] - meta) / scale)),
                "max_diff_vs_checkpoint_in_std": float(np.max(np.abs(ours[stat] - ck) / scale)),
            }
        out[key] = row
    return out, str(path)


def plot(eps, suite_max_steps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, series = "#0b0b0b", "#52514e", "#2a78d6"
    plt.rcParams.update({"axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False})

    def small_multiples(column, xlabel, fname, bins, vline=None):
        fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), sharey=True)
        for ax, suite in zip(axes, SUITES):
            v = eps.loc[eps.suite == suite, column]
            ax.hist(v, bins=bins, color=series, edgecolor="white", linewidth=1)
            if vline:
                ax.axvline(vline[suite], color=muted, linestyle="--", linewidth=1.5)
                ax.text(vline[suite], ax.get_ylim()[1] * 0.95, " eval limit", color=muted, fontsize=8, va="top")
            ax.set_title(f"{suite} (n={len(v)})", fontsize=10, color=ink)
            ax.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("episodes")
        fig.supxlabel(xlabel, fontsize=10, color=ink)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=150)
        plt.close(fig)

    small_multiples("length", "episode length (frames = 20 Hz control steps)", "episode_length.png",
                    bins=30, vline=suite_max_steps)
    small_multiples("gripper_switches", "gripper open/close switches per episode", "gripper_switches.png",
                    bins=np.arange(-0.5, eps.gripper_switches.max() + 1.5, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="lerobot/libero")
    parser.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero",
                        help="checkpoint whose normalizer stats are compared with the data")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    root = dataset_dir(args.dataset)
    info = json.loads((root / "meta" / "info.json").read_text())
    meta_stats = json.loads((root / "meta" / "stats.json").read_text())
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")  # index: task string, column: task_index
    frames = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "task_index", "action",
                                                  "observation.state"])
                       for f in sorted((root / "data").rglob("*.parquet")))
    frames = frames.sort_values(["episode_index", "frame_index"], ignore_index=True)

    suite_of_name = task_to_suite(tasks.index)
    suite_of_index = {int(row.task_index): suite_of_name[name] for name, row in tasks.iterrows()}
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS

    max_steps = {s: TASK_SUITE_MAX_STEPS[s] for s in SUITES}

    eps = per_episode(frames)
    eps["suite"] = eps.task_index.map(suite_of_index)
    eps["task"] = eps.task_index.map({int(r.task_index): name for name, r in tasks.iterrows()})

    by_suite = eps.groupby("suite").agg(
        episodes=("episode_index", "count"),
        frames=("length", "sum"),
        len_min=("length", "min"),
        len_p50=("length", "median"),
        len_max=("length", "max"),
        idle_frac=("idle_frames", lambda s: s.sum()),
        leading_idle_mean=("leading_idle", "mean"),
        gripper_switches_p50=("gripper_switches", "median"),
        saturated_frac=("saturated_frac", "mean"),
    ).reindex(SUITES)
    by_suite["idle_frac"] = by_suite["idle_frac"] / by_suite["frames"]
    by_suite["frame_share"] = by_suite["frames"] / by_suite["frames"].sum()
    by_suite["eval_max_steps"] = [max_steps[s] for s in SUITES]

    by_task = eps.groupby(["suite", "task_index", "task"]).agg(
        episodes=("episode_index", "count"), len_p50=("length", "median")).reset_index()

    flags = flag_anomalies(eps, max_steps)
    stats_check, policy_path = compare_stats(frames, meta_stats, args.policy)

    spatial = sorted(eps.loc[eps.suite == "libero_spatial", "episode_index"].tolist())
    (OUT / "spatial_episodes.json").write_text(json.dumps(spatial))

    audit = {
        "dataset": args.dataset,
        "codebase_version": info.get("codebase_version"),
        "declared_fps": info.get("fps"),
        "eval_control_hz": 20,
        "total_episodes": int(len(eps)),
        "total_frames": int(eps.length.sum()),
        "demos_per_task": {"min": int(by_task.episodes.min()), "max": int(by_task.episodes.max())},
        "by_suite": json.loads(by_suite.round(4).to_json(orient="index")),
        "idle_frames_total": int(eps.idle_frames.sum()),
        "episodes_with_leading_idle": int((eps.leading_idle > 0).sum()),
        "gripper_values": sorted(np.unique(np.stack(frames["action"].to_numpy())[:, 6]).tolist()),
        "anomalies": flags,
        "normalization_check": {"policy": policy_path, **stats_check},
        "spatial_episode_count": len(spatial),
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2))
    by_task.to_csv(OUT / "per_task.csv", index=False)
    eps.to_csv(OUT / "per_episode.csv", index=False)
    plot(eps, max_steps)

    print(by_suite.round(3).to_string())
    print(f"\ndemos per task: {audit['demos_per_task']}  idle frames: {audit['idle_frames_total']}  "
          f"episodes with leading idle: {audit['episodes_with_leading_idle']}")
    print(f"gripper values: {audit['gripper_values']}  declared fps: {audit['declared_fps']} (eval runs 20 Hz)")
    print(f"anomalies: {len(flags)}")
    for f in flags[:15]:
        print("  ", f)
    for key, row in stats_check.items():
        print(f"{key}: mean diff vs meta {row['mean']['max_diff_vs_meta_in_std']:.4f} std, "
              f"vs checkpoint {row['mean']['max_diff_vs_checkpoint_in_std']:.4f} std; "
              f"std diff vs checkpoint {row['std']['max_diff_vs_checkpoint_in_std']:.4f} std")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
