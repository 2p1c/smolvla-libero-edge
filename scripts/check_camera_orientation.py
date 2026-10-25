"""Check that the images a policy sees in eval have the same orientation as the training frames.

Builds the LIBERO env and observation pipeline exactly as lerobot-eval does (make_env ->
preprocess_observation -> env preprocessor, which rotates 180 deg), takes the first frame of each camera,
and compares it with frame 0 of training episodes of the same task under identity / hflip / vflip / rot180.
The best-matching transform should be identity; objects move between episodes but the static scene
(table, cabinet, walls) dominates a downsampled image.

    python scripts/check_camera_orientation.py --task-id 0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "orientation_check"
TRANSFORMS = {
    "identity": lambda x: x,
    "hflip": lambda x: x[:, ::-1],
    "vflip": lambda x: x[::-1, :],
    "rot180": lambda x: x[::-1, ::-1],
}


def eval_frames(suite, task_id, seed):
    """First observation of one episode, through the same pipeline lerobot-eval uses."""
    from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    env_cfg = make_env_config("libero", task=suite, task_ids=[task_id])
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[suite][task_id]
    env_pre, _ = make_env_pre_post_processors(env_cfg, SmolVLAConfig())
    raw, _ = env.reset(seed=[seed])
    obs = env_pre(preprocess_observation(raw))
    frames = {}
    for key, value in obs.items():
        if key.startswith("observation.images."):
            img = value[0].float()  # (C, H, W) in [0, 1]
            frames[key] = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    raw_first = {k: np.asarray(v[0]) for k, v in raw["pixels"].items()}
    env.close()
    return frames, raw_first


def dataset_frames(repo_id, task_text, n_episodes):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    eps = pd.read_csv(ROOT / "results" / "data_audit" / "per_episode.csv")
    ep_ids = sorted(eps.loc[eps.task == task_text, "episode_index"].tolist())[:n_episodes]
    ds = LeRobotDataset(repo_id, episodes=ep_ids)
    # A subset dataset is indexed contiguously over the selected episodes; frame 0 of each is at the
    # running sum of the previous episodes' lengths.
    out = []
    idx = 0
    for ep in ep_ids:
        item = ds[idx]
        assert int(item["episode_index"]) == ep and int(item["frame_index"]) == 0
        out.append({k: (item[k].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    for k in item if k.startswith("observation.images.")})
        idx += int(eps.loc[eps.episode_index == ep, "length"].iloc[0])
    return ep_ids, out


def small_gray(img, size=64):
    t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)[None]
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="area")[0].mean(0)
    return t.numpy()


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--dataset", default="lerobot/libero")
    parser.add_argument("--n-episodes", type=int, default=5)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from libero.libero import benchmark

    task_text = benchmark.get_benchmark_dict()[args.suite]().get_task(args.task_id).language
    sim, sim_raw = eval_frames(args.suite, args.task_id, args.seed)
    ep_ids, data = dataset_frames(args.dataset, task_text, args.n_episodes)

    from PIL import Image

    report = {"task": task_text, "dataset_episodes": ep_ids, "sim_keys": sorted(sim), "cameras": {}}
    for key, sim_img in sim.items():
        scores = {name: float(np.mean([corr(small_gray(np.ascontiguousarray(f(sim_img))), small_gray(d[key]))
                                       for d in data]))
                  for name, f in TRANSFORMS.items()}
        report["cameras"][key] = {"corr_by_transform_applied_to_eval_frame": scores,
                                  "best": max(scores, key=scores.get)}
        report["cameras"][key]["eval_shape"] = list(sim_img.shape)
        report["cameras"][key]["dataset_shape"] = list(data[0][key].shape)
        cam = key.split(".")[-1]
        h, w = data[0][key].shape[:2]
        row = [np.asarray(Image.fromarray(sim_img).resize((w, h), Image.BILINEAR)), data[0][key]]
        Image.fromarray(np.concatenate(row, axis=1)).save(OUT / f"{args.suite}_t{args.task_id}_{cam}_eval_vs_data.png")
    for cam, img in sim_raw.items():
        Image.fromarray(img).save(OUT / f"{args.suite}_t{args.task_id}_{cam}_raw_env.png")

    (OUT / f"{args.suite}_t{args.task_id}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
