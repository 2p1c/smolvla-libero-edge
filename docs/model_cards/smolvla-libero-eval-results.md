---
license: apache-2.0
pretty_name: SmolVLA LIBERO-Spatial closed-loop eval results and failure labels
task_categories:
- robotics
tags:
- smolvla
- libero
- evaluation
- failure-analysis
---

# SmolVLA on LIBERO-Spatial: eval results and failure labels

Closed-loop evaluation results from [github.com/2p1c/smolvla-libero-edge](https://github.com/2p1c/smolvla-libero-edge), comparing the official [`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero) with [`yutongzhu/smolvla-libero-ft20k`](https://huggingface.co/yutongzhu/smolvla-libero-ft20k) and [`yutongzhu/smolvla-libero-spatial-only-10k`](https://huggingface.co/yutongzhu/smolvla-libero-spatial-only-10k).

The training data is the public [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero) and is not duplicated here.

## Contents

```
results/<run>/eval_info.json    per-episode success, rewards, seeds (LeRobot lerobot-eval output)
results/<run>/meta.json         policy, n_action_steps, denoising steps, extra args
results/<run>/runtime.json      precision / compile settings and per-inference latency (BF16 run only)
results/latency_ft20k.json      single-frame latency benchmark (RTX 4090)
failure_labels/*.csv            one row per failed episode: task, episode, category, note, video
failure_labels/README.md        labeling rules
videos/<run>/libero_spatial_<task>/eval_episode_<i>.mp4   failed episodes only
```

All runs use LIBERO-Spatial, 10 episodes per task, seed 1000 and 10 parallel envs, so episode *i* of a task has the same initial scene in every run and runs can be compared episode by episode.

Eval videos are encoded at 80 fps, but each frame is one 20 Hz control step, so play them at 0.25× for real time.

## Failure labels

Each failed episode gets the earliest failure that occurs:

| Category | Rule |
|---|---|
| `approach_wrong_dir` | moves to, or grabs, the wrong object |
| `grasp_miss` | reaches the right object but the gripper closes without holding it |
| `place_offset` | picks the right object but fails or misses on placement |

| Run | Failures | approach_wrong_dir | grasp_miss | place_offset |
|---|---|---|---|---|
| official_n1 | 31 | 23 | 8 | 0 |
| ft20k_n1 | 25 | 14 | 5 | 6 |
| ft20k_n10 | 22 | 10 | 2 | 10 |

Labels are from a single annotator.
