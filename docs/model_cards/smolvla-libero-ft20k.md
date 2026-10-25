---
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
base_model: lerobot/smolvla_base
datasets:
- lerobot/libero
tags:
- smolvla
- vla
- libero
- robotics
- lerobot
model-index:
- name: smolvla-libero-ft20k
  results:
  - task:
      type: robotics
      name: Closed-loop manipulation
    dataset:
      type: libero_spatial
      name: LIBERO-Spatial (10 tasks × 10 episodes, seed 1000)
    metrics:
    - type: success_rate
      value: 78.0
      name: Success rate (n_action_steps=10, 10 denoising steps, FP32)
    - type: success_rate
      value: 81.0
      name: Success rate (n_action_steps=10, 5 denoising steps, BF16 + torch.compile)
---

# SmolVLA fine-tuned on LIBERO (20k steps)

[`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) fine-tuned for 20k steps on all four LIBERO suites ([`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero), 1693 demonstrations) with LeRobot 0.6.1, then evaluated closed-loop in MuJoCo on LIBERO-Spatial.

Code, full results and analysis: [github.com/2p1c/smolvla-libero-edge](https://github.com/2p1c/smolvla-libero-edge)

## Results on LIBERO-Spatial (100 episodes per row)

Seed 1000 and 10 parallel envs for every row, so episode *i* of a task starts from the same scene in every run, and the runs are compared episode by episode (exact McNemar test).

| Model | n_action_steps | Denoising steps | Precision | Success (95% CI) |
|---|---|---|---|---|
| `HuggingFaceVLA/smolvla_libero` (official) | 1 | 10 | FP32 | 69% (59–77) |
| this model | 50 (checkpoint default) | 10 | FP32 | 68% (58–76) |
| this model | 1 | 10 | FP32 | 75% (66–82) |
| this model | **10** | 10 | FP32 | **78% (69–85)** |
| this model | 10 | 5 | BF16 + `torch.compile` | **81% (72–87)** |

None of the differences reach p < 0.05 at 100 episodes (official vs. this model at n=10: 19 episodes went fail→success, 10 went success→fail, p = 0.14).

**Inference latency** (RTX 4090, single frame, p90): 286 ms in FP32 with 10 denoising steps, 37 ms in BF16 + `torch.compile` with 5 denoising steps (7.2× faster). Inference is bound by CPU kernel launches, so BF16 alone barely helps. The first compiled call takes about 2 minutes, so warm up before use.

## Usage

The checkpoint's config keeps `n_action_steps=50`, inherited from `smolvla_base`. That setting scores 10 points lower, so **set `n_action_steps=10` at evaluation time**. The checkpoint expects the cameras as `camera1` and `camera2`, so map LIBERO's `image` and `image2` to them:

```bash
lerobot-eval \
  --policy.path=yutongzhu/smolvla-libero-ft20k \
  --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_spatial \
  --eval.n_episodes=10 --eval.batch_size=10 --seed=1000 \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
```

Add `--policy.num_steps=5` for the faster deployment config.

## Training

| | |
|---|---|
| Base | `lerobot/smolvla_base` |
| Data | `lerobot/libero`, all four suites (spatial, object, goal, libero_10) |
| Steps / batch size | 20,000 / 32 (about 2.3 epochs) |
| Hardware | 1× RTX 4090, 1 h 47 min |
| Final loss | 0.36 |

## Failure analysis

Hand-labeled failed episodes ([labels and videos](https://huggingface.co/datasets/yutongzhu/smolvla-libero-eval-results)):

| Model | Failures | Grabbed the wrong bowl | Missed the grasp | Placement offset |
|---|---|---|---|---|
| official, n=1 | 31 | 23 | 8 | 0 |
| this model, n=1 | 25 | 14 | 5 | 6 |
| this model, n=10 | 22 | 10 | 2 | 10 |

Compared with the official checkpoint, fine-tuning reduces wrong-object and missed-grasp failures but introduces placement offsets. Task 5 ("pick up the black bowl on the ramekin") is the weakest task in every run.

## Limitations

- Evaluated on LIBERO-Spatial only, with 100 episodes per setting.
- Eval renders at 360 px while the training data is 256 px.
- The SmolVLA paper reports 90% on LIBERO-Spatial. Under this project's eval setup the official checkpoint scores 69%, and the gap is not fully explained.
- Simulation only; not tested on a real robot.
