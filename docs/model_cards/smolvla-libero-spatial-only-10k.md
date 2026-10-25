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
- name: smolvla-libero-spatial-only-10k
  results:
  - task:
      type: robotics
      name: Closed-loop manipulation
    dataset:
      type: libero_spatial
      name: LIBERO-Spatial (10 tasks × 10 episodes, seed 1000)
    metrics:
    - type: success_rate
      value: 80.0
      name: Success rate (n_action_steps=10, 10 denoising steps, FP32)
---

# SmolVLA fine-tuned on LIBERO-Spatial only (10k steps)

A data-mixture ablation: [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) fine-tuned for 10k steps on **only the 432 LIBERO-Spatial demonstrations** of [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero), about 26% of the episodes. It is compared with [`yutongzhu/smolvla-libero-ft20k`](https://huggingface.co/yutongzhu/smolvla-libero-ft20k), which was trained for 20k steps on all four suites.

Code, full results and analysis: [github.com/2p1c/smolvla-libero-edge](https://github.com/2p1c/smolvla-libero-edge)

## Results on LIBERO-Spatial (100 episodes per row, seed 1000, n_action_steps=10)

| Model | Training data | Steps | Success (95% CI) |
|---|---|---|---|
| `yutongzhu/smolvla-libero-ft20k` | 4 suites (1693 demos) | 20k | 78% (69–85) |
| **this model** | spatial only (432 demos) | 10k | **80% (71–87)** |
| `HuggingFaceVLA/smolvla_libero` (official, n=1) | — | — | 69% (59–77) |

Paired per-episode comparison with the 4-suite model: 12 episodes went fail→success and 10 went success→fail (exact McNemar p = 0.83). On LIBERO-Spatial, training on the target suite alone matches the full mixture at half the steps. This is not evidence that the other suites are useless: in this comparison each spatial demo is seen about 2× as often (10k steps over 26% of the data vs. 20k steps over all of it), and transfer to the other suites was not measured.

## Usage

As with the 20k model, the config keeps `n_action_steps=50` from `smolvla_base`, so set it explicitly and map the camera names:

```bash
lerobot-eval \
  --policy.path=yutongzhu/smolvla-libero-spatial-only-10k \
  --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_spatial \
  --eval.n_episodes=10 --eval.batch_size=10 --seed=1000 \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
```

## Training

| | |
|---|---|
| Base | `lerobot/smolvla_base` |
| Data | `lerobot/libero`, the 432 libero_spatial episodes (`--dataset.episodes`) |
| Steps / batch size | 10,000 / 32 |
| Hardware | 1× RTX 4090 |
| Normalization | LeRobot uses the full-dataset statistics in `meta/stats.json`, even when training on a subset |

## Limitations

- Evaluated on LIBERO-Spatial only; generalization to the other suites was not tested.
- 100 episodes per setting; the difference from the 4-suite model is within noise.
- Simulation only.
