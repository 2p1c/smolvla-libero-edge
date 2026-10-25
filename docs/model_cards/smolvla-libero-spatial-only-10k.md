---
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
base_model: lerobot/smolvla_base
datasets:
- lerobot/libero
language:
- zh
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
      name: 闭环操作
    dataset:
      type: libero_spatial
      name: LIBERO-Spatial（10 个任务 × 10 回合，种子 1000）
    metrics:
    - type: success_rate
      value: 80.0
      name: 成功率（n_action_steps=10，去噪 10 步，FP32）
---

# SmolVLA 只用 LIBERO-Spatial 数据微调 10k 步

这是一个数据配比实验：以 [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) 为起点，**只用 [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero) 里 LIBERO-Spatial 的 432 条示教**（约占全部回合的 26%）微调 10k 步。对照组是 [`yutongzhu/smolvla-libero-ft20k`](https://huggingface.co/yutongzhu/smolvla-libero-ft20k)：用全部四个子集训练 20k 步。

代码、完整结果和分析：[github.com/2p1c/smolvla-libero-edge](https://github.com/2p1c/smolvla-libero-edge)

## LIBERO-Spatial 结果（每行 100 回合，种子 1000，n_action_steps=10）

| 模型 | 训练数据 | 步数 | 成功率（95% 置信区间） |
|---|---|---|---|
| `yutongzhu/smolvla-libero-ft20k` | 四个子集（1693 条示教） | 20k | 78%（69–85） |
| **本模型** | 只用 spatial（432 条示教） | 10k | **80%（71–87）** |
| `HuggingFaceVLA/smolvla_libero`（官方，n=1） | — | — | 69%（59–77） |

和四子集模型逐回合配对比较：12 个回合由败转胜，10 个由胜转败（精确 McNemar 检验 p=0.83）。在 LIBERO-Spatial 上，只用目标子集训练一半步数，就能和全量混合数据持平。

但这不能说明其他子集没有用：这组对比里，每条 spatial 示教被看到的次数约为对照组的 2 倍（10k 步只用 26% 的数据，对照组 20k 步用全部数据），而且没有测在其他子集上的迁移效果。

## 使用方法

和 20k 模型一样，配置里的 `n_action_steps` 还是继承自 `smolvla_base` 的 50，需要显式设置，并映射相机名：

```bash
lerobot-eval \
  --policy.path=yutongzhu/smolvla-libero-spatial-only-10k \
  --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_spatial \
  --eval.n_episodes=10 --eval.batch_size=10 --seed=1000 \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
```

## 训练配置

| | |
|---|---|
| 起点模型 | `lerobot/smolvla_base` |
| 数据 | `lerobot/libero` 中 libero_spatial 的 432 个回合（`--dataset.episodes`） |
| 步数 / batch size | 10,000 / 32 |
| 硬件 | 1 张 RTX 4090 |
| 归一化 | 只用部分回合训练时，LeRobot 仍使用全量数据的统计量（`meta/stats.json`） |

## 局限性

- 只在 LIBERO-Spatial 上评测，没有测在其他子集上的泛化。
- 每个设置 100 回合，和四子集模型的差距在噪声范围内。
- 只在仿真中测试。
