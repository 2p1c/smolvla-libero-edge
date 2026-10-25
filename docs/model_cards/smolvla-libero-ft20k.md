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
- name: smolvla-libero-ft20k
  results:
  - task:
      type: robotics
      name: 闭环操作
    dataset:
      type: libero_spatial
      name: LIBERO-Spatial（10 个任务 × 10 回合，种子 1000）
    metrics:
    - type: success_rate
      value: 78.0
      name: 成功率（n_action_steps=10，去噪 10 步，FP32）
    - type: success_rate
      value: 81.0
      name: 成功率（n_action_steps=10，去噪 5 步，BF16 + torch.compile）
---

# SmolVLA 在 LIBERO 上微调 20k 步

以 [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) 为起点，用 LeRobot 0.6.1 在 LIBERO 全部四个子集（[`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero)，1693 条示教）上微调 20k 步，然后在 MuJoCo 仿真的 LIBERO-Spatial 上做闭环评测。

代码、完整结果和分析：[github.com/2p1c/smolvla-libero-edge](https://github.com/2p1c/smolvla-libero-edge)

## 演示

本模型在 LIBERO-Spatial 中的闭环执行（n_action_steps=10，第三人称视角）。失败案例的动图按 2 倍速播放。

| 成功（任务 0） | 抓错了碗（任务 5） | 没抓住（任务 8） | 放置偏差（任务 8） |
|---|---|---|---|
| ![成功](https://huggingface.co/yutongzhu/smolvla-libero-ft20k/resolve/main/media/success.gif) | ![抓错物体](https://huggingface.co/yutongzhu/smolvla-libero-ft20k/resolve/main/media/fail_wrong_object.gif) | ![没抓住](https://huggingface.co/yutongzhu/smolvla-libero-ft20k/resolve/main/media/fail_grasp_miss.gif) | ![放置偏差](https://huggingface.co/yutongzhu/smolvla-libero-ft20k/resolve/main/media/fail_place_offset.gif) |

所有已标注失败回合的视频见 [yutongzhu/smolvla-libero-eval-results](https://huggingface.co/datasets/yutongzhu/smolvla-libero-eval-results)。

## LIBERO-Spatial 结果（每行 100 回合）

每次评测都固定种子 1000、10 个环境并行，同一任务的第 i 个回合在各次评测中初始场景相同，所以可以逐回合配对比较（精确 McNemar 检验）。

| 模型 | n_action_steps | 去噪步数 | 精度 | 成功率（95% 置信区间） |
|---|---|---|---|---|
| `HuggingFaceVLA/smolvla_libero`（官方） | 1 | 10 | FP32 | 69%（59–77） |
| 本模型 | 50（checkpoint 默认值） | 10 | FP32 | 68%（58–76） |
| 本模型 | 1 | 10 | FP32 | 75%（66–82） |
| 本模型 | **10** | 10 | FP32 | **78%（69–85）** |
| 本模型 | 10 | 5 | BF16 + `torch.compile` | **81%（72–87）** |

100 回合下，以上差异都没有达到 p<0.05。例如官方 vs 本模型（n=10）：19 个回合由败转胜，10 个由胜转败，p=0.14。

**推理延迟**（RTX 4090，单帧，p90）：FP32、去噪 10 步为 286 ms；BF16 + `torch.compile`、去噪 5 步为 37 ms，快 7.2 倍。推理的瓶颈在 CPU 下发 CUDA kernel，所以单用 BF16 几乎不加速。编译后的第一次推理约需 2 分钟，使用前要先预热。

## 使用方法

checkpoint 配置里的 `n_action_steps` 还是继承自 `smolvla_base` 的 50，这个设置下成功率会低 10 个百分点，所以**评测时要设成 `n_action_steps=10`**。checkpoint 要求的相机名是 `camera1`、`camera2`，需要把 LIBERO 的 `image`、`image2` 映射过去：

```bash
lerobot-eval \
  --policy.path=yutongzhu/smolvla-libero-ft20k \
  --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_spatial \
  --eval.n_episodes=10 --eval.batch_size=10 --seed=1000 \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
```

加上 `--policy.num_steps=5` 就是更快的部署配置。

## 训练配置

| | |
|---|---|
| 起点模型 | `lerobot/smolvla_base` |
| 数据 | `lerobot/libero` 全部四个子集（spatial、object、goal、libero_10） |
| 步数 / batch size | 20,000 / 32（约 2.3 个 epoch） |
| 硬件与耗时 | 1 张 RTX 4090，1 小时 47 分 |
| 最终 loss | 0.36 |

## 失败分析

逐个人工标注失败回合（[标注表和视频](https://huggingface.co/datasets/yutongzhu/smolvla-libero-eval-results)）：

| 模型 | 失败数 | 抓错了碗 | 没抓住 | 放置偏差 |
|---|---|---|---|---|
| 官方，n=1 | 31 | 23 | 8 | 0 |
| 本模型，n=1 | 25 | 14 | 5 | 6 |
| 本模型，n=10 | 22 | 10 | 2 | 10 |

和官方 checkpoint 相比，微调减少了抓错物体和没抓住，但带来了放置偏差。任务 5（“pick up the black bowl on the ramekin”）在每次评测中都是最弱的任务。

## 局限性

- 只在 LIBERO-Spatial 上评测，每个设置 100 回合。
- 评测时渲染分辨率是 360 px，训练数据是 256 px。
- SmolVLA 论文报告 LIBERO-Spatial 为 90%；在本项目的评测设置下，官方 checkpoint 为 69%，差距原因没有完全查明。
- 只在仿真中测试，没有上真机。
