# SmolVLA 在 LIBERO 上的微调与端侧部署分析

在 MuJoCo 仿真（LIBERO）里完成 VLA 的闭环：微调 → 闭环评测 → 部署约束分析 → 推理加速。
基于 [LeRobot](https://github.com/huggingface/lerobot) 0.6.1 和 [SmolVLA](https://huggingface.co/lerobot/smolvla_base)（约 4.5 亿参数，SmolVLM2 主干加 flow-matching 动作专家）。

> 所有方括号 `[ ]` 都等实测后填写。没跑出来的数不写。

## 结果

### 1. 微调后闭环成功率（LIBERO-Spatial，10 个任务 × [10] 回合）

| 模型 | 训练步数 | n_action_steps | 成功率 |
|---|---|---|---|
| 本项目微调 | 5k | 50 | [ ]% |
| 本项目微调 | 10k | 50 | [ ]% |
| 本项目微调 | 20k | 50 | [ ]% |
| 本项目微调 | 20k | [最佳 n] | [ ]% |
| `HuggingFaceVLA/smolvla_libero`（官方参照） | — | 1 | [ ]% |

### 2. 动作块执行步数消融（核心实验）

每次推理出一个 50 步的动作块，执行前 n 步后重新推理。n 越大，对推理速度要求越低（需要推理频率 ≥ 20Hz / n），但开环执行越久，误差越容易累积。

| n_action_steps | 1 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|
| 所需推理频率 | 20 Hz | 4 Hz | 2 Hz | 0.8 Hz | 0.4 Hz |
| 成功率 | [ ]% | [ ]% | [ ]% | [ ]% | [ ]% |

![sweep](results/sweep_official.png)

结论：[例如：n ≤ [ ] 时成功率保持在 [ ]% 以上；n = 50 时降到 [ ]%。]

### 3. 推理延迟（[GPU 型号]，单帧，预热 20 次、测 100 次）

| 配置 | 去噪步数 | p50 ms | p90 ms | 推理 Hz | 20Hz 下最小 n | 显存峰值 GB | 闭环成功率 |
|---|---|---|---|---|---|---|---|
| FP32 | 10 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ]% |
| BF16 | 10 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ]% |
| BF16 | 5 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ]% |
| BF16 + compile | 10 | [ ] | [ ] | [ ] | [ ] | [ ] | — |
| BF16 + compile | 5 | [ ] | [ ] | [ ] | [ ] | [ ] | — |

“20Hz 下最小 n”= ⌈p90 / 50ms⌉：异步执行时，一个动作块至少要撑过一次推理的时间。把它和第 2 节的曲线对照，就能得出这块设备上能达到的成功率。

### 4. 成功和失败案例

| 成功 | 失败 |
|---|---|
| ![](media/success.gif) | ![](media/failure.gif) |

失败原因：[例如：抓取时夹爪闭合过早 / 放置位置偏移]

## 复现

需要 Linux + NVIDIA GPU（24GB 及以上）。LIBERO 只支持 Linux。

```bash
bash scripts/setup.sh
```

先用官方 checkpoint 测出上限，确认闭环链路通了：

```bash
EPISODES=2 scripts/eval.sh HuggingFaceVLA/smolvla_libero official_smoke
```

在 tmux 里开始微调。先看第一分钟的 it/s，再按可用时间重新设 `STEPS`：

```bash
STEPS=20000 BATCH=32 scripts/train.sh
```

训练期间做消融实验（默认用官方 checkpoint，每个任务 5 回合）：

```bash
scripts/sweep_n_action_steps.sh
```

测推理延迟：

```bash
python scripts/bench_latency.py --policy HuggingFaceVLA/smolvla_libero
```

评测所有微调 checkpoint：

```bash
scripts/eval_ckpts.sh
```

评测去噪步数减半后的闭环成功率：

```bash
NUM_STEPS=5 scripts/eval.sh outputs/smolvla_libero_ft/checkpoints/last/pretrained_model ft_last_steps5
```

汇总所有结果成表：

```bash
python scripts/collect_results.py
```

评测视频在 `results/<tag>/videos/`。

## 训练和部署必须一致的地方

- **相机名**：`smolvla_base` 预训练时用 `camera1..3`，LIBERO 叫 `image`、`image2`。训练和评测都要传同一个 `rename_map`（`scripts/common.sh`），`eval.sh` 会按 checkpoint 自动加。
- **归一化统计量**：state 和 action 用 MEAN_STD，统计量随 checkpoint 一起保存在 `policy_preprocessor*` 里。换数据集就要重算。
- **控制模式**：LIBERO 默认 `relative`（末端增量），要和训练数据的动作定义一致。
- **无头渲染**：`MUJOCO_GL=egl`。

## 目录

```
scripts/common.sh               共享路径、数据集、rename_map
scripts/setup.sh                安装依赖，预下载数据集和 checkpoint
scripts/train.sh                从 smolvla_base 微调
scripts/eval.sh                 单次闭环评测 → results/<tag>/
scripts/eval_ckpts.sh           评测一个训练任务的所有 checkpoint
scripts/sweep_n_action_steps.sh n_action_steps 消融并画图
scripts/bench_latency.py        推理延迟和显存 → results/latency.json
scripts/collect_results.py      汇总 results/ 成 markdown 表
```
