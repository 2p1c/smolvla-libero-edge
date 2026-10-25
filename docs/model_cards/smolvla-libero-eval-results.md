---
license: apache-2.0
pretty_name: SmolVLA LIBERO-Spatial 闭环评测结果与失败标注
language:
- zh
task_categories:
- robotics
tags:
- smolvla
- libero
- evaluation
- failure-analysis
---

# SmolVLA 在 LIBERO-Spatial 上的评测结果与失败标注

来自 [github.com/2p1c/smolvla-libero-edge](https://github.com/2p1c/smolvla-libero-edge) 的闭环评测结果，比较了官方 [`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero)、[`yutongzhu/smolvla-libero-ft20k`](https://huggingface.co/yutongzhu/smolvla-libero-ft20k) 和 [`yutongzhu/smolvla-libero-spatial-only-10k`](https://huggingface.co/yutongzhu/smolvla-libero-spatial-only-10k)。

训练数据是公开的 [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero)，这里不重复上传。

## 目录结构

```
results/<run>/eval_info.json    每个回合的成功与否、奖励、种子（lerobot-eval 的输出）
results/<run>/meta.json         模型、n_action_steps、去噪步数、额外参数
results/<run>/runtime.json      精度和编译设置，以及每次推理的延迟（只有 BF16 那次评测有）
results/latency_ft20k.json      单帧推理延迟测试（RTX 4090）
failure_labels/*.csv            每个失败回合一行：task, episode, category, note, video
failure_labels/README.md        标注规范
videos/<run>/libero_spatial_<task>/eval_episode_<i>.mp4   只包含失败回合
```

所有评测都在 LIBERO-Spatial 上进行：每个任务 10 回合，种子 1000，10 个环境并行。同一任务的第 i 个回合在各次评测中初始场景相同，所以可以逐回合比较。

评测视频按 80 fps 编码，但每一帧是一个 20 Hz 控制步，按 0.25 倍速播放才是真实速度。

## 失败标注

| 抓错了碗 | 没抓住 | 放置偏差 |
|---|---|---|
| ![抓错物体](https://huggingface.co/yutongzhu/smolvla-libero-ft20k/resolve/main/media/fail_wrong_object.gif) | ![没抓住](https://huggingface.co/yutongzhu/smolvla-libero-ft20k/resolve/main/media/fail_grasp_miss.gif) | ![放置偏差](https://huggingface.co/yutongzhu/smolvla-libero-ft20k/resolve/main/media/fail_place_offset.gif) |
| `ft20k_n10`，任务 5，回合 0 | `ft20k_n10`，任务 8，回合 2 | `ft20k_n10`，任务 8，回合 0 |

动图按 2 倍速播放。每个失败回合只标最早出现的失败原因：

| 类别 | 判定标准 |
|---|---|
| `approach_wrong_dir` | 去抓了错误的物体，或明显朝错误的物体移动 |
| `grasp_miss` | 朝正确的物体移动，但夹爪闭合后没有抓住 |
| `place_offset` | 抓起了正确的物体，但放置失败或放偏 |

| 评测 | 失败数 | approach_wrong_dir | grasp_miss | place_offset |
|---|---|---|---|---|
| official_n1 | 31 | 23 | 8 | 0 |
| ft20k_n1 | 25 | 14 | 5 | 6 |
| ft20k_n10 | 22 | 10 | 2 | 10 |

标注由一人完成。
