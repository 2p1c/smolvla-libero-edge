# 失败回合标注规范

每个失败回合只标一个类别，按失败最早发生的阶段判定。

| 类别 | 判定标准 |
|---|---|
| `approach_wrong_dir` | 机械臂去抓了错误的物体，或明显朝错误的物体移动 |
| `grasp_miss` | 朝正确的物体移动，但夹爪闭合后没有抓住 |
| `place_offset` | 抓起了正确的物体，但放置失败或放偏 |
| `other` | 以上都不符合，在 `note` 列写明原因 |

边界情况：抓起正确的碗、放到错误位置 → `place_offset`；一开始就抓了另一个碗 → `approach_wrong_dir`。

## 结果

| 模型 | 失败数 | approach_wrong_dir | grasp_miss | place_offset |
|---|---|---|---|---|
| `HuggingFaceVLA/smolvla_libero`，n=1（`failure_labels_official_n1.csv`） | 31/100 | 23（74%） | 8（26%） | 0 |
| 本项目微调 20k，n=1（`failure_labels_ft20k_n1.csv`） | 25/100 | 14（56%） | 5（20%） | 6（24%） |
| 本项目微调 20k，n=10（`failure_labels_ft20k_n10.csv`） | 22/100 | 10（45%） | 2（9%） | 10（45%） |

同为 n=1 时，微调模型已经出现 6 次 `place_offset`，官方模型为 0，所以放置偏差主要来自微调本身；n=10 又多了 4 次。
