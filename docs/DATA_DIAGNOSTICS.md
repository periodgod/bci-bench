# Local EEG Data Diagnostics

本页记录当前本机采集数据的离线诊断结果。数据文件不进入仓库，下面只保留统计结论和复现实验命令。

## 数据概况

扫描目录：

```powershell
wyu-bci-bench scan E:\世界机器人大赛\projects\arm_bci_robot\data\recordings
```

当前扫描到 6 个 NPZ，共 81 个 trial：

- 方向/机械臂任务：`forward/backward/left/right/up/down`，共 35 个 trial。
- 左右手 MI：`left_hand_mi/right_hand_mi`，共 46 个 trial。
- MI 子集类别平衡：左手 23，右手 23。

注意：方向任务和左右手 MI 是不同范式，不能直接混在一个二分类 MI 训练集里。

## 模型结果

只使用后两段左右手 MI 文件：

```powershell
wyu-bci-bench bench `
  E:\世界机器人大赛\projects\arm_bci_robot\data\recordings\2026-06-25\21-46-57_session.npz `
  E:\世界机器人大赛\projects\arm_bci_robot\data\recordings\2026-06-25\22-06-49_session.npz `
  --label-mode binary_mi `
  --channel-mode all `
  --model fbcsp-lda `
  --folds 5 `
  --out E:\世界机器人大赛\wyu-bci-bench\outputs\binary_mi_bench_all_channels
```

结果：

- 32 通道：accuracy = 0.5000，balanced_accuracy = 0.5000。
- 混淆矩阵：`[[11, 12], [11, 12]]`。
- 每类召回：左手 0.4783，右手 0.5217。

三中心通道 `C3/Cz/C4`：

- balanced_accuracy = 0.4565。
- 混淆矩阵：`[[10, 13], [12, 11]]`。

结论：不是因为只选了中心通道导致 50%；全通道也没有显著区分度。

## 频带能量

MI 子集 32 通道均值：

| class | delta | theta | mu | low_beta | high_beta | gamma_low |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| left_hand_mi | 6.7676 | 3.4874 | 1.9646 | 2.0955 | 1.3954 | 1.6514 |
| right_hand_mi | 5.4018 | 3.7405 | 1.9999 | 2.0770 | 1.3739 | 1.6316 |

关键点：

- mu 频带差异很小：左手 1.9646，右手 1.9999。
- low_beta/high_beta 差异也很小。
- delta 差异更明显，但 delta 更容易受到漂移、眼动、体动和基线变化影响，不应作为左右手 MI 的主要判据。

这说明当前左右手 MI 数据里，典型运动想象应当出现的 C3/C4 附近 mu/beta ERD/ERS 分离不明显。

## 信号质量

MI 子集 32 通道统计：

- shape = `[46, 32, 1000]`
- fs = 250 Hz
- amplitude mean = 573.55
- amplitude std = 4366.17
- abs p95 = 9442.30
- abs p99 = 11306.01
- channel variance min = 4540.31
- channel variance median = 108499.27
- channel variance max = 1399358.62

通道方差跨度很大，说明可能存在坏道、接触不稳定、量纲/缩放差异、运动伪迹或个别通道污染。

## 为什么准确率接近 50%

当前证据更支持数据问题，而不是模型实现单点问题：

- MI trial 数只有 46 个，单类 23 个，交叉验证每折样本很少。
- 左右手在 mu/beta 频段几乎没有平均能量分离。
- 低频 delta 差异相对更大，可能代表漂移或伪迹，而不是运动想象。
- 全通道和中心通道基线都接近随机。
- 数据目录里混有方向任务和 MI 任务，训练时必须明确使用 `--label-mode binary_mi`。

## 建议

- 重新采集更多左右手 MI，至少每类 60-100 个 trial。
- 保证左右手 cue、想象时间窗、休息时间窗和标签写入完全同步。
- 采集时重点检查 C3/Cz/C4 及相邻运动区通道阻抗和接触。
- 每次采集后立即运行 `wyu-bci-bench analyze` 检查 PSD、频带能量和通道热图。
- 训练前剔除明显坏道和异常 trial。
- 不要把机械臂方向任务与左右手 MI 混入同一个二分类 MI 数据集。
