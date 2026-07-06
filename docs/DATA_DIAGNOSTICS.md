# 本地 EEG 数据诊断记录

本页记录我们目前在机械手 EEG8 数据上踩过的坑、修正后的处理流程和可复现实验结论。原始数据不进入仓库，只保留统计结论、命令和输出路径格式。

## 新版连续数据流程

适用数据：`arm_bci_robot` 采集的 `raw_stream + trial_records_json` NPZ。

处理顺序：

1. 扫描最新 N 份 raw_stream 文件。
2. 每个文件单独读取采样率、通道名、连续信号和 trial 时间标签。
3. 先做通道 QC，得到自动/严格/手动坏道列表。
4. 对每个源文件单独预处理：线性去趋势、50 Hz 陷波、4-40 Hz 六阶 Butterworth 零相位滤波。
5. 按 `trial_records_json` 直接切 epoch，不允许 epoch 跨源文件边界。
6. 在每个 epoch 内做基线校正。
7. 合并所有有效 epoch 后做 K 折交叉验证。

重要修正：

- 不能把四份有停顿的数据先拼接成连续长数组再做 FFT 滤波。
- 不再使用“舌头想象/心算”等旧任务名；当前八分类是左手/右手/双手/双脚的想象与执行。
- 对 EEGNet 这类深度模型，不再把很少的 trial 当成足够数据；必须同时报告简单模型结果。

## 最新四份数据的参考结果

以下是此前在最新四份机械手八分类数据上得到的结果，供后续复现时对照。

全通道、正确按源文件切 epoch、EEGNet：

- epochs: `(139, 32, 875)`
- accuracy: `0.108`
- balanced_accuracy: `0.108`
- macro_f1: `0.107`

自动坏道筛选后，移除 `Fp1, F4, C3, CP5, PO3`，EEGNet：

- epochs: `(139, 27, 875)`
- accuracy: `0.144`
- balanced_accuracy: `0.150`
- macro_f1: `0.150`

严格坏道筛选后，移除 `Fp1, F4, C3, F8, CP5, PO3, PO4`，结果更差：

- balanced_accuracy: `0.086`

手动只移除 `Fp1,F4,PO3`：

- balanced_accuracy: `0.124`

当前 bench 默认 simple 模型组中最好的一组：

- feature/model: `mi_8_30_logvar + diag_gaussian_nb`
- accuracy: `0.1223`
- balanced_accuracy: `0.1219`
- macro_f1: `0.1092`

旧脚本中曾记录过 `mi_8_30_logvar + nearest_centroid` 的 balanced accuracy 约 `0.159`，但新版 bench 使用更明确的 CLI 默认参数和当前实现复跑后，以本节当前结果为准。

结论：当前效果不好不是单个模型实现问题，更像是数据质量、关键运动区通道质量、trial 数不足和八分类任务过难共同造成的。

## 复现命令

简单模型：

```powershell
wyu-bci-bench bench `
  E:\世界机器人大赛\projects\arm_bci_robot\data\recordings `
  --continuous-raw `
  --latest 4 `
  --bad-channel-mode auto `
  --model simple `
  --folds 5 `
  --out E:\世界机器人大赛\wyu-bci-bench\outputs\arm_latest4_simple
```

EEGNet：

```powershell
wyu-bci-bench bench `
  E:\世界机器人大赛\projects\arm_bci_robot\data\recordings `
  --continuous-raw `
  --latest 4 `
  --bad-channel-mode auto `
  --model eegnet `
  --folds 5 `
  --epochs 30 `
  --out E:\世界机器人大赛\wyu-bci-bench\outputs\arm_latest4_eegnet
```

分析图：

```powershell
wyu-bci-bench analyze `
  E:\世界机器人大赛\projects\arm_bci_robot\data\recordings `
  --continuous-raw `
  --latest 4 `
  --bad-channel-mode auto `
  --out E:\世界机器人大赛\wyu-bci-bench\outputs\arm_latest4_analysis
```

## 目前最可疑的问题

- 八分类任务对当前样本量太难；每类 trial 很少，5 折后每折训练样本更少。
- C3/C4 等运动区关键通道里存在被 QC 判为坏道或边缘坏道的情况，直接削弱左右手 MI 可分性。
- 平均 PSD 图会掩盖空间差异，需要看通道级 mu/beta ERD/ERS。
- 执行任务可能混入肌电和动作伪迹，模型可能学到非脑电线索；想象任务又可能因为被试执行不稳定而分离度低。
- ICA 可能有帮助，但不能盲目删除组件；当前优先级仍是坏道、阻抗、trial 数和任务设计。

## 下次采集建议

- 先做少类别验证，例如左手想象、右手想象、双脚想象或左右手二分类。
- 每类至少 60-100 个有效 trial，再考虑八分类。
- 采集前重点检查 `C3/Cz/C4` 及周边 `FC/CP` 通道接触质量。
- 采集时明确分开想象和执行，执行动作幅度要小，避免肩膀、身体和眼动。
- 每次采集后立即运行 `analyze`，先看坏道、PSD、频带热图和标签数量，再决定是否继续采集。
