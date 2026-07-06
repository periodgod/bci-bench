# wyu-bci-bench

`wyu-bci-bench` 是我们内部使用的 EEG/BCI 离线分析与模型评测库。它把原来分散在机械手项目、实验脚本和 notebook 里的数据读取、预处理、画图、交叉验证和模型入口统一到一个 Python 包里。

这个仓库只保存代码、文档、测试和示例，不上传原始采集数据、处理后数据、模型权重、实验输出或日志。

## 目前整合的流程

- 普通 epoch NPZ：读取 `eeg/y/fs/channel_names` 等常见字段，合并多文件后做离线分析和模型评测。
- 机械手连续 raw_stream：读取 `raw_stream + trial_records_json`，每个采集文件独立预处理，再按标签时间切 epoch，最后合并 trial。
- 标准预处理：线性去趋势、50 Hz 陷波、4-40 Hz 六阶 Butterworth 零相位滤波、优先使用 trial 内 `baseline_start_ms -> cue_onset_ms` 的 pre-cue baseline 校正。
- 质量控制：按文件检测非有限值、平线、异常标准差、鲁棒标准差和 0.1-99.9% 鲁棒峰峰值，支持 `off/auto/strict/manual` 坏道模式。
- 模型：FBCSP 风格 log-variance + diagonal LDA、简单 log-variance 模型组、Riemannian 协方差小样本模型组、MiniRocket 风格固定卷积核特征、XGBoost EEG 特征树模型、sklearn 多机器学习算法组、多模型 hard voting、可选 EEGNet/CapsuleNet。
- 评估：分层 K 折交叉验证，输出 accuracy、balanced accuracy、macro F1、混淆矩阵、每类召回率和 JSON 报告。
- MI/ME 标签模式：同一份 `4 部位 × 想象/执行` 连续采集数据可直接切成 8 类、想象/执行 2 类、身体部位 4 类、只看想象 4 类或只看执行 4 类。

## 安装

基础功能：

```powershell
cd E:\世界机器人大赛\wyu-bci-bench
pip install -e .
```

连续 raw_stream 默认预处理需要 SciPy：

```powershell
pip install -e ".[signal]"
```

EEGNet 需要 PyTorch：

```powershell
pip install -e ".[all]"
```

## 快速命令

分析最新四份机械手连续采集数据：

```powershell
wyu-bci-bench analyze `
  E:\世界机器人大赛\projects\arm_bci_robot\data\recordings `
  --continuous-raw `
  --latest 4 `
  --bad-channel-mode auto `
  --out E:\世界机器人大赛\wyu-bci-bench\outputs\arm_latest4_analysis
```

先跑简单模型，适合当前 trial 数很少的八分类数据：

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

跑 EEGNet：

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

跑 Riemannian 小样本 EEG 模型：

```powershell
wyu-bci-bench bench `
  E:\世界机器人大赛\projects\arm_bci_robot\data\recordings `
  --continuous-raw `
  --latest 4 `
  --bad-channel-mode auto `
  --model riemann `
  --folds 5 `
  --out E:\世界机器人大赛\wyu-bci-bench\outputs\arm_latest4_riemann
```

跑 MiniRocket 风格固定卷积核特征模型：

```powershell
wyu-bci-bench bench `
  E:\世界机器人大赛\projects\arm_bci_robot\data\recordings `
  --continuous-raw `
  --latest 4 `
  --bad-channel-mode auto `
  --model minirocket `
  --folds 5 `
  --out E:\世界机器人大赛\wyu-bci-bench\outputs\arm_latest4_minirocket
```

跑 XGBoost EEG 特征树模型：

```powershell
wyu-bci-bench bench `
  E:\世界机器人大赛\projects\arm_bci_robot\data\recordings `
  --continuous-raw `
  --latest 4 `
  --bad-channel-mode auto `
  --model xgb `
  --folds 5 `
  --out E:\世界机器人大赛\wyu-bci-bench\outputs\arm_latest4_xgb
```

跑 sklearn 多机器学习算法组：

```powershell
wyu-bci-bench bench `
  E:\世界机器人大赛\projects\arm_bci_robot\data\recordings `
  --continuous-raw `
  --latest 4 `
  --bad-channel-mode auto `
  --model ml-suite `
  --folds 5 `
  --out E:\世界机器人大赛\wyu-bci-bench\outputs\arm_latest4_ml_suite
```

跑多机器学习模型投票：

```powershell
wyu-bci-bench bench `
  E:\世界机器人大赛\projects\arm_bci_robot\data\recordings `
  --continuous-raw `
  --latest 4 `
  --bad-channel-mode auto `
  --model vote-simple `
  --folds 5 `
  --out E:\世界机器人大赛\wyu-bci-bench\outputs\arm_latest4_vote_simple
```

跑机器学习 + CapsuleNet 投票：

```powershell
wyu-bci-bench bench `
  E:\世界机器人大赛\projects\arm_bci_robot\data\recordings `
  --continuous-raw `
  --latest 4 `
  --bad-channel-mode auto `
  --model vote-capsule `
  --folds 5 `
  --epochs 20 `
  --out E:\世界机器人大赛\wyu-bci-bench\outputs\arm_latest4_vote_capsule
```

一次跑全部模型：

```powershell
wyu-bci-bench bench `
  E:\世界机器人大赛\projects\arm_bci_robot\data\recordings `
  --continuous-raw `
  --latest 4 `
  --bad-channel-mode auto `
  --model all `
  --out E:\世界机器人大赛\wyu-bci-bench\outputs\arm_latest4_all
```

MI/ME 最新数据常用拆分：

```powershell
# 8 类：左手/右手/双手/双脚 × 想象/执行
wyu-bci-bench bench `
  E:\世界机器人大赛\projects\arm_bci_robot\data\recordings\2026-07-04\11-12-40_mi_me.npz `
  --continuous-raw `
  --label-mode mi_me8 `
  --bad-channel-mode manual `
  --bad-channels F4 `
  --reject-z 999999 `
  --model simple `
  --out E:\世界机器人大赛\wyu-bci-bench\outputs\mi_me_111240_8class

# 2 类：想象 vs 执行
wyu-bci-bench bench `
  E:\世界机器人大赛\projects\arm_bci_robot\data\recordings\2026-07-04\11-12-40_mi_me.npz `
  --continuous-raw `
  --label-mode mi_me_state `
  --bad-channel-mode manual `
  --bad-channels F4 `
  --reject-z 999999 `
  --model simple `
  --out E:\世界机器人大赛\wyu-bci-bench\outputs\mi_me_111240_state

# 只看想象 4 类：左手/右手/双手/双脚
wyu-bci-bench bench `
  E:\世界机器人大赛\projects\arm_bci_robot\data\recordings\2026-07-04\11-12-40_mi_me.npz `
  --continuous-raw `
  --label-mode mi_me_imagery_body4 `
  --bad-channel-mode manual `
  --bad-channels F4 `
  --reject-z 999999 `
  --model simple `
  --out E:\世界机器人大赛\wyu-bci-bench\outputs\mi_me_111240_imagery_body4

# 只看执行 4 类：左手/右手/双手/双脚
wyu-bci-bench bench `
  E:\世界机器人大赛\projects\arm_bci_robot\data\recordings\2026-07-04\11-12-40_mi_me.npz `
  --continuous-raw `
  --label-mode mi_me_execution_body4 `
  --bad-channel-mode manual `
  --bad-channels F4 `
  --reject-z 999999 `
  --model simple `
  --out E:\世界机器人大赛\wyu-bci-bench\outputs\mi_me_111240_execution_body4
```

`--reject-z 999999` 用于复现实验时保留全部可切出的 trial；日常 QC 可以去掉它，使用默认 `8.0` 自动剔除峰值异常 epoch。

## 关键注意事项

- 有停顿的多份连续采集文件不能先拼成一条长信号再 FFT 滤波；本库按源文件独立滤波和切 epoch。
- trial 的 epoch 不能跨源文件边界，越界 trial 会被拒绝。
- 连续采集默认按源文件末端做事件时间线和 `raw_stream` 对齐，避免 GUI 计时与实际样本数差几个点时误丢最后一个 trial；特殊合成数据可用 `--no-align-timeline` 关闭。
- 基线校正优先使用同一 trial 的提示前基线；缺少这些时间戳时才回退到 epoch 开头。不要对整段连续数据做全局基线。
- ICA 不做盲删；当前优先做坏道筛选和数据质量报告。
- 当前八分类任务 trial 数很少，简单 log-variance 模型可能比 EEGNet 更稳。
- 机械臂项目里不要继续新建一次性的 `compare_xxx.py` 来复制读数据和评估逻辑；通用分析先收进本库，再由项目脚本或 notebook 调用本库。

## 目录

```text
wyu-bci-bench/
├── src/wyu_bci_bench/    # 核心库
├── tests/                # smoke tests
├── docs/                 # 诊断和架构文档
├── examples/             # 示例
├── notebooks/            # 离线 notebook
└── outputs/              # 本地输出，不入库
```
## vote-ml-best

`--model vote-ml-best` runs a hard-voting ensemble over the strongest current
classical EEG machine-learning candidates: FBCSP-LDA, ExtraTrees,
GaussianNB tangent-space, RandomForest, RBF-SVM, and XGBoost. Use it when the
task is small-sample MI/ME 8-class classification and the goal is to combine
the best non-deep-learning models already available in the bench.

`--model vote-ml-pruned` removes the weak XGBoost voter from that set.
`--model vote-ml-pruned-capsule` uses the pruned classical ML voters and adds
CapsuleNet as one extra voter. Validate this mode with repeated seeds before
treating it as a stable improvement.
