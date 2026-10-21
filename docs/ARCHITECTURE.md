# wyu-bci-bench 架构

目标是把 EEG 相关代码从具体项目脚本里拆成可复用库，让机械手系统、实验脚本、notebook 和命令行都调用同一套实现。

## 分层

1. `data`
   - 读取已经切好的 epoch NPZ。
   - 兼容 `eeg/y/fs/channel_names` 等常见字段。
   - 合并多文件时统一标签、采样率、通道顺序和样本长度。

2. `continuous`
   - 读取 `raw_stream + trial_records_json` 连续采集文件。
   - 每个采集文件独立预处理，不跨停顿拼接滤波。
   - 支持坏道 QC、50 Hz 陷波、六阶 Butterworth 滤波、epoch 内基线校正和源文件边界保护。

3. `signal`
   - 提供 CAR、FFT 特征滤波、PSD、频带能量和通道选择。
   - 不依赖旧项目路径。

4. `features`
   - 提供 log-bandpower、固定空间滤波、FBCSP 风格 log-variance 和 Fisher 特征筛选。
   - 当前实现尽量避免复杂 native 线性代数，便于在 Windows 环境稳定运行。

5. `models`
   - 提供 diagonal LDA、nearest centroid、Gaussian NB、KNN 和 filter-bank LDA baseline。
   - 这些模型适合小样本快速判断数据是否有可分性。

6. `riemann`
   - 提供 EEG 协方差矩阵、Riemannian mean、MDM、Tangent Space + LDA 和 Filter-bank Tangent Space 模型。
   - 面向小样本 MI/ME 数据，作为 simple/FBCSP 与 EEGNet 之间的稳健机器学习路线。

7. `ssvep` / `erp`
   - `ssvep` 提供参考波 CCA、滤波器组 CCA 和训练折内 TRCA。
   - `erp` 提供已切 epoch 的 xDAWN 空间滤波 + 收缩 LDA；不负责从连续采集中检测事件。

8. `adaptation`
   - 提供无标签 Euclidean Alignment 和 CORAL。
   - 用评估域输入估计对齐统计量属于传导式测试时适配，必须在实验协议中显式声明。

9. `minirocket`
   - 提供 MiniRocket 风格固定零和卷积核、训练折内 bias 分位数、PPV/max 特征和 diagonal LDA。
   - 面向小样本 EEG 的少训练特征工程路线；不作为原始 sktime MiniRocket 的完全复现。

10. `treeboost`
   - 提供 XGBoost 在 EEG-aware 特征上的树模型路线。
   - 输入为 split-local 特征：log-bandpower、filter-bank spatial log-variance 和可选 Riemannian tangent-space；不直接把 raw EEG 喂给树模型。

11. `deep` / `deep_models` / `nets` / `ssl`
   - 提供可选 PyTorch EEGNet、Braindecode 常见网络，以及 M1/M2 和自监督表征工具。
   - 未安装 PyTorch 时，其他功能不受影响。

12. `capsule`
   - 提供轻量 EEG CapsuleNet voter，用于 `vote-capsule` 投票版本。
   - 该模型作为投票成员，不默认替代小样本机器学习主线。

13. `foundation` / `readout`
   - 提供 CBraMod 与 EEGPT 的已核验结构输入适配、微调控制和冻结特征线性读出。
   - EEGPT 要求调用方显式提供数据集幅值乘数；LaBraM 仍在合同审计前拒绝适配调用。

14. `evaluate` / `viz` / `cli`
   - 统一交叉验证、质量报告、画图和命令行入口。
   - 评估输出 JSON，包含混淆矩阵、balanced accuracy、macro F1 和每类召回率。
   - 支持 `vote-simple` hard voting、`vote-ml-best` 最佳经典机器学习 hard voting、`vote-ml-pruned` 去弱模型 hard voting、`vote-ml-pruned-capsule` 剪枝机器学习 + CapsuleNet hard voting、`vote-capsule` 机器学习 + CapsuleNet hard voting。

## 外部项目如何使用

普通 epoch 数据：

```python
from wyu_bci_bench.data import load_dataset
from wyu_bci_bench.evaluate import cross_validate_filterbank_lda

dataset = load_dataset(recordings_dir, label_mode="binary_mi", channel_mode="motor11")
metrics = cross_validate_filterbank_lda(
    dataset.eeg,
    dataset.y,
    dataset.fs,
    dataset.channel_names,
    dataset.label_names,
)
```

机械手连续 raw_stream 数据：

```python
from wyu_bci_bench.continuous import ContinuousPreprocessConfig, load_continuous_dataset
from wyu_bci_bench.evaluate import cross_validate_simple_models

dataset = load_continuous_dataset(
    recordings_dir,
    count_latest=4,
    config=ContinuousPreprocessConfig(bad_channel_mode="auto"),
)
metrics = cross_validate_simple_models(
    dataset.eeg,
    dataset.y,
    dataset.fs,
    dataset.channel_names,
    dataset.label_names,
)
```

如果机械手实时系统后续需要部署模型，可以新增 `deployment.py`，把训练、保存、加载和实时预测接口集中在本库中。
