# wyu-bci-bench Architecture

目标是把脑电相关代码从项目脚本里拆成可复用库，让机械臂系统、实验脚本和 notebook 都通过库函数调用。

## 分层

1. `data`
   - 只负责扫描和读取本机 NPZ。
   - 兼容 `eeg/y/fs/channel_names` 等常见字段。
   - 合并多文件时统一标签、采样率、通道顺序和样本长度。

2. `signal`
   - 提供 CAR、FFT 滤波、PSD、频带能量和通道选择。
   - 不依赖旧项目路径。

3. `features`
   - 提供滤波器组 log-variance、固定运动区空间滤波、Fisher 特征筛选。
   - 默认不使用 `np.linalg`，便于在当前 Windows 环境稳定运行。

4. `models`
   - 提供稳定版 filter-bank spatial LDA。
   - 分类器使用共享对角协方差，避免 native LDA 崩溃。

5. `deep`
   - 可选 PyTorch EEGNet。
   - 未安装 PyTorch 时，其他功能不受影响。

6. `evaluate` / `viz` / `cli`
   - 交叉验证、数据质量分析、绘图和命令行统一入口。

## 外部项目如何使用

其他项目不再复制 EEG 处理代码，只保留业务逻辑：

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

如果机械臂实时系统需要部署模型，后续可以在本库新增 `deployment.py`，把训练、保存、加载和实时预测接口也集中起来。
