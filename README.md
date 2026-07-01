# wyu-bci-bench

`wyu-bci-bench` 是我们内部使用的脑电离线分析和模型评测库。它把原来分散在实验脚本、notebook 和机械臂系统里的 EEG 数据处理、频谱分析、能量图、脑地形图、FBCSP 风格基线和 EEGNet 测试入口统一封装成 Python 包。

这个仓库只保存代码、文档、测试和示例，不上传原始采集数据、处理后数据、模型权重、实验输出或日志。

## 目录结构

```
wyu-bci-bench/
├── src/wyu_bci_bench/    # 核心库
│   ├── signal.py         # 信号处理
│   ├── viz.py            # 可视化
│   └── ...               # 其他模块
├── notebooks/            # 分析 notebook
├── examples/             # 使用示例
├── tests/                # 单元测试
├── outputs/              # 评测报告和图表 (本地)
└── docs/                 # 文档
```

## 快速开始

```bash
cd wyu-bci-bench
pip install -e .
python examples/basic_usage.py
```

## 环境

```bash
pip install numpy scipy scikit-learn matplotlib mne
```

---

**仓库**: https://github.com/wyubci/wyu-bci-bench
