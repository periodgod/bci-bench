# wyu-bci-bench

`wyu-bci-bench` 是我们内部使用的 EEG/BCI 离线分析与模型评测库。它把原来分散在机械手项目、实验脚本和 notebook 里的数据读取、预处理、画图、交叉验证和模型入口统一到一个 Python 包里。

这个仓库只保存代码、文档、测试和示例，不上传原始采集数据、处理后数据、模型权重、实验输出或日志。

## 目前整合的流程

- 普通 epoch NPZ：读取 `eeg/y/fs/channel_names` 等常见字段，合并多文件后做离线分析和模型评测。
- 机械手连续 raw_stream：读取 `raw_stream + trial_records_json`，每个采集文件独立预处理，再按标签时间切 epoch，最后合并 trial。
- 标准预处理：线性去趋势、50 Hz 陷波、4-40 Hz 六阶 Butterworth 零相位滤波、优先使用 trial 内 `baseline_start_ms -> cue_onset_ms` 的 pre-cue baseline 校正。
- 质量控制：按文件检测非有限值、平线、异常标准差、鲁棒标准差和 0.1-99.9% 鲁棒峰峰值，支持 `off/auto/strict/manual` 坏道模式。
- 模型：线性/概率与树模型、滤波器组 CSP+SVM/LDA、黎曼几何基线、MiniRocket、XGBoost、sklearn 多算法组、hard voting、EEGNet/CapsuleNet、M1/M2；另含 SSVEP CCA/FBCCA/TRCA/eTRCA/TDCA、ERP xDAWN+LDA、EA/CORAL/黎曼对齐/MMD/DANN/AdaBN、可选 Braindecode 网络与本地 EEG-ResNet/InceptionTime。
- 在线评估：因果 SOS 滤波、连续流窗口缓存、当前主机推理延迟/吞吐基准和时间漂移分箱摘要；不包含设备接入或自动域适配。
- 评估：分层 K 折交叉验证，输出 accuracy、balanced accuracy、macro F1、混淆矩阵、每类召回率和 JSON 报告。
- 诊断：分组置换重要性、通道/频带留出消融、验证折温度缩放、ECE/Brier 与风险-覆盖率曲线；这些是解释/概率评估工具，不是性能提升声明。
- MI/ME 标签模式：同一份 `4 部位 × 想象/执行` 连续采集数据可直接切成 8 类、想象/执行 2 类、身体部位 4 类、只看想象 4 类或只看执行 4 类。
- EEG 基础模型适配：输入合同核验、冻结线性读出、指定模块解冻，以及 PEFT LoRA / DoRA / PiSSA / AdaLoRA；方法来源和未支持项见 [`docs/foundation_finetuning_methods.md`](docs/foundation_finetuning_methods.md)。

BIOT 冻结特征需安装 `.[biot]`，并向 `wyu_bci_bench.foundation.biot.extract_features` 提供官方 BIOT 源码目录、匹配的 encoder checkpoint、严格有序的官方 16/18 双极导联名、200 Hz EEG 和默认 10 秒窗合同。适配器不会猜测或补造通道映射。

EEGPT 冻结特征需安装 `.[eegpt]`，并调用 `wyu_bci_bench.foundation.eegpt.extract_features`，提供作者源码目录、含 `target_encoder.*` 的匹配 safetensors 文件、58 路官方通道名称和 4 秒 EEG。`amplitude_scale` 必须由数据集预处理方案显式指定；作者预训练数据混合了不同来源，bench 不设统一默认尺度。编码器严格加载与 512-D 冻结特征前向已核验；PROTEUS 通道/尺度适配和下游效果未验证。

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

ShallowFBCSPNet、Deep4Net、EEGConformer、FBCNet、ATCNet、TCN、EEGTCNet、EEGInceptionMI、EEGNeX 和 EEGITNet 需要可选的 Braindecode 依赖：

```powershell
pip install -e ".[braindecode]"
```

真正的传统滤波器组 CSP+SVM 使用 MNE、SciPy 与 scikit-learn：

```powershell
pip install -e ".[csp]"
```

现有模型按架构家族、表征训练目标、读出器和评估协议分层整理，包含可运行状态与当前缺口，见 [`docs/model_catalog.md`](docs/model_catalog.md)。每种已纳入且可运行的算法/机制均有独立 Remotion composition；未通过输入合同审计的基础模型候选不标为已支持。索引、bench 符号和证据边界见 `E:/世界机器人大赛/projects/脑电大模型/animations/src/methods.json`。

范式专用 SSVEP/ERP 解码器、无监督跨域对齐和在线滤波使用 `scipy`，xDAWN+LDA 还需要 `scikit-learn`；黎曼对齐额外使用 `pyriemann`，见 `.[ssvep]`、`.[domain-adaptation]` 与 `.[signal]` 可选依赖组。EA/CORAL/黎曼对齐/AdaBN 若使用评估目标域的无标签输入估计统计量，必须按传导式测试时适配协议单独报告。新增架构、CSP、域适配和自监督方法尚无目标公开数据集效果结果，详见 [`docs/model_catalog.md`](docs/model_catalog.md)。

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

## 跨被试 / 跨会话 / 基础模型协议层（v0.2）

毕业论文（PROTEUS 三指令）、EEG/EMG Challenge 2026 Track 2、OmniEEG Challenge 2026 共用 `grouped / splits / metrics / priors / nets / ssl / foundation / readout / registry / submission` 模块。说明、旧函数对照和核验结果见 [docs/EEG_FOUNDATION_PROTOCOLS.md](docs/EEG_FOUNDATION_PROTOCOLS.md)；新实验从 [examples/proteus_loso_template.py](examples/proteus_loso_template.py) 复制。

表征学习方法对应的已发表论文、作者代码、bench 实现状态与复现边界见 [docs/representation_learning_papers.md](docs/representation_learning_papers.md)。

```powershell
pip install -e ".[ml]"
python scripts/verify_proteus_port.py --workspace D:\脑电大模型
wyu-bci-bench oof-score D:\脑电大模型\challenge2026_track02\proteus_update1\results\representation_learning_20261019\ssl\oof.npz
```

## 能力标签映射

`explorer/capabilities.json` 将关键方法映射到具体模块和接口，并分别标记流程阶段、方法机制、处理目标和声明等级。例如，跨被试 SupCon 的“旨在缓解域偏移”是设计意图；跨会话划分只是评估域偏移下的泛化，不代表它本身会消除域偏移。`explorer/build_catalog.py` 会校验标签、接口引用和声明等级，并把映射加入 catalog；Explorer 的可视化筛选尚待实现。

## 目录

```text
wyu-bci-bench/
├── src/wyu_bci_bench/    # 核心库
├── tests/                # smoke tests + grouped protocol tests
├── docs/                 # 诊断、架构和协议层文档
├── examples/             # 示例与新实验模板
├── scripts/              # 核验脚本
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
