# 跨被试 / 跨会话 / 基础模型协议层（v0.2.0）

v0.1 面向机械手单人 K 折；v0.2 增加按参与者分组的协议层，供毕业论文（PROTEUS）、EEG/EMG Challenge 2026 Track 2 和 OmniEEG Challenge 2026 共用。所有新模块已用 `scripts/verify_proteus_port.py` 对照历史 OOF 与原始代码核验（`--with-bids` 共 29/29 通过：论文与 10-19 复现报告的 cell BA、bootstrap 区间、sign-flip p 值逐位一致；数学先验特征、Welch 特征变体、SupCon、M1/M2 的种子初始化与前向输出与原实现完全相同；从原始 EDF 重新提取 sub-P01 与缓存逐位相同）。

## 模块

2026-11-14 跨数据5折扩展：`splits.subject_kfold` 对参与者做固定种子近等量K折，每人恰好测试一次，验证集只从当前外层训练人中抽取；这是外层K折+单次内层留出，不冒称完整嵌套K折。`eegmat.load_eegmat_windows` 根据源README的 `_1`静息/`_2`心算标签构建19通道、每条件前60秒的无重叠4秒窗口，排除ECG和耳参考差分；两条件使用同一个参与者×session键，防止单类session导致组BA失真。`supervised` 可显式启用CUDA bfloat16 autocast，保留FP32参数/优化器，预测在FP32下计算softmax；默认仍为原FP32。EEGMAT存在任务与记录顺序混杂，开发对比须披露。

2026-11-14 BrainOmni：`foundation.brainomni` 严格加载官方完整 Tiny 权重，复用官方冻结 RVQ、`encode()` 的前11层和时间均值/展平/MLP读出；提供 `frozen`、`last_block`、`full` 与注意力 `qkv/proj` LoRA。`prepare_input` 明确微伏输入的分母为 `std(uV)+10`，等价于官方伏单位的 `std(V)+1e-5`，不能直接在微伏上加 `1e-5`。`dreyer.load_acquisition_windows` 从 EDF 内嵌 769/770 事件提取左右手提示后4秒窗口，显式排除被文件标成 EEG 的 EOG/EMG。适配代码安装入口为 `.[brainomni]`；源码与 MIT 许可位于 `_third_party/brainomni/`，`UPSTREAM.json` 固定提交及每个文件的哈希。Windows 单进程使用 torch.distributed；官方注意力评估时仍有 dropout，移植版仅在 eval 将 dropout 设0。真实 Tiny 缓存分词与直接编码最大差0；梯度检查验证每个声明可训练张量连接任务损失。性能须另看具体实验协议。

2026-11-09 追加审计：`ensemble_fit.fit_simplex_weights` 支持内层 OOF 单纯形权重；`foundation.eegpt` 支持已核验名称子集，官方 19 通道示例与 41 通道真实权重前向均通过；`foundation.labram` 迁移历史 wrapper 并按官方 engine 修正 µV÷100，编码推理不需要 tokenizer；`foundation.token_adaptation` 提供位置保留池化和末层适配，预训练/随机 EEGPT、LaBraM 前缀回放与完整前向最大差均为 0。`ssl.MaskedEEGAutoencoder.encode_tokens` 与 `mae_downstream` 共用同一编码器，掩码保留电极/时间位置，支持折内自监督三臂；目标任务效果待实验，接口验证不是性能证据。

2026-11-09 Track 2 新实验扩展：`phase_slow.PhaseSlowNet` 提供整窗/两阶段/慢波受控消融；`supervised` 提供固定预算、组×类均衡损失与多输入分类训练；`splits.forward_blocks` 构造外层训练行内的向前验证；`foundation.cbramod.extract_tokens/pool_tokens/make_last_block_classifier` 保留时空 token 并支持固定前缀缓存后的末层训练；`csp.FilterBankCSP.predict_proba` 调用原生 LDA 概率；`spatial_csd` 封装 MNE 球样条 CSD；`causal_robust` 提供可重置且只看过去的裁剪/指数矩估计；`ensemble_diagnostics` 与 `proteus_bids.window_index` 提供分阶段误差和事件索引审计。新接口单测单独验证，29 项历史迁移核验并不等价于对所有新算法的历史等价验证。

旧→新映射补充：round4 `run_deep.train_net` 的固定预算无增强训练由 `supervised.train_classifier` 承担（明确训练 RNG 在模型初始化之后重置，因此新实验以 D0 重训作为配对控制）；`run_slow.slow_feats` 复用 `slow_features.slow_dc_features`；CBraMod 新读出使用同一官方 encoder，并以单测核验缓存前缀+末层与完整输出相等。

| 模块 | 内容 |
|---|---|
| `eegmat` | EEGMAT成对静息/心算EDF加载与19个头皮通道、等时长两条件、固定窗口合同 |
| `foundation.brainomni` | 官方 BrainOmni 输入/位置归一化、严格加载、冻结分词缓存、有效主干微调与任务梯度审计；评估 dropout 修正在源文件清单中记录 |
| `dreyer` | Dreyer2023 acquisition EDF 事件读取，27个头皮 EEG 通道、769左手/770右手、固定滤波与提示后4秒窗口 |
| `grouped` | `GroupedEEGDataset`（X/y/subjects/sessions/interfaces）、`load_grouped_npz`、`load_proteus_windows`（(N,41,480) µV @120 Hz 校验） |
| `pretraining` | `load_pretraining_corpus`：分片 `.npy` 内存映射、无标签分组索引、按会话与记录序排序；`resolve_channel_coordinates`：MNE 3D 坐标与未知通道掩码 |
| `pretraining_masks` | `sessionfm_spatiotemporal_mask`：基于米制电极坐标生成 3 cm 空间簇、1 s 时间段掩码及整通道丢弃样本 |
| `external_datasets` | `load_lee2019_mi_windows`、`load_bnci2015_004_windows`：从完成事件/窗口审计的缓存构建 `GroupedEEGDataset`；Lee 二类及 BNCI 三类映射必须由调用方显式提供，BNCI 通道名从缓存读取 |
| `external_event_data` | `read_edf_with_events`、`extract_annotated_windows`、`extract_windows_from_raw`、`extract_sliding_windows`：保留事件注释加载外部 EDF，按事件标签或连续时间切窗，统一重采样、转换为微伏并可逐样本 CAR |
| `fewshot` | `balanced_kshot_indices`、`GaussianDecoderPrior`、`estimate_decoder_prior`、`isotropic_decoder_prior`、`fit_map_logistic`、`semi_supervised_em_logistic`：分层少样本抽样、源解码器低秩/各向同性 Gaussian 先验、MAP Logistic 与半监督 soft-label EM；k=0 严格退化为先验均值 |
| `proteus_bids` | `extract_windows`（NEMAR nm000290 BIDS → 窗口，逐位等于历史缓存）、`save_cache` |
| `window_features` | `extract_variant`（Welch 频带功率/相对功率/Hjorth，`whole`、`pre_post_car` 等变体） |
| `splits` | `loso`、`inner_loso`、`cross_session`、`neuralbench_sealed`、`subject_ratio`（OmniEEG 8:1:1）、`multi_seed_subject_ratio`；全部返回布尔掩码 `Split` 并可 `check_disjoint` |
| `metrics` | `cell_balanced_accuracy`（PROTEUS 主指标）、`stream_mean_balanced_accuracy`（NeuralBench `bal_acc_stream_mean`）、`summarize_predictions`、`paired_subject_bootstrap`、`exact_sign_flip_pvalue`、`bootstrap_mean_ci` |
| `protocol_diagnostics` | `collapse_summary` / `collapse_gate`（逐 cell 预测类占比、被试×类低召回与固定阈值判定）、`guard_comparison`（指定被试的配对守护比较） |
| `priors` | `MathPriorFeatures` / `math_prior_features`：1394 维 log/相对频带功率 + Hjorth（默认 CPU，保证复现 0.5113） |
| `nets` | `MultiScaleTemporalEnergy`、`BandCovarianceProjection`、`MathPriorMultiScaleCovNet`（M1）、`PowerFirstMathPriorNet`（M2）、`make_model`、`train_model`、`predict_proba`、`sampling_weights` |
| `deep_models` | 可选 Braindecode 网络以及本地 EEG-ResNet、InceptionTime 工厂与折内训练；验证集按试次划分，测试窗口概率按试次平均 |
| `frontier_models` / `frontier_advanced` | 前沿 EEG 候选结构与损失。前者含 STMambaNet、EEGSimpleConv、EEG-MFTNet、2025 EEGMamba foundation model、EEGM2；后者含 EEGMamba2024（arXiv:2407.20254）、DCT 系列 GADCN、Phase-SPDNet 及其余候选。部分 Mamba-2 encoder/decoder 需要 Linux/CUDA 可选依赖；可运行结构不代表论文完整训练复现。论文核验和复现边界见模型目录及审计文档 |
| `cocot` | CoCoT-EEG 多尺度卷积 → L2 patch → Fourier 时间位置编码 → Transformer；`make_cocot_model` 构建模型，`fit_predict_cocot` 在训练折内监督微调，支持严格 encoder state 与可选 Mixup |
| `csp` | `FilterBankCSP`：MNE CSP 在每个预设频带训练空间滤波器，拼接 log-average-power 后接标准化线性 SVM 或收缩 LDA；滤波器与分类器均只拟合训练折 |
| `ssvep` | 刺激参考 CCA、滤波器组 CCA（FBCCA 风格）、TRCA/eTRCA 与 MOABB TDCA；真实刺激频率按类别显式传入 |
| `erp` | 已分段 epoch 的 xDAWN 空间滤波 + 收缩 LDA |
| `event_waveform` | `low_frequency_waveforms`（CAR、低通、基线、降采样；反馈前缀先截断）、`XDawnTangentLogistic`（按类 xDAWN + OAS 协方差切空间 + Logistic）、`ClassTemplateProjection`（训练类模板的低秩投影相关特征，可取绝对相关） |
| `adaptation` | Euclidean/Riemannian Alignment、CORAL、MMD、DANN 域头与 AdaBN；使用评估目标域输入拟合时按传导式测试时适配报告，MMD/DANN 需调用方提供编码器与训练循环 |
| `ssl` | `RepresentationModel`、`TS2VecModel`、`MaskedEEGAutoencoder`、SimCLR/VICReg/TS2Vec/掩码/预测/跨通道损失、`cross_subject_supcon`、`train_ssl`、`train_ts2vec`、`train_masked_autoencoder`、`encode` |
| `online` | `CausalSOSFilter`、`StreamWindowBuffer`、`benchmark_streaming`、`temporal_drift_summary`：因果滤波、连续窗切分、当前主机延迟/吞吐和时间分箱摘要 |
| `session_causal` | `causal_center`、`causal_class_balance`、`causal_fixed_point_prior_match`、`SessionCausalState`、`FixedPointPriorMatchState`、`stream_keys`：每个 subject×session 流内只用此前无标签窗口做因果特征中心化、类别均衡或固定点先验匹配，会话开始时重置；属于因果测试时适配，须与无状态模型分开报告 |
| `innovation` | 因果特征适配：`causal_scale`、`causal_rank_normalize`、`exponential_center`；频谱/协方差：`periodic_spectral_features`、`fit_wccn`、`fit_nuisance_subspace` / `remove_nuisance_subspace`；原型适配：`source_prototypes`、`sinkhorn_assignments`、`causal_prototype_prob`、`causal_gaussian_prototype_prob`；稳健训练辅助：`cell_class_sample_weights`、`GroupDRO`；`full_session_center` 仅用于含未来窗口的上限诊断 |
| `sict` | `SICTAdapter` 两层因果交叉注意力残差头（64-D、4 头、128 窗记忆、相对年龄桶）；`SICTStream` 每会话隔离状态并支持 reset，含均匀注意力、记忆置空和固定种子顺序消融 |
| `foundation` | 输入合同注册表 `CONTRACTS`；`require_verified` 拒绝未审计合同；`foundation.cbramod.extract_features`（4 s / 200 Hz / µV÷100）；`foundation.biot.extract_features`（官方有序 16/18 导联、200 Hz、10 s、官方幅值归一化、256-D 冻结编码器）；`foundation.eegpt.extract_features`（官方 58 通道、4 s / 256 Hz，显式 `amplitude_scale`，严格加载 encoder，512-D 冻结特征）；`foundation.finetuning` 的全量/冻结/指定层策略与 PEFT LoRA 系适配 |
| `foundation.pa_fm` | `PAFMSCC`、`train_pafm`：M2 上增加基础模型残差头，支持折内嵌入或编码器输入训练 |
| `foundation.brainomni_adaptation` | BrainOmni 专用 DEFT 迁移、容量匹配残差 MLP、原始掩码 RVQ 预测适配；`AdaLoraStep` 与显式正交损失补齐 tensor-returning 编码器的 PEFT 训练；`subject_pair_order` 保留同人正例且每行每轮一次 |
| `foundation.brainomni_personal` | 显式标注校准/间隔/评分分区，个人 FiLM 与冻结群体 LoRA+个人 LoRA；保存/恢复仅个人参数，需单独按校准协议报告 |
| `foundation.brainomni_multitask` | 已知任务 ID 路由的共享 A/任务独立 B LoRA，与任务各自独立 A/B 的联合训练对照；不使用被试身份路由 |
| `foundation.sessionfm` | `SessionFM`、`SessionBaselineState`、`session_spectral_features`：会话基线条件化的 120 Hz / 4 s 时频双支模型、带训练坐标噪声的 Fourier+MLP 位置编码、criss-cross 编码器、分位置全局重建头、先验锚定读出和因果逐窗状态 |
| `foundation.sessionfm_downstream` | `SessionFMDownstream`：冻结训练折 Logistic 数学先验 + sigmoid 门控 SessionFM 残差；两阶段 LP-FT、30% 频谱掩码损失、逐层 LR 衰减和 SBC/1394-D 先验/CCB 三状态因果流式预测 |
| `foundation.bnci2015` | BNCI2015-004 MOABB 元数据审计与已审计窗口包装；cue 窗口不预设，事件/通道/类别合同须由调用方核验 |
| `splits.ordered_kshot_mask` | 按显式 `recording_order`（或输入行序）选择指定参与者/会话的每类前 k 个样本 |
| `readout` | `linear_probe`、`loso_probe`、`nested_loso_readout`（内层 LOSO 选 C） |
| `registry` | `save_run`（oof.npz + report.json + sha256）、`load_oof`（兼容历史键名） |
| `submission` | `build_zip`（根目录打包 + 清单）、`check_zip`（根目录、单 .py、`reset_state`、`submission_dir`） |

命令行：

```powershell
wyu-bci-bench oof-score path\to\oof.npz
wyu-bci-bench oof-compare a.npz:simclr b.npz:random
```

## 旧函数 → bench 对照

旧脚本保持原样（它们的 sha256 记录在各 report.json 里，改动会破坏可追溯性）。新实验只从 bench 导入。

| 旧实现（`challenge2026_track02/proteus_update1/`） | bench |
|---|---|
| OpenTSLab/BrainOmni `BrainOmni.encode` / `DownstreamModel`（新增官方来源） | `foundation.brainomni.TokenBackbone` / `BrainOmniClassifier`；冻结分词缓存路径与真实官方编码等价，训练复用 `supervised.train_classifier` |
| wythedee/DEFT `LaBraM/disentangle/gaussian_model.py`（commit 875b95fe） | `_third_party.deft.gaussian_model` 原文保留 MIT 许可；`foundation.brainomni_adaptation.BrainOmniDEFT` 使用 BrainOmni token 网格、相同分类头及 RVQ embedding 重建；属于迁移实验，非原论文逐项复现 |
| `differentiable_feature_module_loso.cell_score`、`strict_directions_experiment.cell_balanced_score`、`cbramod_full4s_probe.cell_score` 等 81 处 | `metrics.cell_balanced_accuracy` / `grouped_balanced_accuracy` |
| `all_subject_loso_generalization.summarize`、`ssl_experiment.summarize`、`strict_directions_experiment.summarize` | `metrics.summarize_predictions` |
| `representation_learning_20261019/analyze_results.paired` | `metrics.paired_subject_bootstrap`（10-19 区间的实现） |
| `strict_directions_experiment.paired_cluster_bootstrap` | 同上；旧版按 cell 拼接，不保留 cell 权重抽样方式，勿再用 |
| `pilot_labram/.../postprocess_paper_results.exact_sign_flip_pvalue`、`reporting_20260919/build_results.sign_flip` | `metrics.exact_sign_flip_pvalue` |
| `confirm_power_first_4bin_multiseed.bootstrap_ci`、`论文交付/audit_primary_experiments.bootstrap_mean` | `metrics.bootstrap_mean_ci` |
| Track 2 sealed G-collapse 统计、P04/P10 G-guard 检查 | `protocol_diagnostics.collapse_summary` / `collapse_gate`、`guard_comparison` |
| 各处 `subject_split()`、`foundation_probe_controlled_loso.nested_loso`、`foundation_pool_probe.nested_probe` | `splits.loso` / `inner_loso`、`readout.nested_loso_readout` |
| `cross_session_diagnostic.main` 的 ses-1→ses-2/3 切分 | `splits.cross_session` |
| Track 2 N2 / E0.5 / T26 实验脚本内的 NSP、整会话中心化与类原型更新 | `innovation`；实验参数和折分仍由各实验入口负责 |
| Track 2 SICT causal session-context residual decoder | `sict.SICTAdapter` / `sict.SICTStream`；训练协议由新实验入口管理 |
| Track 2 F1 低频事件相关波形空间支路 | `event_waveform.low_frequency_waveforms`、`XDawnTangentLogistic`、`ClassTemplateProjection`；逐折训练和融合由 `challenge2026_track02/experiments/run_f1_event_waveform.py` 管理 |
| `differentiable_feature_module_loso.DifferentiableFeatureModule` / `get_scaler_features` | `priors.MathPriorFeatures` / `math_prior_features` |
| `math_prior_multiscale_covnet.*`、`spatial_energy_ablation_loso.PowerFirst*`、`make_model`、`train_model`、`all_subject_loso_generalization.sampling_weights` | `nets.*` |
| `representation_learning_20261019/ssl_experiment.*` | `ssl.*` + `readout.probe_proba` |
| `representation_learning_20261019/cbramod_full4s_probe.extract` / `nested` | `foundation.cbramod.extract_features` + `readout.nested_loso_readout` |
| `foundation_probe_proteus.extract_eegpt`（旧合同） | 已作废；改用 `foundation.eegpt.extract_features`，旧实现不复用其未经核验的输入缩放 |
| CoCoT-EEG 论文模型与下游微调 | `cocot.make_cocot_model`、`cocot.fit_predict_cocot`；外层训练折内验证，微调默认全量更新，encoder 权重须由调用方提供并严格加载 |
| Track 2 第三轮 G2 DSPC 少样本适配 | `fewshot.balanced_kshot_indices`、`estimate_decoder_prior`、`fit_map_logistic`、`semi_supervised_em_logistic`；Dreyer runner 只编排事件抽窗、5 折参与者划分、source-LOO 调参和重复统计 |
| 各处 `sha256()`、`save_progress`、`write_json` | `registry.save_run`、`grouped.file_sha256` |
| `reproduction_20261004/package_submissions.package_one`、`build_math_prior_submission.main` 的打包部分 | `submission.build_zip` + `check_zip` |

| `train_proteus_update1.extract_windows` | `proteus_bids.extract_windows`（sub-P01 重新提取与缓存 max\|diff\|=0） |
| `temporal_features.extract_variant` 及其在各提交包里的拷贝 | `window_features.extract_variant` |
| `cross_subject_supcon_loso.cross_subject_supcon` | `ssl.cross_subject_supcon` |
| `analyze_results.mean_summaries` | `metrics.mean_summaries` |
| `论文交付/audit_primary_experiments.py`（读旧 CBraMod 合同） | `论文交付/audit_primary_experiments_bench.py`（bench 版，读 `cbramod_native`） |

尚未迁移（仍在原脚本，需要时再收进 bench）：`ContentStyleNet`/`GroupRiskLinear`、外部数据拼接、pilot_labram 的 EEGPT 下游头与训练逻辑。EEGPT encoder-only 适配已完成严格结构加载和前向验证，但数据集幅值映射、checkpoint 来源认证及目标任务效果仍待验证；LaBraM PROTEUS 编码器合同已于 11-09 完成源代码及真实权重前向审计，修正后任务实验进行中。BIOT 已实现显式 montage/窗长/采样率合同，真实目标任务效果仍未验证。

模型家族、训练目标、读出器和评估协议的分类及当前可运行状态见 [模型目录](model_catalog.md)。Braindecode 深度基线通过 `pip install -e ".[braindecode]"` 安装；本地 EEG-ResNet 与 InceptionTime 使用 PyTorch。当前已有结构前向及浅层网络训练 smoke test，尚未在 PROTEUS 等目标数据集比较效果。`FilterBankCSP` 通过 `pip install -e ".[csp]"` 安装 MNE CSP、SciPy 和 scikit-learn。掩码自编码器与无标签训练入口已实现，但只完成小批合成数据 smoke test。在线推理延迟只代表当前主机测量；时间漂移摘要需要延迟标签做离线审计，不执行自动检测或适配。

核验：`python scripts/verify_proteus_port.py --workspace D:/脑电大模型 --with-bids`（29 项，含原始 EDF 重提取，约 1 分钟）；不加 `--with-bids` 为 28 项。

## 已知注意事项

- 数学先验特征在 CUDA float32 下与 CPU 有微小差异，PROTEUS LOSO 探针会从 0.5113 变为 0.5109；复现历史数字用 CPU。
- 无类别窗口的 cell：PROTEUS 历史口径记 recall=0（`absent="zero"`），NeuralBench 流指标用 `absent="skip"`。
- PROTEUS 十名公开参与者都有研发曝光；bench 里的任何 LOSO 数字都是开发证据，不是 sealed 测试。
- 微调模块只控制梯度参数集合或挂接 PEFT adapter，不替代数据划分、预处理合同、训练循环或内层选参；新策略应在每个外层训练折拟合。
