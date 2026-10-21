# 跨被试 / 跨会话 / 基础模型协议层（v0.2.0）

v0.1 面向机械手单人 K 折；v0.2 增加按参与者分组的协议层，供毕业论文（PROTEUS）、EEG/EMG Challenge 2026 Track 2 和 OmniEEG Challenge 2026 共用。所有新模块已用 `scripts/verify_proteus_port.py` 对照历史 OOF 与原始代码核验（`--with-bids` 共 29/29 通过：论文与 10-19 复现报告的 cell BA、bootstrap 区间、sign-flip p 值逐位一致；数学先验特征、Welch 特征变体、SupCon、M1/M2 的种子初始化与前向输出与原实现完全相同；从原始 EDF 重新提取 sub-P01 与缓存逐位相同）。

## 模块

| 模块 | 内容 |
|---|---|
| `grouped` | `GroupedEEGDataset`（X/y/subjects/sessions/interfaces）、`load_grouped_npz`、`load_proteus_windows`（(N,41,480) µV @120 Hz 校验） |
| `proteus_bids` | `extract_windows`（NEMAR nm000290 BIDS → 窗口，逐位等于历史缓存）、`save_cache` |
| `window_features` | `extract_variant`（Welch 频带功率/相对功率/Hjorth，`whole`、`pre_post_car` 等变体） |
| `splits` | `loso`、`inner_loso`、`cross_session`、`neuralbench_sealed`、`subject_ratio`（OmniEEG 8:1:1）、`multi_seed_subject_ratio`；全部返回布尔掩码 `Split` 并可 `check_disjoint` |
| `metrics` | `cell_balanced_accuracy`（PROTEUS 主指标）、`stream_mean_balanced_accuracy`（NeuralBench `bal_acc_stream_mean`）、`summarize_predictions`、`paired_subject_bootstrap`、`exact_sign_flip_pvalue`、`bootstrap_mean_ci` |
| `priors` | `MathPriorFeatures` / `math_prior_features`：1394 维 log/相对频带功率 + Hjorth（默认 CPU，保证复现 0.5113） |
| `nets` | `MultiScaleTemporalEnergy`、`BandCovarianceProjection`、`MathPriorMultiScaleCovNet`（M1）、`PowerFirstMathPriorNet`（M2）、`make_model`、`train_model`、`predict_proba`、`sampling_weights` |
| `deep_models` | 可选 Braindecode 网络以及本地 EEG-ResNet、InceptionTime 工厂与折内训练；验证集按试次划分，测试窗口概率按试次平均 |
| `csp` | `FilterBankCSP`：MNE CSP 在每个预设频带训练空间滤波器，拼接 log-average-power 后接标准化线性 SVM 或收缩 LDA；滤波器与分类器均只拟合训练折 |
| `ssvep` | 刺激参考 CCA、滤波器组 CCA（FBCCA 风格）、TRCA/eTRCA 与 MOABB TDCA；真实刺激频率按类别显式传入 |
| `erp` | 已分段 epoch 的 xDAWN 空间滤波 + 收缩 LDA |
| `adaptation` | Euclidean/Riemannian Alignment、CORAL、MMD、DANN 域头与 AdaBN；使用评估目标域输入拟合时按传导式测试时适配报告，MMD/DANN 需调用方提供编码器与训练循环 |
| `ssl` | `RepresentationModel`、`TS2VecModel`、`MaskedEEGAutoencoder`、SimCLR/VICReg/TS2Vec/掩码/预测/跨通道损失、`cross_subject_supcon`、`train_ssl`、`train_ts2vec`、`train_masked_autoencoder`、`encode` |
| `online` | `CausalSOSFilter`、`StreamWindowBuffer`、`benchmark_streaming`、`temporal_drift_summary`：因果滤波、连续窗切分、当前主机延迟/吞吐和时间分箱摘要 |
| `foundation` | 输入合同注册表 `CONTRACTS`；`require_verified` 拒绝未审计合同；`foundation.cbramod.extract_features`（4 s / 200 Hz / µV÷100）；`foundation.biot.extract_features`（官方有序 16/18 导联、200 Hz、10 s、官方幅值归一化、256-D 冻结编码器）；`foundation.eegpt.extract_features`（官方 58 通道、4 s / 256 Hz，显式 `amplitude_scale`，严格加载 encoder，512-D 冻结特征）；`foundation.finetuning` 的全量/冻结/指定层策略与 PEFT LoRA 系适配 |
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
| `differentiable_feature_module_loso.cell_score`、`strict_directions_experiment.cell_balanced_score`、`cbramod_full4s_probe.cell_score` 等 81 处 | `metrics.cell_balanced_accuracy` / `grouped_balanced_accuracy` |
| `all_subject_loso_generalization.summarize`、`ssl_experiment.summarize`、`strict_directions_experiment.summarize` | `metrics.summarize_predictions` |
| `representation_learning_20261019/analyze_results.paired` | `metrics.paired_subject_bootstrap`（10-19 区间的实现） |
| `strict_directions_experiment.paired_cluster_bootstrap` | 同上；旧版按 cell 拼接，不保留 cell 权重抽样方式，勿再用 |
| `pilot_labram/.../postprocess_paper_results.exact_sign_flip_pvalue`、`reporting_20260919/build_results.sign_flip` | `metrics.exact_sign_flip_pvalue` |
| `confirm_power_first_4bin_multiseed.bootstrap_ci`、`论文交付/audit_primary_experiments.bootstrap_mean` | `metrics.bootstrap_mean_ci` |
| 各处 `subject_split()`、`foundation_probe_controlled_loso.nested_loso`、`foundation_pool_probe.nested_probe` | `splits.loso` / `inner_loso`、`readout.nested_loso_readout` |
| `cross_session_diagnostic.main` 的 ses-1→ses-2/3 切分 | `splits.cross_session` |
| `differentiable_feature_module_loso.DifferentiableFeatureModule` / `get_scaler_features` | `priors.MathPriorFeatures` / `math_prior_features` |
| `math_prior_multiscale_covnet.*`、`spatial_energy_ablation_loso.PowerFirst*`、`make_model`、`train_model`、`all_subject_loso_generalization.sampling_weights` | `nets.*` |
| `representation_learning_20261019/ssl_experiment.*` | `ssl.*` + `readout.probe_proba` |
| `representation_learning_20261019/cbramod_full4s_probe.extract` / `nested` | `foundation.cbramod.extract_features` + `readout.nested_loso_readout` |
| `foundation_probe_proteus.extract_eegpt`（旧合同） | 已作废；改用 `foundation.eegpt.extract_features`，旧实现不复用其未经核验的输入缩放 |
| 各处 `sha256()`、`save_progress`、`write_json` | `registry.save_run`、`grouped.file_sha256` |
| `reproduction_20261004/package_submissions.package_one`、`build_math_prior_submission.main` 的打包部分 | `submission.build_zip` + `check_zip` |

| `train_proteus_update1.extract_windows` | `proteus_bids.extract_windows`（sub-P01 重新提取与缓存 max\|diff\|=0） |
| `temporal_features.extract_variant` 及其在各提交包里的拷贝 | `window_features.extract_variant` |
| `cross_subject_supcon_loso.cross_subject_supcon` | `ssl.cross_subject_supcon` |
| `analyze_results.mean_summaries` | `metrics.mean_summaries` |
| `论文交付/audit_primary_experiments.py`（读旧 CBraMod 合同） | `论文交付/audit_primary_experiments_bench.py`（bench 版，读 `cbramod_native`） |

尚未迁移（仍在原脚本，需要时再收进 bench）：`ContentStyleNet`/`GroupRiskLinear`、外部数据拼接、pilot_labram 的 EEGPT 下游头与训练逻辑。EEGPT encoder-only 适配已完成严格结构加载和前向验证，但数据集幅值映射、checkpoint 来源认证及目标任务效果仍待验证；LaBraM tokenizer/checkpoint 输入合同仍未完成。BIOT 已实现显式 montage/窗长/采样率合同，真实目标任务效果仍未验证。

模型家族、训练目标、读出器和评估协议的分类及当前可运行状态见 [模型目录](model_catalog.md)。Braindecode 深度基线通过 `pip install -e ".[braindecode]"` 安装；本地 EEG-ResNet 与 InceptionTime 使用 PyTorch。当前已有结构前向及浅层网络训练 smoke test，尚未在 PROTEUS 等目标数据集比较效果。`FilterBankCSP` 通过 `pip install -e ".[csp]"` 安装 MNE CSP、SciPy 和 scikit-learn。掩码自编码器与无标签训练入口已实现，但只完成小批合成数据 smoke test。在线推理延迟只代表当前主机测量；时间漂移摘要需要延迟标签做离线审计，不执行自动检测或适配。

核验：`python scripts/verify_proteus_port.py --workspace D:/脑电大模型 --with-bids`（29 项，含原始 EDF 重提取，约 1 分钟）；不加 `--with-bids` 为 28 项。

## 已知注意事项

- 数学先验特征在 CUDA float32 下与 CPU 有微小差异，PROTEUS LOSO 探针会从 0.5113 变为 0.5109；复现历史数字用 CPU。
- 无类别窗口的 cell：PROTEUS 历史口径记 recall=0（`absent="zero"`），NeuralBench 流指标用 `absent="skip"`。
- PROTEUS 十名公开参与者都有研发曝光；bench 里的任何 LOSO 数字都是开发证据，不是 sealed 测试。
- 微调模块只控制梯度参数集合或挂接 PEFT adapter，不替代数据划分、预处理合同、训练循环或内层选参；新策略应在每个外层训练折拟合。
