# EEG 模型与方法分类

本文档按“可直接运行的模型/特征分类器、表征训练目标、预训练基础模型、读出器、评估协议”分层登记。它们处在不同流程阶段，不应混作同一类模型，也不能仅凭名称推断性能。

## 可直接运行的分类模型

| 家族 | bench 入口 | 主要做法 | 适合用来回答的问题 | 状态与注意事项 |
|---|---|---|---|---|
| 线性与概率基线 | `models`：对角 LDA、Gaussian NB、Logistic Regression、最近质心；`sklearn_ml`：线性 SVM | 在固定特征上拟合简单判别边界或类条件分布 | 深度网络是否真的超过低方差、易解释的基线 | 已实现；标准化、特征选择必须只在训练折拟合 |
| 邻居与树模型 | `models`：KNN；`sklearn_ml`：决策树、随机森林、ExtraTrees、AdaBoost、HistGradientBoosting；`treeboost`：XGBoost | 依据邻近样本或树的分段规则分类 | 非线性边界或非线性特征组合是否有帮助 | 已实现为单模型入口或 ML 套件成员；小样本下应控制树深度并做重复种子评估 |
| FBCSP 风格固定投影特征 | `models.FilterBankLDAModel`、`riemann.FilterBankTangentSpaceLDAModel` | 多频带空间投影/协方差特征，经筛选后分类 | 运动想象任务的低成本对照 | 已实现；`FilterBankLDAModel` 使用固定投影与 log-variance，并不学习 CSP 空间滤波器 |
| 真正的滤波器组 CSP + SVM | `csp.FilterBankCSP` | 对每个固定频带，仅用训练 epoch 拟合 MNE CSP，拼接各频带对数平均功率，再标准化并训练线性 SVM | 与 FBCSP+SVM 经典流程对应的学习式空间滤波基线 | 算法接口已实现并通过合成数据测试；需安装 `.[csp]`；尚无真实目标数据集效果结论 |
| 真正的滤波器组 CSP + 收缩 LDA | `csp.FilterBankCSP(classifier="lda")` | 同一组每带训练 CSP 特征改由标准化 + 收缩 LDA 读出 | 经典小样本线性分类对照 | 接口和合成测试已通过；未在目标公开数据集评估 |
| 黎曼几何 | `riemann`：MDM、滤波器组 MDM、切空间 + LDA | 把 trial 协方差当作对称正定矩阵，在黎曼流形或其切空间上分类 | 小样本、空间协方差结构是否携带判别信息 | 已实现；黎曼参考矩阵和所有统计量必须在训练折估计 |
| 固定卷积核特征 | `minirocket` | 使用不参与梯度训练的卷积核提取时间序列特征，再由分类器读出 | 不训练深层卷积网络时，固定时序特征是否有效 | 已实现；属于特征变换 + 分类器，不等同端到端神经网络 |
| 轻量卷积网络 | `deep`：EEGNet | 深度可分离卷积拆开时间滤波与空间混合，控制参数规模 | 小样本 EEG 中紧凑端到端网络的基准表现 | 已实现；窗口化、归一化和早停需限制在当前训练折 |
| 胶囊网络 | `capsule`：CapsuleNet | 以向量胶囊和动态路由组合局部特征 | 胶囊表示能否作为投票集成成员 | 已实现为可选成员；不要默认它优于普通 CNN |
| 数学先验 + 多尺度网络 | `nets`：M1、M2 | 将频带/Hjorth 数学先验与多尺度时空卷积结合 | 结构先验与网络表征结合后是否改善目标协议 | 已实现；现有 PROTEUS 数字是开发证据，不能视作 sealed 泛化结论 |
| ShallowFBCSPNet | `deep_models`：`shallow_fbcsp` | 浅层时空卷积学习类似频带滤波与空间投影的操作，再以平方/对数类变换形成判别表征 | 与传统 FBCSP/LDA 对照：端到端学习是否有增益 | Braindecode 可选基线；架构可运行，尚无本 bench 任务的效果结论 |
| Deep4Net | `deep_models`：`deep4` | 多层时空卷积与池化逐步提取 EEG 局部模式 | 深层卷积基线是否比浅层网络更适合当前数据规模 | Braindecode 可选基线；需要按训练折选 epoch |
| EEGConformer | `deep_models`：`eegconformer` | 卷积 token/patch 表征后接自注意力模块建模较长程关系 | 注意力模块是否补充卷积的局部时空特征 | Braindecode 可选基线；这里登记的是架构入口，不代表预训练模型 |
| FBCNet | `deep_models`：`fbcnet` | 滤波器组式空间表征与方差/时域统计结合 | 多频带空间特征是否适合端到端学习 | Braindecode 可选基线；与传统手工 FBCSP pipeline 分开登记 |
| ATCNet | `deep_models`：`atcnet` | 时间卷积与注意力/多分支上下文特征组合 | 时间上下文模块是否改善运动想象分类 | Braindecode 可选基线；当前依赖在 500 点输入前向已核验，但未做完整数据集实验 |
| TCN | `deep_models`：`tcn` | 扩张时序卷积；bench 对逐时间点类别分数做均值读出 | 较长时间依赖能否由卷积感受野捕获 | Braindecode 架构接口已实现；前向与试次读出通过形状测试，未做目标数据集训练实验 |
| EEGTCNet | `deep_models`：`eegtcnet` | EEGNet 风格时空特征后接时序卷积网络 | 紧凑时空卷积与 TCN 组合的效果 | Braindecode 架构接口已实现；未做目标数据集效果比较 |
| EEGInceptionMI | `deep_models`：`eeginception_mi` | 并行时间卷积分支提取不同时间尺度，再融合用于 MI 分类 | 多尺度时间滤波是否补充单尺度卷积 | Braindecode 架构接口已实现；不等于完整复现所有论文训练配方 |
| EEGNeX | `deep_models`：`eegnex` | 基于 EEGNet 设计的扩展时空卷积结构 | 较大时间感受野与深度卷积结构的对照 | Braindecode 架构接口已实现；未做目标数据集效果比较 |
| EEGITNet | `deep_models`：`eegitnet` | 卷积前端加时序卷积模块，组合局部信号与上下文特征 | CNN-TCN 混合结构的对照 | Braindecode 架构接口已实现；未做目标数据集效果比较 |
| EEG-ResNet | `deep_models`：`eeg_resnet` | 本地 PyTorch 多层残差时序卷积，逐层下采样后做 trial 级均值读出 | 残差连接对时间特征提取的影响 | 结构前向通过；本地紧凑实现，不声称是 EEG 领域特定论文的逐项复现 |
| InceptionTime | `deep_models`：`inception_time` | 并行短/中/长卷积核，加池化分支融合不同时间尺度 | 多尺度卷积对不同持续时间信号模式的覆盖 | 结构前向通过；本地轻量实现，不声称逐项复现原论文配方 |

## 按 BCI 范式的专用解码器

| 范式 | bench 入口 | 方法 | 训练/测试边界 |
|---|---|---|---|
| SSVEP | `ssvep.SSVEPCCA` | 基于刺激频率正余弦参考信号的 CCA | 频率与谐波由实验设计提供；若相位由训练数据估计，估计必须放在训练折内 |
| SSVEP | `ssvep.FilterBankCCA` | 多频带 CCA 相关系数平方的加权和（FBCCA 风格） | 滤波频带不从测试标签选择；短 epoch 需要满足零相位滤波的 padding 要求 |
| SSVEP | `ssvep.TRCA` | 同类重复试次学习任务相关空间滤波器，再与类别模板计算相关 | 只在训练试次学习空间滤波器和模板；至少每类两条训练试次 |
| SSVEP | `ssvep.EnsembleTRCA` | MOABB eTRCA：跨类别集成 TRCA 空间滤波器；类别顺序必须对应真实刺激频率 | 频率识别基线；接口已验证，合成随机相位信号不用于声明性能，尚无公开 SSVEP 数据集结果 |
| SSVEP | `ssvep.TDCA` | MOABB 任务判别成分分析，利用延迟嵌入的时空信号判别刺激类别 | SSVEP 时空判别基线；接口已验证，尚无公开数据集结果 |
| ERP / P300 | `erp.XDawnLDA` | 训练折目标平均 ERP 的 xDAWN 空间滤波 + 收缩 LDA | 输入是已切好的刺激 epoch；事件检测、目标/非目标标记和 epoch 边界由调用方提供 |

## 无监督域对齐

| 方法 | bench 入口 | 输入与作用 | 评估协议限制 |
|---|---|---|---|
| Euclidean Alignment (EA) | `adaptation.euclidean_align`、`EuclideanAlignment` | 对一个域的无标签 trial 协方差取均值，求逆平方根后对白化 EEG 通道 | 若对齐矩阵用测试被试/测试会话的整批输入估计，这是传导式测试时校准；必须在协议中声明，不可与严格归纳式 LOSO 混报 |
| CORAL | `adaptation.coral_align` | 用目标无标签特征的均值与协方差，将源特征变换到目标统计分布 | 读取整批目标输入同样属于传导式适配；目标标签不得参与对齐、调参或停止条件 |
| 黎曼几何域对齐 | `adaptation.RiemannianAlignment` | 按域计算 SPD trial 协方差的黎曼几何均值，并用其逆平方根白化 | 减少域间协方差几何差异的设计目标；PyRiemann 接口和合成测试通过，无公开目标域性能结论；目标统计需声明访问协议 |
| 最大均值差异（MMD） | `adaptation.maximum_mean_discrepancy` | 对源/目标特征计算多核 RBF MMD²，可与任务损失联合优化 | 衡量或作为域不变表征的分布差异正则；仅提供损失项，不含编码器或优化循环，未做任务效果评估 |
| 域对抗神经网络（DANN） | `adaptation.DomainAdversarialHead`、`domain_adversarial_loss` | 域分类头前设置梯度反转，使共享编码器接收相反的域分类梯度 | 学习难以区分域的特征表示；任务头、数据采样和训练循环由调用方实现，未做任务效果评估 |
| 自适应批归一化（AdaBN） | `adaptation.adapt_batch_norm` | 用无标签目标批次更新 BatchNorm 运行统计，保持其余层处于 eval | 测试时统计校准的设计目标；合成统计更新测试通过，用评估输入校准时属于测试时适配，未做性能评估 |

## 可解释性与概率可靠性

| 方法 | bench 入口 | 测量内容 | 协议边界 |
|---|---|---|---|
| 分组置换重要性 | `interpretability.grouped_permutation_importance` | 在留出特征矩阵上联合打乱特征组，观察 balanced accuracy 等指标下降 | estimator 必须在训练折拟合；只能在验证/测试特征上解释。分数下降是模型依赖诊断，不是因果效应 |
| 通道消融 | `interpretability.channel_ablation_scores` | 将留出 epoch 的指定电极置零，报告 BA 相对完整输入的变化 | 模型不重训；通道顺序必须预先统一。不是删通道重训后的性能比较 |
| 频带消融 | `interpretability.band_ablation_scores` | 用 Butterworth 带阻移除指定频段，报告留出 BA 变化 | 频带需事先指定，测试只做输入扰动；变化不能解释为频段的因果效应 |
| 温度缩放 | `calibration.fit_temperature` + `apply_temperature` | 在 logits 上拟合单一正温度，压平或锐化概率但不改变类别排序 | 温度只在训练流程内的验证折拟合，最后在独立测试折评估；不可用测试标签拟合 |
| ECE / Brier | `calibration.expected_calibration_error`、`multiclass_brier_score` | ECE 衡量置信度与准确率分箱差异；Brier 衡量概率与 one-hot 标签的平方误差 | 都是概率质量指标，需与分类 BA/F1 并列报告，不能替代分类指标 |
| 置信度拒识 | `calibration.risk_coverage_curve` | 由高到低接纳预测，展示覆盖率与接受样本错误率的关系 | 阈值如需选定，应在验证折确定；曲线在独立测试折报告，并注明覆盖率/风险口径 |

这些诊断模块提供计算方法，不附带目标数据集结果。置换/消融回答模型对输入扰动的敏感性；概率校准与拒识回答预测置信度是否可用，不能据此宣称特征具有生理因果作用或准确率得到提升。

本批 SSVEP/ERP/域对齐实现已用合成数据测试算法接口和数值合同；尚未接入标准 SSVEP/ERP 数据集跑协议级基准，因此没有可报告的真实数据效果结论。

`FilterBankCSP`、新增深度架构、SSVEP 扩展、域对齐方法、掩码自编码器和在线评估目前只完成接口/合成数据测试。CSP 的滤波带、CSP 维度与分类器参数须在训练折内确定；深度网络的训练验证继续按原始 trial 分组，不能先切重叠窗口再随机拆分。MMD/DANN 是训练组件，不是端到端现成模型。在线延迟只代表测量时的主机与运行时；漂移摘要依赖延迟标签做离线审计，不是自动检测器。

范式算法的经典来源包括 CCA（Lin et al., IEEE TBME, 2006, DOI: `10.1109/TBME.2006.886577`）、FBCCA（Chen et al., Journal of Neural Engineering, 2015, DOI: `10.1088/1741-2560/12/4/046008`）、TRCA（Nakanishi et al., IEEE Transactions on Biomedical Engineering, 2018, DOI: `10.1109/TBME.2017.2694818`）、xDAWN（Rivet et al., IEEE Transactions on Biomedical Engineering, 2009, DOI: `10.1109/TBME.2009.2012869`）与 EA（He and Wu, IEEE Transactions on Biomedical Engineering, 2020, DOI: `10.1109/TBME.2019.2913914`）。这些引用说明算法来源，不代表本 bench 在特定数据集上的性能结论。

## 表征学习与预训练模型

| 类型 | bench 当前内容 | 说明 |
|---|---|---|
| 自监督训练目标 | `ssl`：SimCLR、VICReg、TS2Vec 风格分层时间对比、patch 级 `MaskedEEGAutoencoder`、预测式损失、跨通道对比损失 | `train_masked_autoencoder` 只接收无标签 EEG tensor batch；使用前先在选定协议内隔离预训练参与者，当前 recipe 只经小批合成数据 smoke test，尚无目标数据集结果 |
| 跨被试监督对比 | `ssl.cross_subject_supcon` | 使用标签与被试信息组织正负样本，设计目标是同类跨被试靠近；该意图不等于已证明消除域偏移 |
| 预训练基础模型 | `foundation.cbramod` | CBraMod 输入合同已登记并可提取冻结特征；需遵守 4 s、200 Hz、µV÷100 和 patch 输入约定 |
| 预训练基础模型 | `foundation.biot` | BIOT 官方 TUAB 合同：200 Hz、10 s、官方 16/18 个有序双极导联；每 epoch/通道按绝对振幅 95 分位归一化；严格加载对应 encoder 权重后输出 256-D 冻结特征。其他窗长需独立验证 |
| 预训练基础模型 | `foundation.eegpt` | 官方 58 通道、256 Hz、4 s / 1024 点、64 点 patch；通道重排和严格 checkpoint encoder 加载已核验，输出 16×4×512 token 后均值池化为 512-D。必须显式提供数据集幅值乘数；未验证 PROTEUS 映射或任务效果 |
| 待审计候选 | LaBraM | 尚未完成可复现的 tokenizer/checkpoint 输入映射和输出合同，不提供可运行适配器 |

## 在线与流式评估

| 能力 | bench 入口 | 作用 | 边界 |
|---|---|---|---|
| 因果滤波 | `online.design_causal_bandpass`、`CausalSOSFilter` | 设计 SOS 带通滤波器并跨输入块保留滤波状态，不读取未来采样点 | 这是单向因果滤波；与零相位离线滤波的相位/边缘行为不同，部署前须按目标采样率和通道顺序验证 |
| 连续窗口 | `online.StreamWindowBuffer` | 从任意长度连续数据块中按固定窗长和步长发出重叠窗口 | 缓冲器只管理窗口，不含设备采集、时钟同步或丢包恢复 |
| 延迟与吞吐 | `online.benchmark_streaming` | 跳过 warm-up 后测量当前主机窗口推理的 p50/p95 延迟、吞吐和截止时间超限数 | 结果必须附主机、设备、批量和运行时版本；不能直接外推到其他设备 |
| 时间漂移摘要 | `online.temporal_drift_summary` | 按时间顺序分箱报告 accuracy、balanced accuracy 及相对首箱变化 | 需要延迟标签做离线审计；该摘要本身不执行在线漂移检测或自动适配 |

## 与模型分开的组件

- **读出器**：`readout.probe_proba`、`loso_probe`、`nested_loso_readout` 把已得到的特征映射到标签。线性探针不是新的 EEG 骨干网络。
- **微调策略**：冻结、指定模块解冻、全量微调，以及 LoRA / DoRA / PiSSA / AdaLoRA 决定更新哪些参数，不定义骨干架构。
- **划分协议**：LOSO、跨会话、受试者比例划分规定训练/测试样本如何分离，不是模型。
- **集成**：hard voting 汇总多个模型的离散预测；它与单个分类器要分别报告。

## 推荐的比较顺序

1. 先跑适配当前任务的简单基线（LDA/SVM 或黎曼 MDM），确认数据与指标管线正常。
2. 加入 EEGNet、ShallowFBCSPNet 和 Deep4Net 作为经典端到端 CNN 对照；按资源情况再跑 FBCNet、ATCNet 和 EEGConformer。
3. 有无标签预训练数据时，单独比较自监督表征 + 固定线性读出；不要把预训练目标与端到端监督网络的训练数据条件混在一起。
4. 只有在骨干输入合同、权重来源和数据划分均核验后，才比较基础模型冻结读出或微调。

所有模型都必须使用同一外层划分和同一指标；窗口模型按试次切分，禁止同一试次的重叠窗口同时进入训练和验证/测试。结果必须注明协议与证据等级。十名公开 PROTEUS 参与者均有研发曝光，相关结果只属于开发证据。Braindecode 模型当前有结构前向与一个浅层模型训练 smoke test；新增网络通过结构前向测试但没有目标数据集训练/性能结果；SSVEP/ERP/域适配和 CSP 仅完成合成数据接口验证。
