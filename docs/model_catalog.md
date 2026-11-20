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
| ShallowFBCSPNet | `deep_models`：`shallow_fbcsp` | 浅层时空卷积学习类似频带滤波与空间投影的操作，再以平方/对数类变换形成判别表征 | 与传统 FBCSP/LDA 对照：端到端学习是否有增益 | 属于 K2M2 的 D6 两个浅层分支；K2M2 N/S M=0.7147，作为新支路候选不重复计分 |
| Deep4Net | `deep_models`：`deep4` | 多层时空卷积与池化逐步提取 EEG 局部模式 | 深层卷积基线是否比浅层网络更适合当前数据规模 | Track 2 40 轮单种子 N/S；D6 家族补充 M=0.7070，低于基线 |
| EEGConformer | `deep_models`：`eegconformer` | 卷积 token/patch 表征后接自注意力模块建模较长程关系 | 注意力模块是否补充卷积的局部时空特征 | 480 点输入用池化窗/步长 30/15 完成 8 轮 N/S；D6 家族补充 M=0.7087 |
| FBCNet | `deep_models`：`fbcnet` | 滤波器组式空间表征与方差/时域统计结合 | 多频带空间特征是否适合端到端学习 | 完成 8 轮 N/S；D6 家族补充 M=0.7108 |
| ATCNet | `deep_models`：`atcnet` | 时间卷积与注意力/多分支上下文特征组合 | 时间上下文模块是否改善运动想象分类 | 480 点输入用 `n_windows=2` 完成 8 轮 N/S；D6 家族补充 M=0.7110，配对差 -0.31 pp（95% CI [-0.81,+0.16]） |
| TCN | `deep_models`：`tcn` | 扩张时序卷积；bench 对逐时间点类别分数做均值读出 | 较长时间依赖能否由卷积感受野捕获 | 完成 8 轮 N/S；D6 家族补充 M=0.7105 |
| EEGTCNet | `deep_models`：`eegtcnet` | EEGNet 风格时空特征后接时序卷积网络 | 紧凑时空卷积与 TCN 组合的效果 | 完成 40 轮 exact N/S；D6 家族补充 M=0.7096 |
| EEGInceptionMI | `deep_models`：`eeginception_mi` | 并行时间卷积分支提取不同时间尺度，再融合用于 MI 分类 | 多尺度时间滤波是否补充单尺度卷积 | 完成 8 轮 N/S；D6 家族补充 M=0.7134；非论文完整训练配方 |
| EEGNeX | `deep_models`：`eegnex` | 基于 EEGNet 设计的扩展时空卷积结构 | 较大时间感受野与深度卷积结构的对照 | 完成 40 轮 exact N/S；D6 家族补充 M=0.7130 |
| EEGITNet | `deep_models`：`eegitnet` | 卷积前端加时序卷积模块，组合局部信号与上下文特征 | CNN-TCN 混合结构的对照 | 完成 12 轮 N/S 排序 pilot；D6 家族补充 M=0.7098 |
| EEG-ResNet | `deep_models`：`eeg_resnet` | 本地 PyTorch 多层残差时序卷积，逐层下采样后做 trial 级均值读出 | 残差连接对时间特征提取的影响 | 完成本地紧凑实现的 8 轮 N/S；D6 家族补充 M=0.7111，不声称论文逐项复现 |
| InceptionTime | `deep_models`：`inception_time` | 并行短/中/长卷积核，加池化分支融合不同时间尺度 | 多尺度卷积对不同持续时间信号模式的覆盖 | 完成本地轻量实现的 8 轮 N/S；D6 家族补充 M=0.7105，不声称论文逐项复现 |
| CoCoT-EEG | `cocot.make_cocot_model`、`cocot.fit_predict_cocot`；Mahuas et al., [arXiv:2607.09543](https://arxiv.org/abs/2607.09543) | 每个电极独立做多尺度时域卷积，先提特征再用 L2 pooling 切 patch；加时间 Fourier 位置编码后，Transformer 在全部电极×时间 token 上做自注意力。读出为 1024 隐层与 2 个残差 GLU 块 | 对比卷积先于 patch 化、全局时空注意力及监督微调在目标 EEG 协议下的表现 | 以论文输入合同重采样到 100 Hz 并逐 epoch/channel median/MAD 归一化，完成 8 轮 N/S；D6 家族补充 M=0.6969、配对差 -1.62 pp（95% CI [-2.90,-0.43]）。无可用预训练权重；默认核列表和残差缩放为 bench 明示配置 |
| STMambaNet (paper-grounded) | `frontier_models.make_stmambanet` | 六路时间卷积嵌入、空间卷积、均值/方差池化和并行空间/时间状态空间分支 | 论文给出 BCIC-IV-2a/2b 实验；本实现用于可运行结构基线，不承载论文精度 | 完成 8 轮 N/S；D6 家族补充 M=0.7088。对角 SSM 递推替代官方 selective scan，非数值等价复现 |
| EEG-SimpleConv | `frontier_models.make_eeg_simpleconv` | 单层输入卷积后堆叠逐级增宽的双卷积/池化块，试次均值读出；可选被试分类辅助头 | 论文的轻量卷积架构是否作为低复杂度 MI 解码器可用 | 按作者仓库拓扑独立移植，完成 8 轮 N/S；D6 家族补充 M=0.7132；重采样由调用方预处理 |
| EEG-MFTNet | `frontier_models.make_eeg_mftnet` | 六个多尺度时间卷积分支与 Transformer 编码器按可学习权重拼接，后接深度空间卷积、可分离时间卷积和分类头 | 多尺度时间滤波与 Transformer 融合对跨会话 MI 解码的影响 | 完成 40 轮单种子 N/S；D6 家族补充 M=0.7134、配对差 -0.21 pp（95% CI [-1.17,+0.58]）；未复现作者 TensorFlow 训练流程 |
| EEGMamba | `frontier_models.make_eegmamba` | 作者模型的 temporal/spectral patch embedding + Mamba-2 encoder + 分类读出 | 预训练 EEGMamba 表征适配下游分类 | MIT 许可作者结构与许可证已纳入；patch contract 通过；Mamba-SSM 为可选 CUDA 依赖，当前 Windows 环境未能测试完整 encoder 前向 |
| EEGM2 | `frontier_models.EEGM2`、`eegm2_temporal_spectral_loss`、`eegm2_representation_summary` | 多尺度输入卷积 + 三层 U-Net 编码器/解码器 + Mamba-2 bottleneck/decoder blocks；时间 L1 与全频谱幅值重建损失 | 长序列 EEG 自监督重建与统计特征读出 | 根据论文和作者公开拓扑独立实现；U-Net/损失/读出结构 smoke test 通过，Mamba-2 可选 CUDA blocks 未在当前 Windows 环境验证；无目标协议效果 |

### Track 2 深度模型筛选记录（2026-11-09）

按 PROTEUS 两半协议 N（十人 LOSO）与 S（每 participant×session×interface 按时间前 2/3 训练、后 1/3 测试）筛选，指标 M=(N+S)/2。统一输入为 41×480、120 Hz，按训练折拟合逐通道尺度；结果只属于十名已曝光参与者的开发证据。当前 K2M2 四支路基线 M=0.7147，候选晋级门槛 M≥0.7247 且按被试配对 95% bootstrap 区间下界>0。直接替换 D6 的 EEGNeX、EEGTCNet、Deep4Net、EEG-MFTNet、EEGITNet 和本页下列候选均未达到晋级门槛。保留 D6 四分之一权重并将其与新模型均分的家族补充筛选中，最高短 pilot 点分为 SPD Token Transformer 的 8 轮结果，M=0.7163（N=0.6262，S=0.8063）；但相同模型按完整 40 轮训练后，M=0.7113（N=0.6293，S=0.7934），总体点差 -0.34 pp，按被试配对均值差 -0.94 pp、95% CI [-2.50,+0.26]。pilot 高分没有复现。EEG-MFTNet 用完整 40 轮单种子训练，家族补充 M=0.7134、配对差 -0.21 pp、95% CI [-1.17,+0.58]。CapsuleNet 用 bench 的早停接口完成 8 轮上限 pilot，深度家族补充 M=0.7014，按被试配对差 -1.52 pp、95% CI [-2.88,-0.36]；其 bench 预测接口输出硬标签，本轮融合使用 one-hot 映射。其余浅筛选、失败原因和每模型 OOF 见[Track 2 现代深度筛选记录](../../challenge2026_track02/experiments/track2_modern_deep_20261109/README.md)。

本轮目录中可直接 raw-epoch 训练且适用三分类 MI 的 Braindecode/bench 监督候选均已完成 N/S 训练或记录失败边界；完整 40 轮结果与 8/12 轮排序 pilot 分开标注，短 pilot 不作为晋级或 sealed 依据。新增 ATCNet 采用 `n_windows=2` 适配 480 点输入，完成 8 轮 N/S，家族补充 M=0.7110、配对差 -0.31 pp（95% CI [-0.81,+0.16]）。FBCNet 与 EEG-TriNet++ 的 8 轮 N/S OOF 家族补充 M 分别为 0.7108 和 0.7107；早期 CUDA FFT 报错在干净进程复跑时未重现。EEGConformer 用池化窗/步长 30/15 完成 8 轮 N/S，家族补充 M=0.7087；这是 120 Hz/480 点配置，不是论文原始 250 Hz 配方。GADCN 默认 DDCT-UNet 分解不收敛；DLDCT rank-16、width-32、eps=1e-3 配置完成 8 轮 N/S，家族补充 M=0.7105。EEGMamba2024 compact width-16/state-4 pilot 已完成，家族补充 M=0.7108；本地递推不等价官方 selective scan。CoCoT-EEG 因输入合同不同单独评估；CBraMod 使用本地官方 checkpoint 完成独立 N/S 特征读出及固定家族补充比较。2026-11-09 后续审计已证实 EEGPT 支持命名通道子集，41 路真实权重前向与前缀回放通过；LaBraM 已修正历史 wrapper 遗漏的 µV÷100，并进入同架构预训练/随机读出与有限微调对照。BIOT 支持变长输入，但当前只有其16条预训练双极导联中的4条可构造，未进入本轮；不可将10秒当作架构硬限制。

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
| 预训练基础模型 | `foundation.cbramod` | 官方 checkpoint 与 4 s、200 Hz、µV÷100 合同已用于 PROTEUS N/S 冻结特征 + 固定线性读出；replacement M=0.7003，家族补充 M=0.7073、配对差 -0.65 pp（95% CI [-1.40,+0.03]） |
| 预训练基础模型 | `foundation.biot` | 官方 TUAB 合同：200 Hz、10 s、官方 16/18 个有序双极导联；每 epoch/通道按绝对振幅 95 分位归一化；接口输出 256-D 冻结特征。当前 PROTEUS 缓存的通道名与窗长均不匹配，不纳入本轮比较 |
| 预训练基础模型 | `foundation.eegpt` | 官方 58 通道、256 Hz、4 s / 1024 点、64 点 patch；真实 checkpoint 严格加载已核验，输出 512-D 冻结特征，幅值乘数必填。当前 PROTEUS 缓存为 41 路，缺少已核验的通道补齐映射，暂不评分 |
| 待审计候选 | LaBraM | 尚未完成可复现的 tokenizer/checkpoint 输入映射和输出合同；当前 PROTEUS 输入映射未核验，不提供本轮评分 |

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

所有模型都必须使用同一外层划分和同一指标；窗口模型按试次切分，禁止同一试次的重叠窗口同时进入训练和验证/测试。结果必须注明协议与证据等级。十名公开 PROTEUS 参与者均有研发曝光，相关结果只属于开发证据。本页同合同的端到端监督模型已有 Track 2 N/S 筛选记录；SSVEP/ERP、域适配和 CSP 的目标范式/公开数据集性能仍未验证，不能据合成接口测试推断效果。

## 2024–2026 前沿架构候选

用户提供的 Mamba、Transformer、CNN、SPD 和跨被试域适应候选均有结构或方法组件入口。核验结果、来源链接、元数据差异和每项实现边界见[前沿架构清单核验](frontier_eeg_architecture_audit_20261106.md)。`EEGMamba2024` 专指 arXiv:2407.20254；`frontier_models.make_eegmamba` 专指另一篇 2025 foundation model。`GADCN` 支持 DCT/DLDCT/DDCT-UNet 变体，`PhaseSpaceSPDNet` 使用延迟轨迹协方差与 SPDNet 层。可运行结构不等于完整训练协议复现；未核验的 SOTA 或精度数字不沿用。

2026-11-06 复核修订：`GraphCSPNet`、`TensorCSPNet` 调用带非商业许可的作者模型，输入使用论文要求的分段/频带协方差张量；已针对上游两个运行错误做本地修复，但作者几何优化与运行时复数特征值转换警告仍需后续验证。`EEGTriNetPlusPlus` 包含 CNN 空间/频谱 patch + BiLSTM + Transformer + 可微 MAML episode loss；作者论文为 2026-03，非清单所写 2025-03；NAS 搜索器和论文训练协议未复现。其余已核验身份的条目按论文/作者公开拓扑实现；身份未核实的名称只提供摘要级参考架构或方法组件。当前 14 项专项测试覆盖前向、若干梯度及接口，不代表论文完整复现或目标协议性能。

## Track 2 深度模型筛选（2026-11-09）

使用 PROTEUS 十名已曝光参与者的两半 N/S 开发协议（N=LOSO，S=每 participant×session×interface 按时间前 2/3 训练、后 1/3 测试），比较目录监督模型对 K2M2 四支路融合的替换与家族补充。晋级门槛为 M=(N+S)/2 ≥0.7247，且相对 K2M2 的按被试配对 bootstrap 95% CI 下界>0。没有候选通过。候选逐项结果与覆盖边界见[实验 README](../../challenge2026_track02/experiments/track2_modern_deep_20261109/README.md)及下文广度筛选表。

完整 40 轮中，EEG-MFTNet 家族补充 M=0.7134、配对差 -0.21 pp（95% CI [-1.17,+0.58]）；SPD Token Transformer 家族补充 M=0.7113、配对差 -0.94 pp（[-2.50,+0.26]）。8 轮 pilot 中最高点为 SPD Token M=0.7163，但 40 轮复核未复现。EEGInceptionMI、EEG-SimpleConv、PhaseSpaceSPDNet 等 8 轮家族补充 M 分别为 0.7134、0.7132、0.7124，均未超过基线。CoCoT-EEG 使用单独的 100 Hz polyphase 重采样与 median/MAD 输入合同，8 轮家族补充 M=0.6969，配对差 -1.62 pp（95% CI [-2.90,-0.43]）。详细分数和失败状态见 [Track 2 modern deep model screen](../../challenge2026_track02/experiments/track2_modern_deep_20261109/README.md)。这些仅是公开参与者开发证据，不能视为 sealed 泛化结果。

补充完成的 FBCNet 与 EEG-TriNet++ 8 轮 N/S pilots 的家族补充 M 分别为 0.7108（配对差 -0.45 pp，95% CI [-1.15,+0.10]）和 0.7107（-0.60 pp，[-1.41,+0.09]），均未超过 K2M2。两者早期 CUDA FFT 失败在后续干净进程完整重跑中未重现，OOF 已纳入汇总报告。

EEGConformer 输入适配、GADCN DLDCT 配置和 EEGMamba2024 compact pilot 的家族补充 M 分别为 0.7087（配对差 -0.34 pp，95% CI [-1.00,+0.29]）、0.7105（-0.63 pp，[-1.47,+0.11]）和 0.7108（-0.52 pp，[-1.27,+0.14]）。EEGMamba2024 使用 width=16/state_size=4，完成 8 轮 N/S 用时 1,834 秒；bench 递推不等价官方 Mamba selective scan。CBraMod 使用本地官方 checkpoint 另行完成 N/S 冻结特征读出，replacement M=0.7003，family M=0.7073、配对差 -0.65 pp（95% CI [-1.40,+0.03]）。这些候选均未达到晋级门槛。

本轮建议保留 K2M2 当前 D6 深度支路。目录中可运行、输入合同匹配的监督候选已完成筛选，ATCNet 的 480 点配置为 `n_windows=2`；CBraMod 本地官方 checkpoint 也已完成独立 N/S 读出与固定家族比较，但未提升基线。EEGPT 权重本地存在，然而 58→41 通道映射未核验；BIOT 要求当前缓存不具备的双极导联名与 10 秒窗；LaBraM tokenizer/checkpoint 输入合同未完成审计。不得使用同一批 OOF 测试预测拟合融合权重并将其报告为无偏增益。

2026-11-09 后续七方向执行状态与逐臂 N/S/M、配对区间见 `challenge2026_track02/experiments/track2_next_stage_20261109/README.md`（工作区路径）。当前 E3 CSD 专家固定10%融合 M=0.7189，仍未达到0.7247门槛；不替换已提交的 K2M2。
