# 2024–2026 EEG 架构清单核验

核验日期：2026-11-07。此页只记录可追溯的论文/代码元数据，不代表模型已接入、复现或在 bench 协议下评估。文献存在、代码可访问、架构可运行和目标数据效果是四种不同证据。

| 用户清单名称 | 核验结果 | 证据与差异 | Bench 状态 |
|---|---|---|---|
| STMambaNet | 已核实论文；bench 初版结构已接入 | [arXiv:2409.09627](https://arxiv.org/abs/2409.09627)，题名 *Spatial-Temporal Mamba Network for EEG-based Motor Imagery Classification*。原文描述六路时间卷积、空间卷积、均值/方差池化及空间/时间 Mamba 编码器。 | `frontier_models.make_stmambanet`；结构和梯度测试通过。由于缺少作者代码，卷积宽度、核宽和池化参数为 bench 配置；SSM 内核使用 PyTorch 对角递推而非官方 Mamba selective scan，不宣称等价复现；未评估目标协议效果 |
| EEGMamba | 清单项论文已核实；另有同名 2025 foundation model | 清单项为 [arXiv:2407.20254](https://arxiv.org/abs/2407.20254)，摘要明确包含 ST-Adaptive、双向 Mamba、task-aware MoE 和 universal expert；作者摘要称代码将发布。官方 [wjq-learning/EEGMamba](https://github.com/wjq-learning/EEGMamba) 则是不同的 2025 *Neural Networks* foundation model。 | 新增 `EEGMamba2024`，独立重建空间/时间自适应卷积、class token、双向输入依赖对角 SSM、task 条件稀疏 MoE 和 universal expert；结构/梯度测试通过。状态递推是 PyTorch 参考实现，不等价原始 Mamba selective scan；未复现论文全训练流程或目标协议效果。旧 `make_eegmamba` 仍指 2025 foundation model；其作者 patch embedding 和 Mamba2 CUDA 编码器依赖另列。 |
| MI-Mamba | 论文/摘要已核实，架构近似实现 | PMID [39844431](https://pubmed.ncbi.nlm.nih.gov/39844431/)，DOI `10.1111/nyas.15288`；摘要描述卷积提取局部空间特征、Mamba 建模全局时间依赖。清单月份/精度须以论文最终版本及表格为准。 | `frontier_advanced.MIMamba` 实现卷积前端、PyTorch selective-state 递推与分类头；形状/梯度测试通过。未核实完整超参和论文训练流程；未评估目标协议效果。 |
| EEGM2 | 论文与代码已核实，清单描述不符；架构按论文独立复现 | [arXiv:2502.17873](https://arxiv.org/abs/2502.17873)，*An Efficient Self-Supervised Framework for Long-Sequence EEG Modeling*；[Emotiv/EEGM2-ICDM2025](https://github.com/Emotiv/EEGM2-ICDM2025) README 说明发表于 AI4TS Workshop @ ICDM 2025，报告 TUAB 与多 Emotiv 任务，不是“6 数据集 SOTA”。仓库未见许可证文件。 | `frontier_models.EEGM2` 按论文 U-shaped encoder-decoder、Mamba2 residual blocks、插值上采样/skip-connections 实现；配有 temporal-spectral loss 与 9-statistic representation summary；结构 smoke test 以注入 Identity SSM 验证，CUDA Mamba2 encoder 未验证 |
| CFSPMNet | 已核实，年份错误 | [arXiv:2605.10111](https://arxiv.org/abs/2605.10111) 与 [作者仓库](https://github.com/wxk1224/CFSPMNet) 一致，工作为 2026 年。仓库有模型源码但未见许可证文件，故不能直接复制其代码。 | `frontier_advanced.CFSPMNet` 按论文/仓库可见模块独立实现多尺度生理 tokenizer、Fourier rhythmic modeling、selective-state mixer 与共享/私有原型伪标签筛选；14 项专项测试包含前向与有限梯度检查。作者完整训练/在线更新细节尚未核实，不能称完整复现；未评估目标协议。 |
| MGEC | 未核实 | 未找到能对应清单所述“互引导双专家、域特异/域不变”的 EEG 论文标识。 | `DualExpertTransformer` 为描述级互引导双分支参考结构，未确认对应原论文；未评估目标协议。 |
| PTSM | 论文已核实；实现覆盖主要训练结构 | [arXiv:2508.11357](https://arxiv.org/abs/2508.11357)，题名 *Physiology-aware and Task-invariant Spatio-temporal Modeling for Cross-Subject EEG Decoding*。用户清单中的月份为 2025-08，而不是 2025-08 后的其他日期。 | `PTSM` 实现个体/共享时空掩码凸组合、任务/被试投影和分类头；`ptsm_objective` 含掩码正则、任务分类、特征去相关/正交及监督对比项。结构与有限梯度测试通过；损失权重及完整 zero-shot 训练程序未逐项复现。 |
| EEG-TriNet++ | 论文已核实，清单年份错 | DOI [10.3390/bioengineering13030307](https://doi.org/10.3390/bioengineering13030307)，MDPI *Bioengineering* 13(3), 307，2026-03-06。论文摘要列明 spatial-spectral CNN、BiLSTM、Transformer、patch tokenization、NAS 与 MAML；Crossref 许可为 CC BY 4.0。 | `EEGTriNetPlusPlus` 有 spatial-spectral patch CNN + BiLSTM + Transformer；`maml_episode_loss` 支持可微 inner update。NAS、论文完整多分支与训练配置尚未复现；未测论文协议。 |
| SPD-Token Transformer | 论文已核实；论文身份与清单吻合 | [arXiv:2601.21521](https://arxiv.org/abs/2601.21521)，题名 *A Unified SPD Token Transformer Framework for EEG Classification: Systematic Comparison of Geometric Embeddings*，2026-01-29。摘要比较 BWSPD、Log-Euclidean、Euclidean embedding 与 BN-Embed。 | `SPDTokenTransformer` 支持三种 embedding、可选 BN-Embed、raw epoch 分窗协方差和预计算多带 SPD tokens；结构/输入梯度测试通过。论文大规模比较、精确优化细节和报告精度未复现。 |
| EEG-MFTNet | 论文及作者代码已核实；bench 已按结构独立实现 | [arXiv:2604.05843](https://arxiv.org/abs/2604.05843)；作者仓库 [pandrikopoul/EEG-MFTNet](https://github.com/pandrikopoul/EEG-MFTNet) 提供 Keras 网络定义，但未声明许可证。 | `frontier_models.make_eeg_mftnet`；保留多尺度 temporal branches、Transformer residual、learnable weighted concatenation、depthwise spatial conv、separable conv 和分类头，转为 PyTorch；未迁移训练/数据脚本；目标协议效果未验证 |
| EEG-SimpleConv | 作者代码仓库已定位；清单出版元数据待补 | 作者仓库 [elouayas/EEGSimpleConv](https://github.com/elouayas/EEGSimpleConv) 含 `models/EEGSimpleConv.py`，未声明许可证。清单所写 arXiv/NeurIPS 评审状态未据仓库确认。 | `frontier_models.make_eeg_simpleconv` 独立按网络拓扑移植；卷积/输出结构测试通过；原始重采样前处理需由调用方处理；目标协议效果未验证 |
| AGTCNet | 论文与作者仓库已定位；代码许可限制 | DOI [10.1109/ACCESS.2025.3627419](https://doi.org/10.1109/ACCESS.2025.3627419)，作者仓库 [galvinlim/AGTCNet](https://github.com/galvinlim/AGTCNet)。仓库 `LICENSE.md` 为 CC BY-NC-ND 4.0，不直接复制或改编其源文件。 | `AGTCNet` 现有动态 channel graph attention、图信号混合和多尺度 temporal branches 参考实现，专项前向/梯度测试通过；尚未从论文逐模块核验全部设计，不是作者源码迁移或完整论文复现。 |
| Graph-CSPNet | 论文与作者实现已定位 | 与 Tensor-CSPNet 共用作者仓库 [GeometricBCI/Tensor-CSPNet-and-Graph-CSPNet](https://github.com/GeometricBCI/Tensor-CSPNet-and-Graph-CSPNet)，仓库许可允许非商业用途下复制/修改并要求保留声明；模型是 Graph BiMap + ReEig/BatchNormSPD/LogEig。 | `GraphCSPNet` 使用 `(B, segments, C, C)` 协方差张量、图传播和多层 BiMap/谱映射；切空间归一化近似作者 Karcher 均值，未复现作者 Riemannian optimizer。合成结构测试通过。 |
| Tensor-CSPNet | 论文与作者实现已定位 | DOI [10.1109/TNNLS.2022.3172108](https://doi.org/10.1109/TNNLS.2022.3172108)，发表于 TNNLS 2023；作者仓库同上。源码说明输入为 `(B, windows, bands, C, C)`，BiMap 几何模块后接时间 CNN。 | `TensorCSPNet` 使用多频带/时间窗协方差、BiMap/ReEig/LogEig 和时间卷积头；归一化/优化器差异同 Graph-CSPNet。合成结构测试通过。 |
| GADCN | 论文已核实；用户描述不完整 | [arXiv:2511.18940](https://arxiv.org/abs/2511.18940)，题名 *Geometry-Aware Deep Congruence Networks for Manifold Learning in Cross-Subject Motor Imagery*。摘要明确列出 DCT、DLDCT、DDCT-UNet，并报告 transductive cross-subject 实验；“深度一致性”并非其名称。 | `GADCN` 已改为 DCT congruence block，并提供 `dct`、`dldct`、`ddct_unet` 三种深度；支持从 raw epoch 构造协方差、SPD eigenvalue rectification、DDCT skip fusion 与 Fisher-style classifier head。形状测试通过；自定义 logistic-regression objective、transductive 协议及完整论文超参未实现。 |
| SPDNet + 相空间增强 | 论文已核实；实现按公开代码流程独立重建 | [arXiv:2403.05645](https://arxiv.org/abs/2403.05645)，DOI `10.1088/1741-2551/ad88a2`，题名 *Geometric Neural Network based on Phase Space for BCI-EEG decoding*。论文摘要明确使用 Augmented Covariance Method 与 SPDNet；作者仓库没有声明许可证，因此未复制其源码。 | `PhaseSpaceSPDNet` 以 order/lag 延迟轨迹生成相空间协方差，再执行 BiMap、ReEig、可选 embedding BN、LogEig 上三角向量化及分类。亦可接收预计算协方差；结构/梯度测试通过。MOABB 完整 scaler/CV/优化器协议未复现。 |
| CS-Divergence FM | 论文已核实；清单名称为缩写 | [arXiv:2507.21037](https://arxiv.org/abs/2507.21037)，题名 *Brain foundation model-guided source-selective domain adaptation for cross-subject EEG decoding*，2025-07-28，方法名 BFM-MSDA。摘要确认 foundation-model 引导来源检索、相关性加权 CS 边缘对齐与 CCS 条件决策对齐。 | `BFMMSDA` 含 top-k 源域选择、来源加权双对齐、warm-up 与 source classification loss；专项 loss/梯度测试通过。论文预训练 foundation embedding、完整源域构造和目标数据性能未复现。 |
| Subject-Aligned | 同名论文已核实，清单月份错误 | DOI [10.1038/s41598-026-69220-z](https://doi.org/10.1038/s41598-026-69220-z)，*Scientific Reports*，2026-09-03，作者为 Lopes 等。摘要给出 subject encoder bank、共享分类器及低秩新被试校准；仍需全文核实所列方法是否同一工作。 | `SubjectAlignedEncoder` 使用 subject-specific encoder bank、共享分类头与低秩通道 adapter，并要求新被试 subject ID；结构测试通过，论文训练/校准流程未复现。 |

## 接入准则

## 2026-11-06 描述级实现状态

未完成论文级接入的候选仍有描述级临时入口；映射与缺失模块以本页表格“Bench 状态”为准。Graph-CSPNet/Tensor-CSPNet 与 EEG-TriNet++ 已更新为更贴近论文结构的实现，但未完成作者训练流程、精确几何优化或论文协议复现。其余架构入口不得作为已完成迁移的证据。

入口位于 `wyu_bci_bench.frontier_advanced`，从公共包导出；覆盖形状与损失梯度的合成测试通过与否以本次交接日志为准。论文身份不明确的条目仍需补原文/作者实现核查，之后才能把这些参考结构升级为论文复现。

- 只有核实论文身份、原文结构、代码来源和许可证后，才把条目作为可复现架构登记。
- MIT 许可的 EEGMamba 源码可以在保留版权与许可证的前提下评估移植；依赖 Mamba CUDA 扩展的部分还须验证本工作区 Windows/PyTorch 环境能否运行。
- EEGM2 与 CFSPMNet 仓库未见许可证文件；在取得许可前不复制其实现。可按论文独立实现，但必须标明复现假设，并用作者给出的输入/结构/训练设置逐项核验。
- 本页的论文与代码检索不验证用户清单里的准确率、SOTA、模型大小或推理速度。所有性能主张都需要原文表格和明确协议。
