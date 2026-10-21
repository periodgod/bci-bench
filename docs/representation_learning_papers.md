# 表征学习论文与 bench 实现

核验日期：2026-10-21。会议级别按中国计算机学会《推荐国际学术会议和期刊目录》常用分类记录；CCF 分类说明会议影响力，不代表某种方法必然适合本数据或会提高分数。只纳入正式会议论文集中的已发表工作；arXiv 预印本不单独算作发表。

| 论文 | 已发表出处 | 方法要点 | bench 状态 |
|---|---|---|---|
| Zhihan Yue et al., “TS2Vec: Towards Universal Representation of Time Series” | AAAI 2022, *Proceedings of the AAAI Conference on Artificial Intelligence*, 36(8), 8980–8987. CCF-A。DOI: [10.1609/aaai.v36i8.20881](https://doi.org/10.1609/aaai.v36i8.20881)。[AAAI 正式版](https://ojs.aaai.org/index.php/AAAI/article/view/20881)，[作者代码](https://github.com/yuezhihan/ts2vec)。 | 对两个增强视图逐层时间池化，并在时间点与实例两个层面进行对比学习。 | **已实现核心目标的 EEG 适配版**：`TS2VecModel`、`hierarchical_contrastive_loss`、`train_ts2vec`。编码器和增强策略不是作者代码的逐项复刻。 |
| Navid Mohammadi Foumani et al., “EEG2Rep: Enhancing Self-supervised EEG Representation Through Informative Masked Inputs” | ACM SIGKDD 2024, *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, 5544–5555。CCF-A。DOI: [10.1145/3637528.3671600](https://doi.org/10.1145/3637528.3671600)，[ACM 正式版](https://dl.acm.org/doi/10.1145/3637528.3671600)，[作者代码](https://github.com/Navidfoumani/EEG2Rep)。 | Semantic Subsequence Preserving（SSP）保留语义子序列；预测目标在潜在表征空间，而非直接重建被遮挡的原始 EEG。作者报告在其六项任务中保留 50% 记录的平均表现最好。 | **待实现**。先审阅作者模型、SSP 掩码和预训练数据划分；不能仅把通用随机遮挡重建称作 EEG2Rep。 |
| “Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI” (LaBraM) | ICLR 2024 正式会议论文，顶级机器学习会议。[OpenReview 论文记录](https://openreview.net/forum?id=QzTpTRVtrP)，[作者代码](https://github.com/935963004/LaBraM)。 | 将 EEG 切成通道 patch，以向量量化神经频谱 tokenizer 产生离散目标，再做 masked neural-code prediction；论文报告约 2,500 小时、约 20 个数据集的预训练。 | **未集成**。CBraMod 与 EEGPT 已有适配；LaBraM 的通道顺序、patch 化、tokenizer/checkpoint 与数据合同均未审计。 |
| “EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals” | NeurIPS 2024 正式会议论文。[作者代码](https://github.com/BINE022/EEGPT)。 | 以掩码式双重自监督学习跨通道与时间表征；公开大模型规格为 58 通道、256 Hz、4 秒窗口、64 点 patch。 | **结构适配已实现**：官方 encoder 可严格加载并输出 16×4×512 tokens，bench 提供显式幅值乘数与 512-D 冻结特征；checkpoint 来源认证、目标数据的尺度与效果仍未核验。 |
| Chaoqi Yang et al., “BIOT: Biosignal Transformer for Cross-data Learning in the Wild” | NeurIPS 2023, *Advances in Neural Information Processing Systems 36*。CCF-A。DOI: [10.52202/075280-3420](https://doi.org/10.52202/075280-3420)，[NeurIPS 正式论文页](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract-Conference.html)，[作者代码](https://github.com/ycq091044/BIOT)。 | 面向多种生理信号及不同通道/采样设置的 Transformer；将通道信号划为 patch 并加入通道身份编码，目标是跨数据集建模。 | **未集成**。可作为统一多数据源编码器候选；接入前需审计采样率、patch 与通道映射，不等于已经缓解域偏移。 |

## 实现说明与边界

`ssl.TS2VecModel` 输出 `(batch, time, embedding)`。`ssl.hierarchical_contrastive_loss` 在两个增强视图上，分别计算同一窗口内跨时间点的 temporal contrast、同一时间点跨窗口的 instance contrast，并逐层时间池化；`ssl.train_ts2vec` 用现有被试均衡采样器和保持时间位置的增广训练。下游窗口向量由时间均值池化得到。

这些方法是候选算法依据，不是已经验证有效的研究结果。TS2Vec 风格实现与作者完整配置存在差异；EEG2Rep、LaBraM、BIOT 只登记了已发表论文与候选关系，尚未声称代码实现。EEGPT 有 encoder-only 结构适配，但其 checkpoint 来源、PROTEUS 通道/幅值映射及目标任务效果未验证。任何新结果都必须在外层训练折中拟合编码器和读出器，并与同一划分、同一种子预算的随机初始化及现有 SimCLR/VICReg 基线配对比较；PROTEUS 十人仍仅属于开发证据。不得根据训练损失下降声称跨被试域偏移得到缓解。

## 来源核验

- KDD / AAAI / NeurIPS 的出版信息与 DOI 由 Crossref DOI 元数据核对；CCF-A 标注依据 [CCF 推荐国际学术会议和期刊目录](https://www.ccf.org.cn/Academic_Evaluation/By_category/)中对应会议类别。
- LaBraM 的 ICLR 2024 发表状态和方法描述由作者仓库 README 与 OpenReview 论文记录核对；作者仓库明确要求加载预训练权重并提供输入通道顺序，故未把它当作可直接替换的通用 EEG 编码器。
- EEG2Rep 的方法概述由论文摘要和作者仓库 README 核对。性能数字属于论文作者在其数据与协议下的报告，不是 bench 结果。
