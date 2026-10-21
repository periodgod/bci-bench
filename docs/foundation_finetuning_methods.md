# EEG 基础模型微调方法

核验日期：2026-10-21。下表分开记录常见训练策略、已发表方法和本 bench 的实际实现。论文报告的收益只适用于各自模型与数据协议，不能直接外推到 EEG 或 PROTEUS。

| 方法 | 类别与适用场景 | 资源 / 风险提示 | 出处 | bench 状态 |
|---|---|---|---|---|
| 冻结编码器 + 线性读出（linear probe） | 小样本迁移的首选对照；只训练简单分类头，便于检验预训练表征是否可分 | 训练便宜；不能适应骨干内部表征 | 通用评估范式；结合本项目 `readout.linear_probe` | 已有线性探针与冻结 CBraMod 特征流程 |
| 部分微调（partial fine-tuning） | 只解冻最后若干层或指定层；可在数据较少时限制更新范围 | 哪些层解冻须在验证折确定；不得按测试折结果挑层 | 训练策略，不对应单一论文 | 已实现按真实模块名指定解冻；由实验配置定义层集合 |
| 全量微调（full fine-tuning） | 对骨干和任务头全部更新；适合目标数据相对充足且算力允许的对照 | 显存、过拟合与灾难性遗忘风险较高 | 经典迁移学习策略 | 已实现参数开关；训练循环由调用方提供 |
| LoRA | 在指定线性层加入低秩可训练矩阵；常用参数高效微调基线 | 需核对骨干层名、rank、缩放与注入位置 | Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” ICLR 2022。[OpenReview](https://openreview.net/forum?id=nZeVKeeFYf9)，[arXiv](https://doi.org/10.48550/arXiv.2106.09685) | PEFT adapter 已实现；本机 toy Transformer/MLP 注入测试通过 |
| QLoRA | 4-bit 量化冻结骨干 + LoRA adapter，减少显存占用 | 依赖量化库、硬件和算子兼容；量化可能改变数值行为 | Dettmers et al., NeurIPS 2023。[NeurIPS 论文集](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html)，[arXiv](https://doi.org/10.48550/arXiv.2305.14314) | 未实现；当前环境没有 `bitsandbytes`，不宣称可运行 |
| AdaLoRA | 根据重要性预算动态分配不同矩阵的低秩预算 | 需要训练过程按计划更新 rank 分配；配置注入不等于完整训练日程 | Zhang et al., ICLR 2023。[OpenReview](https://openreview.net/forum?id=lq62uWRJjiY)，[arXiv](https://doi.org/10.48550/arXiv.2303.10512) | 已实现 PEFT 配置和注入；动态预算训练日程由训练器负责，尚无端到端 recipe |
| DoRA | 将权重方向与幅值分开适配，以低秩方式更新方向并学习幅值 | 额外参数/计算与适用层有关；必须单独比较同预算 LoRA | Liu et al., ICML 2024。[PMLR 论文集](https://proceedings.mlr.press/v235/)，[arXiv](https://doi.org/10.48550/arXiv.2402.09353) | PEFT `use_dora` 配置已实现；toy 模型前向通过 |
| VeRA | 多层共享冻结随机矩阵，只学习小型缩放向量，进一步压低 adapter 参数量 | 可训练参数更少不保证 EEG 表现更好；需核验 PEFT 版本和目标模块支持 | Kopiczko et al., ICML 2024。[PMLR 论文集](https://proceedings.mlr.press/v235/)，[arXiv](https://doi.org/10.48550/arXiv.2310.10534) | 记录为候选；当前统一接口尚未加入，避免绕过专属配置合同 |
| PiSSA | 用主奇异方向初始化低秩分量，减少普通 LoRA 随机初始化带来的优化差异 | 初始化会重组权重分量；保存、合并和转换 adapter 时须按 PEFT 文档处理 | Meng et al., NeurIPS 2024。[NeurIPS 论文集](https://proceedings.neurips.cc/paper_files/paper/2024)，DOI [10.52202/079017-3846](https://doi.org/10.52202/079017-3846)，[arXiv](https://doi.org/10.48550/arXiv.2404.02948) | PEFT PiSSA 初始化已实现；toy 模型前向通过 |
| LoRA+ | LoRA 两个低秩因子使用不同学习率，改善训练效率 | 需要优化器按 A/B 因子分组；单纯挂接 LoRA 不等于 LoRA+ | Hayou et al., ICML 2024。[PMLR 论文集](https://proceedings.mlr.press/v235/)，[arXiv](https://doi.org/10.48550/arXiv.2402.12354) | 候选，尚未实现专用 optimizer 参数组 |

## Bench 接口

```python
from wyu_bci_bench.foundation import (
    attach_peft_adapter,
    configure_trainable_parameters,
    count_trainable_parameters,
)

# 冻结骨干，仅训练分类头；名称应先从 model.named_modules() 核对。
counts = configure_trainable_parameters(model, "selected", module_names=["classifier"])

# 或把 PEFT LoRA adapter 注入选定线性层。模型输入合同仍需独立核验。
model = attach_peft_adapter(model, method="lora", target_modules=["q_proj", "v_proj"])
counts = count_trainable_parameters(model)
print(counts.trainable, counts.total, counts.fraction)
```

`selected` 接受精确模块名，并将该模块的全部子模块解冻；不猜测某个模型的“最后几层”命名。`attach_peft_adapter` 当前支持 `lora`、`dora`、`pissa`、`adalora`。AdaLoRA 调用时必须传入正数 `total_steps`，供 rank-allocation schedule 使用。PEFT 属于可选环境依赖；可用 `pip install -e ".[peft]"` 安装。QLoRA 需要独立量化依赖，本工作区当前未安装 `bitsandbytes`。

微调实验先满足 `foundation.require_verified(model_name)` 对应的输入合同，再使用 bench 分组协议。归一化、特征变换、adapter 拟合、训练轮数与超参数选择均应限制在外层训练折；通过内层分组验证选参。记录实际可训练参数量、显存/耗时、随机种子和 adapter 配置，并与冻结线性探针、全量微调及随机初始化同骨干对照。公开 PROTEUS 十人结果仍是开发证据，不可称为独立测试结论。

## 出处核验说明

LoRA、AdaLoRA、QLoRA 的发表年份与会议信息分别以 ICLR/OpenReview、NeurIPS 正式论文集为准。DoRA、VeRA、LoRA+ 的正式会议记录位于 ICML 2024 PMLR 第 235 卷；PiSSA 为 NeurIPS 2024。arXiv DOI 仅用于定位预印本，会议出处才是发表状态依据。方法介绍用于确定算法机制；表中的 bench 状态只依据本仓库接口和测试，不意味着已在 EEG 数据上训练或验证有效。
