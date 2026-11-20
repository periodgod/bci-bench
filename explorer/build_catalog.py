"""Scan the wyu-bci-bench source tree (AST only, nothing imported) into a catalog dict.

    python build_catalog.py <bench_root> [out.json]

Used by explorer_app.py at start-up so the explorer always shows the code as it
is on disk; the JSON form is also bundled into the exe as an offline snapshot.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ANIMATION_ROOT = Path(r"E:\世界机器人大赛\projects\脑电大模型\animations")

LAYERS = [
    ("ingest", "数据接入", "读取 NPZ / 连续采集 / BIDS 原始数据，形成带分组键的窗口数据"),
    ("signal", "信号处理", "CAR、滤波、PSD、通道选择等无学习的预处理"),
    ("features", "特征工程", "频带功率、Hjorth、空间滤波、数学先验等固定特征"),
    ("classic", "经典机器学习", "LDA、黎曼、MiniRocket、XGBoost、sklearn 多模型"),
    ("deep", "深度模型与自监督", "EEGNet、CapsuleNet、M1/M2、多种经典 EEG 深度网络与自监督表征"),
    ("foundation", "基础模型适配", "预训练模型输入合同、冻结读出与参数高效微调"),
    ("protocol", "划分与评估", "LOSO / 跨会话 / 8:1:1 划分、交叉验证、线性读出"),
    ("stats", "指标与统计", "cell 等权 BA、流式 BA、配对 bootstrap、sign-flip"),
    ("results", "结果与提交", "OOF 保存与读取、竞赛 ZIP 打包与检查"),
    ("interface", "可视化与命令行", "画图与 wyu-bci-bench 命令行入口"),
]

MODULES = {
    "data": ("ingest", "v0.1", "epoch NPZ 读取、标签映射、多文件合并"),
    "continuous": ("ingest", "v0.1", "机械手 raw_stream 连续采集：逐文件预处理、坏道 QC、切 epoch"),
    "grouped": ("ingest", "v0.2", "GroupedEEGDataset：带 subject/session/interface 的窗口数据"),
    "proteus_bids": ("ingest", "v0.2", "PROTEUS BIDS 原始 EDF → (N,41,480) 窗口缓存"),
    "signal": ("signal", "v0.1", "CAR、FFT 滤波、PSD、频带能量、通道选择"),
    "features": ("features", "v0.1", "log-bandpower、固定空间滤波、FBCSP 风格 log-variance、Fisher 筛选"),
    "csp": ("classic", "v0.2", "MNE Common Spatial Pattern 多频带投影 + 标准化线性 SVM"),
    "window_features": ("features", "v0.2", "Welch 频带功率 / 相对功率 / Hjorth 变体（pre_post_car 等）"),
    "priors": ("features", "v0.2", "1394 维数学先验特征（M0/M1/M2 共用）"),
    "models": ("classic", "v0.1", "对角 LDA、最近质心、高斯 NB、KNN、filter-bank LDA"),
    "riemann": ("classic", "v0.1", "协方差、黎曼均值、MDM、切空间 + LDA"),
    "minirocket": ("classic", "v0.1", "MiniRocket 风格固定卷积核特征"),
    "treeboost": ("classic", "v0.1", "XGBoost EEG 特征树模型"),
    "sklearn_ml": ("classic", "v0.1", "sklearn 多机器学习算法组"),
    "deep": ("deep", "v0.1", "可选 PyTorch EEGNet"),
    "deep_models": ("deep", "v0.2", "12 种 EEG 深度网络：ShallowFBCSPNet、Deep4Net、EEGConformer、FBCNet、ATCNet、TCN、EEGTCNet、EEGInceptionMI、EEGNeX、EEGITNet、EEG-ResNet、InceptionTime；前十种使用 Braindecode，后两种为本地轻量实现"),
    "cocot": ("deep", "v0.2", "CoCoT-EEG：多尺度卷积先于 L2 patch 化，再接 Transformer；含折内监督微调入口，预训练权重需调用方提供"),
    "ssvep": ("classic", "v0.2", "SSVEP 专用 CCA、FBCCA、TRCA/eTRCA 与 TDCA 解码；频率显式映射到类别"),
    "erp": ("classic", "v0.2", "epoch 级 xDAWN 空间滤波与收缩 LDA 的 ERP/P300 基线"),
    "adaptation": ("protocol", "v0.2", "Euclidean Alignment、黎曼对齐、CORAL、MMD、DANN 与 AdaBN；使用目标域统计时属于传导式适配"),
    "interpretability": ("protocol", "v0.2", "留出特征组置换重要性、通道遮蔽与频带消融；用于解释预测依赖，不是因果效应"),
    "calibration": ("protocol", "v0.2", "验证折拟合温度缩放、测试折 ECE/Brier 评估，以及按置信度计算风险-覆盖率"),
    "capsule": ("deep", "v0.1", "轻量 CapsuleNet 投票成员"),
    "nets": ("deep", "v0.2", "PROTEUS M1/M2：数学先验 + 多尺度残差网络及训练"),
    "ssl": ("deep", "v0.2", "SimCLR / VICReg / SupCon / TS2Vec；关联 KDD 2024 EEG2Rep、ICLR 2024 LaBraM、NeurIPS 2023 BIOT 候选"),
    "foundation": ("foundation", "v0.2", "基础模型输入合同与微调策略；未审计合同拒绝调用"),
    "foundation.cbramod": ("foundation", "v0.2", "CBraMod 4 s / 200 Hz / µV÷100 冻结特征"),
    "foundation.finetuning": ("foundation", "v0.2", "全量、冻结、指定模块微调；PEFT LoRA / DoRA / PiSSA / AdaLoRA"),
    "splits": ("protocol", "v0.2", "LOSO、内层 LOSO、跨会话、NeuralBench sealed、OmniEEG 8:1:1"),
    "evaluate": ("protocol", "v0.1", "分层 K 折交叉验证、投票集成、质量报告"),
    "readout": ("protocol", "v0.2", "线性探针、LOSO 探针、嵌套 LOSO 选 C"),
    "metrics": ("stats", "v0.2", "cell / 流式平衡准确率、配对 bootstrap、sign-flip"),
    "registry": ("results", "v0.2", "统一 OOF + report.json 格式，兼容历史文件"),
    "submission": ("results", "v0.2", "Codabench ZIP 打包与静态检查"),
    "viz": ("interface", "v0.1", "PSD、频带热图、地形图、标签分布 SVG"),
    "cli": ("interface", "v0.1+v0.2", "scan / analyze / bench / oof-score / oof-compare"),
}

# These public classes are the beginner-facing entry points shown in the model
# browser. Their identifiers stay unchanged for code navigation, while the
# Chinese gloss explains the modeling idea instead of translating the name only.
SYMBOLS_ZH = {
    ("models", "NearestCentroidClassifier"): "先对特征标准化，再计算每个类别的中心；新样本归到距离最近的类别中心。",
    ("models", "DiagonalGaussianNB"): "为每个类别、每个特征拟合高斯分布，并假设特征在给定类别后彼此独立。",
    ("models", "KNearestNeighborsClassifier"): "查找特征空间中最近的 k 个训练样本并投票；票数相同时按距离和类别顺序稳定裁决。",
    ("models", "StandardizedDiagonalLDA"): "先标准化特征，再用各类别均值和共享的对角协方差进行线性判别。",
    ("models", "FilterBankLDAModel"): "运动想象基线：提取多频带空间对数方差，用 Fisher 分数筛选特征，再用对角 LDA 分类。",
    ("riemann", "FilterBankTangentSpaceLDAModel"): "在多个频带计算协方差切空间特征，进行有监督特征筛选后用对角 LDA 分类。",
    ("ssl", "TS2VecModel"): "TS2Vec（AAAI 2022）风格逐时间点编码器；bench 使用轻量 EEG 适配结构，不是作者模型逐层复刻。",
    ("ssl", "hierarchical_contrastive_loss"): "TS2Vec（AAAI 2022）分层时间对比目标的 EEG 适配实现：跨视图比较时间点与样本，并逐层池化。",
    ("ssl", "train_ts2vec"): "使用无标签 EEG 训练 TS2Vec 风格编码器；论文依据与和原论文配置的差异见表征学习论文清单。",
    ("deep_models", "available_models"): "列出可构建的十种 Braindecode EEG 深度网络：ShallowFBCSPNet、Deep4Net、EEGConformer、FBCNet、ATCNet、TCN、EEGTCNet、EEGInceptionMI、EEGNeX、EEGITNet。",
    ("deep_models", "make_braindecode_model"): "按照通道数、采样点数、类别数和采样率创建指定的 Braindecode EEG 网络；输入 epoch 形状为（批次、通道、采样点）。",
    ("deep_models", "fit_predict_braindecode"): "在外层训练折内部按试次再划分训练集和验证集，训练所选网络，并把一个试次多个窗口的概率平均成试次级概率。",
    ("cocot", "make_cocot_model"): "创建 CoCoT-EEG 多尺度卷积 tokenizer、跨通道与时间 Transformer、残差 GLU 读出器；卷积核列表可显式配置。",
    ("cocot", "fit_predict_cocot"): "按 CoCoT-EEG 的 100 Hz 与 median/MAD 合同预处理，在外层训练折内按 trial 留出验证集并监督微调；可选严格 encoder 权重和 Mixup。",
    ("cocot", "resample_epochs"): "用 polyphase 重采样把（trial，channel，time）数据转换到 CoCoT-EEG 论文采用的 100 Hz。",
    ("csp", "FilterBankCSP"): "真正的滤波器组 CSP：每个预设频带只用训练 epoch 学空间投影，提取对数平均功率并拼接，再由标准化线性 SVM 分类；它与固定投影的 FBCSP 风格特征不同。",
    ("ssl", "masked_reconstruction_loss"): "掩码重建目标只在被遮挡的 EEG 位置计算均方误差，避免未遮挡输入位置主导训练损失。",
    ("ssvep", "EnsembleTRCA"): "集成 TRCA（eTRCA）：在 MOABB 的 TRCA 中启用跨类别空间滤波器集成；调用时需按排序后的类别提供各自真实刺激频率。",
    ("ssvep", "TDCA"): "任务判别成分分析：通过时间延迟扩展 EEG，在类间/类内散度准则下学习共享判别空间；由 MOABB 实现核心算法。",
    ("adaptation", "RiemannianAlignment"): "用每个域的协方差黎曼几何均值估计白化矩阵，使域内几何均值映射到单位阵；目标域输入访问需符合传导式协议。",
    ("adaptation", "maximum_mean_discrepancy"): "多核 RBF MMD²：衡量源域与目标域特征分布的差异，可与任务损失联合训练；本身不提供编码器或训练循环。",
    ("adaptation", "DomainAdversarialHead"): "DANN 的域分类头通过梯度反转层训练特征提取器，使其难以识别被试/域；任务分类头和优化循环由调用方提供。",
    ("adaptation", "adapt_batch_norm"): "仅用无标签目标批次更新 BatchNorm 运行均值/方差；其他层保持评估状态，不读取目标标签。使用测试批次统计时需声明测试时适配。",
    ("interpretability", "grouped_permutation_importance"): "在留出特征上联合置换一组列，测量预测指标下降；重要性是模型依赖诊断，不是因果效应。",
    ("interpretability", "channel_ablation_scores"): "将留出 epoch 的指定通道置零并记录指标变化；模型不重训，因此反映输入敏感度而非最优通道子集。",
    ("interpretability", "band_ablation_scores"): "对留出 EEG 指定频带做带阻后记录分数变化；用于检查模型频带依赖，不代表频带具有因果作用。",
    ("calibration", "fit_temperature"): "只用训练流程中的验证折 logits 拟合正温度；最终校准质量必须在独立测试折评估。",
    ("calibration", "expected_calibration_error"): "按最大类别概率分箱，计算置信度与实际正确率的加权绝对差。",
    ("calibration", "multiclass_brier_score"): "计算多分类概率与 one-hot 标签之间的均方概率误差。",
    ("calibration", "risk_coverage_curve"): "按置信度从高到低逐步接纳预测，报告接受比例与错误率的关系，用于评估拒识策略。",
    ("ssvep", "SSVEPCCA"): "为每个刺激频率生成正弦/余弦参考波，用典型相关系数比较 EEG 与参考波，再选择得分最高的频率类别。",
    ("ssvep", "FilterBankCCA"): "先把 EEG 分成多个频带，分别计算 CCA 相关系数平方，再按频带权重汇总，形成 FBCCA 风格分数。",
    ("ssvep", "TRCA"): "从同类别重复试次学习能让试次间活动最同步的空间滤波器，并将测试试次与类别模板比较。",
    ("ssvep", "reference_signals"): "按采样率、刺激频率和谐波数构造 SSVEP 正弦/余弦参考信号。",
    ("erp", "XDawnLDA"): "从训练折目标 ERP 学习增强诱发响应的空间滤波器，再将滤波后的 epoch 交给收缩 LDA 分类。",
    ("adaptation", "EuclideanAlignment"): "逐域用无标签试次协方差估计白化矩阵；若统计量来自测试域，评估协议必须允许测试时校准。",
    ("adaptation", "euclidean_align"): "用当前域无标签 EEG 的平均协方差白化各通道，常用于跨被试/跨会话输入对齐。",
    ("adaptation", "coral_align"): "用无标签目标特征的均值和协方差把源域特征变换到目标域统计分布；不使用目标标签。",
    ("foundation.finetuning", "configure_trainable_parameters"): "按全量、冻结或明确指定的模块控制哪些参数参与梯度更新；指定名称来自模型真实模块树。",
    ("foundation.finetuning", "attach_peft_adapter"): "用 PEFT 为已核验输入的骨干注入 LoRA、DoRA、PiSSA 或 AdaLoRA；目标层名称必须逐模型核对。",
    ("foundation.finetuning", "count_trainable_parameters"): "统计总参数量与可训练参数量，用于比较不同微调策略的参数开销。",
}

PROJECTS = {
    "v0.1": ["机械手 BCI"],
    "v0.2": ["毕业论文", "NeurIPS 赛题", "OmniEEG 赛题"],
    "v0.1+v0.2": ["机械手 BCI", "毕业论文", "NeurIPS 赛题", "OmniEEG 赛题"],
}
HEAVY = {"torch": "PyTorch", "scipy": "SciPy", "sklearn": "scikit-learn", "xgboost": "XGBoost",
         "mne": "MNE", "pandas": "pandas", "matplotlib": "matplotlib"}


def _sig(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        args = ast.unparse(node.args)
    except Exception:  # pragma: no cover
        args = "..."
    ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"({args}){ret}"


def _doc(node) -> str:
    return (ast.get_docstring(node) or "").strip()


def _first(doc: str) -> str:
    return doc.split("\n\n")[0].replace("\n", " ").strip()


def _symbols(body, lines, owner=None, nested=False):
    out = []
    for node in body:
        if isinstance(node, ast.If):  # `if nn is not None:` guarded torch classes
            out += _symbols(node.body, lines, owner, nested=True)
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            end = getattr(node, "end_lineno", node.lineno)
            item = {
                "name": node.name,
                "kind": "class" if isinstance(node, ast.ClassDef) else ("method" if owner else "function"),
                "line": node.lineno, "end": end, "lines": end - node.lineno + 1,
                "public": not node.name.startswith("_"),
                "doc": _doc(node), "summary": _first(_doc(node)),
                "source": "".join(lines[node.lineno - 1 - len(node.decorator_list):end]),
                "torch_guarded": nested,
            }
            if isinstance(node, ast.ClassDef):
                item["bases"] = [ast.unparse(b) for b in node.bases]
                item["signature"] = f"({', '.join(item['bases'])})" if item["bases"] else ""
                item["methods"] = [m for m in _symbols(node.body, lines, owner=node.name)
                                   if m["kind"] == "method" and (m["public"] or m["name"] == "__init__")]
                for m in item["methods"]:
                    m.pop("source", None)
            else:
                item["signature"] = _sig(node)
            out.append(item)
    return out


def scan_module(path: Path, pkg_root: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    rel = path.relative_to(pkg_root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    name = ".".join(parts) or "__init__"
    internal, external = set(), set()
    package = parts if path.name == "__init__.py" else parts[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                if node.module:
                    internal.add(".".join([*base, node.module]))
                else:
                    internal.update(".".join([*base, a.name]) for a in node.names)
            elif node.module:
                external.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for a in node.names:
                external.add(a.name.split(".")[0])
    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for sub in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                if isinstance(sub, ast.Import):
                    guarded.update(a.name.split(".")[0] for a in sub.names)
                elif isinstance(sub, ast.ImportFrom) and sub.module and not sub.level:
                    guarded.add(sub.module.split(".")[0])
    layer, version, desc = MODULES.get(name, ("other", "?", ""))
    symbols = _symbols(tree.body, lines)
    for symbol in symbols:
        symbol["summary_zh"] = SYMBOLS_ZH.get((name, symbol["name"]), "")
    return {
        "name": name, "file": str(path.relative_to(pkg_root.parent.parent)).replace("\\", "/"),
        "layer": layer, "version": version, "projects": PROJECTS.get(version, []), "desc": desc,
        "doc": _doc(tree), "summary": _first(_doc(tree)), "loc": len(lines),
        "symbols": symbols,
        "imports_internal": sorted(i for i in internal if i and i != name),
        "requires": sorted(HEAVY[e] for e in external if e in HEAVY),
        "optional": sorted(HEAVY[e] for e in guarded if e in HEAVY),
        "source": text,
    }


def scan_scripts(root: Path) -> list[dict]:
    out = []
    for folder, kind in (("tests", "测试"), ("scripts", "核验脚本"), ("examples", "示例"), ("explorer", "浏览器")):
        for p in sorted((root / folder).glob("*.py")):
            text = p.read_text(encoding="utf-8")
            tree = ast.parse(text)
            tests = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
            out.append({"name": p.name, "file": f"{folder}/{p.name}", "kind": kind, "loc": len(text.splitlines()),
                        "summary": _first(_doc(tree)), "tests": tests, "source": text})
    return out


def parse_migration(doc: Path) -> list[dict]:
    if not doc.exists():
        return []
    rows, on = [], False
    for line in doc.read_text(encoding="utf-8").splitlines():
        if line.startswith("## 旧函数"):
            on = True
            continue
        if on and line.startswith("## "):
            break
        m = re.match(r"^\| (.+?) \| (.+?) \|$", line) if on else None
        if m and not set(m.group(1)) <= set("-| ") and not m.group(1).startswith("旧实现"):
            rows.append({"old": m.group(1), "new": m.group(2)})
    return rows


def attach_capabilities(modules: list[dict], path: Path) -> dict:
    """Load semantic method/purpose tags and bind them to real module symbols."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"unsupported capability schema in {path}")
    dimensions = data.get("dimensions", {})
    if set(dimensions) != {"stage", "method", "goal"}:
        raise ValueError("capability dimensions must define stage, method, and goal")
    labels = {}
    for dimension, values in dimensions.items():
        ids = [item.get("id") for item in values]
        if any(not item.get("label") for item in values) or len(ids) != len(set(ids)):
            raise ValueError(f"invalid or duplicate {dimension} capability label")
        labels[dimension] = set(ids)
    levels = data.get("claim_levels", {})
    if not levels:
        raise ValueError("capability claim_levels must not be empty")

    module_index = {module["name"]: module for module in modules}
    symbol_index = {}
    for module in modules:
        module["capabilities"] = []
        for symbol in module["symbols"]:
            symbol["capabilities"] = []
            symbol_index[(module["name"], symbol["name"])] = symbol

    seen = set()
    for entry in data.get("entries", []):
        target = (entry.get("module"), entry.get("symbol"))
        if target in seen:
            raise ValueError(f"duplicate capability target: {target[0]}.{target[1]}")
        seen.add(target)
        if target not in symbol_index:
            raise ValueError(f"capability target does not exist in source: {target[0]}.{target[1]}")
        if entry.get("claim_level") not in levels:
            raise ValueError(f"unknown claim level for {target[0]}.{target[1]}")
        for dimension in labels:
            selected = entry.get(dimension, [])
            unknown = set(selected) - labels[dimension]
            if unknown:
                raise ValueError(f"unknown {dimension} tags for {target[0]}.{target[1]}: {sorted(unknown)}")
        symbol_index[target]["capabilities"].append(entry)
        module_index[target[0]]["capabilities"].append(entry)
    return {"schema_version": data["schema_version"], "dimensions": dimensions,
            "claim_levels": levels, "entries": data.get("entries", [])}


def git_info(root: Path) -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
        except Exception:
            return ""
    status = run("status", "--short")
    return {"branch": run("branch", "--show-current"), "commit": run("log", "-1", "--format=%h %ad %s", "--date=short"),
            "dirty_files": len([l for l in status.splitlines() if l.strip()])}


def scan_animations(root: Path, module_names: set[str]) -> list[dict]:
    manifest = ANIMATION_ROOT / "src" / "methods.json"
    if not manifest.is_file():
        return []
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    output = ANIMATION_ROOT / "out" / "methods"
    animations = []
    for entry in entries:
        bench_ref = entry["bench"].split("wyu_bci_bench.", 1)[-1]
        module = next((name for name in sorted(module_names, key=len, reverse=True)
                       if bench_ref == name or bench_ref.startswith(name + ".")),
                      bench_ref.split(".", 1)[0].split(" ", 1)[0])
        video_path = output / f"{entry['id']}.mp4"
        animations.append({
            "id": entry["id"], "title": entry["title"], "group": entry["group"],
            "bench": entry["bench"], "claim": entry["claim"], "module": module,
            "video": str(video_path), "available": video_path.is_file(),
        })
    return animations


def build_catalog(root: str | Path) -> dict:
    root = Path(root)
    pkg = root / "src" / "wyu_bci_bench"
    if not pkg.is_dir():
        raise FileNotFoundError(f"{pkg} not found")
    modules = [scan_module(p, pkg) for p in sorted(pkg.rglob("*.py")) if "__pycache__" not in p.parts]
    modules = [m for m in modules if m["name"] != "__init__"]
    names = {m["name"] for m in modules}
    for m in modules:
        fixed = set()
        for imp in m["imports_internal"]:
            while imp and imp not in names and "." in imp:
                imp = imp.rsplit(".", 1)[0]
            if imp in names and imp != m["name"]:
                fixed.add(imp)
        m["imports_internal"] = sorted(fixed)
        m["used_by"] = []
    for m in modules:
        for imp in m["imports_internal"]:
            next(x for x in modules if x["name"] == imp)["used_by"].append(m["name"])
    capabilities = attach_capabilities(modules, root / "explorer" / "capabilities.json")
    version = re.search(r'__version__ = "([^"]+)"', (pkg / "__init__.py").read_text(encoding="utf-8"))
    init_tree = ast.parse((pkg / "__init__.py").read_text(encoding="utf-8"))
    exported = []
    for node in init_tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "__all__" for t in node.targets):
            exported = [ast.literal_eval(e) for e in node.value.elts]
    verify = root / "outputs" / "last_verify.json"
    scripts = scan_scripts(root)
    animations = scan_animations(root, names)
    pub = [s for m in modules for s in m["symbols"] if s["public"]]
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "root": str(root),
        "version": version.group(1) if version else "?", "git": git_info(root),
        "layers": [{"id": i, "name": n, "desc": d} for i, n, d in LAYERS],
        "modules": modules, "scripts": scripts, "exported": exported, "capabilities": capabilities,
        "animations": animations,
        "migration": parse_migration(root / "docs" / "EEG_FOUNDATION_PROTOCOLS.md"),
        "verify": json.loads(verify.read_text(encoding="utf-8")) if verify.exists() else None,
        "stats": {"modules": len(modules), "loc": sum(m["loc"] for m in modules),
                  "functions": sum(s["kind"] == "function" for s in pub),
                  "classes": sum(s["kind"] == "class" for s in pub),
                  "tests": sum(len(s["tests"]) for s in scripts),
                  "animations": len(animations), "animation_videos": sum(a["available"] for a in animations),
                  "v02_modules": sum(m["version"].endswith("v0.2") for m in modules)},
    }


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    cat = build_catalog(root)
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    payload = json.dumps(cat, ensure_ascii=False)
    if out:
        out.write_text(payload, encoding="utf-8")
    print(json.dumps(cat["stats"], ensure_ascii=False), "->", out or "(stdout suppressed)")
