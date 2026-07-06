from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE = Path("E:/\u4e16\u754c\u673a\u5668\u4eba\u5927\u8d5b/wyu-bci-bench/outputs")
OUT = BASE / "ljy74_final_comparison_report"

DATASETS = [
    {
        "id": "all5",
        "name": "\u5168\u90e85\u4e2a\u6587\u4ef6",
        "prefix": "ljy74_all",
        "files": [
            "15-54-41_mi_me.npz",
            "16-03-06_mi_me.npz",
            "16-15-12_mi_me.npz",
            "16-31-06_mi_me.npz",
            "16-44-21_mi_me.npz",
        ],
    },
    {
        "id": "latest2",
        "name": "\u6700\u65b02\u4e2a\u5b8c\u6574\u6587\u4ef6",
        "prefix": "ljy74_latest2",
        "files": ["16-31-06_mi_me.npz", "16-44-21_mi_me.npz"],
    },
]

TASKS = [
    ("mi_me8", "8\u7c7b\uff1a4\u90e8\u4f4d\u00d7\u60f3\u8c61/\u6267\u884c", 0.125),
    ("mi_me_state", "2\u7c7b\uff1a\u60f3\u8c61 vs \u6267\u884c", 0.5),
    ("mi_me_body4", "4\u7c7b\uff1a\u8eab\u4f53\u90e8\u4f4d", 0.25),
    ("mi_me_imagery_body4", "4\u7c7b\uff1a\u53ea\u770b\u60f3\u8c61", 0.25),
    ("mi_me_execution_body4", "4\u7c7b\uff1a\u53ea\u770b\u6267\u884c", 0.25),
]

MODELS = [
    ("simple", "simple_logvar", "simple_logvar(best)"),
    ("fbcsp", "fbcsp_lda_stable", "FBCSP+LDA"),
    ("riemann", "riemannian", "Riemannian(best)"),
    ("eegnet", "eegnet", "EEGNet"),
]

COLORS = {
    ("all5", "simple"): (47, 110, 179),
    ("all5", "fbcsp"): (46, 139, 87),
    ("all5", "riemann"): (20, 163, 165),
    ("all5", "eegnet"): (127, 140, 141),
    ("latest2", "simple"): (232, 110, 27),
    ("latest2", "fbcsp"): (125, 91, 214),
    ("latest2", "riemann"): (194, 24, 91),
    ("latest2", "eegnet"): (161, 98, 7),
}

BLACK = (17, 24, 39)
GRAY = (102, 112, 133)
LIGHT = (229, 231, 235)
GRID = (208, 215, 222)
RED = (239, 68, 68)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for item in candidates:
        if item.exists():
            return ImageFont.truetype(str(item), size=size)
    return ImageFont.load_default()


F_TITLE = font(34, True)
F_SUB = font(18)
F_TEXT = font(18)
F_SMALL = font(15)
F_TINY = font(13)
F_TABLE = font(15)
F_TABLE_SMALL = font(13)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: object, fill=BLACK, font_obj=F_TEXT, anchor: str | None = None) -> None:
    draw.text(xy, str(value), fill=fill, font=font_obj, anchor=anchor)


def text_width(draw: ImageDraw.ImageDraw, value: str, font_obj: ImageFont.FreeTypeFont) -> int:
    box = draw.textbbox((0, 0), value, font=font_obj)
    return int(box[2] - box[0])


def ellipsize(draw: ImageDraw.ImageDraw, value: object, max_width: int, font_obj: ImageFont.FreeTypeFont) -> str:
    raw = str(value)
    if text_width(draw, raw, font_obj) <= max_width:
        return raw
    result = raw
    while result and text_width(draw, result + "...", font_obj) > max_width:
        result = result[:-1]
    return result + "..." if result else "..."


def clean_output() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in [OUT / "task_charts", OUT / "confusion_matrices"]:
        if sub.exists():
            shutil.rmtree(sub)
        sub.mkdir(parents=True, exist_ok=True)
    for item in OUT.glob("*.png"):
        item.unlink()


def load_row(dataset: dict[str, object], task_id: str, task_name: str, chance: float, model_id: str, model_key: str, model_name: str) -> dict[str, object]:
    report = BASE / f"{dataset['prefix']}_{task_id}_{model_id}_precue" / "benchmark_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    raw = payload["models"][model_key]
    if "best" in raw:
        item = raw["best"]
        detail = f"{item.get('feature_set', '')} + {item.get('model', '')}".strip(" +")
    else:
        item = raw
        detail = {
            "fbcsp": "filter-bank spatial logvar + diagonal LDA",
            "eegnet": "PyTorch EEGNet, 30 epochs",
        }.get(model_id, model_name)
    ds = payload["dataset"]
    prep = ds.get("metadata", {}).get("continuous_preprocessing", {})
    return {
        "dataset_id": dataset["id"],
        "dataset": dataset["name"],
        "task_id": task_id,
        "task": task_name,
        "model_id": model_id,
        "model": model_name,
        "model_detail": detail,
        "accuracy": float(item["accuracy"]),
        "balanced_accuracy": float(item["balanced_accuracy"]),
        "macro_f1": float(item["macro_f1"]),
        "chance": float(chance),
        "shape": ds.get("shape", []),
        "source_files": [Path(p).name for p in ds.get("source_files", dataset["files"])],
        "source_trial_counts": ds.get("source_trial_counts", []),
        "bad_channels_removed": prep.get("bad_channels_removed", []),
        "label_order": item.get("label_order") or ds.get("label_names", []),
        "confusion_matrix": item.get("confusion_matrix"),
        "report_path": str(report),
    }


def collect_rows() -> list[dict[str, object]]:
    rows = []
    for dataset in DATASETS:
        for task_id, task_name, chance in TASKS:
            for model_id, model_key, model_name in MODELS:
                rows.append(load_row(dataset, task_id, task_name, chance, model_id, model_key, model_name))
    return rows


def write_tables(rows: list[dict[str, object]]) -> None:
    (OUT / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = [
        "dataset",
        "task",
        "model",
        "model_detail",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "chance",
        "shape",
        "source_files",
        "source_trial_counts",
        "bad_channels_removed",
        "report_path",
    ]
    with (OUT / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row[key] for key in columns if key not in {"shape", "source_files", "source_trial_counts", "bad_channels_removed"}},
                    "shape": "x".join(map(str, row["shape"])),
                    "source_files": "; ".join(row["source_files"]),
                    "source_trial_counts": "; ".join(map(str, row["source_trial_counts"])),
                    "bad_channels_removed": ", ".join(row["bad_channels_removed"]),
                }
            )


def y_to_px(value: float, top: int, height: int) -> int:
    return int(top + (1.0 - value) * height)


def draw_axes(draw: ImageDraw.ImageDraw, left: int, top: int, width: int, height: int) -> None:
    for i in range(11):
        val = i / 10.0
        y = y_to_px(val, top, height)
        draw.line((left, y, left + width, y), fill=GRID, width=1)
        text(draw, (left - 42, y - 9), f"{val:.1f}", fill=GRAY, font_obj=F_TINY)
    draw.rectangle((left, top, left + width, top + height), outline=(156, 163, 175), width=1)


def plot_overview(rows: list[dict[str, object]]) -> None:
    img = Image.new("RGB", (1900, 1050), "white")
    draw = ImageDraw.Draw(img)
    text(draw, (60, 45), "ljy7.4 EEG/MI-ME Bench 汇总 Balanced Accuracy", font_obj=F_TITLE)
    text(draw, (60, 92), "全部5个文件 vs 最新2个完整40-trial文件；红线为随机水平。", fill=GRAY, font_obj=F_SUB)
    left, top, width, height = 110, 165, 1700, 650
    draw_axes(draw, left, top, width, height)

    specs = [
        ("all5", "simple", "全部-simple"),
        ("all5", "fbcsp", "全部-FBCSP"),
        ("all5", "riemann", "全部-Riemann"),
        ("all5", "eegnet", "全部-EEGNet"),
        ("latest2", "simple", "最新2-simple"),
        ("latest2", "fbcsp", "最新2-FBCSP"),
        ("latest2", "riemann", "最新2-Riemann"),
        ("latest2", "eegnet", "最新2-EEGNet"),
    ]
    group_w = width / len(TASKS)
    bar_w = 22
    for task_idx, (task_id, task_name, chance) in enumerate(TASKS):
        group_x = left + task_idx * group_w
        draw.line((group_x + 22, y_to_px(chance, top, height), group_x + group_w - 22, y_to_px(chance, top, height)), fill=RED, width=2)
        label = task_name
        text(draw, (int(group_x + group_w / 2), top + height + 24), label, fill=BLACK, font_obj=F_TINY, anchor="mm")
        for spec_idx, (dataset_id, model_id, _) in enumerate(specs):
            row = next(r for r in rows if r["dataset_id"] == dataset_id and r["model_id"] == model_id and r["task_id"] == task_id)
            val = row["balanced_accuracy"]
            x0 = int(group_x + 42 + spec_idx * (bar_w + 6))
            x1 = x0 + bar_w
            y0 = y_to_px(val, top, height)
            draw.rectangle((x0, y0, x1, top + height), fill=COLORS[(dataset_id, model_id)], outline=BLACK)
            text(draw, (x0 + bar_w // 2, y0 - 20), f"{val:.2f}", fill=GRAY, font_obj=F_TINY, anchor="mm")

    legend_x, legend_y = 120, 885
    for idx, (dataset_id, model_id, label) in enumerate(specs):
        x = legend_x + (idx % 4) * 410
        y = legend_y + (idx // 4) * 45
        draw.rectangle((x, y, x + 28, y + 20), fill=COLORS[(dataset_id, model_id)], outline=BLACK)
        text(draw, (x + 38, y - 2), label, font_obj=F_SMALL)
    draw.line((legend_x, legend_y + 94, legend_x + 28, legend_y + 94), fill=RED, width=3)
    text(draw, (legend_x + 38, legend_y + 82), "随机水平", font_obj=F_SMALL)
    img.save(OUT / "00_balanced_accuracy_overview.png")


def plot_task_charts(rows: list[dict[str, object]]) -> None:
    for task_id, task_name, chance in TASKS:
        img = Image.new("RGB", (1500, 820), "white")
        draw = ImageDraw.Draw(img)
        text(draw, (50, 35), f"{task_name} - Balanced Accuracy", font_obj=F_TITLE)
        left, top, width, height = 95, 115, 1320, 510
        draw_axes(draw, left, top, width, height)
        draw.line((left + 20, y_to_px(chance, top, height), left + width - 20, y_to_px(chance, top, height)), fill=RED, width=2)
        bars = []
        for dataset in DATASETS:
            for model_id, _, model_name in MODELS:
                row = next(r for r in rows if r["dataset_id"] == dataset["id"] and r["model_id"] == model_id and r["task_id"] == task_id)
                bars.append((dataset["id"], model_id, f"{dataset['name']}\n{model_name}", row["balanced_accuracy"]))
        slot = width / len(bars)
        bar_w = 68
        for idx, (dataset_id, model_id, label, val) in enumerate(bars):
            cx = left + int(slot * idx + slot / 2)
            y0 = y_to_px(val, top, height)
            draw.rectangle((cx - bar_w // 2, y0, cx + bar_w // 2, top + height), fill=COLORS[(dataset_id, model_id)], outline=BLACK)
            text(draw, (cx, y0 - 20), f"{val:.3f}", fill=GRAY, font_obj=F_SMALL, anchor="mm")
            first, second = label.split("\n")
            text(draw, (cx, top + height + 25), first, font_obj=F_TINY, anchor="mm")
            text(draw, (cx, top + height + 47), second, font_obj=F_TINY, anchor="mm")
        draw.line((1220, 55, 1260, 55), fill=RED, width=3)
        text(draw, (1270, 45), "随机水平", font_obj=F_SMALL)
        img.save(OUT / "task_charts" / f"{task_id}.png")


def plot_table(rows: list[dict[str, object]]) -> None:
    columns = [
        ("数据", 175),
        ("分类任务", 235),
        ("模型", 185),
        ("最佳特征/模型细节", 475),
        ("Acc", 90),
        ("Bal Acc", 105),
        ("F1", 90),
        ("Chance", 95),
        ("Shape", 165),
    ]
    row_h = 38
    width = 40 + sum(item[1] for item in columns) + 40
    height = 130 + row_h * (len(rows) + 1) + 40
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    text(draw, (40, 28), "ljy7.4 完整结果表", font_obj=F_TITLE)
    text(draw, (40, 78), "metric=balanced accuracy；完整文件列表见 README/CSV。", fill=GRAY, font_obj=F_SUB)
    x = 40
    y = 120
    for title, col_w in columns:
        draw.rectangle((x, y, x + col_w, y + row_h), fill=BLACK)
        text(draw, (x + 8, y + 9), title, fill="white", font_obj=F_TABLE)
        x += col_w
    for row_idx, row in enumerate(rows):
        x = 40
        y = 120 + row_h * (row_idx + 1)
        fill = (248, 250, 252) if row_idx % 2 == 0 else "white"
        values = [
            row["dataset"],
            row["task"],
            row["model"],
            row["model_detail"],
            f"{row['accuracy']:.3f}",
            f"{row['balanced_accuracy']:.3f}",
            f"{row['macro_f1']:.3f}",
            f"{row['chance']:.3f}",
            "x".join(map(str, row["shape"])),
        ]
        for col_idx, ((_, col_w), value) in enumerate(zip(columns, values)):
            draw.rectangle((x, y, x + col_w, y + row_h), fill=fill, outline=LIGHT)
            value = ellipsize(draw, value, col_w - 14, F_TABLE_SMALL)
            color = BLACK
            if col_idx == 5 and float(row["balanced_accuracy"]) >= 0.6:
                color = (22, 101, 52)
            text(draw, (x + 8, y + 10), value, fill=color, font_obj=F_TABLE_SMALL)
            x += col_w
    img.save(OUT / "01_summary_table.png")


def blue(value: float) -> tuple[int, int, int]:
    value = max(0.0, min(1.0, float(value)))
    low = (239, 246, 255)
    high = (30, 64, 175)
    return tuple(int(low[i] * (1.0 - value) + high[i] * value) for i in range(3))


def plot_confusion(row: dict[str, object]) -> None:
    matrix = row.get("confusion_matrix")
    if not matrix:
        return
    arr = np.asarray(matrix, dtype=float)
    labels = [str(item) for item in (row["label_order"] or range(arr.shape[0]))]
    n = len(labels)
    cell = 76 if n <= 4 else 68
    left, top = 170, 135
    width = left + n * cell + 80
    height = top + n * cell + 130
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    text(draw, (30, 25), f"{row['dataset']} | {row['task']} | {row['model']}", font_obj=F_SUB)
    text(draw, (30, 58), f"Bal Acc={row['balanced_accuracy']:.3f}", fill=GRAY, font_obj=F_SMALL)
    denom = np.maximum(arr.sum(axis=1, keepdims=True), 1.0)
    norm = arr / denom
    for idx, label in enumerate(labels):
        text(draw, (left - 10, top + idx * cell + cell // 2 - 10), ellipsize(draw, label, 140, F_TINY), font_obj=F_TINY, anchor="ra")
        text(draw, (left + idx * cell + cell // 2, top - 22), ellipsize(draw, label, cell + 20, F_TINY), font_obj=F_TINY, anchor="mm")
    for i in range(n):
        for j in range(n):
            x0 = left + j * cell
            y0 = top + i * cell
            val = float(norm[i, j])
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=blue(val), outline="white")
            color = "white" if val > 0.5 else BLACK
            text(draw, (x0 + cell // 2, y0 + cell // 2 - 12), int(arr[i, j]), fill=color, font_obj=F_SMALL, anchor="mm")
            text(draw, (x0 + cell // 2, y0 + cell // 2 + 12), f"{val:.2f}", fill=color, font_obj=F_TINY, anchor="mm")
    text(draw, (left + n * cell // 2, height - 54), "预测", font_obj=F_SMALL, anchor="mm")
    text(draw, (38, top + n * cell // 2), "真实", font_obj=F_SMALL)
    img.save(OUT / "confusion_matrices" / f"{row['dataset_id']}_{row['task_id']}_{row['model_id']}_confusion.png")


def write_readme(rows: list[dict[str, object]]) -> None:
    best_all = max((r for r in rows if r["dataset_id"] == "all5"), key=lambda r: r["balanced_accuracy"])
    best_latest = max((r for r in rows if r["dataset_id"] == "latest2"), key=lambda r: r["balanced_accuracy"])
    best_by_task = [
        max((r for r in rows if r["dataset_id"] == "latest2" and r["task_id"] == task_id), key=lambda r: r["balanced_accuracy"])
        for task_id, _, _ in TASKS
    ]
    lines = [
        "# ljy7.4 合并数据 Bench 结果",
        "",
        "## 数据",
        "",
        "- 来源文件夹：`E:\\世界机器人大赛\\projects\\arm_bci_robot\\data\\recordings\\2026-07-04\\ljy7.4`",
        "- 全部 5 个文件：`15-54-41_mi_me.npz`, `16-03-06_mi_me.npz`, `16-15-12_mi_me.npz`, `16-31-06_mi_me.npz`, `16-44-21_mi_me.npz`",
        "- 最新 2 个完整 40-trial 文件：`16-31-06_mi_me.npz`, `16-44-21_mi_me.npz`",
        "- 采样率：250 Hz；F4 手动移除；pre-cue baseline；5-fold cross validation。",
        "- 全部 5 个文件：180 个有效 epoch，2 个未完成 trial 被拒绝。",
        "- 最新 2 个完整文件：80 个有效 epoch，8 类各 10 次。",
        "",
        "## 模型",
        "",
        "- `simple_logvar(best)`：多组 log-variance / spatial log-variance 特征 + 小样本分类器，取最佳组合。",
        "- `FBCSP+LDA`：filter-bank spatial log-variance + diagonal LDA。",
        "- `Riemannian(best)`：协方差矩阵 + Riemannian MDM / Tangent Space + LDA / Filter-bank Tangent Space，取最佳组合。",
        "- `EEGNet`：PyTorch EEGNet，30 epochs。",
        "",
        "## 最新2个完整文件：每个任务最佳",
        "",
        "| 分类任务 | 最佳模型 | Acc | Bal Acc | Macro F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in best_by_task:
        lines.append(f"| {row['task']} | {row['model']} ({row['model_detail']}) | {row['accuracy']:.3f} | {row['balanced_accuracy']:.3f} | {row['macro_f1']:.3f} |")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 全部 5 个文件最佳：{best_all['task']} / {best_all['model']}，Bal Acc = {best_all['balanced_accuracy']:.3f}。",
            f"- 最新 2 个完整文件最佳：{best_latest['task']} / {best_latest['model']}，Bal Acc = {best_latest['balanced_accuracy']:.3f}。",
            "- 新加的 Riemannian/filter-bank tangent-space 方法在最新 2 个完整文件上是当前最好的机器学习方向：2 类、身体 4 类、只看想象 4 类都超过旧基线。",
            "- 8 类直接分类仍不稳，最新 2 个文件最高 Bal Acc = 0.375；建议控制策略先做层级分类或先聚焦想象 4 类。",
            "- 小样本下 EEGNet 没有超过传统 EEG 特征工程；当前不建议把端到端深度学习作为主模型。",
            "",
            "## 输出文件",
            "",
            "- `00_balanced_accuracy_overview.png`：全部任务总览柱状图。",
            "- `01_summary_table.png`：完整结果表 PNG。",
            "- `task_charts/`：按分类任务拆分的 PNG。",
            "- `confusion_matrices/`：每个数据集、任务、模型的混淆矩阵 PNG。",
            "- `summary.csv` / `summary.json`：机器可读结果。",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    clean_output()
    rows = collect_rows()
    write_tables(rows)
    plot_overview(rows)
    plot_table(rows)
    plot_task_charts(rows)
    for row in rows:
        plot_confusion(row)
    write_readme(rows)
    print(f"wrote {OUT}")
    print(f"rows={len(rows)} pngs={len(list(OUT.rglob('*.png')))}")
    for row in rows:
        print(f"{row['dataset']} | {row['task']} | {row['model']} | bal={row['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
