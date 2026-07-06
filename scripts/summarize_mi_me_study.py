"""Summarize the standardized LJY MI/ME study benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CASE_ORDER = [
    "A_ljy74_all",
    "B_ljy742_all",
    "C_ljy74_latest2_plus_ljy742_all",
]

LABEL_ORDER = [
    "mi_me_state",
    "mi_me_imagery_body4",
    "mi_me_execution_body4",
    "mi_me_body4",
    "mi_me8",
]

MODEL_ORDER = [
    "simple",
    "fbcsp-lda",
    "riemann",
    "minirocket",
    "xgb",
    "eegnet",
]

CHANCE = {
    "mi_me_state": 0.5,
    "mi_me_imagery_body4": 0.25,
    "mi_me_execution_body4": 0.25,
    "mi_me_body4": 0.25,
    "mi_me8": 0.125,
}

COLORS = {
    "simple": "#3b82f6",
    "fbcsp-lda": "#f59e0b",
    "riemann": "#10b981",
    "minirocket": "#ef4444",
    "xgb": "#0f766e",
    "eegnet": "#8b5cf6",
}

RGB_COLORS = {
    "simple": (59, 130, 246),
    "fbcsp-lda": (245, 158, 11),
    "riemann": (16, 185, 129),
    "minirocket": (239, 68, 68),
    "xgb": (15, 118, 110),
    "eegnet": (139, 92, 246),
}

CSV_FIELDS = [
    "case_id",
    "label_mode",
    "model",
    "model_family",
    "feature_set",
    "inner_model",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "chance_balanced_accuracy",
    "n_trials",
    "n_channels",
    "n_samples",
    "report_path",
]


def bench_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def _rank_value(value: str, order: list[str]) -> int:
    return order.index(value) if value in order else 999


def _extract_model_metrics(model_name: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    models = payload.get("models", {})
    if model_name == "simple":
        block = models["simple_logvar"]
        return "simple_logvar", block["best"]
    if model_name == "riemann":
        block = models["riemannian"]
        return "riemannian", block["best"]
    if model_name == "fbcsp-lda":
        return "fbcsp_lda_stable", models["fbcsp_lda_stable"]
    if model_name == "eegnet":
        return "eegnet", models["eegnet"]
    if model_name == "minirocket":
        block = models["minirocket"]
        return "minirocket", block["best"]
    if model_name == "xgb":
        block = models["xgb"]
        return "xgb", block["best"]
    raise ValueError(f"Unsupported model folder: {model_name}")


def collect_rows(benchmark_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in sorted(benchmark_root.rglob("benchmark_report.json")):
        rel = report.relative_to(benchmark_root).parts
        if len(rel) < 4:
            continue
        case_id, label_mode, model_name = rel[0], rel[1], rel[2]
        if model_name not in MODEL_ORDER:
            continue
        payload = json.loads(report.read_text(encoding="utf-8"))
        family, metrics = _extract_model_metrics(model_name, payload)
        shape = payload.get("dataset", {}).get("shape", [])
        rows.append(
            {
                "case_id": case_id,
                "label_mode": label_mode,
                "model": model_name,
                "model_family": family,
                "feature_set": metrics.get("feature_set", ""),
                "inner_model": metrics.get("model", ""),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)),
                "macro_f1": float(metrics.get("macro_f1", 0.0)),
                "chance_balanced_accuracy": CHANCE.get(label_mode, ""),
                "n_trials": int(shape[0]) if len(shape) > 0 else 0,
                "n_channels": int(shape[1]) if len(shape) > 1 else 0,
                "n_samples": int(shape[2]) if len(shape) > 2 else 0,
                "report_path": str(report),
                "label_order": metrics.get("label_order", []),
                "confusion_matrix": metrics.get("confusion_matrix", []),
            }
        )
    rows.sort(
        key=lambda row: (
            _rank_value(str(row["case_id"]), CASE_ORDER),
            _rank_value(str(row["label_mode"]), LABEL_ORDER),
            _rank_value(str(row["model"]), MODEL_ORDER),
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def _font(size: int = 14) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _draw_rotated_text(
    image: Image.Image,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (31, 41, 55),
    angle: int = 45,
) -> None:
    bbox = font.getbbox(text)
    w = max(1, bbox[2] - bbox[0] + 8)
    h = max(1, bbox[3] - bbox[1] + 8)
    tile = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    d = ImageDraw.Draw(tile)
    d.text((4, 4), text, font=font, fill=fill + (255,))
    rotated = tile.rotate(angle, expand=True)
    image.alpha_composite(rotated, (int(xy[0]), int(xy[1])))


def _draw_text_center(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (31, 41, 55),
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    x = box[0] + (box[2] - box[0] - (bbox[2] - bbox[0])) // 2
    y = box[1] + (box[3] - box[1] - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), text, font=font, fill=fill)


def _nice_value(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: list[dict[str, Any]] = []
    for case_id in CASE_ORDER:
        for label_mode in LABEL_ORDER:
            subset = [row for row in rows if row["case_id"] == case_id and row["label_mode"] == label_mode]
            if subset:
                best.append(
                    max(
                        subset,
                        key=lambda row: (
                            float(row["balanced_accuracy"]),
                            float(row["macro_f1"]),
                            float(row["accuracy"]),
                        ),
                    )
                )
    return best


def plot_eight_class(rows: list[dict[str, Any]], path: Path) -> None:
    width, height = 1200, 700
    margin_left, margin_right, margin_top, margin_bottom = 95, 30, 70, 115
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = _font(24)
    label_font = _font(15)
    small_font = _font(13)
    draw.text((margin_left, 22), "8-class MI/ME classification", font=title_font, fill=(17, 24, 39))
    present_models = [model for model in MODEL_ORDER if any(row["model"] == model and row["label_mode"] == "mi_me8" for row in rows)]
    max_y = 0.4
    for tick in np.linspace(0.0, max_y, 5):
        y = margin_top + plot_h - int((tick / max_y) * plot_h)
        draw.line((margin_left, y, width - margin_right, y), fill=(229, 231, 235), width=1)
        draw.text((18, y - 8), f"{tick:.2f}", font=small_font, fill=(75, 85, 99))
    chance_y = margin_top + plot_h - int((CHANCE["mi_me8"] / max_y) * plot_h)
    draw.line((margin_left, chance_y, width - margin_right, chance_y), fill=(107, 114, 128), width=2)
    draw.text((width - margin_right - 92, chance_y + 8), "chance", font=small_font, fill=(75, 85, 99))
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill=(55, 65, 81), width=2)
    draw.line((margin_left, margin_top + plot_h, width - margin_right, margin_top + plot_h), fill=(55, 65, 81), width=2)
    group_w = plot_w / len(CASE_ORDER)
    bar_w = min(48, int(group_w * 0.72 / max(len(present_models), 1)))
    labels = ["A ljy7.4 all", "B ljy7.42 all", "C latest+ljy7.42"]
    for case_idx, case_id in enumerate(CASE_ORDER):
        group_left = margin_left + case_idx * group_w
        group_center = group_left + group_w / 2.0
        for model_idx, model in enumerate(present_models):
            match = [
                row
                for row in rows
                if row["case_id"] == case_id and row["label_mode"] == "mi_me8" and row["model"] == model
            ]
            if not match:
                continue
            value = float(match[0]["balanced_accuracy"])
            x0 = int(group_center - (len(present_models) * bar_w) / 2 + model_idx * bar_w + 4)
            x1 = int(x0 + bar_w - 8)
            y1 = margin_top + plot_h
            y0 = margin_top + plot_h - int((value / max_y) * plot_h)
            draw.rectangle((x0, y0, x1, y1), fill=RGB_COLORS.get(model, (107, 114, 128)))
            _draw_text_center(draw, (x0 - 12, y0 - 24, x1 + 12, y0 - 4), _nice_value(value), small_font)
        _draw_text_center(draw, (int(group_left), margin_top + plot_h + 18, int(group_left + group_w), margin_top + plot_h + 45), labels[case_idx], label_font)
    legend_x = margin_left
    legend_y = height - 42
    for model in present_models:
        color = RGB_COLORS.get(model, (107, 114, 128))
        draw.rectangle((legend_x, legend_y, legend_x + 18, legend_y + 18), fill=color)
        draw.text((legend_x + 26, legend_y - 1), model, font=small_font, fill=(31, 41, 55))
        legend_x += 125
    image.convert("RGB").save(path)


def plot_all_tasks(rows: list[dict[str, Any]], path: Path) -> None:
    width, height = 1800, 820
    margin_left, margin_right, margin_top, margin_bottom = 95, 30, 70, 190
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = _font(24)
    small_font = _font(13)
    draw.text((margin_left, 22), "Auxiliary task overview", font=title_font, fill=(17, 24, 39))
    present_models = [model for model in MODEL_ORDER if any(row["model"] == model for row in rows)]
    max_y = 0.75
    for tick in np.linspace(0.0, max_y, 6):
        y = margin_top + plot_h - int((tick / max_y) * plot_h)
        draw.line((margin_left, y, width - margin_right, y), fill=(229, 231, 235), width=1)
        draw.text((18, y - 8), f"{tick:.2f}", font=small_font, fill=(75, 85, 99))
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill=(55, 65, 81), width=2)
    draw.line((margin_left, margin_top + plot_h, width - margin_right, margin_top + plot_h), fill=(55, 65, 81), width=2)
    task_count = len(CASE_ORDER) * len(LABEL_ORDER)
    group_w = plot_w / task_count
    bar_w = max(5, min(18, int(group_w * 0.78 / max(len(present_models), 1))))
    for case_idx, case_id in enumerate(CASE_ORDER):
        for label_idx, label_mode in enumerate(LABEL_ORDER):
            pos = case_idx * len(LABEL_ORDER) + label_idx
            group_left = margin_left + pos * group_w
            group_center = group_left + group_w / 2.0
            chance_y = margin_top + plot_h - int((CHANCE[label_mode] / max_y) * plot_h)
            draw.line((int(group_left + 3), chance_y, int(group_left + group_w - 3), chance_y), fill=(156, 163, 175), width=1)
            for model_idx, model in enumerate(present_models):
                match = [
                    row
                    for row in rows
                    if row["case_id"] == case_id and row["label_mode"] == label_mode and row["model"] == model
                ]
                if not match:
                    continue
                value = float(match[0]["balanced_accuracy"])
                x0 = int(group_center - (len(present_models) * bar_w) / 2 + model_idx * bar_w + 1)
                x1 = int(x0 + bar_w - 2)
                y1 = margin_top + plot_h
                y0 = margin_top + plot_h - int((value / max_y) * plot_h)
                draw.rectangle((x0, y0, x1, y1), fill=RGB_COLORS.get(model, (107, 114, 128)))
            label = f"{case_id.split('_')[0]} {label_mode.replace('mi_me_', '')}"
            _draw_rotated_text(image, (int(group_left + 4), margin_top + plot_h + 12), label, small_font, angle=45)
    legend_x = margin_left
    legend_y = height - 42
    for model in present_models:
        color = RGB_COLORS.get(model, (107, 114, 128))
        draw.rectangle((legend_x, legend_y, legend_x + 18, legend_y + 18), fill=color)
        draw.text((legend_x + 26, legend_y - 1), model, font=small_font, fill=(31, 41, 55))
        legend_x += 125
    image.convert("RGB").save(path)


def plot_eight_confusions(best_eight: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in best_eight:
        cm = np.asarray(row.get("confusion_matrix", []), dtype=float)
        if cm.size == 0:
            continue
        row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        cm_norm = cm / row_sum
        n_classes = cm.shape[0]
        cell = 54
        margin_left, margin_top, margin_right, margin_bottom = 85, 90, 35, 75
        width = margin_left + n_classes * cell + margin_right
        height = margin_top + n_classes * cell + margin_bottom
        image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        draw = ImageDraw.Draw(image)
        title_font = _font(16)
        small_font = _font(12)
        draw.text(
            (margin_left, 22),
            f"{row['case_id']}  mi_me8 best={row['model']} bal={float(row['balanced_accuracy']):.3f}",
            font=title_font,
            fill=(17, 24, 39),
        )
        names = [f"C{idx + 1}" for idx in range(n_classes)]
        for i in range(n_classes):
            for j in range(n_classes):
                intensity = int(255 - 180 * float(cm_norm[i, j]))
                fill = (intensity, min(245, intensity + 15), 255)
                x0 = margin_left + j * cell
                y0 = margin_top + i * cell
                x1 = x0 + cell
                y1 = y0 + cell
                draw.rectangle((x0, y0, x1, y1), fill=fill, outline=(219, 234, 254))
                text_color = (255, 255, 255) if cm_norm[i, j] >= 0.55 else (17, 24, 39)
                _draw_text_center(draw, (x0, y0, x1, y1), str(int(cm[i, j])), small_font, fill=text_color)
        for idx, name in enumerate(names):
            _draw_text_center(draw, (margin_left + idx * cell, margin_top - 28, margin_left + (idx + 1) * cell, margin_top - 6), name, small_font)
            _draw_text_center(draw, (margin_left - 50, margin_top + idx * cell, margin_left - 8, margin_top + (idx + 1) * cell), name, small_font)
        draw.text((margin_left + n_classes * cell // 2 - 30, height - 42), "Predicted", font=small_font, fill=(55, 65, 81))
        draw.text((18, margin_top + n_classes * cell // 2 - 8), "True", font=small_font, fill=(55, 65, 81))
        image.convert("RGB").save(out_dir / f"{row['case_id']}__mi_me8__best_{row['model']}.png")


def write_readme(path: Path, rows: list[dict[str, Any]], best_eight: list[dict[str, Any]]) -> None:
    lines = [
        "# MI/ME Study Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Primary focus: direct 8-class MI/ME classification (`mi_me8`).",
        "",
        "## 8-Class Results",
        "",
        "| Case | Best model | Inner model | Balanced acc | Accuracy | Macro F1 | Chance |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in best_eight:
        lines.append(
            f"| `{row['case_id']}` | `{row['model']}` | `{row['inner_model'] or row['feature_set']}` | "
            f"{float(row['balanced_accuracy']):.3f} | {float(row['accuracy']):.3f} | "
            f"{float(row['macro_f1']):.3f} | {CHANCE['mi_me8']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary_all_models.csv`",
            "- `summary_all_models.json`",
            "- `summary_eight_class.csv`",
            "- `best_eight_class.csv`",
            "- `plots/eight_class_balanced_accuracy.png`",
            "- `mi_me8_seed_sweep_summary.csv`",
            "- `plots/eight_class_seed_sweep.png`",
            "- `plots/all_tasks_auxiliary_overview.png`",
            "- `plots/confusion_matrices_8class/*.png`",
            "",
            "Auxiliary 2-class and 4-class tasks are kept as plots/tables only; use them to diagnose the paradigm, not as the main result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", default="mi_me_ljy_20260704_study")
    args = parser.parse_args()

    bench_root = bench_root_from_script()
    study_root = bench_root / "outputs" / args.study_id
    benchmark_root = study_root / "01_benchmarks"
    report_root = study_root / "02_reports"
    plots_root = report_root / "plots"
    cm8_root = plots_root / "confusion_matrices_8class"
    report_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(benchmark_root)
    write_csv(report_root / "summary_all_models.csv", rows)
    (report_root / "summary_all_models.json").write_text(
        json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"), "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    eight_rows = [row for row in rows if row["label_mode"] == "mi_me8"]
    write_csv(report_root / "summary_eight_class.csv", eight_rows)

    best_eight = []
    for case_id in CASE_ORDER:
        subset = [row for row in eight_rows if row["case_id"] == case_id]
        if subset:
            best_eight.append(
                max(subset, key=lambda row: (float(row["balanced_accuracy"]), float(row["macro_f1"]), float(row["accuracy"])))
            )
    write_csv(report_root / "best_eight_class.csv", best_eight)

    all_best = best_rows(rows)
    write_csv(report_root / "best_by_task.csv", all_best)
    plot_eight_class(rows, plots_root / "eight_class_balanced_accuracy.png")
    plot_all_tasks(rows, plots_root / "all_tasks_auxiliary_overview.png")
    plot_eight_confusions(best_eight, cm8_root)
    write_readme(report_root / "README.md", rows, best_eight)

    print(f"rows={len(rows)}")
    print(f"eight_class_rows={len(eight_rows)}")
    print(f"report_root={report_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
