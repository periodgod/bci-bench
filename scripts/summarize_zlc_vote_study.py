"""Summarize ZLC 2026-07-04 MI/ME voting ensemble results."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


MODELS = ["vote-simple", "vote-capsule", "vote-ml-best", "vote-ml-pruned", "vote-ml-pruned-capsule"]
REPORT_KEYS = {
    "vote-simple": "vote_simple",
    "vote-capsule": "vote_capsule",
    "vote-ml-best": "vote_ml_best",
    "vote-ml-pruned": "vote_ml_pruned",
    "vote-ml-pruned-capsule": "vote_ml_pruned_capsule",
}
COLORS = {
    "vote-simple": (59, 130, 246),
    "vote-capsule": (139, 92, 246),
    "vote-ml-best": (16, 185, 129),
    "vote-ml-pruned": (20, 184, 166),
    "vote-ml-pruned-capsule": (245, 158, 11),
}
ML_COLOR = (15, 118, 110)


def bench_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _font(size: int = 14) -> ImageFont.ImageFont:
    for candidate in [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _draw_center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont, fill=(31, 41, 55)) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    x = box[0] + (box[2] - box[0] - (bbox[2] - bbox[0])) // 2
    y = box[1] + (box[3] - box[1] - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), text, font=font, fill=fill)


def load_rows(study_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    voter_rows: list[dict[str, Any]] = []
    bench_root_path = study_root / "01_benchmarks" / "zlc74" / "mi_me8"
    for model in MODELS:
        report = bench_root_path / model / "benchmark_report.json"
        if not report.exists():
            continue
        payload = json.loads(report.read_text(encoding="utf-8"))
        metrics = payload["models"][REPORT_KEYS[model]]
        shape = payload.get("dataset", {}).get("shape", [])
        rows.append(
            {
                "subject": "zlc",
                "case_id": "zlc74",
                "label_mode": "mi_me8",
                "model": model,
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "n_trials": int(shape[0]) if len(shape) > 0 else 0,
                "n_channels": int(shape[1]) if len(shape) > 1 else 0,
                "n_samples": int(shape[2]) if len(shape) > 2 else 0,
                "report_path": str(report),
                "confusion_matrix": metrics["confusion_matrix"],
                "label_order": metrics["label_order"],
            }
        )
        for voter_name, voter_metrics in metrics.get("voter_metrics", {}).items():
            voter_rows.append(
                {
                    "ensemble": model,
                    "voter": voter_name,
                    "accuracy": float(voter_metrics["accuracy"]),
                    "balanced_accuracy": float(voter_metrics["balanced_accuracy"]),
                    "macro_f1": float(voter_metrics["macro_f1"]),
                }
            )
    return rows, voter_rows


def load_ml_suite_rows(study_root: Path) -> list[dict[str, Any]]:
    report = study_root / "01_benchmarks" / "zlc74" / "mi_me8" / "ml-suite" / "benchmark_report.json"
    if not report.exists():
        return []
    payload = json.loads(report.read_text(encoding="utf-8"))
    block = payload["models"]["ml_suite"]
    rows = []
    for metrics in block.get("models", []):
        rows.append(
            {
                "subject": "zlc",
                "case_id": "zlc74",
                "label_mode": "mi_me8",
                "model": metrics.get("model", ""),
                "feature_set": metrics.get("feature_set", ""),
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "report_path": str(report),
            }
        )
    rows.sort(key=lambda row: (row["balanced_accuracy"], row["macro_f1"], row["accuracy"]), reverse=True)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_bar(rows: list[dict[str, Any]], path: Path) -> None:
    width, height = 900, 560
    margin_left, margin_right, margin_top, margin_bottom = 90, 35, 70, 95
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = _font(24)
    small_font = _font(13)
    label_font = _font(15)
    draw.text((margin_left, 25), "ZLC 8-class voting ensemble", font=title_font, fill=(17, 24, 39))
    max_y = 0.55
    for tick in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        y = margin_top + plot_h - int((tick / max_y) * plot_h)
        draw.line((margin_left, y, width - margin_right, y), fill=(229, 231, 235), width=1)
        draw.text((20, y - 8), f"{tick:.2f}", font=small_font, fill=(75, 85, 99))
    chance_y = margin_top + plot_h - int((0.125 / max_y) * plot_h)
    draw.line((margin_left, chance_y, width - margin_right, chance_y), fill=(107, 114, 128), width=2)
    draw.text((width - margin_right - 85, chance_y + 7), "chance", font=small_font, fill=(75, 85, 99))
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill=(55, 65, 81), width=2)
    draw.line((margin_left, margin_top + plot_h, width - margin_right, margin_top + plot_h), fill=(55, 65, 81), width=2)
    group_w = plot_w / len(rows)
    for idx, row in enumerate(rows):
        value = float(row["balanced_accuracy"])
        center = margin_left + idx * group_w + group_w / 2
        bar_w = 90
        x0 = int(center - bar_w / 2)
        x1 = int(center + bar_w / 2)
        y1 = margin_top + plot_h
        y0 = margin_top + plot_h - int((value / max_y) * plot_h)
        draw.rectangle((x0, y0, x1, y1), fill=COLORS[row["model"]])
        _draw_center(draw, (x0 - 12, y0 - 25, x1 + 12, y0 - 4), f"{value:.3f}", small_font)
        _draw_center(draw, (int(center - group_w / 2), margin_top + plot_h + 22, int(center + group_w / 2), margin_top + plot_h + 48), row["model"], label_font)
    image.convert("RGB").save(path)


def plot_ml_suite(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    width, height = 1200, 680
    margin_left, margin_right, margin_top, margin_bottom = 95, 35, 70, 165
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = _font(24)
    small_font = _font(12)
    draw.text((margin_left, 25), "ZLC 8-class ML algorithm suite", font=title_font, fill=(17, 24, 39))
    max_y = 0.5
    for tick in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        y = margin_top + plot_h - int((tick / max_y) * plot_h)
        draw.line((margin_left, y, width - margin_right, y), fill=(229, 231, 235), width=1)
        draw.text((20, y - 8), f"{tick:.2f}", font=small_font, fill=(75, 85, 99))
    chance_y = margin_top + plot_h - int((0.125 / max_y) * plot_h)
    draw.line((margin_left, chance_y, width - margin_right, chance_y), fill=(107, 114, 128), width=2)
    draw.text((width - margin_right - 85, chance_y + 7), "chance", font=small_font, fill=(75, 85, 99))
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill=(55, 65, 81), width=2)
    draw.line((margin_left, margin_top + plot_h, width - margin_right, margin_top + plot_h), fill=(55, 65, 81), width=2)
    group_w = plot_w / len(rows)
    for idx, row in enumerate(rows):
        value = float(row["balanced_accuracy"])
        center = margin_left + idx * group_w + group_w / 2
        bar_w = max(36, min(78, int(group_w * 0.55)))
        x0 = int(center - bar_w / 2)
        x1 = int(center + bar_w / 2)
        y1 = margin_top + plot_h
        y0 = margin_top + plot_h - int((value / max_y) * plot_h)
        draw.rectangle((x0, y0, x1, y1), fill=ML_COLOR)
        _draw_center(draw, (x0 - 10, y0 - 23, x1 + 10, y0 - 4), f"{value:.3f}", small_font)
        label = str(row["model"]).replace("_", "\n")
        _draw_center(draw, (int(center - group_w / 2), margin_top + plot_h + 12, int(center + group_w / 2), margin_top + plot_h + 105), label, small_font)
    image.convert("RGB").save(path)


def plot_confusion(row: dict[str, Any], path: Path) -> None:
    matrix = row["confusion_matrix"]
    n = len(matrix)
    cell = 56
    margin_left, margin_top, margin_right, margin_bottom = 85, 90, 35, 75
    width = margin_left + n * cell + margin_right
    height = margin_top + n * cell + margin_bottom
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = _font(16)
    small_font = _font(12)
    draw.text(
        (margin_left, 24),
        f"{row['model']}  bal={float(row['balanced_accuracy']):.3f} acc={float(row['accuracy']):.3f}",
        font=title_font,
        fill=(17, 24, 39),
    )
    for i in range(n):
        row_sum = max(sum(matrix[i]), 1)
        for j in range(n):
            norm = matrix[i][j] / row_sum
            intensity = int(255 - 180 * norm)
            fill = (intensity, min(245, intensity + 15), 255)
            x0 = margin_left + j * cell
            y0 = margin_top + i * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=(219, 234, 254))
            text_color = (255, 255, 255) if norm >= 0.55 else (17, 24, 39)
            _draw_center(draw, (x0, y0, x1, y1), str(matrix[i][j]), small_font, text_color)
    labels = [f"C{i + 1}" for i in range(n)]
    for idx, label in enumerate(labels):
        _draw_center(draw, (margin_left + idx * cell, margin_top - 28, margin_left + (idx + 1) * cell, margin_top - 6), label, small_font)
        _draw_center(draw, (margin_left - 50, margin_top + idx * cell, margin_left - 8, margin_top + (idx + 1) * cell), label, small_font)
    draw.text((margin_left + n * cell // 2 - 30, height - 42), "Predicted", font=small_font, fill=(55, 65, 81))
    draw.text((18, margin_top + n * cell // 2 - 8), "True", font=small_font, fill=(55, 65, 81))
    image.convert("RGB").save(path)


def main() -> int:
    study_root = bench_root() / "outputs" / "mi_me_zlc_20260704_study"
    report_root = study_root / "02_reports"
    plot_root = report_root / "plots"
    report_root.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)
    rows, voter_rows = load_rows(study_root)
    ml_rows = load_ml_suite_rows(study_root)
    write_csv(
        report_root / "zlc74_mi_me8_vote_summary.csv",
        rows,
        ["subject", "case_id", "label_mode", "model", "accuracy", "balanced_accuracy", "macro_f1", "n_trials", "n_channels", "n_samples", "report_path"],
    )
    write_csv(
        report_root / "zlc74_mi_me8_voter_metrics.csv",
        voter_rows,
        ["ensemble", "voter", "accuracy", "balanced_accuracy", "macro_f1"],
    )
    write_csv(
        report_root / "zlc74_mi_me8_ml_suite.csv",
        ml_rows,
        ["subject", "case_id", "label_mode", "model", "feature_set", "accuracy", "balanced_accuracy", "macro_f1", "report_path"],
    )
    (report_root / "zlc74_mi_me8_vote_summary.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "rows": rows,
                "voter_rows": voter_rows,
                "ml_suite_rows": ml_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_bar(rows, plot_root / "zlc74_mi_me8_vote_comparison.png")
    plot_ml_suite(ml_rows, plot_root / "zlc74_mi_me8_ml_suite.png")
    for row in rows:
        plot_confusion(row, plot_root / f"zlc74_mi_me8_{row['model']}_confusion.png")
    lines = [
        "# ZLC 8-Class Voting Results",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Model | Accuracy | Balanced accuracy | Macro F1 |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| `{row['model']}` | {row['accuracy']:.3f} | {row['balanced_accuracy']:.3f} | {row['macro_f1']:.3f} |")
    lines.extend(
        [
            "",
            "## ML Suite",
            "",
            "| Algorithm | Accuracy | Balanced accuracy | Macro F1 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in ml_rows:
        lines.append(f"| `{row['model']}` | {row['accuracy']:.3f} | {row['balanced_accuracy']:.3f} | {row['macro_f1']:.3f} |")
    lines.extend(
        [
            "",
            "Files:",
            "- `zlc74_mi_me8_vote_summary.csv`",
            "- `zlc74_mi_me8_voter_metrics.csv`",
            "- `zlc74_mi_me8_ml_suite.csv`",
            "- `plots/zlc74_mi_me8_vote_comparison.png`",
            "- `plots/zlc74_mi_me8_ml_suite.png`",
            "- `plots/zlc74_mi_me8_vote-simple_confusion.png`",
            "- `plots/zlc74_mi_me8_vote-capsule_confusion.png`",
            "- `plots/zlc74_mi_me8_vote-ml-best_confusion.png`",
            "- `plots/zlc74_mi_me8_vote-ml-pruned_confusion.png`",
            "- `plots/zlc74_mi_me8_vote-ml-pruned-capsule_confusion.png`",
        ]
    )
    (report_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"rows={len(rows)}")
    print(f"voter_rows={len(voter_rows)}")
    print(f"ml_suite_rows={len(ml_rows)}")
    print(f"report_root={report_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
