"""Run repeated-seed 8-class MI/ME benchmarks for the LJY study."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from wyu_bci_bench.continuous import ContinuousPreprocessConfig, load_continuous_dataset
from wyu_bci_bench.evaluate import (
    cross_validate_filterbank_lda,
    cross_validate_minirocket_models,
    cross_validate_riemannian_models,
    cross_validate_simple_models,
)


CASES = [
    {
        "case_id": "A_ljy74_all",
        "files": [
            "ljy7.4/15-54-41_mi_me.npz",
            "ljy7.4/16-03-06_mi_me.npz",
            "ljy7.4/16-15-12_mi_me.npz",
            "ljy7.4/16-31-06_mi_me.npz",
            "ljy7.4/16-44-21_mi_me.npz",
        ],
    },
    {
        "case_id": "B_ljy742_all",
        "files": [
            "ljy7.42/17-29-54_mi_me.npz",
            "ljy7.42/17-43-26_mi_me.npz",
        ],
    },
    {
        "case_id": "C_ljy74_latest2_plus_ljy742_all",
        "files": [
            "ljy7.4/16-31-06_mi_me.npz",
            "ljy7.4/16-44-21_mi_me.npz",
            "ljy7.42/17-29-54_mi_me.npz",
            "ljy7.42/17-43-26_mi_me.npz",
        ],
    },
]

SEEDS = [7, 13, 21, 42, 84]
MODELS = ["simple", "fbcsp-lda", "riemann", "minirocket"]
MODELS_BY_CASE = {
    "A_ljy74_all": ["riemann"],
    "B_ljy742_all": ["minirocket"],
    "C_ljy74_latest2_plus_ljy742_all": ["fbcsp-lda"],
}
COLORS = {
    "simple": (59, 130, 246),
    "fbcsp-lda": (245, 158, 11),
    "riemann": (16, 185, 129),
    "minirocket": (239, 68, 68),
}


def _font(size: int = 14) -> ImageFont.ImageFont:
    for candidate in [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_root(bench_root: Path) -> Path:
    return bench_root.parent / "projects" / "arm_bci_robot" / "data" / "recordings" / "2026-07-04"


def _extract(model: str, result: dict[str, Any]) -> dict[str, Any]:
    if model == "simple":
        metrics = result["best"]
    elif model == "riemann":
        metrics = result["best"]
    elif model == "minirocket":
        metrics = result["best"]
    else:
        metrics = result
    return {
        "inner_model": metrics.get("model", ""),
        "feature_set": metrics.get("feature_set", ""),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)),
        "macro_f1": float(metrics.get("macro_f1", 0.0)),
    }


def _run_model(model: str, dataset: Any, seed: int) -> dict[str, Any]:
    if model == "simple":
        return cross_validate_simple_models(
            dataset.eeg,
            dataset.y,
            dataset.fs,
            dataset.channel_names,
            dataset.label_names,
            folds=5,
            seed=seed,
        )
    if model == "fbcsp-lda":
        return cross_validate_filterbank_lda(
            dataset.eeg,
            dataset.y,
            dataset.fs,
            dataset.channel_names,
            dataset.label_names,
            folds=5,
            seed=seed,
            top_k=48,
        )
    if model == "riemann":
        return cross_validate_riemannian_models(
            dataset.eeg,
            dataset.y,
            dataset.fs,
            dataset.channel_names,
            dataset.label_names,
            folds=5,
            seed=seed,
        )
    if model == "minirocket":
        return cross_validate_minirocket_models(
            dataset.eeg,
            dataset.y,
            dataset.fs,
            dataset.channel_names,
            dataset.label_names,
            folds=5,
            seed=seed,
        )
    raise ValueError(f"Unknown model: {model}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "model",
        "seed",
        "inner_model",
        "feature_set",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "n_trials",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for case in CASES:
        case_id = case["case_id"]
        for model in MODELS:
            vals = [float(row["balanced_accuracy"]) for row in rows if row["case_id"] == case_id and row["model"] == model]
            if not vals:
                continue
            summary.append(
                {
                    "case_id": case_id,
                    "model": model,
                    "seeds": len(vals),
                    "balanced_mean": mean(vals),
                    "balanced_std": pstdev(vals) if len(vals) > 1 else 0.0,
                    "balanced_min": min(vals),
                    "balanced_max": max(vals),
                }
            )
    return summary


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["case_id", "model", "seeds", "balanced_mean", "balanced_std", "balanced_min", "balanced_max"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def plot_summary(rows: list[dict[str, Any]], path: Path) -> None:
    width, height = 1200, 700
    margin_left, margin_right, margin_top, margin_bottom = 95, 30, 70, 115
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = _font(24)
    label_font = _font(15)
    small_font = _font(13)
    draw.text((margin_left, 22), "8-class seed stability, mean +/- std", font=title_font, fill=(17, 24, 39))
    max_y = 0.4
    for tick in [0.0, 0.1, 0.2, 0.3, 0.4]:
        y = margin_top + plot_h - int((tick / max_y) * plot_h)
        draw.line((margin_left, y, width - margin_right, y), fill=(229, 231, 235), width=1)
        draw.text((18, y - 8), f"{tick:.2f}", font=small_font, fill=(75, 85, 99))
    chance_y = margin_top + plot_h - int((0.125 / max_y) * plot_h)
    draw.line((margin_left, chance_y, width - margin_right, chance_y), fill=(107, 114, 128), width=2)
    draw.text((width - margin_right - 92, chance_y + 8), "chance", font=small_font, fill=(75, 85, 99))
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill=(55, 65, 81), width=2)
    draw.line((margin_left, margin_top + plot_h, width - margin_right, margin_top + plot_h), fill=(55, 65, 81), width=2)
    group_w = plot_w / len(CASES)
    bar_w = 48
    for case_idx, case in enumerate(CASES):
        case_id = case["case_id"]
        group_left = margin_left + case_idx * group_w
        group_center = group_left + group_w / 2
        for model_idx, model in enumerate(MODELS):
            match = [row for row in rows if row["case_id"] == case_id and row["model"] == model]
            if not match:
                continue
            row = match[0]
            value = float(row["balanced_mean"])
            std = float(row["balanced_std"])
            x0 = int(group_center - (len(MODELS) * bar_w) / 2 + model_idx * bar_w + 4)
            x1 = int(x0 + bar_w - 8)
            y1 = margin_top + plot_h
            y0 = margin_top + plot_h - int((value / max_y) * plot_h)
            draw.rectangle((x0, y0, x1, y1), fill=COLORS.get(model, (107, 114, 128)))
            err_top = margin_top + plot_h - int((min(max_y, value + std) / max_y) * plot_h)
            err_bottom = margin_top + plot_h - int((max(0.0, value - std) / max_y) * plot_h)
            mid = (x0 + x1) // 2
            draw.line((mid, err_top, mid, err_bottom), fill=(17, 24, 39), width=2)
            draw.line((mid - 6, err_top, mid + 6, err_top), fill=(17, 24, 39), width=2)
            draw.line((mid - 6, err_bottom, mid + 6, err_bottom), fill=(17, 24, 39), width=2)
            draw.text((x0 - 4, y0 - 22), f"{value:.3f}", font=small_font, fill=(31, 41, 55))
        label = ["A ljy7.4 all", "B ljy7.42 all", "C latest+ljy7.42"][case_idx]
        bbox = draw.textbbox((0, 0), label, font=label_font)
        draw.text((int(group_center - (bbox[2] - bbox[0]) / 2), margin_top + plot_h + 24), label, font=label_font, fill=(31, 41, 55))
    legend_x = margin_left
    legend_y = height - 42
    for model in MODELS:
        draw.rectangle((legend_x, legend_y, legend_x + 18, legend_y + 18), fill=COLORS[model])
        draw.text((legend_x + 26, legend_y - 1), model, font=small_font, fill=(31, 41, 55))
        legend_x += 125
    image.convert("RGB").save(path)


def main() -> int:
    bench_root = _bench_root()
    data_root = _data_root(bench_root)
    out_root = bench_root / "outputs" / "mi_me_ljy_20260704_study" / "02_reports"
    plots_root = out_root / "plots"
    out_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)
    config = ContinuousPreprocessConfig(
        bad_channel_mode="manual",
        bad_channels=("F4",),
        reject_z=999999,
        baseline_mode="pre_cue",
    )

    rows: list[dict[str, Any]] = []
    for case in CASES:
        paths = [data_root / rel for rel in case["files"]]
        print(f"load {case['case_id']}", flush=True)
        dataset = load_continuous_dataset(paths, config=config, label_mode="mi_me8")
        for seed in SEEDS:
            for model in MODELS_BY_CASE[str(case["case_id"])]:
                print(f"run case={case['case_id']} model={model} seed={seed}", flush=True)
                result = _run_model(model, dataset, seed)
                metrics = _extract(model, result)
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "model": model,
                        "seed": seed,
                        "n_trials": int(dataset.shape[0]),
                        **metrics,
                    }
                )
    summary = summarize(rows)
    write_csv(out_root / "mi_me8_seed_sweep.csv", rows)
    write_summary_csv(out_root / "mi_me8_seed_sweep_summary.csv", summary)
    (out_root / "mi_me8_seed_sweep.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "seeds": SEEDS,
                "rows": rows,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_summary(summary, plots_root / "eight_class_seed_sweep.png")
    print(f"rows={len(rows)}")
    print(f"out_root={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
