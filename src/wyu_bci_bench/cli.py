"""Command line interface for WYU BCI Bench."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .data import load_dataset, load_npz_recording, scan_recordings, summarize_recordings
from .evaluate import cross_validate_eegnet, cross_validate_filterbank_lda, dataset_quality_report
from .viz import (
    plot_band_energy_heatmap,
    plot_class_band_topomap,
    plot_label_distribution,
    plot_psd,
)


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wyu-bci-bench", description="Offline EEG analysis and benchmark toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Scan local NPZ recordings")
    scan.add_argument("recordings", type=Path, nargs="+")
    scan.add_argument("--json", type=Path, default=None, help="Optional summary JSON path")

    analyze = sub.add_parser("analyze", help="Generate offline analysis report and plots")
    analyze.add_argument("recordings", type=Path, nargs="+")
    analyze.add_argument("--out", type=Path, default=Path("outputs/offline_analysis"))
    analyze.add_argument("--label-mode", choices=["all", "binary_mi", "six_movement"], default="all")
    analyze.add_argument("--channel-mode", choices=["all", "motor", "motor11", "central3"], default="motor11")

    bench = sub.add_parser("bench", help="Run offline model benchmarks")
    bench.add_argument("recordings", type=Path, nargs="+")
    bench.add_argument("--out", type=Path, default=Path("outputs/benchmarks"))
    bench.add_argument("--label-mode", choices=["all", "binary_mi", "six_movement"], default="all")
    bench.add_argument("--channel-mode", choices=["all", "motor", "motor11", "central3"], default="motor11")
    bench.add_argument("--model", choices=["fbcsp-lda", "eegnet", "both"], default="fbcsp-lda")
    bench.add_argument("--folds", type=int, default=5)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--epochs", type=int, default=30)
    bench.add_argument("--top-k", type=int, default=48)
    return parser


def cmd_scan(args: argparse.Namespace) -> int:
    files = scan_recordings(args.recordings)
    print(f"Found {len(files)} NPZ recordings")
    for item in summarize_recordings(args.recordings):
        if "error" in item:
            print(f"ERROR {item['path']}: {item['error']}")
        else:
            print(f"{item['path']} shape={item['shape']} fs={item['sampling_rate_hz']} labels={item['label_counts']}")
    if args.json:
        _write_json(args.json, {"generated_utc": datetime.now(timezone.utc).isoformat(), "recordings": summarize_recordings(args.recordings)})
        print(f"Summary JSON: {args.json.resolve()}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.recordings, label_mode=args.label_mode, channel_mode=args.channel_mode)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    quality = dataset_quality_report(dataset.eeg, dataset.y, dataset.fs, dataset.channel_names, dataset.label_names)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset.summary(),
        "quality": quality,
        "plots": {},
    }
    payload["plots"]["label_distribution"] = str(plot_label_distribution(dataset.labels, out / "label_distribution.png"))
    payload["plots"]["psd"] = str(plot_psd(dataset.eeg, dataset.fs, out / "psd.png", dataset.y, dataset.label_names))
    payload["plots"]["band_energy_heatmap"] = str(plot_band_energy_heatmap(dataset.eeg, dataset.fs, dataset.channel_names, out / "band_energy_heatmap.png"))
    topomap = plot_class_band_topomap(dataset.eeg, dataset.y, dataset.fs, dataset.channel_names, out / "mu_class_difference_topomap.png")
    if topomap is not None:
        payload["plots"]["mu_class_difference_topomap"] = str(topomap)
    report = out / "analysis_report.json"
    _write_json(report, payload)
    print(f"Dataset shape: {dataset.shape}, labels={dict((name, int((dataset.labels == name).sum())) for name in dataset.label_names)}")
    print(f"Analysis report: {report.resolve()}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.recordings, label_mode=args.label_mode, channel_mode=args.channel_mode)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset.summary(),
        "models": {},
    }
    if args.model in {"fbcsp-lda", "both"}:
        results["models"]["fbcsp_lda_stable"] = cross_validate_filterbank_lda(
            dataset.eeg,
            dataset.y,
            dataset.fs,
            dataset.channel_names,
            dataset.label_names,
            folds=args.folds,
            seed=args.seed,
            top_k=args.top_k,
        )
    if args.model in {"eegnet", "both"}:
        results["models"]["eegnet"] = cross_validate_eegnet(
            dataset.eeg,
            dataset.y,
            dataset.fs,
            dataset.channel_names,
            dataset.label_names,
            folds=args.folds,
            seed=args.seed,
            epochs=args.epochs,
        )
    report = out / "benchmark_report.json"
    _write_json(report, results)
    for name, metrics in results["models"].items():
        print(f"{name}: accuracy={metrics['accuracy']:.4f} balanced={metrics['balanced_accuracy']:.4f}")
    print(f"Benchmark report: {report.resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "scan":
        return cmd_scan(args)
    if args.command == "analyze":
        return cmd_analyze(args)
    if args.command == "bench":
        return cmd_bench(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
