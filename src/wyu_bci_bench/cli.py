"""Command line interface for WYU BCI Bench."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .continuous import ContinuousPreprocessConfig, load_continuous_dataset
from .data import load_dataset, load_npz_recording, scan_recordings, summarize_recordings
from .evaluate import (
    cross_validate_eegnet,
    cross_validate_filterbank_lda,
    cross_validate_ml_suite_models,
    cross_validate_minirocket_models,
    cross_validate_riemannian_models,
    cross_validate_simple_models,
    cross_validate_voting_ensemble,
    cross_validate_xgb_models,
    dataset_quality_report,
)
from .viz import (
    plot_band_energy_heatmap,
    plot_class_band_topomap,
    plot_label_distribution,
    plot_psd,
)

LABEL_MODE_CHOICES = [
    "all",
    "binary_mi",
    "six_movement",
    "mi_me8",
    "mi_me_state",
    "mi_me_body",
    "mi_me_body4",
    "mi_me_imagery_body",
    "mi_me_imagery_body4",
    "mi_me_execution_body",
    "mi_me_execution_body4",
]


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
    analyze.add_argument("--label-mode", choices=LABEL_MODE_CHOICES, default="all")
    analyze.add_argument("--channel-mode", choices=["all", "motor", "motor11", "central3"], default="motor11")
    analyze.add_argument("--continuous-raw", action="store_true", help="Load raw_stream + trial_records_json recordings")
    analyze.add_argument("--latest", type=int, default=None, help="Use latest N raw_stream recordings")
    analyze.add_argument("--bad-channel-mode", choices=["off", "auto", "strict", "manual"], default="auto")
    analyze.add_argument("--bad-channels", default="", help="Comma-separated manual bad channels for continuous raw")
    analyze.add_argument("--reject-z", type=float, default=8.0, help="Epoch peak rejection threshold in robust z units for continuous raw")
    analyze.add_argument("--baseline-mode", choices=["pre_cue", "epoch_start", "none"], default="pre_cue")
    analyze.add_argument("--no-align-timeline", action="store_true", help="Disable end-alignment between GUI event times and raw_stream samples")
    analyze.add_argument("--no-filter", action="store_true", help="Disable continuous Butterworth preprocessing")

    bench = sub.add_parser("bench", help="Run offline model benchmarks")
    bench.add_argument("recordings", type=Path, nargs="+")
    bench.add_argument("--out", type=Path, default=Path("outputs/benchmarks"))
    bench.add_argument("--label-mode", choices=LABEL_MODE_CHOICES, default="all")
    bench.add_argument("--channel-mode", choices=["all", "motor", "motor11", "central3"], default="motor11")
    bench.add_argument("--continuous-raw", action="store_true", help="Load raw_stream + trial_records_json recordings")
    bench.add_argument("--latest", type=int, default=None, help="Use latest N raw_stream recordings")
    bench.add_argument("--bad-channel-mode", choices=["off", "auto", "strict", "manual"], default="auto")
    bench.add_argument("--bad-channels", default="", help="Comma-separated manual bad channels for continuous raw")
    bench.add_argument("--reject-z", type=float, default=8.0, help="Epoch peak rejection threshold in robust z units for continuous raw")
    bench.add_argument("--baseline-mode", choices=["pre_cue", "epoch_start", "none"], default="pre_cue")
    bench.add_argument("--no-align-timeline", action="store_true", help="Disable end-alignment between GUI event times and raw_stream samples")
    bench.add_argument("--no-filter", action="store_true", help="Disable continuous Butterworth preprocessing")
    bench.add_argument(
        "--model",
        choices=[
            "fbcsp-lda",
            "simple",
            "riemann",
            "minirocket",
            "xgb",
            "ml-suite",
            "vote-simple",
            "vote-ml-best",
            "vote-ml-pruned",
            "vote-ml-pruned-capsule",
            "vote-capsule",
            "eegnet",
            "both",
            "all",
        ],
        default="fbcsp-lda",
    )
    bench.add_argument("--folds", type=int, default=5)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--epochs", type=int, default=30)
    bench.add_argument("--top-k", type=int, default=None, help="Feature count for FBCSP or simple models")
    return parser


def _continuous_config(args: argparse.Namespace) -> ContinuousPreprocessConfig:
    bad_channels = tuple(name.strip() for name in str(args.bad_channels).split(",") if name.strip())
    mode = "manual" if bad_channels and args.bad_channel_mode == "manual" else args.bad_channel_mode
    return ContinuousPreprocessConfig(
        apply_filter=not bool(args.no_filter),
        bad_channel_mode=mode,
        bad_channels=bad_channels,
        reject_z=float(args.reject_z),
        baseline_mode=str(args.baseline_mode),
        align_timeline_to_raw_end=not bool(args.no_align_timeline),
    )


def _load_dataset_from_args(args: argparse.Namespace):
    if getattr(args, "continuous_raw", False):
        return load_continuous_dataset(
            args.recordings,
            config=_continuous_config(args),
            count_latest=args.latest,
            label_mode=args.label_mode,
        )
    return load_dataset(args.recordings, label_mode=args.label_mode, channel_mode=args.channel_mode)


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
    dataset = _load_dataset_from_args(args)
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
    dataset = _load_dataset_from_args(args)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset.summary(),
        "models": {},
    }
    if args.model in {"fbcsp-lda", "both", "all"}:
        results["models"]["fbcsp_lda_stable"] = cross_validate_filterbank_lda(
            dataset.eeg,
            dataset.y,
            dataset.fs,
            dataset.channel_names,
            dataset.label_names,
            folds=args.folds,
            seed=args.seed,
            top_k=args.top_k if args.top_k is not None else 48,
        )
    if args.model in {"simple", "all"}:
        results["models"]["simple_logvar"] = cross_validate_simple_models(
            dataset.eeg,
            dataset.y,
            dataset.fs,
            dataset.channel_names,
            dataset.label_names,
            folds=args.folds,
            seed=args.seed,
            top_k=args.top_k if args.top_k is not None and args.top_k > 0 else None,
        )
    if args.model in {"riemann", "all"}:
        results["models"]["riemannian"] = cross_validate_riemannian_models(
            dataset.eeg,
            dataset.y,
            dataset.fs,
            dataset.channel_names,
            dataset.label_names,
            folds=args.folds,
            seed=args.seed,
        )
    if args.model in {"minirocket", "all"}:
        results["models"]["minirocket"] = cross_validate_minirocket_models(
            dataset.eeg,
            dataset.y,
            dataset.fs,
            dataset.channel_names,
            dataset.label_names,
            folds=args.folds,
            seed=args.seed,
        )
    if args.model in {"xgb", "all"}:
        results["models"]["xgb"] = cross_validate_xgb_models(
            dataset.eeg,
            dataset.y,
            dataset.fs,
            dataset.channel_names,
            dataset.label_names,
            folds=args.folds,
            seed=args.seed,
        )
    if args.model in {"ml-suite", "all"}:
        results["models"]["ml_suite"] = cross_validate_ml_suite_models(
            dataset.eeg,
            dataset.y,
            dataset.fs,
            dataset.channel_names,
            dataset.label_names,
            folds=args.folds,
            seed=args.seed,
        )
    if args.model in {"vote-simple", "all"}:
        results["models"]["vote_simple"] = cross_validate_voting_ensemble(
            dataset.eeg,
            dataset.y,
            dataset.fs,
            dataset.channel_names,
            dataset.label_names,
            folds=args.folds,
            seed=args.seed,
            include_capsule=False,
        )
    if args.model in {"vote-ml-best", "all"}:
        results["models"]["vote_ml_best"] = cross_validate_voting_ensemble(
            dataset.eeg,
            dataset.y,
            dataset.fs,
            dataset.channel_names,
            dataset.label_names,
            folds=args.folds,
            seed=args.seed,
            include_capsule=False,
            voter_set="ml_best",
        )
    if args.model in {"vote-ml-pruned", "all"}:
        results["models"]["vote_ml_pruned"] = cross_validate_voting_ensemble(
            dataset.eeg,
            dataset.y,
            dataset.fs,
            dataset.channel_names,
            dataset.label_names,
            folds=args.folds,
            seed=args.seed,
            include_capsule=False,
            voter_set="ml_pruned",
        )
    if args.model in {"vote-ml-pruned-capsule", "all"}:
        results["models"]["vote_ml_pruned_capsule"] = cross_validate_voting_ensemble(
            dataset.eeg,
            dataset.y,
            dataset.fs,
            dataset.channel_names,
            dataset.label_names,
            folds=args.folds,
            seed=args.seed,
            include_capsule=True,
            capsule_epochs=args.epochs,
            voter_set="ml_pruned",
        )
    if args.model in {"vote-capsule", "all"}:
        results["models"]["vote_capsule"] = cross_validate_voting_ensemble(
            dataset.eeg,
            dataset.y,
            dataset.fs,
            dataset.channel_names,
            dataset.label_names,
            folds=args.folds,
            seed=args.seed,
            include_capsule=True,
            capsule_epochs=args.epochs,
        )
    if args.model in {"eegnet", "both", "all"}:
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
        if "best" in metrics:
            best = metrics["best"]
            print(
                f"{name}: best={best['feature_set']}+{best['model']} "
                f"accuracy={best['accuracy']:.4f} balanced={best['balanced_accuracy']:.4f}"
            )
        else:
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
