from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wyu_bci_bench.continuous import ContinuousPreprocessConfig, load_continuous_dataset
from wyu_bci_bench.data import load_dataset, load_npz_recording
from wyu_bci_bench.evaluate import (
    cross_validate_filterbank_lda,
    cross_validate_ml_suite_models,
    cross_validate_minirocket_models,
    cross_validate_riemannian_models,
    cross_validate_simple_models,
    cross_validate_voting_ensemble,
    cross_validate_xgb_models,
)
from wyu_bci_bench.features import filterbank_spatial_logvar_features
from wyu_bci_bench.models import StandardizedDiagonalLDA
from wyu_bci_bench.viz import plot_band_energy_heatmap, plot_psd


CHANNELS = ["FC3", "FCz", "FC4", "C3", "C1", "Cz", "C2", "C4", "CP3", "CPz", "CP4"]


def synthetic_eeg(trials: int = 24, fs: int = 250, samples: int = 500) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(samples, dtype=np.float32) / float(fs)
    rng = np.random.default_rng(7)
    eeg = 0.05 * rng.standard_normal((trials, len(CHANNELS), samples)).astype(np.float32)
    labels = []
    c3 = CHANNELS.index("C3")
    c4 = CHANNELS.index("C4")
    for idx in range(trials):
        if idx % 2 == 0:
            eeg[idx, c3] += 1.2 * np.sin(2 * np.pi * 10.0 * t)
            labels.append("left_hand_mi")
        else:
            eeg[idx, c4] += 1.2 * np.sin(2 * np.pi * 10.0 * t)
            labels.append("right_hand_mi")
    return eeg.astype(np.float32), np.asarray(labels, dtype=object)


class SmokeTests(unittest.TestCase):
    def test_load_dataset_from_npz(self) -> None:
        eeg, labels = synthetic_eeg()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.npz"
            np.savez(path, eeg=eeg, y=labels, fs=250, channel_names=np.asarray(CHANNELS, dtype=object))
            recording = load_npz_recording(path)
            self.assertEqual(recording.shape, eeg.shape)
            dataset = load_dataset(tmp, label_mode="binary_mi", channel_mode="motor11")
            self.assertEqual(dataset.shape[0], 24)
            self.assertEqual(dataset.label_names, ["left_hand_mi", "right_hand_mi"])

    def test_features_and_lda(self) -> None:
        eeg, labels = synthetic_eeg()
        y = np.asarray([0 if item == "left_hand_mi" else 1 for item in labels], dtype=np.int64)
        features, names = filterbank_spatial_logvar_features(eeg, 250, CHANNELS)
        self.assertEqual(features.shape[0], 24)
        self.assertGreater(features.shape[1], len(CHANNELS))
        self.assertEqual(features.shape[1], len(names))
        model = StandardizedDiagonalLDA().fit(features, y)
        pred = model.predict(features)
        self.assertGreaterEqual(float(np.mean(pred == y)), 0.9)

    def test_cross_validate_filterbank_lda(self) -> None:
        eeg, labels = synthetic_eeg()
        y = np.asarray([0 if item == "left_hand_mi" else 1 for item in labels], dtype=np.int64)
        result = cross_validate_filterbank_lda(eeg, y, 250, CHANNELS, ["left_hand_mi", "right_hand_mi"], folds=4)
        self.assertIn("balanced_accuracy", result)
        self.assertGreaterEqual(float(result["balanced_accuracy"]), 0.8)

    def test_cross_validate_simple_models(self) -> None:
        eeg, labels = synthetic_eeg(trials=32)
        y = np.asarray([0 if item == "left_hand_mi" else 1 for item in labels], dtype=np.int64)
        result = cross_validate_simple_models(
            eeg,
            y,
            250,
            CHANNELS,
            ["left_hand_mi", "right_hand_mi"],
            folds=4,
            top_k=16,
        )
        self.assertIn("best", result)
        self.assertIn("models", result)
        self.assertGreaterEqual(float(result["best"]["balanced_accuracy"]), 0.8)

    def test_cross_validate_riemannian_models(self) -> None:
        eeg, labels = synthetic_eeg(trials=32)
        y = np.asarray([0 if item == "left_hand_mi" else 1 for item in labels], dtype=np.int64)
        result = cross_validate_riemannian_models(
            eeg,
            y,
            250,
            CHANNELS,
            ["left_hand_mi", "right_hand_mi"],
            folds=4,
        )
        self.assertIn("best", result)
        self.assertIn("models", result)
        self.assertGreaterEqual(float(result["best"]["balanced_accuracy"]), 0.8)

    def test_cross_validate_minirocket_models(self) -> None:
        eeg, labels = synthetic_eeg(trials=32)
        y = np.asarray([0 if item == "left_hand_mi" else 1 for item in labels], dtype=np.int64)
        result = cross_validate_minirocket_models(
            eeg,
            y,
            250,
            CHANNELS,
            ["left_hand_mi", "right_hand_mi"],
            folds=4,
            seed=7,
        )
        self.assertIn("best", result)
        self.assertIn("models", result)
        self.assertGreaterEqual(float(result["best"]["balanced_accuracy"]), 0.6)

    def test_cross_validate_xgb_models(self) -> None:
        try:
            import xgboost  # noqa: F401
        except Exception:
            self.skipTest("xgboost is not installed")
        eeg, labels = synthetic_eeg(trials=32)
        y = np.asarray([0 if item == "left_hand_mi" else 1 for item in labels], dtype=np.int64)
        result = cross_validate_xgb_models(
            eeg,
            y,
            250,
            CHANNELS,
            ["left_hand_mi", "right_hand_mi"],
            folds=4,
            seed=7,
        )
        self.assertIn("best", result)
        self.assertIn("models", result)

    def test_cross_validate_ml_suite_models(self) -> None:
        try:
            import sklearn  # noqa: F401
        except Exception:
            self.skipTest("scikit-learn is not installed")
        eeg, labels = synthetic_eeg(trials=32)
        y = np.asarray([0 if item == "left_hand_mi" else 1 for item in labels], dtype=np.int64)
        result = cross_validate_ml_suite_models(
            eeg,
            y,
            250,
            CHANNELS,
            ["left_hand_mi", "right_hand_mi"],
            folds=4,
            seed=7,
        )
        self.assertIn("best", result)
        self.assertIn("models", result)

    def test_cross_validate_vote_simple(self) -> None:
        try:
            import xgboost  # noqa: F401
        except Exception:
            self.skipTest("xgboost is not installed")
        eeg, labels = synthetic_eeg(trials=24)
        y = np.asarray([0 if item == "left_hand_mi" else 1 for item in labels], dtype=np.int64)
        result = cross_validate_voting_ensemble(
            eeg,
            y,
            250,
            CHANNELS,
            ["left_hand_mi", "right_hand_mi"],
            folds=2,
            seed=7,
            include_capsule=False,
        )
        self.assertIn("balanced_accuracy", result)
        self.assertIn("voter_metrics", result)

    def test_cross_validate_vote_ml_best_smoke(self) -> None:
        try:
            import sklearn  # noqa: F401
            import xgboost  # noqa: F401
        except Exception:
            self.skipTest("scikit-learn or xgboost is not installed")
        eeg, labels = synthetic_eeg(trials=24)
        y = np.asarray([0 if item == "left_hand_mi" else 1 for item in labels], dtype=np.int64)
        result = cross_validate_voting_ensemble(
            eeg,
            y,
            250,
            CHANNELS,
            ["left_hand_mi", "right_hand_mi"],
            folds=2,
            seed=7,
            voter_set="ml_best",
        )
        self.assertIn("balanced_accuracy", result)
        self.assertEqual(result["model"], "vote_ml_best")
        self.assertIn("extra_trees_full_top128", result["voter_metrics"])

    def test_cross_validate_vote_ml_pruned_smoke(self) -> None:
        try:
            import sklearn  # noqa: F401
        except Exception:
            self.skipTest("scikit-learn is not installed")
        eeg, labels = synthetic_eeg(trials=24)
        y = np.asarray([0 if item == "left_hand_mi" else 1 for item in labels], dtype=np.int64)
        result = cross_validate_voting_ensemble(
            eeg,
            y,
            250,
            CHANNELS,
            ["left_hand_mi", "right_hand_mi"],
            folds=2,
            seed=7,
            voter_set="ml_pruned",
        )
        self.assertIn("balanced_accuracy", result)
        self.assertEqual(result["model"], "vote_ml_pruned")
        self.assertIn("extra_trees_full_top128", result["voter_metrics"])
        self.assertNotIn("xgb_eeg", result["voter_metrics"])

    def test_cross_validate_vote_capsule_smoke(self) -> None:
        try:
            import torch  # noqa: F401
            import xgboost  # noqa: F401
        except Exception:
            self.skipTest("torch or xgboost is not installed")
        eeg, labels = synthetic_eeg(trials=24)
        y = np.asarray([0 if item == "left_hand_mi" else 1 for item in labels], dtype=np.int64)
        result = cross_validate_voting_ensemble(
            eeg,
            y,
            250,
            CHANNELS,
            ["left_hand_mi", "right_hand_mi"],
            folds=2,
            seed=7,
            include_capsule=True,
            capsule_epochs=1,
        )
        self.assertIn("balanced_accuracy", result)
        self.assertIn("capsule_net", result["voter_metrics"])

    def test_cross_validate_vote_ml_pruned_capsule_smoke(self) -> None:
        try:
            import sklearn  # noqa: F401
            import torch  # noqa: F401
        except Exception:
            self.skipTest("scikit-learn or torch is not installed")
        eeg, labels = synthetic_eeg(trials=24)
        y = np.asarray([0 if item == "left_hand_mi" else 1 for item in labels], dtype=np.int64)
        result = cross_validate_voting_ensemble(
            eeg,
            y,
            250,
            CHANNELS,
            ["left_hand_mi", "right_hand_mi"],
            folds=2,
            seed=7,
            include_capsule=True,
            capsule_epochs=1,
            voter_set="ml_pruned",
        )
        self.assertEqual(result["model"], "vote_ml_pruned_capsule")
        self.assertIn("capsule_net", result["voter_metrics"])
        self.assertNotIn("xgb_eeg", result["voter_metrics"])

    def test_continuous_raw_loader_epoching_and_bad_channels(self) -> None:
        fs = 250
        channel_names = np.asarray(["Fp1", "C3", "Cz", "C4"], dtype=object)
        samples = 1400
        t = np.arange(samples, dtype=np.float32) / float(fs)

        def write_segment(path: Path, phase: float) -> None:
            raw = 0.02 * np.sin(2 * np.pi * 10.0 * t[:, None] + phase).astype(np.float32)
            raw = np.repeat(raw, len(channel_names), axis=1)
            raw[:, 0] = 1000.0 * np.sin(2 * np.pi * 1.0 * t + phase)
            raw[120:620, 1] += 1.0 * np.sin(2 * np.pi * 10.0 * t[:500])
            raw[620:1120, 3] += 1.0 * np.sin(2 * np.pi * 12.0 * t[:500])
            records = [
                {"class_id": 1, "task_start_ms": 480.0, "task_end_ms": 2480.0},
                {"class_id": 2, "task_start_ms": 2480.0, "task_end_ms": 4480.0},
                {"class_id": 1, "task_start_ms": 5200.0, "task_end_ms": 6100.0},
            ]
            np.savez(
                path,
                raw_stream=raw.astype(np.float32),
                fs=fs,
                raw_channel_names=channel_names,
                trial_records_json=np.asarray(json.dumps(records, ensure_ascii=False)),
            )

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.npz"
            second = Path(tmp) / "second.npz"
            write_segment(first, 0.0)
            write_segment(second, 0.4)
            dataset = load_continuous_dataset(
                tmp,
                config=ContinuousPreprocessConfig(
                    eeg_channels=4,
                    apply_filter=False,
                    task_offset_s=0.0,
                    baseline_s=0.1,
                    bad_channel_mode="auto",
                    align_timeline_to_raw_end=False,
                ),
            )
            self.assertEqual(dataset.shape, (4, 3, 500))
            self.assertNotIn("Fp1", dataset.channel_names)
            self.assertEqual(dataset.source_trial_counts, [2, 2])
            self.assertEqual(dataset.label_names, ["C1 左手想象", "C2 左手执行"])
            self.assertEqual(
                dataset.metadata["epoch_info"]["reject_reasons"]["crosses_source_boundary"],
                2,
            )

    def test_continuous_mi_me_label_modes(self) -> None:
        fs = 250
        channel_names = np.asarray(["C3", "Cz", "C4", "Pz"], dtype=object)
        raw = 0.01 * np.random.default_rng(11).standard_normal((9000, len(channel_names))).astype(np.float32)
        records = [
            {"class_id": cid, "task_start_ms": float(idx * 1000), "task_end_ms": float(idx * 1000 + 800)}
            for idx, cid in enumerate(range(1, 9))
        ]

        with tempfile.TemporaryDirectory() as tmp:
            np.savez(
                Path(tmp) / "mi_me.npz",
                raw_stream=raw,
                fs=fs,
                raw_channel_names=channel_names,
                trial_records_json=np.asarray(json.dumps(records, ensure_ascii=False)),
            )
            cfg = ContinuousPreprocessConfig(
                eeg_channels=4,
                apply_filter=False,
                task_offset_s=0.0,
                baseline_s=0.1,
                min_epoch_s=0.1,
                bad_channel_mode="off",
            )

            state = load_continuous_dataset(tmp, config=cfg, label_mode="mi_me_state")
            self.assertEqual(state.shape[0], 8)
            self.assertEqual(state.label_names, ["想象", "执行"])
            self.assertEqual(state.y.tolist(), [0, 1, 0, 1, 0, 1, 0, 1])

            body = load_continuous_dataset(tmp, config=cfg, label_mode="mi_me_body4")
            self.assertEqual(body.label_names, ["左手", "右手", "双手", "双脚"])
            self.assertEqual(body.y.tolist(), [0, 0, 1, 1, 2, 2, 3, 3])

            imagery = load_continuous_dataset(tmp, config=cfg, label_mode="mi_me_imagery_body4")
            self.assertEqual(imagery.shape[0], 4)
            self.assertEqual(imagery.y.tolist(), [0, 1, 2, 3])

            execution = load_continuous_dataset(tmp, config=cfg, label_mode="mi_me_execution_body4")
            self.assertEqual(execution.shape[0], 4)
            self.assertEqual(execution.y.tolist(), [0, 1, 2, 3])

    def test_svg_plots(self) -> None:
        eeg, labels = synthetic_eeg()
        y = np.asarray([0 if item == "left_hand_mi" else 1 for item in labels], dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmp:
            psd = plot_psd(eeg, 250, Path(tmp) / "psd.png", y, ["left_hand_mi", "right_hand_mi"])
            heatmap = plot_band_energy_heatmap(eeg, 250, CHANNELS, Path(tmp) / "heatmap.png")
            self.assertEqual(psd.suffix, ".svg")
            self.assertEqual(heatmap.suffix, ".svg")
            self.assertIn("<svg", psd.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
