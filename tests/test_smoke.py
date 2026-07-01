from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from wyu_bci_bench.data import load_dataset, load_npz_recording
from wyu_bci_bench.evaluate import cross_validate_filterbank_lda
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
