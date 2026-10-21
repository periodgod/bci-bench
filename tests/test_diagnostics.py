from __future__ import annotations

import unittest

import numpy as np


class InterpretationTests(unittest.TestCase):
    def test_grouped_permutation_uses_joint_feature_groups(self):
        from sklearn.linear_model import LogisticRegression
        from wyu_bci_bench.interpretability import grouped_permutation_importance

        rng = np.random.default_rng(4)
        y = np.tile([0, 1], 80)
        X = rng.normal(size=(len(y), 3))
        X[:, 0] += (y * 2 - 1) * 2
        model = LogisticRegression().fit(X[:100], y[:100])
        result = grouped_permutation_importance(model, X[100:], y[100:], {"signal": [0], "noise": [1, 2]}, n_repeats=8)
        self.assertGreater(result["signal"]["mean"], result["noise"]["mean"])

    def test_channel_ablation_reports_score_change_not_retraining_gain(self):
        from wyu_bci_bench.interpretability import channel_ablation_scores

        y = np.tile([0, 1], 20)
        X = np.zeros((len(y), 2, 32), dtype=float)
        X[:, 0, :] = np.where(y[:, None] == 1, 1.0, -1.0)
        predict = lambda values: (values[:, 0].mean(axis=1) > 0).astype(int)
        result = channel_ablation_scores(predict, X, y, {"C3": [0]})
        self.assertEqual(result["baseline"], 1.0)
        self.assertLess(result["C3"], 0)

    def test_band_ablation_removes_only_requested_frequency_region(self):
        from wyu_bci_bench.interpretability import band_ablation_scores

        fs, n = 128, 512
        t = np.arange(n) / fs
        y = np.tile([0, 1], 16)
        X = np.random.default_rng(3).normal(scale=0.08, size=(len(y), 2, n))
        X[:, 0, :] += (y[:, None] == 1) * np.sin(2 * np.pi * 10 * t)
        predict = lambda values: (values[:, 0].std(axis=1) > 0.5).astype(int)
        scores = band_ablation_scores(predict, X, y, fs, {"mu": (8, 13), "beta": (18, 25)})
        self.assertEqual(scores["baseline"], 1.0)
        self.assertLess(scores["mu"], scores["beta"])


class CalibrationTests(unittest.TestCase):
    def test_temperature_fit_uses_validation_logits_and_improves_nll(self):
        from wyu_bci_bench.calibration import apply_temperature, fit_temperature

        rng = np.random.default_rng(5)
        y = np.tile(np.arange(3), 100)
        logits = rng.normal(size=(len(y), 3))
        logits[np.arange(len(y)), y] += 0.9
        temperature = fit_temperature(logits, y)
        scaled = apply_temperature(logits, temperature)
        self.assertGreater(temperature, 0)
        self.assertLessEqual(float(np.mean(-np.log(scaled[np.arange(len(y)), y]))),
                             float(np.mean(-np.log(apply_temperature(logits, 1.0)[np.arange(len(y)), y]))))

    def test_calibration_scores_and_risk_coverage_are_well_formed(self):
        from wyu_bci_bench.calibration import expected_calibration_error, multiclass_brier_score, risk_coverage_curve

        p = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.7, 0.3]])
        y = np.array([0, 1, 1, 0])
        self.assertGreaterEqual(expected_calibration_error(p, y, n_bins=4), 0)
        self.assertGreaterEqual(multiclass_brier_score(p, y), 0)
        coverage, risk = risk_coverage_curve(p, y)
        np.testing.assert_allclose(coverage[[0, -1]], [0, 1])
        self.assertEqual(len(coverage), len(y) + 1)
        self.assertEqual(len(risk), len(y) + 1)

    def test_invalid_probabilities_and_temperature_are_rejected(self):
        from wyu_bci_bench.calibration import apply_temperature, expected_calibration_error

        with self.assertRaisesRegex(ValueError, "sum to one"):
            expected_calibration_error(np.array([[0.2, 0.2]]), np.array([0]))
        with self.assertRaisesRegex(ValueError, "positive"):
            apply_temperature(np.ones((2, 2)), 0)


class OnlineEvaluationTests(unittest.TestCase):
    def test_causal_sos_filter_is_chunk_boundary_invariant(self):
        from wyu_bci_bench.online import CausalSOSFilter, design_causal_bandpass

        rng = np.random.default_rng(33)
        signal = rng.normal(size=(3, 2000))
        sos = design_causal_bandpass(4, 35, 128)
        whole = CausalSOSFilter(sos, channels=3).process(signal)
        streaming = CausalSOSFilter(sos, channels=3)
        pieces = np.concatenate([
            streaming.process(signal[:, :117]),
            streaming.process(signal[:, 117:901]),
            streaming.process(signal[:, 901:]),
        ], axis=1)
        np.testing.assert_allclose(pieces, whole, atol=1e-12)

    def test_stream_buffer_emits_overlapping_windows_with_absolute_offsets(self):
        from wyu_bci_bench.online import StreamWindowBuffer

        signal = np.arange(40, dtype=float).reshape(2, 20)
        buffer = StreamWindowBuffer(channels=2, window_samples=8, step_samples=4)
        windows = buffer.push(signal[:, :5]) + buffer.push(signal[:, 5:13]) + buffer.push(signal[:, 13:])
        self.assertEqual([start for start, _ in windows], [0, 4, 8, 12])
        for start, window in windows:
            np.testing.assert_array_equal(window, signal[:, start:start + 8])

    def test_streaming_benchmark_excludes_warmup_and_reports_deadlines(self):
        from wyu_bci_bench.online import benchmark_streaming

        seen = []
        result = benchmark_streaming(lambda x: seen.append(x.shape), [np.ones((2, 4))] * 4,
                                     warmup=1, deadline_ms=100)
        self.assertEqual(result["windows"], 3)
        self.assertEqual(len(seen), 4)
        self.assertEqual(result["deadline_misses"], 0)
        self.assertGreater(result["throughput_windows_s"], 0)

    def test_temporal_drift_is_binned_in_timestamp_order(self):
        from wyu_bci_bench.online import temporal_drift_summary

        truth = np.array([0, 1, 0, 1, 0, 1])
        pred = np.array([0, 1, 0, 0, 1, 0])
        time_order = np.array([5, 4, 3, 2, 1, 0])
        result = temporal_drift_summary(truth, pred, time_order, bins=3)
        self.assertEqual([row["count"] for row in result], [2, 2, 2])
        self.assertEqual([row["time_start"] for row in result], [0.0, 2.0, 4.0])
        self.assertEqual(result[0]["balanced_accuracy_delta_from_first"], 0.0)


if __name__ == "__main__":
    unittest.main()
