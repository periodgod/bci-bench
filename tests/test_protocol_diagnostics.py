import unittest

import numpy as np

from wyu_bci_bench.protocol_diagnostics import collapse_gate, collapse_summary, guard_comparison


class ProtocolDiagnosticsTests(unittest.TestCase):
    def test_collapse_summary_and_thresholds(self):
        y = np.tile(np.arange(3), 4)
        pred = y.copy()
        subjects = np.repeat(["P1", "P2"], 6)
        sessions = np.tile(np.repeat(["S1", "S2"], 3), 2)
        interfaces = np.full(len(y), "I1")
        summary = collapse_summary(y, pred, subjects, sessions, interfaces)
        self.assertEqual(summary["n_cells"], 4)
        self.assertEqual(summary["cells_with_zero_predicted_class"], 0)
        self.assertEqual(summary["cells_with_class_proportion_below_10pct"], 0)
        self.assertTrue(collapse_gate(summary)["pass"])

    def test_collapse_flags_missing_prediction_class(self):
        y = np.tile(np.arange(3), 4)
        pred = np.zeros_like(y)
        groups = (np.repeat(["P1", "P2"], 6), np.tile(np.repeat(["S1", "S2"], 3), 2),
                  np.full(len(y), "I1"))
        summary = collapse_summary(y, pred, *groups)
        self.assertEqual(summary["cells_with_zero_predicted_class"], 4)
        self.assertEqual(summary["n_subject_class_recalls_below_20pct"], 4)
        self.assertFalse(collapse_gate(summary)["pass"])

    def test_guard_comparison_applies_per_subject_floor(self):
        anchor = {"sub-P04": 0.5, "sub-P10": 0.6}
        self.assertTrue(guard_comparison({"sub-P04": 0.47, "sub-P10": 0.61}, anchor)["pass"])
        self.assertFalse(guard_comparison({"sub-P04": 0.469, "sub-P10": 0.61}, anchor)["pass"])

    def test_guard_requires_all_scores(self):
        with self.assertRaises(ValueError):
            guard_comparison({"sub-P04": 0.5}, {"sub-P04": 0.5})


if __name__ == "__main__":
    unittest.main()
