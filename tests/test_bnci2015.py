import unittest

import numpy as np

from wyu_bci_bench.foundation.bnci2015 import (
    BNCI2015_004_CHANNELS,
    BNCI2015_004_EVENT_IDS,
    BNCI2015_004_TASKS,
    make_grouped_dataset,
)


class BNCI2015Tests(unittest.TestCase):
    def test_moabb_event_contract_is_five_class_dataset(self):
        self.assertEqual(BNCI2015_004_EVENT_IDS,
                         {"math": 1, "letter": 2, "rotation": 3, "count": 4, "baseline": 5})
        self.assertEqual(BNCI2015_004_TASKS["math"], "mental_subtraction")
        self.assertEqual(BNCI2015_004_TASKS["letter"], "word_association")
        self.assertEqual(len(BNCI2015_004_CHANNELS), 30)

    def test_wraps_audited_three_class_windows(self):
        x = np.zeros((6, 30, 1024), dtype=np.float32)
        dataset = make_grouped_dataset(
            x, np.array([0, 1, 2, 0, 1, 2]), np.array(["P1"] * 6),
            np.array(["day1"] * 3 + ["day2"] * 3), np.array(["all"] * 6),
            source="audited.npz", channel_names=BNCI2015_004_CHANNELS,
            class_names=["MI", "math", "word"],
        )
        self.assertEqual(dataset.X.shape, (6, 30, 1024))
        self.assertEqual(dataset.fs, 256.0)
        with self.assertRaisesRegex(ValueError, "expected 30"):
            make_grouped_dataset(np.zeros((2, 29, 1024)), np.array([0, 1]),
                                 np.array(["P1"] * 2), np.array(["day1"] * 2),
                                 np.array(["all"] * 2), source="bad.npz",
                                 channel_names=BNCI2015_004_CHANNELS,
                                 class_names=["MI", "math", "word"])


if __name__ == "__main__":
    unittest.main()
