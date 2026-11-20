import unittest

import mne
import numpy as np

from wyu_bci_bench.external_event_data import (
    extract_annotated_windows,
    extract_sliding_windows,
    extract_windows_from_raw,
)


class ExternalEventDataTests(unittest.TestCase):
    def setUp(self):
        sfreq = 20.0
        times = np.arange(80) / sfreq
        data = np.vstack([np.sin(2 * np.pi * times), np.cos(2 * np.pi * times)]) * 1e-6
        self.raw = mne.io.RawArray(data, mne.create_info(["C3", "C4"], sfreq, "eeg"), verbose="ERROR")

    def test_event_windows_resample_car_and_preserve_order(self):
        windows = extract_windows_from_raw(self.raw, [0.0, 1.0], channel_names=["C3", "C4"],
                                           duration_seconds=1.0, target_fs=10.0)
        self.assertEqual(windows.shape, (2, 2, 10))
        np.testing.assert_allclose(windows.mean(axis=1), 0, atol=1e-6)
        self.assertFalse(np.allclose(windows[0], windows[1]))

    def test_sliding_windows_include_only_complete_windows(self):
        windows = extract_sliding_windows(self.raw, channel_names=["C3", "C4"], duration_seconds=1.0,
                                          step_seconds=0.5, target_fs=10.0)
        self.assertEqual(windows.shape, (7, 2, 10))

    def test_missing_channel_and_out_of_bounds_window_raise(self):
        with self.assertRaises(ValueError):
            extract_windows_from_raw(self.raw, [0], channel_names=["Fz"], duration_seconds=1)
        with self.assertRaises(ValueError):
            extract_windows_from_raw(self.raw, [3.5], channel_names=["C3"], duration_seconds=1)

    def test_annotated_windows_map_labels_and_keep_recording_order(self):
        self.raw.set_annotations(mne.Annotations([2.0, 0.5], [0, 0], ["770", "769"]))
        windows, labels, onsets = extract_annotated_windows(
            self.raw, {"769": 0, "770": 1}, channel_names=["C3", "C4"],
            offset_seconds=0.25, duration_seconds=0.5, target_fs=10,
        )
        np.testing.assert_array_equal(labels, [0, 1])
        np.testing.assert_allclose(onsets, [0.5, 2.0])
        self.assertEqual(windows.shape, (2, 2, 5))
        with self.assertRaises(ValueError):
            extract_annotated_windows(self.raw, {"768": 0}, channel_names=["C3"],
                                       offset_seconds=0, duration_seconds=0.5)


if __name__ == "__main__":
    unittest.main()
