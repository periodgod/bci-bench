import unittest

import mne
import numpy as np
from unittest.mock import patch

from wyu_bci_bench.pretraining import preprocess_unlabelled_recording, read_unlabelled_edf


class PretrainingRecordTests(unittest.TestCase):
    def test_edf_loader_discards_event_annotations(self):
        raw = mne.io.RawArray(
            np.zeros((1, 1200)), mne.create_info(["Cz"], 120, ch_types="eeg"), verbose="ERROR"
        )
        raw.set_annotations(mne.Annotations([1.0], [0.0], ["left_hand"]))
        with patch("mne.io.read_raw_edf", return_value=raw):
            loaded = read_unlabelled_edf("synthetic.edf")
        self.assertEqual(len(loaded.annotations), 0)

    def test_coordinate_resolution_normalizes_common_eeg_labels(self):
        from wyu_bci_bench.pretraining import resolve_channel_coordinates

        coords, known = resolve_channel_coordinates(["EEG Fp1", "T3", "ECG ECG"])
        self.assertEqual(known.tolist(), [True, True, False])
        self.assertTrue(np.isfinite(coords[:2]).all())

    def test_recording_is_filtered_windowed_and_label_free(self):
        sfreq = 250.0
        times = np.arange(2500) / sfreq
        signal = np.stack([
            20e-6 * np.sin(2 * np.pi * 10 * times),
            -20e-6 * np.sin(2 * np.pi * 10 * times),
        ])
        raw = mne.io.RawArray(signal, mne.create_info(["C3", "C4"], sfreq, ch_types="eeg"), verbose="ERROR")

        windows, metadata = preprocess_unlabelled_recording(raw)

        self.assertEqual(windows.shape, (4, 2, 480))
        self.assertEqual(windows.dtype, np.float16)
        self.assertEqual(metadata["n_windows_total"], 4)
        self.assertEqual(metadata["channel_names"], ["C3", "C4"])
        self.assertEqual(metadata["dropped_channels"], [])
        self.assertEqual(raw.info["sfreq"], sfreq)
        self.assertEqual(raw.n_times, 2500)
        self.assertFalse(np.any(np.abs(windows) > 5))


if __name__ == "__main__":
    unittest.main()
