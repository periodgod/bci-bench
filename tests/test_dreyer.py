"""Keep mislabeled ocular/muscle channels out of the model, with audited events."""
import unittest
import numpy as np
import mne
from wyu_bci_bench.dreyer import DREYER_EEG_CHANNELS, DREYER_EVENTS
from wyu_bci_bench.external_event_data import extract_annotated_windows


class DreyerTests(unittest.TestCase):
    def test_event_map_and_cue_windows(self):
        names = list(DREYER_EEG_CHANNELS)
        self.assertEqual(len(names), 27)
        self.assertFalse(set(names) & {'EOG1', 'EOG2', 'EOG3', 'EMGg', 'EMGd'})
        info = mne.create_info(names, 256, 'eeg')
        raw = mne.io.RawArray(np.ones((27, 2560)) * 2e-6, info, verbose='ERROR')
        raw.set_annotations(mne.Annotations([0, 1, 5], [0, 0, 0], ['768', '769', '770']))
        x, y, onset = extract_annotated_windows(raw, DREYER_EVENTS,
            channel_names=names, offset_seconds=0, duration_seconds=4,
            target_fs=256, car=False)
        self.assertEqual(x.shape, (2, 27, 1024))
        np.testing.assert_array_equal(y, [0, 1])
        np.testing.assert_array_equal(onset, [1, 5])
        np.testing.assert_allclose(x, 2)
