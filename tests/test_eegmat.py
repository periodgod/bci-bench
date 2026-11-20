import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import mne
import numpy as np
from wyu_bci_bench.eegmat import EEGMAT_CHANNELS, load_eegmat_windows


class EEGMATTests(unittest.TestCase):
    def test_paired_labels_reference_exclusion_and_subject_metadata(self):
        names = [f'EEG {name}' for name in EEGMAT_CHANNELS]+['EEG A2-A1', 'ECG ECG']
        rng = np.random.default_rng(4)
        raw = mne.io.RawArray(rng.normal(0, 5e-6, (21, 256*40)),
            mne.create_info(names, 256, ch_types='eeg'), verbose='ERROR')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'README.txt').write_text('_1 baseline; _2 arithmetic', encoding='utf-8')
            for person in range(5):
                for suffix in ['1', '2']:
                    (root/f'Subject{person:02d}_{suffix}.edf').write_bytes(b'synthetic')
            with patch('wyu_bci_bench.eegmat.read_edf_with_events', side_effect=lambda path: raw.copy()):
                dataset = load_eegmat_windows(root, seconds_per_condition=4)
        self.assertEqual(dataset.X.shape, (10, 19, 1024))
        self.assertEqual(len(dataset.people), 5)
        self.assertEqual(dataset.unit, 'uV')
        self.assertEqual(dataset.channel_names, list(EEGMAT_CHANNELS))
        for person in dataset.people:
            ix = dataset.subjects == person
            self.assertEqual(set(dataset.y[ix]), {0, 1})
            self.assertEqual(len(set(dataset.sessions[ix])), 1)
            np.testing.assert_array_equal(dataset.recording_order[ix], [0, 1])
        self.assertEqual(len(dataset.metadata['records']), 10)
        self.assertEqual(len(raw.ch_names), 21)  # reader inputs are not modified in-place
