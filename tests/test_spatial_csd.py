import unittest
import numpy as np
from wyu_bci_bench.spatial_csd import csd_matrix, apply_csd


class CSDTests(unittest.TestCase):
    def test_matches_mne_epochs_and_removes_common_reference(self):
        import mne
        names = ["F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4", "Oz"]
        sphere = (0., 0., 0.04, 0.09)
        a, _ = csd_matrix(names, sphere=sphere)
        x = np.random.default_rng(9).normal(size=(3, len(names), 128))
        info = mne.create_info(names, 120, ch_types="eeg")
        info.set_montage(mne.channels.make_standard_montage("standard_1020"))
        ep = mne.EpochsArray(x, info, verbose=False)
        direct = mne.preprocessing.compute_current_source_density(ep, sphere=sphere).get_data()
        np.testing.assert_allclose(apply_csd(x, a), direct, rtol=1e-5, atol=1e-3)
        np.testing.assert_allclose(apply_csd(x + 500, a), apply_csd(x, a), rtol=1e-5, atol=1e-3)
        with self.assertRaises(ValueError):
            csd_matrix(["invented", "Cz", "Fz", "Pz"])
