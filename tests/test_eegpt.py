import unittest

import numpy as np


class EEGPTAdapterTests(unittest.TestCase):
    def setUp(self):
        from wyu_bci_bench.foundation.eegpt import EEGPT_CHANNELS

        self.channels = EEGPT_CHANNELS

    def test_prepares_official_order_resampling_and_explicit_scale(self):
        from scipy.signal import resample
        from wyu_bci_bench.foundation.eegpt import prepare_input

        order = np.random.default_rng(4).permutation(len(self.channels))
        source_channels = [self.channels[i] for i in order]
        X = np.zeros((2, len(source_channels), 800), dtype=np.float32)
        for index in range(len(source_channels)):
            X[:, index] = (index + 1) * np.sin(np.arange(800) / 15)

        prepared = prepare_input(X, 200, source_channels, amplitude_scale=0.001)
        expected = X[:, [source_channels.index(name) for name in self.channels]]
        expected = resample(expected, 1024, axis=-1).astype(np.float32) * 0.001
        self.assertEqual(prepared.shape, (2, 58, 1024))
        np.testing.assert_allclose(prepared, expected, rtol=1e-6, atol=1e-7)

    def test_requires_exact_channel_identity_and_unique_names(self):
        from wyu_bci_bench.foundation.eegpt import prepare_input

        X = np.zeros((1, 57, 1024), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "missing"):
            prepare_input(X, 256, self.channels[:-1], amplitude_scale=1.0)
        duplicate = list(self.channels)
        duplicate[-1] = duplicate[0]
        X = np.zeros((1, 58, 1024), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "uniquely"):
            prepare_input(X, 256, duplicate, amplitude_scale=1.0)

    def test_rejects_bad_duration_scale_and_nonfinite_samples(self):
        from wyu_bci_bench.foundation.eegpt import prepare_input

        with self.assertRaisesRegex(ValueError, "4-second"):
            prepare_input(np.zeros((1, 58, 512)), 256, self.channels, amplitude_scale=1.0)
        with self.assertRaisesRegex(ValueError, "amplitude_scale"):
            prepare_input(np.zeros((1, 58, 1024)), 256, self.channels, amplitude_scale=0.0)
        X = np.zeros((1, 58, 1024), dtype=np.float32)
        X[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            prepare_input(X, 256, self.channels, amplitude_scale=1.0)

    def test_contract_is_registered_and_verified(self):
        from wyu_bci_bench.foundation import require_verified

        self.assertTrue(require_verified("eegpt").verified)


if __name__ == "__main__":
    unittest.main()
