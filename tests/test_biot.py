import unittest

import numpy as np


class BIOTAdapterTests(unittest.TestCase):
    def setUp(self):
        from wyu_bci_bench.foundation.biot import BIOT_16_CHANNELS

        self.channels = BIOT_16_CHANNELS

    def test_prepares_exact_official_order_and_normalization(self):
        from wyu_bci_bench.foundation.biot import prepare_input

        order = np.random.default_rng(4).permutation(len(self.channels))
        source_channels = [self.channels[i] for i in order]
        X = np.zeros((2, len(source_channels), 2000), dtype=np.float32)
        for index, name in enumerate(source_channels):
            X[:, index] = (index + 1) * np.sin(np.arange(2000) / 15)
        prepared = prepare_input(X, 200, source_channels, self.channels)
        self.assertEqual(prepared.shape, (2, 16, 2000))
        self.assertTrue(np.isfinite(prepared).all())
        expected = X[:, [source_channels.index(name) for name in self.channels]]
        expected /= np.quantile(np.abs(expected), 0.95, axis=-1, keepdims=True) + 1e-8
        np.testing.assert_allclose(prepared, expected, rtol=1e-6, atol=1e-6)

    def test_missing_or_noncanonical_channels_refuse(self):
        from wyu_bci_bench.foundation.biot import prepare_input

        X = np.zeros((1, 15, 2000), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "missing"):
            prepare_input(X, 200, self.channels[:15], self.channels)
        with self.assertRaisesRegex(ValueError, "official ordered"):
            prepare_input(X, 200, self.channels[:15], self.channels[:15])

    def test_resamples_to_ten_seconds_at_200_hz(self):
        from wyu_bci_bench.foundation.biot import prepare_input

        X = np.random.default_rng(9).normal(size=(1, 16, 1200)).astype(np.float32)
        prepared = prepare_input(X, 120, self.channels, self.channels)
        self.assertEqual(prepared.shape, (1, 16, 2000))

    def test_rejects_wrong_duration_and_nonfinite_input(self):
        from wyu_bci_bench.foundation.biot import prepare_input

        X = np.zeros((1, 16, 1000), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "duration"):
            prepare_input(X, 200, self.channels, self.channels)
        X[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            prepare_input(X, 200, self.channels, self.channels)

    def test_load_encoder_rejects_invalid_channels_before_loading_repo(self):
        from wyu_bci_bench.foundation.biot import load_encoder

        with self.assertRaisesRegex(ValueError, "16 or 18"):
            load_encoder("missing", "missing", n_channels=17)


if __name__ == "__main__":
    unittest.main()
