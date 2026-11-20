import unittest

import numpy as np

from wyu_bci_bench.event_waveform import (
    ClassTemplateProjection,
    XDawnTangentLogistic,
    low_frequency_waveforms,
)


class EventWaveformTests(unittest.TestCase):
    def test_low_frequency_prefix_filters_only_prefix(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(4, 5, 480))
        changed_future = x.copy()
        changed_future[..., 240:] += 20
        first = low_frequency_waveforms(x, prefix_seconds=2.0)
        second = low_frequency_waveforms(changed_future, prefix_seconds=2.0)
        self.assertEqual(first.shape, (4, 5, 20))
        np.testing.assert_allclose(first, second, atol=1e-6)
        self.assertTrue(np.isfinite(first).all())

    def test_xdawn_tangent_predicts_finite_probabilities(self):
        rng = np.random.default_rng(4)
        y = np.repeat(np.arange(3), 12)
        X = rng.normal(scale=0.2, size=(len(y), 6, 30))
        for cls in range(3):
            X[y == cls, cls, 5:20] += 0.8
        model = XDawnTangentLogistic(n_components_per_class=2).fit(X, y)
        p = model.predict_proba(X[:5])
        self.assertEqual(p.shape, (5, 3))
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)

    def test_template_projection_and_absolute_variant(self):
        rng = np.random.default_rng(5)
        y = np.repeat(np.arange(3), 10)
        X = rng.normal(size=(len(y), 7, 25))
        for cls in range(3):
            X[y == cls, cls, :] += np.sin(np.linspace(0, np.pi, 25))
        signed = ClassTemplateProjection(n_components_per_class=2).fit(X, y)
        absolute = ClassTemplateProjection(
            n_components_per_class=2, absolute_correlation=True
        ).fit(X, y)
        p = absolute.predict_proba(-X[:4])
        self.assertEqual(signed.transform(X[:4]).shape, (4, 6))
        self.assertEqual(p.shape, (4, 3))
        self.assertTrue(np.isfinite(p).all())

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            low_frequency_waveforms(np.full((2, 3, 20), np.nan))


if __name__ == "__main__":
    unittest.main()
