import unittest

import numpy as np

from wyu_bci_bench.session_causal import causal_center, causal_euclidean_align
from wyu_bci_bench.splits import chronological_cell_holdout, seen_later_session


class CausalCenterInitTest(unittest.TestCase):
    def test_default_unchanged_and_init_start(self):
        rng = np.random.default_rng(0)
        Z = rng.normal(size=(12, 4))
        keys = np.array(["a"] * 7 + ["b"] * 5)
        base = causal_center(Z, keys)
        self.assertTrue(np.allclose(base[0], Z[0]))
        init = rng.normal(size=4)
        out = causal_center(Z, keys, init=init)
        self.assertTrue(np.allclose(out[0], Z[0] - init))
        self.assertTrue(np.allclose(out[7], Z[7] - init))
        n = 3
        expected = Z[3] - (init + n / (n + 32.0) * (Z[:3].mean(0) - init))
        self.assertTrue(np.allclose(out[3], expected))

    def test_init_is_causal(self):
        rng = np.random.default_rng(1)
        Z = rng.normal(size=(10, 3))
        keys = np.array(["s"] * 10)
        init = np.ones(3)
        a = causal_center(Z, keys, init=init)
        Z2 = Z.copy()
        Z2[6:] += 100.0
        b = causal_center(Z2, keys, init=init)
        self.assertTrue(np.allclose(a[:6], b[:6]))


class CausalEuclideanAlignTest(unittest.TestCase):
    def test_causal_and_reset(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(9, 4, 50))
        keys = np.array(["a"] * 5 + ["b"] * 4)
        init = np.eye(4) * 2.0
        out = causal_euclidean_align(X, keys, init_cov=init)
        self.assertTrue(np.allclose(out[0], X[0] / np.sqrt(2.0)))
        self.assertTrue(np.allclose(out[5], X[5] / np.sqrt(2.0)))
        X2 = X.copy()
        X2[3:5] *= 50.0
        out2 = causal_euclidean_align(X2, keys, init_cov=init)
        self.assertTrue(np.allclose(out[:3], out2[:3]))
        self.assertTrue(np.allclose(out[5:], out2[5:]))

    def test_converges_to_whitening(self):
        rng = np.random.default_rng(3)
        A = np.eye(3) + 0.3 * rng.normal(size=(3, 3))
        X = np.einsum("cd,ndt->nct", A, rng.normal(size=(400, 3, 200)))
        out = causal_euclidean_align(X, np.zeros(400, dtype=int), alpha=1.0)
        tail = out[-50:] - out[-50:].mean(-1, keepdims=True)
        cov = np.einsum("nct,ndt->cd", tail, tail) / (50 * 200)
        self.assertTrue(np.allclose(cov, np.eye(3), atol=0.15))


class SplitTest(unittest.TestCase):
    def test_chronological_cell_holdout(self):
        cells = np.array(["x"] * 9 + ["y"] * 6)
        split = chronological_cell_holdout(cells, test_fraction=1 / 3)
        self.assertEqual(split.test.tolist(), [False] * 6 + [True] * 3 + [False] * 4 + [True] * 2)
        split.check_disjoint()

    def test_seen_later_session(self):
        subjects = np.array(["P04", "P04", "P04", "P01"])
        sessions = np.array(["ses-1", "ses-2", "ses-3", "ses-1"])
        split = seen_later_session(subjects, sessions, "ses-3", ["P04"])
        self.assertEqual(split.test.tolist(), [False, False, True, False])
        self.assertEqual(split.train.tolist(), [True, True, False, True])


class SoftmaxRegressionTest(unittest.TestCase):
    def test_feature_multiplier_and_frozen_logits(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        from wyu_bci_bench.linear_torch import fit_softmax_regression

        rng = np.random.default_rng(8)
        features = rng.normal(size=(180, 4))
        labels = (features[:, 0] > 0).astype(int) + (features[:, 1] > 0).astype(int)
        multiplier = np.array([2., 1., .5, 1.])
        weighted = fit_softmax_regression(features, labels, C=.1, feature_multiplier=multiplier)
        transformed = StandardScaler().fit_transform(features) * multiplier
        reference = LogisticRegression(C=.1, tol=1e-10).fit(transformed, labels)
        np.testing.assert_allclose(weighted.predict_proba(features), reference.predict_proba(transformed), atol=1e-3)

        offset = np.eye(3)[labels] * 4
        residual = fit_softmax_regression(features, labels, C=.1, offset_logits=offset)
        combined = residual.predict_proba(features, offset_logits=offset)
        np.testing.assert_array_equal(combined.argmax(axis=1), labels)
        self.assertLess(np.linalg.norm(residual.weight), np.linalg.norm(weighted.weight))
        with self.assertRaises(ValueError):
            residual.predict_proba(features, offset_logits=offset[:, :2])

    def test_matches_sklearn(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        from wyu_bci_bench.linear_torch import fit_softmax_regression

        rng = np.random.default_rng(4)
        F = rng.normal(size=(300, 20))
        y = (F[:, 0] + 0.5 * F[:, 1] > 0).astype(int) + (F[:, 2] > 1).astype(int)
        model = fit_softmax_regression(F, y, C=0.1)
        sk = LogisticRegression(C=0.1, max_iter=5000, tol=1e-10).fit(StandardScaler().fit_transform(F), y)
        z = StandardScaler().fit(F).transform(F)
        p_sk = sk.predict_proba(z)
        self.assertLess(np.abs(model.predict_proba(F) - p_sk).max(), 1e-3)


class EnergyAdaptationTest(unittest.TestCase):
    def test_zero_gain_preserves_m2_and_temporal_gains_reduce_training_loss(self):
        import torch
        from torch.nn import functional as F

        from wyu_bci_bench.energy_adaptation import fit_energy_gains, gain_logits, spatial_energy_terms
        from wyu_bci_bench.nets import PowerFirstMathPriorNet

        torch.manual_seed(11)
        model = PowerFirstMathPriorNet(np.zeros(5), np.ones(5), np.zeros((3, 5)), np.zeros(3),
                                       bins=4, n_channels=4).eval().requires_grad_(False)
        eeg = torch.randn(9, 4, 120) * 15
        features = torch.zeros(9, 5)
        energy = model.multiscale.channel_log_energy(eeg)
        mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 0]])
        whole, regional = spatial_energy_terms(energy, model.multiscale.spatial_weight, mask)
        args = (model.temporal_head, model.temporal_gate, model.math_prior(features), whole, regional)
        initial = gain_logits(*args, torch.zeros(3, 2, 4))
        torch.testing.assert_close(initial, model(eeg, features))
        labels = torch.arange(9) % 3
        gains = fit_energy_gains(*args, labels, by_half=True)
        self.assertLess(float(F.cross_entropy(gain_logits(*args, gains), labels)),
                        float(F.cross_entropy(initial, labels)))
        torch.testing.assert_close(gains[..., 0], gains[..., 1])
        torch.testing.assert_close(gains[..., 2], gains[..., 3])
        self.assertLessEqual(float(gains.abs().max()), .5)


class ShallowConvNetTest(unittest.TestCase):
    def test_forward_shape(self):
        try:
            import torch
        except Exception:  # pragma: no cover
            self.skipTest("torch missing")
        from wyu_bci_bench.deep import make_eegnet, make_shallow_convnet

        model = make_shallow_convnet(41, 120.0, 3, 480)
        self.assertEqual(tuple(model(torch.zeros(5, 41, 480)).shape), (5, 3))
        net = make_eegnet(41, 120.0, 3)
        self.assertEqual(tuple(net(torch.zeros(2, 1, 41, 480)).shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
