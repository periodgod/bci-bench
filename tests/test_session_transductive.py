import unittest

import numpy as np

from wyu_bci_bench.connectivity import imaginary_coherency, logeuclid_filterbank
from wyu_bci_bench.session_transductive import (
    causal_adaptive_lda,
    causal_zscore,
    knn_vote_proba,
    neighbor_smooth,
)


def _rand_probs(rng, n):
    p = rng.random((n, 3)) + 0.1
    return p / p.sum(1, keepdims=True)


class CausalityTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.keys = np.array(["a"] * 12 + ["b"] * 8)
        self.Z = self.rng.normal(size=(20, 5))
        self.P = _rand_probs(self.rng, 20)

    def _future_change(self, fn):
        out1 = fn(self.P, self.Z)
        Z2, P2 = self.Z.copy(), self.P.copy()
        Z2[7:12] += 50.0
        P2[7:12] = P2[7:12][:, ::-1]
        out2 = fn(P2, Z2)
        self.assertTrue(np.allclose(out1[:7], out2[:7]))
        self.assertTrue(np.allclose(out1[12:], out2[12:]))

    def test_zscore(self):
        self._future_change(lambda P, Z: causal_zscore(Z, self.keys, init_mean=np.zeros(5), init_var=np.ones(5)))
        out = causal_zscore(self.Z, self.keys, init_mean=np.ones(5), init_var=4 * np.ones(5))
        self.assertTrue(np.allclose(out[0], (self.Z[0] - 1) / 2, atol=1e-6))
        self.assertTrue(np.allclose(out[12], (self.Z[12] - 1) / 2, atol=1e-6))

    def test_neighbor_smooth(self):
        self._future_change(lambda P, Z: neighbor_smooth(P, Z, self.keys, k=3))
        out = neighbor_smooth(self.P, self.Z, self.keys, k=3)
        self.assertTrue(np.allclose(out[0], self.P[0]))
        self.assertTrue(np.allclose(out.sum(1), 1.0))

    def test_adaptive_lda(self):
        mu = self.rng.normal(size=(3, 5))
        fn = lambda P, Z: causal_adaptive_lda(P, Z, self.keys, mu, np.eye(5), burn=2)
        self._future_change(fn)
        out = fn(self.P, self.Z)
        self.assertTrue(np.allclose(out[:2], self.P[:2]))
        self.assertTrue(np.allclose(out.sum(1), 1.0))


class KnnVoteTest(unittest.TestCase):
    def test_votes(self):
        train = np.array([[1, 0], [1, 0.1], [0, 1], [0.1, 1]], dtype=float)
        y = np.array([0, 0, 1, 1])
        p = knn_vote_proba(train, y, np.array([[1, 0.05]]), k=2, n_classes=3)
        self.assertTrue(np.allclose(p, [[3 / 5, 1 / 5, 1 / 5]]))


class ConnectivityTest(unittest.TestCase):
    def test_imaginary_coherency(self):
        rng = np.random.default_rng(1)
        fs, t = 120.0, 240
        tt = np.arange(t) / fs
        base = np.sin(2 * np.pi * 10 * tt)
        lagged = np.sin(2 * np.pi * 10 * tt - np.pi / 2)
        x = np.stack([base, base, lagged]) + 0.01 * rng.normal(size=(3, t))
        ic = imaginary_coherency(x[None], fs, (8, 13))
        self.assertEqual(ic.shape, (1, 3))
        self.assertLess(abs(ic[0, 0]), 0.05)
        self.assertGreater(abs(ic[0, 1]), 0.5)

    def test_logeuclid_shape(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(4, 5, 120))
        v = logeuclid_filterbank(x, 120.0, [(4, 8), (8, 13)], [slice(0, 60), slice(60, 120)])
        self.assertEqual(v.shape, (4, 2 * 2 * 15))
        self.assertTrue(np.isfinite(v).all())


class SlowFeatureTest(unittest.TestCase):
    def test_means_and_slope(self):
        from wyu_bci_bench.slow_features import slow_dc_features

        t = np.arange(16, dtype=np.float64)
        x = np.stack([3.0 + 0 * t, 2.0 * t])[None]
        f = slow_dc_features(x, n_segments=4)
        self.assertEqual(f.shape, (1, 2 * 4 + 2))
        self.assertTrue(np.allclose(f[0, :4], 3.0))
        self.assertTrue(np.allclose(f[0, 4:8], 2.0 * np.array([1.5, 5.5, 9.5, 13.5])))
        self.assertTrue(np.allclose(f[0, 8:], [0.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
