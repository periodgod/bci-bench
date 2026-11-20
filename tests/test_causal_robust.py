import unittest
import numpy as np
from wyu_bci_bench.causal_robust import robust_causal_zscore, RobustZScoreState
from wyu_bci_bench.session_causal import causal_fixed_point_prior_match


class CausalRobustTests(unittest.TestCase):
    def test_unclipped_matches_original_and_first_observation_updates(self):
        from wyu_bci_bench.session_transductive import causal_zscore
        x = np.random.default_rng(3).normal(size=(50, 3)).astype(np.float32)
        keys = np.array(["a"] * len(x))
        np.testing.assert_allclose(robust_causal_zscore(x, keys, clip=np.inf), causal_zscore(x, keys), atol=1e-6)
        changed = x.copy(); changed[0] += 2
        self.assertFalse(np.allclose(robust_causal_zscore(x, keys)[1], robust_causal_zscore(changed, keys)[1]))

    def test_stream_step_and_batch_equivalence_and_reset(self):
        x = np.random.default_rng(5).normal(size=(80, 3))
        for mode in ("clip", "ewma"):
            state = RobustZScoreState(np.zeros(3), np.ones(3), mode=mode)
            a = np.stack([state.step(row) for row in x])
            b = robust_causal_zscore(x, np.zeros(len(x)), mode=mode)
            np.testing.assert_array_equal(a, b)
            state.reset()
            np.testing.assert_array_equal(a, np.stack([state.step(row) for row in x]))

    def test_future_perturbation_and_reset_invariance(self):
        x = np.arange(24, dtype=float).reshape(12, 2)
        keys = np.array(["a"]*6+["b"]*6)
        for mode in ("clip", "ewma"):
            p = robust_causal_zscore(x, keys, mode=mode)
            changed = x.copy(); changed[4:] += 500
            q = robust_causal_zscore(changed, keys, mode=mode)
            np.testing.assert_array_equal(p[:4], q[:4])
            np.testing.assert_array_equal(p[:6], robust_causal_zscore(x[:6], keys[:6], mode=mode))
            np.testing.assert_array_equal(p[6:], robust_causal_zscore(x[6:], keys[6:], mode=mode))

    def test_fixed_point_history_window_reset_and_default_equivalence(self):
        from wyu_bci_bench.session_causal import FixedPointPriorMatchState
        rng = np.random.default_rng(8)
        p = rng.dirichlet([1, 2, 3], size=20)
        keys = np.array(["a"]*10+["b"]*10)
        np.testing.assert_allclose(causal_fixed_point_prior_match(p, keys),
                                   causal_fixed_point_prior_match(p, keys, history_window=None))
        got = causal_fixed_point_prior_match(p, keys, history_window=4)
        np.testing.assert_allclose(got[10:], causal_fixed_point_prior_match(p[10:], keys[10:], history_window=4))
        self.assertFalse(np.allclose(got[8:10], causal_fixed_point_prior_match(p, keys)[8:10]))
        state = FixedPointPriorMatchState(history_window=4)
        streamed = np.stack([state.update(row) for row in p[:10]])
        np.testing.assert_allclose(streamed, got[:10])
        state.reset()
        np.testing.assert_allclose(np.stack([state.update(row) for row in p[10:]]), got[10:])
        changed = p.copy(); changed[9:] = 1/3
        np.testing.assert_allclose(got[:9], causal_fixed_point_prior_match(changed, keys, history_window=4)[:9])


if __name__ == "__main__":
    unittest.main()
