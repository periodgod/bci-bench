import unittest

import numpy as np

from wyu_bci_bench.session_causal import (
    FixedPointPriorMatchState,
    SessionCausalState,
    causal_center,
    causal_class_balance,
    causal_fixed_point_prior_match,
    stream_keys,
)


class SessionCausalTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.keys = stream_keys(["a"] * 7 + ["b"] * 5, ["s1"] * 12)
        self.Z = rng.normal(size=(12, 4)) + 3.0
        logits = rng.normal(size=(12, 3))
        self.P = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)

    def test_first_window_of_each_stream_is_unchanged(self):
        C = causal_center(self.Z, self.keys, alpha=4.0)
        B = causal_class_balance(self.P, self.keys, alpha=4.0)
        for first in (0, 7):
            np.testing.assert_allclose(C[first], self.Z[first])
            np.testing.assert_allclose(B[first], self.P[first])

    def test_output_does_not_depend_on_future_or_other_streams(self):
        C = causal_center(self.Z, self.keys, alpha=4.0)
        B = causal_class_balance(self.P, self.keys, alpha=4.0)
        Z2, P2 = self.Z.copy(), self.P.copy()
        Z2[4:7] += 100.0
        Z2[7:] -= 50.0
        P2[4:7] = P2[4:7][:, ::-1]
        np.testing.assert_allclose(causal_center(Z2, self.keys, 4.0)[:4], C[:4])
        np.testing.assert_allclose(causal_class_balance(P2, self.keys, 4.0)[:4], B[:4])
        self.assertFalse(np.allclose(causal_center(Z2, self.keys, 4.0)[5], C[5]))
        np.testing.assert_allclose(causal_class_balance(P2, self.keys, 4.0)[7:], B[7:])

    def test_streaming_state_matches_batch_reference(self):
        C = causal_center(self.Z, self.keys, alpha=4.0)
        B = causal_class_balance(self.P, self.keys, alpha=4.0)
        state = SessionCausalState(4.0, 4.0)
        got_c, got_b = [], []
        for i in range(12):
            if i in (0, 7):
                state.reset()
            got_c.append(state.center(self.Z[i])[0])
            got_b.append(state.balance(self.P[i])[0])
        np.testing.assert_allclose(np.array(got_c), C, atol=1e-12)
        np.testing.assert_allclose(np.array(got_b), B, atol=1e-12)

    def test_balance_outputs_probabilities_and_removes_constant_bias(self):
        P = np.tile([[0.7, 0.2, 0.1]], (200, 1))
        B = causal_class_balance(P, np.zeros(200, int), alpha=1.0)
        np.testing.assert_allclose(B.sum(1), 1.0)
        self.assertLess(abs(B[-1, 0] - 1 / 3), 0.01)

    def test_fixed_point_match_is_causal_and_has_burn_in(self):
        keys = stream_keys(["a"] * 12 + ["b"] * 4)
        P = np.tile([[0.75, 0.20, 0.05]], (16, 1))
        got = causal_fixed_point_prior_match(P, keys, alpha=4.0, burn_in=3)
        np.testing.assert_allclose(got[[0, 1, 2, 12, 13, 14]], P[[0, 1, 2, 12, 13, 14]])
        self.assertFalse(np.allclose(got[3], P[3]))
        changed = P.copy()
        changed[8:12] = changed[8:12][:, ::-1]
        got_changed = causal_fixed_point_prior_match(changed, keys, alpha=4.0, burn_in=3)
        np.testing.assert_allclose(got_changed[:8], got[:8])
        np.testing.assert_allclose(got_changed[12:], got[12:])
        np.testing.assert_allclose(got.sum(1), 1.0)

    def test_fixed_point_stream_matches_batch_and_reset_replays(self):
        keys = stream_keys(["a"] * 7 + ["b"] * 5)
        batch = causal_fixed_point_prior_match(self.P, keys, alpha=4.0, burn_in=2)
        state = FixedPointPriorMatchState(3, alpha=4.0, burn_in=2)
        streamed = []
        for i, row in enumerate(self.P):
            if i in (0, 7):
                state.reset()
            streamed.append(state.update(row))
        np.testing.assert_allclose(streamed, batch, atol=1e-10)
        state.reset()
        replay = np.array([state.update(row) for row in self.P[:7]])
        np.testing.assert_allclose(replay, batch[:7], atol=1e-10)

    def test_fixed_point_match_rejects_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            causal_fixed_point_prior_match([[0.5, np.nan, 0.5]], ["a"])
        state = FixedPointPriorMatchState(3)
        with self.assertRaises(ValueError):
            state.update([0.2, -0.1, 0.9])


if __name__ == "__main__":
    unittest.main()
