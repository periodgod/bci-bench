import unittest

import numpy as np

from wyu_bci_bench.innovation import (
    causal_prototype_prob,
    causal_gaussian_prototype_prob,
    causal_rank_normalize,
    causal_scale,
    class_scatter,
    fit_nuisance_subspace,
    fit_wccn,
    exponential_center,
    cell_class_balanced_cross_entropy,
    cell_class_sample_weights,
    GroupDRO,
    group_dro_cross_entropy,
    full_session_center,
    nuisance_covariance,
    remove_nuisance_subspace,
    sinkhorn_assignments,
    source_prototypes,
    periodic_spectral_features,
)


class InnovationTests(unittest.TestCase):
    def test_full_session_center_is_groupwise_and_zero_mean(self):
        x = np.array([[1, 3], [3, 5], [10, 20]], dtype=np.float32)
        got = full_session_center(x, ["a", "a", "b"])
        np.testing.assert_allclose(got, [[-1, -1], [1, 1], [0, 0]])

    def test_nuisance_covariance_uses_stream_ordered_blocks(self):
        z = np.arange(24, dtype=float).reshape(12, 2)
        cov, count = nuisance_covariance(z, ["a"] * 8 + ["b"] * 4, block_size=2)
        self.assertEqual(count, 6)
        self.assertEqual(cov.shape, (2, 2))
        self.assertTrue(np.all(np.linalg.eigvalsh(cov) >= -1e-10))

    def test_fit_and_remove_nuisance_subspace(self):
        rng = np.random.default_rng(19)
        y = np.tile(np.arange(3), 20)
        signal = np.eye(3)[y] @ rng.normal(size=(3, 5))
        nuisance = np.repeat(np.linspace(-2, 2, 10), 6)[:, None]
        z = signal + nuisance * np.array([[1., 0., 0., 0., 0.]])
        keys = np.repeat(np.arange(10), 6)
        values, vectors, blocks = fit_nuisance_subspace(z, y, keys, 2, block_size=3)
        projected = remove_nuisance_subspace(z, vectors)
        self.assertEqual(values.shape, (2,))
        self.assertEqual(blocks, 20)
        self.assertEqual(vectors.shape, (5, 2))
        self.assertEqual(projected.shape, z.shape)
        self.assertTrue(np.isfinite(class_scatter(z, y)).all())
        np.testing.assert_allclose(projected, z - (z @ vectors) @ vectors.T)

    def test_sinkhorn_has_balanced_marginals(self):
        scores = np.array([[3., 0., -1.], [1., 2., 0.], [0., 1., 3.], [1., 0., 2.]])
        q = sinkhorn_assignments(scores, epsilon=0.2, iterations=100)
        np.testing.assert_allclose(q.sum(1), 1 / 4, atol=1e-5)
        np.testing.assert_allclose(q.sum(0), 1 / 3, atol=1e-5)

    def test_causal_prototypes_do_not_see_current_or_future_window(self):
        emb = np.tile(np.eye(3, dtype=np.float32), (15, 1))
        proto = np.eye(3, dtype=np.float32)
        prob = np.full((len(emb), 3), 1 / 3, dtype=np.float32)
        kwargs = dict(memory=8, burn_in=4, shrink_kappa=4, temperature=0.2)
        base = causal_prototype_prob(emb, proto, prob, **kwargs)
        changed = emb.copy()
        changed[10:] = changed[10:, ::-1]
        later = causal_prototype_prob(changed, proto, prob, **kwargs)
        np.testing.assert_allclose(base[:10], later[:10])
        replay = causal_prototype_prob(emb, proto, prob, **kwargs)
        np.testing.assert_array_equal(base, replay)

    def test_source_prototypes_reject_missing_class_and_nonfinite_inputs(self):
        with self.assertRaises(ValueError):
            source_prototypes(np.ones((3, 2)), np.array([0, 0, 1]))
        with self.assertRaises(ValueError):
            full_session_center(np.array([[np.nan]]), ["a"])

    def test_causal_feature_transforms_reset_by_stream_and_use_history(self):
        x = np.array([[1., 4.], [3., 2.], [2., 5.], [10., 10.]])
        keys = ["a", "a", "a", "b"]
        scale = causal_scale(x, keys, np.array([2., 2.]), beta=2)
        rank = causal_rank_normalize(x, keys, np.tile(x[:3], (2, 1)), beta=2)
        exp = exponential_center(x, keys, half_life=2)
        np.testing.assert_array_equal(scale[-1], x[-1])
        np.testing.assert_array_equal(exp[-1], x[-1])
        changed = x.copy()
        changed[:3] += 100
        rank_changed = causal_rank_normalize(changed, keys, np.tile(x[:3], (2, 1)), beta=2)
        np.testing.assert_array_equal(rank[-1], rank_changed[-1])
        altered = x.copy()
        altered[0] += 10
        changed_scale = causal_scale(altered, keys, np.array([2., 2.]), beta=2)
        self.assertFalse(np.allclose(scale[2], changed_scale[2]))

    def test_wccn_and_periodic_spectral_features(self):
        rng = np.random.default_rng(5)
        z = rng.normal(size=(30, 4))
        y = np.tile(np.arange(3), 10)
        transform = fit_wccn(z, y, shrinkage=0.5)
        self.assertEqual(transform.shape, (4, 4))
        raw = rng.normal(size=(5, 3, 480)).astype(np.float32)
        periodic, aperiodic = periodic_spectral_features(raw)
        self.assertEqual(periodic.shape, (5, 2 * 3 * (7 + 3)))
        self.assertEqual(aperiodic.shape, (5, 2 * 3 * 2))
        self.assertTrue(np.isfinite(periodic).all() and np.isfinite(aperiodic).all())

    def test_gaussian_prototype_adaptation_is_causal_and_oracle_labels_are_optional(self):
        z = np.tile(np.array([[1., 0.], [0., 1.], [-1., 0.]]), (6, 1))
        means = np.array([[1., 0.], [0., 1.], [-1., 0.]])
        var = np.full((3, 2), 0.2)
        p = np.full((len(z), 3), 1 / 3)
        base = causal_gaussian_prototype_prob(z, means, var, p, labels=np.tile([0, 1, 2], 6))
        changed = z.copy()
        changed[12:] = -changed[12:]
        later = causal_gaussian_prototype_prob(changed, means, var, p,
                                               labels=np.tile([0, 1, 2], 6))
        np.testing.assert_allclose(base[:12], later[:12])
        em = causal_gaussian_prototype_prob(z, means, var, p, labels=None)
        self.assertTrue(np.isfinite(em).all())
        np.testing.assert_allclose(em.sum(1), 1.0)

    def test_cell_class_weights_equalize_groups_and_groupdro_tracks_hard_groups(self):
        import torch

        labels = np.array([0, 0, 0, 1, 1, 0, 1])
        cells = np.array(["a"] * 5 + ["b"] * 2)
        weights = cell_class_sample_weights(labels, cells)
        for cell, cls in (("a", 0), ("a", 1), ("b", 0), ("b", 1)):
            self.assertAlmostEqual(weights[(cells == cell) & (labels == cls)].sum(),
                                   weights[(cells == "a") & (labels == 0)].sum(), places=5)
        logits = torch.tensor([[2., 0.], [0., 2.], [1., 0.]], requires_grad=True)
        target = torch.tensor([0, 1, 1])
        loss = cell_class_balanced_cross_entropy(logits, target, np.ones(3))
        self.assertTrue(torch.isfinite(loss))
        state = GroupDRO(2, eta=0.1)
        before = state.weights.copy()
        robust = group_dro_cross_entropy(logits, target, torch.tensor([0, 1, 1]), state)
        self.assertTrue(torch.isfinite(robust))
        self.assertGreater(state.weights[1], before[1])


if __name__ == "__main__":
    unittest.main()
