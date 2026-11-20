import unittest

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from wyu_bci_bench.fewshot import (
    balanced_kshot_indices,
    balance_unlabeled_probabilities,
    estimate_decoder_prior,
    fit_map_logistic,
    isotropic_decoder_prior,
    predict_map_logistic,
    semi_supervised_em_logistic,
)


class FewShotTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.tile(np.repeat([0, 1], 20), 2)
        self.runs = np.repeat(["R1", "R2"], 40)

    def test_kshot_sampling_is_deterministic_balanced_and_run_stratified(self):
        first = balanced_kshot_indices(self.labels, self.runs, 16, repeats=20, seed=9)
        second = balanced_kshot_indices(self.labels, self.runs, 16, repeats=20, seed=9)
        np.testing.assert_array_equal(first, second)
        for ix in first:
            self.assertEqual(len(ix), 16)
            self.assertEqual(np.sum(self.labels[ix] == 0), 8)
            self.assertEqual(np.sum(self.labels[ix] == 1), 8)
            for run in ("R1", "R2"):
                for cls in (0, 1):
                    self.assertEqual(np.sum((self.runs[ix] == run) & (self.labels[ix] == cls)), 4)

    def test_two_shot_is_class_balanced_and_eighty_shot_has_one_unique_set(self):
        two = balanced_kshot_indices(self.labels, self.runs, 2, repeats=30, seed=3)
        self.assertTrue(all(np.unique(self.labels[ix], return_counts=True)[1].tolist() == [1, 1]
                            for ix in two))
        full = balanced_kshot_indices(self.labels, self.runs, 80, repeats=20, seed=3)
        self.assertTrue(all(set(ix.tolist()) == set(range(80)) for ix in full))
        self.assertTrue(all(set(row.tolist()) == set(full[0].tolist()) for row in full[1:]))

    def test_invalid_strata_and_k_are_rejected(self):
        with self.assertRaises(ValueError):
            balanced_kshot_indices([0, 0, 1, 1], ["R1"] * 4, 2)
        with self.assertRaises(ValueError):
            balanced_kshot_indices(self.labels, self.runs, 6)

    def test_prior_estimation_is_low_rank_and_zero_shot_is_exact_base(self):
        rng = np.random.default_rng(5)
        base = np.zeros(5)
        source = rng.normal(size=(8, 5))
        prior = estimate_decoder_prior(source, base, rank=3)
        self.assertEqual(prior.directions.shape, (5, 3))
        self.assertTrue(np.all(prior.variances >= 0))
        np.testing.assert_array_equal(fit_map_logistic(np.empty((0, 4)), np.empty(0), prior), base)
        full = estimate_decoder_prior(source, base, rank=None)
        self.assertEqual(full.directions.shape[1], 5)

    def test_map_logistic_is_finite_and_predicts_valid_probabilities(self):
        rng = np.random.default_rng(11)
        X = rng.normal(size=(40, 3))
        y = (X[:, 0] - 0.4 * X[:, 1] > 0).astype(float)
        source = np.vstack([np.zeros(4), np.ones(4) * 0.1, -np.ones(4) * 0.1])
        prior = isotropic_decoder_prior(source, np.zeros(4), variance_multiplier=2.0)
        theta = fit_map_logistic(X, y, prior)
        probabilities = predict_map_logistic(X, theta)
        self.assertTrue(np.isfinite(theta).all())
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_dual_map_matches_independent_primal_solution(self):
        rng = np.random.default_rng(23)
        X = rng.normal(size=(18, 3))
        y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(float)
        base = rng.normal(scale=0.05, size=4)
        source = base + rng.normal(scale=0.2, size=(8, 4))
        prior = estimate_decoder_prior(source, base, rank=2)
        theta = fit_map_logistic(X, y, prior)
        U = prior.directions
        covariance = U @ np.diag(prior.variances + prior.residual_variance) @ U.T
        covariance += prior.residual_variance * (np.eye(4) - U @ U.T)
        precision = np.linalg.inv(covariance)
        design = np.column_stack([X, np.ones(len(X))])

        def objective(value):
            delta = value - prior.mean
            logits = design @ value
            loss = np.logaddexp(0, logits) - y * logits
            return float(loss.sum() + 0.5 * delta @ precision @ delta)

        def gradient(value):
            delta = value - prior.mean
            logits = design @ value
            return design.T @ (expit(logits) - y) + precision @ delta

        reference = minimize(objective, prior.mean, jac=gradient, method="L-BFGS-B",
                             options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-10})
        self.assertTrue(reference.success)
        np.testing.assert_allclose(theta, reference.x, atol=2e-6, rtol=2e-6)

    def test_map_is_stable_for_ill_conditioned_low_rank_prior(self):
        rng = np.random.default_rng(31)
        scales = np.asarray([1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
        source = rng.normal(size=(20, len(scales))) * scales
        prior = estimate_decoder_prior(source, np.zeros(len(scales)), rank=3)
        X = rng.normal(size=(32, len(scales) - 1))
        y = (X[:, 0] - 0.2 * X[:, 1] > 0).astype(float)
        theta = fit_map_logistic(X, y, prior, max_iter=200)
        self.assertTrue(np.isfinite(theta).all())
        probabilities = predict_map_logistic(X, theta)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_uniform_balancing_and_em_keep_probabilities_valid(self):
        p = np.tile([[0.8, 0.2]], (25, 1))
        q = balance_unlabeled_probabilities(p)
        np.testing.assert_allclose(q.mean(axis=0), [0.5, 0.5], atol=1e-8)
        rng = np.random.default_rng(12)
        X = rng.normal(size=(30, 2))
        y = (X[:10, 0] > 0).astype(float)
        source = rng.normal(scale=0.2, size=(6, 3))
        prior = estimate_decoder_prior(source, np.zeros(3), rank=2)
        theta, pseudo = semi_supervised_em_logistic(X[:10], y, X[10:], prior, iterations=3)
        self.assertTrue(np.isfinite(theta).all())
        np.testing.assert_allclose(pseudo.sum(axis=1), 1.0)
        np.testing.assert_allclose(pseudo.mean(axis=0), [0.5, 0.5], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
