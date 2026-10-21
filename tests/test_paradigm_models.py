import unittest

import numpy as np


class SSVEPTests(unittest.TestCase):
    def setUp(self):
        self.fs = 250
        self.frequencies = (10.0, 12.0, 15.0)
        time = np.arange(500) / self.fs
        rng = np.random.default_rng(31)
        epochs, labels = [], []
        for label, frequency in enumerate(self.frequencies):
            for _ in range(8):
                phase = rng.uniform(0, 2 * np.pi)
                base = np.sin(2 * np.pi * frequency * time + phase)
                epochs.append(np.stack([base, 0.7 * base + 0.1 * rng.normal(size=len(time))]))
                labels.append(label)
        self.X = np.asarray(epochs, dtype=np.float32)
        self.y = np.asarray(labels)

    def test_reference_cca_decodes_synthetic_frequency_trials(self):
        from wyu_bci_bench.ssvep import SSVEPCCA

        model = SSVEPCCA(self.frequencies, self.fs, harmonics=2)
        np.testing.assert_array_equal(model.predict(self.X), self.y)

    def test_filterbank_cca_decodes_synthetic_frequency_trials(self):
        from wyu_bci_bench.ssvep import FilterBankCCA

        model = FilterBankCCA(self.frequencies, self.fs, harmonics=2,
                              bands=((6, 90), (14, 90), (22, 90)))
        np.testing.assert_array_equal(model.predict(self.X), self.y)

    def test_filterbank_defaults_respect_low_sampling_rate_nyquist(self):
        from wyu_bci_bench.ssvep import FilterBankCCA

        time = np.arange(600) / 120
        X = np.asarray([[np.sin(2 * np.pi * f * time)] * 2 for f in (10, 12)])
        model = FilterBankCCA((10, 12), 120, harmonics=2)
        np.testing.assert_array_equal(model.predict(X), np.array([0, 1]))

    def test_trca_learns_repeated_class_templates(self):
        from wyu_bci_bench.ssvep import TRCA

        model = TRCA().fit(self.X, self.y)
        self.assertGreater(np.mean(model.predict(self.X) == self.y), 0.8)

    def test_ensemble_trca_uses_stimulus_frequency_mapping(self):
        from wyu_bci_bench.ssvep import EnsembleTRCA

        labels = np.where(self.y == 0, 10, np.where(self.y == 1, 20, 30))
        model = EnsembleTRCA(self.frequencies, self.fs, n_fbands=3).fit(self.X, labels)
        prediction = model.predict(self.X)
        self.assertEqual(prediction.shape, labels.shape)
        self.assertTrue(np.isin(prediction, labels).all())
        np.testing.assert_allclose(model.estimator_.peaks_, self.frequencies)

    def test_tdca_learns_temporal_discriminants_with_non_frequency_labels(self):
        from wyu_bci_bench.ssvep import TDCA

        labels = np.where(self.y == 0, "alpha", np.where(self.y == 1, "beta", "gamma"))
        model = TDCA(self.frequencies, self.fs, n_fbands=3, n_delay=3).fit(self.X, labels)
        self.assertEqual(model.predict(self.X).shape, labels.shape)
        self.assertGreater(np.mean(model.predict(self.X) == labels), 0.7)

    def test_reference_rejects_harmonics_at_or_above_nyquist(self):
        from wyu_bci_bench.ssvep import reference_signals

        with self.assertRaisesRegex(ValueError, "Nyquist"):
            reference_signals(100, 100, (20,), harmonics=3)


class ERPTests(unittest.TestCase):
    def test_xdawn_lda_separates_evoked_target_epochs(self):
        from wyu_bci_bench.erp import XDawnLDA

        rng = np.random.default_rng(22)
        X = rng.normal(scale=0.35, size=(48, 6, 80))
        y = np.tile([0, 1], 24)
        response = np.exp(-0.5 * ((np.arange(80) - 35) / 5) ** 2)
        X[y == 1, 0, :] += response
        X[y == 1, 1, :] += 0.5 * response
        model = XDawnLDA(n_components=3).fit(X[:40], y[:40])
        predicted = model.predict(X[40:])
        self.assertEqual(predicted.shape, (8,))
        self.assertGreater(np.mean(predicted == y[40:]), 0.75)
        self.assertEqual(model.predict_proba(X[40:]).shape, (8, 2))

    def test_xdawn_rejects_invalid_component_count(self):
        from wyu_bci_bench.erp import XDawnLDA

        with self.assertRaisesRegex(ValueError, "n_components"):
            XDawnLDA(n_components=5).fit(np.ones((4, 2, 20)), np.array([0, 1, 0, 1]))

    def test_xdawn_rejects_single_class_training_data(self):
        from wyu_bci_bench.erp import XDawnLDA

        with self.assertRaisesRegex(ValueError, "two target classes"):
            XDawnLDA().fit(np.ones((4, 2, 20)), np.ones(4, dtype=int))


class DomainAdaptationTests(unittest.TestCase):
    def test_euclidean_alignment_normalizes_each_unlabeled_domain(self):
        from wyu_bci_bench.adaptation import EuclideanAlignment

        rng = np.random.default_rng(12)
        source = rng.normal(size=(20, 3, 40))
        target = np.einsum("cd,ndt->nct", np.diag([3.0, 1.5, 0.5]),
                           rng.normal(size=(20, 3, 40)))
        X = np.concatenate([source, target])
        domains = np.array(["source"] * 20 + ["target"] * 20)
        aligned = EuclideanAlignment().fit_transform(X, domains)
        for key in ("source", "target"):
            domain_epochs = aligned[domains == key]
            centered = domain_epochs - domain_epochs.mean(axis=-1, keepdims=True)
            cov = np.mean(centered @ np.swapaxes(centered, 1, 2) / centered.shape[-1], axis=0)
            np.testing.assert_allclose(cov, np.eye(3), atol=2e-5)

    def test_coral_maps_source_mean_and_covariance_to_target(self):
        from wyu_bci_bench.adaptation import coral_align

        rng = np.random.default_rng(8)
        source = rng.normal(size=(500, 3)) @ np.diag([0.5, 1.0, 2.0])
        target = rng.normal(loc=[2, -1, 0.5], size=(600, 3)) @ np.diag([2.0, 0.7, 1.2])
        aligned = coral_align(source, target)
        np.testing.assert_allclose(aligned.mean(axis=0), target.mean(axis=0), atol=1e-10)
        np.testing.assert_allclose(np.cov(aligned, rowvar=False), np.cov(target, rowvar=False), atol=1e-9)

    def test_euclidean_alignment_rejects_unfitted_domains(self):
        from wyu_bci_bench.adaptation import EuclideanAlignment

        model = EuclideanAlignment().fit(np.ones((3, 2, 10)), np.array([0, 0, 1]))
        with self.assertRaisesRegex(ValueError, "unknown domain"):
            model.transform(np.ones((1, 2, 10)), np.array([2]))

    def test_riemannian_alignment_whitens_each_domain(self):
        from wyu_bci_bench.adaptation import RiemannianAlignment

        rng = np.random.default_rng(19)
        source = rng.normal(size=(16, 3, 48))
        target = np.einsum("cd,ndt->nct", np.diag([3.0, 1.5, 0.6]), rng.normal(size=(16, 3, 48)))
        X = np.concatenate((source, target))
        domains = np.array(["s"] * len(source) + ["t"] * len(target))
        aligned = RiemannianAlignment().fit_transform(X, domains)
        for key in ("s", "t"):
            segment = aligned[domains == key]
            centered = segment - segment.mean(axis=-1, keepdims=True)
            covariance = np.mean(centered @ np.swapaxes(centered, 1, 2) / centered.shape[-1], axis=0)
            np.testing.assert_allclose(covariance, np.eye(3), atol=0.08)

    def test_mmd_detects_shift_and_propagates_gradients(self):
        import torch
        from wyu_bci_bench.adaptation import maximum_mean_discrepancy

        source = torch.tensor([[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], requires_grad=True)
        same = source.detach().clone()
        shifted = same + torch.tensor([2.0, -1.0])
        self.assertAlmostEqual(float(maximum_mean_discrepancy(source, same).detach()), 0.0, places=6)
        discrepancy = maximum_mean_discrepancy(source, shifted)
        self.assertGreater(float(discrepancy.detach()), 0.05)
        discrepancy.backward()
        self.assertTrue(torch.isfinite(source.grad).all())

    def test_dann_head_reverses_encoder_gradient(self):
        import torch
        from wyu_bci_bench.adaptation import DomainAdversarialHead

        head = DomainAdversarialHead(3, 2, hidden=4, strength=1.0)
        values = torch.randn(5, 3, requires_grad=True)
        head(values).sum().backward()
        reversed_gradient = values.grad.detach().clone()
        values.grad.zero_()
        head.classifier(values).sum().backward()
        ordinary_gradient = values.grad.detach().clone()
        torch.testing.assert_close(reversed_gradient, -ordinary_gradient)

    def test_adabn_updates_only_running_statistics_from_unlabeled_batches(self):
        import torch
        from torch import nn
        from wyu_bci_bench.adaptation import adapt_batch_norm

        model = nn.Sequential(nn.BatchNorm1d(2), nn.Dropout(p=0.9), nn.Linear(2, 2))
        model.eval()
        initial_mean = model[0].running_mean.clone()
        seen = adapt_batch_norm(model, [torch.full((8, 2), 3.0), torch.full((8, 2), 4.0)])
        self.assertEqual(seen, 2)
        self.assertTrue(torch.all(model[0].running_mean > initial_mean))
        self.assertFalse(model.training)
        self.assertFalse(model[1].training)


class CSPTests(unittest.TestCase):
    def test_filterbank_csp_fits_csp_per_band_and_predicts_trials(self):
        from wyu_bci_bench.csp import FilterBankCSP

        rng = np.random.default_rng(41)
        t = np.arange(320) / 128
        labels = np.tile([0, 1], 20)
        X = rng.normal(scale=0.12, size=(len(labels), 4, len(t)))
        X[labels == 0, 0] += np.sin(2 * np.pi * 10 * t) * 1.4
        X[labels == 1, 1] += np.sin(2 * np.pi * 10 * t) * 1.4
        model = FilterBankCSP(bands=((8, 14), (14, 24)), n_components=2).fit(X[:32], labels[:32], 128)
        predictions = model.predict(X[32:])
        self.assertEqual(predictions.shape, (8,))
        self.assertGreater(np.mean(predictions == labels[32:]), 0.75)
        self.assertEqual(len(model.csp_filters_), 2)

    def test_filterbank_csp_supports_shrinkage_lda_readout(self):
        from wyu_bci_bench.csp import FilterBankCSP

        rng = np.random.default_rng(42)
        t = np.arange(256) / 128
        y = np.tile([0, 1], 12)
        X = rng.normal(scale=0.15, size=(len(y), 4, len(t)))
        X[y == 0, 0] += np.sin(2 * np.pi * 10 * t)
        X[y == 1, 1] += np.sin(2 * np.pi * 10 * t)
        model = FilterBankCSP(bands=((8, 14),), n_components=2, classifier="lda").fit(X[:16], y[:16], 128)
        self.assertEqual(model.predict(X[16:]).shape, (8,))

    def test_csp_rejects_test_channel_mismatch(self):
        from wyu_bci_bench.csp import FilterBankCSP

        rng = np.random.default_rng(9)
        X = rng.normal(size=(12, 3, 256))
        y = np.tile([0, 1], 6)
        model = FilterBankCSP(bands=((8, 20),), n_components=2).fit(X, y, 128)
        with self.assertRaisesRegex(ValueError, "channel count"):
            model.predict(X[:, :2])


class MaskedReconstructionTests(unittest.TestCase):
    def test_loss_only_uses_masked_signal_values(self):
        import torch
        from wyu_bci_bench.ssl import masked_reconstruction_loss

        target = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
        prediction = torch.tensor([[[100.0, 2.0, -100.0, 5.0]]])
        mask = torch.tensor([[[False, True, False, True]]])
        loss = masked_reconstruction_loss(prediction, target, mask)
        self.assertAlmostEqual(float(loss), 2.5, places=6)

    def test_loss_rejects_empty_mask_and_shape_mismatch(self):
        import torch
        from wyu_bci_bench.ssl import masked_reconstruction_loss

        values = torch.ones(2, 3, 4)
        with self.assertRaisesRegex(ValueError, "at least one"):
            masked_reconstruction_loss(values, values, torch.zeros(2, 3, 4, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            masked_reconstruction_loss(values, torch.ones(2, 3, 5), torch.ones(2, 3, 4, dtype=torch.bool))

    def test_predictive_objective_respects_horizon_mask(self):
        import torch
        from wyu_bci_bench.ssl import predictive_forecast_loss

        target = torch.tensor([[1.0, 100.0, 3.0]])
        prediction = torch.tensor([[2.0, -100.0, 3.0]], requires_grad=True)
        mask = torch.tensor([[True, False, True]])
        loss = predictive_forecast_loss(prediction, target, mask)
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()), 0.5)
        self.assertEqual(float(prediction.grad[0, 1]), 0.0)

    def test_cross_channel_contrastive_loss_is_differentiable(self):
        import torch
        from wyu_bci_bench.ssl import cross_channel_contrastive_loss

        embedding = torch.randn(4, 3, 8, requires_grad=True)
        loss = cross_channel_contrastive_loss(embedding)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(embedding.grad).all())

    def test_masked_autoencoder_runs_unsupervised_training_recipe(self):
        import torch
        from wyu_bci_bench.ssl import MaskedEEGAutoencoder, train_masked_autoencoder

        model = MaskedEEGAutoencoder(n_channels=3, patch_size=8, hidden=16,
                                     heads=4, depth=1, max_patches=12)
        batches = [torch.randn(4, 3, 64), torch.randn(4, 3, 64)]
        history = train_masked_autoencoder(model, batches, epochs=2, mask_ratio=0.5)
        self.assertEqual(len(history), 2)
        self.assertTrue(np.isfinite(history).all())
        detail = model(torch.randn(2, 3, 64), return_details=True)
        self.assertEqual(tuple(detail["prediction"].shape), (2, 3, 8, 8))
        self.assertGreater(float(detail["mask"].float().mean()), 0)

    def test_masked_autoencoder_rejects_label_pairs_and_invalid_lengths(self):
        import torch
        from wyu_bci_bench.ssl import MaskedEEGAutoencoder, train_masked_autoencoder

        model = MaskedEEGAutoencoder(n_channels=2, patch_size=8, hidden=16, heads=4, depth=1)
        with self.assertRaisesRegex(ValueError, "divisible"):
            model(torch.randn(2, 2, 63))
        with self.assertRaisesRegex(TypeError, "task labels"):
            train_masked_autoencoder(model, [(torch.randn(2, 2, 64), torch.tensor([0, 1]))])


if __name__ == "__main__":
    unittest.main()
