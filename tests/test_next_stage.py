import unittest
import numpy as np
import torch
from torch import nn
from wyu_bci_bench.phase_slow import PhaseSlowNet
from wyu_bci_bench.supervised import balanced_group_class_weights, train_classifier, predict_classifier
from wyu_bci_bench.ensemble_diagnostics import stream_positions


class NextStageTests(unittest.TestCase):
    def test_forward_blocks_are_temporal_and_outer_isolated(self):
        from wyu_bci_bench.splits import forward_blocks
        keys = np.array(["a"] * 10 + ["b"] * 10)
        order = np.tile(np.arange(10)[::-1], 2)
        eligible = order < 8
        for split in forward_blocks(keys, eligible, order):
            self.assertFalse(np.any((split.train | split.test) & ~eligible))
            for key in ("a", "b"):
                self.assertLess(order[split.train & (keys == key)].max(), order[split.test & (keys == key)].min())

    def test_stream_order_and_reset(self):
        np.testing.assert_array_equal(stream_positions(["a", "b", "a", "b", "c"]), [0, 0, 1, 1, 0])

    def test_group_class_mass(self):
        y = np.array([0, 0, 1, 1, 1, 0])
        g = np.array(["a", "a", "a", "b", "b", "b"])
        w = balanced_group_class_weights(y, g)
        self.assertAlmostEqual(float(w.mean()), 1.)
        mass = [w[(g == k) & (y == c)].sum() for k, c in (("a", 0), ("a", 1), ("b", 0), ("b", 1))]
        np.testing.assert_allclose(mass, mass[0])

    def test_phase_readout_retains_half_identity(self):
        # Replace encoder with a known temporal feature; equal global averages
        # but swapped halves must produce distinct phase representations.
        model = PhaseSlowNet(channels=1, width=1, classes=2, phase=True)
        model.features = nn.Identity()
        model.classifier = nn.Identity()
        x = torch.cat((torch.zeros(1, 1, 8), torch.ones(1, 1, 8)), -1)
        torch.testing.assert_close(model(x), torch.tensor([[0., 1.]]))
        torch.testing.assert_close(model(x.flip(-1)), torch.tensor([[1., 0.]]))

    def test_slow_branch_receives_gradient(self):
        model = PhaseSlowNet(channels=3, width=4, phase=True, slow_features=6)
        model(torch.randn(4, 3, 64), torch.randn(4, 6)).square().mean().backward()
        self.assertGreater(model.slow[0].weight.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            model(torch.randn(4, 3, 64))

    def test_trainer_reduces_loss_and_predicts_probabilities(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(60, 2)).astype(np.float32)
        y = (x[:, 0] > 0).astype(np.int64)
        torch.manual_seed(4)
        model = nn.Linear(2, 2)
        model, info = train_classifier(model, (x,), y, epochs=25, seed=4, lr=.1, batch=30, device="cpu")
        p = predict_classifier(model, (x,), device="cpu")
        self.assertGreater((p.argmax(1) == y).mean(), .95)
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
        self.assertEqual(info["epochs"], 25)

    def test_csp_native_probability_order(self):
        from wyu_bci_bench.csp import FilterBankCSP
        rng = np.random.default_rng(3)
        x = rng.normal(size=(30, 4, 128))
        y = np.tile([2, 5, 9], 10)
        for i, c in enumerate([2, 5, 9]):
            x[y == c, i] *= 4
        model = FilterBankCSP(bands=((8, 24),), n_components=3, classifier="lda").fit(x, y, 128)
        p = model.predict_proba(x)
        np.testing.assert_array_equal(model.classes_, [2, 5, 9])
        np.testing.assert_array_equal(model.classes_[p.argmax(1)], model.predict(x))
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
        with self.assertRaises(ValueError):
            model.predict_proba(x[:, :3])
        svm = FilterBankCSP(bands=((8, 24),), n_components=3).fit(x, y, 128)
        with self.assertRaises(ValueError):
            svm.predict_proba(x)


if __name__ == "__main__":
    unittest.main()
