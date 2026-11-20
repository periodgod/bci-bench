import unittest

import numpy as np
import torch
from torch import nn

from wyu_bci_bench.prior_residual import PriorResidualEEG
from wyu_bci_bench.supervised import train_classifier


class PriorResidualTests(unittest.TestCase):
    def test_initial_prior_and_bounded_independent_corrections(self):
        torch.set_num_threads(2)
        for target in ("mi", "sub", "word"):
            model = PriorResidualEEG(target, 3, 4).eval()
            local = torch.randn(3, 3, 120)
            full = torch.randn(3, 4, 120)
            prior = torch.tensor([-2.0, 0.0, 1.0])
            torch.testing.assert_close(model(local, full, prior)[:, 1], prior, atol=0, rtol=0)
            with torch.no_grad():
                model.heads[0][-1].bias.fill_(100)
                model.heads[1][-1].bias.fill_(-100)
            local_delta, full_delta = model.corrections(local, full)
            torch.testing.assert_close(local_delta, torch.full_like(prior, 0.75))
            torch.testing.assert_close(full_delta, torch.full_like(prior, -0.75))
            torch.testing.assert_close(model(local, full, prior)[:, 1], prior)

    def test_convergence_at_minimum_learning_rate(self):
        x = np.ones((4, 1), dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        model = nn.Linear(1, 2, bias=False)
        nn.init.zeros_(model.weight)
        _, report = train_classifier(model, (x,), y, epochs=None, seed=1, device="cpu", lr=0.001,
                                     plateau_patience=2, plateau_delta=0.001, min_lr=0.0001)
        self.assertEqual(report["stop_reason"], "training_loss_plateau_at_min_lr")
        self.assertLessEqual(report["learning_rates"][-1], 0.0001)
        self.assertAlmostEqual(report["initial_train_loss"], report["best_train_loss"], places=6)

