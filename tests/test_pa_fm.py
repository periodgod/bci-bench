import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required")
class PAFMTests(unittest.TestCase):
    def test_gate_zero_matches_m2(self):
        from wyu_bci_bench.nets import PowerFirstMathPriorNet
        from wyu_bci_bench.foundation.pa_fm import PAFMSCC

        rng = np.random.default_rng(2)
        m2 = PowerFirstMathPriorNet(
            np.zeros(6, np.float32), np.ones(6, np.float32),
            rng.normal(size=(3, 6)).astype(np.float32), np.zeros(3, np.float32),
            bins=2, n_channels=4,
        )
        model = PAFMSCC(m2, foundation_dim=5)
        model.foundation_gate.data.fill_(-100.0)
        model.eval()
        raw = torch.randn(3, 4, 480)
        feat = torch.randn(3, 6)
        emb = torch.randn(3, 5)
        with torch.no_grad():
            torch.testing.assert_close(model(raw, feat, foundation_embedding=emb), m2(raw, feat),
                                       atol=1e-5, rtol=1e-5)

    def test_token_pooling_and_trainable_count(self):
        from wyu_bci_bench.foundation.pa_fm import PAFMSCC
        from wyu_bci_bench.nets import PowerFirstMathPriorNet

        m2 = PowerFirstMathPriorNet(
            np.zeros(6, np.float32), np.ones(6, np.float32),
            np.zeros((3, 6), np.float32), np.zeros(3, np.float32),
            bins=2, n_channels=4,
        )
        model = PAFMSCC(m2, foundation_dim=5)
        self.assertEqual(tuple(model.pool_embedding(torch.ones(2, 4, 3, 5)).shape), (2, 5))
        trainable, total = model.trainable_parameter_count()
        self.assertGreater(trainable, 0)
        self.assertGreaterEqual(total, trainable)

    def test_early_stopping_returns_selected_validation_epoch(self):
        from wyu_bci_bench.foundation.pa_fm import PAFMSCC, train_pafm_early_stopping

        class Prior(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.readout = torch.nn.Linear(8, 3)

            def forward(self, raw, features):
                return self.readout(torch.cat((raw.flatten(1), features), dim=1))

        rng = np.random.default_rng(4)

        def data(prefix):
            n = 9
            return {
                "raw": rng.normal(size=(n, 2, 2)).astype(np.float32),
                "math_features": rng.normal(size=(n, 4)).astype(np.float32),
                "foundation": rng.normal(size=(n, 5)).astype(np.float32),
                "y": np.tile(np.arange(3), 3).astype(np.int64),
                "subjects": np.asarray([f"{prefix}{i // 3}" for i in range(n)]),
                "sessions": np.asarray(["ses-1"] * n),
                "interfaces": np.asarray(["left"] * n),
            }

        model = PAFMSCC(Prior(), foundation_dim=5)
        model, report = train_pafm_early_stopping(
            model, data("tr"), data("va"), seed=17, max_epochs=3, patience=2,
            device="cpu", batch_size=3,
        )
        self.assertIn(report["best_epoch"], {row["epoch"] for row in report["history"]})
        self.assertLessEqual(report["stopped_epoch"], 3)
        self.assertEqual(model.training, False)


if __name__ == "__main__":
    unittest.main()
