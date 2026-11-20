import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required")
class CoCoTEEGTests(unittest.TestCase):
    def small_model_kwargs(self):
        return {
            "model_dim": 32,
            "ffn_dim": 64,
            "layers": 1,
            "heads": 4,
            "patch_size": 10,
            "kernel_sizes": (3, 7),
            "head_dim": 32,
            "head_layers": 1,
        }

    def test_architecture_emits_one_logit_vector_per_epoch(self):
        from wyu_bci_bench.cocot import make_cocot_model

        model = make_cocot_model(4, 103, 3, **self.small_model_kwargs())
        scores = model(torch.randn(2, 4, 103))
        self.assertEqual(tuple(scores.shape), (2, 3))

    def test_encoder_checkpoint_load_is_strict(self):
        from wyu_bci_bench.cocot import make_cocot_model

        kwargs = self.small_model_kwargs()
        source = make_cocot_model(4, 100, 3, **kwargs)
        target = make_cocot_model(4, 100, 3, **kwargs)
        target.load_encoder_state_dict(source.encoder.state_dict())
        for left, right in zip(source.encoder.parameters(), target.encoder.parameters()):
            torch.testing.assert_close(left, right)
        with self.assertRaises(RuntimeError):
            target.load_encoder_state_dict({})

    def test_robust_normalization_uses_channel_median_and_mad(self):
        from wyu_bci_bench.cocot import robust_normalize

        x = np.asarray([[[0, 1, 2, 1000]]], dtype=np.float32)
        result = robust_normalize(x)
        self.assertAlmostEqual(float(np.median(result)), 0.0)
        self.assertAlmostEqual(float(np.median(np.abs(result))), 1.0, places=5)

    def test_fold_local_supervised_finetuning_returns_trial_predictions(self):
        from wyu_bci_bench.cocot import fit_predict_cocot

        rng = np.random.default_rng(42)
        y = np.tile(np.array([0, 1, 2], dtype=np.int64), 4)
        train = rng.normal(size=(len(y), 4, 100)).astype(np.float32)
        train[y == 1, 0] += 0.5
        train[y == 2, 1] += 0.5
        test = rng.normal(size=(3, 4, 100)).astype(np.float32)
        pred, metadata = fit_predict_cocot(
            train, y, test, 3, sampling_rate=100, epochs=1, patience=1,
            batch_size=4, model_kwargs=self.small_model_kwargs(),
        )
        self.assertEqual(pred.shape, (3,))
        self.assertTrue(np.isin(pred, [0, 1, 2]).all())
        self.assertEqual(metadata["model"], "cocot_eeg")
        self.assertGreater(metadata["parameter_count"], 0)

    def test_mixup_finetuning_path_returns_trial_predictions(self):
        from wyu_bci_bench.cocot import fit_predict_cocot

        rng = np.random.default_rng(7)
        y = np.tile(np.array([0, 1, 2], dtype=np.int64), 3)
        train = rng.normal(size=(len(y), 4, 100)).astype(np.float32)
        test = rng.normal(size=(3, 4, 100)).astype(np.float32)
        pred, metadata = fit_predict_cocot(
            train, y, test, 3, sampling_rate=100, epochs=1, patience=1,
            batch_size=4, mixup_alpha=0.2, model_kwargs=self.small_model_kwargs(),
        )
        self.assertEqual(pred.shape, (3,))
        self.assertEqual(metadata["mixup_alpha"], 0.2)

    def test_training_resamples_to_paper_sampling_rate(self):
        from wyu_bci_bench.cocot import fit_predict_cocot

        rng = np.random.default_rng(9)
        y = np.tile(np.array([0, 1, 2], dtype=np.int64), 3)
        train = rng.normal(size=(len(y), 4, 120)).astype(np.float32)
        test = rng.normal(size=(3, 4, 120)).astype(np.float32)
        pred, _ = fit_predict_cocot(
            train, y, test, 3, sampling_rate=120, epochs=1, patience=1,
            batch_size=4, model_kwargs=self.small_model_kwargs(),
        )
        self.assertEqual(pred.shape, (3,))


if __name__ == "__main__":
    unittest.main()
