import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required")
class BraindecodeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import braindecode  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("optional Braindecode package is unavailable") from exc

    def test_supported_models_accept_bench_epoch_shape(self):
        from wyu_bci_bench.deep_models import available_models, make_braindecode_model

        for name in available_models():
            with self.subTest(name=name):
                model = make_braindecode_model(
                    name, channels=8, samples=500, classes=3, sampling_rate=250
                )
                scores = model(torch.randn(2, 8, 500))
                self.assertEqual(tuple(scores.shape), (2, 3))

    def test_fold_local_training_returns_trial_predictions(self):
        from wyu_bci_bench.deep_models import fit_predict_braindecode

        rng = np.random.default_rng(7)
        train = rng.normal(size=(12, 8, 500)).astype(np.float32)
        labels = np.tile(np.array([0, 1, 2], dtype=np.int64), 4)
        train[labels == 1, 0, :] += 0.3
        train[labels == 2, 1, :] += 0.3
        test = rng.normal(size=(3, 8, 500)).astype(np.float32)
        pred, metadata = fit_predict_braindecode(
            "shallow_fbcsp", train, labels, test, 250, 3,
            seed=4, max_epochs=2, patience=2,
        )
        self.assertEqual(pred.shape, (3,))
        self.assertTrue(np.isin(pred, [0, 1, 2]).all())
        self.assertEqual(metadata["model"], "shallow_fbcsp")
        self.assertGreater(metadata["parameter_count"], 0)

    def test_unknown_model_is_rejected(self):
        from wyu_bci_bench.deep_models import make_braindecode_model
        with self.assertRaisesRegex(ValueError, "unknown model"):
            make_braindecode_model("not-a-model", channels=8, samples=500,
                                   classes=3, sampling_rate=250)

    def test_temporal_model_families_return_class_scores(self):
        from wyu_bci_bench.deep_models import make_braindecode_model

        for name in ("tcn", "eegtcnet", "eeginception_mi", "eegnex", "eegitnet"):
            with self.subTest(name=name):
                model = make_braindecode_model(
                    name, channels=8, samples=500, classes=3, sampling_rate=250
                )
                scores = model(torch.randn(2, 8, 500))
                self.assertEqual(tuple(scores.shape), (2, 3))

    def test_local_resnet_and_inception_time_accept_odd_sample_counts(self):
        from wyu_bci_bench.deep_models import make_braindecode_model

        for name in ("eeg_resnet", "inception_time"):
            with self.subTest(name=name):
                model = make_braindecode_model(
                    name, channels=8, samples=503, classes=3, sampling_rate=250
                )
                self.assertEqual(tuple(model(torch.randn(2, 8, 503)).shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
