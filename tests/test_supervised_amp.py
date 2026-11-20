import unittest
import numpy as np
import torch
from wyu_bci_bench.supervised import train_classifier, predict_classifier


class SupervisedAMPTests(unittest.TestCase):
    def test_default_fp32_training_and_probability_contract(self):
        torch.manual_seed(1)
        model = torch.nn.Linear(3, 2)
        x = np.array([[1, 0, 0], [-1, 0, 0], [2, 0, 0], [-2, 0, 0]], np.float32)
        y = np.array([0, 1, 0, 1])
        initial = model.weight.detach().clone()
        model, report = train_classifier(model, (x,), y, epochs=2, seed=3, batch=2, device='cpu')
        self.assertFalse(torch.equal(initial, model.weight))
        self.assertIsNone(report['amp_dtype'])
        p = predict_classifier(model, (x,), device='cpu', batch=3)
        self.assertEqual(p.shape, (4, 2))
        self.assertEqual(p.dtype, np.float32)
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)

    def test_invalid_precision_or_cpu_bfloat16_raise_before_fit(self):
        model = torch.nn.Linear(2, 2)
        x, y = np.ones((4, 2), np.float32), np.array([0, 1, 0, 1])
        for precision in ['float16', 'bfloat16']:
            with self.assertRaises(ValueError):
                train_classifier(model, (x,), y, epochs=1, seed=0, device='cpu', amp_dtype=precision)
