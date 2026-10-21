import unittest

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = nn = None


@unittest.skipIf(torch is None, "PyTorch is required")
class FineTuningTests(unittest.TestCase):
    def setUp(self):
        from wyu_bci_bench.foundation.finetuning import configure_trainable_parameters
        self.configure = configure_trainable_parameters
        self.model = nn.Sequential(
            nn.Linear(4, 8), nn.Sequential(nn.Linear(8, 8), nn.ReLU()), nn.Linear(8, 2)
        )

    def test_full_and_frozen_parameter_counts(self):
        from wyu_bci_bench.foundation.finetuning import count_trainable_parameters
        full = self.configure(self.model, "full")
        self.assertEqual(full.trainable, full.total)
        frozen = self.configure(self.model, "frozen")
        self.assertEqual(frozen.trainable, 0)
        self.assertEqual(count_trainable_parameters(self.model).fraction, 0.0)

    def test_selected_parent_unfreezes_only_its_subtree(self):
        result = self.configure(self.model, "selected", module_names=["1.0"])
        expected = sum(p.numel() for p in self.model[1][0].parameters())
        self.assertEqual(result.trainable, expected)
        self.assertTrue(all(p.requires_grad for p in self.model[1][0].parameters()))
        self.assertFalse(any(p.requires_grad for p in self.model[0].parameters()))

    def test_bad_selection_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            self.configure(self.model, "selected")
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.configure(self.model, "selected", module_names=["encoder.blocks.99"])

    def test_peft_lora_injection_and_trainable_report(self):
        try:
            from peft import PeftModel
        except ImportError:
            self.skipTest("optional PEFT package is unavailable")
        from wyu_bci_bench.foundation.finetuning import attach_peft_adapter, count_trainable_parameters

        model = attach_peft_adapter(nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2)),
                                    method="lora", target_modules=["0", "1"], rank=2, alpha=4)
        self.assertIsInstance(model, PeftModel)
        counts = count_trainable_parameters(model)
        self.assertGreater(counts.trainable, 0)
        self.assertLess(counts.trainable, counts.total)
        self.assertEqual(tuple(model(torch.ones(3, 4)).shape), (3, 2))

    def test_supported_peft_variants_run_on_toy_backbone(self):
        try:
            import peft  # noqa: F401
        except ImportError:
            self.skipTest("optional PEFT package is unavailable")
        from wyu_bci_bench.foundation.finetuning import attach_peft_adapter

        for method in ("dora", "pissa", "adalora"):
            with self.subTest(method=method):
                model = attach_peft_adapter(
                    nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2)), method=method,
                    target_modules=["0", "1"], rank=2, alpha=4,
                    total_steps=100 if method == "adalora" else None,
                )
                self.assertEqual(tuple(model(torch.ones(2, 4)).shape), (2, 2))

    def test_adalora_requires_training_schedule(self):
        from wyu_bci_bench.foundation.finetuning import attach_peft_adapter
        with self.assertRaisesRegex(ValueError, "total_steps"):
            attach_peft_adapter(nn.Sequential(nn.Linear(4, 2)), method="adalora",
                                target_modules=["0"], rank=2)


if __name__ == "__main__":
    unittest.main()
