"""Contract, true encode/cached equivalence, eval dropout and task-gradient checks."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn

from wyu_bci_bench.foundation.brainomni import (prepare_input, normalize_positions,
    TokenBackbone, BrainOmniClassifier, audit_task_gradients, load_encoder)
from wyu_bci_bench._third_party.brainomni.model_utils.attn import SelfAttention


def fake_encoder():
    return SimpleNamespace(lm_dim=8, projection=nn.Identity(),
        tokenizer=SimpleNamespace(encoder=SimpleNamespace(neuros=nn.Parameter(torch.randn(3, 8)))),
        blocks=nn.ModuleList([nn.Linear(8, 8) for _ in range(3)]))


class BrainOmniTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_microvolt_conversion_matches_official_volt_epsilon(self):
        x = np.random.default_rng(3).normal(0, 25, (2, 4, 512))
        volts = x * 1e-6
        volts -= volts.mean(1, keepdims=True)
        expected = volts / (volts.std((1, 2), keepdims=True) + 1e-5)
        np.testing.assert_allclose(prepare_input(x), expected, rtol=1e-6, atol=1e-7)
        for bad, fs in [(x, 200), (x[:, :, :255], 256), (x * np.nan, 256)]:
            with self.assertRaises(ValueError):
                prepare_input(bad, fs=fs)

    def test_one_second_window_normalizes_only_observed_samples(self):
        x = np.random.default_rng(9).normal(0, 25, (2, 4, 256))
        volts = x * 1e-6
        volts -= volts.mean(1, keepdims=True)
        expected = volts / (volts.std((1, 2), keepdims=True) + 1e-5)
        actual = prepare_input(x)
        self.assertEqual(actual.shape, x.shape)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        padded_first = prepare_input(np.pad(x, ((0, 0), (0, 0), (0, 256))))[:, :, :256]
        self.assertGreater(float(np.max(np.abs(actual-padded_first))), .1)

    def test_one_second_flat_window_stays_finite(self):
        actual = prepare_input(np.ones((1, 4, 256))*25)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_array_equal(actual, np.zeros_like(actual))

    def test_geometry_translation_and_scale_invariance(self):
        xyz = np.array([[1, 0, 0], [-1, 0, 0], [0, 2, 1]], float)
        p = normalize_positions(xyz)
        np.testing.assert_allclose(p, normalize_positions(xyz * 3 + 9), atol=1e-7)
        np.testing.assert_allclose(np.mean(p[:, :3] ** 2) * 9, 1, atol=1e-6)
        self.assertTrue((p[:, 3:] == 0).all())
        with self.assertRaises(ValueError):
            normalize_positions(np.zeros((3, 3)))

    def test_cached_path_omits_final_block_and_freezes_source_vectors(self):
        encoder = fake_encoder()
        backbone = TokenBackbone(encoder)
        x = torch.randn(2, 3, 4, 8)
        expected = x + encoder.tokenizer.encoder.neuros.detach()[None, :, None]
        for block in encoder.blocks[:-1]:
            expected = block(expected)
        expected = torch.nn.functional.normalize(expected, dim=-1, eps=1e-6)
        torch.testing.assert_close(backbone(x), expected)
        self.assertEqual(len(backbone.blocks), 2)
        self.assertFalse(backbone.neuro.requires_grad)

    def test_selected_scope_and_actual_gradients(self):
        for strategy in ['frozen', 'last_block', 'full']:
            model = BrainOmniClassifier(fake_encoder(), strategy=strategy)
            audit = audit_task_gradients(model, torch.randn(4, 3, 4, 8), torch.tensor([0, 1, 0, 1]))
            self.assertTrue(audit['connected'])
            names = [n for n, p in model.named_parameters() if p.requires_grad]
            if strategy == 'frozen':
                self.assertTrue(all(n.startswith('class_head.') for n in names))
                self.assertFalse(model.backbone.training)
            elif strategy == 'last_block':
                self.assertFalse(any(n.startswith('backbone.blocks.0.') for n in names))
                self.assertTrue(any(n.startswith('backbone.blocks.1.') for n in names))

    def test_gradient_audit_rejects_unused_trainable_parameter(self):
        model = BrainOmniClassifier(fake_encoder(), strategy='full')
        model.unused = nn.Parameter(torch.ones(1))
        with self.assertRaisesRegex(RuntimeError, 'unused'):
            audit_task_gradients(model, torch.randn(2, 3, 4, 8), torch.tensor([0, 1]))

    def test_eval_attention_is_deterministic_but_training_dropout_remains(self):
        attention = SelfAttention(16, 4, .5).eval()
        x = torch.randn(2, 5, 16)
        torch.testing.assert_close(attention(x), attention(x), atol=0, rtol=0)
        attention.train()
        self.assertFalse(torch.equal(attention(x), attention(x)))

    def test_strict_checkpoint_loading_rejects_missing_key(self):
        from wyu_bci_bench._third_party.brainomni.brainomni.model import BrainOmni
        cfg = dict(window_length=64, n_filters=2, ratios=[2, 2], kernel_size=3,
            last_kernel_size=3, n_dim=16, n_head=2, n_neuro=4, dropout=0.,
            codebook_dim=16, codebook_size=8, num_quantizers=2, rotation_trick=False,
            quantize_optimize_method='ema', overlap_ratio=.25, lm_dim=16,
            lm_head=4, lm_depth=3, lm_dropout=.1, mask_ratio=.5, num_quantizers_used=2)
        model = BrainOmni(**cfg)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory/'model_cfg.json').write_text(json.dumps(cfg), encoding='utf-8')
            state = model.state_dict()
            torch.save(state, directory/'BrainOmni.pt')
            loaded = load_encoder(directory)
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
            state.pop('mask_token')
            torch.save(state, directory/'BrainOmni.pt')
            with self.assertRaises(RuntimeError):
                load_encoder(directory)


if __name__ == '__main__':
    unittest.main()
