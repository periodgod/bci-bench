"""Mechanism checks: rank allocation, initialization, SSL equivalence and sampling."""
import copy
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from wyu_bci_bench.foundation.brainomni import BrainOmniClassifier
from wyu_bci_bench.foundation.brainomni_adaptation import (subject_pair_order,
    adalora_objective, AdaLoraStep, BrainOmniMaskedTokens, BrainOmniDEFT, BrainOmniResidualMLP)


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(8, 8)
        self.proj = nn.Linear(8, 8)

    def forward(self, x):
        return x + self.proj(self.qkv(x).tanh())


def encoder():
    return SimpleNamespace(lm_dim=8, projection=nn.Identity(),
        tokenizer=SimpleNamespace(encoder=SimpleNamespace(neuros=nn.Parameter(torch.randn(3, 8)))),
        blocks=nn.ModuleList([AttentionBlock() for _ in range(3)]))


class AdaptationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_pair_sampler_has_positives_and_preserves_every_row(self):
        subjects = np.repeat(np.arange(5), 6)
        ix = subject_pair_order(subjects, 42)
        np.testing.assert_array_equal(np.sort(ix), np.arange(30))
        self.assertTrue((subjects[ix[::2]] == subjects[ix[1::2]]).all())
        self.assertFalse(np.array_equal(ix, subject_pair_order(subjects, 43)))
        with self.assertRaises(ValueError):
            subject_pair_order([0, 0, 0], 1)

    def test_dora_and_pissa_preserve_initial_function(self):
        x = torch.randn(4, 3, 4, 8)
        for strategy in ['dora', 'pissa']:
            e = encoder()
            torch.manual_seed(11)
            original = BrainOmniClassifier(copy.deepcopy(e), strategy='frozen').eval()
            torch.manual_seed(11)
            adapted = BrainOmniClassifier(copy.deepcopy(e), strategy=strategy, rank=2).eval()
            torch.testing.assert_close(original(x), adapted(x), atol=2e-6, rtol=2e-5)

    def test_adalora_actually_updates_and_finalizes_rank(self):
        m = BrainOmniClassifier(encoder(), strategy='adalora', rank=2, total_steps=30)
        callback = AdaLoraStep()
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=.001)
        x, y = torch.randn(4, 3, 4, 8), torch.tensor([0, 1, 0, 1])
        for step in range(30):
            opt.zero_grad(set_to_none=True)
            loss = adalora_objective(m, (x,), y, None)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            opt.step()
            callback(step, m)
        cfg = m.backbone.peft_config['default']
        self.assertTrue(cfg.rank_pattern)
        total_rank = sum(sum(v) for v in cfg.rank_pattern.values())
        self.assertEqual(total_rank, 2 * len(cfg.rank_pattern))
        self.assertLess(total_rank, 4 * len(cfg.rank_pattern))
        self.assertEqual(callback.trace[-1]['step'], 29)

    def test_deft_all_auxiliary_paths_connected_and_mlp_capacity_matched(self):
        base = BrainOmniClassifier(encoder(), strategy='full')
        m = BrainOmniDEFT(base)
        loss, parts = m.loss(torch.randn(4, 3, 4, 8), torch.tensor([0, 1, 0, 1]), torch.tensor([0, 0, 1, 1]))
        self.assertEqual(set(parts), {'task','style_cl','style_invar','var_ratio','recon'})
        self.assertTrue(all(torch.isfinite(v) for v in parts.values()))
        loss.backward()
        missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
        self.assertEqual(missing, [])
        adapter_params = sum(p.numel() for n, p in m.named_parameters()
            if not n.startswith(('deft.backbone.', 'deft.classifier.')))
        control = BrainOmniResidualMLP(BrainOmniClassifier(encoder(), strategy='full'), adapter_params)
        count = sum(p.numel() for p in control.adapter.parameters())
        self.assertLessEqual(abs(count-adapter_params), (2*8+1)/2)

    def test_cached_ssl_matches_official_mask_loss_and_updates_readout_block(self):
        from wyu_bci_bench._third_party.brainomni.brainomni.model import BrainOmni
        cfg = dict(window_length=64, n_filters=2, ratios=[2, 2], kernel_size=3,
            last_kernel_size=3, n_dim=16, n_head=2, n_neuro=4, dropout=0.,
            codebook_dim=16, codebook_size=8, num_quantizers=2, rotation_trick=False,
            quantize_optimize_method='ema', overlap_ratio=.25, lm_dim=16,
            lm_head=4, lm_depth=3, lm_dropout=0., mask_ratio=.5, num_quantizers_used=2)
        e = BrainOmni(**cfg).eval()
        tokens = torch.randn(2, 4, 5, 16)
        labels = torch.randint(0, 8, (2, 4, 5, 2))
        # Bypass tokenizer with identical cached outputs: random sequence is exact.
        e.tokenizer.tokenize = lambda *args: (tokens, labels)
        torch.manual_seed(12)
        expected = e(None, None, None)['loss']
        adapted = BrainOmniMaskedTokens(e).eval()
        torch.manual_seed(12)
        actual = adapted(tokens, labels)
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
        actual.backward()
        active = [n for n, p in adapted.named_parameters() if p.requires_grad]
        self.assertTrue(active)
        self.assertTrue(all(n.startswith('encoder.blocks.1.') for n in active))
        self.assertTrue(all(p.grad is not None for p in e.blocks[-2].parameters()))


if __name__ == '__main__':
    unittest.main()
