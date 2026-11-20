import unittest
import numpy as np


class CBRaModTokenTests(unittest.TestCase):
    def test_token_shape_and_global_mean_equivalence(self):
        from wyu_bci_bench.foundation.cbramod import extract_features, extract_tokens
        from pathlib import Path
        import os
        repo = Path(os.environ.get("CBRAMOD_REPO", "D:/脑电大模型/tmp_repos/cbramod"))
        weights = repo / "pretrained_weights/pretrained_weights.pth"
        if not weights.exists():
            self.skipTest("local verified CBraMod weights are unavailable")
        rng = np.random.default_rng(7)
        x = rng.normal(size=(2, 41, 480)).astype(np.float32)
        tokens = extract_tokens(x, 120, repo, weights, device="cpu", batch=2)
        features = extract_features(x, 120, repo, weights, device="cpu", batch=2)
        self.assertEqual(tokens.shape, (2, 41, 4, 200))
        np.testing.assert_allclose(tokens.mean((1, 2)), features, rtol=2e-5, atol=2e-5)
        self.assertTrue(np.isfinite(tokens).all())

    def test_prefix_last_block_matches_full_encoder(self):
        import torch
        from pathlib import Path
        from wyu_bci_bench.foundation.cbramod import extract_tokens, make_last_block_classifier
        repo = Path("D:/脑电大模型/tmp_repos/cbramod")
        weights = repo / "pretrained_weights/pretrained_weights.pth"
        if not weights.exists():
            self.skipTest("local checkpoint unavailable")
        x = np.random.default_rng(2).normal(size=(1, 3, 480)).astype(np.float32)
        prefix = extract_tokens(x, 120, repo, weights, device="cpu", before_last=True)
        full = extract_tokens(x, 120, repo, weights, device="cpu")
        net = make_last_block_classifier(repo, weights).eval()
        with torch.no_grad():
            reconstructed = net.block(torch.from_numpy(prefix)).numpy()
        np.testing.assert_allclose(reconstructed, full, rtol=1e-5, atol=1e-5)
        net(torch.from_numpy(prefix)).square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in net.block.parameters()))

    def test_region_pooling_retains_time_and_channel_identity(self):
        from wyu_bci_bench.foundation.cbramod import pool_tokens
        t = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
        np.testing.assert_allclose(pool_tokens(t, "region_phase", [[0, 1], [2, 3]]), [[2.5, 4.5, 10.5, 12.5]])
        with self.assertRaises(ValueError):
            pool_tokens(t, "region_phase", [[0, 1], [1, 2]])


if __name__ == "__main__":
    unittest.main()
