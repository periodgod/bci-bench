import unittest
import hashlib
import json
import tempfile
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from wyu_bci_bench.foundation.sessionfm import SessionFM
from wyu_bci_bench.foundation.sessionfm_downstream import (
    SessionFMDownstream, SessionFMState, configure_sessionfm_stage,
    sessionfm_parameter_groups,
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_sessionfm_k5 import REQUIRED_K4, verify_k4_summary


class SessionFMDownstreamTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.backbone = SessionFM(depth=3, heads_per_axis=4, hidden=32, drop_path=0)
        self.model = SessionFMDownstream(self.backbone)
        self.x = torch.randn(4, 41, 480)
        self.coords = torch.randn(41, 3)

    def test_zero_gate_returns_frozen_prior_logits(self):
        self.backbone.gate.data.fill_(-1000)
        pf = torch.randn(4, 1394)
        anchor = torch.randn(4, 3)
        result = self.model(self.x, coords=self.coords, prior_features=pf, prior_logits=anchor)
        torch.testing.assert_close(result["logits"], anchor, rtol=0, atol=0)

    def test_stream_matches_batch_causal_reference(self):
        self.backbone.eval()
        with torch.no_grad():
            batch_spec = __import__("wyu_bci_bench.foundation.sessionfm", fromlist=["session_spectral_features"]).session_spectral_features(self.x)
            batch_prior = self.model.prior_features(self.x).numpy()
            from wyu_bci_bench.session_causal import causal_center, causal_class_balance
            pcenter = causal_center(batch_prior, np.zeros(4, dtype=int))
            logits = self.model.prior_logits(torch.tensor(pcenter, dtype=torch.float32))
            baseline_rows = []
            running = []
            for i in range(4):
                if running:
                    n = len(running); w = n / (n + 32)
                    baseline_rows.append(w * np.mean(running, axis=0) +
                                         (1-w) * self.backbone.prior_mean[:41].numpy())
                else:
                    baseline_rows.append(self.backbone.prior_mean[:41].numpy())
                running.append(batch_spec[i].numpy().reshape(41, 4, 10).mean(1))
            bresult = self.backbone(self.x, coords=self.coords, baseline=torch.tensor(np.stack(baseline_rows)),
                                    n_seen=torch.arange(4), prior_logits=logits)
            expected = causal_class_balance(torch.softmax(bresult["logits"], -1).numpy(), np.zeros(4, dtype=int))
            state = SessionFMState()
            observed = torch.cat([self.model.predict_stream(self.x[i], coords=self.coords, state=state)
                                 for i in range(4)]).numpy()
        np.testing.assert_allclose(observed, expected, atol=1e-6, rtol=0)

    def test_stateless_predictions_do_not_depend_on_window_order(self):
        self.backbone.eval()
        with torch.no_grad():
            p1 = [self.model.predict_stream(x, coords=self.coords, state=SessionFMState(), stateful=False)
                  for x in self.x]
            order = [2, 0, 3, 1]
            p2 = [self.model.predict_stream(self.x[i], coords=self.coords, state=SessionFMState(), stateful=False)
                  for i in order]
        for i, j in enumerate(order):
            torch.testing.assert_close(p1[j], p2[i], rtol=0, atol=1e-6)

    def test_stage_one_keeps_backbone_frozen_except_readout(self):
        configure_sessionfm_stage(self.model, 1)
        self.model.eval()
        before = {n: p.detach().clone() for n, p in self.backbone.named_parameters()}
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=4e-4)
        out = self.model(self.x, coords=self.coords, prior_logits=torch.randn(4, 3))
        out["logits"].sum().backward(); optimizer.step()
        for n, p in self.backbone.named_parameters():
            if n not in {"pool_query", "gate"} and not n.startswith("residual_head."):
                torch.testing.assert_close(p, before[n])

    def test_layer_learning_rates_decay_from_top_by_point_65(self):
        groups = sessionfm_parameter_groups(self.model)
        block_lrs = []
        for block in self.backbone.blocks:
            block_parameter_ids = {id(p) for p in block.parameters()}
            block_lrs.append(next(g["lr"] for g in groups
                                  if block_parameter_ids.intersection(id(p) for p in g["params"])))
        self.assertEqual(len(block_lrs), 3)
        self.assertAlmostEqual(block_lrs[0], 1e-4 * 0.65 ** 2)
        self.assertAlmostEqual(block_lrs[1], 1e-4 * 0.65)
        self.assertAlmostEqual(block_lrs[2], 1e-4)

    def test_k5_preflight_rejects_missing_k4_arm_and_accepts_matching_synthetic_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "main_20261101").mkdir()
            checkpoint = root / "main_20261101" / "weights.pt"
            checkpoint.write_bytes(b"synthetic checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            summary = root / "summary.json"
            summary.write_text(json.dumps({"runs": [{"name": "main_20261101", "weights_sha256": digest}]}),
                               encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing required arms"):
                verify_k4_summary(summary)
            for name in REQUIRED_K4[1:]:
                (root / name).mkdir()
                (root / name / "weights.pt").write_bytes(b"synthetic checkpoint")
            items = [{"name": name, "weights_sha256": digest} for name in REQUIRED_K4]
            summary.write_text(json.dumps({"runs": items}), encoding="utf-8")
            self.assertEqual(len(verify_k4_summary(summary)["runs"]), 7)


if __name__ == "__main__":
    unittest.main()
