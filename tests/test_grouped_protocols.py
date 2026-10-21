from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from wyu_bci_bench.grouped import GroupedEEGDataset, load_grouped_npz
from wyu_bci_bench.metrics import (cell_balanced_accuracy, exact_sign_flip_pvalue, grouped_balanced_accuracy,
                                   paired_subject_bootstrap, stream_mean_balanced_accuracy, summarize_predictions)
from wyu_bci_bench.registry import load_oof, save_run
from wyu_bci_bench.splits import cross_session, inner_loso, loso, neuralbench_sealed, subject_ratio
from wyu_bci_bench.submission import build_zip, check_zip


def toy(n_people=4, per_cell=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_people):
        for ses in ("ses-1", "ses-2"):
            for itf in ("Graz", "BrainHero"):
                for i in range(per_cell):
                    rows.append((f"sub-P{p + 1:02d}", ses, itf, i % 3))
    subjects, sessions, interfaces, y = (np.array(c) for c in zip(*rows))
    X = rng.standard_normal((len(y), 4, 32)).astype(np.float32)
    return X, y.astype(np.int64), subjects, sessions, interfaces


class MetricTests(unittest.TestCase):
    def test_perfect_and_chance(self):
        _, y, s, ses, itf = toy()
        self.assertAlmostEqual(cell_balanced_accuracy(y, y, s, ses, itf), 1.0)
        self.assertAlmostEqual(cell_balanced_accuracy(y, np.zeros_like(y), s, ses, itf), 1 / 3)

    def test_cells_are_equal_weight(self):
        y = np.array([0, 1, 2] * 2 + [0, 1, 2] * 10)
        s = np.array(["a"] * 6 + ["b"] * 30)
        pred = y.copy()
        pred[6:] = 0  # big cell at chance, small cell perfect
        score = cell_balanced_accuracy(y, pred, s, np.full(36, "x"), np.full(36, "x"))
        self.assertAlmostEqual(score, (1 + 1 / 3) / 2)

    def test_absent_class_policy(self):
        y = np.array([0, 0, 1, 1])
        pred = y.copy()
        zero = grouped_balanced_accuracy(y, pred, (np.zeros(4),), 3, absent="zero")[0]
        skip = grouped_balanced_accuracy(y, pred, (np.zeros(4),), 3, absent="skip")[0]
        self.assertAlmostEqual(zero, 2 / 3)
        self.assertAlmostEqual(skip, 1.0)

    def test_stream_mean_ignores_interface(self):
        _, y, s, ses, itf = toy()
        self.assertAlmostEqual(stream_mean_balanced_accuracy(y, y, s, ses), 1.0)

    def test_bootstrap_and_sign_flip(self):
        _, y, s, ses, itf = toy(n_people=6)
        good = summarize_predictions(y, y, s, ses, itf)
        bad = summarize_predictions(y, np.zeros_like(y), s, ses, itf)
        r = paired_subject_bootstrap(good, bad, n_boot=500, seed=1)
        self.assertAlmostEqual(r["cell_difference"], 2 / 3)
        self.assertEqual(r["improved_subjects"], 6)
        self.assertAlmostEqual(exact_sign_flip_pvalue([1.0] * 6), 2 / 64)
        self.assertAlmostEqual(exact_sign_flip_pvalue([1.0, -1.0]), 1.0)


class SplitTests(unittest.TestCase):
    def test_loso_and_inner(self):
        _, _, s, ses, _ = toy()
        folds = list(loso(s))
        self.assertEqual(len(folds), 4)
        for f in folds:
            f.check_disjoint(s)
            inner = list(inner_loso(s, f.train))
            self.assertEqual(len(inner), 3)
            for i in inner:
                self.assertFalse(np.any(i.test & f.test))
                i.check_disjoint(s)

    def test_cross_session_and_sealed(self):
        _, _, s, ses, _ = toy()
        x = cross_session(s, ses, "sub-P01")
        self.assertTrue(set(ses[x.train]) == {"ses-1"} and set(ses[x.test]) == {"ses-2"})
        self.assertTrue(set(s[x.train] ) == set(s[x.test]) == {"sub-P01"})
        sealed = neuralbench_sealed(s, ses, ["sub-P02"], sealed_sessions=["ses-2"])
        self.assertTrue(np.all(s[sealed.test] == "sub-P02") and np.all(ses[sealed.test] == "ses-2"))

    def test_subject_ratio_is_subject_isolated_and_seeded(self):
        s = np.repeat([f"p{i}" for i in range(20)], 5)
        a, b = subject_ratio(s, 1), subject_ratio(s, 1)
        self.assertTrue(np.array_equal(a.test, b.test))
        self.assertNotEqual(subject_ratio(s, 2).info["test_subjects"], a.info["test_subjects"])
        self.assertEqual(len(a.info["test_subjects"]), 2)


class WindowFeatureTests(unittest.TestCase):
    def test_variant_dimensions(self):
        from wyu_bci_bench.window_features import extract_variant

        X = (np.random.default_rng(0).standard_normal((5, 41, 480)) * 10).astype(np.float32)
        self.assertEqual(extract_variant(X, "whole").shape, (5, 41 * 17))
        self.assertEqual(extract_variant(X, "pre_post_car").shape, (5, 2 * 41 * 17))
        self.assertEqual(extract_variant(X, "post_no_high").shape, (5, 41 * 15))
        with self.assertRaises(ValueError):
            extract_variant(X, "middle")
        with self.assertRaises(ValueError):
            extract_variant(X[:, :10], "whole")


class IOTests(unittest.TestCase):
    def test_grouped_npz_and_registry_roundtrip(self):
        X, y, s, ses, itf = toy()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "w.npz"
            np.savez(p, X=X, y=y, subjects=s, sessions=ses, interfaces=itf, channels=np.array(["a", "b", "c", "d"]))
            d = load_grouped_npz(p, fs=8, class_names=["a", "b", "c"])
            self.assertIsInstance(d, GroupedEEGDataset)
            self.assertEqual(d.summary()["cells"], 16)
            probs = np.eye(3, dtype=np.float32)[y]
            report = save_run(Path(tmp) / "run", y=y, subjects=s, sessions=ses, interfaces=itf,
                              probs={"oracle": probs, "seeds": np.stack([probs, probs])},
                              protocol={"split": "toy"}, sources={"cache": p})
            self.assertAlmostEqual(report["results"]["oracle"]["cell_ba"], 1.0)
            oof = load_oof(Path(tmp) / "run" / "oof.npz")
            self.assertEqual(oof.ensemble("seeds").shape, probs.shape)
            with self.assertRaises(ValueError):
                save_run(Path(tmp) / "bad", y=y, subjects=s, sessions=ses, interfaces=itf,
                         probs={"bad": probs * 2}, protocol={})

    def test_submission_zip_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "pkg"
            folder.mkdir()
            (folder / "submission.py").write_text(
                "def reset_state():\n    pass\n\ndef load(meta):\n    return meta['submission_dir']\n", encoding="utf-8")
            (folder / "weights.npz").write_bytes(b"0")
            manifest = build_zip(folder, Path(tmp) / "pkg.zip")
            self.assertEqual(set(manifest["members"]), {"submission.py", "weights.npz"})
            self.assertTrue(check_zip(manifest["zip"])["ok"])
            bad = Path(tmp) / "bad.zip"
            with zipfile.ZipFile(bad, "w") as zf:
                zf.writestr("pkg/submission.py", "x = 1\n")
            self.assertFalse(check_zip(bad)["ok"])


class TorchTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

    def test_prior_dim_and_nets(self):
        from wyu_bci_bench import nets
        from wyu_bci_bench.priors import math_prior_features, prior_feature_dim

        rng = np.random.default_rng(0)
        X = (rng.standard_normal((12, 41, 480)) * 20).astype(np.float32)
        y = np.arange(12) % 3
        f = math_prior_features(X, device="cpu")
        self.assertEqual(f.shape, (12, prior_feature_dim()))
        for v in nets.VARIANTS:
            m = nets.make_model(f, y, v)
            self.assertEqual(nets.predict_proba(m, X, f, "cpu").shape, (12, 3))

    def test_ssl_losses_finite(self):
        import torch

        from wyu_bci_bench.ssl import RepresentationModel, augment, simclr_loss, vicreg_loss

        torch.manual_seed(0)
        x = torch.randn(16, 41, 480) * 20
        model = RepresentationModel()
        g = torch.Generator().manual_seed(1)
        za, zb = model(torch.cat((augment(x, g), augment(x, g)))).chunk(2)
        for fn in (simclr_loss, vicreg_loss):
            self.assertTrue(torch.isfinite(fn(za, zb)))

    def test_ts2vec_hierarchy_shapes_gradients_and_pooling(self):
        import torch

        from wyu_bci_bench.ssl import TS2VecModel, hierarchical_contrastive_loss

        model = TS2VecModel(n_channels=5, width=8, out_features=6, depth=2)
        x = torch.randn(4, 5, 33)
        z1, z2 = model(x), model(x + 0.01 * torch.randn_like(x))
        self.assertEqual(tuple(z1.shape), (4, 33, 6))
        self.assertEqual(tuple(model.pool(z1).shape), (4, 6))
        loss = hierarchical_contrastive_loss(z1, z2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.input.weight.grad).all())
        self.assertTrue(torch.isfinite(hierarchical_contrastive_loss(z1[:, :1], z2[:, :1])))

    def test_ts2vec_training_entrypoint_runs(self):
        import torch

        from wyu_bci_bench.ssl import TS2VecModel, train_ts2vec

        model = TS2VecModel(n_channels=3, width=8, out_features=4, depth=1)
        raw = torch.randn(8, 3, 17)
        trace = train_ts2vec(model, raw, np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                             seed=3, epochs=1, batch=4)
        self.assertEqual(len(trace), 1)
        self.assertTrue(np.isfinite(trace[0]))

    def test_supcon_uses_cross_subject_positives_only(self):
        import torch

        from wyu_bci_bench.ssl import cross_subject_supcon

        emb = torch.nn.functional.normalize(torch.randn(6, 4), dim=1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        same_person = torch.zeros(6, dtype=torch.long)
        self.assertEqual(float(cross_subject_supcon(emb, labels, same_person, 0.1)), 0.0)
        self.assertGreater(float(cross_subject_supcon(emb, labels, torch.arange(6), 0.1)), 0.0)

    def test_foundation_contract_gates_verified_and_pending_models(self):
        from wyu_bci_bench.foundation import require_verified

        self.assertTrue(require_verified("cbramod").verified)
        self.assertTrue(require_verified("eegpt").verified)
        with self.assertRaises(NotImplementedError):
            require_verified("labram")

    def test_biot_contract_is_verified_and_specific(self):
        from wyu_bci_bench.foundation import get_contract, require_verified

        self.assertTrue(require_verified("biot").verified)
        contract = get_contract("biot")
        self.assertIn("200 Hz", " ".join(contract.steps))
        self.assertIn("256-D", contract.output)


if __name__ == "__main__":
    unittest.main()
