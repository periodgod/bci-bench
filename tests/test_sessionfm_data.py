from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wyu_bci_bench.external_datasets import load_bnci2015_004_windows, load_lee2019_mi_windows
from wyu_bci_bench.grouped import GroupedEEGDataset, load_grouped_npz
from wyu_bci_bench.pretraining import load_pretraining_corpus, resolve_channel_coordinates


class SessionFMDataTests(unittest.TestCase):
    def _external_cache(self, path: Path, n_channels: int, labels: np.ndarray) -> None:
        n = len(labels)
        np.savez_compressed(
            path,
            X=np.zeros((n, n_channels, 16), dtype=np.float32),
            y=labels,
            subjects=np.array(["P1"] * n),
            sessions=np.array(["ses-1"] * n),
            interfaces=np.array(["all"] * n),
            channels=np.array([f"C{i}" for i in range(n_channels)]),
            recording_order=np.arange(n),
        )

    def test_grouped_dataset_preserves_recording_order_in_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ordered.npz"
            np.savez(path, X=np.zeros((3, 2, 4)), y=[0, 1, 0], subjects=["P"] * 3,
                     sessions=["S"] * 3, interfaces=["all"] * 3, channels=["Cz", "Fz"],
                     recording_order=[2, 1, 3])
            dataset = load_grouped_npz(path, fs=10, class_names=["a", "b"])
            self.assertIsInstance(dataset, GroupedEEGDataset)
            self.assertEqual(dataset.subset(np.array([True, False, True])).recording_order.tolist(), [2, 3])

    def test_pretraining_shard_is_memmapped_and_session_ordered_without_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "signals.npy", np.arange(48, dtype=np.float16).reshape(3, 2, 8))
            np.savez(root / "index.npz", subjects=["P", "P", "P"], sessions=["S", "S", "S"],
                     recording_order=[3, 1, 2])
            manifest = {"shards": [{"name": "s1", "dataset": "toy", "signals": "signals.npy",
                                    "index": "index.npz", "channels": ["Cz", "Fz"], "fs": 120,
                                    "unit": "uV"}]}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            corpus = load_pretraining_corpus(manifest_path)
            self.assertIsInstance(corpus.shards[0].X, np.memmap)
            self.assertEqual(corpus.n_windows, 3)
            self.assertEqual(corpus.shards[0].ordered_session_indices()[("P", "S")].tolist(), [1, 2, 0])
            self.assertFalse(hasattr(corpus.shards[0], "y"))
            corpus.close()

    def test_pretraining_rejects_task_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "signals.npy", np.zeros((1, 1, 8), dtype=np.float16))
            np.savez(root / "index.npz", subjects=["P"], sessions=["S"], recording_order=[0], y=[0])
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"shards": [{"name": "s1", "dataset": "toy",
                "signals": "signals.npy", "index": "index.npz", "channels": ["Cz"],
                "fs": 120, "unit": "uV"}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not contain task labels"):
                load_pretraining_corpus(manifest_path, verify_hashes=False)

    def test_external_loaders_require_explicit_audited_fields_and_class_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            lee = Path(tmp) / "lee.npz"
            self._external_cache(lee, 62, np.array([0, 1]))
            dataset = load_lee2019_mi_windows(lee, class_names=["left", "right"])
            self.assertEqual(dataset.X.shape, (2, 62, 16))
            self.assertEqual(dataset.class_names, ["left", "right"])
            bnci = Path(tmp) / "bnci.npz"
            self._external_cache(bnci, 30, np.array([0, 1, 2]))
            bnci_data = load_bnci2015_004_windows(bnci, class_names=["MI", "math", "word"])
            self.assertEqual(bnci_data.X.shape[1], 30)
            with self.assertRaisesRegex(ValueError, "explicit class names"):
                load_bnci2015_004_windows(bnci, class_names=["only_two", "classes"])

    def test_coordinate_resolution_marks_unknown_channels(self):
        coords, known = resolve_channel_coordinates(["Cz", "unknown"])
        self.assertEqual(coords.shape, (2, 3))
        self.assertEqual(known.tolist(), [True, False])
        self.assertTrue(np.isnan(coords[1]).all())


class SessionFMModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch is not installed")

    def test_parameter_count_and_zero_film_equivalence(self):
        import torch
        from wyu_bci_bench.foundation.sessionfm import SessionFM

        torch.manual_seed(0)
        model = SessionFM(drop_path=0).eval()
        self.assertGreaterEqual(model.parameter_count(), 1_300_000)
        self.assertLessEqual(model.parameter_count(), 1_600_000)
        x = torch.randn(1, 3, 480)
        coords = torch.randn(1, 3, 3)
        baseline = torch.randn(1, 3, 10)
        with torch.inference_mode():
            conditioned = model(x, coords=coords, baseline=baseline, n_seen=13)["tokens"]
            unconditional = model(x, coords=coords, baseline=baseline, n_seen=0)["tokens"]
        torch.testing.assert_close(conditioned, unconditional, atol=0, rtol=0)

    def test_sbc_film_ablation_ignores_session_context(self):
        import torch
        from wyu_bci_bench.foundation.sessionfm import SessionFM

        torch.manual_seed(11)
        model = SessionFM(drop_path=0, use_sbc_film=False).eval()
        x = torch.randn(1, 2, 480)
        coords = torch.randn(2, 3)
        baseline_a = torch.zeros(1, 2, 10)
        with torch.inference_mode():
            a = model(x, coords=coords, baseline=baseline_a, n_seen=0)["tokens"]
            b = model(x, coords=coords, baseline=baseline_a, n_seen=32)["tokens"]
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_masked_waveform_and_spectrum_cannot_leak_into_tokens(self):
        import torch
        from wyu_bci_bench.foundation.sessionfm import SessionFM

        torch.manual_seed(1)
        model = SessionFM(drop_path=0).eval()
        x = torch.randn(1, 2, 480)
        changed = x.clone()
        mask = torch.zeros(1, 2, 8, dtype=torch.bool)
        mask[:, 0, 2:4] = True
        changed[:, 0, 120:240] += 100
        coords = torch.zeros(1, 2, 3)
        with torch.inference_mode():
            a = model(x, coords=coords, mask=mask)
            b = model(changed, coords=coords, mask=mask)
        torch.testing.assert_close(a["tokens"], b["tokens"], atol=0, rtol=0)
        self.assertEqual(a["spec_reconstruction"].shape, (1, 2, 4, 10))
        self.assertEqual(a["wave_reconstruction"].shape, (1, 2, 8, 60))

    def test_stream_batch_order_matches_and_zero_seen_is_stateless(self):
        import torch
        from wyu_bci_bench.foundation.sessionfm import SessionBaselineState, SessionFM

        torch.manual_seed(2)
        model = SessionFM(drop_path=0).eval()
        x = torch.randn(4, 2, 480)
        coords = torch.zeros(2, 3)
        prior_mean = torch.randn(2, 10)
        model.prior_mean[:2].copy_(prior_mean)
        state = SessionBaselineState(alpha=32)
        sequential = []
        with torch.inference_mode():
            for row in x:
                sequential.append(model.forward_stream(row, state, coords=coords)["logits"])
        got = torch.cat(sequential)
        state.reset()
        baselines, counts = [], []
        for row in x:
            baselines.append(state.baseline(prior_mean, n_channels=2).clone())
            counts.append(state.count)
            state.update(__import__("wyu_bci_bench.foundation.sessionfm", fromlist=["session_spectral_features"])
                         .session_spectral_features(row.unsqueeze(0)))
        with torch.inference_mode():
            batch = model(x, coords=coords, baseline=torch.stack(baselines), n_seen=torch.tensor(counts))["logits"]
            stateless = model(x[:1], coords=coords, baseline=prior_mean, n_seen=0)["logits"]
            empty_state = SessionBaselineState()
            stream_stateless = model.forward_stream(x[:1], empty_state, coords=coords)["logits"]
        torch.testing.assert_close(got, batch, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(stream_stateless, stateless, atol=1e-6, rtol=1e-6)

    def test_global_reconstruction_uses_each_masked_position(self):
        import torch
        from wyu_bci_bench.foundation.sessionfm import SessionFM

        model = SessionFM(drop_path=0).eval()
        with torch.inference_mode():
            result = model(torch.randn(2, 3, 480), coords=torch.randn(3, 3))
            global_token = result["global_token"]
            position = result["position_encoding"]
            pair_position = position.reshape(2, 3, 4, 2, model.dim).mean(3)[:, 0, :2]
            pair = model.global_head(torch.cat((global_token[:, None].expand(-1, 2, -1), pair_position), dim=-1))
        self.assertEqual(result["global_reconstruction"].shape, (2, 3, 4, 10))
        self.assertFalse(torch.allclose(pair[:, 0], pair[:, 1]))

    def test_absolute_spectrum_variant_uses_raw_target_for_global_and_spectral_losses(self):
        import torch
        from wyu_bci_bench.foundation.sessionfm import sessionfm_pretraining_spectral_target

        raw = torch.randn(2, 3, 4, 10)
        relative = raw - torch.randn(2, 3, 1, 10)
        target = sessionfm_pretraining_spectral_target(raw, relative, "absolute_spectrum")
        self.assertIs(target, raw)
        torch.testing.assert_close(target, raw)

    def test_global_token_norm_is_channel_count_stable(self):
        import torch
        from wyu_bci_bench.foundation.sessionfm import SessionFM

        torch.manual_seed(7)
        model = SessionFM(drop_path=0).eval()
        norms = []
        with torch.inference_mode():
            for channels in (19, 27):
                output = model(torch.randn(2, channels, 480), coords=torch.randn(channels, 3))
                norms.append(output["global_token"].norm(dim=-1).mean().item())
        self.assertLess(max(norms) / max(min(norms), 1e-8), 1.5)

    def test_training_coordinate_noise_is_stochastic_but_eval_is_stable(self):
        import torch
        from wyu_bci_bench.foundation.sessionfm import SessionFM

        model = SessionFM(drop_path=0)
        x, coords = torch.randn(1, 3, 480), torch.randn(3, 3)
        model.train()
        first, second = model(x, coords=coords)["position_encoding"], model(x, coords=coords)["position_encoding"]
        self.assertFalse(torch.equal(first, second))
        model.eval()
        with torch.inference_mode():
            first, second = model(x, coords=coords)["position_encoding"], model(x, coords=coords)["position_encoding"]
        torch.testing.assert_close(first, second, atol=0, rtol=0)


class SessionFMMaskTests(unittest.TestCase):
    @staticmethod
    def _mask_statistics(mask, dropped, coordinates):
        import torch
        mask = torch.as_tensor(mask)
        dropped = torch.as_tensor(dropped)
        segments = mask.reshape(len(mask), mask.shape[1], 4, 2).all(-1).numpy()
        coords = torch.as_tensor(coordinates, dtype=torch.float32)
        distances = torch.cdist(coords, coords).numpy()
        adjacent = (distances <= 0.03) & (distances > 0)
        pairs = segments[:, :, None, :] & segments[:, None, :, :]
        focal = np.broadcast_to(segments[:, :, None, :], pairs.shape)
        co_rate = pairs[:, adjacent].sum() / max(focal[:, adjacent].sum(), 1)
        return (float(mask.float().mean()), float(co_rate), float(dropped.float().mean()))

    def test_adaptive_mask_radius_and_spatial_statistics_for_three_channel_groups(self):
        import torch
        from wyu_bci_bench.pretraining_masks import (
            SessionFMMaskGenerator, sessionfm_mask_statistics,
        )

        manifest_path = (Path(__file__).resolve().parents[2] / "challenge2026_track02" / "data"
                         / "sessionfm_corpus_k3_fixed_20261027" / "manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shards = {item["name"].split("_")[0]: item for item in manifest["shards"]
                  if item["name"] in {"Dreyer2023_000000", "EEGMAT_000000"}}
        from wyu_bci_bench.pretraining import resolve_channel_coordinates
        proteus_path = (Path(__file__).resolve().parents[2] / "challenge2026_track02" / "proteus_update1"
                        / "cache" / "proteus_windows_v1.npz")
        with np.load(proteus_path, allow_pickle=False) as cache:
            proteus_names = cache["channels"].astype(str).tolist()
        proteus_coords, known = resolve_channel_coordinates(proteus_names)
        groups = {
            "Dreyer2023_27": np.asarray(shards["Dreyer2023"]["coordinates_m"], np.float32),
            "EEGMAT": np.asarray(shards["EEGMAT"]["coordinates_m"], np.float32),
            "PROTEUS_41": np.asarray(proteus_coords[known], np.float32),
        }
        expected_channels = {"Dreyer2023_27": 27, "EEGMAT": 19, "PROTEUS_41": 41}
        for name, coords in groups.items():
            self.assertEqual(coords.shape, (expected_channels[name], 3))
        for index, (name, coordinates) in enumerate(groups.items()):
            with self.subTest(group=name):
                mask, dropped = SessionFMMaskGenerator(9200 + index).generate(coordinates, 1000)
                stats = sessionfm_mask_statistics(mask, dropped, coordinates)
                self.assertGreaterEqual(stats["mean_neighbor_count"], 2.0, (name, stats))
                self.assertGreater(stats["neighbor_comask_rate"],
                                   stats["independent_mask_baseline"] + 0.10, (name, stats))
                self.assertTrue(0.50 <= stats["mask_fraction"] <= 0.60, (name, stats))
                self.assertAlmostEqual(stats["radius_m"], 1.5 * stats["median_nearest_neighbor_m"], places=7)
                print(f"SessionFM adaptive mask {name}: {stats}")

    def test_masked_trivial_losses_use_zero_and_featurewise_medians(self):
        import torch
        from wyu_bci_bench.pretraining_masks import sessionfm_masked_trivial_losses

        wave = torch.ones(2, 2, 8, 3)
        spectral = torch.arange(2 * 2 * 4 * 2, dtype=torch.float32).reshape(2, 2, 4, 2)
        mask = torch.zeros(2, 2, 8, dtype=torch.bool)
        mask[:, :, 0:2] = True
        losses = sessionfm_masked_trivial_losses(wave, spectral, spectral + 1, mask)
        self.assertAlmostEqual(losses["wave"].item(), 1.0)
        self.assertGreater(losses["spectrum"].item(), 0.0)
        self.assertAlmostEqual(losses["spectrum"].item(), losses["global"].item())

    def test_mask_rate_spatial_correlation_and_channel_drop_fraction(self):
        import torch
        from wyu_bci_bench.pretraining_masks import sessionfm_spatiotemporal_mask

        coordinates = torch.tensor([[(i % 5) * 0.025, (i // 5) * 0.025, 0.0] for i in range(25)])
        mask, channel_dropped = sessionfm_spatiotemporal_mask(
            coordinates, batch_size=512, generator=torch.Generator().manual_seed(19)
        )
        mask_rate = mask.float().mean().item()
        self.assertTrue(0.50 <= mask_rate <= 0.60, mask_rate)
        self.assertAlmostEqual(channel_dropped.float().mean().item(), 0.10, delta=0.02)

        from wyu_bci_bench.pretraining_masks import sessionfm_mask_statistics
        stats = sessionfm_mask_statistics(mask, channel_dropped, coordinates)
        self.assertGreater(stats["neighbor_comask_rate"], stats["independent_mask_baseline"] + 0.10)


if __name__ == "__main__":
    unittest.main()
