from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from explorer.build_catalog import build_catalog


class CapabilityCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = build_catalog(ROOT)

    def test_every_capability_points_to_an_existing_symbol_and_known_tags(self):
        modules = {module["name"]: module for module in self.catalog["modules"]}
        capabilities = self.catalog["capabilities"]
        dimensions = capabilities["dimensions"]
        known = {dimension: {item["id"] for item in items} for dimension, items in dimensions.items()}

        for entry in capabilities["entries"]:
            with self.subTest(target=f"{entry['module']}.{entry['symbol']}"):
                module = modules[entry["module"]]
                self.assertIn(entry["symbol"], {symbol["name"] for symbol in module["symbols"]})
                self.assertIn(entry["claim_level"], capabilities["claim_levels"])
                for dimension, ids in known.items():
                    self.assertTrue(set(entry[dimension]) <= ids)

    def test_domain_shift_is_marked_as_intent_or_assessment(self):
        entries = {(e["module"], e["symbol"]): e for e in self.catalog["capabilities"]["entries"]}
        supcon = entries[("ssl", "cross_subject_supcon")]
        session_split = entries[("splits", "cross_session")]
        self.assertIn("domain_shift_mitigation", supcon["goal"])
        self.assertEqual(supcon["claim_level"], "design_intent")
        self.assertIn("domain_shift_assessment", session_split["goal"])
        self.assertEqual(session_split["claim_level"], "evaluation")

    def test_class_cards_have_chinese_explanations(self):
        modules = {module["name"]: module for module in self.catalog["modules"]}
        expected = {
            "NearestCentroidClassifier",
            "DiagonalGaussianNB",
            "KNearestNeighborsClassifier",
            "StandardizedDiagonalLDA",
            "FilterBankLDAModel",
        }
        symbols = {s["name"]: s for s in modules["models"]["symbols"]}
        for name in expected:
            with self.subTest(symbol=name):
                self.assertTrue(symbols[name]["summary_zh"])

    def test_braindecode_models_are_catalogued_by_architecture_family(self):
        modules = {module["name"]: module for module in self.catalog["modules"]}
        model_module = modules["deep_models"]
        self.assertIn("ShallowFBCSPNet", model_module["desc"])
        self.assertIn("EEGTCNet", model_module["desc"])
        self.assertIn("EEGInceptionMI", model_module["desc"])
        self.assertIn("EEG-ResNet", model_module["desc"])
        self.assertIn("InceptionTime", model_module["desc"])
        symbols = {symbol["name"]: symbol for symbol in model_module["symbols"]}
        for name in ("available_models", "make_braindecode_model", "fit_predict_braindecode"):
            with self.subTest(symbol=name):
                self.assertTrue(symbols[name]["summary_zh"])
        self.assertIn("ShallowFBCSPNet", symbols["available_models"]["summary_zh"])
        entry = next(
            e for e in self.catalog["capabilities"]["entries"]
            if e["module"] == "deep_models" and e["symbol"] == "make_braindecode_model"
        )
        labels = {item["id"] for item in self.catalog["capabilities"]["dimensions"]["method"]}
        self.assertTrue(set(entry["method"]) <= labels)
        self.assertEqual(len(entry["method"]), 8)

    def test_true_csp_and_masked_reconstruction_are_catalogued(self):
        modules = {module["name"]: module for module in self.catalog["modules"]}
        self.assertIn("MNE", modules["csp"]["requires"])
        self.assertTrue(any(s["name"] == "FilterBankCSP" for s in modules["csp"]["symbols"]))
        self.assertTrue(any(s["name"] == "masked_reconstruction_loss" for s in modules["ssl"]["symbols"]))

    def test_paradigm_and_adaptation_modules_are_catalogued(self):
        modules = {module["name"]: module for module in self.catalog["modules"]}
        self.assertIn("TRCA", modules["ssvep"]["desc"])
        self.assertIn("eTRCA", modules["ssvep"]["desc"])
        self.assertIn("TDCA", modules["ssvep"]["desc"])
        self.assertIn("xDAWN", modules["erp"]["desc"])
        self.assertIn("Euclidean Alignment", modules["adaptation"]["desc"])
        targets = {(entry["module"], entry["symbol"])
                   for entry in self.catalog["capabilities"]["entries"]}
        for target in (("ssvep", "SSVEPCCA"), ("ssvep", "FilterBankCCA"),
                       ("ssvep", "TRCA"), ("erp", "XDawnLDA"),
                       ("adaptation", "coral_align")):
            with self.subTest(target=target):
                self.assertIn(target, targets)

    def test_extended_ssvep_and_adaptation_methods_are_catalogued(self):
        targets = {(entry["module"], entry["symbol"])
                   for entry in self.catalog["capabilities"]["entries"]}
        for target in (("ssvep", "EnsembleTRCA"), ("ssvep", "TDCA"),
                       ("adaptation", "RiemannianAlignment"),
                       ("adaptation", "maximum_mean_discrepancy"),
                       ("adaptation", "DomainAdversarialHead"),
                       ("adaptation", "adapt_batch_norm")):
            with self.subTest(target=target):
                self.assertIn(target, targets)

    def test_interpretability_and_calibration_methods_are_catalogued(self):
        modules = {module["name"]: module for module in self.catalog["modules"]}
        self.assertIn("留出特征组置换", modules["interpretability"]["desc"])
        self.assertIn("ECE/Brier", modules["calibration"]["desc"])
        targets = {(entry["module"], entry["symbol"])
                   for entry in self.catalog["capabilities"]["entries"]}
        for target in (("interpretability", "grouped_permutation_importance"),
                       ("interpretability", "channel_ablation_scores"),
                       ("interpretability", "band_ablation_scores"),
                       ("calibration", "fit_temperature"),
                       ("calibration", "expected_calibration_error"),
                       ("calibration", "multiclass_brier_score"),
                       ("calibration", "risk_coverage_curve")):
            with self.subTest(target=target):
                self.assertIn(target, targets)


if __name__ == "__main__":
    unittest.main()
