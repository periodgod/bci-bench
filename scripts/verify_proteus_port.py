"""Check that the bench reproduces the archived PROTEUS results and original code exactly.

    python scripts/verify_proteus_port.py --workspace D:/脑电大模型

1. Recompute headline cell BA / bootstrap intervals from archived OOF files with
   ``wyu_bci_bench.metrics`` and compare with the numbers in the reports.
2. Import the original modules and compare math-prior features, M1/M2 seeded
   initialisation and forward outputs, and ``cell_score`` with the bench ports.
Exit code 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

from wyu_bci_bench.grouped import load_proteus_windows
from wyu_bci_bench.metrics import (cell_balanced_accuracy, exact_sign_flip_pvalue, grouped_balanced_accuracy,
                                   paired_subject_bootstrap)
from wyu_bci_bench.registry import load_oof

# (oof file relative to proteus_update1/results, condition, expected cell BA, source)
EXPECTED = [
    ("representation_learning_20261019/ssl/oof.npz", "random", 0.4204, "复现报告 §3"),
    ("representation_learning_20261019/ssl/oof.npz", "simclr", 0.4770, "复现报告 §3"),
    ("representation_learning_20261019/ssl/oof.npz", "vicreg", 0.4755, "复现报告 §3"),
    ("representation_learning_20261019/ssl/oof.npz", "math_prior", 0.5113, "复现报告 §3"),
    ("representation_learning_20261019/ssl/bn_control_oof.npz", "random_bn_calibrated", 0.4570, "复现报告 §3"),
    ("representation_learning_20261019/cbramod_native/oof.npz", "pretrained", 0.4236, "复现报告 §4"),
    ("representation_learning_20261019/cbramod_native/oof.npz", "random_20261016", 0.4214, "复现报告 §4"),
    ("representation_learning_20261019/cbramod_native/oof.npz", "random_20261017", 0.4132, "复现报告 §4"),
    ("representation_learning_20261019/cbramod_native/oof.npz", "random_20261018", 0.4186, "复现报告 §4"),
    ("power_first_4bin_multiseed_confirmation_20261015_oof.npz", "power_first_4bin_ensemble", 0.5460, "论文 4.1 M2"),
    ("power_first_4bin_multiseed_confirmation_20261015_oof.npz", "current_full_head_2bin_ensemble", 0.5292, "论文 4.1 M1"),
]
# (left, right, expected difference, expected 95% CI)
CONTRASTS = [
    ("simclr", "random", 0.0566, (0.0224, 0.1140)),
    ("vicreg", "random", 0.0551, (0.0246, 0.0942)),
    ("simclr", "random_bn_calibrated", 0.0201, (-0.0044, 0.0548)),
    ("vicreg", "simclr", -0.0015, (-0.0313, 0.0218)),
    ("simclr", "math_prior", -0.0343, (-0.0831, 0.0185)),
]
TOL = 6e-5  # reports print 4 decimals


class Checker:
    def __init__(self):
        self.rows, self.failed = [], 0

    def check(self, name, ok, detail=""):
        self.rows.append((name, bool(ok), detail))
        self.failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def check_oof_numbers(c: Checker, results: Path) -> None:
    cache = {}
    for rel, name, expected, src in EXPECTED:
        path = results / rel
        if not path.exists():
            c.check(f"{name}", False, f"missing {path}")
            continue
        oof = cache.setdefault(rel, load_oof(path))
        got = oof.summary(name)["cell_ba"]
        c.check(f"cell BA {name}", abs(got - expected) < TOL, f"got {got:.5f} expected {expected:.4f} ({src})")
    ssl = cache["representation_learning_20261019/ssl/oof.npz"]
    bn = cache["representation_learning_20261019/ssl/bn_control_oof.npz"]
    summaries = {k: ssl.summary(k) for k in ("random", "simclr", "vicreg", "math_prior")}
    summaries["random_bn_calibrated"] = bn.summary("random_bn_calibrated")
    for left, right, d, (lo, hi) in CONTRASTS:
        r = paired_subject_bootstrap(summaries[left], summaries[right])
        ci = r["cell_cluster_bootstrap_95ci"]
        ok = abs(r["cell_difference"] - d) < TOL and abs(ci[0] - lo) < TOL and abs(ci[1] - hi) < TOL
        c.check(f"bootstrap {left}-{right}", ok, f"{r['cell_difference']:+.4f} [{ci[0]:.4f},{ci[1]:.4f}]")
    m2 = cache["power_first_4bin_multiseed_confirmation_20261015_oof.npz"]
    s2, s1 = m2.summary("power_first_4bin_ensemble"), m2.summary("current_full_head_2bin_ensemble")
    diffs = [s2["subjects"][p] - s1["subjects"][p] for p in sorted(s2["subjects"])]
    p = exact_sign_flip_pvalue(diffs)
    c.check("sign-flip M2-M1", abs(p - 0.0703) < 5e-4, f"p={p:.4f} expected 0.0703; improved {sum(d > 0 for d in diffs)}/10")


def check_code_equivalence(c: Checker, proteus: Path, X: np.ndarray, data) -> None:
    import torch

    sys.path.insert(0, str(proteus))
    orig_dfm = importlib.import_module("differentiable_feature_module_loso")
    orig_cov = importlib.import_module("math_prior_multiscale_covnet")
    orig_sea = importlib.import_module("spatial_energy_ablation_loso")
    from wyu_bci_bench import nets, priors

    sub = X[:256]
    a = orig_dfm.get_scaler_features(sub, "cpu")
    b = priors.math_prior_features(sub, device="cpu")
    c.check("math prior features", a.shape == b.shape == (256, 1394) and np.array_equal(a, b),
            f"shape {b.shape} max|diff|={np.abs(a - b).max():.2e}")
    y = data.y
    pred = (y + (np.arange(len(y)) % 4 == 0)) % 3
    o = orig_dfm.cell_score(y, pred, *data.meta)[0]
    n = cell_balanced_accuracy(y, pred, *data.meta)
    c.check("cell_score equivalence", o == n, f"{o:.6f} vs {n:.6f}")
    feats = b
    for variant in nets.VARIANTS:
        orig_dfm.torch.manual_seed(7)
        torch.manual_seed(7)
        m_old = orig_sea.make_model(feats, y[:256], variant)
        torch.manual_seed(7)
        m_new = nets.make_model(feats, y[:256], variant)
        same_state = all(torch.equal(m_old.state_dict()[k], m_new.state_dict()[k]) for k in m_old.state_dict())
        same_keys = list(m_old.state_dict()) == list(m_new.state_dict())
        m_old.eval(), m_new.eval()
        with torch.inference_mode():
            xr, xf = torch.from_numpy(sub[:32]), torch.from_numpy(feats[:32])
            same_out = torch.allclose(m_old(xr, xf), m_new(xr, xf), atol=1e-6)
        c.check(f"net {variant}", same_keys and same_state and same_out,
                f"keys={same_keys} init={same_state} forward={same_out}")

    from wyu_bci_bench import window_features
    orig_tf = importlib.import_module("temporal_features")
    for variant in ("whole", "pre_post_car", "pre_post_scale_invariant_clipped", "post_no_high_car"):
        a, b = orig_tf.extract_variant(sub, variant), window_features.extract_variant(sub, variant)
        c.check(f"window features {variant}", a.shape == b.shape and np.array_equal(a, b), f"shape {b.shape}")

    from wyu_bci_bench.ssl import cross_subject_supcon
    orig_sc = importlib.import_module("cross_subject_supcon_loso")
    g = torch.Generator().manual_seed(3)
    emb = torch.nn.functional.normalize(torch.randn(64, 16, generator=g), dim=1)
    lab, sj = torch.randint(0, 3, (64,), generator=g), torch.randint(0, 5, (64,), generator=g)
    a = orig_sc.cross_subject_supcon(emb, lab, sj, 0.1)
    b = cross_subject_supcon(emb, lab, sj, 0.1)
    c.check("cross-subject SupCon", torch.equal(a, b), f"{float(a):.6f} vs {float(b):.6f}")


def check_bids_extraction(c: Checker, proteus: Path, data, subject: str = "sub-P01") -> None:
    """Re-extract one participant from raw EDF and compare with the cache rows (slow, ~1 min)."""
    from wyu_bci_bench.proteus_bids import extract_windows

    raw_root = proteus / "data" / "Dreyer2026Proteus" / "download" / "nm000290"
    if not raw_root.exists():
        c.check("BIDS extraction", False, f"missing {raw_root}")
        return
    new = extract_windows(raw_root, subjects=[subject])
    old = data.subset(data.subjects == subject)
    same_meta = all(np.array_equal(getattr(new, k), getattr(old, k)) for k in ("y", "subjects", "sessions", "interfaces"))
    diff = float(np.abs(new.X - old.X).max()) if new.X.shape == old.X.shape else float("inf")
    c.check(f"BIDS extraction {subject}", same_meta and diff < 1e-3 and new.channel_names == old.channel_names,
            f"{new.X.shape} vs {old.X.shape}, max|diff|={diff:.2e} uV")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, default=Path("D:/脑电大模型"))
    ap.add_argument("--skip-code", action="store_true", help="only recompute numbers from OOF files")
    ap.add_argument("--with-bids", action="store_true", help="also re-extract sub-P01 from raw EDF")
    args = ap.parse_args()
    proteus = args.workspace / "challenge2026_track02" / "proteus_update1"
    c = Checker()
    data = load_proteus_windows(proteus / "cache" / "proteus_windows_v1.npz")
    c.check("PROTEUS cache", data.X.shape == (2655, 41, 480) and len(data.people) == 10,
            json.dumps({k: data.summary()[k] for k in ("shape", "cells")}))
    check_oof_numbers(c, proteus / "results")
    if not args.skip_code:
        check_code_equivalence(c, proteus, data.X, data)
    if args.with_bids:
        check_bids_extraction(c, proteus, data)
    print(f"\n{len(c.rows) - c.failed}/{len(c.rows)} checks passed")
    out = Path(__file__).resolve().parents[1] / "outputs" / "last_verify.json"
    out.parent.mkdir(exist_ok=True)
    import time
    out.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "passed": len(c.rows) - c.failed,
                               "total": len(c.rows), "with_bids": args.with_bids,
                               "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in c.rows]},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if c.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
