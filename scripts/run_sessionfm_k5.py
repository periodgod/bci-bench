"""K5 preregistration and preflight; this preparation build never evaluates PROTEUS.

Preregistered arms: a=SCC-M2 (reuse its existing OOF); b=SessionFM K4 main;
c=random-init SessionFM; d=stateless SessionFM; e1-e4=K4 A1-A4 checkpoints;
e5=K4 main checkpoint with prior anchoring removed. Seeds are paired
20261101->33, 20261102->34, 20261103->35; report the mean of the three
probability matrices. G1 (b vs c): cell BA gain >=1.0 pp, participant-cluster
bootstrap 95% lower bound >0, and >=7/10 participants improve. G2 (b vs a):
the same gain/interval/person rules and worst-participant decline <=3 pp.
The fixed inner validation participant for outer fold k is ordered
SUBJECTS[(k+1)%10]. Stage 1 is frozen-backbone LP-FT for 10 epochs at 4e-4.
Stage 2 selects epochs (12..45, patience 8) on that participant only.

Execution is deliberately disabled in this preparation build. Preflight first
requires all three main seeds and A1-A4 with matching SHA-256 in K4 summary;
an absent arm is a hard error before any dataset can be opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED_K4 = ("main_20261101", "main_20261102", "main_20261103",
               "A1", "A2", "A3", "A4")


def verify_k4_summary(summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = {item["name"]: item for item in summary.get("runs", [])}
    missing = [name for name in REQUIRED_K4 if name not in runs]
    if missing:
        raise RuntimeError(f"K4 summary missing required arms: {', '.join(missing)}")
    root = summary_path.parent
    for name in REQUIRED_K4:
        run = runs[name]
        expected = run.get("weights_sha256")
        checkpoint = root / name / "weights.pt"
        if not expected or not checkpoint.is_file():
            raise RuntimeError(f"K4 checkpoint or SHA-256 missing for {name}")
        actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"K4 checkpoint SHA-256 mismatch for {name}")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k4-summary", type=Path, required=True)
    args = parser.parse_args(argv)
    verify_k4_summary(args.k4_summary)
    raise RuntimeError("K5 evaluation is disabled in this preparation build; no data was opened")


if __name__ == "__main__":
    raise SystemExit(main())
