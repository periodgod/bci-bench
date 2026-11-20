"""Protocol-specific diagnostics for class collapse and guarded comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def collapse_summary(y, pred, subjects, sessions, interfaces, n_classes: int = 3) -> dict[str, object]:
    """Summarize underrepresented predicted classes per cell and low recalls per subject.

    A cell is one subject x session x interface group. Subject-class recall pools
    that subject's windows across sessions and interfaces.
    """
    y, pred = np.asarray(y), np.asarray(pred)
    groups = [np.asarray(g).astype(str) for g in (subjects, sessions, interfaces)]
    if y.ndim != 1 or pred.shape != y.shape or any(g.shape != y.shape for g in groups):
        raise ValueError("y, pred, and group arrays must be aligned 1-D arrays")
    if not len(y) or n_classes < 2 or not np.isin(y, np.arange(n_classes)).all() \
            or not np.isin(pred, np.arange(n_classes)).all():
        raise ValueError("expected non-empty labels in [0, n_classes)")

    cells = []
    for key in sorted(set(zip(*(g.tolist() for g in groups)))):
        mask = np.ones(len(y), dtype=bool)
        for group, value in zip(groups, key):
            mask &= group == value
        proportions = np.bincount(pred[mask], minlength=n_classes) / mask.sum()
        recalls = [float(np.mean(pred[mask & (y == c)] == c)) if np.any(mask & (y == c)) else None
                   for c in range(n_classes)]
        cells.append({"group": list(key), "n": int(mask.sum()),
                      "predicted_proportions": proportions.tolist(),
                      "zero_classes": np.flatnonzero(proportions == 0).astype(int).tolist(),
                      "class_recalls": recalls})

    low_recall = []
    for person in sorted(set(groups[0].tolist())):
        person_mask = groups[0] == person
        for cls in range(n_classes):
            cls_mask = person_mask & (y == cls)
            recall = float(np.mean(pred[cls_mask] == cls)) if np.any(cls_mask) else None
            if recall is not None and recall < 0.20:
                low_recall.append({"subject": person, "class": cls, "recall": recall})
    return {
        "n_cells": len(cells),
        "min_predicted_class_proportion": float(min(min(c["predicted_proportions"]) for c in cells)),
        "cells_with_zero_predicted_class": sum(bool(c["zero_classes"]) for c in cells),
        "cells_with_class_proportion_below_10pct": sum(
            any(p < 0.10 for p in c["predicted_proportions"]) for c in cells),
        "subject_class_recalls_below_20pct": low_recall,
        "n_subject_class_recalls_below_20pct": len(low_recall),
        "cells": cells,
    }


def collapse_gate(summary: Mapping[str, object], *, max_low_cells: int = 4,
                  max_low_recalls: int = 1) -> dict[str, object]:
    """Apply the fixed PROTEUS collapse thresholds to ``collapse_summary`` output."""
    checks = {
        "no_zero_class_cells": summary["cells_with_zero_predicted_class"] == 0,
        "low_class_cells_within_limit": summary["cells_with_class_proportion_below_10pct"] <= max_low_cells,
        "low_subject_class_recalls_within_limit": summary["n_subject_class_recalls_below_20pct"] <= max_low_recalls,
    }
    return {"checks": checks, "pass": all(checks.values())}


def guard_comparison(candidate: Mapping[str, float], anchor: Mapping[str, float],
                     subjects: Sequence[str] = ("sub-P04", "sub-P10"),
                     max_drop: float = 0.03) -> dict[str, object]:
    """Check candidate cross-session scores against the same-protocol anchor scores."""
    if max_drop < 0 or not subjects:
        raise ValueError("max_drop must be non-negative and subjects non-empty")
    missing = [p for p in subjects if p not in candidate or p not in anchor]
    if missing:
        raise ValueError(f"missing candidate or anchor scores for {missing}")
    diffs = {p: float(candidate[p]) - float(anchor[p]) for p in subjects}
    if not np.isfinite(list(diffs.values())).all():
        raise ValueError("guard scores must be finite")
    return {"differences": diffs, "max_drop": float(max_drop),
            "pass": all(value >= -max_drop - 1e-12 for value in diffs.values())}
