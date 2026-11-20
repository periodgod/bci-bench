"""Descriptive diagnostics for aligned out-of-fold probability ensembles."""
from __future__ import annotations

import numpy as np
from .metrics import summarize_predictions


def stream_positions(keys):
    """Zero-based position in each stream, preserving the supplied row order."""
    counts, positions = {}, []
    for key in np.asarray(keys).astype(str):
        positions.append(counts.get(key, 0))
        counts[key] = counts.get(key, 0) + 1
    return np.asarray(positions, dtype=np.int64)


def ensemble_diagnostics(y, baseline, branches, subjects, sessions, interfaces, keys):
    y = np.asarray(y)
    pred = np.asarray(baseline).argmax(1)
    correct = pred == y
    out = {"confusion": np.zeros((3, 3), dtype=int).tolist(), "branches": {}, "stages": {}}
    confusion = np.zeros((3, 3), dtype=int)
    np.add.at(confusion, (y, pred), 1)
    out["confusion"] = confusion.tolist()
    for name, p in branches.items():
        hit = np.asarray(p).argmax(1) == y
        out["branches"][name] = {
            "rescued": int((hit & ~correct).sum()), "harmed": int((~hit & correct).sum()),
            "rescue_rate_given_baseline_wrong": float(hit[~correct].mean()) if (~correct).any() else None,
            "harm_rate_given_baseline_correct": float((~hit[correct]).mean()) if correct.any() else None,
            "agreement": float((np.asarray(p).argmax(1) == pred).mean()),
        }
    names = list(branches)
    out["correct_intersection"] = {
        a: {b: int(((np.asarray(branches[a]).argmax(1) == y) &
                    (np.asarray(branches[b]).argmax(1) == y)).sum()) for b in names} for a in names}
    pos = stream_positions(keys)
    for name, mask in (("1-8", pos < 8), ("9-32", (pos >= 8) & (pos < 32)), (">32", pos >= 32)):
        # Sparse early cells may lack classes: expose counts and pooled accuracy as well.
        out["stages"][name] = {"n": int(mask.sum()), "class_counts": np.bincount(y[mask], minlength=3).tolist(),
                               "accuracy": float(correct[mask].mean()),
                               **summarize_predictions(y[mask], pred[mask], np.asarray(subjects)[mask],
                                                       np.asarray(sessions)[mask], np.asarray(interfaces)[mask])}
    return out
