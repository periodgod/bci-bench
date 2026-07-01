"""Analyze local arm_bci_robot recordings through wyu_bci_bench."""

from __future__ import annotations

from pathlib import Path

from wyu_bci_bench.data import load_dataset
from wyu_bci_bench.evaluate import cross_validate_filterbank_lda, dataset_quality_report


ROOT = Path(r"E:\世界机器人大赛")
RECORDINGS = ROOT / "projects" / "arm_bci_robot" / "data" / "recordings"


def main() -> None:
    dataset = load_dataset(RECORDINGS, label_mode="all", channel_mode="motor11")
    print("shape:", dataset.shape)
    print("labels:", dataset.label_names)
    print("channels:", dataset.channel_names)

    quality = dataset_quality_report(
        dataset.eeg,
        dataset.y,
        dataset.fs,
        dataset.channel_names,
        dataset.label_names,
    )
    print("amplitude:", quality["amplitude"])

    result = cross_validate_filterbank_lda(
        dataset.eeg,
        dataset.y,
        dataset.fs,
        dataset.channel_names,
        dataset.label_names,
        folds=5,
    )
    print("balanced_accuracy:", result["balanced_accuracy"])
    print("confusion_matrix:", result["confusion_matrix"])


if __name__ == "__main__":
    main()
