"""WY University BCI benchmark toolkit."""

from __future__ import annotations

from .continuous import ContinuousPreprocessConfig, compute_channel_qc, load_continuous_dataset
from .data import EEGDataset, Recording, load_npz_recording, merge_recordings, scan_recordings
from .evaluate import (
    cross_validate_eegnet,
    cross_validate_filterbank_lda,
    cross_validate_riemannian_models,
    cross_validate_simple_models,
)
from .models import (
    DiagonalGaussianNB,
    FilterBankLDAModel,
    KNearestNeighborsClassifier,
    NearestCentroidClassifier,
    StandardizedDiagonalLDA,
)

__all__ = [
    "ContinuousPreprocessConfig",
    "DiagonalGaussianNB",
    "EEGDataset",
    "FilterBankLDAModel",
    "KNearestNeighborsClassifier",
    "NearestCentroidClassifier",
    "Recording",
    "StandardizedDiagonalLDA",
    "compute_channel_qc",
    "cross_validate_eegnet",
    "cross_validate_filterbank_lda",
    "cross_validate_riemannian_models",
    "cross_validate_simple_models",
    "load_continuous_dataset",
    "load_npz_recording",
    "merge_recordings",
    "scan_recordings",
]

__version__ = "0.1.0"
