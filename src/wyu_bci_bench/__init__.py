"""WY University BCI benchmark toolkit."""

from __future__ import annotations

from .continuous import ContinuousPreprocessConfig, compute_channel_qc, load_continuous_dataset
from .data import EEGDataset, Recording, load_npz_recording, merge_recordings, scan_recordings
from .grouped import GroupedEEGDataset, load_grouped_npz, load_proteus_windows
from .metrics import (
    cell_balanced_accuracy,
    exact_sign_flip_pvalue,
    paired_subject_bootstrap,
    stream_mean_balanced_accuracy,
    summarize_predictions,
)
from .registry import load_oof, save_run
from .splits import cross_session, inner_loso, loso, neuralbench_sealed, subject_ratio
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
from .adaptation import (
    DomainAdversarialHead,
    EuclideanAlignment,
    RiemannianAlignment,
    adapt_batch_norm,
    coral_align,
    domain_adversarial_loss,
    euclidean_align,
    maximum_mean_discrepancy,
)
from .erp import XDawnLDA
from .ssvep import EnsembleTRCA, FilterBankCCA, SSVEPCCA, TDCA, TRCA
from .csp import FilterBankCSP
from .interpretability import band_ablation_scores, channel_ablation_scores, grouped_permutation_importance
from .calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    multiclass_brier_score,
    risk_coverage_curve,
)
from .online import (CausalSOSFilter, StreamWindowBuffer, benchmark_streaming,
                     design_causal_bandpass, temporal_drift_summary)
from .foundation.eegpt import EEGPT_CHANNELS
from .foundation.eegpt import extract_features as extract_eegpt_features
from .foundation.eegpt import load_encoder as load_eegpt_encoder
from .foundation.eegpt import prepare_input as eegpt_prepare_input

__all__ = [
    "GroupedEEGDataset",
    "cell_balanced_accuracy",
    "cross_session",
    "exact_sign_flip_pvalue",
    "inner_loso",
    "load_grouped_npz",
    "load_oof",
    "load_proteus_windows",
    "loso",
    "neuralbench_sealed",
    "paired_subject_bootstrap",
    "save_run",
    "stream_mean_balanced_accuracy",
    "subject_ratio",
    "summarize_predictions",
    "ContinuousPreprocessConfig",
    "DiagonalGaussianNB",
    "EEGDataset",
    "FilterBankLDAModel",
    "FilterBankCCA",
    "FilterBankCSP",
    "band_ablation_scores",
    "channel_ablation_scores",
    "grouped_permutation_importance",
    "apply_temperature",
    "expected_calibration_error",
    "fit_temperature",
    "multiclass_brier_score",
    "risk_coverage_curve",
    "KNearestNeighborsClassifier",
    "NearestCentroidClassifier",
    "EuclideanAlignment",
    "RiemannianAlignment",
    "DomainAdversarialHead",
    "EnsembleTRCA",
    "Recording",
    "StandardizedDiagonalLDA",
    "SSVEPCCA",
    "TRCA",
    "TDCA",
    "XDawnLDA",
    "coral_align",
    "adapt_batch_norm",
    "compute_channel_qc",
    "cross_validate_eegnet",
    "cross_validate_filterbank_lda",
    "cross_validate_riemannian_models",
    "cross_validate_simple_models",
    "euclidean_align",
    "maximum_mean_discrepancy",
    "domain_adversarial_loss",
    "load_continuous_dataset",
    "CausalSOSFilter",
    "StreamWindowBuffer",
    "benchmark_streaming",
    "design_causal_bandpass",
    "temporal_drift_summary",
    "load_npz_recording",
    "merge_recordings",
    "scan_recordings",
    "EEGPT_CHANNELS",
    "eegpt_prepare_input",
    "load_eegpt_encoder",
    "extract_eegpt_features",
]

__version__ = "0.2.0"
