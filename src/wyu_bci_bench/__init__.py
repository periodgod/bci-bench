"""WY University BCI benchmark toolkit."""

from __future__ import annotations

from .data import EEGDataset, Recording, load_npz_recording, merge_recordings, scan_recordings
from .evaluate import cross_validate_eegnet, cross_validate_filterbank_lda
from .models import FilterBankLDAModel, StandardizedDiagonalLDA

__all__ = [
    "EEGDataset",
    "FilterBankLDAModel",
    "Recording",
    "StandardizedDiagonalLDA",
    "cross_validate_eegnet",
    "cross_validate_filterbank_lda",
    "load_npz_recording",
    "merge_recordings",
    "scan_recordings",
]

__version__ = "0.1.0"
