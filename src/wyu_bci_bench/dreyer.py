"""Audited Dreyer2023 acquisition-run labels from the EDF's embedded events."""
from pathlib import Path
import numpy as np
from .external_event_data import read_edf_with_events, extract_annotated_windows
from .grouped import GroupedEEGDataset, file_sha256

# Excludes EOG1/2/3 and EMGg/EMGd, which these EDFs misleadingly mark EEG.
DREYER_EEG_CHANNELS = tuple(('Fz FCz Cz CPz Pz C1 C3 C5 C2 C4 C6 F4 FC2 FC4 '
    'FC6 CP2 CP4 CP6 P4 F3 FC1 FC3 FC5 CP1 CP3 CP5 P3').split())
DREYER_EVENTS = {'769': 0, '770': 1}


def load_acquisition_windows(root, subjects, *, runs=('R2acquisition',), duration=4.0):
    """Cue-relative [0,4) seconds, 256 Hz; recording-wise fixed filters only.

    769=left and 770=right match the MOABB Dreyer2023 loader. Acquisition runs
    avoid online feedback as a pilot confound. No artifact-based label selection.
    """
    arrays, labels, people, sessions, orders, provenance = [], [], [], [], [], []
    for subject in subjects:
        for run in runs:
            path = Path(root) / subject / 'eeg' / f'{subject}_task-{run}_eeg.edf'
            raw = read_edf_with_events(path).pick(list(DREYER_EEG_CHANNELS))
            raw.notch_filter([50, 60], verbose='ERROR')
            raw.resample(256, verbose='ERROR')
            raw.filter(0.1, 96, verbose='ERROR')
            x, y, onsets = extract_annotated_windows(raw, DREYER_EVENTS,
                channel_names=DREYER_EEG_CHANNELS, offset_seconds=0,
                duration_seconds=duration, target_fs=256, car=False)
            arrays.append(x)
            labels.append(y)
            people.extend([subject] * len(y))
            sessions.extend([run] * len(y))
            orders.extend(range(len(y)))
            provenance.append(dict(path=str(path), sha256=file_sha256(path),
                                   trials=len(y), counts=np.bincount(y, minlength=2).tolist(),
                                   cue_onsets_seconds=onsets.tolist()))
    x, y = np.concatenate(arrays), np.concatenate(labels)
    return GroupedEEGDataset(x, y, np.asarray(people), np.asarray(sessions),
        np.full(len(y), 'Dreyer2023'), fs=256, channel_names=list(DREYER_EEG_CHANNELS),
        class_names=['left_hand', 'right_hand'], source=str(root),
        recording_order=np.asarray(orders), metadata={'records': provenance,
            'preprocessing': '27 scalp EEG only; notch50/60; resample256; FIR0.1-96; cue[0,4)s; uV; no CAR yet',
            'event_map': DREYER_EVENTS, 'positions': 'caller must supply geometry'})
