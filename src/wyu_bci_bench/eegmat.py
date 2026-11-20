"""EEGMAT paired baseline/arithmetic records for participant-isolated evaluation."""
from pathlib import Path
import numpy as np
from .external_event_data import read_edf_with_events, extract_windows_from_raw
from .grouped import GroupedEEGDataset, file_sha256

EEGMAT_CHANNELS = tuple('Fp1 Fp2 F3 F4 F7 F8 T3 T4 C3 C4 T5 T6 P3 P4 O1 O2 Fz Cz Pz'.split())


def load_eegmat_windows(root, *, seconds_per_condition=60):
    """_1 baseline, _2 arithmetic per bundled README; matched nonoverlapping4s windows.

    Retain only19 scalp channels. Exclude ECG and the A2-A1 reference difference.
    Both conditions share one participant/session key so group BA has both classes.
    Already published artifact-cleaned source; native bandwidth is not restored.
    """
    root = Path(root)
    subjects = sorted({p.stem.rsplit('_', 1)[0] for p in root.glob('Subject*_*.edf')})
    if len(subjects) < 5:
        raise ValueError('need at least5 paired EEGMAT subjects')
    if seconds_per_condition < 4 or seconds_per_condition % 4:
        raise ValueError('seconds_per_condition must be a positive multiple of4')
    arrays, labels, people, order, records = [], [], [], [], []
    starts = np.arange(0, seconds_per_condition, 4, dtype=float)
    for subject in subjects:
        for suffix, label in [('1', 0), ('2', 1)]:
            path = root/f'{subject}_{suffix}.edf'
            raw = read_edf_with_events(path)
            rename = {name: name.removeprefix('EEG ').strip() for name in raw.ch_names}
            raw.rename_channels(rename)
            raw.pick(list(EEGMAT_CHANNELS))
            raw.notch_filter([50, 60], verbose='ERROR')
            raw.resample(256, verbose='ERROR')
            raw.filter(.1, 96, verbose='ERROR')
            x = extract_windows_from_raw(raw, starts, channel_names=EEGMAT_CHANNELS,
                duration_seconds=4, target_fs=256, car=False)
            arrays.append(x)
            labels.extend([label]*len(x))
            people.extend([subject]*len(x))
            order.extend(np.arange(len(x))+label*len(x))
            records.append(dict(path=str(path), sha256=file_sha256(path), condition=label,
                                start_seconds=starts.tolist(), trials=len(x)))
            raw.close()
    y = np.asarray(labels, np.int64)
    return GroupedEEGDataset(np.concatenate(arrays), y, np.asarray(people),
        np.full(len(y), 'paired_baseline_arithmetic'), np.full(len(y), 'EEGMAT'),
        fs=256, channel_names=list(EEGMAT_CHANNELS), class_names=['baseline', 'arithmetic'],
        recording_order=np.asarray(order), source=str(root),
        metadata={'records': records, 'label_source': str(root/'README.txt'),
            'label_source_sha256': file_sha256(root/'README.txt'),
            'preprocessing': '19 scalp channels; notch50/60; resample256; FIR0.1-96; first60s each condition; nonoverlap4s; uV; no CAR yet',
            'limitation': 'Baseline precedes arithmetic; recording/condition order and source preprocessing are confounds. Adjacent windows are not independent subjects.'})
