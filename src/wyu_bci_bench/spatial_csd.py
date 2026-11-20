"""MNE spherical-spline current source density with a fixed montage contract."""
import numpy as np


def csd_matrix(channel_names, *, montage="standard_1020", sphere="auto", spline_order=4,
               lambda2=1e-5, n_legendre_terms=50):
    """Build the label-independent linear map once, preserving channel order.

    Coordinates are template coordinates, not participant-digitized positions.
    Applying to uV gives uV/m²; no CAR or duplicate-channel substitution occurs.
    """
    import mne
    names = list(channel_names)
    if len(names) < 4 or len(set(names)) != len(names):
        raise ValueError("need at least four distinct named electrodes")
    info = mne.create_info(names, sfreq=120, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage(montage), on_missing="raise")
    identity = mne.EvokedArray(np.eye(len(names)), info, tmin=0)
    transformed = mne.preprocessing.compute_current_source_density(identity, sphere=sphere, lambda2=lambda2,
        stiffness=spline_order, n_legendre_terms=n_legendre_terms, copy=True)
    positions = {name: info["chs"][i]["loc"][:3].tolist() for i, name in enumerate(names)}
    return transformed.data.copy(), {"channel_names": names, "montage": montage, "sphere": sphere,
        "spline_order": spline_order, "lambda2": lambda2, "n_legendre_terms": n_legendre_terms,
        "positions_m": positions, "units": "uV/m^2 for uV input"}


def apply_csd(eeg, matrix):
    x, a = np.asarray(eeg), np.asarray(matrix)
    if x.ndim != 3 or a.shape != (x.shape[1], x.shape[1]) or not np.isfinite(x).all() or not np.isfinite(a).all():
        raise ValueError("finite N,C,T input and matching C,C map required")
    return np.einsum("cd,ndt->nct", a, x).astype(np.float32)
