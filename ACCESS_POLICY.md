# Access Policy

`wyu-bci-bench` is private/internal project infrastructure.

Do not commit or upload:

- Raw EEG/EMG/EOG recordings.
- Processed datasets or exported trial arrays.
- Model weights, checkpoints, joblib bundles, or ONNX files.
- Experiment outputs, reports, plots, logs, or notebook checkpoints.
- Participant identifiers or consent-related files.

Allowed in this repository:

- Source code.
- Tests using synthetic data only.
- Documentation.
- Example scripts that reference local paths but do not include data.

Before pushing, run:

```powershell
git status --short
git ls-files
```

If any file under `data/`, `recordings/`, `outputs/`, `models/`, or any large EEG/data/model extension appears in the staged set, unstage it before publishing.
