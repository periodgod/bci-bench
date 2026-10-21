# wyu-bci-bench recipes

All snippets assume `from wyu_bci_bench import ...` in the `.venv` interpreter.

## 1. New PROTEUS experiment (thesis line)

```python
from pathlib import Path
from wyu_bci_bench.grouped import load_proteus_windows
from wyu_bci_bench.splits import loso
from wyu_bci_bench.metrics import summarize_predictions, paired_subject_bootstrap
from wyu_bci_bench.registry import save_run, load_oof

ROOT = Path("D:/脑电大模型/challenge2026_track02/proteus_update1")
data = load_proteus_windows(ROOT / "cache/proteus_windows_v1.npz")
probs = np.full((len(data.y), 3), np.nan, np.float32)
for split in loso(data.subjects):
    # fit everything on split.train only; predict split.test
    probs[split.test] = my_fit_predict(data.subset(split.train), data.subset(split.test))
save_run(ROOT / "results/<topic>_<yyyymmdd>", y=data.y, subjects=data.subjects, sessions=data.sessions,
         interfaces=data.interfaces, probs={"mine": probs},
         protocol={"split": "10-subject LOSO", "seeds": [33, 34, 35], "selection": "inner LOSO"},
         sources={"cache": ROOT / "cache/proteus_windows_v1.npz", "script": Path(__file__)})
```

Compare against an archived baseline on the same rows:

```python
base = load_oof(ROOT / "results/representation_learning_20261019/ssl/oof.npz")
paired_subject_bootstrap(summarize_predictions(data.y, probs, *data.meta), base.summary("math_prior"))
```

Reference numbers. Verified by `scripts/verify_proteus_port.py`; the comparison names are the OOF condition keys.

| Condition | cell BA |
|---|---:|
| M2 (`power_first_4bin_ensemble`) | 0.5460 |
| M1 | 0.5292 |
| math prior probe | 0.5113 |
| SimCLR | 0.4770 |
| VICReg | 0.4755 |
| random + BN | 0.4570 |
| random | 0.4204 |
| CBraMod pretrained | 0.4236 |
| CBraMod random ×3 | 0.4214 / 0.4132 / 0.4186 |

M2 − M1 sign-flip test: p = 0.0703.

## 2. Selecting C or epochs without leakage

```python
from wyu_bci_bench.splits import loso, inner_loso
from wyu_bci_bench.readout import nested_loso_readout
scores, folds, probs = nested_loso_readout({"cbramod": F_pre, "random": F_rand}, data.y, *data.meta)
```

For networks, run `inner_loso(data.subjects, outer.train)` inside each outer fold. Choose the epoch by inner cell BA, then retrain on `outer.train` with `nets.train_model(..., epochs=chosen)`.

## 3. M1 / M2

```python
from wyu_bci_bench.priors import math_prior_features
from wyu_bci_bench import nets
F = math_prior_features(data.X)                       # CPU, 1394-D
model = nets.make_model(F[tr], data.y[tr], "power_first_4bin")
model = nets.train_model(model, data.X[tr], F[tr], data.y[tr], data.subjects[tr], epochs=E, seed=33, device="cuda")
p = nets.predict_proba(model, data.X[te], F[te], "cuda")
```

## 4. SSL encoder + frozen probe

```python
import torch
from wyu_bci_bench import ssl, nets
from wyu_bci_bench.readout import probe_proba
nets.seed_all(seed + fold * 100)
model = ssl.RepresentationModel().to(dev)
raw = torch.as_tensor(data.X, device=dev)
ssl.train_ssl(model, raw[tr_t], data.subjects[tr], "vicreg", seed + fold * 100)
p = probe_proba(ssl.encode(model, raw[tr_t]), data.y[tr], ssl.encode(model, raw[te_t]))
```

## 5. Hierarchical time-series SSL (TS2Vec)

```python
import torch
from wyu_bci_bench import ssl

model = ssl.TS2VecModel(n_channels=data.X.shape[1]).to(dev)
raw = torch.as_tensor(data.X[tr], dtype=torch.float32, device=dev)
ssl.train_ts2vec(model, raw, data.subjects[tr], seed=seed, epochs=15)
train_features = model.pool(model(raw)).detach().cpu().numpy()
test_raw = torch.as_tensor(data.X[te], dtype=torch.float32, device=dev)
test_features = model.pool(model(test_raw)).detach().cpu().numpy()
p = probe_proba(train_features, data.y[tr], test_features)
```

This uses TS2Vec's hierarchical temporal contrastive objective with the bench's compact EEG temporal encoder. Fit it only on the outer training fold when reporting LOSO. The code implements the objective, not a bit-for-bit reproduction of the authors' full architecture or augmentation pipeline. See [representation learning papers](../../representation_learning_papers.md).

## 6. CBraMod frozen features

```python
from wyu_bci_bench.foundation.cbramod import extract_features
repo = "D:/脑电大模型/tmp_repos/cbramod"
F_pre = extract_features(data.X, data.fs, repo, f"{repo}/pretrained_weights/pretrained_weights.pth")
F_r16 = extract_features(data.X, data.fs, repo, None, seed=20261016)   # same-architecture random control
```

Always pair a pretrained condition with seeded random-initialisation controls of the same architecture.

## 6. Cross-session diagnostic (thesis title claim)

```python
from wyu_bci_bench.splits import cross_session
s = cross_session(data.subjects, data.sessions, "sub-P10", train_sessions=["ses-1"])
```

Report this as "first-session labelled calibration", not zero-calibration transfer. Only P04 and P10 currently have enough sessions, and both are exposed participants.

## 7. EEG/EMG Challenge 2026 Track 2 (NeuralBench / Codabench)

- Metric: `metrics.stream_mean_balanced_accuracy(y, pred, subjects, sessions)`.
- Local sealed-style split: `splits.neuralbench_sealed(subjects, sessions, evaluation_subjects)`.
- Package:

  ```python
  from wyu_bci_bench.submission import build_zip, check_zip
  m = build_zip("pkg_dir", "out/track02_<name>.zip")
  report = check_zip(m["zip"])
  assert report["ok"], report["problems"]
  ```

- The package must:
  - load weights from `meta['submission_dir']`
  - implement `reset_state()`
  - avoid per-session recalibration

## 8. OmniEEG Challenge 2026

- Rematch and final rounds: 14 datasets, 200 Hz, subject-isolated 8:1:1, three organiser seeds. Select checkpoints by validation BA; the test split is only for final scoring.
- Until the organiser publishes index lists, use `splits.multi_seed_subject_ratio(subjects, seeds)`, then replace them with `splits.predefined` once published.
- Rematch: public weights only (OmniEEG, LaBraM, CodeBrain, ...).
- Final: random initialisation only. Assert that no `load_state_dict` call is made from external weights.
- Per track, score = mean BA over the 2 datasets. Main track = sum of track points. Report per-dataset BA, PR-AUC, ROC-AUC and F1.

## 9. Not yet in the bench

Still in original scripts; port with tests when needed: `ContentStyleNet` / `GroupRiskLinear`, external-corpus concatenation, and the LaBraM adapter. EEGPT has an encoder-only adapter in `foundation.eegpt`; downstream heads and training loops remain in the original scripts.

## 10. Thesis statistics

`论文交付/audit_primary_experiments_bench.py` recomputes the thesis M1/M2 and CBraMod tables from OOF files with bench metrics and writes `十人主实验统一复核_20261020.json` + figure. Re-run it after any new OOF that the thesis cites.

## 11. SSVEP and ERP/P300 baselines

```python
from wyu_bci_bench import FilterBankCCA, SSVEPCCA, TRCA, XDawnLDA

# X is (trials, channels, samples); y is an integer class index for each stimulus frequency.
cca = SSVEPCCA(frequencies=[8.0, 9.0, 10.0], sampling_rate=250, harmonics=3)
predicted = cca.predict(X)
fbcca = FilterBankCCA(frequencies=[8.0, 9.0, 10.0], sampling_rate=250, harmonics=3)
predicted_fb = fbcca.predict(X)

# TRCA learns spatial filters/templates from training repetitions only.
trca = TRCA().fit(X_train, y_train)
predicted_trca = trca.predict(X_test)

# XDawnLDA expects epochs already cut around events and binary target labels.
erp = XDawnLDA(n_components=4, positive_label=1).fit(X_train, y_train)
target_probabilities = erp.predict_proba(X_test)
```

The CCA reference methods use the known stimulation frequencies and do not require training labels. TRCA and xDAWN+LDA must be fit inside the current training fold. See [model_catalog.md](../../model_catalog.md) for assumptions and source references.

## 12. Unsupervised domain alignment

```python
from wyu_bci_bench import EuclideanAlignment, coral_align

# Fit each EA matrix only from that domain's unlabeled trials.
aligned_epochs = EuclideanAlignment().fit_transform(X, domain_ids)

# CORAL maps labeled source features to the unlabeled target distribution.
aligned_source_features = coral_align(source_features, unlabeled_target_features)
```

If `domain_ids` or `unlabeled_target_features` include held-out evaluation inputs, the result is transductive/test-time adaptation. The protocol must explicitly permit this and must never use target labels.
