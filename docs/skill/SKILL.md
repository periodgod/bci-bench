---
name: wyu-bci-bench
description: Mandatory entry point for any EEG/BCI model, data-loading, split, metric, statistics, foundation-model or submission code in the 脑电大模型 workspace (graduation thesis on PROTEUS, EEG/EMG Challenge 2026 Track 2, OmniEEG Challenge 2026). Use before writing or editing such code so it reuses the wyu-bci-bench library instead of re-implementing loaders, LOSO splits, cell balanced accuracy, bootstrap, math-prior features, M1/M2 nets, SSL, CBraMod adapters or ZIP packaging.
metadata:
  version: '1.0'
  updated: '2026-10-21'
  repo: https://github.com/wyubci/wyu-bci-bench
  local: D:\脑电大模型\wyu-bci-bench (branch feat/eeg-foundation-protocols)
---

# wyu-bci-bench

The workspace once held 327 scripts with no shared library; `cell_score` alone was copied into 81 files and `subject_split` into 6. The bench now owns these pieces. New code imports them; it does not copy them.

## Hard rules

1. Before writing a loader, split, metric, bootstrap/sign-flip, feature extractor, network, SSL loss, foundation adapter, OOF writer or ZIP packager, check the module map below. If it exists, import it.
2. If something reusable is missing, add it to `wyu-bci-bench/src/wyu_bci_bench/` with a test in `tests/`, then import it. One-off experiment scripts only hold the experiment-specific part (see `examples/proteus_loso_template.py`).
3. Do not edit historical scripts under `challenge2026_track02/proteus_update1/` or `pilot_labram/`. Their sha256 hashes are recorded in report.json files. Port and verify instead.
4. Save every result with `registry.save_run` (oof.npz + report.json + sha256). Never report a number that has no OOF file behind it.
5. After touching bench code, run both checks and report the counts:
   - `python -m unittest discover -s tests` (42 tests)
   - `python scripts/verify_proteus_port.py --workspace D:/脑电大模型 --with-bids` (29 checks)

## Workspace workflow

- Before starting work in this workspace, read `交接文档.md` sections 1 and 3, the latest three work-log entries, and section 4 before changing any recorded decision. Use the snapshot, open tasks, and blockers to avoid repeating completed work.
- When work changes workspace files, runs an experiment, or establishes a new finding, update `交接文档.md` before finishing: add a dated entry at the top of section 5 using the section 6 template, update the section 1 snapshot if status or key numbers changed, maintain section 3 tasks, record consequential decisions in section 4, update the last-updated line, and run `node tools/handoff_check.cjs`.
- Keep handoff notes factual: record paths, commands, counts, failures, and unverified items; never include credentials or tokens. A read-only Q&A with no new finding needs no log entry.
- Get the user's approval before deleting files, pushing Git, submitting a competition entry, or changing global configuration. Mark pending decisions as `等确认` in section 3. For file moves, maintain a reversible manifest and fix affected links.
- Historical scripts under `challenge2026_track02/proteus_update1/` and `pilot_labram/` are immutable; port reusable work into the bench instead. Use `challenge2026_track02\.venv\Scripts\python.exe` for workspace Python tasks.
- Treat results from the ten public PROTEUS participants as development evidence, not sealed-test evidence, and label every result with its protocol and evidence level.

## Environment

- Python: `D:\脑电大模型\challenge2026_track02\.venv\Scripts\python.exe` (3.12, torch 2.14 cu126, sklearn 1.9, numpy 2.5). The bench is installed editable there (`pip install -e ./wyu-bci-bench --no-deps`).
- The system Python312 has torch 2.6 and sklearn. Install with `-e .[ml]` if you use it.
- PROTEUS cache: `challenge2026_track02/proteus_update1/cache/proteus_windows_v1.npz`
  - `X` (2655,41,480) µV @120 Hz
  - `y` 0/1/2
  - `subjects` / `sessions` / `interfaces`
  - 10 people, 28 cells

## Module map

| Need | Use |
|---|---|
| Load windows with groups | `grouped.load_proteus_windows`, `grouped.load_grouped_npz`, `GroupedEEGDataset.meta` |
| Rebuild cache from raw BIDS | `proteus_bids.extract_windows`, `proteus_bids.save_cache` |
| Welch feature variants | `window_features.extract_variant(X, "pre_post_car")` |
| LOSO / inner LOSO | `splits.loso`, `splits.inner_loso` |
| Cross-session (ses-1 → later) | `splits.cross_session` |
| NeuralBench sealed sessions 4–6 | `splits.neuralbench_sealed` |
| OmniEEG subject 8:1:1 × 3 seeds | `splits.subject_ratio`, `splits.multi_seed_subject_ratio` |
| PROTEUS headline metric | `metrics.cell_balanced_accuracy` (participant×session×interface, equal weight) |
| NeuralBench metric | `metrics.stream_mean_balanced_accuracy` (subject×session, `absent="skip"`) |
| Full summary | `metrics.summarize_predictions` → cell_ba, subject_ba, per-subject, recalls |
| Paired comparison | `metrics.paired_subject_bootstrap` (seed 20261019, 20000 draws), `metrics.exact_sign_flip_pvalue` |
| Math prior 1394-D | `priors.math_prior_features` (CPU default; CUDA moves 0.5113 → 0.5109) |
| M1 / M2 | `nets.make_model(features, y, "current_full_head_2bin" or "power_first_4bin")`, `nets.train_model`, `nets.predict_proba` |
| SimCLR / VICReg / SupCon / TS2Vec | `ssl.RepresentationModel`, `ssl.TS2VecModel`, `ssl.train_ssl`, `ssl.train_ts2vec`, `ssl.hierarchical_contrastive_loss`, `ssl.encode`, `ssl.cross_subject_supcon` |
| Average random-seed controls | `metrics.mean_summaries` (mean of scores, not a probability ensemble) |
| Foundation model / fine-tuning | `foundation.require_verified(name)`, `foundation.cbramod.extract_features`, `foundation.eegpt.extract_features`, `foundation.finetuning.configure_trainable_parameters`, `attach_peft_adapter` |
| SSVEP decoding | `ssvep.SSVEPCCA`, `ssvep.FilterBankCCA`, `ssvep.TRCA` |
| ERP/P300 epochs | `erp.XDawnLDA` (requires already labeled target/non-target epochs) |
| Unsupervised alignment | `adaptation.EuclideanAlignment`, `euclidean_align`, `coral_align` |
| Linear readout | `readout.probe_proba`, `readout.loso_probe`, `readout.nested_loso_readout` |
| Save / load results | `registry.save_run`, `registry.load_oof` (reads historical key styles) |
| Competition ZIP | `submission.build_zip`, `submission.check_zip` |
| Quick CLI | `wyu-bci-bench oof-score x.npz`, `wyu-bci-bench oof-compare a.npz:k b.npz:k` |

Recipes for each task are in [references/recipes.md](references/recipes.md). Read it before starting a new experiment.
Fine-tuning method references and bench implementation boundaries are in [foundation_finetuning_methods.md](../../foundation_finetuning_methods.md).

## Guardrails

- **Fit inside the fold.** Scalers, priors, SSL encoders, C and epoch selection are fitted inside the training fold only; selection uses `inner_loso`. Never shuffle windows across people.
- **Foundation contracts.** CBraMod and EEGPT have verified structure/input adapters; EEGPT requires an explicit dataset-specific `amplitude_scale` and strict official-architecture checkpoint loading. LaBraM remains blocked until its tokenizer, channel mapping, preprocessing and checkpoint contract are audited and recorded in `foundation.CONTRACTS`.
- **Fine-tuning.** The `foundation.finetuning` module provides full/frozen/explicit-module training and PEFT LoRA, DoRA, PiSSA, and AdaLoRA adapter attachment. Check target layer names against the loaded model; AdaLoRA also requires the planned `total_steps`. QLoRA is not available in the current environment because `bitsandbytes` is not installed. Fine-tune and select hyperparameters inside training folds only.
- **Paradigm-specific models.** Use the SSVEP algorithms only when stimulus frequencies/trial labels are known; TRCA templates and spatial filters are trained inside each training fold. `XDawnLDA` expects segmented ERP epochs and target labels; event detection and epoch construction stay with the data loader.
- **Unsupervised alignment.** EA and CORAL do not use target labels, but estimating transforms from evaluation-domain inputs is transductive test-time adaptation. Declare that access in the protocol and do not compare it as strict inductive LOSO.
- **Protocols differ.** PROTEUS LOSO (cross-subject), NeuralBench (within-subject cross-session, stream BA) and OmniEEG (subject 8:1:1, 3 organiser seeds, valid-BA checkpoint) are different protocols. Never put their numbers in one table as comparable.
- **Development evidence only.** All 10 public PROTEUS participants have development exposure. Bench LOSO numbers are development evidence, not sealed tests.
- **Report precisely.**
  - Name the protocol, the seeds, and whether a score is a probability ensemble or a mean of seed scores.
  - Give paired intervals.
  - State when an interval crosses zero.

## Extending the bench

1. Port code verbatim, keeping module construction order so seeded initialisation is unchanged.
2. Turn hard-coded 41/120/480 values into arguments whose defaults equal the originals.
3. Add a unit test, and add an equivalence check to `scripts/verify_proteus_port.py` when an original implementation exists.
4. Update `docs/EEG_FOUNDATION_PROTOCOLS.md`, both the module table and the old→new mapping.
5. Commit to a feature branch of `wyubci/wyu-bci-bench`. Do not push to main without the user's request.
