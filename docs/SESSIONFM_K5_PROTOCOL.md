# SessionFM K5 preregistration

K5 is E3 / S1 from `01_毕业论文/毕业论文完整实验方案_20261027.md`. This file records the fixed protocol before any PROTEUS downstream evaluation. The current `scripts/run_sessionfm_k5.py` only performs K4 checkpoint preflight and then stops; evaluation remains disabled.

| Arm | Definition |
|---|---|
| a | Existing SCC-M2 OOF, reused without rerun |
| b | SessionFM K4 main pretrained weights, stateful SBC + causal prior centering + CCB |
| c | Randomly initialized SessionFM with the same stateful adaptation |
| d | SessionFM stateless inference: `n=0`, no feature centering, no CCB |
| e1-e4 | K4 A1-A4 pretrained weights, same downstream recipe |
| e5 | K4 main weights with prior anchoring removed |

Pretraining/downstream seed pairs are 20261101/33, 20261102/34, and 20261103/35. Average the three probability matrices before computing arm summaries. K4 preflight requires exactly the three main seed run names plus A1-A4 and verifies each `weights.pt` SHA-256 against `summary.json`. Any missing run, checkpoint, or hash is a hard failure before dataset access.

For each outer LOSO fold, the fixed inner validation participant is `SUBJECTS[(fold+1)%10]`. Fit the 1394-feature Logistic anchor on training-fold features only after past-only causal centering. Stage 1 freezes the SessionFM encoder and trains the attention pool query, residual MLP, and gate for exactly 10 epochs at 4e-4. Stage 2 unfreezes the model and uses AdamW (weight decay 0.05), top backbone layer LR 1e-4 with per-layer multiplier 0.65 toward lower layers, and head LR 4e-4. Use label smoothing 0.1, gradient clipping 1.0, and `0.9 * CE + 0.1 * masked spectral MSE`; mask 30% of the eight token positions during training. Select stage-2 epoch only on the fixed inner validation participant, with minimum 12, maximum 45, and patience 8.

G1 (b versus c) passes only if cell-balanced-accuracy gain is at least 1.0 percentage point, participant-cluster-bootstrap 95% lower bound is greater than zero, and at least 7/10 participants improve. G2 (b versus a) uses the same criteria and additionally requires worst-participant decline no greater than 3 percentage points. e1-e5 are descriptive mechanism checks without pass thresholds and cannot select a model.

Stateful inference predicts before adding the current window to three session-local sums: SBC spectral features, 1394-D prior features, and uncorrected class probabilities for CCB. `reset_state()` clears all three. The stateless arm uses none of these histories. All synthetic tests are score-free; no PROTEUS score computation, display, or persistence is part of preparation.
