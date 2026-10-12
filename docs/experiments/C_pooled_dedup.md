# C: Pooled, leaked rows removed

<!-- generated:start -->
| Registry field | Value |
| --- | --- |
| Status | done |
| Phase | 0 |
| Importance | gate |
| Depends on | [B](B_pooled_fleet.md), [E02](E02_leakage_audit.md) |
| Judged against | [B](B_pooled_fleet.md), [A](A_isolated_per_event.md) |
| Unblocks | [F1](F1_fedavg.md), [F2](F2_personalised_fl.md), [F3](F3_fedprox_sweep.md), [F4](F4_private_fl.md) |
| Notebook | [`notebooks/12_arm_pooled_dedup.ipynb`](../../notebooks/12_arm_pooled_dedup.ipynb) |
| Config | [`configs/arm_pooled_dedup.yaml`](../../configs/arm_pooled_dedup.yaml) |
| Runs | `results/runs/C/` |
<!-- generated:end -->

## Question
How much of the A-B gap (0.045 CARE) is contamination, and how much is specialisation?

## Why it matters
The companion PDF calls Phase 0 a prerequisite: federating contaminated data reproduces the contamination inside the clients. C also replaces B as the honest centralised baseline that [F1](F1_fedavg.md) should match.

## Method: change from B
Exactly one change. Remove from the pooled training corpus every row whose fingerprint (6 decimals, same columns as the audit) appears in any event's prediction window. Same turbine holdout, scaler, architecture and threshold rule as B. Record how many rows were removed per event.

As implemented in the notebook:
- The fingerprint is B's leakage measurement: `row_fingerprints` over all 238 selected `_avg` sensor columns, rounded to 6 decimals.
- The filter covers every normal training row, so the 3 held-out validation turbines (early stopping and threshold) are cleaned as well as the fit turbines. Within-training duplicates are kept; removing them is ablation X6.
- The notebook also re-measures leakage exactly as B did before removing anything (at seed 42 this should reproduce B's 25.4%), then measures it on the rows that actually enter training. That number must be zero, and the notebook asserts it.

Run: `WT_DATA_ROOT=<dir containing "Wind Farm C"> WT_SEED=42`, then execute the notebook. On Kaggle the first cell clones this repo.

## Inputs and outputs
| Direction | Artifact | Path |
| --- | --- | --- |
| in | Event cache, fingerprints, B's config | E00, E02, B |
| out | `results.json`, `per_event.csv`, `scores.parquet`, `removed_rows.csv` (per event), `leakage_before_removal.csv` | `results/runs/C/<run>/` |

## Decision rule
The documented decomposition predicts CARE near 0.572 (B minus about 0.011).

| Result (mean of 3 seeds) | Reading |
| --- | --- |
| C close to 0.572 | Decomposition confirmed: most of the A-C gap is specialisation. Go to D and the federated arms. |
| C well below 0.572 | Contamination was helping more than estimated; re-check the decomposition before trusting B anywhere. |
| C close to A | The gap was mostly contamination; personalisation (F2) has less to recover. |

## Results
Run locally on CPU (TensorFlow 2.21, pandas 3.0, op determinism on), 2026-10-11, from an uncommitted working tree on top of `fd1cfb5`, so every manifest has `"dirty": true`. Run directories are under `results/runs/C/`.

| Run id | Seed | Held-out turbines | Threshold | CARE | Coverage | Earliness | Reliability | Accuracy | TP/FP/FN/TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20261011T181204Z_22adc845_s42` | 42 | 33, 44, 50 | 19.2 | 0.543 | 0.156 | 0.104 | 0.465 | 0.994 | 4/0/23/31 |
| `20261011T182445Z_b9ea4251_s43` | 43 | 5, 15, 35 | 161.0 | 0.000 | 0.002 | 0.001 | 0.000 | 0.999 | 0/0/27/31 |
| `20261011T184331Z_cda4e6dd_s44` | 44 | 2, 32, 53 | 177.2 | 0.437 | 0.034 | 0.016 | 0.143 | 0.996 | 1/1/26/30 |
| **mean ± sd** | | | | **0.326 ± 0.288** | | | | | |

Removal check: at seed 42 the "before" leakage reproduces B exactly: 2,281,373 pooled training rows, and 40,216 of 158,528 prediction rows (25.4%) present in training. Removal took 78,802 training rows and 9,270 validation rows. Leakage after removal is 0 in every run.

**Reading.** At seed 42, the seed B used, C is 0.543. That is 0.029 below the predicted 0.572, so more than the 0.013 noise floor, but the environments differ (B was a Kaggle GPU run). The 3-seed mean, 0.326, is "well below 0.572" under the decision rule. The cause, though, is not contamination; it is B's validation design, which C keeps unchanged:
- Early stopping and the single global threshold both come from 3 held-out turbines. Which 3 changes with the seed.
- At seeds 43 and 44 the held-out set contains a turbine the fleet model reconstructs badly. Validation MSE is 16 and 36 respectively, against 0.66 at seed 42.
- The 99th-percentile threshold therefore rises about 9×, early stopping keeps the epoch-1 model, and almost nothing is flagged.

B's 0.583 is very likely one lucky holdout draw, not a stable property of pooled training. Before C-B can be read as "what contamination was worth", B has to be re-run at the same seeds in the same environment (the [S](S_seed_robustness.md) experiment). That pairing is also needed to fix the decomposition.

## Risks and open questions
- The pooled arms are fragile to the validation-turbine draw. A fix (for example, more held-out turbines, or a per-turbine threshold as in [ADR 0002](../adr/0002-local-scaler-and-threshold.md)) would be a change to B, so it belongs in its own experiment, not in C.
- Removing leaked rows also changes the training set size; report it so the result is not read as a pure data-quality effect.
- Removing within-training duplicates as well is a separate change; it is ablation X6, not part of C.
