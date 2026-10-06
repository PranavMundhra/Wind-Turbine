# E02: Leakage audit

<!-- generated:start -->
| Registry field | Value |
| --- | --- |
| Status | done |
| Phase | R |
| Importance | gate |
| Depends on | none |
| Judged against | none |
| Unblocks | [B](B_pooled_fleet.md), [C](C_pooled_dedup.md), [D](D_local_per_turbine.md), [E03](E03_client_dataset.md), [F1](F1_fedavg.md), [F2](F2_personalised_fl.md), [F3](F3_fedprox_sweep.md), [F4](F4_private_fl.md), [S](S_seed_robustness.md) |
| Notebook | [`notebooks/02_leakage_audit.ipynb`](../../notebooks/02_leakage_audit.ipynb) |
| Config | none yet |
| Runs | `results/runs/E02/` |
<!-- generated:end -->

## Question
Are rows duplicated across event files, and if so, exactly how?

## Why it matters
It is the reason Arm B's Accuracy cannot be trusted, the reason Arm C exists, and the reason federated clients must be turbines, not event files ([ADR 0001](../adr/0001-federated-client-is-asset-id.md)). It also defines the fingerprint used to remove leaked rows in C, D, E03 and every federated arm.

## Method
Fingerprint rows on 12 columns to find alignment, verify on 20 disjoint holdout columns and on all 226 columns, check contiguity and offsets, and run cross-turbine and wrong-shift controls.

## Inputs and outputs
| Direction | Artifact | Path |
| --- | --- | --- |
| in | Raw Farm C CSVs | `$WT_DATA_ROOT` |
| out | Pair tests, controls, summary | [`results/leakage_audit/`](../../results/leakage_audit/) |
| out | Fingerprint method (6-decimal row hash) | `windturbine.audit.row_fingerprints` |

## Results
From [`evidence_summary.json`](../../results/leakage_audit/evidence_summary.json): 64 within-turbine pairs tested, 49 with enough matches, all 49 verified duplicates; worst holdout and full-width match 1.000 over 226 columns; 0 cross-turbine false positives; time shifts of -365, 0 and +365 days.

## Risks and open questions
- The notebook writes three files that are not in the repo: `strict_informative_verification.csv`, `zero_contamination_audit.csv`, `zero_contamination_by_column.csv`. Re-run and commit them, or remove the references.
- No run manifest exists for this audit; the next run should write one.
