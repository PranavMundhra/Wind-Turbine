# C: Pooled, leaked rows removed

<!-- generated:start -->
| Registry field | Value |
| --- | --- |
| Status | planned |
| Phase | 0 |
| Importance | gate |
| Depends on | [B](B_pooled_fleet.md), [E02](E02_leakage_audit.md) |
| Judged against | [B](B_pooled_fleet.md), [A](A_isolated_per_event.md) |
| Unblocks | [F1](F1_fedavg.md), [F2](F2_personalised_fl.md), [F3](F3_fedprox_sweep.md), [F4](F4_private_fl.md) |
| Notebook | none yet |
| Config | [`configs/arm_pooled_dedup.yaml`](../../configs/arm_pooled_dedup.yaml) |
| Runs | `results/runs/C/` |
<!-- generated:end -->

## Question
How much of the A-B gap (0.045 CARE) is contamination, and how much is specialisation?

## Why it matters
The companion PDF calls Phase 0 a prerequisite: federating contaminated data reproduces the contamination inside the clients. C also replaces B as the honest centralised baseline that [F1](F1_fedavg.md) should match.

## Method: change from B
Exactly one change. Remove from the pooled training corpus every row whose fingerprint (6 decimals, same columns as the audit) appears in any event's prediction window. Same turbine holdout, scaler, architecture and threshold rule as B. Record how many rows were removed per event.

## Inputs and outputs
| Direction | Artifact | Path |
| --- | --- | --- |
| in | Event cache, fingerprints, B's config | E00, E02, B |
| out | `results.json`, `per_event.csv`, `scores.parquet`, removed-row counts | `results/runs/C/<run>/` |

## Decision rule
The documented decomposition predicts CARE near 0.572 (B minus about 0.011).

| Result (mean of 3 seeds) | Reading |
| --- | --- |
| C close to 0.572 | Decomposition confirmed: most of the A-C gap is specialisation. Go to D and the federated arms. |
| C well below 0.572 | Contamination was helping more than estimated; re-check the decomposition before trusting B anywhere. |
| C close to A | The gap was mostly contamination; personalisation (F2) has less to recover. |

## Results
Not run.

## Risks and open questions
- Removing leaked rows also changes the training set size; report it so the result is not read as a pure data-quality effect.
- Removing within-training duplicates as well is a separate change; it is ablation X6, not part of C.
