# B: Pooled, one fleet-wide model

<!-- generated:start -->
| Registry field | Value |
| --- | --- |
| Status | done-unverified |
| Phase | done |
| Importance | baseline |
| Depends on | [E00](E00_data_cache.md), [E02](E02_leakage_audit.md) |
| Judged against | [A](A_isolated_per_event.md) |
| Unblocks | [C](C_pooled_dedup.md), [F1](F1_fedavg.md), [F2](F2_personalised_fl.md), [F3](F3_fedprox_sweep.md), [F4](F4_private_fl.md), [S](S_seed_robustness.md) |
| Notebook | [`notebooks/11_arm_pooled.ipynb`](../../notebooks/11_arm_pooled.ipynb) |
| Config | [`configs/arm_pooled.yaml`](../../configs/arm_pooled.yaml) |
| Runs | `results/runs/B/` |
<!-- generated:end -->

## Question
Does one model trained on the whole fleet beat per-event models?

## Why it matters
It is the centralised baseline, and the experiment in which the duplication was first measured. Its Accuracy is inflated by contamination, so it must be read alongside [C](C_pooled_dedup.md).

## Method: change from A
One model and one threshold for all 58 events, trained on 2,281,373 pooled normal rows with one global scaler. Validation holds out whole turbines (3 of 22), because an event-level split would put duplicated rows on both sides.

## Inputs and outputs
| Direction | Artifact | Path |
| --- | --- | --- |
| in | Event cache; fingerprint method | E00, E02 |
| in (hidden, to remove) | `/kaggle/working/results_isolated.json` from A, read for the comparison table | replaced by `scripts/compare_runs.py` |
| out | `results_pooled.json`, `per_event_pooled.csv`, `leakage_pooled.csv`, `per_event_with_leakage.csv` | target: `results/runs/B/<run>/` |

## Results
Reported in prose only. A second run with identical code gave CARE 0.570 (TP 6), which sets the noise floor.

| Run id | Seed | CARE | Coverage | Earliness | Reliability | Accuracy | TP/FP/FN/TN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none recorded | 42 | 0.583 | 0.227 | 0.161 | 0.593 | 0.967 | 7/1/20/30 |

Contamination: 40,216 of 158,528 prediction rows (25.4%) were in the training corpus, across 27 of 58 events. The documented decomposition estimates it adds about +0.011 CARE.

## Risks and open questions
- The comparison cell only works if A ran earlier in the same Kaggle session.
- About 25 minutes on a P100.
