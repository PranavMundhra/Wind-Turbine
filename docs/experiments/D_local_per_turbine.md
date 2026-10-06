# D: Local per-turbine, no collaboration

<!-- generated:start -->
| Registry field | Value |
| --- | --- |
| Status | planned |
| Phase | 0 |
| Importance | gate |
| Depends on | [E03](E03_client_dataset.md) |
| Judged against | [A](A_isolated_per_event.md), [C](C_pooled_dedup.md) |
| Unblocks | [F2](F2_personalised_fl.md), [F3](F3_fedprox_sweep.md), [F4](F4_private_fl.md) |
| Notebook | none yet |
| Config | [`configs/arm_local_turbine.yaml`](../../configs/arm_local_turbine.yaml) |
| Runs | `results/runs/D/` |
<!-- generated:end -->

## Question
How well does a turbine do with only its own data? This is the floor any federated scheme must beat to justify collaborating.

## Why it matters
This arm is new to the plan. The companion PDF frames the federated curve as running between A and B, but federated clients are turbines, not events, so the correct "no sharing" endpoint is a per-turbine model, and the correct "full sharing" endpoint is C. D is also what FedProx approaches as mu goes to 0 ([F3](F3_fedprox_sweep.md)). Compared with A, it shows whether specialising per event adds anything beyond specialising per turbine.

## Method: change from A
22 models, one per `asset_id`, each trained on that turbine's leak-free rows from [E03](E03_client_dataset.md). One scaler and threshold per turbine. Every event is scored by its turbine's model, then CARE is computed over all 58 events exactly as in A.

## Inputs and outputs
| Direction | Artifact | Path |
| --- | --- | --- |
| in | Client partitions | E03 |
| out | `results.json`, `per_event.csv`, `scores.parquet` | `results/runs/D/<run>/` |

## Decision rule
| Result | Reading |
| --- | --- |
| D ≈ A | Turbine-level specialisation is enough; federated personalisation targets D. |
| D < A by more than noise | Event-level context matters (seasons, operating regime per file); consider local thresholds per event in F2. |
| D < C | The fleet helps even without personalisation; F1 alone may suffice. |

## Results
Not run.

## Risks and open questions
- Leak removal may shrink some turbines' training sets heavily; report rows removed per turbine.
- The 3 turbines with a single event get the same data as their Arm A model, so for them D and A should agree within noise. That is a free sanity check.
