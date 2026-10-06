# E00: Data cache

<!-- generated:start -->
| Registry field | Value |
| --- | --- |
| Status | done |
| Phase | R |
| Importance | supporting |
| Depends on | none |
| Judged against | none |
| Unblocks | [A](A_isolated_per_event.md), [B](B_pooled_fleet.md), [C](C_pooled_dedup.md), [D](D_local_per_turbine.md), [E03](E03_client_dataset.md), [F1](F1_fedavg.md), [F2](F2_personalised_fl.md), [F3](F3_fedprox_sweep.md), [F4](F4_private_fl.md), [S](S_seed_robustness.md), [X](X_ablations.md) |
| Notebook | [`notebooks/00_build_cache.ipynb`](../../notebooks/00_build_cache.ipynb) |
| Config | none yet |
| Runs | `results/runs/E00/` |
<!-- generated:end -->

## Question
Can every experiment read the same 58 Farm C events from one fast, verified copy?

## Why it matters
Arms A and B both read this cache; E03 should too (today it re-parses raw CSVs, see [E03](E03_client_dataset.md)). If two experiments read different copies, a difference in their results could be a data difference.

## Method
Parse each event CSV once, keep the descriptive columns (`time_stamp`, `asset_id`, `id`, `train_test`, `status_type_id`) plus the sensor columns, store as `event_<id>.parquet`, and write `care_cache_manifest.json`. Run on a CPU session.

## Inputs and outputs
| Direction | Artifact | Path |
| --- | --- | --- |
| in | Raw Farm C CSVs, `event_info.csv`, `feature_description.csv` | `$WT_DATA_ROOT` |
| out | One parquet per event | `data/cache/event_<id>.parquet` |
| out | Cache manifest (hash it into every run manifest) | `data/cache/care_cache_manifest.json` |

## Decision rule
The cache is valid when its manifest hash matches the one recorded by the run being reproduced.

## Results
Not a scored experiment.

## Risks and open questions
- The cache helpers are copied into both arm notebooks; move them to `windturbine.data` (ADR 0003).
