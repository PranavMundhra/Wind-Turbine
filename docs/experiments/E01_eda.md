# E01: Exploratory data analysis

<!-- generated:start -->
| Registry field | Value |
| --- | --- |
| Status | done |
| Phase | R |
| Importance | supporting |
| Depends on | none |
| Judged against | none |
| Unblocks | [D](D_local_per_turbine.md), [E03](E03_client_dataset.md), [F1](F1_fedavg.md), [F2](F2_personalised_fl.md), [F3](F3_fedprox_sweep.md), [F4](F4_private_fl.md) |
| Notebook | [`notebooks/01_eda.ipynb`](../../notebooks/01_eda.ipynb) |
| Config | none yet |
| Runs | `results/runs/E01/` |
<!-- generated:end -->

## Question
Which channels are usable, redundant or implausible, and is the avg-only feature set justified?

## Why it matters
It justifies two choices every arm makes: avg-only features (238 selected, 243 after the angle transform) and dropping 12 of 17 metadata-proposed angle columns whose ranges are not angles. Today its outputs are not consumed by anything, so those choices live only in notebook code.

## Method
Data quality, missingness, sentinels, flatlines, redundancy, power curve, seasonality on Farm C.

## Inputs and outputs
| Direction | Artifact | Path |
| --- | --- | --- |
| in | Raw Farm C CSVs | `$WT_DATA_ROOT` |
| out | `channel_quality.csv`, `recommended_drop_columns.csv`, `redundant_channel_pairs.csv` and others | `results/eda/` (proposed) |
| out (new) | `feature_schema.json`: the one feature list every arm and client uses | `results/eda/feature_schema.json` |

## Decision rule
The feature schema is frozen before Phase 0. Every arm loads it; none derives its own.

## Results
Outputs are embedded in the notebook (5.1 MB). Export the CSVs to `results/eda/` and commit `feature_schema.json`.

## Risks and open questions
- Federated clients must share an identical input dimension; a schema derived per client breaks averaging. That is why the schema is a file, not a computation.
