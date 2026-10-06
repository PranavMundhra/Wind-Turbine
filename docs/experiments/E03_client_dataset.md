# E03: Leak-free per-turbine client dataset

<!-- generated:start -->
| Registry field | Value |
| --- | --- |
| Status | partial |
| Phase | 0 |
| Importance | gate |
| Depends on | [E00](E00_data_cache.md), [E01](E01_eda.md), [E02](E02_leakage_audit.md) |
| Judged against | none |
| Unblocks | [D](D_local_per_turbine.md), [F1](F1_fedavg.md), [F2](F2_personalised_fl.md), [F3](F3_fedprox_sweep.md), [F4](F4_private_fl.md) |
| Notebook | [`notebooks/20_federated_clients.ipynb`](../../notebooks/20_federated_clients.ipynb) |
| Config | [`configs/clients.yaml`](../../configs/clients.yaml) |
| Runs | `results/runs/E03/` |
<!-- generated:end -->

## Question
Can we build 22 per-turbine partitions that use the arms' exact preprocessing and contain no training row that is evaluated anywhere for that turbine?

## Why it matters
Every federated arm and Arm D train on these partitions. It is the bridge between the centralised work and the federated work, and today that bridge is broken.

## Method: change from the current notebook
The current builder (`20_federated_clients.ipynb`) partitions by `asset_id` correctly but is disconnected from the rest of the pipeline:

| Current builder | Required |
| --- | --- |
| Reads raw CSVs itself | Read the E00 cache |
| No avg-only selection, angle transform or status/power-curve normal mask | Apply `feature_schema.json` from E01 and the arms' normal mask |
| Chronological validation split | Blocked 5-day split, as in the arms (the chronological split produced a 42x threshold spread in Arm A) |
| No leakage removal | Remove every training row whose fingerprint appears in any prediction window of the same turbine |

The last row is the important correction. Partitioning by turbine removes leakage between clients, but the duplication E02 found is within a turbine: a row that is `prediction` in one event file of turbine 53 is `train` in another file of turbine 53. Simply dropping exact duplicate rows does not remove it; dropping training rows seen in that turbine's prediction windows does.

## Inputs and outputs
| Direction | Artifact | Path |
| --- | --- | --- |
| in | Event cache, feature schema, fingerprint method | E00, E01, E02 |
| out | Per-client train / validation / prediction partitions | `data/clients/client_asset_<id>/` |
| out | `client_manifest.csv`: rows before and after leak removal, per client | `results/clients/` |

## Decision rule
Accept when a test confirms zero fingerprints shared between any client's training rows and any prediction row of the same client, and every client has the same feature columns in the same order.

## Results
Not run in the leak-free form.

## Risks and open questions
- Leak removal shrinks some clients; record by how much, because FedAvg weights clients by row count.
