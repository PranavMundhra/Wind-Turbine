# F1: FedAvg, global model, local thresholds

<!-- generated:start -->
| Registry field | Value |
| --- | --- |
| Status | planned |
| Phase | 2 |
| Importance | gate |
| Depends on | [E03](E03_client_dataset.md), [C](C_pooled_dedup.md) |
| Judged against | [C](C_pooled_dedup.md), [D](D_local_per_turbine.md) |
| Unblocks | [F2](F2_personalised_fl.md), [F3](F3_fedprox_sweep.md), [F4](F4_private_fl.md) |
| Notebook | none yet |
| Config | [`configs/fed_fedavg.yaml`](../../configs/fed_fedavg.yaml) |
| Runs | `results/runs/F1/` |
<!-- generated:end -->

## Question
How does federated averaging compare with central training on exactly the same prepared client data?

## Why it matters
Federated and centralized training should first be checked on a restricted gradient-step fixture. That identity does not guarantee equal CARE with multiple local epochs, Adam or nonconvex optimization. Historical C also differs in validation membership, scaling and calibration, so C alone cannot validate the federated implementation.

## Method: change from C
Load the exact exported [E03](E03_client_dataset.md) partitions used by regenerated D through `windturbine.data.clients.load_client_dataset`. No client splits, feature selection or scaler refitting occur inside FL. Train one global model in Flower simulation, every client every round, with weights equal to actual retained fitting rows. Keep the agreed fixed input width, local scaling and local thresholds ([ADR 0002](../adr/0002-local-scaler-and-threshold.md)). Inactive inputs are zero-filled; all output dimensions contribute to loss and anomaly scores as in D.

Create a **matched centralized comparator** by training the same model on the union of those exact transformed fitting partitions, using the same calibration sets and local threshold rule. State initialization, optimizer, selection policy and training budget. Historical C remains an additional reference, not the implementation acceptance test.

Clients report unrounded per-event component values and event decisions. The server concatenates those event tables and calls `windturbine.scoring.aggregate_event_metrics` once. Event confusion counts alone cannot reconstruct Earliness or the other pointwise components. Do not average client CARE or Reliability, and do not weight CARE by training rows. Event IDs must be unique and the received cohort must match the expected evaluation events.

Implementation check before the real run: with full participation and one full-batch gradient step per round, FedAvg equals centralised gradient descent. Test that on a toy model first.

## Inputs and outputs
| Direction | Artifact | Path |
| --- | --- | --- |
| in | Client partitions; C's result for comparison | E03, C |
| out | `results.json`, `per_event.csv`, `round_log.csv` (loss per round per client) | `results/runs/F1/<run>/` |

## Decision rule
| Result | Reading |
| --- | --- |
| Restricted one-step gradient test and metric parity pass | The tested mechanics are correct; start an exploratory real-data pilot. |
| F1 matches the matched comparator within a predeclared practical margin | Report equivalence only with sufficient paired uncertainty evidence. |
| F1 differs from the matched comparator | Investigate local-step and optimizer effects; a difference is not automatically a software bug. |
| F1 differs from historical C | Report the preprocessing, membership and calibration differences; do not assign the whole gap to federation. |

## Results
Not run.

The new export-backed D seed-42 reference has now completed with CARE 0.639852
(14 detected anomalies, 6 false alarms). Its [saved execution evidence](../../results/evidence/D/20261015T014242Z_2525b30e_s42/README.md)
records the E03 dataset hash. Use that same export for the initial FL pilot; the
single D run does not replace the matched centralized comparator or seed study.

## Risks and open questions
- C and F1 differ in scaler scope (global vs local) as well as training mechanism; X5 separates the two.
- Communication is not a constraint: about 363 KB per model, under 1.6 GB for 22 clients and 100 rounds (companion PDF).
