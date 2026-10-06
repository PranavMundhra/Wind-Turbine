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
Does naive federated averaging reproduce the centralised result, as the companion PDF predicts?

## Why it matters
The PDF states the prediction before running: FedAvg's fixed point roughly minimises the weighted average of client losses, which is what the pooled arm optimised, so FedAvg should land near the centralised score. Confirming it shows the gap is caused by the shared-model objective, not by the data-sharing mechanism, and it validates the federated code before F2 and F3 build on it.

## Method: change from C
Same leak-free data as D (22 turbine clients from [E03](E03_client_dataset.md)), but trained as one global model by FedAvg in Flower simulation, every client every round, clients weighted by row count after leak removal. Scaler and threshold stay local ([ADR 0002](../adr/0002-local-scaler-and-threshold.md)). Each client reports per-event TP/FP/FN/TN; the server computes CARE.

Implementation check before the real run: with full participation and one full-batch gradient step per round, FedAvg equals centralised gradient descent. Test that on a toy model first.

## Inputs and outputs
| Direction | Artifact | Path |
| --- | --- | --- |
| in | Client partitions; C's result for comparison | E03, C |
| out | `results.json`, `per_event.csv`, `round_log.csv` (loss per round per client) | `results/runs/F1/<run>/` |

## Decision rule
| Result | Reading |
| --- | --- |
| F1 ≈ C (within noise) | Prediction confirmed; federation itself costs nothing here. Proceed to F2. |
| F1 well below C | Client drift or a bug; check the toy-equivalence test and local-epoch count before going further. |
| F1 above C | Local scalers/thresholds are already doing personalisation work; measure that with ablation X5. |

## Results
Not run.

## Risks and open questions
- C and F1 differ in scaler scope (global vs local) as well as training mechanism; X5 separates the two.
- Communication is not a constraint: about 363 KB per model, under 1.6 GB for 22 clients and 100 rounds (companion PDF).
