# F2: Personalised FL, shared encoder, local decoder

<!-- generated:start -->
| Registry field | Value |
| --- | --- |
| Status | partial |
| Phase | 3 |
| Importance | headline |
| Depends on | [F1](F1_fedavg.md), [D](D_local_per_turbine.md) |
| Judged against | [C](C_pooled_dedup.md), [D](D_local_per_turbine.md), [A](A_isolated_per_event.md) |
| Unblocks | [F4](F4_private_fl.md) |
| Notebook | [`notebooks/31_fed_personalised.ipynb`](../../notebooks/31_fed_personalised.ipynb) |
| Config | [`configs/fed_personalised.yaml`](../../configs/fed_personalised.yaml) |
| Runs | `results/runs/F2/` |
<!-- generated:end -->

## Question
Can sharing what generalises (the encoder) and personalising what does not (decoder, scaler, threshold) beat both centralised training (C) and local-only training (D)?

## Why it matters
This is the candidate headline result: per-turbine specificity with fleet-wide sample size. The companion PDF notes nobody has tested it on this benchmark.

## Method: change from F1
Only the encoder layers (243 → 133 → 83 → 20) are averaged each round. The decoder layers (20 → 83 → 133 → 243) stay on the client. This is the shared-base, personal-head pattern. Each client ends with its own model.

**Implemented as FedPer** (`windturbine.federated.run_federated`, `strategy: fedper`). Every round each turbine loads the broadcast encoder and its own decoder, trains all layers for one local epoch with a fresh Adam optimizer, and returns only the encoder. Flower's FedAvg averages the encoders by retained training rows; decoders never leave the turbine. FedRep-style alternating training was considered and not chosen, because it costs about three times as much per round. Each round costs about the same as F1's FedAvg branch alone.

Everything else matches F1: the same E03 export and dataset hash, 100 fixed rounds, local scalers and thresholds, and seeded per-client shuffles. Selection uses the round with the lowest row-weighted validation MSE, where each turbine scores with its own model. Prediction windows never enter selection. Each turbine's selected model is saved to `fedper/selected/client_<id>.npz`. The run projects its runtime after the first rounds and stops early if it would exceed `max_hours`.

## Inputs and outputs
| Direction | Artifact | Path |
| --- | --- | --- |
| in | Client partitions; F1's federated code; D and C results | E03, F1, D, C |
| out | `results.json`, `per_event.csv`, `round_log.csv` | `results/runs/F2/<run>/` |

## Decision rule
| Result (3 seeds) | Reading |
| --- | --- |
| F2 > max(C, D) beyond noise | Headline: personalised FL beats both centralised and isolated training. |
| F2 ≈ D | Sharing the encoder adds nothing; the fleet carries no transferable signal at this model size. |
| F2 ≈ C | Personalising the decoder is not enough; try personalising more layers or event-level thresholds. |

## Results
Implemented ([notebook](../../notebooks/31_fed_personalised.ipynb)); the real-data run has not happened. The synthetic smoke run and unit tests check mechanics only. Those tests confirm the encoder arrays are identical across clients while the decoders differ, plus checkpoint readback and CARE parity.

## Risks and open questions
- Which layers to share is itself a choice; sharing only the first encoder layer is a cheap variant to try if the main result is ambiguous.
