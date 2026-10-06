# F2: Personalised FL, shared encoder, local decoder

<!-- generated:start -->
| Registry field | Value |
| --- | --- |
| Status | planned |
| Phase | 3 |
| Importance | headline |
| Depends on | [F1](F1_fedavg.md), [D](D_local_per_turbine.md) |
| Judged against | [C](C_pooled_dedup.md), [D](D_local_per_turbine.md), [A](A_isolated_per_event.md) |
| Unblocks | [F4](F4_private_fl.md) |
| Notebook | none yet |
| Config | [`configs/fed_personalised.yaml`](../../configs/fed_personalised.yaml) |
| Runs | `results/runs/F2/` |
<!-- generated:end -->

## Question
Can sharing what generalises (the encoder) and personalising what does not (decoder, scaler, threshold) beat both centralised training (C) and local-only training (D)?

## Why it matters
This is the candidate headline result: per-turbine specificity with fleet-wide sample size. The companion PDF notes nobody has tested it on this benchmark.

## Method: change from F1
Only the encoder layers (243 → 133 → 83 → 20) are averaged each round. The decoder layers (20 → 83 → 133 → 243) stay on the client and are fine-tuned locally for a few epochs each round. This is the shared-base, personal-head pattern. Each client ends with its own model.

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
Not run.

## Risks and open questions
- Which layers to share is itself a choice; sharing only the first encoder layer is a cheap variant to try if the main result is ambiguous.
