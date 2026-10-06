# F4: Secure aggregation and differential privacy

<!-- generated:start -->
| Registry field | Value |
| --- | --- |
| Status | planned |
| Phase | 5 |
| Importance | supporting |
| Depends on | [F2](F2_personalised_fl.md), [F3](F3_fedprox_sweep.md) |
| Judged against | [F2](F2_personalised_fl.md), [F3](F3_fedprox_sweep.md) |
| Unblocks | none |
| Notebook | none yet |
| Config | [`configs/fed_private.yaml`](../../configs/fed_private.yaml) |
| Runs | `results/runs/F4/` |
<!-- generated:end -->

## Question
What does formal privacy cost in CARE?

## Why it matters
Federated learning removes raw-data sharing, which is the commercial blocker, but does not by itself give a formal privacy guarantee: weight updates can leak training data. This turns "privacy-preserving" from a claim into a measured trade-off.

## Method: change from the best of F2 / F3
1. Add secure aggregation for weight updates (and scaler statistics, if a global scaler is ever used). The server then sees only sums. CARE should be unchanged, so this step doubles as a correctness check.
2. Add differential privacy with clipped, noised updates. Sweep the noise level and report CARE against the resulting privacy budget epsilon.

## Inputs and outputs
| Direction | Artifact | Path |
| --- | --- | --- |
| in | Best F2 / F3 configuration | F2, F3 |
| out | `care_vs_epsilon.csv`, plot | `results/runs/F4/` |

## Decision rule
Report the epsilon at which CARE falls to D's level: past that point, collaborating under privacy is worse than not collaborating.

## Results
Not run.

## Risks and open questions
- Noise blurs exactly the tail behaviour an anomaly detector relies on, so Coverage and Earliness may fall faster than Accuracy. Report the components, not only CARE.
- Choose the DP library and accounting method when this phase starts; check their current maintenance status then.
