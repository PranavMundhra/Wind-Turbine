# F4: Secure aggregation and differential privacy

<!-- generated:start -->
| Registry field | Value |
| --- | --- |
| Status | partial |
| Phase | 5 |
| Importance | supporting |
| Depends on | [F2](F2_personalised_fl.md), [F3](F3_fedprox_sweep.md) |
| Judged against | [F2](F2_personalised_fl.md), [F3](F3_fedprox_sweep.md) |
| Unblocks | none |
| Notebook | [`notebooks/33_fed_private.ipynb`](../../notebooks/33_fed_private.ipynb) |
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

### Implemented design
- **Base.** `base_config` in `configs/fed_private.yaml`, defaulting to F2's FedPer. Override it with `WT_F4_BASE`, and give `WT_F4_MU` when the base is FedProx. The private settings are merged on top, so the base budget and layers stay the same.
- **Secure aggregation (simulated).** Each client encodes rows × weights in 64-bit fixed point (2^24 scale) and adds one pairwise mask per other client. The server sums only masked vectors, and the masks cancel exactly modulo 2^64. Every round is checked against Flower's plain FedAvg, and the masked sum becomes the global update. Masks come from a seeded generator, with no key agreement or dropout recovery. This verifies the arithmetic, not a deployed protocol.
- **Record-level DP-SGD.** Every local step clips each record's gradient to `dp_clip_norm` (1.0). Gaussian noise with standard deviation `noise × clip` is added to the clipped sum, and the result is divided by the batch size. Per-example norms are computed exactly for the Dense stack, using |a_i|²|g_i|² per layer, without building per-example gradients. A test matches them to a full per-example Jacobian. This runs about 1.3× slower than plain training, not 7×.
- **Accounting.** The `dp-accounting` RDP accountant computes (epsilon, delta = 1e-6) separately per turbine, from its sampling rate and step count. The headline is the worst turbine. Batches are shuffled, not Poisson-sampled, so epsilon is the standard approximation for that sampler and not a proven bound. Not covered: the server's checkpoint selection sees each turbine's validation MSE, and thresholds use non-private validation rows.
- **Grid.** Noise multipliers 0, 0.5, 1, 2, 4, one per Kaggle session (`WT_F4_NOISE`). Noise 0 is secure aggregation only, the correctness check. The notebook's final section writes `care_vs_epsilon.csv` from the attached session outputs.

## Inputs and outputs
| Direction | Artifact | Path |
| --- | --- | --- |
| in | Best F2 / F3 configuration | F2, F3 |
| out | `care_vs_epsilon.csv`, plot | `results/runs/F4/` |

## Decision rule
Report the epsilon at which CARE falls to D's level: past that point, collaborating under privacy is worse than not collaborating.

## Results
Implemented ([notebook](../../notebooks/33_fed_private.ipynb)); no real-data run yet. Run noise 0 first and compare its CARE with the matching base run.

## Risks and open questions
- Noise blurs exactly the tail behaviour an anomaly detector relies on, so Coverage and Earliness may fall faster than Accuracy. Report the components, not only CARE.
- DP library: `dp-accounting` (Google's maintained accountant) for epsilon only. Clipping and noise are implemented in the repo, because `tensorflow-privacy` does not support Keras 3. On Kaggle without internet, attach its wheel like `flwr`.
