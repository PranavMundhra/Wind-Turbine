# A: Isolated, one model per event

<!-- generated:start -->
| Registry field | Value |
| --- | --- |
| Status | done-unverified |
| Phase | done |
| Importance | baseline |
| Depends on | [E00](E00_data_cache.md) |
| Judged against | none |
| Unblocks | [S](S_seed_robustness.md), [X](X_ablations.md) |
| Notebook | [`notebooks/10_arm_isolated.ipynb`](../../notebooks/10_arm_isolated.ipynb) |
| Config | [`configs/arm_isolated.yaml`](../../configs/arm_isolated.yaml) |
| Runs | `results/runs/A/` |
<!-- generated:end -->

## Question
How well does the paper-faithful per-event autoencoder score on Farm C?

## Why it matters
It is the reference point for everything else and the only arm that cannot be contaminated by cross-file duplication, because each model only ever sees one file.

## Method
58 autoencoders, one per event dataset, about 38,000 training rows each. Architecture 243 → 133 → 83 → 20 → 83 → 133 → 243, Adam at 0.0056 with `ReduceLROnPlateau`, batch 64, up to 200 epochs, early stopping patience 10. Validation is a blocked split (5-day blocks, every 7th held out). Threshold is the 99th percentile of reconstruction error on held-out normal rows, one per event. Full detail: [MODEL_DOCUMENTATION.md](../MODEL_DOCUMENTATION.md).

## Inputs and outputs
| Direction | Artifact | Path |
| --- | --- | --- |
| in | Event cache | E00 |
| out | `results_isolated.json`, `per_event_isolated.csv`, `leakage_counterfactual.csv` | today: `/kaggle/working`; target: `results/runs/A/<run>/` |

## Results
Reported in prose only (single seed, no run directory, notebook outputs cleared):

| Run id | Seed | CARE | Coverage | Earliness | Reliability | Accuracy | TP/FP/FN/TN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none recorded | 42 | 0.628 | 0.451 | 0.332 | 0.611 | 0.874 | 16/10/11/21 |

## Risks and open questions
- Not reproducible from the repo until re-run with a run manifest.
- Three events (58, 60, 85) hit the 200-epoch cap without early stopping.
- About 113 minutes on a P100, so the seed study ([S](S_seed_robustness.md)) costs about 6 GPU-hours for 3 seeds.
- A is per-event, but federated clients are per-turbine; [D](D_local_per_turbine.md) is the correct local-only anchor for federated comparisons.
