# X: Ablations

<!-- generated:start -->
| Registry field | Value |
| --- | --- |
| Status | planned |
| Phase | any |
| Importance | supporting |
| Depends on | [A](A_isolated_per_event.md) |
| Judged against | [A](A_isolated_per_event.md), [D](D_local_per_turbine.md), [F1](F1_fedavg.md) |
| Unblocks | none |
| Notebook | none yet |
| Config | none yet |
| Runs | `results/runs/X/` |
<!-- generated:end -->

## Question
Do the untested assumptions change any conclusion?

## Why it matters
Each item below is a documented assumption that could move a headline number. None blocks the main path, but a reviewer will ask about each.

## Method
| ID | Assumption tested | Change | Needs retraining? | Run on |
| --- | --- | --- | --- | --- |
| X1 | Bottleneck width 20 suits a 243-input model (sizes were tuned for the paper's unknown feature set) | Widths 10, 20, 40 | Yes | A or D |
| X2 | Earliness weighting matches the paper's Figure 1 | Re-score with the paper's weighting | No, if `scores.parquet` is saved | A, B |
| X3 | Threshold at the 99th percentile | Quantiles 0.95 to 0.999 | No, if `scores.parquet` is saved | A, D |
| X4 | Blocked validation split | Blocked vs chronological | Yes | A |
| X5 | Local scaler | Global scaler from secure-summed count, sum, sum of squares | Yes | F1 |
| X6 | Pooled training with repeated rows | Also keep only unique training rows | Yes | C |

## Inputs and outputs
Each sub-ablation writes `results/runs/X<n>/<run>/` with the same contract as the arms. X2 and X3 read the parent run's `scores.parquet` instead of retraining, which is why saving per-row scores is part of the run contract.

## Decision rule
An ablation matters if it changes the ranking of any two arms beyond the noise rule in [S](S_seed_robustness.md).

## Results
Not run.

## Risks and open questions
- X2 is the most urgent: it affects one of the four CARE components for every arm and is flagged as unverified in the model documentation.
