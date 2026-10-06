# S: Seed robustness and confidence intervals

<!-- generated:start -->
| Registry field | Value |
| --- | --- |
| Status | planned |
| Phase | 0 |
| Importance | gate |
| Depends on | [A](A_isolated_per_event.md), [B](B_pooled_fleet.md) |
| Judged against | none |
| Unblocks | none |
| Notebook | none yet |
| Config | none yet |
| Runs | `results/runs/S/` |
<!-- generated:end -->

## Question
Which differences between experiments are larger than run-to-run noise?

## Why it matters
Two identical runs of B gave CARE 0.570 and 0.583, so differences of about 0.013 are noise. The A-B Reliability gap (0.611 vs 0.593) is inside that. The 58 events also come from only 22 turbines (one turbine contributes 6), so they are not 58 independent samples. Every comparison in the graph depends on S being applied.

## Method
1. Turn on deterministic ops (`windturbine.runs.set_seeds(seed, deterministic=True)`) and re-run B twice at one seed: if CARE now matches exactly, the old noise was cuDNN non-determinism.
2. Run every scored experiment at 3 or more seeds; report mean and standard deviation.
3. For each pairwise comparison, bootstrap by resampling turbines (not events) and report a 95% interval for the CARE difference.

## Inputs and outputs
| Direction | Artifact | Path |
| --- | --- | --- |
| in | Run directories of every scored experiment | `results/runs/*/` |
| out | `seed_table.csv`, `pairwise_ci.csv` | `results/summary/` |

## Decision rule
A difference counts only if its turbine-bootstrap interval excludes zero.

## Results
Not run. Existing evidence: B twice at seed 42, CARE 0.570 and 0.583.

## Risks and open questions
- S is drawn depending on A and B because it starts there, but its rule applies to every scored node in the graph.
