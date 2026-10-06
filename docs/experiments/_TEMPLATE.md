# <ID>: <Title>

<!-- generated:start -->
<!-- generated:end -->

## Question
One sentence. What will we know after this run that we do not know now?

## Why it matters
Where it sits in the [experiment graph](EXPERIMENT_GRAPH.md): what it unblocks, what it is judged against.

## Method: change from the parent experiment
State only what differs from the experiment it builds on. One change per experiment, so any result difference has one cause.

## Inputs and outputs
| Direction | Artifact | Path |
| --- | --- | --- |

## Decision rule
What result means what, written before running. Differences under the noise floor (about 0.013 CARE, see [S](S_seed_robustness.md)) are not differences.

## Results
| Run id | Seed | CARE | Coverage | Earliness | Reliability | Accuracy | TP/FP/FN/TN |
| --- | --- | --- | --- | --- | --- | --- | --- |

Every row cites a run directory under `results/runs/<ID>/`. No run id, no number.

## Risks and open questions
