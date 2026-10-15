# F3: FedProx mu sweep

<!-- generated:start -->
| Registry field | Value |
| --- | --- |
| Status | planned |
| Phase | 4 |
| Importance | headline |
| Depends on | [F1](F1_fedavg.md), [D](D_local_per_turbine.md) |
| Judged against | [C](C_pooled_dedup.md), [D](D_local_per_turbine.md), [F2](F2_personalised_fl.md) |
| Unblocks | [F4](F4_private_fl.md) |
| Notebook | none yet |
| Config | [`configs/fed_fedprox.yaml`](../../configs/fed_fedprox.yaml) |
| Runs | `results/runs/F3/` |
<!-- generated:end -->

## Question
How does the strength of the proximal penalty affect federated optimization and CARE?

## Why it matters
FedProx adds a penalty (mu/2)·‖θ − θ_global‖² to each client's loss. With matching local solver and aggregation settings, mu = 0 is FedAvg, not independent local training. Larger mu constrains local updates; it does not guarantee the centralized result or a monotone curve between C and D. Treat C, regenerated D, and the matched centralized comparator as reference methods, not guaranteed endpoints. See the [original FedProx paper](https://proceedings.mlsys.org/paper_files/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf).

## Method: change from F1
Replace FedAvg with FedProx; sweep mu over the grid in `configs/fed_fedprox.yaml` (starting grid 0 to 10, refined around wherever CARE changes fastest). Plot CARE against mu with C and D as horizontal reference lines, and F2 as a third line.

## Inputs and outputs
| Direction | Artifact | Path |
| --- | --- | --- |
| in | Client partitions; F1 code; C, D, F2 results | E03, F1 |
| out | One run directory per mu; `care_vs_mu.csv` and the plot | `results/runs/F3/` |

## Decision rule
| Curve shape | Reading |
| --- | --- |
| A nonzero mu improves on matched FedAvg | The proximal constraint helps under the tested solver and data settings. |
| No mu improves on FedAvg or regenerated D | No benefit was measured for this penalty and search grid; do not infer that all collaboration fails. |
| Flat | No sensitivity was measured in this grid and budget. |

## Results
Not run.

## Risks and open questions
- Cost: 6 grid points × 3 seeds = 18 federated runs. Run the grid at one seed first.
