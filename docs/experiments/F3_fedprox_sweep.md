# F3: FedProx mu sweep

<!-- generated:start -->
| Registry field | Value |
| --- | --- |
| Status | partial |
| Phase | 4 |
| Importance | headline |
| Depends on | [F1](F1_fedavg.md), [D](D_local_per_turbine.md) |
| Judged against | [C](C_pooled_dedup.md), [D](D_local_per_turbine.md), [F2](F2_personalised_fl.md) |
| Unblocks | [F4](F4_private_fl.md) |
| Notebook | [`notebooks/32_fedprox.ipynb`](../../notebooks/32_fedprox.ipynb) |
| Config | [`configs/fed_fedprox.yaml`](../../configs/fed_fedprox.yaml) |
| Runs | `results/runs/F3/` |
<!-- generated:end -->

## Question
How does the strength of the proximal penalty affect federated optimization and CARE?

## Why it matters
FedProx adds a penalty (mu/2)·‖θ − θ_global‖² to each client's loss. With matching local solver and aggregation settings, mu = 0 is FedAvg, not independent local training. Larger mu constrains local updates; it does not guarantee the centralized result or a monotone curve between C and D. Treat C, regenerated D, and the matched centralized comparator as reference methods, not guaranteed endpoints. See the [original FedProx paper](https://proceedings.mlsys.org/paper_files/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf).

## Method: change from F1
Replace FedAvg with FedProx; sweep mu over the grid in `configs/fed_fedprox.yaml` (starting grid 0 to 10, refined around wherever CARE changes fastest). Plot CARE against mu with C and D as horizontal reference lines, and F2 as a third line.

**Implemented** (`run_federated`, `strategy: fedprox`). Each local step adds mu·(w − w_broadcast) to the data gradient, which is the gradient of the proximal term. Every client trains for **two local epochs**, so local drift is large enough for the penalty to matter. With one epoch FedProx barely differs from FedAvg. Flower averages all weights by retained training rows.

With mu = 0 the run is FedAvg at two local epochs. A test checks this bit for bit, and another shows the same engine reproduces F1's FedAvg branch exactly. That mu = 0 run is the sweep's own baseline.

Each mu costs about one full F1 run (100 rounds × 2 epochs), so **one Kaggle session runs one mu**, chosen with `WT_F3_MU`. The notebook's final section verifies every attached session output and writes `care_vs_mu.csv` with the curve.

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
Implemented ([notebook](../../notebooks/32_fedprox.ipynb)); no real-data mu has run. Run mu = 0 first.

## Risks and open questions
- Cost: 6 grid points × 3 seeds = 18 federated runs. Run the grid at one seed first.
