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
What does performance look like along the whole dial from local-only to fully shared training?

## Why it matters
FedProx adds a penalty (mu/2)·‖θ − θ_global‖² to each client's loss. At large mu every client is pinned to the global model (F1, close to C); at mu = 0 with enough local epochs clients barely interact (close to D). The companion PDF says the shape of that curve is the paper. Correction to the PDF: its endpoints are A and B, but with turbine clients and leak-free data they are D and C.

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
| Interior maximum above both ends | Partial sharing wins; report the best mu and compare it with F2. |
| Monotone towards D | Collaboration only hurts; the local model is the answer for this farm. |
| Flat | mu does not matter at this scale; F2's architecture split is the only lever. |

## Results
Not run.

## Risks and open questions
- Cost: 6 grid points × 3 seeds = 18 federated runs. Run the grid at one seed first.
