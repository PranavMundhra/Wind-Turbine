# ADR-0001: The federated client is the turbine (`asset_id`), not the event file

**Status:** Proposed
**Date:** 2026-10-03
**Deciders:** Project owner

## Context
Farm C has 58 event files from 22 turbines; 19 turbines have more than one file. Rows are
duplicated across files of the same turbine: `results/leakage_audit/evidence_summary.json`
records 49 of 49 within-turbine pairs with enough fingerprint matches verified as duplicates (64 pairs were tested) and zero cross-turbine false
positives. Pooling across events put 25.4% of Arm B's prediction rows into its training
corpus (docs/MODEL_DOCUMENTATION.md section 3).

## Decision
One federated client per `asset_id` (22 clients). Within each client, every training row
whose fingerprint appears in any of that client's prediction windows is removed before
training ([E03](../experiments/E03_client_dataset.md)). `notebooks/20_federated_clients.ipynb`
already partitions by turbine but does not yet remove these rows.

Correction (2026-10-03): an earlier draft said turbine partitioning plus row
deduplication was enough. It is not: the duplication is *within* a turbine (a row that is
`prediction` in one of turbine 53's files is `train` in another), so turbine partitioning
removes cross-client leakage only, and exact-duplicate removal does not remove a row that
appears once in training and once in prediction.

## Options Considered

### Option A: client = `asset_id`
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Leakage risk | Removes cross-client duplicates by construction |
| Realism | Matches an asset owner holding a turbine's data |

**Pros:** no row appears on two clients; cross-silo setting (few clients, reliable).
**Cons:** 22 clients; clients with several events still need within-client leak removal.

### Option B: client = event file
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Leakage risk | High: overlapping files recreate the contamination inside the federation |

**Pros:** 58 clients, simpler mapping to the benchmark.
**Cons:** invalidates evaluation for the same reason Arm B is contaminated.

## Trade-off Analysis
Option B is faster to build and wrong; the project's central finding is why.

## Consequences
- Easier: evaluation stays clean without a separate filter.
- Harder: client sizes are unequal (the largest turbine has 6 events), which affects FedAvg weights.
- Revisit: whether within-client deduplication changes any client's size materially.

## Action Items
1. [ ] Remove training rows seen in the same client's prediction windows (E03).
2. [ ] Test: no client's training fingerprints intersect its own or any other client's prediction fingerprints.
