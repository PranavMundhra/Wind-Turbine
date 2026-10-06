# ADR-0002: Scaler and threshold stay local to each client

**Status:** Proposed (to be confirmed by the phase 2 and 3 experiments)
**Date:** 2026-10-03
**Deciders:** Project owner

## Context
The threshold is the 99th percentile of reconstruction error on a turbine's own held-out
healthy rows, so it depends on that turbine's error distribution. Isolated models (Arm A,
CARE 0.628) beat the pooled model (Arm B, 0.583); the pooled model's single wide
manifold lets faults look normal somewhere in the fleet. The companion PDF proposes
sharing what generalises (encoder) and keeping local what does not.

## Decision
Scaler and threshold are always local. The first federated baseline (phase 2) is plain
FedAvg with local thresholds; phase 3 adds a shared encoder with local decoder fine-tuning.

## Options Considered

### Option A: local scaler and threshold
**Pros:** removes per-turbine sensor bias before the shared model; threshold fits the
client's own error; needs no extra communication.
**Cons:** each client needs enough healthy rows to estimate a stable quantile.

### Option B: global scaler (secure-sum of count, sum, sum of squares) and global threshold
**Pros:** one scaling for everyone; simpler server logic.
**Cons:** a single threshold across turbines repeats Arm B's behaviour; adds a secure
aggregation round before training can start.

## Trade-off Analysis
A global threshold is the part of Arm B least likely to transfer. A global scaler is a
reasonable ablation, not the default.

## Consequences
- Easier: personalisation and privacy (fewer statistics leave the client).
- Harder: comparing against the centralised arms needs the same CARE aggregation on integer counts per event.
- Revisit: if local thresholds are unstable on small clients.

## Action Items
1. [ ] Implement per-client threshold selection reusing `scoring.pick_threshold`.
2. [ ] Report per-event TP/FP/FN/TN from each client so the server can compute CARE.
