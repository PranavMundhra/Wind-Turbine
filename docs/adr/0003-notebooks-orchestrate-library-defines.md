# ADR-0003: Notebooks orchestrate; the `windturbine` package defines

**Status:** Accepted for the restructure (extraction is in progress)
**Date:** 2026-10-03
**Deciders:** Project owner

## Context
The two arm notebooks were intended to share code "generated from a single source"
(docs/MODEL_DOCUMENTATION.md), but 26 of 27 functions are identical copies, and 7 more are
copied from the cache builder. Every notebook hard-codes a Kaggle path. A fix in one copy
can silently diverge the arms, and federated work would add a third and fourth copy.

## Decision
Move shared logic into `src/windturbine/` (importable, tested, configured by YAML).
Notebooks only call it and display results. Done so far: `scoring` (CARE building blocks,
threshold), `audit` (row fingerprints), `config`. Still in notebooks: `data`, `models`.

## Options Considered

### Option A: shared package + thin notebooks
| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium (one-off extraction) |
| Reproducibility | High: configs and tests |
| Kaggle friendliness | Needs `pip install` of the repo, or upload as a dataset |

### Option B: keep self-contained notebooks
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low now, rising with each new arm |
| Reproducibility | Low: no tests, path edits per run |

## Trade-off Analysis
Self-contained notebooks suit a Kaggle one-off; they do not suit four experiment arms
that must stay comparable.

## Consequences
- Easier: Arm C and federated runs reuse identical preprocessing and scoring.
- Harder: running on Kaggle needs the package available in the session.
- Revisit: extracted functions were verified identical to the notebook originals on random
  inputs, but `data` and `models` still need to be checked against real Farm C data.

## Action Items
1. [ ] Extract `data` and `models`, replacing notebook globals with `Config` fields.
2. [ ] Re-run both arms from configs; accept if CARE is within the documented ±0.013 noise.
3. [ ] Replace the notebook bodies with calls into the package.
