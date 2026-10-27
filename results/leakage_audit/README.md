# Leakage audit evidence

Outputs of `notebooks/02_leakage_audit.ipynb` (eight tests plus a trivial-match audit).
`evidence_summary.json` summarises them: 49 of 49 within-turbine pairs with enough fingerprint matches (of 64 tested) verified as
duplicates, zero cross-turbine false positives, rows byte-identical across 226 columns,
time shifts of -365, 0 and +365 days. Figures are in `figures/`.

## Share of prediction rows seen in pooled training data

In the preliminary pooled experiment (one model for the whole farm, every event's
normal training rows pooled), the fingerprints of every training row were collected
and every event's prediction rows tested against them:
**40,216 of 158,528 prediction rows (25.4%)** were present in the pooled training
data, across 27 of 58 events. This is the "approximately one quarter" quoted in
Section III of the paper. It was recorded from notebook output of that experiment
(see `docs/MODEL_DOCUMENTATION.md` in commit `33e11e6`); the per-turbine
client export used by every paper run removes all such rows before training.
