# Leakage audit evidence

Outputs of `notebooks/02_leakage_audit.ipynb` (eight tests plus a trivial-match audit).
`evidence_summary.json` summarises them: 49 of 49 within-turbine pairs with enough fingerprint matches (of 64 tested) verified as
duplicates, zero cross-turbine false positives, rows byte-identical across 226 columns,
time shifts of -365, 0 and +365 days. Figures are in `figures/`.
