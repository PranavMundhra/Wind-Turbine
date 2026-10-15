# Federated Phase Transition Assessment

Updated 15 October 2026 after the supplied Kaggle Arm D execution.

**Yes: move into federated implementation and a small exploratory pilot.** The real client-data bridge has now supported verified loading and completed training for all 22 turbines and evaluation for all 58 events. The previous requirement to demonstrate an E03-backed D run is satisfied by saved execution evidence.

This does not mark the full statistical study complete. The federated trainer and its restricted gradient-step check are not implemented, the new D baseline has one seed, and a matched centralized comparator remains required before attributing performance differences to federation.

## Evidence that changes the decision

The [preserved notebook](../results/evidence/D/20261015T014242Z_2525b30e_s42/arm-d.executed.ipynb) records run `20261015T014242Z_2525b30e_s42`. All code cells ran in sequence without Python error outputs. Every client was loaded with verification enabled before training. The saved source identity matches all 27 referenced local source/config/notebook files at import, including the loader and evaluator.

The matching loader checks artifact hashes, common schema, raw fingerprints, transformed arrays, fitting-only scaler statistics, evaluation masks, local split intersections, global prediction intersections and cross-client fitting-validation intersections. Completion of all 22 client iterations therefore supports these checks on the consumed export. It does not replace an independent local rebuild from the original data.

| Observed item | Result |
| --- | --- |
| Clients trained | 22 |
| Events scored | 58: 27 anomaly, 31 normal |
| Prediction rows | 158,528 |
| Status-eligible prediction rows | 139,647 |
| Fitting / validation rows | 1,845,037 / 347,569 |
| CARE | 0.639852 |
| Coverage / Earliness | 0.369207 / 0.294633 |
| Reliability / Accuracy | 0.654206 / 0.940607 |
| TP / FP / FN / TN | 14 / 6 / 13 / 25 |
| Training time | 109.584875 minutes |

The 58-row printed event table independently reproduces the displayed components, support, confusion counts and CARE within absolute tolerance `1e-6`. The reconstructed CARE is 0.6398517733471789. All recorded event decisions agree with peak criticality reaching 72.

The printed E03 manifest SHA256 is `bb2ed3ffc84adbf79bbba624941090acd91831097f427b38b2e20885d4b0b1d1`. Use this export identity for the first FL comparison, or explicitly record a new dataset version and regenerate its baseline.

## Stage decisions

| Stage | Decision | Reason |
| --- | --- | --- |
| Shared client-data implementation | Completed | Shared build/load code and integration tests exist. |
| Real-data E03 consumption and D baseline | Completed at seed 42, notebook-backed | All clients passed the loading checks and completed training/scoring. |
| Synthetic federated implementation | Go now | The needed data/model/metric contracts are defined. |
| Small real-data federated pilot | Proceed after mechanics checks | First verify restricted update equivalence, event cohort and server metric parity in the actual FL implementation. |
| Claims that FL helps, matches centralized training, or improves privacy | Not established | Require matched methods, repeated experiments and the relevant privacy mechanism. |

## Work to do next

1. Implement the smallest full-participation FedAvg simulation. Use a common initialization, fixed named parameter order and weights equal to retained fitting-row counts.
2. Verify one full-batch plain-SGD step against centralized gradient descent on the same transformed data, including unequal client sizes. Passing this restricted test does not certify multi-epoch Adam training.
3. Verify exact evaluation event coverage and client/server CARE parity. The existing shared aggregator supports this, but the actual FL reporting path still needs to demonstrate it. Never average client CARE or weight evaluation by training rows.
4. Run a short pilot using the same E03 manifest and model contract as D. Save round logs, optimizer-state policy, local-step budget and checkpoint-selection policy.
5. Add central training on the union of the identical prepared client fitting partitions, with the same local transformations, calibration data and threshold policy. Historical C alone does not isolate the effect of federation.

FL does not need to outperform D to justify running the experiment. A negative result can be useful when the comparison is controlled. FedProx with zero mu remains FedAvg under matching solver settings; its sweep has no guaranteed D/C endpoints.

## Evidence still needed for research conclusions

Archive the original full-precision run directory and E03 manifest/payloads. The repository currently preserves the executed notebook, displayed event table, printed training counts, source identity and an evidence record. These should not be presented as recovered full-precision artifacts.

Repeat the new D and compared methods at paired seeds, separate split randomness from optimization randomness, and estimate uncertainty by resampling turbines while keeping their events together. State a practical equivalence margin in advance if claiming two methods are equivalent. Do not reuse the historical 0.013 difference as a universal noise threshold.

The earlier [council report](FEDERATED_READINESS_COUNCIL_2026_10_15.md) and [implementation notes](CLIENT_BRIDGE.md) describe the conditions before this run. This assessment updates their pending real-data gate; it does not claim that the council reran the Kaggle experiment or that local tests trained these models again.
