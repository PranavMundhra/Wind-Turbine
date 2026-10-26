# F1 implementation and council review

F1 now has a Kaggle notebook and a shared training implementation. Two developer
agents worked on the training code and notebook. Four specialist agents reviewed
the design and then the implementation, one specialist at a time. Each made an
independent assessment from the source. They did not run the full turbine study.

The decision is to proceed to an exploratory F1 pilot. The real synthetic
integration checks passed. There is no real-data F1 score in this change, and it
does not approve moving directly to F2 or claiming an improvement.

## What was built

- [30_fedavg.ipynb](../notebooks/30_fedavg.ipynb) runs Flower FedAvg, then a matched
  centralized baseline, from the same frozen E03 client export.
- [fedavg.py](../src/windturbine/federated/fedavg.py) visits every client each
  round. All clients start with the same global weights. Actual retained training
  rows determine each update's weight.
- Both methods start with identical tensors and visit the same training rows for
  the same total passes. Normal validation rows choose the checkpoint. Neither
  method uses prediction labels or prediction scores to select its model.
- Each selected global model sets a separate turbine threshold from that
  turbine's validation scores. Complete event reports must reproduce direct CARE.
- Saved outputs include data/source identity, configuration, package versions,
  logs, row scores, thresholds and reload-checked model weights. A completed run
  is published atomically with checksums and a `COMPLETE` marker.

## Council decisions

| Specialist | Decision from source review | Main reason and limit |
| --- | --- | --- |
| Data and leakage | Accepted; fix verified | Frozen E03 arrays are reused, splits and scalers are not refitted, final source hashes are checked. Alternative exports now need a written reason even when explicitly pinned. Model seed and partition seed are recorded separately. |
| Statistics and evaluation | Go for an exploratory pilot | Both methods have equal row exposure and validation-selection opportunities, including initial weights. Threshold calibration occurs after selection; event metrics and cohort coverage are checked. Paired seeds are still needed for research claims. |
| Federated learning | Go subject to runtime checks | Real Flower aggregation validates every update; local models share the round broadcast and start with fresh Adam. Unequal-client SGD equivalence and model checkpoint readback test the mechanics. |
| Systems and Kaggle | Go after integration validation | Sequential loading, source attachment checks, verified checkpoint reload and atomic publication are implemented. The requested top-level seed alias was added for the existing comparison script. |

The data reviewer found a real gap: a user could select a different export hash
without explaining the change. The implementation now compares against the
accepted D reference and requires an override reason for any alternative. Tests
cover both pinned and unpinned alternatives.

## What the comparison can tell us

The central branch keeps Adam state and processes shuffled client blocks. F1
starts a fresh local Adam for each client and round, then averages the resulting
weights. Equal row exposure controls the training budget but does not make these
updates equal. A score difference measures these complete procedures; it cannot
be attributed to communication alone.

The simple SGD preflight has a narrower purpose. With one full-batch gradient
step from common weights, sample-weighted FedAvg should equal centralized SGD.
It checks unequal client sizes, multiple parameter arrays and several rounds.
It does not prove equality for Adam or multiple local steps.

## Verification

The real notebook integration passed: **3 tests passed, none skipped**. Every
code cell ran against a tiny synthetic export with real TensorFlow and Flower,
including both training branches, preflight, checkpoint readback and artifact
hash checks. Attached-source tests accepted a fresh bundle and rejected an
E03-only bundle. The core F1 suite passed **19 tests**, including a two-round,
two-local-epoch experiment with unequal clients, optimizer-step totals, distinct
model/partition seeds, saved thresholds and checkpoint selection. The existing
suite also passed **53 tests**: **75 tests passed in total, none skipped**.

The tested environment used Python 3.11 on Windows, TensorFlow 2.21.0, Flower
1.40.0, Keras 3.15.1, NumPy 2.4.6, pandas 3.0.6, PyArrow 26.0.0 and protobuf
6.33.6. `pip check` found no broken requirements. TensorFlow's long header paths
required installing into a shorter temporary directory. Kaggle execution still
needs its own real-data validation; local CPU checks do not demonstrate a full
Kaggle GPU run.

Static lint, the experiment registry and notebook compilation pass. The source
bundle includes F1 code, its notebook, configuration and guide. Automated tests
cover fresh attached-bundle loading and rejection of an older E03-only bundle.

## Next real-data gate

Use the E03 export whose manifest hash is recorded in D's evidence. First run a
short pilot, then the declared full budget. Archive the whole completed output
and verify 22 clients, 58 events, checkpoint reload and CARE parity. Compare F1
with its matched central result and D. Review those outputs before developing
later federated experiments. See [FEDAVG.md](FEDAVG.md) for the exact workflow.
