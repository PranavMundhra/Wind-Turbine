# Arm D Kaggle Run Evidence

Run: `20261015T014242Z_2525b30e_s42`, seed 42, 15 October 2026.

This folder preserves the supplied executed `arm-d.ipynb` and values extracted from its saved output. All 22 turbines completed verified client loading and local training. All 58 events were scored. The notebook contains no Python error outputs, and its 27 recorded source-file hashes match the source used in this workspace when the evidence was imported.

| Metric | Printed value |
| --- | --- |
| CARE | 0.639852 |
| Coverage | 0.369207 |
| Earliness | 0.294633 |
| Reliability | 0.654206 |
| Accuracy | 0.940607 |
| TP / FP / FN / TN | 14 / 6 / 13 / 25 |
| Training time | 109.584875 minutes |

The run used 1,845,037 fitting rows, 347,569 validation rows, and 158,528 prediction rows. Status-only evaluation included 139,647 prediction rows. Each of the 27 anomalous events supports Coverage and Earliness; all 31 normal events support Accuracy. Thresholds ranged from 9.352313 to 16.971849.

The E03 dataset manifest hash printed by the notebook is:

```text
bb2ed3ffc84adbf79bbba624941090acd91831097f427b38b2e20885d4b0b1d1
```

## Files and precision

- [arm-d.executed.ipynb](arm-d.executed.ipynb): exact bytes of the supplied notebook, including code, execution timestamps and outputs.
- [results_display.json](results_display.json): the six-decimal printed metrics and exact printed confusion/support counts.
- [per_event_display.csv](per_event_display.csv): the complete 58-row printed event table. Numeric components have display precision, not original full precision.
- [training_counts_and_thresholds.csv](training_counts_and_thresholds.csv): the 22 printed fitting/validation counts and thresholds. This is not the original fitting log; epochs and validation losses were not printed.
- [source_identity.json](source_identity.json): source identity copied from notebook output.
- [evidence.json](evidence.json): checksums, dataset identity, execution metadata, consistency checks, totals and missing-artifact inventory.

Recomputing CARE from the displayed event table gives 0.6398517733471789, consistent with the printed 0.639852 within an absolute tolerance of 0.000001. The event detections also agree with the criticality threshold of 72.

The original run directory, full-precision result files, row scores, model contract and actual E03 payloads are not in this folder. Do not treat this evidence capture as a replacement for those artifacts or mix its rounded tables into full-precision run comparisons. Keep future FL runs tied to the exact E03 hash above.

For the current readiness decision, see [the transition assessment](../../../../docs/FEDERATED_TRANSITION_2026_10_15.md).
