# D: Local per-turbine, no collaboration

<!-- generated:start -->
| Registry field | Value |
| --- | --- |
| Status | done |
| Phase | 0 |
| Importance | gate |
| Depends on | [E03](E03_client_dataset.md) |
| Judged against | [A](A_isolated_per_event.md), [C](C_pooled_dedup.md) |
| Unblocks | [F1](F1_fedavg.md), [F2](F2_personalised_fl.md), [F3](F3_fedprox_sweep.md), [F4](F4_private_fl.md) |
| Notebook | [`notebooks/13_arm_local_turbine.ipynb`](../../notebooks/13_arm_local_turbine.ipynb) |
| Config | [`configs/arm_local_turbine.yaml`](../../configs/arm_local_turbine.yaml) |
| Runs | `results/runs/D/` |
<!-- generated:end -->

## Question
How well does each turbine perform without collaborating? This local baseline anchors the comparison with federated training on the same E03 export.

## Method
The current notebook loads the shared E03 artifacts through `load_client_dataset(..., verify=True)`. It does not rebuild splits, reselect features or refit scalers. Each turbine trains the configured autoencoder on its saved fitting rows and selects its threshold from its saved validation rows. Prediction evaluation uses the separate status-only mask and shared CARE evaluator.

All clients use one ordered input schema. Local inactive inputs are zero-filled without dropping dimensions; all outputs contribute to reconstruction loss and scores. Raw float64 fingerprints, global prediction exclusion and training-only angle/scaler decisions define the new preprocessing version. See [the bridge contract](../CLIENT_BRIDGE.md).

The client is a turbine, not an event file. D is a useful independent-training reference, but it is not the zero-mu endpoint of FedProx: zero mu gives FedAvg under matching settings. See [F3](F3_fedprox_sweep.md).

## Current Kaggle result

Run `20261015T014242Z_2525b30e_s42` completed on 15 October 2026, using seed 42. The [executed notebook and extracted tables](../../results/evidence/D/20261015T014242Z_2525b30e_s42/README.md) are preserved in the repository. All 22 turbines completed verified loading and training, and all 58 events were evaluated. There are no Python error outputs. The recorded source hashes match all 27 referenced files in the workspace at import.

| Metric | Printed result |
| --- | --- |
| CARE | **0.639852** |
| Coverage | 0.369207 |
| Earliness | 0.294633 |
| Reliability | 0.654206 |
| Accuracy | 0.940607 |
| TP / FP / FN / TN | **14 / 6 / 13 / 25** |
| Supporting anomaly / normal events | 27 / 31 |
| Training time | 109.584875 minutes |

The model detected 14 of 27 anomaly events. It raised 6 false alarms among the 31 normal events. Fitting used 1,845,037 rows, validation used 347,569, and prediction covered 158,528 rows; 139,647 prediction rows were eligible under the status-only evaluation mask. Local thresholds ranged from 9.352313 to 16.971849.

The recorded E03 manifest SHA256 is:

```text
bb2ed3ffc84adbf79bbba624941090acd91831097f427b38b2e20885d4b0b1d1
```

Recomputing the complete score from the printed per-event table gives CARE 0.6398517733471789, consistent with 0.639852 to display precision. Event decisions match the criticality threshold of 72. These are output-consistency checks, not an independent model retraining.

The evidence includes six-decimal displayed metrics and the displayed event table. The original full-precision `results.json`, `per_event.csv`, `scores.parquet`, `fit_log.csv`, `model_contract.json`, run manifest and E03 payloads are not available locally. Archive them and keep future FL runs tied to the same dataset hash.

## Transition decision

**Proceed to federated implementation and an exploratory pilot.** The shared client bridge has now been exercised through a completed full-cohort local training run. This removes the earlier requirement to demonstrate a real E03-backed D execution.

Before interpreting the federated pilot, pass the restricted full-batch SGD equivalence test, test the exact received event cohort and server CARE parity, and add a centralized comparator using the same transformed fitting partitions and calibration policy. Use actual retained fitting-row counts for aggregation weights. See [the transition assessment](../FEDERATED_TRANSITION_2026_10_15.md).

A single seed does not establish robustness or prove a benefit from collaboration. Historical A/B/C results have different preparation or environments and cannot isolate federation's effect. Repeated paired seeds and turbine-level uncertainty estimates remain part of the research study.

## Inputs and outputs

| Direction | Artifact | Path |
| --- | --- | --- |
| in | Verified E03 partitions, local scalers and event metadata | `WT_CLIENT_DATASET` |
| out | Original results, per-event metrics, row scores, fit log and model contract | `/kaggle/working/results/runs/D/<run>/` |
| preserved here | Executed notebook, displayed metrics/event table, printed training counts and source identity | `results/evidence/D/20261015T014242Z_2525b30e_s42/` |

## Historical results

The previous in-memory pipeline was reported on CPU on 11 and 12 October 2026 from an uncommitted tree above `fd1cfb5`. These values are retained as historical references; they are not repeated runs of the new exported-data pipeline and must not be pooled with it.

| Run id | Seed | CARE | Coverage | Earliness | Reliability | Accuracy | TP/FP/FN/TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20261011T185853Z_2525b30e_s42` | 42 | 0.625 | 0.322 | 0.231 | 0.659 | 0.956 | 12/4/15/27 |
| `20261011T190243Z_605bad76_s43` | 43 | 0.607 | 0.310 | 0.230 | 0.606 | 0.944 | 12/6/15/25 |
| `20261011T190246Z_3e6c8da8_s44` | 44 | 0.600 | 0.331 | 0.236 | 0.556 | 0.939 | 11/7/16/24 |
| **mean and sample sd** | | **0.611 +/- 0.013** | 0.321 | 0.232 | 0.607 | 0.946 | |

The new score is numerically higher than the old seed-42 score, but that difference includes preprocessing, evaluation and execution changes. It does not measure an isolated improvement in the model or the effect of leakage removal.
