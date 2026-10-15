# E03: Leak-free per-turbine client dataset

<!-- generated:start -->
| Registry field | Value |
| --- | --- |
| Status | done |
| Phase | 0 |
| Importance | gate |
| Depends on | [E00](E00_data_cache.md), [E01](E01_eda.md), [E02](E02_leakage_audit.md) |
| Judged against | none |
| Unblocks | [D](D_local_per_turbine.md), [F1](F1_fedavg.md), [F2](F2_personalised_fl.md), [F3](F3_fedprox_sweep.md), [F4](F4_private_fl.md) |
| Notebook | [`notebooks/20_federated_clients.ipynb`](../../notebooks/20_federated_clients.ipynb) |
| Config | [`configs/clients.yaml`](../../configs/clients.yaml) |
| Runs | `results/runs/E03/` |
<!-- generated:end -->

## Question
Can we build 22 per-turbine partitions that use the arms' exact preprocessing and contain no training row that is evaluated anywhere for that turbine?

## Why it matters
Every federated arm and the updated Arm D consume these partitions. The shared implementation is in `src/windturbine/data/clients.py`; the completed Kaggle D run now records verified consumption of the full 22-client/58-event export. See [Kaggle instructions and artifact contract](../CLIENT_BRIDGE.md).

## Method

The notebook calls a CPU-only shared builder. It selects one CSV variant per event, validates all event metadata, and fails rather than silently skipping events. Defaults require all 58 events and 22 turbines.

Fingerprints use ordered raw float64 sensor values rounded to six decimals, before angles, imputation or scaling, with signed zero normalized. The union of **all** prediction fingerprints is excluded from both fitting and validation. Training copies of a turbine's remaining validation rows are then removed. All three local intersections and global prediction intersections are checked.

The default blocked split follows D's retained-row blocks, not five elapsed calendar days; short events use a recorded chronological fallback. There is no temporal embargo. Angle-range decisions use only cleaned fitting data. Every client retains the same ordered feature dimensions. Local means, scales and inactive-feature masks use cleaned fitting rows only; inactive inputs become zero without dropping columns. Reconstruction loss and anomaly scores include every output dimension, including those zero-filled inputs, consistently in D and the proposed matched FL comparison.

Training and calibration use the status and power filters. Prediction scoring uses a separate status-only mask. Event identities, timestamps and anomaly-window metadata remain attached to prediction rows.

This changes the historical D preprocessing and requires new baseline runs. Legacy E00 caches do not establish the raw float64 measurement identity; the default build reads source CSVs directly instead of silently using them.

## Inputs and outputs
| Direction | Artifact | Path |
| --- | --- | --- |
| in | Event cache, feature schema, fingerprint method | E00, E01, E02 |
| out | Per-client fitting / validation / prediction arrays and raw audit readings | `/kaggle/working/windturbine_clients/clients/<asset_id>/*.npz` |
| out | Common schema, raw-source hashes, per-client counts and overlap checks | `dataset_manifest.json` and `clients/<asset_id>/client.json` |
| out | Successful complete-export manifest hash | `COMPLETE` |

## Decision rule
Accept the real-data gate only after the complete cohort has built and reloaded successfully: common ordered features, nonempty required partitions, zero fitting-validation and fitting/validation-prediction intersections, correct source/artifact hashes, and saved removal counts. Within-partition repeats are retained and counted separately from unique fingerprints. FedAvg weights use the actual retained fitting-row count.

## Results
The exported build and loader are covered by synthetic integration tests, including planted cross-event and cross-client overlap, missing metadata, different missing-feature patterns, output mutation and clean notebook execution.

The supplied Kaggle Arm D run `20261015T014242Z_2525b30e_s42` loaded and verified all 22 exported clients and scored all 58 events. Its executed source matches the current loader and includes raw fingerprint/scaler/schema/mask/overlap verification before each client's training. The complete run therefore supplies real-data consumption evidence for the E03 gate. The printed dataset manifest SHA256 is `bb2ed3ffc84adbf79bbba624941090acd91831097f427b38b2e20885d4b0b1d1`.

See the [preserved run evidence](../../results/evidence/D/20261015T014242Z_2525b30e_s42/README.md). The actual E03 payload and manifest are not available locally for an independent rebuild; archive them for reproducibility and use the identical manifest for FL. The updated D score is 0.639852 at seed 42. This is separate from the old in-memory baseline results.

## Risks and open questions
- Leak removal shrinks some clients; record by how much, because FedAvg weights clients by row count.
