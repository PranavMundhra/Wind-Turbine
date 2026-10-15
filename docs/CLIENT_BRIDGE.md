# Client Dataset Bridge for Kaggle and Federated Learning

The E03 notebook now builds the client data once. Arm D and future federated code load the same saved partitions. The builder runs on CPU and needs NumPy, pandas and PyYAML. It does not train a model or require TensorFlow.

The complete Farm C build must contain 22 turbines and 58 events. Local tests use a small synthetic dataset. They establish that the code works on controlled cases; they do not replace running the real dataset on Kaggle.

**Run update, 15 October 2026:** the supplied Kaggle Arm D execution completed all
22 clients and 58 events using verified E03 exports, with seed-42 CARE 0.639852.
The [preserved execution evidence](../results/evidence/D/20261015T014242Z_2525b30e_s42/README.md)
supplies the previously pending real-data consumption and D-training evidence.
The [transition assessment](FEDERATED_TRANSITION_2026_10_15.md) now approves starting
federated implementation and an exploratory pilot after its mechanics checks.

## Run the client notebook on Kaggle

1. Upload [20_federated_clients.ipynb](../notebooks/20_federated_clients.ipynb) to a CPU Kaggle notebook.
2. Attach the CARE to Compare source dataset containing `Wind Farm C/datasets`, `event_info.csv` and `feature_description.csv`. Comma-prefixed metadata and event exports are supported too. The notebook finds the farm under `/kaggle/input`; set `WT_DATA_ROOT` only if detection needs help.
3. Attach the supplied `windturbine-client-bridge.zip` as a separate Kaggle Dataset. It contains the exact source, configurations and notebook versions for these changes. The bootstrap supports either the ZIP itself or the source directories if Kaggle extracts it during upload. It checks the file hashes before importing code. Internet can stay off.
4. Run all cells. Outputs go to `/kaggle/working/windturbine_clients`, independently of the read-only source attachment.
5. Check that the final cell reports 22 verified clients and that all overlap counts are zero. Save the notebook version with its output dataset. Keep `dataset_manifest.json`, the client sidecars and the `COMPLETE` file with the arrays.

The source bundle is useful before the code is published. After publishing, an internet-enabled notebook can instead set `WT_REPO_REF` to a commit or tag containing the bridge. A fresh Kaggle session will clone that revision. The old reviewed commit `37e2ea5` does not contain this implementation. No push or publication was performed as part of the local changes.

An existing local checkout also works: open the notebook from that checkout, or set `WT_REPO_ROOT`. Explicit source selectors are authoritative; conflicting selectors or an invalid root cause a clear failure.

## Useful settings

Set these environment variables in a small cell before the bootstrap when needed:

| Setting | Meaning |
| --- | --- |
| `WT_DATA_ROOT` | Source directory containing the farm, or the farm directory itself |
| `WT_REPO_ROOT` | Exact local or attached source checkout containing `src/windturbine/data/clients.py` |
| `WT_REPO_REF` | Published Git commit or tag to clone; requires internet; cannot be combined with `WT_REPO_ROOT` |
| `WT_CLIENT_OUTPUT_ROOT` | New output directory; on Kaggle it must be inside `/kaggle/working` |
| `WT_SEED` | Client split seed during E03; model seed during D |
| `WT_CLIENT_SMOKE=1` | Build two synthetic clients and four events instead of Farm C; only for checking the notebook |

`WT_CLIENT_OUTPUT` is a compatibility alias for `WT_CLIENT_OUTPUT_ROOT`. A successful dataset is immutable: rerunning into the same directory fails rather than appending duplicate rows. Use a new output directory for another build. The notebook never deletes an existing dataset for you.

The default configuration is [base.yaml](../configs/base.yaml) plus [clients.yaml](../configs/clients.yaml). Keep the 22-client and 58-event expectations for the real benchmark. The synthetic smoke mode explicitly substitutes its smaller cohort.

## What the builder changes

The builder chooses one CSV variant for each event and validates complete metadata. Missing or corrupt events cause failure. It processes one turbine at a time and uses temporary disk staging rather than holding the entire farm in memory.

It creates fingerprints from ordered raw float64 sensor values, rounded to six decimal places. Signed zero is normalized. These identifiers are calculated before angles, scaling and missing-value replacement. They detect rounded equal measurements, not all statistical dependence or nearby correlated readings.

The prediction fingerprint union across all clients is removed from both fitting and validation data. Fitting copies of a turbine's retained validation rows are removed too. A separate check rejects cross-client fitting-validation copies, because a future shared model would otherwise train on another client's calibration readings. A rejection should be investigated, not bypassed by disabling checks.

Within-partition repeats are retained. The audit reports total rows and unique fingerprints separately. Future FedAvg weights use actual retained fitting rows, including retained repeats, rather than pre-cleaning counts or unique counts.

The blocked split follows the earlier D policy: blocks count retained healthy rows at six nominal rows per hour. They are not necessarily five elapsed calendar days after filtering. Short events fall back to a recorded chronological split. There is no temporal embargo.

One ordered feature schema is shared by every client. Angle candidates come from metadata and are range-checked using only cleaned fitting rows. Local means, scales and missingness masks also use cleaned fitting rows only. Unsupported dimensions become zero without changing the model width. Healthy constant features use scale one so deviations at prediction time remain visible. All model output dimensions contribute to reconstruction loss and anomaly scores, including outputs for zero-filled inputs. D and the future matched comparator must use this same convention.

Fitting and calibration use the stricter training status/power filter. Prediction evaluation uses a separate status-only mask, so low-power normal-status anomalies are not silently removed from evaluation.

Legacy E00 caches lack the precision and provenance required by the new raw float64 contract. This version deliberately rejects `cache_dir` rather than treating a converted float32 cache as original readings. Use the default raw CSV path. This costs parsing time but avoids an unverified data substitution.

## Saved artifacts

```text
windturbine_clients/
  COMPLETE
  dataset_manifest.json
  clients/
    <asset_id>/
      train.npz
      val.npz
      prediction.npz
      val_fingerprints.npy
      prediction_fingerprints.npy
      client.json
```

Each NPZ contains the finite float32 model matrix `X`, raw float64 sensor readings, canonical fingerprints, event IDs, turbine IDs and timestamps. Prediction data also contain status IDs, `eval_mask` and pointwise ground truth. Each client sidecar contains event windows and labels, the ordered schema, local scaler parameters and removal counts.

The dataset manifest records the cohort, schema and preprocessing hashes, source CSV and metadata hashes, artifact hashes, configuration, software versions, implementation identity and the notebook's source identity. `COMPLETE` contains the final manifest hash. A failed build cannot leave a complete dataset at the requested destination.

Raw audit readings and transformed arrays are both stored, which uses more disk than transformed arrays alone. Temporary raw staging is removed as final clients are exported. Verification loads one client at a time; small fingerprint indexes allow global checks without repeatedly reading the whole farm.

To check a saved export independently:

```python
from windturbine.data.clients import validate_client_dataset, load_client_dataset

manifest = validate_client_dataset("/kaggle/working/windturbine_clients")
client = load_client_dataset("/kaggle/working/windturbine_clients", manifest["clients"][0])
X_train = client["train"]["X"]
n_fit = len(X_train)  # Future FedAvg aggregation weight
```

Loading checks hashes, recomputes raw fingerprints and transformed arrays, reconstructs fitting-only scaler statistics, and rechecks masks and overlaps. It verifies saved-data consistency. It is not a fresh reconstruction of every split decision from the original source CSVs, and hashes are not an authenticity guarantee against somebody replacing all artifacts and manifests together.

## Run the local baseline on these exact clients

Use [13_arm_local_turbine.ipynb](../notebooks/13_arm_local_turbine.ipynb) with the same source bundle. Attach the completed E03 output and set `WT_CLIENT_DATASET` to the folder containing `dataset_manifest.json` if automatic discovery is ambiguous. The notebook also finds E03 data already under `/kaggle/working/windturbine_clients`.

Arm D loads the arrays directly. It does not re-split, reselect features or refit scalers. Its model seed may vary while the recorded preprocessing seed stays fixed. Other preparation settings must match the export. TensorFlow and PyArrow are required for D's training and row-score output; these are included in the project's training extra.

Each run writes a manifest linked to the exact E03 manifest, unrounded per-event metrics, row scores, a fitting log and `model_contract.json`. Model layers have explicit encoder/decoder names and shared parameter shapes are checked across clients. `WT_OUTPUT_ROOT` can select the run directory, inside `/kaggle/working` on Kaggle.

The reusable D notebook remains output-free. Its completed Kaggle run is preserved separately in the evidence folder. Earlier historical CARE values used different preparation and evaluation rules; do not pool them with the new seed-42 result.

## Handoff to the federated implementation

Use the exported fitting counts for update weighting and the fixed schema/model contract for parameter compatibility. The shared `evaluate_care` returns an unrounded event table. Clients may have only normal or only anomalous events, so a client CARE value can legitimately be undefined.

Concatenate the client event tables and call `aggregate_event_metrics` once, passing `expected_event_ids` from the export metadata. The server checks the expected cohort and duplicates, averages supported event components, computes global Reliability and applies CARE's guards. Do not average client CARE or weight evaluation by training rows.

Before interpreting an FL result, implement the restricted full-batch SGD equivalence check and train a matched centralized comparator on the same transformed fitting partitions and calibration sets. The builder does not implement Flower or certify full FL training. The real E03-backed D execution is now recorded; next implement and test the federated mechanics and retain the original full-precision artifacts.

## Local checks and packaging

Before the subsequent Kaggle run, the same four council specialists reviewed the implemented bridge, sequentially:

| Specialist | Implementation decision |
| --- | --- |
| Data integrity | Accepted: canonical raw checks, split separation and common schema are implemented; real cohort execution remains required. |
| Statistics | Accepted: supported fixtures match the prior evaluator, intentional mask/support fixes are explicit, and client/server metric parity passes. |
| Federated algorithms | Accepted: exported arrays, named layers and model contracts support the next pilot; matched centralized training remains future work. |
| Systems | Accepted conditionally: Kaggle source/bootstrap and provenance checks pass locally; full Kaggle execution and D training are not yet demonstrated. |

The complete local suite passes 53 tests, including 18 client/notebook tests and
13 CARE helper/evaluator tests. Lint, registry links and generated graph checks
also pass. The later notebook-backed real-data result is linked above; the council's
implementation review did not itself execute that training.

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check src tests scripts
.\.venv\Scripts\python scripts/check_registry.py
.\.venv\Scripts\python scripts/render_graph.py --check
.\.venv\Scripts\python scripts/package_kaggle_bridge.py
```

The final command creates `dist/windturbine-client-bridge.zip` from the current source. Rebuild it after changing source, configs or the included notebooks. It contains no raw data, run outputs, environment files or credentials. Notebook tests execute every E03 code cell on synthetic data without TensorFlow and exercise offline ZIP bootstrapping and corruption rejection.
