# Running F1 on Kaggle

F1 trains one shared autoencoder with all 22 turbine clients. It uses Flower's
FedAvg aggregation in one Python process, visiting clients one at a time. This
keeps memory use lower than loading all client matrices together. It is a
simulation: it does not provide distributed execution or a privacy guarantee.

Open [30_fedavg.ipynb](../notebooks/30_fedavg.ipynb). The reusable implementation is
[fedavg.py](../src/windturbine/federated/fedavg.py), with the named model in
[autoencoder.py](../src/windturbine/models/autoencoder.py).

## Kaggle setup

1. Build the source attachment with `python scripts/package_kaggle_bridge.py`.
   Attach `dist/windturbine-client-bridge.zip` as a Kaggle dataset. Rebuild it after
   source changes; an old bridge does not contain F1.
2. Attach the **E03 client export used by the recorded D run**, including its
   `dataset_manifest.json`, schema, client files and checksums. Raw event CSVs or
   legacy caches cannot replace this export. Set `WT_CLIENT_DATASET` to its root
   if automatic discovery cannot find it.
3. Install the federated dependencies before training. The pinned pair is
   `tensorflow==2.21.0` and `flwr==1.40.0`; `pyarrow` is also needed. Locally use
   `pip install -e ".[dev,federated]"`. On Kaggle use an installation cell or an
   attached compatible wheel dataset. The notebook checks dependencies and does
   not silently install or replace packages. Use a fresh session after changing
   TensorFlow. Internet is optional when source, wheels and data are attached.
4. Run every cell. Output goes to a new directory under Kaggle's writable working
   folder. `WT_F1_OUTPUT_ROOT` can select another new directory. Existing outputs
   are never overwritten. Download the entire completed run.

The default [configuration](../configs/fed_fedavg.yaml) uses seed 42, 100 rounds,
one local epoch and every client in every round. `WT_F1_ROUNDS` makes an explicit
short pilot; the chosen budget is saved in the run manifest. Set `WT_F1_SMOKE=1`
for a separate two-client, two-round synthetic check. Synthetic results do not
measure turbine fault detection.

**Runtime.** The full budget is 100 FedAvg rounds plus 100 centralized epochs, each
about 28,800 batch-64 steps over the 1,845,037 retained rows (about 5.8 million
steps in total). Training loops over batches inside one compiled TensorFlow graph
per model. Nothing is published until both branches finish, so a Kaggle timeout
loses the run. After each of the first three rounds the run projects its total
time and stops with a suggested `WT_F1_ROUNDS` if it would pass `max_hours`
(default 11 in the config). Check the printed per-round seconds in the first
minutes of a session.

The default dataset manifest SHA-256 is
`bb2ed3ffc84adbf79bbba624941090acd91831097f427b38b2e20885d4b0b1d1`.
It ties F1 to the saved D evidence. A different export needs an explicit config
change and a nonempty `dataset_override_reason`; it loses that direct D match.
The model seed can change while the exported partition seed stays fixed.

## Training and comparison rules

No splits, features or scalers are fitted again. F1 verifies the export, uses its
exact transformed arrays, and checks input hashes again before publishing. Each
client starts from the same round weights with a fresh Adam optimizer. Flower
weights its update by actual retained training rows. Missing clients, wrong
counts, incompatible parameters and nonfinite values stop the run.

The matched centralized branch starts from identical weights and visits the same
rows for the same total number of passes. It keeps one Adam optimizer and visits
shuffled client blocks. These optimizer updates differ from FedAvg; equal data
exposure does not imply identical training or scores. Historical C has different
preprocessing and is an additional reference only.

Both branches complete their fixed budgets. The lowest validation MSE, weighted
by validation-row count, selects the checkpoint; initial weights are eligible,
ties keep the earliest checkpoint, and centralized selection occurs after each
matching local-epoch budget. There is no prediction-based selection, early
stopping or learning-rate reduction. Each turbine then calibrates its local
99th-percentile threshold using the selected global model and its own validation
rows. Prediction windows are scored after both branches finish training.

The server combines unrounded event components for the complete cohort and
checks its CARE result against direct evaluation. It never averages client CARE.

## Saved evidence and completion

A successful run has a `COMPLETE` marker containing the hash of `manifest.json`.
The manifest records config, source identity, package versions, data hashes,
training counts, row exposures, optimizer steps and selected checkpoints.
Each branch saves results, all event metrics, validation and prediction scores,
thresholds, histories, final weights and selected weights. The selected checkpoint
is loaded into a fresh model and its validation loss checked before scoring.
Root `results.json` and `per_event.csv` mirror the FedAvg branch for the existing
comparison script. Treat directories without `COMPLETE` as unfinished.

The preflight checks actual Flower aggregation against centralized full-batch
SGD on unequal synthetic clients, and checks server/direct CARE parity. This
restricted identity does not prove Adam or multiple-local-step equivalence.
The integration tests also run the complete notebook with real TensorFlow and
Flower on a tiny synthetic E03 export.

## Current decision

The real synthetic notebook integration passed with TensorFlow 2.21.0 and Flower
1.40.0. See the [council report](F1_IMPLEMENTATION_COUNCIL_2026_10_15.md) for
validation and review details. No real-data F1 result is included in this change. After running on
Kaggle, archive the complete outputs, check all 22 clients and 58 events, and
compare F1 with its matched centralized branch and the recorded D baseline.
Run paired seeds before making claims about superiority or equivalence. F2 and
F3 remain pending until the F1 evidence has been reviewed.
