# Wind-Turbine: normal-behaviour autoencoders for CARE to Compare, Wind Farm C

Fault detection on wind-turbine SCADA data, evaluated with the CARE score. Two training
strategies are compared under identical preprocessing, architecture and scoring:

| | Arm A: isolated | Arm B: pooled |
|---|---|---|
| Models | 58, one per event dataset | 1, fleet-wide |
| CARE | 0.628 | 0.583 |
| Events detected | 16 of 27 | 7 of 27 |

These numbers are reported in [`docs/MODEL_DOCUMENTATION.md`](docs/MODEL_DOCUMENTATION.md);
the arm notebooks in this repo have no stored outputs, so they are not yet reproducible
from the repo alone. Two historical pooled runs differed by 0.013 CARE; that is
not an established universal noise threshold.

The historical Phase 0 arms C and D were reported at 3 seeds each (CPU; full run artifacts are not included in this checkout):

| | Arm C: pooled, leaked rows removed | Arm D: one model per turbine, leak-free |
|---|---|---|
| CARE, mean ± sd of 3 seeds | 0.326 ± 0.288 | 0.611 ± 0.013 |
| CARE at seed 42 | 0.543 | 0.625 |

D's historical result is close to A, and C's results are sensitive to the three-turbine
validation holdout. Matched reruns are needed to identify the causes: details in
[C](docs/experiments/C_pooled_dedup.md) and [D](docs/experiments/D_local_per_turbine.md).

The updated D notebook has now completed a Kaggle run using the shared E03 exports
and corrected evaluation eligibility. Seed 42 achieved **CARE 0.639852**, detecting
**14 of 27 anomalies**, with **6 false alarms** across 31 normal events. All 22
clients and 58 events completed. The [saved run evidence](results/evidence/D/20261015T014242Z_2525b30e_s42/README.md)
includes the executed notebook and extracted event table.

This is one run of the new pipeline; the earlier table remains historical.
**Federated implementation and an exploratory pilot can proceed**, with restricted
SGD-equivalence, cohort/metric-parity checks and a matched centralized comparator
required before interpreting results. See the [transition assessment](docs/FEDERATED_TRANSITION_2026_10_15.md).

**Key finding:** the published data duplicates rows across event files. 25.4% of the pooled
arm's evaluation rows were also in its training set. The evidence is in
[`results/leakage_audit/`](results/leakage_audit/). **Next step:** federated learning with
one client per turbine; see [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Experiments

All experiments, from Arm A to the planned federated arms, are defined in
[`docs/experiments/`](docs/experiments/README.md) and connected in the
[experiment graph](docs/experiments/EXPERIMENT_GRAPH.md). The single source of truth is
[`experiments/registry.yaml`](experiments/registry.yaml); `make check` verifies the repo
against it. What was wrong with the original layout and how it is fixed:
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Layout

```
src/windturbine/   reusable code (client data, config, scoring, audit, runs, paths; models/federated still partial)
experiments/       registry.yaml: every experiment, its files and its dependencies
configs/           YAML run configs (base + per arm)
notebooks/         00 cache, 01 EDA, 02 leakage audit, 10 isolated, 11 pooled, 20 federated clients
results/           leakage_audit/ evidence; runs/ for new outputs (git-ignored)
docs/              model documentation, roadmap, architecture, ADRs, companion PDF
scripts/           registry check, graph renderer, run comparison, synthetic data
tests/             pytest suite
```

## Setup

```bash
pip install -e ".[dev]"      # tests, lint
pip install -e ".[train]"    # adds TensorFlow etc. for the notebooks
make check test
```

For Kaggle, run [`notebooks/20_federated_clients.ipynb`](notebooks/20_federated_clients.ipynb)
on CPU with the source dataset and the packaged bridge attached. It builds immutable,
verified clients; the updated Arm D loads those exact arrays. See the
[Kaggle instructions and federated handoff](docs/CLIENT_BRIDGE.md). Build the source
bundle with `python scripts/package_kaggle_bridge.py`; internet is optional.

The data is the CARE to Compare Wind Farm C set (Kaggle:
`azizkasimov/wind-turbine-scada-data-for-early-fault-detection`). Set `WT_DATA_ROOT` or `dataset_root` in `configs/base.yaml`; the notebooks still contain the Kaggle path until they adopt `windturbine.paths`.

## Status

Restructure in progress (roadmap phase R). Done and tested: `scoring`, `audit`, `config`, `runs`, `paths`, the experiment registry and its checks.
The extracted functions were checked against the notebook originals on random inputs.
The E03 data bridge and D integration are implemented and tested locally. The supplied
Kaggle execution records verified loading of all 22 clients and completed the new D
baseline at seed 42. Full-precision run artifacts still need archiving; multiple-seed
matched comparisons and federated training remain outstanding.
Not done: extraction of the older A/B/C data/model code, `runners`, `federated`, a pinned environment (the TensorFlow
version used was never recorded), and a LICENSE (the owner must choose one).
