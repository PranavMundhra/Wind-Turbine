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
from the repo alone. Run-to-run noise is about ±0.013 CARE.

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
src/windturbine/   reusable code (config, scoring, audit, runs, paths done; data, models, federated TODO)
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

The data is the CARE to Compare Wind Farm C set (Kaggle:
`azizkasimov/wind-turbine-scada-data-for-early-fault-detection`). Set `WT_DATA_ROOT` or `dataset_root` in `configs/base.yaml`; the notebooks still contain the Kaggle path until they adopt `windturbine.paths`.

## Status

Restructure in progress (roadmap phase R). Done and tested: `scoring`, `audit`, `config`, `runs`, `paths`, the experiment registry and its checks.
The extracted functions were checked against the notebook originals on random inputs.
Not done: `data`, `models`, `runners`, `federated`, a pinned environment (the TensorFlow
version used was never recorded), and a LICENSE (the owner must choose one).
