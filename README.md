# Federated normal-behaviour models on CARE to Compare

[![ci](https://github.com/PranavMundhra/Wind-Turbine/actions/workflows/ci.yml/badge.svg)](https://github.com/PranavMundhra/Wind-Turbine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code and results for *Federated normal-behaviour models for early fault detection on wind turbine SCADA
data* (P. Mundhra, D. Poojary, A. Sharma, P. K. Loharkar; in preparation).

The paper gives a federated baseline on Wind Farm C of the public
[CARE to Compare](https://doi.org/10.3390/data9120138) benchmark: 22 turbines, one federated client each,
58 events scored with the CARE score. It also reports that individual rows are duplicated across event files
of the same turbine, so every client dataset here excludes any training row that also appears in a
prediction window.

**Main result (seed 42):** no federated setting scored higher than local training (CARE 0.640). FedAvg
and FedProx with μ ≤ 0.1 were within 0.008 of it; stronger proximal regularisation scored lower.

## Contents

- [Quick start](#quick-start)
- [Results](#results)
- [Reproducing the paper](#reproducing-the-paper)
- [Multi-seed runs](#multi-seed-runs)
- [Where each part of the paper lives](#where-each-part-of-the-paper-lives)
- [Notes on the seed-42 results](#notes-on-the-seed-42-results)
- [Citation and licence](#citation-and-licence)

## Quick start

Rebuild Table I and Fig. 4 from the committed run files. This needs no GPU and no data download:

```bash
git clone https://github.com/PranavMundhra/Wind-Turbine.git && cd Wind-Turbine
pip install -e ".[dev]" matplotlib
python -m windturbine summarise --runs results/paper --out results/paper   # Table I + bootstrap intervals
python -m windturbine figures                                               # Fig. 4
```

The outputs are `results/paper/table_I.md`, `table_I.csv`, `bootstrap_vs_D.csv` and
`figures/fig4_validation_and_care.pdf`.

## Results

Seed 42, 58 events (27 with an anomaly, 31 normal). Higher is better for every score.

| Approach | Experiment | CARE | Coverage | Earliness | Reliability | Accuracy | Detected / 27 | False alarms / 31 |
|---|---|---|---|---|---|---|---|---|
| Local training (D) | `D` | 0.640 | 0.369 | 0.295 | 0.654 | 0.941 | 14 | 6 |
| Centralised comparator¹ | `central_sequential` | 0.640 | 0.316 | 0.244 | 0.723 | 0.960 | 12 | 2 |
| FedAvg (F1) | `F1` | 0.636 | 0.353 | 0.294 | 0.654 | 0.940 | 14 | 6 |
| Personalised (F2) | `F2` | 0.625 | 0.296 | 0.251 | 0.663 | 0.958 | 11 | 3 |
| FedProx (F3), μ = 0 | `F3_mu0` | 0.636 | 0.347 | 0.287 | 0.657 | 0.945 | 13 | 5 |
| FedProx (F3), μ = 0.001 | `F3_mu0.001` | 0.632 | 0.337 | 0.299 | 0.657 | 0.933 | 13 | 5 |
| FedProx (F3), μ = 0.01 | `F3_mu0.01` | 0.636 | 0.351 | 0.309 | 0.654 | 0.933 | 14 | 6 |
| FedProx (F3), μ = 0.1 | `F3_mu0.1` | 0.639 | 0.354 | 0.292 | 0.657 | 0.945 | 13 | 5 |
| FedProx (F3), μ = 1 | `F3_mu1` | 0.616 | 0.266 | 0.188 | 0.667 | 0.979 | 10 | 2 |
| FedProx (F3), μ = 10 | `F3_mu10` | 0.590 | 0.240 | 0.146 | 0.604 | 0.980 | 11 | 5 |

¹ The comparator as run for the paper; see [Notes](#notes-on-the-seed-42-results). Use `central` for new runs.

**Exploratory private variant (F4).** F2 with simulated secure aggregation and record-level DP-SGD
(clip norm 1, δ = 10⁻⁶, RDP accountant). With noise, the models barely trained, so these rows give no
usable relation between privacy and CARE.

| Noise multiplier σ | Experiment | CARE | ε |
|---|---|---|---|
| 0 (secure aggregation only) | `F4_sigma0` | 0.614 | – |
| 0.5 | `F4_sigma0.5` | 0.590 | 22.2 |
| 1 | `F4_sigma1` | 0.590 | 2.82 |
| 2 | `F4_sigma2` | 0.596 | 1.05 |
| 4 | `F4_sigma4` | 0.596 | 0.47 |

**Uncertainty.** These are single-seed results. `bootstrap_vs_D.csv` gives a 95% interval for each
CARE difference from D, resampling turbines rather than events (the 58 events come from 22 turbines,
so events of one turbine are not independent). At seed 42 every interval contains zero.

## Reproducing the paper

### 1. Install

Python 3.10 or later. Training needs PyTorch: install the CUDA build for your GPU from
[pytorch.org](https://pytorch.org) first, then:

```bash
pip install -e ".[train,dev]"
python -m pytest -q          # about 3 minutes on CPU; training tests use tiny synthetic data
```

Keras 3 runs on the PyTorch backend, which the package selects automatically, so training uses the GPU on
Windows as well as Linux. `requirements-lock.txt` lists the exact versions of the seed-42 runs
(torch 2.11.0 with CUDA 12.8, Keras 3.15.1, Flower 1.40.0).

### 2. Get the data

CARE to Compare is published on Zenodo under CC BY-SA 4.0 and is not redistributed here:
[doi:10.5281/zenodo.14006163](https://doi.org/10.5281/zenodo.14006163). Download and unpack it.

### 3. Build the leak-free clients

```bash
python -m windturbine build-clients --raw /path/to/CARE_To_Compare --out /path/to/clients
```

`--raw` is the folder that contains `Wind Farm C`. The command writes one partition per turbine and prints
the SHA-256 of `dataset_manifest.json`. Every experiment pins the expected hash (`configs/paper/_dataset.yaml`)
and stops before training if it differs. To use another export on purpose, add
`--dataset-sha <hash> --dataset-reason "<why>"` to `run`; the reason is recorded in each run's manifest.

### 4. Run the experiments

```bash
python -m windturbine list                                              # the 16 experiments
python -m windturbine run all --seeds 42 --clients /path/to/clients --jobs 3
```

Each run is written to `results/runs/<experiment>/s<seed>/` (`results/runs/` is git-ignored). `all`
covers every row of Table I with the corrected comparator `central`; name `central_sequential` explicitly to
reproduce the paper's comparator. `--jobs 3` runs three experiments at once, each in its own process,
with logs in `results/runs/logs/`.

Before a long batch, check the environment with two rounds of everything (a few minutes):

```bash
python -m windturbine run all --seeds 0 --clients /path/to/clients --out results/smoke --rounds 2 --set max_epochs=2
```

### 5. Build Table I and Fig. 4

```bash
python -m windturbine summarise --runs results/runs --out results/paper
python -m windturbine figures
```

`summarise` copies the small files of each finished run into `results/paper/` (no weights or per-row scores)
and rebuilds the tables from them. Commit `results/paper/` to publish new runs.

The `Makefile` wraps these steps: `make seeds CLIENTS=/path/to/clients`, `make table`, `make figures`,
`make smoke`.

## Multi-seed runs

The client partitions are fixed by the export, so a seed changes only the model initialisation and the
minibatch order. Runs that already exist are skipped, so an interrupted batch resumes when you run the same
command again.

```bash
# Priority subset: the rows the paper's comparisons rest on
python -m windturbine run D central F1 F2 F3_mu0.1 --seeds 42 43 44 45 46 --clients /path/to/clients --jobs 3

# Then the remaining rows
python -m windturbine run all --seeds 43 44 45 46 --clients /path/to/clients --jobs 3

python -m windturbine summarise --runs results/runs --out results/paper
```

With several seeds, `table_I.md` reports mean ± SD per experiment, and `bootstrap_vs_D.csv` averages
each experiment over its seeds before taking the difference from D.

**Runtime.** On one RTX 5070 Ti with three runs in parallel, a single run took 0.9–2.1 h. FedProx is the
slowest because it trains two local epochs per round. One seed of all experiments is about 20 run-hours,
roughly 7 hours of wall time with `--jobs 3`; the priority subset is roughly 2 hours of wall time per seed.

## Where each part of the paper lives

| Paper | Code | Config / results |
|---|---|---|
| Leakage audit (Sec. III, Fig. 1) | `src/windturbine/audit/`, `notebooks/02_leakage_audit.ipynb` | `results/leakage_audit/` |
| Leak-free clients (Sec. IV) | `src/windturbine/data/clients.py` | `configs/clients.yaml` |
| Autoencoder and training step | `src/windturbine/models/autoencoder.py` | `configs/base.yaml` |
| Local training (D) | `src/windturbine/local.py` | `configs/paper/D.yaml` |
| FedAvg (F1), centralised comparator | `src/windturbine/federated/fedavg.py` | `configs/paper/F1.yaml`, `central.yaml` |
| FedPer (F2), FedProx (F3), private (F4) | `src/windturbine/federated/strategies.py` | `configs/paper/F2.yaml`, `F3_mu*.yaml`, `F4_sigma*.yaml` |
| CARE score and thresholds | `src/windturbine/scoring/` | – |
| Table I, bootstrap, Fig. 4 | `src/windturbine/summary.py` | `results/paper/` |
| Command line | `src/windturbine/__main__.py`, `experiments.py` | – |

Each committed run folder in `results/paper/<experiment>/s<seed>/` contains `results.json` (CARE and its
components), `per_event.csv` (decision and scores per event), `manifest.json` (full config, dataset hash,
package versions, runtime), the training curve (`rounds.csv` or `epochs.csv`) and the per-turbine
thresholds. F4 runs add `privacy.json`.

## Notes on the seed-42 results

- **D and F1–F4 were trained in different environments.** D was trained on Kaggle with TensorFlow, on
  client export `bb2ed3ff…`; its `results.json` and `per_event.csv` were copied from notebook output at
  six decimals (`results/paper/D/s42/evidence.json`). F1–F4 were trained on an RTX 5070 Ti with Keras 3 on
  PyTorch, on a local rebuild of the export (`1a43906d…`) with identical per-turbine row counts. New D runs
  use the same PyTorch training step as F1–F4, so the multi-seed results do not have this difference.
- **The paper's centralised comparator drifted.** In `central_sequential`, each epoch trains on one turbine
  at a time in random order, so the model moves toward the last turbine of the epoch. Its validation error
  was lowest at epoch 2 (MSE 0.50) and rose to 0.91 by epoch 100, against about 0.41 for FedAvg.
  `central` shuffles minibatches across all turbines and is the comparator to use from now on.

The earlier project history (arms A–C, Kaggle notebooks, planning documents) is in commit `33e11e6`, and
the exact code of the seed-42 runs is commit `73ffab0`.

## Citation and licence

Citation metadata is in [`CITATION.cff`](CITATION.cff); please cite the paper once it is published. Please
also cite the dataset:

> C. Gück, C. M. A. Roelofs and S. Faulstich, "CARE to Compare: A real-world benchmark dataset for early
> fault detection in wind turbine data," *Data*, vol. 9, no. 12, art. no. 138, 2024,
> doi: 10.3390/data9120138.

The code is released under the [MIT licence](LICENSE). The CARE to Compare data keep their own
CC BY-SA 4.0 licence.
