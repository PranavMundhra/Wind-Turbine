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

**Main result (mean of five seeds):** federated learning scored about the same as local training. Local
training averaged CARE 0.627 ± 0.004; FedAvg averaged 0.634 ± 0.007 and the corrected centralised comparator
0.635 ± 0.013, but no 95% turbine-bootstrap interval for a difference from local training excludes zero
(FedAvg +0.007, interval −0.031 to +0.043). FedPer was lower (0.611 ± 0.019). In the rows that have only seed 42,
stronger proximal regularisation (μ ≥ 1) and any privacy noise scored lower.

**Interactive demo and reproduction guide:** <https://pranavmundhra.github.io/Wind-Turbine/> (source in [`site/`](site/)).

## Contents

- [Quick start](#quick-start)
- [Results](#results)
- [Reproducing the paper](#reproducing-the-paper)
- [Multi-seed runs](#multi-seed-runs)
- [Where each part of the paper lives](#where-each-part-of-the-paper-lives)
- [Notes on the results](#notes-on-the-results)
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

For a guided version, open [`notebooks/reproduce.ipynb`](notebooks/reproduce.ipynb). It shows the same
tables and figure with explanations, the convergence checks behind the notes below, which events change
between local training and FedAvg, and the leakage-audit evidence. It is stored with its outputs, so it can
be read on GitHub without running anything.

## Results

58 events (27 with an anomaly, 31 normal). Rows with five seeds (42–46) show mean ± SD over seeds; the other rows
are seed 42 only. Higher is better for every score.

| Approach | Experiment | Seeds | CARE | Coverage | Earliness | Reliability | Accuracy | Detected / 27 | False alarms / 31 |
|---|---|---|---|---|---|---|---|---|---|
| Local training (D) | `D` | 5 | 0.627 ± 0.004 | 0.372 ± 0.009 | 0.295 ± 0.008 | 0.597 ± 0.019 | 0.936 ± 0.008 | 14.2 ± 0.8 | 8.8 ± 0.8 |
| Centralised comparator (corrected) | `central` | 5 | 0.635 ± 0.013 | 0.327 ± 0.028 | 0.254 ± 0.040 | 0.680 ± 0.037 | 0.958 ± 0.010 | 12.6 ± 1.7 | 3.8 ± 1.3 |
| Centralised comparator, as in the paper¹ | `central_sequential` | 1 | 0.640 | 0.316 | 0.244 | 0.723 | 0.960 | 12.0 | 2.0 |
| FedAvg (F1) | `F1` | 5 | 0.634 ± 0.007 | 0.344 ± 0.010 | 0.298 ± 0.004 | 0.650 ± 0.020 | 0.939 ± 0.005 | 13.8 ± 0.4 | 6.0 ± 0.7 |
| Personalised (F2) | `F2` | 5 | 0.611 ± 0.019 | 0.285 ± 0.024 | 0.230 ± 0.027 | 0.609 ± 0.060 | 0.966 ± 0.011 | 10.4 ± 1.5 | 4.2 ± 1.6 |
| FedProx (F3), μ = 0 | `F3_mu0` | 1 | 0.636 | 0.347 | 0.287 | 0.657 | 0.945 | 13.0 | 5.0 |
| FedProx (F3), μ = 0.001 | `F3_mu0.001` | 1 | 0.632 | 0.337 | 0.299 | 0.657 | 0.933 | 13.0 | 5.0 |
| FedProx (F3), μ = 0.01 | `F3_mu0.01` | 1 | 0.636 | 0.351 | 0.309 | 0.654 | 0.933 | 14.0 | 6.0 |
| FedProx (F3), μ = 0.1 | `F3_mu0.1` | 5 | 0.631 ± 0.007 | 0.351 ± 0.008 | 0.291 ± 0.004 | 0.626 ± 0.021 | 0.943 ± 0.002 | 12.6 ± 0.5 | 5.8 ± 0.4 |
| FedProx (F3), μ = 1 | `F3_mu1` | 1 | 0.616 | 0.266 | 0.188 | 0.667 | 0.979 | 10.0 | 2.0 |
| FedProx (F3), μ = 10 | `F3_mu10` | 1 | 0.590 | 0.240 | 0.146 | 0.604 | 0.980 | 11.0 | 5.0 |

¹ The comparator as run for the paper (one seed); see [Notes](#notes-on-the-results). Use `central` for new runs.

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

**Uncertainty.** `bootstrap_vs_D.csv` gives a 95% interval for each CARE difference from D, resampling turbines
rather than events (the 58 events come from 22 turbines, so events of one turbine are not independent). Each
experiment is averaged over its seeds before the difference from D is taken. Every one of the 15 intervals contains zero, so no ranking is established. Rows that have only seed 42 are compared with
the five-seed mean of D and should be read as indicative. For how CARE itself is computed, see the
[CARE maths](https://pranavmundhra.github.io/Wind-Turbine/#maths) page.

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

CARE to Compare is published on Zenodo under CC BY-SA 4.0 and is not redistributed here. All runs used
**version 6** (9 July 2025): [doi:10.5281/zenodo.15846963](https://doi.org/10.5281/zenodo.15846963). Download and unpack it.

Use v6 or later. Zenodo advises against v2 (`10.5281/zenodo.14006163`), whose timestamps do not match the
event-info files. Wind Farm C's labels are the same across versions (27 anomaly and 31 normal events);
benchmark-wide, v6 has 45 anomaly and 50 normal datasets, because v5 relabelled one Wind Farm A event.

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

The committed results already include seeds 42–46 for D, central, F1, F2 and F3_mu0.1.
With several seeds, `table_I.md` reports mean ± SD per experiment, and `bootstrap_vs_D.csv` averages
each experiment over its seeds before taking the difference from D.

**Runtime.** On one RTX 5070 Ti with three runs in parallel, a single run took 0.9–2.1 h. FedProx is the
slowest because it trains two local epochs per round. One seed of all experiments is about 20 run-hours,
roughly 7 hours of wall time with `--jobs 3`; the priority subset is roughly 2 hours of wall time per seed.

## Where each part of the paper lives

| Paper | Code | Config / results |
|---|---|---|
| Exploratory analysis behind the feature choice (Sec. III) | `notebooks/01_eda.ipynb` | – |
| Leakage audit (Sec. III, Fig. 1) | `src/windturbine/audit/`, `notebooks/02_leakage_audit.ipynb` | `results/leakage_audit/` |
| Leak-free clients (Sec. IV) | `src/windturbine/data/clients.py` | `configs/clients.yaml` |
| Autoencoder and training step | `src/windturbine/models/autoencoder.py` | `configs/base.yaml` |
| Local training (D) | `src/windturbine/local.py` | `configs/paper/D.yaml` |
| FedAvg (F1), centralised comparator | `src/windturbine/federated/fedavg.py` | `configs/paper/F1.yaml`, `central.yaml` |
| FedPer (F2), FedProx (F3), private (F4) | `src/windturbine/federated/strategies.py` | `configs/paper/F2.yaml`, `F3_mu*.yaml`, `F4_sigma*.yaml` |
| CARE score and thresholds | `src/windturbine/scoring/` | [CARE maths](https://pranavmundhra.github.io/Wind-Turbine/#maths) |
| Table I, bootstrap, Fig. 4 | `src/windturbine/summary.py`, `notebooks/reproduce.ipynb` | `results/paper/` |
| Command line | `src/windturbine/__main__.py`, `experiments.py` | – |

Each committed run folder in `results/paper/<experiment>/s<seed>/` contains `results.json` (CARE and its
components), `per_event.csv` (decision and scores per event), `manifest.json` (full config, dataset hash,
package versions, runtime), the training curve (`rounds.csv` or `epochs.csv`) and the per-turbine
thresholds. F4 runs add `privacy.json`.

## Notes on the results

- **Local training was rerun at every seed.** D was first run on Kaggle with TensorFlow, on client export
  `bb2ed3ff…` (CARE 0.640 at seed 42; the notebook output is kept in `results/paper/D/s42/evidence.json`). Table I
  now uses reruns of D at seeds 42–46 with the same PyTorch training step as the federated runs, on a local rebuild
  of the export (`1a43906d…`, identical per-turbine row counts). Seed 42 alone gives 0.632 with that step. The
  federated runs were trained on an RTX 5070 Ti with Keras 3 on PyTorch.
- **The paper's centralised comparator drifted.** In `central_sequential`, each epoch trains on one turbine
  at a time in random order, so the model moves toward the last turbine of the epoch. Its validation error
  was lowest at epoch 2 (MSE 0.50) and rose to 0.91 by epoch 100, against about 0.41 for FedAvg.
  `central` shuffles minibatches across all turbines and is the comparator to use from now on.
- **Only five rows have five seeds.** D, `central`, F1, F2 and F3_mu0.1 do; the rest of the FedProx sweep and the
  privacy runs are seed 42 only.

The earlier project history (arms A–C, Kaggle notebooks, planning documents) is in commit `33e11e6`, and
the exact code of the first seed-42 runs is commit `73ffab0`.

## Citation and licence

Citation metadata is in [`CITATION.cff`](CITATION.cff); please cite the paper once it is published. Please
also cite the dataset:

> C. Gück, C. M. A. Roelofs and S. Faulstich, "CARE to Compare: A real-world benchmark dataset for early
> fault detection in wind turbine data," *Data*, vol. 9, no. 12, art. no. 138, 2024,
> doi: 10.3390/data9120138.
>
> C. Gück and C. M. A. Roelofs, "Wind turbine SCADA data for early fault detection," Zenodo, version 6,
> Jul. 2025, doi: 10.5281/zenodo.15846963.

The code is released under the [MIT licence](LICENSE). The CARE to Compare data keep their own
CC BY-SA 4.0 licence.
