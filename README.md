# Federated normal-behaviour models on CARE to Compare, Wind Farm C

Code and results for the paper *Federated normal-behaviour models for early fault detection on wind turbine
SCADA data* (P. Mundhra, D. Poojary, A. Sharma, P. K. Loharkar; in preparation).

The repository reproduces every number in the paper:

- **The leakage audit** (Section III, Fig. 1): rows duplicated across event files of the same turbine.
- **The leak-free client export** (Section IV): 22 per-turbine clients in which no training row appears
  in any prediction window.
- **Table I and Fig. 4**: local training (D), FedAvg (F1), personalised FedPer (F2), the FedProx sweep
  (F3), the exploratory private variant (F4) and the matched centralised comparator, all scored with CARE.

## Results

`results/paper/` holds the small files of every run behind the paper (results, per-event decisions,
thresholds, training curves, privacy accounting and the full run manifest); model weights and per-row
scores are not committed. Table I is rebuilt from those files with:

```bash
python -m windturbine summarise --runs results/paper --out results/paper
```

Seed 42, as reported in the paper (`results/paper/table_I.md`):

| Experiment | CARE | Coverage | Earliness | Reliability | Accuracy | Detected / 27 | False alarms / 31 |
|---|---|---|---|---|---|---|---|
| Local training (D) | 0.640 | 0.369 | 0.295 | 0.654 | 0.941 | 14 | 6 |
| Centralised comparator, sequential order | 0.640 | 0.316 | 0.244 | 0.723 | 0.960 | 12 | 2 |
| FedAvg (F1) | 0.636 | 0.353 | 0.294 | 0.654 | 0.940 | 14 | 6 |
| Personalised (F2) | 0.625 | 0.296 | 0.251 | 0.663 | 0.958 | 11 | 3 |
| FedProx μ = 0 / 0.001 / 0.01 / 0.1 | 0.636 / 0.632 / 0.636 / 0.639 | | | | | | |
| FedProx μ = 1 / 10 | 0.616 / 0.590 | | | | | | |
| Private (F4), σ = 0 / 0.5 / 1 / 2 / 4 | 0.614 / 0.590 / 0.590 / 0.596 / 0.596 | | | | | | |

F4's ε (δ = 10⁻⁶, RDP accountant) is 22.2, 2.82, 1.05 and 0.47 for σ = 0.5, 1, 2 and 4
(`results/paper/F4_sigma*/s42/fedper_private/privacy.json`).

`summarise` also writes `bootstrap_vs_D.csv`: 95% intervals for each CARE difference from D, resampling
turbines rather than events (the 58 events come from 22 turbines). At seed 42 every interval contains zero.

### Provenance and two caveats

- **D** was trained on Kaggle (TensorFlow, client export `bb2ed3ff…`); its committed `results.json`
  and `per_event.csv` were copied from notebook output at six decimals (`results/paper/D/s42/evidence.json`).
  **All other rows** were trained on one RTX 5070 Ti with Keras 3 on PyTorch, on a local rebuild of the
  export (`1a43906d…`) with identical per-turbine row counts. New runs of D use the same PyTorch training
  step as F1–F4 (`src/windturbine/local.py`), so multi-seed runs remove this difference.
- **The comparator in the paper** (`central_sequential`) trained each epoch one turbine at a time in
  random turbine order. The model drifts toward the last turbine of each epoch: its best validation MSE
  came at epoch 2 (0.50), rising to 0.91 by epoch 100. The `central` experiment shuffles minibatches
  across all turbines and is the comparator to use from now on.

## Installation

Python 3.10 or later. Training needs PyTorch; install the CUDA build for your GPU first
(see pytorch.org), then:

```bash
pip install -e ".[train,dev]"
python -m pytest -q
```

`requirements-lock.txt` lists the exact versions of the seed-42 runs. Scoring, the audit and the
summary run without PyTorch (`pip install -e ".[dev]"`).

## Data

CARE to Compare is published on Zenodo under CC BY-SA 4.0:
[doi:10.5281/zenodo.14006163](https://doi.org/10.5281/zenodo.14006163). It is not redistributed here.
Download it and point the commands below at the folder that contains `Wind Farm C/`.

## Reproducing the paper

```bash
# 1. Leak-free per-turbine clients (prints the dataset manifest hash)
python -m windturbine build-clients --raw /path/to/CARE_To_Compare --out /path/to/clients

# 2. Experiments; a run that already exists is skipped, so rerun the same command to resume
python -m windturbine list
python -m windturbine run all --seeds 42 --clients /path/to/clients --jobs 3

# 3. Table I (mean ± SD over seeds, turbine-bootstrap intervals) and Fig. 4
python -m windturbine summarise --runs results/runs --out results/paper
python -m windturbine figures
```

`run` writes `results/runs/<experiment>/s<seed>/` and refuses to overwrite an existing run. Configs are in
`configs/paper/`, one file per row of Table I. If your client export has a different manifest hash from
the pinned one, `run` stops before training; pass `--dataset-sha <hash> --dataset-reason "<why>"`.

**Multi-seed runs.** Partitions are fixed by the export, so seeds change only model initialisation and
minibatch order:

```bash
# Priority subset first: the rows the paper's comparisons rest on
python -m windturbine run D central F1 F2 F3_mu0.1 --seeds 42 43 44 45 46 --clients /path/to/clients --jobs 3
# Then everything else
python -m windturbine run all --seeds 43 44 45 46 --clients /path/to/clients --jobs 3
```

**Runtime.** On one RTX 5070 Ti with three runs in parallel, each run took 0.9–2.1 h (F3 with two local
epochs is the slowest). One seed of all 15 experiments is about 20 run-hours, roughly 7 hours of wall
time with `--jobs 3`. Use `--rounds 2` and `--set max_epochs=2` for a quick check that everything runs.

## Layout

```
src/windturbine/
  audit/            row fingerprints for the leakage audit
  data/clients.py   leak-free per-turbine client export (its hash is pinned in every run)
  models/           autoencoder and the PyTorch training step shared by all experiments
  local.py          D: one model per turbine
  federated/        F1 FedAvg and the centralised comparator; F2-F4 (FedPer, FedProx, DP-SGD, secure sum)
  scoring/          CARE, thresholds, reconstruction error
  experiments.py    config loading and dispatch; __main__.py is the command line
  summary.py        Table I, turbine bootstrap, Fig. 4
configs/            base.yaml, clients.yaml (export), paper/ (one file per Table I row)
notebooks/          02_leakage_audit.ipynb: the audit behind Fig. 1, with its outputs
results/            paper/ (committed run files), leakage_audit/ (audit evidence)
tests/              pytest; the training tests run on CPU with tiny synthetic data
```

The project history, including the earlier arms A–C, Kaggle notebooks and planning documents, is kept
in commit `33e11e6` (tag it `pre-paper-cleanup` if you like); the exact code of the seed-42 runs is commit `73ffab0`.

## Citation

See `CITATION.cff`, or cite the paper once published. Please also cite the dataset:
C. Gück, C. M. A. Roelofs and S. Faulstich, "CARE to Compare: A real-world benchmark dataset for early
fault detection in wind turbine data," *Data*, vol. 9, no. 12, art. no. 138, 2024.

## Licence

Code: MIT (see `LICENSE`). The CARE to Compare data keep their own CC BY-SA 4.0 licence.
