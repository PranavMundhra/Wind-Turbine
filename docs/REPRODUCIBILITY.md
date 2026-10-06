# Reproducibility: what was wrong and what fixes it

The original repo had three kinds of problem: structure (copied code, no package), reproducibility (no recorded runs, environment or seeds) and connection (files that depended on each other without saying so). This page lists each problem, its fix, and whether the fix is in place.

| # | Problem in the original repo | Fix | Where | Status |
| --- | --- | --- | --- | --- |
| 1 | 26 of 27 functions copy-pasted between the isolated and pooled notebooks | Shared package; notebooks call it | `src/windturbine/` ([ADR 0003](adr/0003-notebooks-orchestrate-library-defines.md)) | Partial: `scoring`, `audit`, `config`, `runs`, `paths` done; `data`, `models` to extract |
| 2 | Kaggle path hard-coded in every notebook | `resolve_data_root()`: argument → `WT_DATA_ROOT` → config → Kaggle autodetect | `src/windturbine/paths.py` | Done; notebooks to adopt |
| 3 | Pooled notebook silently reads `/kaggle/working/results_isolated.json` from Arm A | Comparisons read run directories | `scripts/compare_runs.py` | Done |
| 4 | Headline numbers exist only in prose; arm notebook outputs cleared | Run contract: `manifest.json`, `config.yaml`, `results.json`, `per_event.csv`, `scores.parquet` per run | `src/windturbine/runs.py` | Done; notebooks to adopt |
| 5 | TensorFlow version never recorded | Manifest records package versions and git SHA; pin `requirements.lock` after the next Kaggle run | `runs.write_manifest` | Done (recording); pin pending |
| 6 | Run-to-run noise of 0.013 CARE from cuDNN | `set_seeds(seed, deterministic=True)` enables TensorFlow op determinism when available | `runs.set_seeds` | Done; untested on GPU |
| 7 | Docs linked a `figs/` folder and scripts that did not exist | Every relative markdown link checked in CI | `scripts/check_registry.py` | Done |
| 8 | No list of experiments; nothing tied notebooks, configs and docs together | Experiment registry; every notebook's first cell names its experiment ID | `experiments/registry.yaml` | Done |
| 9 | Client builder disconnected from the arms (own CSV parsing, own split, no feature schema, no leak removal) | Spec rewritten as E03 | [E03](experiments/E03_client_dataset.md) | Specified; not implemented |
| 10 | EDA outputs consumed by nothing; feature choices live in code | `feature_schema.json` as an explicit artifact used by all arms and clients | [E01](experiments/E01_eda.md) | Specified |
| 11 | Three audit outputs written by the notebook are missing from the repo | Re-run E02 with a manifest and commit them | [E02](experiments/E02_leakage_audit.md) | Open |
| 12 | Single seed per arm | 3+ seeds and turbine-level bootstrap | [S](experiments/S_seed_robustness.md) | Specified |
| 13 | Per-row scores never saved, so every threshold or earliness change needs retraining | `scores.parquet` in the run contract | `runs.py` docstring, [X](experiments/X_ablations.md) | Specified |

## Reproducing a run

```bash
pip install -e ".[train]"
export WT_DATA_ROOT=/path/to/care-to-compare   # or set dataset_root in configs/base.yaml
make check                                      # registry, links, graph freshness
# then run the experiment's notebook or runner; it writes results/runs/<ID>/<run id>/
python scripts/compare_runs.py --pairs B-A C-B
```

A run is reproducible when its `manifest.json` shows a clean git tree (`"dirty": false`), and re-running at that commit with its `config.yaml` and the same data manifest hash gives the same `results.json`. With op determinism this should be exact; without it, within the S noise band.
