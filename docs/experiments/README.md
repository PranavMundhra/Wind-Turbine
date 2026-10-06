# Experiments

Every experiment has an ID, an entry in [`experiments/registry.yaml`](../../experiments/registry.yaml), a doc in this folder, a config in [`configs/`](../../configs/) and, once run, run directories under `results/runs/<ID>/`. The table and the [graph](EXPERIMENT_GRAPH.md) are generated from the registry by `make graph`; CI fails if they are stale.

<!-- generated:start -->
| ID | Experiment | Phase | Status | Importance | Depends on | Unblocks | Doc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E00 | Data cache | R | done | supporting | none | 11 | [E00_data_cache.md](E00_data_cache.md) |
| E01 | Exploratory data analysis | R | done | supporting | none | 6 | [E01_eda.md](E01_eda.md) |
| E02 | Leakage audit | R | done | gate | none | 9 | [E02_leakage_audit.md](E02_leakage_audit.md) |
| E03 | Leak-free per-turbine client dataset | 0 | partial | gate | E00, E01, E02 | 5 | [E03_client_dataset.md](E03_client_dataset.md) |
| A | Isolated, one model per event | done | done-unverified | baseline | E00 | 2 | [A_isolated_per_event.md](A_isolated_per_event.md) |
| B | Pooled, one fleet-wide model | done | done-unverified | baseline | E00, E02 | 6 | [B_pooled_fleet.md](B_pooled_fleet.md) |
| C | Pooled, leaked rows removed | 0 | planned | gate | B, E02 | 4 | [C_pooled_dedup.md](C_pooled_dedup.md) |
| D | Local per-turbine, no collaboration | 0 | planned | gate | E03 | 3 | [D_local_per_turbine.md](D_local_per_turbine.md) |
| F1 | FedAvg, global model, local thresholds | 2 | planned | gate | E03, C | 3 | [F1_fedavg.md](F1_fedavg.md) |
| F2 | Personalised FL, shared encoder, local decoder | 3 | planned | headline | F1, D | 1 | [F2_personalised_fl.md](F2_personalised_fl.md) |
| F3 | FedProx mu sweep | 4 | planned | headline | F1, D | 1 | [F3_fedprox_sweep.md](F3_fedprox_sweep.md) |
| F4 | Secure aggregation and differential privacy | 5 | planned | supporting | F2, F3 | 0 | [F4_private_fl.md](F4_private_fl.md) |
| S | Seed robustness and confidence intervals | 0 | planned | gate | A, B | 0 | [S_seed_robustness.md](S_seed_robustness.md) |
| X | Ablations | any | planned | supporting | A | 0 | [X_ablations.md](X_ablations.md) |
<!-- generated:end -->

## Rules that keep experiments connected

1. **Register first.** A notebook, config or experiment doc that is not in the registry fails `make check`.
2. **One change per experiment.** Each doc states its change from the parent; two changes at once recreate the A-vs-B confound.
3. **Decision rule before results.** Each doc says what each outcome would mean before the run.
4. **No number without a run id.** Results tables cite `results/runs/<ID>/<run id>/`, whose `manifest.json` records the git commit, config hash, seed, package versions and upstream run ids.
5. **Compare through run directories.** Use `python scripts/compare_runs.py --pairs C-B F2-D`; never read another notebook's working directory.
6. **Noise rule.** A difference counts only if it survives [S](S_seed_robustness.md).

New experiment: copy [`_TEMPLATE.md`](_TEMPLATE.md), add the registry entry, run `make graph check`.
