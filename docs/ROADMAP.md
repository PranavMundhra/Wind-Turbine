# Roadmap

Experiments are defined in [`experiments/`](experiments/README.md) and drawn in the [experiment graph](experiments/EXPERIMENT_GRAPH.md). Phases come from the companion PDF, plus phase R (this restructure) and Arm D, which the original plan did not have. The PDF put client repartitioning in phase 1; it moves to phase 0 as E03 because Arm D needs it.

| Phase | Experiments | Exit test | Status |
|---|---|---|---|
| R | [E00](experiments/E00_data_cache.md), [E01](experiments/E01_eda.md), [E02](experiments/E02_leakage_audit.md); extract `data` and `models` into the package | A and B re-run from configs with run manifests, matching 0.628 / 0.583 within noise | Package partly extracted; E00-E02 done |
| 0 | [E03](experiments/E03_client_dataset.md), [C](experiments/C_pooled_dedup.md), [D](experiments/D_local_per_turbine.md), [S](experiments/S_seed_robustness.md) | 22 verified clients on one feature schema; matched baseline runs and uncertainty estimates | E03 verified during the completed Kaggle D run: 22 clients / 58 events, seed-42 CARE 0.639852. Full-precision artifact archive and matched multi-seed baseline/uncertainty study remain open. |
| 1 | Federated plumbing: Flower simulation, shared per-event CARE reporting | Full-participation, one full-batch SGD step matches centralized GD; server metrics match direct evaluation | Ready to start; the real-data bridge and D execution are demonstrated. FL implementation is not yet present. |
| 2 | [F1](experiments/F1_fedavg.md) | Restricted SGD and metric parity pass; compare FL with a matched centralized replay of the same client partitions | Not started |
| 3 | [F2](experiments/F2_personalised_fl.md) | F2 above both C and D | Not started |
| 4 | [F3](experiments/F3_fedprox_sweep.md) | Measure CARE and optimization sensitivity to mu; zero mu matches FedAvg under the same settings, with no guaranteed D/C endpoints | Not started |
| 5 | [F4](experiments/F4_private_fl.md) | CARE cost per privacy budget | Not started |
| any | [X](experiments/X_ablations.md) | No ablation changes an arm ranking | X2 (earliness weighting) most urgent |

## Open questions
The Kaggle workflow is in [CLIENT_BRIDGE.md](CLIENT_BRIDGE.md).

The subsequent Kaggle run now supplies full-cohort execution evidence. The current
[transition assessment](FEDERATED_TRANSITION_2026_10_15.md) approves phase 1 and an
exploratory pilot after its mechanics checks; it does not mark the entire phase-0
statistical study complete.

- Can `notebooks/archive/Baseline_model.ipynb` be deleted?
- Does the earliness weighting match the paper's Figure 1?
- Which TensorFlow / Keras versions produced the published numbers?
- Dataset licence: may derived results be committed?
