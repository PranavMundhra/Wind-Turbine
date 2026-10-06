# Roadmap

Experiments are defined in [`experiments/`](experiments/README.md) and drawn in the [experiment graph](experiments/EXPERIMENT_GRAPH.md). Phases come from the companion PDF, plus phase R (this restructure) and Arm D, which the original plan did not have. The PDF put client repartitioning in phase 1; it moves to phase 0 as E03 because Arm D needs it.

| Phase | Experiments | Exit test | Status |
|---|---|---|---|
| R | [E00](experiments/E00_data_cache.md), [E01](experiments/E01_eda.md), [E02](experiments/E02_leakage_audit.md); extract `data` and `models` into the package | A and B re-run from configs with run manifests, matching 0.628 / 0.583 within noise | Package partly extracted; E00-E02 done |
| 0 | [E03](experiments/E03_client_dataset.md), [C](experiments/C_pooled_dedup.md), [D](experiments/D_local_per_turbine.md), [S](experiments/S_seed_robustness.md) | 22 leak-free clients on one feature schema; contamination vs specialisation separated; both ends of the federated dial measured at 3 seeds | E03 builder exists but is not leak-free; the rest not started |
| 1 | Federated plumbing: Flower simulation, per-client CARE counts reported to the server | FedAvg equals centralised gradient descent on a toy model (see F1) | Not started |
| 2 | [F1](experiments/F1_fedavg.md) | F1 ≈ C confirms the stated prediction | Not started |
| 3 | [F2](experiments/F2_personalised_fl.md) | F2 above both C and D | Not started |
| 4 | [F3](experiments/F3_fedprox_sweep.md) | CARE-vs-mu curve between D and C | Not started |
| 5 | [F4](experiments/F4_private_fl.md) | CARE cost per privacy budget | Not started |
| any | [X](experiments/X_ablations.md) | No ablation changes an arm ranking | X2 (earliness weighting) most urgent |

## Open questions
- Can `notebooks/archive/Baseline_model.ipynb` be deleted?
- Does the earliness weighting match the paper's Figure 1?
- Which TensorFlow / Keras versions produced the published numbers?
- Dataset licence: may derived results be committed?
