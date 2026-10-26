# Architecture

```
runners / notebooks  ->  src/windturbine  <-  configs/*.yaml
                          data | models | scoring | audit | federated
data (CSV -> parquet cache)  ->  library  ->  results/ (runs, leakage_audit)
```

| Package | Status | Contents |
|---|---|---|
| `config` | done | `Config` dataclass; defaults mirror the notebook constants; YAML overrides |
| `scoring` | done | `fbeta_from_counts`, `criticality`, `event_detected`, `accuracy_care`, `earliness_ws`, `pick_threshold`, `reconstruction_error` |
| `audit` | done | `row_fingerprints`, `leaked_fraction` |
| `runs` | done | run directories, `manifest.json`, result contract, `set_seeds`, `config_hash` |
| `paths` | done | `resolve_data_root` (no hard-coded Kaggle paths) |
| `data` | E03 implemented | `clients`: shared raw parsing, canonical fingerprints, cleaning, fixed schema, local scaling, atomic exports and independent reload checks; D consumes these exports. Older A/B/C helpers still need extraction. |
| `models` | F1 implemented | Named dense autoencoder with shared feature width; explicit minibatch training and validation loss; optional TensorFlow imports. |
| `federated` | F1 implemented | Sequential clients, real Flower FedAvg aggregation, matched centralized training, validation selection and verified artifacts. Personalisation and private FL remain planned. |
| `runners` | TODO | CLI entry points per arm |

Experiments: [`experiments/`](experiments/README.md) and `experiments/registry.yaml` at the repo root. Decisions: see `docs/adr/`. Reproducibility fixes: [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Model and results detail: `docs/MODEL_DOCUMENTATION.md`.

Client artifact contract and Kaggle execution: [CLIENT_BRIDGE.md](CLIENT_BRIDGE.md).
The shared CARE evaluator and event-table aggregation are in `scoring/evaluation.py`.
The client-data bridge prepares inputs; the separate F1 trainer consumes them.
See [FEDAVG.md](FEDAVG.md) for its execution and artifact contract.
