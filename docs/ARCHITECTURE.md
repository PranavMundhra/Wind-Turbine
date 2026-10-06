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
| `data` | TODO | cache, features, normal mask, splits (still in notebooks) |
| `models` | TODO | autoencoder 243→133→83→20→83→133→243, training loop (still in notebooks) |
| `federated` | TODO | server, clients, personalisation (roadmap phases 1-5) |
| `runners` | TODO | CLI entry points per arm |

Experiments: [`experiments/`](experiments/README.md) and `experiments/registry.yaml` at the repo root. Decisions: see `docs/adr/`. Reproducibility fixes: [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Model and results detail: `docs/MODEL_DOCUMENTATION.md`.
