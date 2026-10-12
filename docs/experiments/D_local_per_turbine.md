# D: Local per-turbine, no collaboration

<!-- generated:start -->
| Registry field | Value |
| --- | --- |
| Status | done |
| Phase | 0 |
| Importance | gate |
| Depends on | [E03](E03_client_dataset.md) |
| Judged against | [A](A_isolated_per_event.md), [C](C_pooled_dedup.md) |
| Unblocks | [F2](F2_personalised_fl.md), [F3](F3_fedprox_sweep.md), [F4](F4_private_fl.md) |
| Notebook | [`notebooks/13_arm_local_turbine.ipynb`](../../notebooks/13_arm_local_turbine.ipynb) |
| Config | [`configs/arm_local_turbine.yaml`](../../configs/arm_local_turbine.yaml) |
| Runs | `results/runs/D/` |
<!-- generated:end -->

## Question
How well does a turbine do with only its own data? This is the floor any federated scheme must beat to justify collaborating.

## Why it matters
This arm is new to the plan. The companion PDF frames the federated curve as running between A and B, but federated clients are turbines, not events, so the correct "no sharing" endpoint is a per-turbine model, and the correct "full sharing" endpoint is C. D is also what FedProx approaches as mu goes to 0 ([F3](F3_fedprox_sweep.md)). Compared with A, it shows whether specialising per event adds anything beyond specialising per turbine.

## Method: change from A
22 models, one per `asset_id`, each trained on that turbine's leak-free rows from [E03](E03_client_dataset.md). One scaler and threshold per turbine. Every event is scored by its turbine's model, then CARE is computed over all 58 events exactly as in A.

As implemented in the notebook:
- The E03 partitions are built in memory from the E00 cache, using the arms' features, normal mask and fingerprints. The on-disk builder (`20_federated_clients.ipynb`) is still the old one, so E03 stays `partial` until the federated arms need files.
- Each event file gets A's blocked split (5-day blocks, every 7th held out). The turbine's train and validation slices are then the union of its files' slices.
- The E03 rule: any training or validation row whose fingerprint appears in any of the turbine's prediction windows is dropped.
- Training rows whose fingerprint also appears in the turbine's validation slice are dropped too. A turbine's files share rows (E02), so without this a row could sit in one file's training blocks and another file's validation blocks, which would bias the threshold low. A has no such case because it splits a single file. The copy is dropped from training, not from validation: the first version dropped it from validation, and that emptied the validation sets of turbines 32, 38 and 42, whose files are near-copies of each other.
- Counts per turbine are written to `client_manifest.csv`.

Run: `WT_DATA_ROOT=<dir containing "Wind Farm C"> WT_SEED=42`, then execute the notebook. On Kaggle the first cell clones this repo.

## Inputs and outputs
| Direction | Artifact | Path |
| --- | --- | --- |
| in | Client partitions (E03 rule, built in memory) | E00, E03 |
| out | `results.json`, `per_event.csv`, `scores.parquet`, `client_manifest.csv` (rows removed per turbine), `fit_log.csv` | `results/runs/D/<run>/` |

## Decision rule
| Result | Reading |
| --- | --- |
| D ≈ A | Turbine-level specialisation is enough; federated personalisation targets D. |
| D < A by more than noise | Event-level context matters (seasons, operating regime per file); consider local thresholds per event in F2. |
| D < C | The fleet helps even without personalisation; F1 alone may suffice. |

## Results
Run locally on CPU (TensorFlow 2.21, pandas 3.0, op determinism on), 2026-10-11/12, from an uncommitted working tree on top of `fd1cfb5` (`"dirty": true`). Seed 42 was run in VS Code. Run directories are under `results/runs/D/`.

| Run id | Seed | CARE | Coverage | Earliness | Reliability | Accuracy | TP/FP/FN/TN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `20261011T185853Z_2525b30e_s42` | 42 | 0.625 | 0.322 | 0.231 | 0.659 | 0.956 | 12/4/15/27 |
| `20261011T190243Z_605bad76_s43` | 43 | 0.607 | 0.310 | 0.230 | 0.606 | 0.944 | 12/6/15/25 |
| `20261011T190246Z_3e6c8da8_s44` | 44 | 0.600 | 0.331 | 0.236 | 0.556 | 0.939 | 11/7/16/24 |
| **mean ± sd** | | **0.611 ± 0.013** | 0.321 | 0.232 | 0.607 | 0.946 | |

Rows: 2,209,346 normal training rows before filtering. 74,249 were dropped as seen in the turbine's own prediction windows, and 361,002 as copies of validation rows. Of 347,214 validation rows, 13,757 were dropped as seen in prediction windows. The smallest validation set is 5,040 rows, and no threshold exceeded 10× its validation median (worst 7.3×).

**Reading.**
- **D vs A:** D ≈ A. Seed 42 gives 0.625 against A's 0.628. The 3-seed mean, 0.611, is 0.017 lower: about one noise floor, from a different environment. Turbine-level specialisation recovers essentially all of A's per-event result, so **federated personalisation should target D**.
- **D vs C:** D is well above C (0.611 ± 0.013 against 0.326 ± 0.288, and 0.625 against 0.543 at seed 42). It is also far more stable across seeds, because each threshold is local. The fleet model does not help without personalisation, so F1 alone is not expected to suffice.
- **Single-event turbines:** turbines 5, 13 and 14 (events 89, 29 and 85, all normal) get Accuracy 0.997, 0.896 and 0.999 at seed 42. Compare these with A's per-event results once A is re-run with a run directory.

## Risks and open questions
- Leak removal may shrink some turbines' training sets heavily; report rows removed per turbine.
- The 3 turbines with a single event get the same data as their Arm A model, so for them D and A should agree within noise. That is a free sanity check.
