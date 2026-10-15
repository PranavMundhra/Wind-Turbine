# Wind Turbine Repository Quality and Leakage Review

Review date: 15 October 2026. Reviewed commit: `37e2ea5b465cd9b9971759fc2814f21bdbecd56d`.

The repository contains strong evidence that some event files reuse the same sensor measurements. Pooling these files can put evaluation measurements into training. The new Arm C and Arm D notebooks remove matching rows in sensible places, before fitting their scalers and models. However, the leakage charts, scoring rules, and saved evidence still need improvement before the results can support strong research claims.

Three reviewers checked the leakage measurement, the experiment notebooks, and the supporting code. This report separates what the code does, what saved outputs show, and what still needs a new experiment. No experiment code was changed.

## What leakage means in this project

SCADA data are sensor readings collected from a turbine, usually every ten minutes. An event file contains a training period and a prediction period. The prediction period is the evaluation or test period.

Training data teach the model. Validation data help choose when to stop training and where to set the anomaly threshold, the score above which a row is flagged. Test data measure the finished system. Test measurements should not influence those earlier choices.

Leakage happens when information from evaluation influences model fitting or model selection. For example, a measurement might be marked `prediction` in event file 1 but `train` in event file 16. A pooled model can then train on a measurement that it will later be tested on. This can make its test result misleading even though each individual file has separate train and prediction labels.

The same measurements can also appear with shifted dates. Matching timestamps alone would therefore miss some copies. Matching sensor values is a useful way to find them.

The project compares four experiments:

| Arm | What is trained | Duplicate handling |
| --- | --- | --- |
| A | One model per event file | No cross-file pooling; a small amount of within-file value overlap is reported |
| B | One model for the whole farm | Overlap is measured, but matching rows remain in training |
| C | One model for the whole farm | Copies of prediction rows are removed from training and validation |
| D | One model per turbine | Copies of that turbine's prediction rows are removed; training copies of validation rows are also removed |

Sources: [model documentation](MODEL_DOCUMENTATION.md), [Arm C](../notebooks/12_arm_pooled_dedup.ipynb), and [Arm D](../notebooks/13_arm_local_turbine.ipynb).

## How the leak was reported

### The headline percentage

Arm B gives each sensor row a fingerprint: a short identifier calculated from its selected sensor values. The default uses 238 raw average sensor columns, rounded to six decimal places, followed by a pandas hash. The 243 number in the architecture describes the model inputs after angle transformations; it is not the number of raw columns used in this check.

The arm notebooks load sensor values as 32-bit floating-point numbers. The check therefore compares parsed and rounded values, rather than original CSV text. Very small differences can disappear during conversion or rounding.

The code builds a set of fingerprints from normal training rows belonging to the fitting turbines. It then checks every prediction row against that set. The validation turbines are excluded from this training reference in the usual turbine-holdout branch.

The reported calculation is:

```text
prediction rows with a fingerprint found in fitting data
------------------------------------------------------ x 100
                all prediction rows

40,216 / 158,528 x 100 = 25.37%, rounded to 25.4%
```

Repeated prediction rows count separately. The denominator also includes prediction rows that the CARE scoring mask may later exclude. This percentage describes overlap with fitting data; it does not count all possible exposure through validation or preprocessing.

The saved Arm C seed-42 output repeats the same before-removal counts and reports 27 of 58 events affected. Therefore, the headline has supporting saved output in the current repository. The raw dataset and the detailed per-event leakage CSV are unavailable in this checkout, so the counts were not independently regenerated during this review.

Sources: [Arm B](../notebooks/11_arm_pooled.ipynb), cells 15, 17, and 21; [Arm C](../notebooks/12_arm_pooled_dedup.ipynb), cells 18 and 20. Cell numbers here use the notebook JSON index, starting at zero.

### The separate duplication investigation

The leakage audit first finds possible matches using 12 changing sensor columns rounded to three decimal places. It keeps only status-0 rows and removes fingerprints that occur more than once within a file. These choices make one-to-one alignment easier, but they discard rows.

The audit then performs stronger checks. It tests whether matches follow a consistent time shift, compares columns that were not used to discover the matches, and checks 226 non-discovery average columns for exact numerical agreement. It also tries incorrect time shifts and pairs from different turbines. A further check separates zero or missing values from informative, nonzero matches.

The [saved summary](../results/leakage_audit/evidence_summary.json) and notebook outputs report:

| Check | Recorded result | What it supports |
| --- | --- | --- |
| Pairs of files from the same turbine screened | 64 | The size of this screening exercise |
| Pairs with enough candidate matches | 49 | These pairs reached the verification stage |
| Candidate pairs verified as duplicates | 49 | Strong evidence of repeated measurements in those pairs |
| Lowest agreement across the 226 verification columns | 100% | Agreement extends beyond the discovery columns |
| Cross-turbine control pairs | 30, with no duplicate verdicts | Useful evidence against obvious false matches in those controls |
| Observed time shifts | Minus 365 days, zero, plus 365 days | Dates alone are not enough to identify copies |

The remaining 15 pairs had too few discovery matches. That does not prove that they contain no repeated measurements. Also, exact numerical agreement after parsing CSV values is more precise wording than the repository's claim of byte-identical rows. The audit does not compare all 957 original columns or the original file bytes.

Source: [leakage audit notebook](../notebooks/02_leakage_audit.ipynb), cells 1 through 11 and 13.

## How accurate the reporting method is

The evidence is convincing for the existence of duplicated measurements. It is weaker as a claim about a universal detection accuracy, an exact false-positive rate, or the effect of leakage on model performance. Those are different questions.

### One chart uses a different denominator

The audit notebook's visualization in cell 12 uses the reduced discovery rows: status 0, unique fingerprints, 12 columns, and three decimal places. It focuses on turbine 53. Its labels describe percentages of prediction rows, which can be read as percentages of all prediction rows. That interpretation is incorrect.

Controlled examples using the current notebook function showed why this matters:

| Simplified sensor values | Chart result after filtering | Overlap among all original prediction rows |
| --- | --- | --- |
| Training `[1]`, prediction `[1, 2, 2]` | 100% | 33.3% |
| Training `[1, 1]`, prediction `[1]` | 0% | 100% |

In the first example, the two repeated `2` rows disappear from the denominator. In the second, both repeated training values disappear from the reference set. This is a reporting problem in that visualization. It does not show that Arm B's separately calculated 25.37% is wrong.

Cell 12 also counts discovery hashes directly without the stronger verification used earlier. Rows with discovery values `1.0001` and `1.0002` have the same three-decimal fingerprint even if another sensor is completely different. A controlled example produced a match despite different full rows.

The chart should either say "candidate matches among retained discovery rows" or verify the full selected sensor rows and count overlap against the original, clearly defined prediction population.

### The small Arm A overlap is not a proven chance rate

The documentation calls 13 matches among 158,528 prediction rows an empirical false-positive floor and compares Arm B with it as roughly 3,000 times the chance rate.

The code establishes only that these are within-file value matches. Different row labels or timestamps do not prove that the underlying measurements are independent. The explanation that all 13 are harmless shutdown or missing-value rows needs direct matched-row inspection. Arm A's reference is also collected before its fitting and validation split, whereas Arm B's usual reference excludes validation turbines.

Use the wording "13 observed within-file fingerprint matches" until those rows are inspected. Do not present their ratio as a measured probability of a false match. Likewise, zero matches in 30 control pairs is useful evidence, but it does not establish a universal zero false-positive rate.

### Fingerprints have a limited meaning

Matching hundreds of sensor values is a useful test for repeated rows. It does not detect every related measurement. Copies with small changes, partially repeated rows, or nearby readings from the same operating period may remain. Hash matches should be checked against actual sensor values when producing final evidence.

A smaller implementation issue was reproduced: six-decimal rounding can produce positive and negative zero, which pandas hashes differently. Values `+0.000000001` and `-0.000000001` therefore round to numerically equal values but can escape the rounded-value match. Normalize signed zero before hashing and test this case. This issue does not prevent detection of identical copies with the same representation.

Sources: [fingerprint implementation](../src/windturbine/audit/fingerprints.py), [audit tests](../tests/test_audit.py), and [model documentation](MODEL_DOCUMENTATION.md), sections 2 and 3.

## How leakage was handled

### Arm C cleans both training and validation

Arm C collects fingerprints from every prediction window, then removes matching normal rows from both the fitting data and the held-out validation data. It fits the scaler and autoencoder afterward. Cleaning validation matters because validation affects early stopping and the anomaly threshold.

The stored seed-42 output records:

| Item | Recorded count |
| --- | --- |
| Training rows before removal | 2,281,373 |
| Training rows removed | 78,802 |
| Validation rows removed | 9,270 |
| Prediction rows matching fitting data before removal | 40,216 of 158,528 |
| Prediction rows matching fitting data after removal | 0 of 158,528 |

The removed training count can exceed the affected prediction count because one prediction measurement may have several training copies. Arm C asserts that the final fitting-data overlap is zero. Its code also filters validation, although the printed final overlap assertion covers fitting data only.

Using prediction identities to remove copies, without using their fault labels or scores to train the model, is a reasonable contamination-control step. For a stronger final check, save and assert all three intersections: training with prediction, validation with prediction, and training with validation.

Source: [Arm C notebook](../notebooks/12_arm_pooled_dedup.ipynb), cells 18, 20, and 22.

### Arm D also separates training from validation

Arm D groups event files by turbine and builds its own cleaned partitions in memory. It removes that turbine's prediction copies from training and validation. It then removes training rows whose fingerprints remain in validation. Each turbine's scaler is fitted only on its cleaned training data.

The stored seed-42 output records 22 trained models, 74,249 training rows removed as prediction copies, 361,002 training rows removed as validation copies, and 13,757 validation rows removed as prediction copies. It reports CARE 0.625.

This is a useful improvement. The notebook should also save explicit zero-overlap checks after filtering; its current assertion checks that the resulting arrays are nonempty. Its local cleaning rule is appropriate to its separate turbine models. Any future model that combines clients must recheck overlap at that wider scope.

Source: [Arm D notebook](../notebooks/13_arm_local_turbine.ipynb), cells 18 and 20.

### The exported federated clients are unfinished

The [client builder](../notebooks/20_federated_clients.ipynb) is still the older implementation. Its integrity checks cover basic client IDs and counts, not duplicate separation. The [E03 specification](experiments/E03_client_dataset.md) openly marks this work as partial.

Arm D does not use those exported files. Its cleaned in-memory partitions should become the shared implementation before federated training uses the exported clients.

## Remaining issues that affect correctness

| Priority | Finding and consequence | Recommended change |
| --- | --- | --- |
| High | All four arms use a training power filter during CARE evaluation. A normal-status row with wind speed 8 and near-zero power is excluded, although the documented evaluation rule is based on status. This changes metric support and the criticality counter. | Separate training eligibility from evaluation eligibility, then rescore saved predictions or rerun where needed. |
| Medium | Angle selection reads the whole first event file, including prediction values, to decide which columns receive angle transforms. Test values can influence preprocessing even after duplicate removal. | Fix the schema from metadata or derive it from training data only. |
| Medium | Cache acceptance checks column names, while the recorded cache manifest has no hashes of the actual CSV or parquet contents. Changed or stale data may keep the same recorded identity. | Record and verify content hashes and metadata versions for every input. |
| Medium | C and D have stored seed-42 outputs, but `results/runs/` contains only `.gitkeep`. The six reported run manifests and detailed result artifacts are unavailable here. | Publish small manifests, resolved configs, results, removal counts, and per-event tables in a durable location. |
| Medium | Run comparison groups only by experiment and discards config/data identities. Different experimental settings can be averaged as though they were repeats. | Separate settings and data versions while allowing random seeds to vary within a valid group. |
| Medium | Native Windows text decoding causes five test failures and a false stale-graph warning. | Read and write repository text explicitly as UTF-8; include Windows in CI. |
| Medium | Arm A records event metadata before skipping events with too few training rows. Evaluation then expects a missing prediction record and raises `KeyError`. | Handle skipped events explicitly and report their effect on metric support. |
| Medium | Accuracy returns 1.0 when a normal event has no eligible rows. Unsupported events can appear perfectly scored. | Return an undefined result and exclude unsupported events from the aggregate. |
| Medium | `write_results` writes `results.json` before validating per-event columns. A rejected run can remain visible to comparison scripts. | Validate all outputs before publishing a completed result. |

Key code references: [Arm C preprocessing](../notebooks/12_arm_pooled_dedup.ipynb), cells 6 through 8; [Arm D partitioning](../notebooks/13_arm_local_turbine.ipynb), cell 18; [Arm A training loop](../notebooks/10_arm_isolated.ipynb), cell 15; [CARE helpers](../src/windturbine/scoring/care.py); [run comparison](../scripts/compare_runs.py); [run writer](../src/windturbine/runs.py); [registry checker](../scripts/check_registry.py); and [graph generator](../scripts/render_graph.py).

Lower-priority issues include same-second run-directory collisions and failure to accept comma-prefixed exports consistently in the audit and EDA notebooks. The blocked validation split also has adjacent training and validation rows at block boundaries. It counts retained rows rather than elapsed time, so it should not be described as a full temporal gap.

## What the performance numbers can support

The experiment documents report these CARE results:

| Arm | Seed 42 | Reported mean and sample standard deviation over three seeds |
| --- | --- | --- |
| A | 0.628 | Not provided |
| B | 0.583 | Not provided |
| C | 0.543 | 0.326 plus or minus 0.288 |
| D | 0.625 | 0.611 plus or minus 0.013 |

C and D have supporting seed-42 notebook outputs. Their other seed results are currently supported by the experiment documents, without the corresponding run artifacts in this checkout.

C varies greatly across seeds. Changing the seed changes which three turbines supply validation, as well as training randomness. The documents report thresholds around 19, 161, and 177. These logs support concern about the validation design; they do not by themselves isolate its causal effect.

The old estimate that contamination adds about 0.011 CARE comes from a subgroup calculation. It is not a controlled retraining result. Subtracting C's result from historical B cannot isolate leakage either: the environments differ, B lacks matched repeated runs, and the scoring concern above affects the comparison. D versus C also changes model scope and threshold scope together.

For a defensible final-year project comparison, run B and C with the same environment, seeds, turbine splits, feature schema, and corrected scoring. Report each paired difference and the row counts removed. Repeat A and D under the same conditions. When estimating uncertainty, group by turbine because multiple events from one turbine are related.

Sources: [Arm C results](experiments/C_pooled_dedup.md), [Arm D results](experiments/D_local_per_turbine.md), and [historical model results](MODEL_DOCUMENTATION.md).

## Checks completed

| Check | Result |
| --- | --- |
| GitHub CI for the reviewed commit | Passed |
| Ruff on source, scripts, and tests | Passed |
| Registry and links with UTF-8 | Passed |
| Generated graph freshness with UTF-8 | Passed |
| Full pytest suite with UTF-8 | 30 passed |
| Full pytest suite with native Windows decoding | 25 passed, 5 failed with `UnicodeDecodeError` |
| Native Windows graph check | False stale-graph warning |
| Synthetic leakage and scoring cases | Confirmed the chart denominator, candidate-match, signed-zero, and scoring-mask issues described above |

The latest [GitHub CI run](https://github.com/PranavMundhra/Wind-Turbine/actions/runs/38366650246) passed for this commit. The earlier missing Arm C and D notebooks have been added, so the previous broken-link finding is resolved.

The tests check small library functions and repository consistency. They do not run the full training notebooks or establish the accuracy of the real-data leakage rate. Raw Farm C data and the complete run artifacts were unavailable for a full retraining and recount during this review.

Useful local commands are:

```powershell
.\.venv\Scripts\python -X utf8 -m pytest -q
.\.venv\Scripts\python -m ruff check src tests scripts
.\.venv\Scripts\python -X utf8 scripts/check_registry.py
.\.venv\Scripts\python -X utf8 scripts/render_graph.py --check
```

## Recommended next work

1. Correct the evaluation mask and prevent test values from choosing preprocessing rules.
2. Give each leakage table a precise denominator, reference set, column list, and rounding rule. Keep discovery charts separate from confirmed full-row overlap counts.
3. Save verified matching row pairs, their source event IDs, time shifts, and before/after counts. Inspect the 13 within-file matches instead of assuming their cause.
4. Add tests for the complete cleaning process, including training, validation, and prediction intersections. Share that implementation across C, D, and the client builder.
5. Publish the small run artifacts and detailed audit CSVs. Hash actual input files and pin the environment.
6. Run paired experiments before claiming how much leakage changes CARE or which training strategy is best.

The project is a useful research prototype with a convincing duplication investigation and meaningful duplicate removal in C and D. Its next step is to make the measurement definitions, scoring, and saved evidence as reliable as the core idea.
