# Two normal-behaviour autoencoders for CARE to Compare, Wind Farm C

Reproduction and extension of the fault-detection baseline for the *CARE to Compare*
benchmark, evaluated with the CARE score. Two training strategies are compared under
identical preprocessing, architecture and scoring:

| | **Arm A — isolated** | **Arm B — pooled** |
|---|---|---|
| Models | 58, one per event dataset | 1, fleet-wide |
| Training rows per model | ~38,000 | 2,281,373 |
| Scaler | one per event | one global |
| Threshold | one per event | one global |
| Cross-event information | none | all of it, including duplicated rows |

Everything except the training strategy is shared code, generated from a single
source so the two arms cannot drift apart. Any difference in results is attributable
to the strategy, not to an accidental difference in preprocessing or scoring.

---

## 1. The network

![Autoencoder architecture](figs/fig1_autoencoder.svg)

A symmetric undercomplete autoencoder. It is trained only on data the turbine
produced while healthy, so it learns to reconstruct the *normal-operation manifold*.
Feed it a row from a degrading turbine and the reconstruction is poor, because the
network has no representation for that state. The residual is the anomaly score.

### Layer table

| Layer | Shape | Weights | Biases | Params | Activation |
|---|---|---|---|---|---|
| input | — | — | — | — | — |
| encode 1 | 243 → 133 | 32,319 | 133 | 32,452 | ReLU |
| encode 2 | 133 → 83 | 11,039 | 83 | 11,122 | ReLU |
| **bottleneck** | 83 → 20 | 1,660 | 20 | 1,680 | ReLU |
| decode 1 | 20 → 83 | 1,660 | 83 | 1,743 | ReLU |
| decode 2 | 83 → 133 | 11,039 | 133 | 11,172 | ReLU |
| output | 133 → 243 | 32,319 | 243 | 32,562 | linear |
| | | | | **90,731** | |

### Why these choices

**The bottleneck is the whole mechanism.** 243 → 20 is a 12.2× compression. Wind
turbine SCADA channels are massively correlated — power, rotor speed, generator
speed, several temperatures and pitch angles all move together with wind speed — so
20 dimensions is enough to capture healthy operation but not enough to also encode a
fault signature. If the bottleneck were too wide the network would learn the identity
function, reconstruct anomalies perfectly, and detect nothing.

**Linear output, ReLU hidden.** The inputs are standardised and therefore signed;
a ReLU output layer could not produce negative values and would floor half the
feature space at zero. ReLU in the hidden layers is the usual choice for speed and
gradient behaviour.

**No denoising noise** (`NOISE_STD = 0.0`). Farm C's baseline configuration does not
use it. The `GaussianNoise` layer is present but inactive, so switching it on is a
one-line change.

**MSE loss on standardised features.** Every feature contributes equally to the loss.
Without standardisation, a channel measured in kW would dominate one measured in
degrees Celsius purely through scale.

### Optimiser and training

| Setting | Value | Note |
|---|---|---|
| Optimiser | Adam | |
| Initial LR | 0.0056 | paper Table 3, Wind Farm C |
| LR schedule | `ReduceLROnPlateau`, factor 0.5, patience 4, min 1e-6 | **deviation — see §5** |
| Batch size | 64 | |
| Max epochs | 200 | |
| Early stopping | `val_loss`, patience 10, restore best weights | |
| Loss | MSE | |
| Weight init | Keras default (Glorot uniform, zero bias) | |
| Seed | 42 | see §7 on determinism |

**Anomaly score** = L2 norm of the residual over all 243 features:
`score = ||x − x̂||₂`. One scalar per 10-minute record.

---

## 2. Arm A — event-isolated

![Isolated pipeline](figs/fig2_pipeline_isolated.svg)

One model per event dataset. The loop body is entirely self-contained: read one
file, filter it, split it, fit a scaler on it, fit a model on it, derive a threshold
from it, score its own prediction window, discard everything and move on.

**Isolation is guaranteed by construction, not by a check.** Training consumes only
rows matching `train_test == "train"` and the normal mask; scoring consumes only
`train_test == "prediction"` rows from the same file. Those partitions are disjoint,
so no evaluation row can reach the optimiser.

**The fingerprint audit measures something else.** It hashes every row's 243 sensor
values and looks for prediction rows whose hash appears among that event's training
rows. It found 13 out of 158,528 — 0.008%. These are not leaks; they are two
different timestamps whose readings are identical to six decimal places, which
happens on flatlined or all-NaN rows during shutdowns. That number is worth keeping:
it is the **empirical false-positive floor of the fingerprint method**, and it is the
baseline against which Arm B's cross-file leakage has to be judged.

### Why this arm exists

It is the paper-faithful configuration. The dataset FAQ states that individual
autoencoder models per turbine were used, and per-event is that at the strictest
granularity. It is also the *only* configuration in which the published data's
cross-file row duplication cannot contaminate the evaluation.

---

## 3. Arm B — pooled (fleet-wide)

![Pooled pipeline](figs/fig3_pipeline_pooled.svg)

One model for the whole farm. Pass 1 pools every event's normal training rows; a
single model and a single threshold are fitted; pass 2 scores all 58 prediction
windows.

**Validation holds out whole turbines, not events.** This matters: because the same
physical rows appear in several event files, an event-level split would put identical
rows on both sides and make `val_loss` meaningless. Three of 22 turbines are held
out, so validation measures what it should — generalisation to a turbine the model
has never seen.

**Contamination is measured, not assumed.** Fingerprints of every training row are
collected into a set, and every event's prediction rows are tested against it.

> 40,216 of 158,528 prediction rows (**25.4%**) were present in the training corpus,
> across 27 of 58 events.

Against the 0.008% collision floor from Arm A, that is roughly 3,000× the chance
rate — unambiguously real duplication.

### Why the contamination exists

It is a property of the published dataset, not of this pipeline. The FAQ states that
overlaps across different event CSV files are an expected side effect of the
anonymization procedure, and that event files may share timestamp ranges even when
they represent different events. Independent verification on this copy of the data
went further: rows are byte-identical across 226 columns, matched on a single rigid
index offset, in contiguous runs with a median longest run of 720 records (five days),
with zero false positives across 30 cross-turbine control pairs. Some pairs match
only after a ±365-day shift, consistent with per-file timestamp anonymisation.

Consequently a row that is `prediction` in file A is `train` in file B, and any
pipeline that pools training data across events trains on its own evaluation rows.

---

## 4. Shared data flow

Both arms run the identical sequence below. Only the *scope* differs — one event, or
all of them.

```
event CSV (~55k rows × 957 cols)
  │
  ├─ parse 243 of 957 columns          avg features only; Min/Max/Std excluded
  │
  ├─ feature build
  │     ├─ 226 linear features         passed through
  │     └─ 5 angle features            → sin, cos   (243 = 226 + 5×2 + …)
  │
  ├─ normal mask                       status_type_id ∈ {0,2}
  │                                    AND NOT (3 ≤ wind ≤ 12 AND |power| ≈ 0)
  │
  ├─ split                             Arm A: blocked in time
  │                                    Arm B: by turbine
  │
  ├─ StandardScaler                    fitted on TRAINING rows only
  │     └─ drop features >30% NaN
  │     └─ NaN → 0 AFTER scaling       i.e. imputed to the feature mean
  │
  ├─ autoencoder                       243 → 20 → 243
  │
  ├─ threshold                         99th percentile of ||x−x̂||₂ on held-out
  │                                    NORMAL rows
  │
  └─ prediction window → scores → flags → criticality → CARE
```

### Preprocessing decisions and their reasons

**Avg features only (238 of 952 selected, 243 after angle transform).** The FAQ notes
that Avg features are often the safest starting point because there are implausible
values in some of the Min, Max and Std features.

**Angle transform with a range check.** Metadata-driven detection proposed 17 angle
columns, but a range check against the first file showed 12 of them spanning ranges
like [−2.00, 89.75] or [0.00, 3.43] — not angles. Applying sin/cos to those would
have silently corrupted 12 features. They are dropped automatically and the table is
printed on every run, so the decision is auditable rather than trusted.

**NaN imputation after scaling.** Filling with 0 *before* standardisation maps a
missing value to `−mean/scale`, which for a channel centred at 600 kW is an enormous
synthetic outlier that the autoencoder then has to reconstruct. Filling after scaling
places it at the feature mean, which is the neutral choice.

**Power-curve filter.** Wind between cut-in and rated while power is near zero is
physically implausible — it indicates curtailment or a sensor fault, not normal
operation. Including those rows would teach the model that zero power at 8 m/s is
normal.

---

## 5. Fidelity to the original CARE to Compare model

### What matches

| Element | Status |
|---|---|
| Normal-behaviour autoencoder trained on healthy data | matches |
| One model per turbine (Arm A) | matches the FAQ's stated approach |
| Hidden layers `[133, 83, 20, 83, 133]` | Table 3, Wind Farm C |
| Adam, initial LR 0.0056, batch 64 | Table 3, Wind Farm C |
| Reconstruction error as anomaly score | matches |
| Status filtering for normal behaviour | FAQ: intended use of `status_type_id` |
| Angle transform, NaN handling, invalid-measurement filtering, scaling | the FAQ's listed preprocessing steps |
| Coverage ground truth = `event_start` … `event_end` | Farm C rule |
| Abnormal-status timestamps excluded from pointwise evaluation | FAQ |
| Criticality counter, threshold 72 | FAQ's documented default |
| Reliability computed **once** across all events, not averaged | FAQ |
| CARE weights (1, 1, 1, 2) with Accuracy double-weighted | matches |

### Deliberate deviations

| Deviation | Reason |
|---|---|
| **Feature set: 243 avg-derived inputs.** The paper's input dimension is unknown to this reproduction. | Avg-only follows the FAQ's recommendation. Note the layer sizes were tuned for the authors' feature set, so `243 → 133` is a mild compression where theirs may have been steeper. This is the largest untested assumption in the reproduction. |
| **`ReduceLROnPlateau` added** (`USE_LR_PLATEAU = True`). | At a fixed LR of 0.0056 the *training* loss increased across epochs while validation plateaued — the optimiser was overshooting, and early stopping then restored an under-trained model. The schedule keeps the paper's starting LR and only decays it when validation stalls. Set `False` to reproduce the paper exactly. |
| **Early-stopping patience 10.** | Patience 3 stopped on noise before convergence. |
| **Blocked validation split** (5-day blocks, every 7th). | A random split interleaves validation rows minutes from training rows, so the threshold it yields is calibrated for "same week" and over-flags months later. A chronological split on a single event makes the held-out tail one unseen season, which produced a 42× spread in thresholds across events and silently disabled detection for 13 of them. Blocking gives temporal separation *and* full seasonal coverage. |
| **Threshold = 99th percentile of held-out normal reconstruction error.** | The FAQ is explicit that the target is `event_label`, not `status_type_id`; a normal-behaviour model has no anomaly labels at fit time, so a high quantile of healthy-data error is the principled choice. `SCORE_QUANTILE` is a config constant for sweeping. |
| **Earliness weighting**: flat 1.0 through the first half of the event window, decaying linearly to 0 at the end. | Carried over from the original baseline and **not verified against the paper's Figure 1**. It affects one of four CARE components. This should be checked before publication. |

---

## 6. Scoring

**Coverage** `F̄β` — pointwise F₀.₅ inside labelled event windows, anomaly events only,
restricted to normal-status timestamps.

**Earliness** `W̄S` — weighted coverage inside the window, early detections weighted
higher, same row mask as Coverage.

**Reliability** `EFβ` — a single event-wise F₀.₅ across all 58 events. An event counts
as detected when the criticality counter reaches 72.

**Accuracy** `Ācc` — true-negative rate on the 31 normal events.

**Criticality counter** — on a normal-status timestamp: `+1` if flagged, `−1` if not,
floored at zero; unchanged on abnormal-status timestamps. Reaching 72 means roughly
12 hours of net accumulated evidence.

```
CARE = (Coverage + Earliness + Reliability + 2·Accuracy) / 5
       with guards: 0 if nothing was flagged anywhere;
                    = Accuracy if Accuracy < 0.5;
                    undefined (NaN) if any component lacks supporting events
```

That last guard matters for subgroup analysis — a leaked-only slice containing no
normal events would otherwise report Accuracy 0.0 and trip the `Acc < 0.5` rule.

---

## 7. Results

```
                isolated    pooled
Coverage          0.451      0.227
Earliness         0.332      0.161
Reliability       0.611      0.593
Accuracy          0.874      0.967
CARE              0.628      0.583
tp/fp/fn/tn    16/10/11/21  7/1/20/30
```

**Isolated detects 16 of 27 events; pooled detects 7.** Coverage is exactly double.
Pooled pays for its low detection rate with near-perfect Accuracy — 1 false alarm
against 10 — and because Accuracy is double-weighted, it contributes 66% of pooled's
CARE numerator versus 56% of isolated's.

### Measured effect of contamination

Leaked normal events score 0.9966 Accuracy against 0.9398 for clean ones. Working
back from the overall 0.967, roughly **15 of 31 normal events are substantially
leaked**. Scoring the pooled arm on clean normals only gives Accuracy ≈ 0.940 and
CARE ≈ 0.572, so **contamination is worth about +0.011 CARE** to the pooled arm.

A second hypothesis was **not** supported. Training on anomalous rows should suppress
their detection, since the model learns to reconstruct them. The measurement gives
Spearman(leaked_pct, coverage) = **+0.349, p = 0.075** — the opposite sign, and not
significant. The reason is visible in the distribution: leakage on anomaly events has
a median of 0.00% and only one event above 50%. The contamination sits almost entirely
in the normal events, so there was little signal to find.

### The confound

Arm A differs from Arm B in **two** ways: it has no contamination, and each model
specialises to one turbine. The decomposition above says contamination accounts for
only ~0.011 of the 0.045 gap, implying most of it is specialisation. That is an
inference from a decomposition, not a direct measurement. A third arm — pooled
training with every leaked row removed from the corpus — would separate the two
cleanly.

### Determinism

Two runs of the pooled arm with identical code produced CARE 0.570 and 0.583 (tp 6
then 7). `tf.random.set_seed` does not make cuDNN reductions deterministic. **Treat
±0.013 CARE as run-to-run noise.** The Coverage gap between arms is far outside that;
the Reliability gap (0.611 vs 0.593) is not, and should be treated as noise.

---

## 8. Known limitations

1. **Effective sample size is far below 58.** The 58 events span 22 turbines, 19 of
   which contribute more than one event; the largest contributes 6, or 10.3% of the
   "independent" evaluations. CARE averages over events as though they were
   independent. Error bars are wider than n = 58 implies.
2. **Three events hit the 200-epoch cap** (58, 60, 85) without early stopping, so
   their thresholds rest on models that never met the validation criterion.
3. **Single seed per arm.** Given the ±0.013 noise floor, conclusions should rest on
   3+ repeats.
4. **Earliness weighting unverified** against the paper's Figure 1.
5. **Layer sizes were tuned for a different input dimension.** A sensitivity check on
   the bottleneck width would be cheap and informative.
6. **Same-farm correlation is irreducible.** All 58 events come from one wind farm, so
   neighbouring turbines see correlated wind, temperature and grid conditions
   regardless of deduplication. This is inherent to the benchmark and affects the
   published baselines equally.

---

## 9. Files

| File | Purpose |
|---|---|
| `care_farmC_00_build_cache.ipynb` | Parse all 58 CSVs once into parquet. Run on a **CPU** session. |
| `care_farmC_isolated.ipynb` | Arm A. ~113 min on a P100. |
| `care_farmC_pooled.ipynb` | Arm B. ~25 min on a P100. |
| `make_fake.py`, `run_test.py` | Synthetic Farm C with the duplication pathology planted; runs both arms end-to-end on CPU in ~2 min. |
| `figs/*.svg` | The three figures above. |

Key config constants, all at the top of every notebook: `USE_AVG_ONLY`,
`VAL_SPLIT_MODE`, `VAL_BLOCK_HOURS`, `SCORE_QUANTILE`, `CRITICALITY_THRESHOLD`,
`USE_LR_PLATEAU`, `HIDDEN_LAYERS`, `LEARNING_RATE`.
