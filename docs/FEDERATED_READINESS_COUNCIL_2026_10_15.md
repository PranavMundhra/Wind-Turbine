# Federated Learning Readiness Council

Date: 15 October 2026. Repository reviewed: `37e2ea5b465cd9b9971759fc2814f21bdbecd56d`.

**Subsequent update:** the shared bridge was implemented and the supplied Kaggle
Arm D run completed the full cohort at seed 42. See the
[updated transition decision](FEDERATED_TRANSITION_2026_10_15.md). The council's
original independent assessments below are retained as the pre-implementation record.

**The work has a sound foundation for exploring federated learning. Start building and testing the federated machinery on synthetic data now. Do not yet treat the project as ready for defensible real-data federated comparisons.** A small real-data pilot becomes reasonable after the client-data, feature-schema, scoring, and comparison checks below pass.

The main issue is the bridge from the current notebooks to shared federated clients. Arm D cleans its partitions in memory, but the exported client builder does not yet apply the same rules. Parts of the planned experiment interpretation also need correction, especially the claimed FedProx endpoints.

## How the council worked

Four specialist agents reviewed the work sequentially: data integrity, statistics, federated algorithms, then software and systems. Each received a fresh context and read primary repository files. They were told not to read the earlier repository review, other council verdicts, or each other's messages. Each finished before the next started. Their verdicts were combined only afterward.

These are independent agent assessments from different technical perspectives, not four independently reproduced real-data experiments. The raw Farm C dataset and complete run artifacts were unavailable. Saved notebook outputs support some observations, but do not replace a new run.

## Individual verdicts

| Specialist | Is the direction correct | Start synthetic FL | Start real-data pilot today | Defensible research comparisons today |
| --- | --- | --- | --- | --- |
| Data integrity and leakage | Yes | Go | Not yet; finish and verify E03 | No go |
| Statistics and evaluation | Yes | Go | Not yet; resolve data and scoring gates | No go |
| Federated algorithms and optimization | Mixed; the research question is useful, but some predictions are wrong | Go | Conditional on the entry gates | No go |
| Software engineering and systems | Yes | Go | Conditional on the entry gates | No go |

"Conditional" does not mean the conditions have passed. The practical agreement is that synthetic implementation can begin now, while real-data entry still needs work. No specialist required FL to beat Arm D before starting an experiment.

## Data integrity verdict

**Reasoning:** Turbine identity is a sensible client boundary. The duplication investigation uses separate discovery and verification columns, negative controls, and informative nonzero matches. This is stronger evidence than a single hash comparison. C and D also remove matching rows before fitting scalers and models.

Arm C's saved seed-42 output shows 40,216 of 158,528 prediction rows matching fitting data before removal, and zero afterward under its fingerprint definition. Arm D additionally removes training copies of validation rows. Those are meaningful improvements.

The remaining blocker is the exported client dataset. Its builder still uses separate raw parsing and chronological splitting without the cleaning used by D. A saved message saying that integrity checks passed covers basic identifiers and counts; it does not establish separation of training, validation, and prediction data.

There is also a model-compatibility risk. Each local `Standardizer` decides which columns to keep based on that client's missingness. Different clients could keep different features. Even equal input widths are insufficient if column positions have different meanings. Actual differences in the current 22 clients were not independently measured.

**Expert verdict:** Continue the direction. Start synthetic FL now. Do not start meaningful real-data comparisons until the exported clients pass explicit checks.

Primary evidence: [audit notebook](../notebooks/02_leakage_audit.ipynb), cells 2 through 8 and 13; [Arm C](../notebooks/12_arm_pooled_dedup.ipynb), cells 18 and 20; [Arm D](../notebooks/13_arm_local_turbine.ipynb), preprocessing and cell 18; [client builder](../notebooks/20_federated_clients.ipynb); and [E03 requirements](experiments/E03_client_dataset.md).

## Statistics verdict

**Reasoning:** C and D provide useful starting observations, but they do not yet isolate why performance differs. C's reported three-seed CARE is 0.326 plus or minus 0.288; D's is 0.611 plus or minus 0.013. C's instability matters. However, changing C's seed changes both the validation turbines and training randomness. It does not independently measure either source of variation.

The current metrics also need attention. The evaluation mask includes a power-curve training filter, so some normal-status low-power timestamps cannot contribute to the score or criticality counter. In addition, a normal event with no eligible timestamps receives Accuracy 1.0. The reviewer reproduced an unsupported event contributing perfect Accuracy. Its effect on existing results is unknown.

The project correctly recognizes that 58 events belong to 22 turbines. Events from the same turbine are related. Uncertainty should therefore be estimated by resampling turbines while keeping their events together, then recomputing the complete CARE score.

Two historical runs differing by 0.013 do not establish a universal noise floor. A difference smaller than that value is not automatically meaningless, and a larger difference is not automatically convincing. Matched runs and uncertainty estimates are needed.

**Expert verdict:** The research direction is sensible. Synthetic FL is appropriate now. A real-data pilot needs a stable scoring contract and a matched comparator; strong conclusions need repeated paired runs and uncertainty estimates.

Primary evidence: [Arm C scoring](../notebooks/12_arm_pooled_dedup.ipynb), cells 8 and 11; [C results](experiments/C_pooled_dedup.md); [D results](experiments/D_local_per_turbine.md); and [seed study](experiments/S_seed_robustness.md).

## Federated algorithm verdict

**Reasoning:** Sharing an encoder while keeping a local decoder is a credible hypothesis. D is a useful local comparator because its client unit matches the proposed turbine clients. Full participation and weighting updates by actual fitting-row counts are reasonable choices for this simulation.

However, two planned interpretations are unsound.

First, F1 changes more than the training location. It proposes D's partitions, local scaling, and local thresholds. C uses different validation membership, global scaling, and a global threshold. Therefore, F1 being above or below C cannot by itself validate the FL code or identify the cost of federation. Add a centralized comparator that uses exactly the same client partitions and transformations as F1.

Second, the roadmap describes FedProx as a dial from independent local models D at zero `mu` toward centralized C at large `mu`. **At zero `mu`, FedProx reduces to FedAvg under the same local solver and aggregation settings.** Clients still receive and update a shared model. Large `mu` restricts local movement; it does not guarantee C's result. The sweep should test optimization behavior, not assume those endpoints. This correction is supported by the original [FedProx paper](https://proceedings.mlsys.org/paper_files/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf).

The planned server metric contract is also incomplete. Event confusion counts recover Reliability, but do not contain timing information for Earliness or the event-level information needed for Coverage and Accuracy. Clients must return unrounded component values and support, or equivalent sufficient statistics. The server must apply the final CARE rules once.

**Expert verdict:** Mixed direction because the experimental question is worthwhile but the algorithm interpretation needs repair. Start synthetic implementation. Permit a real-data pilot after the entry gates. Do not make research comparisons under the current F1 and F3 decision rules.

Primary evidence: [F1 specification](experiments/F1_fedavg.md), [F3 specification](experiments/F3_fedprox_sweep.md), [FedProx config](../configs/fed_fedprox.yaml), [local-scaling decision](adr/0002-local-scaler-and-threshold.md), and [Arm D](../notebooks/13_arm_local_turbine.ipynb), preprocessing and scoring cells.

## Software and systems verdict

**Reasoning:** The repository has useful configuration, result-writing, seed-recording, and audit helpers. D provides a good reference for extracting shared client preparation. Separating TensorFlow from basic scoring and audit dependencies also helps testing.

The federated package is still a placeholder, which is acceptable at this stage. The concern is ensuring that the new implementation has clear contracts rather than copying another notebook pipeline. Current federated configs contain unresolved choices such as `local_epochs: TBD`. Before a real run, they need numeric validated values and explicit rules for parameter order, optimizer state, weighting, and local training.

Run manifests record useful metadata, but the cache manifest does not identify actual input contents. Hashing a description of columns and counts cannot detect changed parquet data. Also, this checkout has no saved run directories beyond `.gitkeep`, so complete provenance for the reported results cannot be inspected.

The privacy roadmap correctly separates ordinary FL from formal privacy. Production security, secure aggregation, and differential privacy need not block a local simulation. They are later requirements for privacy claims; those requirements must also cover metrics and other client disclosures.

**Expert verdict:** Yes on direction. Start synthetic development. Real-data entry is conditional on verified shared preparation, explicit contracts, and traceable artifacts. Research comparisons are not ready today.

Primary evidence: [Arm D](../notebooks/13_arm_local_turbine.ipynb), cells 7 and 18; [client builder](../notebooks/20_federated_clients.ipynb); [configuration validation](../src/windturbine/config.py); [run records](../src/windturbine/runs.py); [F1](experiments/F1_fedavg.md); and [privacy phase](experiments/F4_private_fl.md).

## Points checked after the independent reviews

The statistics reviewer requested clarification of CARE's zero-score rule and Earliness weighting. These concerns were checked against the benchmark paper rather than automatically treated as bugs.

The paper describes zero CARE when no anomaly events are detected. The notebook's event-level zero guard is consistent with that interpretation; the documentation's phrase "nothing flagged anywhere" is less precise. The paper also describes full weight in the first half of an event and a linear fall to zero in the second half, matching the implemented weight shape. These two points should be documented and regression-tested, but they are not confirmed formula defects or independent reasons to block FL. The evaluation-mask and empty-support issues remain separate concerns. See [CARE to Compare, sections on Earliness and CARE](https://publica-rest.fraunhofer.de/server/api/core/bitstreams/6816cb77-9c66-424f-a41d-704432079311/content), pages 8 through 10.

The reviewers suggested different numerical tolerances for toy equivalence. Those are proposed acceptance criteria, not measured properties of existing FL code. The project has no implemented federated pipeline to certify yet.

## What can start now

Begin roadmap phase 1: a small synthetic client-server simulation. Use deliberately different client sizes and distributions, with known expected results. This work can proceed while the real-data preparation is corrected.

The first implementation checks should be:

1. Start every client from identical weights. With full participation and one full-batch gradient step using plain SGD, match a centralized gradient step on the same transformed data. Use no momentum, Adam, noise, or callbacks for this check. For a small float64 fixture, predeclare a maximum absolute parameter error such as `1e-10`; use and justify a suitable tolerance for other numerical settings.
2. Weight updates by actual fitting rows after cleaning. Test unequal client sizes so an accidental unweighted mean fails visibly.
3. Reconstruct CARE through simulated client reports and compare every component and the final score with direct evaluation. Include clients with only normal events, clients with only anomaly events, absent support, and no detected events. Do not average client CARE scores.
4. Reject mismatched feature identities, order, or parameter shapes before aggregation. Test a case where widths match but feature orders differ.

The restricted gradient-step identity follows from the [original FedAvg formulation](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf). Passing it will validate that restricted case, not multi-epoch training with Adam.

## Gates before a real-data pilot

| Gate | Evidence required to pass |
| --- | --- |
| Shared client preparation | A single preparation path used by D and FL produces 22 clients and maps all 58 events, or fails with an explicit exclusion record. The old disk builder must not silently remain the source. |
| Verified separation | Exported data have zero declared fingerprint intersections between fitting and validation, fitting and prediction, and validation and prediction. Check the global fitting corpus against all evaluated prediction sets as well. Save before/after counts and the fingerprint definition. |
| Common model schema | Every client has identical ordered transformed features and compatible shared parameter names and shapes. Local means and scales may differ. Missing features must follow a declared policy rather than silently changing model inputs. |
| Stable scoring | Separate training eligibility from the intended evaluation rule. Handle zero eligible support explicitly. Pass complete evaluator and client-server metric parity tests. |
| Matched centralized comparator | Train centrally on the union of exactly the FL fitting partitions, using the same local transformations and calibration policy. Record initialization, model selection, and training budget. Keep historical C as an additional comparison. |
| Executable and traceable pilot | Resolve all required config values. Save input-content hashes, schema, split membership, source revision, environment, retained counts, and results. Complete a deterministic smoke round with finite updates, followed by a short multi-round run. |

These are pilot-entry checks. They do not require a full FedProx sweep, production deployment, formal privacy, or proof that FL improves CARE. A pilot may legitimately find that sharing does not help.

## Gates before research claims

After the pilot works, repeat matched comparisons with at least three paired seeds. Keep validation-split randomness separate from optimization randomness. Save per-event metrics and row-level/calibration scores so results can be checked and rescored.

Report all CARE components and event confusion counts. Estimate paired differences by resampling turbines and recomputing the full score. If claiming that two methods are equivalent, define a practically acceptable difference in advance; an insignificant result alone does not establish equivalence.

Specify personalization details before evaluating final events: which layers are shared, how those names map to model layers, whether local decoders persist between rounds, training order, optimizer-state handling, and which final model is evaluated. Select `mu`, local epochs, and personalization settings using validation evidence, not the final event outcomes.

Publish the small run artifacts and correct the F1 and F3 interpretation before making claims about the cost of federation, the benefit of personalization, or a local-to-global performance curve. Formal privacy claims additionally require the choices described in the privacy phase, including a threat model and accounting for disclosed metrics.

## Basis and limits of the decision

The data reviewer ran the suite with UTF-8 enabled and obtained 30 passing tests; ordinary Windows decoding caused five registry failures. The systems reviewer ran 11 targeted configuration and result-contract tests successfully. These checks support the existing helpers. They do not establish complete client-data separation, federated update correctness, or real-data metric parity, which still need dedicated tests.

The council supports the underlying question: can turbines benefit from sharing a model or encoder while keeping useful local behavior? The current work gives enough evidence to build that experiment. It does not yet give enough control over data, scoring, and comparison to interpret the final answer reliably.

**Decision: proceed with synthetic federated implementation now, complete the pilot-entry gates next, then run a small real-data pilot before committing to the full federated experiment program.**
