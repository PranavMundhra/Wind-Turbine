# Experiment graph

The headline experiments (F2, F3) sit behind three unrun gates: **C**, **D** (which needs the leak-free client dataset E03) and **F1**. The leakage audit E02 is the one finished gate, and it feeds all of them.

<!-- generated:start -->
```mermaid
flowchart LR
  subgraph lane_r["Phase R: data and evidence"]
    E00["E00: Data cache<br/>done · unblocks 11"]
    E01["E01: Exploratory data analysis<br/>done · unblocks 6"]
    E02["E02: Leakage audit<br/>done · unblocks 9"]
  end
  subgraph lane_done["Done: baselines"]
    A["A: Isolated, one model per event<br/>done-unverified · unblocks 2"]
    B["B: Pooled, one fleet-wide model<br/>done-unverified · unblocks 6"]
  end
  subgraph lane_0["Phase 0: clean baselines, client data, noise"]
    E03["E03: Leak-free per-turbine client dataset<br/>partial · unblocks 5"]
    C["C: Pooled, leaked rows removed<br/>planned · unblocks 4"]
    D["D: Local per-turbine, no collaboration<br/>planned · unblocks 3"]
    S["S: Seed robustness and confidence intervals<br/>planned · unblocks 0"]
  end
  subgraph lane_fl["Phases 2-5: federated arms"]
    F1["F1: FedAvg, global model, local thresholds<br/>planned · unblocks 3"]
    F2["F2: Personalised FL, shared encoder, local decoder<br/>planned · unblocks 1"]
    F3["F3: FedProx mu sweep<br/>planned · unblocks 1"]
    F4["F4: Secure aggregation and differential privacy<br/>planned · unblocks 0"]
  end
  subgraph lane_any["Any phase"]
    X["X: Ablations<br/>planned · unblocks 0"]
  end
  E00 --> E03
  E01 --> E03
  E02 ==> E03
  E00 --> A
  E00 --> B
  E02 ==> B
  B --> C
  E02 ==> C
  E03 ==> D
  E03 ==> F1
  C ==> F1
  F1 ==> F2
  D ==> F2
  F1 ==> F3
  D ==> F3
  F2 --> F4
  F3 --> F4
  A --> S
  B --> S
  A --> X
  B -.-|vs| A
  C -.-|vs| A
  D -.-|vs| A
  D -.-|vs| C
  F1 -.-|vs| D
  F2 -.-|vs| C
  F2 -.-|vs| A
  F3 -.-|vs| C
  F3 -.-|vs| F2
  X -.-|vs| D
  X -.-|vs| F1
  classDef gate fill:#fff4e5,stroke:#d97706,stroke-width:3px,color:#111
  classDef headline fill:#e0ecff,stroke:#2563eb,stroke-width:3px,color:#111
  classDef baseline fill:#eeeeee,stroke:#555,stroke-width:1.5px,color:#111
  classDef supporting fill:#ffffff,stroke:#999,stroke-width:1px,color:#111
  class E02,E03,C,D,F1,S gate
  class F2,F3 headline
  class A,B baseline
  class E00,E01,F4,X supporting
```
<!-- generated:end -->

**How to read it.** Lanes run left to right in the order the work must happen. A thick arrow leaves a gate: the next experiment cannot be trusted until the gate's result is in. A thin arrow is a data dependency. A dashed "vs" line marks the baseline an experiment is judged against. "Unblocks N" counts every experiment downstream, which is the simplest measure of how much rides on that node. Colours: orange border = gate, blue = headline, grey = done baseline, white = supporting.

## What each comparison answers

| Comparison | Question it answers | Status |
| --- | --- | --- |
| B vs A | Is one fleet model better than per-event models? (confounded: contamination and specialisation differ at once) | done: 0.583 vs 0.628 |
| C vs B | What was contamination worth? | needs C |
| C vs A | How much does specialisation matter, cleanly? | needs C |
| D vs A | Is per-turbine specialisation as good as per-event? | needs D |
| D vs C | Local-only vs centralised: the two ends of the federated dial | needs C, D |
| F1 vs C | Does federation itself cost anything? (predicted: no) | needs F1 |
| F2 vs max(C, D) | Does personalised FL beat both ends? (headline) | needs F2 |
| F3 curve between D and C | Where on the sharing dial is the optimum? (headline) | needs F3 |
| F4 vs best of F2 / F3 | What does formal privacy cost? | needs F4 |

## Why D was added

The original plan bracketed the federated work between Arm A and Arm B. With turbines as clients and leaked rows removed, the honest brackets are **D** (each turbine alone) and **C** (everything centralised, clean). A is per-event and B is contaminated, so neither is an endpoint of the federated dial.
