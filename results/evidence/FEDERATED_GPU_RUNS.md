# F1-F4 federated runs (local RTX 5070 Ti, seed 42)

Executed on Windows with Keras 3 on the PyTorch backend (CUDA 12.8): TensorFlow >= 2.11 has no native-Windows GPU
support. The training step (MSE, Keras Adam formula, FedProx term, exact per-example DP-SGD clipping) was ported
to torch in `src/windturbine/models/autoencoder.py` and replayed in CUDA graphs; the preflight gates (Flower
FedAvg = centralized SGD, FedProx mu=0 = FedAvg, open-clip DP = plain training, CARE parity) passed on GPU.
Dataset: local E03 rebuild, manifest `1a43906d...` (same per-turbine counts as the Kaggle export `bb2ed3ff...`).
Single seed: differences of a few thousandths of CARE are within run-to-run noise (historical pooled reruns
differed by 0.013). Reference: Arm D (Kaggle, seed 42) CARE **0.639852**.

| Exp | Branch | CARE | Coverage | Earliness | Reliability | Accuracy | TP/FP | Selected round | epsilon | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 | centralized  | 0.640495 | 0.3161 | 0.2438 | 0.7229 | 0.9598 | 12/2 | 2 |  | `20261026T171256_313611Z` |
| F1 | fedavg  | 0.636185 | 0.3527 | 0.2942 | 0.6542 | 0.9399 | 14/6 | 26 |  | `20261026T171256_313611Z` |
| F2 | fedper  | 0.625111 | 0.2957 | 0.2512 | 0.6627 | 0.9580 | 11/3 | 42 |  | `20261026T185302_930994Z` |
| F3 | fedprox mu=0.001 | 0.631631 | 0.3368 | 0.2988 | 0.6566 | 0.9330 | 13/5 | 94 |  | `mu_0.001_20261026T185008_202732Z` |
| F3 | fedprox mu=0.01 | 0.636157 | 0.3509 | 0.3093 | 0.6542 | 0.9332 | 14/6 | 94 |  | `mu_0.01_20261026T205303_936568Z` |
| F3 | fedprox mu=0.1 | 0.638510 | 0.3537 | 0.2924 | 0.6566 | 0.9450 | 13/5 | 99 |  | `mu_0.1_20261026T171256_320778Z` |
| F3 | fedprox mu=0.0 | 0.636243 | 0.3470 | 0.2871 | 0.6566 | 0.9453 | 13/5 | 26 |  | `mu_0_20261026T171256_422119Z` |
| F3 | fedprox mu=10.0 | 0.590075 | 0.2396 | 0.1458 | 0.6044 | 0.9803 | 11/5 | 100 |  | `mu_10_20261026T210319_455397Z` |
| F3 | fedprox mu=1.0 | 0.615583 | 0.2659 | 0.1881 | 0.6667 | 0.9786 | 10/2 | 100 |  | `mu_1_20261026T190002_276151Z` |
| F4 | fedper_private noise=0.5 | 0.590153 | 0.2401 | 0.1459 | 0.6044 | 0.9802 | 11/5 | 88 | 22.22 | `noise_0.5_20261026T210135_259322Z` |
| F4 | fedper_private noise=0.0 | 0.614167 | 0.3174 | 0.2322 | 0.6024 | 0.9594 | 10/4 | 98 |  | `noise_0_20261026T195720_060878Z` |
| F4 | fedper_private noise=1.0 | 0.590110 | 0.2397 | 0.1458 | 0.6044 | 0.9803 | 11/5 | 87 | 2.82 | `noise_1_20261026T223000_895383Z` |
| F4 | fedper_private noise=2.0 | 0.595562 | 0.2392 | 0.1451 | 0.6322 | 0.9807 | 11/4 | 0 | 1.05 | `noise_2_20261026T233328_266806Z` |
| F4 | fedper_private noise=4.0 | 0.595562 | 0.2392 | 0.1451 | 0.6322 | 0.9807 | 11/4 | 0 | 0.47 | `noise_4_20261027T002554_473271Z` |

## Observations

- F1 FedAvg (0.636), F3 with mu <= 0.1 (0.632-0.639) and the centralized comparator (0.640) all sit within
  about 0.008 CARE of Arm D (0.640). One seed cannot separate them.
- F2 FedPer (0.625) is slightly lower; large proximal pull hurts (mu = 1: 0.616, mu = 10: 0.590).
- F4 with any DP noise (noise >= 0.5) lands at CARE ~0.59 with validation MSE ~1.14 versus ~0.38 for trained
  models: at this privacy level the models barely learn. With noise >= 2, Adam training is unstable
  (validation MSE spikes into the hundreds or thousands) and checkpoint selection keeps round 0, the initial
  weights. The executed notebooks for those runs stop at the post-training sharing check, which did not allow
  a round-0 selection; that check is fixed in notebooks 31-33. The saved run folders are complete and verified.
- Next: paired seeds before any claim; DP-SGD with plain SGD or a lower learning rate for usable epsilon.

Executed notebooks: `notebooks/executed/`. Weights and per-row scores stay in `results/runs/` (git-ignored).
