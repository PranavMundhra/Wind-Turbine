window.WT_DATA = {
 "meta": {
  "reference": "D",
  "dataset": "CARE to Compare, Wind Farm C (version 6)",
  "n_events": 58,
  "n_anomaly": 27,
  "n_normal": 31,
  "n_turbines": 22,
  "fingerprint": "1d5c65301069"
 },
 "groups": {
  "arms": [
   "D",
   "central_sequential",
   "F1",
   "F2"
  ],
  "mu": [
   "F3_mu0",
   "F3_mu0.001",
   "F3_mu0.01",
   "F3_mu0.1",
   "F3_mu1",
   "F3_mu10"
  ],
  "priv": [
   "F4_sigma0",
   "F4_sigma0.5",
   "F4_sigma1",
   "F4_sigma2",
   "F4_sigma4"
  ]
 },
 "events": [
  {
   "id": 1,
   "asset": 53,
   "anomaly": false
  },
  {
   "id": 4,
   "asset": 34,
   "anomaly": true
  },
  {
   "id": 5,
   "asset": 32,
   "anomaly": true
  },
  {
   "id": 6,
   "asset": 38,
   "anomaly": false
  },
  {
   "id": 8,
   "asset": 50,
   "anomaly": false
  },
  {
   "id": 9,
   "asset": 55,
   "anomaly": true
  },
  {
   "id": 11,
   "asset": 43,
   "anomaly": true
  },
  {
   "id": 12,
   "asset": 2,
   "anomaly": true
  },
  {
   "id": 15,
   "asset": 12,
   "anomaly": true
  },
  {
   "id": 16,
   "asset": 53,
   "anomaly": true
  },
  {
   "id": 18,
   "asset": 34,
   "anomaly": true
  },
  {
   "id": 20,
   "asset": 53,
   "anomaly": false
  },
  {
   "id": 28,
   "asset": 52,
   "anomaly": true
  },
  {
   "id": 29,
   "asset": 13,
   "anomaly": false
  },
  {
   "id": 30,
   "asset": 16,
   "anomaly": true
  },
  {
   "id": 31,
   "asset": 35,
   "anomaly": true
  },
  {
   "id": 32,
   "asset": 42,
   "anomaly": false
  },
  {
   "id": 33,
   "asset": 43,
   "anomaly": true
  },
  {
   "id": 35,
   "asset": 53,
   "anomaly": true
  },
  {
   "id": 36,
   "asset": 2,
   "anomaly": false
  },
  {
   "id": 37,
   "asset": 23,
   "anomaly": false
  },
  {
   "id": 39,
   "asset": 52,
   "anomaly": true
  },
  {
   "id": 41,
   "asset": 33,
   "anomaly": false
  },
  {
   "id": 43,
   "asset": 52,
   "anomaly": false
  },
  {
   "id": 44,
   "asset": 44,
   "anomaly": true
  },
  {
   "id": 46,
   "asset": 16,
   "anomaly": false
  },
  {
   "id": 47,
   "asset": 21,
   "anomaly": true
  },
  {
   "id": 48,
   "asset": 35,
   "anomaly": false
  },
  {
   "id": 49,
   "asset": 33,
   "anomaly": true
  },
  {
   "id": 50,
   "asset": 12,
   "anomaly": false
  },
  {
   "id": 54,
   "asset": 52,
   "anomaly": false
  },
  {
   "id": 55,
   "asset": 50,
   "anomaly": true
  },
  {
   "id": 56,
   "asset": 34,
   "anomaly": false
  },
  {
   "id": 57,
   "asset": 55,
   "anomaly": false
  },
  {
   "id": 58,
   "asset": 35,
   "anomaly": false
  },
  {
   "id": 59,
   "asset": 32,
   "anomaly": false
  },
  {
   "id": 60,
   "asset": 53,
   "anomaly": false
  },
  {
   "id": 61,
   "asset": 43,
   "anomaly": false
  },
  {
   "id": 62,
   "asset": 21,
   "anomaly": false
  },
  {
   "id": 63,
   "asset": 56,
   "anomaly": false
  },
  {
   "id": 64,
   "asset": 15,
   "anomaly": false
  },
  {
   "id": 65,
   "asset": 16,
   "anomaly": false
  },
  {
   "id": 66,
   "asset": 12,
   "anomaly": true
  },
  {
   "id": 67,
   "asset": 35,
   "anomaly": true
  },
  {
   "id": 70,
   "asset": 23,
   "anomaly": true
  },
  {
   "id": 75,
   "asset": 44,
   "anomaly": false
  },
  {
   "id": 76,
   "asset": 53,
   "anomaly": true
  },
  {
   "id": 78,
   "asset": 15,
   "anomaly": true
  },
  {
   "id": 79,
   "asset": 16,
   "anomaly": true
  },
  {
   "id": 80,
   "asset": 23,
   "anomaly": false
  },
  {
   "id": 81,
   "asset": 38,
   "anomaly": true
  },
  {
   "id": 85,
   "asset": 14,
   "anomaly": false
  },
  {
   "id": 88,
   "asset": 55,
   "anomaly": false
  },
  {
   "id": 89,
   "asset": 5,
   "anomaly": false
  },
  {
   "id": 90,
   "asset": 56,
   "anomaly": true
  },
  {
   "id": 91,
   "asset": 42,
   "anomaly": true
  },
  {
   "id": 93,
   "asset": 43,
   "anomaly": false
  },
  {
   "id": 94,
   "asset": 34,
   "anomaly": false
  }
 ],
 "experiments": {
  "D": {
   "label": "Local training (D)",
   "group": "arms",
   "note": "Each turbine trains alone. Seed 42 was trained on Kaggle with TensorFlow; the federated runs used PyTorch.",
   "seeds": [
    42
   ],
   "metrics": {
    "care": 0.639852,
    "coverage": 0.369207,
    "earliness": 0.294633,
    "reliability": 0.654206,
    "accuracy": 0.940607
   },
   "sd": null,
   "detected": 14.0,
   "false_alarms": 6.0,
   "ci": null,
   "epsilon": null,
   "path": "results/paper/D/s42/",
   "detected_events": [
    5,
    9,
    11,
    16,
    29,
    31,
    32,
    37,
    39,
    44,
    47,
    49,
    55,
    57,
    59,
    66,
    70,
    76,
    91,
    94
   ]
  },
  "central_sequential": {
   "label": "Centralised comparator",
   "group": "arms",
   "note": "The paper's comparator trains one turbine at a time within each epoch and drifted toward the last turbine.",
   "seeds": [
    42
   ],
   "metrics": {
    "care": 0.640495,
    "coverage": 0.316136,
    "earliness": 0.24376,
    "reliability": 0.722892,
    "accuracy": 0.959842
   },
   "sd": null,
   "detected": 12.0,
   "false_alarms": 2.0,
   "ci": {
    "diff": 0.000643,
    "lo": -0.025852,
    "hi": 0.028513,
    "excludes_zero": false,
    "draws": 200
   },
   "epsilon": null,
   "path": "results/paper/central_sequential/s42/",
   "detected_events": [
    5,
    9,
    11,
    16,
    31,
    32,
    39,
    44,
    47,
    49,
    55,
    59,
    66,
    91
   ]
  },
  "F1": {
   "label": "FedAvg",
   "group": "arms",
   "note": "One global model, local scalers and thresholds.",
   "seeds": [
    42
   ],
   "metrics": {
    "care": 0.636185,
    "coverage": 0.352729,
    "earliness": 0.29421,
    "reliability": 0.654206,
    "accuracy": 0.939891
   },
   "sd": null,
   "detected": 14.0,
   "false_alarms": 6.0,
   "ci": {
    "diff": -0.003667,
    "lo": -0.040277,
    "hi": 0.042168,
    "excludes_zero": false,
    "draws": 200
   },
   "epsilon": null,
   "path": "results/paper/F1/s42/",
   "detected_events": [
    4,
    5,
    9,
    11,
    16,
    20,
    31,
    32,
    37,
    39,
    44,
    47,
    49,
    50,
    55,
    57,
    59,
    66,
    76,
    91
   ]
  },
  "F2": {
   "label": "Personalised (FedPer)",
   "group": "arms",
   "note": "Shared encoder, a decoder kept on each turbine.",
   "seeds": [
    42
   ],
   "metrics": {
    "care": 0.625111,
    "coverage": 0.295712,
    "earliness": 0.25118,
    "reliability": 0.662651,
    "accuracy": 0.958006
   },
   "sd": null,
   "detected": 11.0,
   "false_alarms": 3.0,
   "ci": {
    "diff": -0.014741,
    "lo": -0.072865,
    "hi": 0.030542,
    "excludes_zero": false,
    "draws": 200
   },
   "epsilon": null,
   "path": "results/paper/F2/s42/",
   "detected_events": [
    5,
    9,
    11,
    16,
    29,
    32,
    39,
    44,
    47,
    49,
    59,
    66,
    76,
    91
   ]
  },
  "F3_mu0": {
   "label": "μ = 0",
   "group": "mu",
   "note": "No proximal term, so this is FedAvg.",
   "seeds": [
    42
   ],
   "metrics": {
    "care": 0.636243,
    "coverage": 0.34696,
    "earliness": 0.28708,
    "reliability": 0.656566,
    "accuracy": 0.945305
   },
   "sd": null,
   "detected": 13.0,
   "false_alarms": 5.0,
   "ci": {
    "diff": -0.003609,
    "lo": -0.044563,
    "hi": 0.031648,
    "excludes_zero": false,
    "draws": 200
   },
   "epsilon": null,
   "path": "results/paper/F3_mu0/s42/",
   "detected_events": [
    5,
    9,
    11,
    16,
    20,
    31,
    32,
    37,
    39,
    44,
    47,
    49,
    50,
    55,
    59,
    66,
    76,
    91
   ]
  },
  "F3_mu0.001": {
   "label": "μ = 0.001",
   "group": "mu",
   "note": "",
   "seeds": [
    42
   ],
   "metrics": {
    "care": 0.631631,
    "coverage": 0.336754,
    "earliness": 0.298763,
    "reliability": 0.656566,
    "accuracy": 0.933036
   },
   "sd": null,
   "detected": 13.0,
   "false_alarms": 5.0,
   "ci": {
    "diff": -0.008221,
    "lo": -0.054252,
    "hi": 0.041884,
    "excludes_zero": false,
    "draws": 200
   },
   "epsilon": null,
   "path": "results/paper/F3_mu0.001/s42/",
   "detected_events": [
    4,
    5,
    9,
    11,
    16,
    20,
    32,
    37,
    39,
    44,
    47,
    49,
    50,
    55,
    57,
    66,
    76,
    91
   ]
  },
  "F3_mu0.01": {
   "label": "μ = 0.01",
   "group": "mu",
   "note": "",
   "seeds": [
    42
   ],
   "metrics": {
    "care": 0.636157,
    "coverage": 0.350939,
    "earliness": 0.309287,
    "reliability": 0.654206,
    "accuracy": 0.933177
   },
   "sd": null,
   "detected": 14.0,
   "false_alarms": 6.0,
   "ci": {
    "diff": -0.003695,
    "lo": -0.04406,
    "hi": 0.042288,
    "excludes_zero": false,
    "draws": 200
   },
   "epsilon": null,
   "path": "results/paper/F3_mu0.01/s42/",
   "detected_events": [
    4,
    5,
    9,
    11,
    16,
    20,
    31,
    32,
    37,
    39,
    44,
    47,
    49,
    50,
    55,
    57,
    59,
    66,
    76,
    91
   ]
  },
  "F3_mu0.1": {
   "label": "μ = 0.1",
   "group": "mu",
   "note": "",
   "seeds": [
    42
   ],
   "metrics": {
    "care": 0.63851,
    "coverage": 0.35365,
    "earliness": 0.292363,
    "reliability": 0.656566,
    "accuracy": 0.944985
   },
   "sd": null,
   "detected": 13.0,
   "false_alarms": 5.0,
   "ci": {
    "diff": -0.001342,
    "lo": -0.042871,
    "hi": 0.032621,
    "excludes_zero": false,
    "draws": 200
   },
   "epsilon": null,
   "path": "results/paper/F3_mu0.1/s42/",
   "detected_events": [
    5,
    9,
    11,
    16,
    20,
    31,
    32,
    37,
    39,
    44,
    47,
    49,
    50,
    55,
    59,
    66,
    76,
    91
   ]
  },
  "F3_mu1": {
   "label": "μ = 1",
   "group": "mu",
   "note": "A strong pull toward the global model reduced Coverage and Earliness.",
   "seeds": [
    42
   ],
   "metrics": {
    "care": 0.615583,
    "coverage": 0.265932,
    "earliness": 0.188083,
    "reliability": 0.666667,
    "accuracy": 0.978616
   },
   "sd": null,
   "detected": 10.0,
   "false_alarms": 2.0,
   "ci": {
    "diff": -0.024269,
    "lo": -0.100177,
    "hi": 0.028569,
    "excludes_zero": false,
    "draws": 200
   },
   "epsilon": null,
   "path": "results/paper/F3_mu1/s42/",
   "detected_events": [
    5,
    9,
    11,
    31,
    32,
    39,
    44,
    47,
    49,
    59,
    66,
    91
   ]
  },
  "F3_mu10": {
   "label": "μ = 10",
   "group": "mu",
   "note": "The proximal term dominates training.",
   "seeds": [
    42
   ],
   "metrics": {
    "care": 0.590075,
    "coverage": 0.239577,
    "earliness": 0.145827,
    "reliability": 0.604396,
    "accuracy": 0.980288
   },
   "sd": null,
   "detected": 11.0,
   "false_alarms": 5.0,
   "ci": {
    "diff": -0.049776,
    "lo": -0.130364,
    "hi": 0.000933,
    "excludes_zero": false,
    "draws": 200
   },
   "epsilon": null,
   "path": "results/paper/F3_mu10/s42/",
   "detected_events": [
    5,
    9,
    11,
    31,
    32,
    37,
    39,
    44,
    47,
    49,
    59,
    61,
    66,
    67,
    75,
    91
   ]
  },
  "F4_sigma0": {
   "label": "σ = 0",
   "group": "priv",
   "note": "Secure aggregation only, no differential-privacy noise.",
   "seeds": [
    42
   ],
   "metrics": {
    "care": 0.614167,
    "coverage": 0.317369,
    "earliness": 0.2322,
    "reliability": 0.60241,
    "accuracy": 0.959428
   },
   "sd": null,
   "detected": 10.0,
   "false_alarms": 4.0,
   "ci": {
    "diff": -0.025685,
    "lo": -0.077216,
    "hi": 0.01598,
    "excludes_zero": false,
    "draws": 200
   },
   "epsilon": null,
   "path": "results/paper/F4_sigma0/s42/",
   "detected_events": [
    5,
    9,
    11,
    16,
    32,
    39,
    44,
    47,
    49,
    50,
    59,
    61,
    66,
    91
   ]
  },
  "F4_sigma0.5": {
   "label": "σ = 0.5",
   "group": "priv",
   "note": "With noise the models barely trained.",
   "seeds": [
    42
   ],
   "metrics": {
    "care": 0.590153,
    "coverage": 0.240093,
    "earliness": 0.145948,
    "reliability": 0.604396,
    "accuracy": 0.980164
   },
   "sd": null,
   "detected": 11.0,
   "false_alarms": 5.0,
   "ci": {
    "diff": -0.049699,
    "lo": -0.130117,
    "hi": 0.000903,
    "excludes_zero": false,
    "draws": 200
   },
   "epsilon": 22.223,
   "path": "results/paper/F4_sigma0.5/s42/",
   "detected_events": [
    5,
    9,
    11,
    31,
    32,
    37,
    39,
    44,
    47,
    49,
    59,
    61,
    66,
    67,
    75,
    91
   ]
  },
  "F4_sigma1": {
   "label": "σ = 1",
   "group": "priv",
   "note": "With noise the models barely trained.",
   "seeds": [
    42
   ],
   "metrics": {
    "care": 0.59011,
    "coverage": 0.239741,
    "earliness": 0.145751,
    "reliability": 0.604396,
    "accuracy": 0.980331
   },
   "sd": null,
   "detected": 11.0,
   "false_alarms": 5.0,
   "ci": {
    "diff": -0.049742,
    "lo": -0.13021,
    "hi": 0.000982,
    "excludes_zero": false,
    "draws": 200
   },
   "epsilon": 2.82,
   "path": "results/paper/F4_sigma1/s42/",
   "detected_events": [
    5,
    9,
    11,
    31,
    32,
    37,
    39,
    44,
    47,
    49,
    59,
    61,
    66,
    67,
    75,
    91
   ]
  },
  "F4_sigma2": {
   "label": "σ = 2",
   "group": "priv",
   "note": "Training was unstable and checkpoint selection kept the initial weights.",
   "seeds": [
    42
   ],
   "metrics": {
    "care": 0.595562,
    "coverage": 0.239238,
    "earliness": 0.145062,
    "reliability": 0.632184,
    "accuracy": 0.980663
   },
   "sd": null,
   "detected": 11.0,
   "false_alarms": 4.0,
   "ci": {
    "diff": -0.04429,
    "lo": -0.120262,
    "hi": 0.00262,
    "excludes_zero": false,
    "draws": 200
   },
   "epsilon": 1.048,
   "path": "results/paper/F4_sigma2/s42/",
   "detected_events": [
    5,
    9,
    11,
    31,
    32,
    39,
    44,
    47,
    49,
    59,
    61,
    66,
    67,
    75,
    91
   ]
  },
  "F4_sigma4": {
   "label": "σ = 4",
   "group": "priv",
   "note": "Training was unstable and checkpoint selection kept the initial weights.",
   "seeds": [
    42
   ],
   "metrics": {
    "care": 0.595562,
    "coverage": 0.239238,
    "earliness": 0.145062,
    "reliability": 0.632184,
    "accuracy": 0.980663
   },
   "sd": null,
   "detected": 11.0,
   "false_alarms": 4.0,
   "ci": {
    "diff": -0.04429,
    "lo": -0.120262,
    "hi": 0.00262,
    "excludes_zero": false,
    "draws": 200
   },
   "epsilon": 0.474,
   "path": "results/paper/F4_sigma4/s42/",
   "detected_events": [
    5,
    9,
    11,
    31,
    32,
    39,
    44,
    47,
    49,
    59,
    61,
    66,
    67,
    75,
    91
   ]
  }
 }
};
