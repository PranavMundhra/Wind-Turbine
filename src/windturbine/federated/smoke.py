"""Tiny synthetic two-turbine export that exercises the federated notebooks on CPU.

The rows only exist to run the export, training and readback path end to end.
Its results are execution checks, not evidence about wind turbines.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from windturbine.config import load_config
from windturbine.data.clients import build_client_dataset
from windturbine.runs import file_hash


def build_synthetic_clients(work_root, config_dir, *, farm, source_identity=None):
    """Write raw CARE-format CSVs for 2 turbines / 4 events and export them.

    Returns the client dataset folder plus the Config field and ``extra``
    overrides a smoke run needs (small model, two rounds, pinned manifest).
    """
    work_root, config_dir = Path(work_root), Path(config_dir)
    farm_dir = work_root / "synthetic_care" / farm
    datasets = farm_dir / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    events = []
    for eid in range(1, 5):
        aid = (eid - 1) // 2 + 1
        times = pd.date_range("2024-01-01", periods=100, freq="10min")
        values = np.arange(100, dtype=float) + eid * 1000
        frame = pd.DataFrame({"time_stamp": times, "asset_id": aid,
                              "train_test": ["train"] * 80 + ["prediction"] * 20,
                              "status_type_id": 0, "sensor_0_avg": values,
                              "wind_speed_3_avg": 6.0 + values / 10000,
                              "power_30_avg": 0.5 + values / 10000})
        columns = ["sensor_0_avg", "wind_speed_3_avg", "power_30_avg"]
        frame.loc[80, columns] = frame.loc[0, columns].to_numpy()
        frame.to_csv(datasets / f"comma_{eid}.csv", index=False)
        events.append({"event_id": eid, "event_label": "anomaly" if eid % 2 else "normal",
                       "event_start": times[85], "event_end": times[99]})
    pd.DataFrame(events).to_csv(farm_dir / "comma_event_info.csv", index=False)
    pd.DataFrame({"sensor_name": ["sensor_0", "wind_speed_3", "power_30"],
                  "description": ["temperature", "wind speed", "active power"]}).to_csv(
        farm_dir / "comma_feature_description.csv", index=False)
    split_cfg = replace(load_config(config_dir / "base.yaml", config_dir / "clients.yaml"),
                        val_block_hours=1, val_split_mode="blocked")
    client_root = work_root / "synthetic_clients"
    build_client_dataset(farm_dir.parent, client_root, split_cfg, expected_clients=2, expected_events=4,
                         source_identity=source_identity)
    fields = {"hidden_layers": (8, 4, 2, 4, 8), "batch_size": 16, "val_block_hours": 1,
              "val_split_mode": "blocked", "criticality_threshold": 2}
    extra = {"rounds": 2, "n_clients": 2, "expected_clients": 2, "expected_events": 4,
             "expected_dataset_manifest_sha256": file_hash(client_root / "dataset_manifest.json"),
             "dataset_override_reason": "Synthetic CPU smoke only; not a research comparison"}
    return client_root, fields, extra
