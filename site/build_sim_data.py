"""Build site/sim_data.js: month-long snapshots of real Wind Farm C readings for the Test a turbine replay.

The raw CARE to Compare files are not in the repository (CC BY-SA 4.0, see README), so this script runs on a machine
that has them and its output is committed. It needs two files from turbine 35: event 67 (a fault event) and event 48
(a normal event).

    python site/build_sim_data.py --fault "Wind Farm C/datasets/67.csv" --normal "Wind Farm C/datasets/48.csv"

No trained model runs here. Each reading gets a transparent stand-in score: how far one sensor sits from its usual
value at the same wind speed and time of year, in units of its usual spread, measured on the file's own normal-status
training year. A reading is flagged when the score passes that file's 99th percentile on healthy training readings, the
same q = 0.99 rule as the project's per-turbine thresholds. The page then applies the CARE criticality counter.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "sim_data.js"
W, P, S = "wind_speed_236_avg", "power_2_avg", "sensor_210_avg"
NORMAL, Q, THETA = [0, 2], 0.99, 72
BINS = np.arange(0, 42, 2)
# (key, source, start, label, kind, note). Windows were chosen after looking at the counter month by month.
SNAPSHOTS = [
    ("fault", "fault", "2027-02-01", "Fault event, February", "fault",
     "The prediction window of a labelled fault event. sensor_210 runs far below its usual level for the wind."),
    ("before", "fault", "2026-02-01", "Same turbine, a year earlier", "normal",
     "The same calendar month in the turbine's normal training year."),
    ("close", "fault", "2026-12-01", "A close call", "normal",
     "A normal month where flags bunch up and the counter climbs, but never reaches the threshold."),
    ("healthy", "normal", "2020-04-12", "A labelled normal event", "normal",
     "Ends with the prediction window of an event the benchmark labels as normal."),
]


def load(path):
    head = pd.read_csv(path, nrows=0).columns
    for c in (W, P, S):
        assert c in head, f"{c} missing from {path}"
    d = pd.read_csv(path, usecols=["time_stamp", "train_test", "status_type_id", W, P, S], parse_dates=["time_stamp"])
    return d.sort_values("time_stamp").reset_index(drop=True)


def score(d):
    """Median and spread of S on normal training rows at the same wind-speed bin within +-30 days of the year."""
    ok = d.status_type_id.isin(NORMAL)
    tr = d[(d.train_test == "train") & ok]
    doy, tdoy = d.time_stamp.dt.dayofyear.to_numpy(), tr.time_stamp.dt.dayofyear.to_numpy()
    kb, tkb, tv = np.digitize(d[W].fillna(0), BINS), np.digitize(tr[W].fillna(0), BINS), tr[S].to_numpy()
    med, iqr, cache = np.full(len(d), np.nan), np.full(len(d), np.nan), {}
    for i in range(len(d)):
        key = (doy[i] // 7, kb[i])
        if key not in cache:
            near = np.abs(((tdoy - doy[i] + 182) % 365) - 182) <= 30
            v = tv[near & (tkb == kb[i])]
            cache[key] = (np.median(v), np.subtract(*np.percentile(v, [75, 25]))) if len(v) > 50 else (np.nan, np.nan)
        med[i], iqr[i] = cache[key]
    iqr[iqr <= 1e-9] = np.nan
    d["med"], d["iqr"], d["z"] = med, iqr, (d[S] - med) / iqr
    tau = float(np.nanquantile(np.abs(d.loc[ok & (d.train_test == "train"), "z"]), Q))
    d["flag"] = (np.abs(d.z) > tau).astype(int)
    return d, tau


def counter(status, flag):
    c, out = 0, []
    for s, f in zip(status, flag):
        if s in NORMAL:
            c = c + 1 if f else max(c - 1, 0)
        out.append(c)
    return out


def r(v, n):
    return [None if pd.isna(x) else round(float(x), n) for x in v]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fault", required=True, help="event 67 csv (turbine 35, fault)")
    ap.add_argument("--normal", required=True, help="event 48 csv (turbine 35, normal)")
    a = ap.parse_args()
    files = {"fault": score(load(a.fault)), "normal": score(load(a.normal))}
    snaps = []
    for key, src, start, label, kind, note in SNAPSHOTS:
        d, tau = files[src]
        t0 = pd.Timestamp(start)
        m = d[(d.time_stamp >= t0) & (d.time_stamp < t0 + pd.Timedelta("30D"))].reset_index(drop=True)
        assert len(m) > 2000, f"{key}: only {len(m)} rows"
        peak = max(counter(m.status_type_id, m.flag))
        assert (peak >= THETA) == (kind == "fault"), f"{key}: counter peak {peak} does not match its label {kind}"
        band = tau * m.iqr
        snaps.append({
            "key": key, "label": label, "kind": kind, "note": note, "tau": round(tau, 3),
            "event": 67 if src == "fault" else 48, "start": str(m.time_stamp.iloc[0]), "step_min": 10,
            "t": [int((t - m.time_stamp.iloc[0]).total_seconds() // 600) for t in m.time_stamp],
            "status": m.status_type_id.astype(int).tolist(),
            "split": ["p" if x == "prediction" else "t" for x in m.train_test],
            "wind": r(m[W], 1), "power": r(m[P], 3), "sensor": r(m[S], 2),
            "lo": r(m.med - band, 2), "hi": r(m.med + band, 2), "flag": m.flag.tolist(),
        })
    data = {"turbine": 35, "channels": {"wind": W, "power": P, "sensor": S}, "normal_status": NORMAL, "q": Q,
            "theta_default": THETA, "snapshots": snaps}
    OUT.write_text("window.WT_SIM = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.name}: {sum(len(s['t']) for s in snaps)} readings in {len(snaps)} snapshots")


if __name__ == "__main__":
    main()
