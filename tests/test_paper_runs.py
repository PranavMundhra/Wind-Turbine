"""The paper's experiment configs, the run CLI and Table I on a synthetic export."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from windturbine.experiments import experiment_names, load_experiment, run_experiment
from windturbine.runs import config_dict

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
TABLE_I = ["D", "F1", "F2", "F3_mu0", "F3_mu0.001", "F3_mu0.01", "F3_mu0.1", "F3_mu1", "F3_mu10",
           "F4_sigma0", "F4_sigma0.5", "F4_sigma1", "F4_sigma2", "F4_sigma4", "central", "central_sequential"]
BOOKKEEPING = {"runner", "client_unit", "remove_rows_seen_in_prediction", "fingerprint_decimals", "mu_grid",
               "base_config", "dataset_override_reason", "central_order"}


def test_every_table_row_has_a_config():
    assert sorted(experiment_names(CONFIGS)) == sorted(TABLE_I)


@pytest.mark.parametrize("name", [n for n in TABLE_I if n not in ("D", "central")])
def test_configs_reproduce_the_seed42_runs(name):
    """Every training setting equals the config recorded in the committed seed-42 manifest."""
    manifest = json.loads((REPO / "results/paper" / name / "s42/manifest.json").read_text(encoding="utf-8"))
    recorded, current = manifest["config"], config_dict(load_experiment(name, config_dir=CONFIGS))
    labels = {"extra", "arm"}  # arm is a label; the comparator ran inside the F1 run
    assert ({k: v for k, v in recorded.items() if k not in labels}
            == {k: v for k, v in current.items() if k not in labels})
    keys = (set(recorded["extra"]) | set(current["extra"])) - BOOKKEEPING
    assert {k: recorded["extra"].get(k) for k in keys} == {k: current["extra"].get(k) for k in keys}


def test_seed_and_overrides_win_last():
    cfg = load_experiment("F3_mu0.1", seed=7, config_dir=CONFIGS, overrides={"extra": {"rounds": 3}})
    assert (cfg.seed, cfg.extra["rounds"], cfg.extra["mu"], cfg.extra["local_epochs"]) == (7, 3, 0.1, 2)
    with pytest.raises(KeyError):
        load_experiment("F9", config_dir=CONFIGS)


def test_committed_runs_rebuild_table_i():
    from windturbine.summary import find_runs, seed_table

    table = seed_table(find_runs(REPO / "results/paper"))
    paper = {"D": 0.640, "central_sequential": 0.640, "F1": 0.636, "F2": 0.625, "F3_mu0": 0.636,
             "F3_mu0.001": 0.632, "F3_mu0.01": 0.636, "F3_mu0.1": 0.639, "F3_mu1": 0.616, "F3_mu10": 0.590}
    for name, care in paper.items():
        assert round(table.loc[name, "care_mean"], 3) == care, name


def test_fast_bootstrap_matches_reference_care():
    from windturbine.scoring import aggregate_event_metrics
    from windturbine.summary import _care_from_counts, _turbine_sums

    frame = pd.read_csv(REPO / "results/paper/F3_mu10/s42/per_event.csv")
    turbines = np.array(sorted(frame.asset_id.unique()))
    sums = _turbine_sums(frame, turbines)
    rng = np.random.default_rng(3)
    for _ in range(10):
        counts = rng.multinomial(len(turbines), np.full(len(turbines), 1 / len(turbines)))
        parts = [frame[frame.asset_id == aid].assign(event_id=lambda f, a=aid, j=j: f.event_id.astype(str) + f"{a}_{j}")
                 for aid, n in zip(turbines, counts) for j in range(n)]
        reference = aggregate_event_metrics(pd.concat(parts), beta=0.5, care_weights=(1, 1, 1, 2))["CARE"]
        assert _care_from_counts(counts[None].astype(float), sums, 0.5, (1, 1, 1, 2))[0] == pytest.approx(reference)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("flwr")
    from windturbine.federated import build_synthetic_clients

    root = tmp_path_factory.mktemp("paper")
    clients, fields, extra = build_synthetic_clients(root, CONFIGS, farm="Wind Farm C")
    overrides = {**fields, "max_epochs": 3, "early_stop_patience": 2, "lr_plateau_patience": 1,
                 "extra": {**extra, "max_hours": None}}
    return root, clients, overrides


def _run(synthetic, name, out, seed=42):
    root, clients, overrides = synthetic
    return run_experiment(name, seed=seed, clients=clients, output_root=root / out / name / f"s{seed}",
                          config_dir=CONFIGS, overrides=overrides)


@pytest.mark.parametrize("name", ["D", "F1", "central", "central_sequential", "F2", "F3_mu0.1", "F4_sigma1"])
def test_each_runner_publishes_a_complete_scored_run(synthetic, name):
    manifest = _run(synthetic, name, "runs")
    run = synthetic[0] / "runs" / name / "s42"
    assert (run / "COMPLETE").is_file() and manifest["experiment"] == name
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert 0 <= results["care"] <= 1 and len(pd.read_csv(run / "per_event.csv")) == 4
    if name.startswith("central"):
        assert manifest["central_order"] == ("client_sequential" if name.endswith("sequential") else "pooled_shuffle")
        assert list(manifest["branches"]) == ["centralized"]
    if name == "F1":
        assert list(manifest["branches"]) == ["fedavg"]


def test_split_branches_equal_a_joint_run(synthetic):
    """F1 and the comparator in separate runs give exactly the joint run's results."""
    from windturbine.federated import run_fedavg

    root, clients, _ = synthetic
    _run(synthetic, "F1", "split")
    _run(synthetic, "central", "split")
    cfg = load_experiment("central", config_dir=CONFIGS, overrides=synthetic[2])
    joint = run_fedavg(clients, root / "joint", cfg, run_centralized=True, run_federated_branch=True)
    for branch, name in (("fedavg", "F1"), ("centralized", "central")):
        split = json.loads((root / "split" / name / "s42" / branch / "results.json").read_text(encoding="utf-8"))
        together = json.loads((root / "joint" / branch / "results.json").read_text(encoding="utf-8"))
        assert split == together
        assert joint["branches"][branch]["selected_validation_mse"] == pytest.approx(
            json.loads((root / "split" / name / "s42" / "manifest.json").read_text())["branches"][branch]
            ["selected_validation_mse"], abs=0)


def test_cli_runs_seeds_resumes_and_summarises(synthetic, tmp_path, capsys):
    import yaml

    from windturbine.__main__ import main

    _, clients, overrides = synthetic
    out = tmp_path / "runs"
    sets = []
    for key, value in overrides.items():
        for k, v in (value.items() if key == "extra" else [(key, value)]):
            name = f"extra.{k}" if key == "extra" else k
            sets += ["--set", f"{name}={yaml.safe_dump(list(v) if isinstance(v, tuple) else v, default_flow_style=True).strip().removesuffix('...').strip()}"]
    argv = ["run", "D", "F1", "--seeds", "1", "2", "--clients", str(clients), "--out", str(out), *sets]
    main(argv)
    assert sorted(p.parent.relative_to(out).as_posix() for p in out.glob("*/s*/COMPLETE")) == [
        "D/s1", "D/s2", "F1/s1", "F1/s2"]
    capsys.readouterr()
    main(argv)  # everything is complete: nothing reruns
    assert "4 already complete, 0 to run" in capsys.readouterr().out
    main(["summarise", "--runs", str(out), "--out", str(tmp_path / "paper"), "--boot", "50"])
    table = pd.read_csv(tmp_path / "paper" / "table_I.csv", index_col=0)
    assert table.loc["F1", "n_seeds"] == 2 and table.loc["D", "n_seeds"] == 2
    assert not list((tmp_path / "paper").rglob("*.npz"))  # weights stay out of the committed folder
    assert (tmp_path / "paper" / "bootstrap_vs_D.csv").is_file()
