"""Execute the CPU export notebook end-to-end without CARE data or TensorFlow."""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_client_notebook_runs_all_cells_offline(tmp_path):
    notebook = REPO / "notebooks/20_federated_clients.ipynb"
    runner = """
import json, pathlib, sys
nb = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))
ns = {'__name__': '__main__'}
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), f'notebook cell {i}', 'exec'), ns)
assert 'tensorflow' not in sys.modules
assert len(ns['manifest']['clients']) == 2
assert ns['manifest']['n_events'] == 4
assert ns['summary']['train'].min() > 0
assert ns['summary']['val'].min() > 0
"""
    env = os.environ | {"WT_CLIENT_SMOKE": "1", "WT_REPO_ROOT": str(REPO),
                        "WT_WORK_ROOT": str(tmp_path),
                        "WT_CLIENT_OUTPUT_ROOT": str(tmp_path / "exports"), "PYTHONUTF8": "1"}
    result = subprocess.run([sys.executable, "-c", runner, str(notebook)], cwd=tmp_path,
                            check=False, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "exports/dataset_manifest.json").is_file()


def test_notebooks_have_clean_valid_code():
    for name in ("20_federated_clients.ipynb", "13_arm_local_turbine.ipynb"):
        notebook = json.loads((REPO / "notebooks" / name).read_text(encoding="utf-8"))
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert not cell["outputs"]
                compile("".join(cell["source"]), f"{name}:{i}", "exec")



def _run_bootstrap(tmp_path, extra_env=None, fake_kaggle=None):
    nb = json.loads((REPO / "notebooks/20_federated_clients.ipynb").read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    if fake_kaggle is not None:
        source = source.replace("/kaggle", fake_kaggle.as_posix())
    runner = tmp_path / "bootstrap.py"
    runner.write_text(source, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in {"WT_REPO_ROOT", "WT_REPO_REF", "WT_CLIENT_OUTPUT_ROOT", "WT_CLIENT_OUTPUT"}}
    env.update({"WT_WORK_ROOT": str(tmp_path / "working"), "PYTHONUTF8": "1"})
    env.update(extra_env or {})
    return subprocess.run([sys.executable, str(runner)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, encoding="utf-8", check=False, timeout=60)


def test_explicit_invalid_source_root_fails(tmp_path):
    result = _run_bootstrap(tmp_path, {"WT_REPO_ROOT": str(tmp_path / "missing")})
    assert result.returncode != 0
    assert "WT_REPO_ROOT does not contain" in result.stderr


def test_explicit_source_selectors_cannot_conflict(tmp_path):
    result = _run_bootstrap(tmp_path, {"WT_REPO_ROOT": str(REPO), "WT_REPO_REF": "v1"})
    assert result.returncode != 0
    assert "Choose WT_REPO_ROOT or WT_REPO_REF" in result.stderr


def test_zip_bootstrap_offline_and_corrupt_bundle(tmp_path):
    import runpy
    import zipfile

    kaggle = tmp_path / "kaggle"
    input_dir = kaggle / "input/source"
    input_dir.mkdir(parents=True)
    package = runpy.run_path(str(REPO / "scripts/package_kaggle_bridge.py"))["package"]
    archive = package(REPO, input_dir / "windturbine-client-bridge.zip")
    env = {"WT_WORK_ROOT": str(kaggle / "working")}
    result = _run_bootstrap(tmp_path, env, fake_kaggle=kaggle)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"bundle_sha256":' in result.stdout
    assert list((kaggle / "working").glob("windturbine-source-*/src/windturbine/data/clients.py"))
    with zipfile.ZipFile(archive) as original:
        entries = {name: original.read(name) for name in original.namelist()}
    entries["configs/clients.yaml"] += b"\n# invalid checksum\n"
    with zipfile.ZipFile(archive, "w") as corrupt:
        for name, data in entries.items():
            corrupt.writestr(name, data)
    result = _run_bootstrap(tmp_path, env, fake_kaggle=kaggle)
    assert result.returncode != 0
    assert "Source bundle hash mismatch" in result.stderr
