"""Run the F1 notebook's real Flower/TensorFlow synthetic pipeline."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks/30_fedavg.ipynb"


def test_fedavg_notebook_clean_and_compiles():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert not cell["outputs"]
            compile("".join(cell["source"]), f"F1 notebook cell {index}", "exec")


def test_fedavg_notebook_real_cpu_smoke(tmp_path):
    if any(importlib.util.find_spec(name) is None for name in ("flwr", "tensorflow", "pyarrow")):
        pytest.skip("Install the federated training extras to run real Flower/TensorFlow notebook smoke")
    runner = """
import json, pathlib, sys
notebook = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))
ns = {'__name__': '__main__'}
for index, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), f'F1 notebook cell {index}', 'exec'), ns)
assert ns['preflight']['passed']
assert ns['run_manifest']['preflight']['passed']
assert set(ns['run_manifest']['branches']) == {'fedavg', 'centralized'}
assert set(ns['checkpoint_counts']) == {'fedavg', 'centralized'}
assert all(count > 0 for count in ns['checkpoint_counts'].values())
assert ns['CFG'].extra['rounds'] == 2
assert ns['CFG'].extra['local_epochs'] == 1
assert ns['CFG'].extra['dataset_override_reason']
assert 'ray' not in sys.modules
assert (ns['OUTPUT_ROOT'] / 'COMPLETE').is_file()
for branch, data in ns['run_manifest']['branches'].items():
    assert 0 <= data['selected_epoch'] <= 2
    assert (ns['OUTPUT_ROOT'] / data['history']).is_file()
assert len(ns['comparison']) == 2
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("WT_")}
    env.update({"WT_F1_SMOKE": "1", "WT_REPO_ROOT": str(REPO), "WT_WORK_ROOT": str(tmp_path),
                "WT_F1_OUTPUT_ROOT": str(tmp_path / "run"), "PYTHONUTF8": "1",
                "CUDA_VISIBLE_DEVICES": "-1", "FLWR_TELEMETRY_ENABLED": "0",
                "TF_CPP_MIN_LOG_LEVEL": "2", "TF_NUM_INTRAOP_THREADS": "1",
                "TF_NUM_INTEROP_THREADS": "1"})
    result = subprocess.run([sys.executable, "-c", runner, str(NOTEBOOK)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, encoding="utf-8",
                            timeout=240, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "D reference is not comparable" in result.stdout
    assert (tmp_path / "run/COMPLETE").is_file()


def test_fedavg_notebook_bootstrap_from_attached_zip(tmp_path):
    import runpy
    import zipfile

    kaggle = tmp_path / "kaggle"
    attachment = kaggle / "input/source/windturbine-client-bridge.zip"
    package = runpy.run_path(str(REPO / "scripts/package_kaggle_bridge.py"))["package"]
    package(REPO, attachment)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    bootstrap = next("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    # Substitute the mount prefix only; source selection and ZIP verification run unchanged.
    bootstrap = bootstrap.replace("/kaggle", kaggle.as_posix())
    bootstrap += "\nassert (REPO / 'src/windturbine/federated/fedavg.py').is_file()\n"
    bootstrap += "assert 'notebooks/30_fedavg.ipynb' in SOURCE_IDENTITY['source_files']\n"
    bootstrap += "assert SOURCE_IDENTITY['bundle_sha256']\n"
    env = {k: v for k, v in os.environ.items() if not k.startswith("WT_")}
    env.update({"WT_WORK_ROOT": str(kaggle / "working"), "PYTHONUTF8": "1"})
    result = subprocess.run([sys.executable, "-c", bootstrap], cwd=tmp_path, env=env,
                            capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr

    # An internally valid E03-only bundle must not be accepted as F1 source.
    with zipfile.ZipFile(attachment) as original:
        entries = {name: original.read(name) for name in original.namelist()
                   if not name.startswith("src/windturbine/federated/")
                   and name not in {"notebooks/30_fedavg.ipynb", "docs/FEDAVG.md"}}
    package_manifest = json.loads(entries["bridge_bundle_manifest.json"])
    package_manifest["files"] = {name: digest for name, digest in package_manifest["files"].items()
                                 if name in entries}
    entries["bridge_bundle_manifest.json"] = json.dumps(package_manifest).encode("utf-8")
    with zipfile.ZipFile(attachment, "w") as old_bundle:
        for name, content in entries.items():
            old_bundle.writestr(name, content)
    result = subprocess.run([sys.executable, "-c", bootstrap], cwd=tmp_path, env=env,
                            capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
    assert result.returncode != 0
    assert "containing src/windturbine/federated/fedavg.py" in result.stderr
