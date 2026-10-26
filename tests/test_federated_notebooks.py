"""Run the F2-F4 notebooks' real Flower/TensorFlow synthetic pipelines."""
import importlib.util
import json
import os
import runpy
import subprocess
import sys

import pytest

from test_fedavg_notebook import REPO

NOTEBOOKS = {"F2": ("31_fed_personalised.ipynb", "fedper", ()),
             "F3": ("32_fedprox.ipynb", "fedprox", ()),
             "F4": ("33_fed_private.ipynb", "fedper_private", ("dp_accounting",))}


@pytest.mark.parametrize("tag", NOTEBOOKS)
def test_notebook_clean_and_compiles(tag):
    notebook = json.loads((REPO / "notebooks" / NOTEBOOKS[tag][0]).read_text(encoding="utf-8"))
    assert f"Experiment {tag}" in "".join(notebook["cells"][0]["source"])
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert not cell["outputs"]
            compile("".join(cell["source"]), f"{tag} notebook cell {index}", "exec")


@pytest.mark.parametrize("tag", NOTEBOOKS)
def test_notebook_real_cpu_smoke(tmp_path, tag):
    name, branch, extra = NOTEBOOKS[tag]
    if any(importlib.util.find_spec(m) is None for m in ("flwr", "tensorflow", "pyarrow", *extra)):
        pytest.skip("Install the federated training extras to run the real notebook smoke")
    runner = f"""
import json, pathlib, sys
notebook = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))
ns = {{'__name__': '__main__'}}
for index, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), f'{tag} notebook cell {{index}}', 'exec'), ns)
assert ns['preflight']['fedavg']['passed'] and ns['preflight']['strategy']['passed']
assert ns['BRANCH'] == {branch!r} and ns['run_manifest']['experiment'] == {tag!r}
assert ns['CFG'].extra['rounds'] == 2 and ns['CFG'].extra['dataset_override_reason']
assert (ns['OUTPUT_ROOT'] / 'COMPLETE').is_file() and len(ns['selected']) == 2
assert 'ray' not in sys.modules
if {tag!r} == 'F2':
    assert ns['run_manifest']['shared_parameter_indices'] == [0, 1, 2, 3, 4, 5]
if {tag!r} == 'F3':
    assert ns['MU'] == 0.1 and len(ns['sweep']) == 1 and ns['sweep_path'].is_file()
if {tag!r} == 'F4':
    assert ns['run_manifest']['privacy']['epsilon_max'] > 0
    assert ns['run_manifest']['strategy']['secure_aggregation'] and ns['sweep_path'].is_file()
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("WT_")}
    env.update({f"WT_{tag}_SMOKE": "1", "WT_REPO_ROOT": str(REPO), "WT_WORK_ROOT": str(tmp_path),
                f"WT_{tag}_OUTPUT_ROOT": str(tmp_path / "runs" / "run"), "PYTHONUTF8": "1",
                "CUDA_VISIBLE_DEVICES": "-1", "FLWR_TELEMETRY_ENABLED": "0", "TF_CPP_MIN_LOG_LEVEL": "2",
                "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1", "MPLBACKEND": "Agg"})
    result = subprocess.run([sys.executable, "-c", runner, str(REPO / "notebooks" / name)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, encoding="utf-8", timeout=420, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "D reference is not comparable" in result.stdout


def test_bundle_carries_f2_f4_source_and_notebooks(tmp_path):
    package = runpy.run_path(str(REPO / "scripts/package_kaggle_bridge.py"))["package"]
    import zipfile

    with zipfile.ZipFile(package(REPO, tmp_path / "bundle.zip")) as archive:
        names = set(archive.namelist())
    assert {"src/windturbine/federated/strategies.py", "src/windturbine/federated/smoke.py",
            "configs/fed_personalised.yaml", "configs/fed_fedprox.yaml", "configs/fed_private.yaml",
            *(f"notebooks/{name}" for name, _, _ in NOTEBOOKS.values())} <= names
