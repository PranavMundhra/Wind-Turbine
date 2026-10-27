"""Run F1-F4 notebooks on the local GPU (Keras on PyTorch), 3 lanes in parallel.

Executed notebooks -> notebooks/executed/, run folders -> results/runs/F1..F4,
live cell output -> results/runs/logs/<name>.log. Run from VS Code (Run > Run
Without Debugging); keep the window open until it prints ALL DONE.
"""
import os
import subprocess
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(r"D:\wt_work\windturbine_clients")
OUT_NB = REPO / "notebooks" / "executed"
LOGS = REPO / "results" / "runs" / "logs"

# (name, notebook, extra env). F1 first in lane A so later notebooks can compare against it.
LANES = [
    [("F1_fedavg", "30_fedavg.ipynb", {}),
     ("F2_fedper", "31_fed_personalised.ipynb", {}),
     *[(f"F4_noise_{n:g}", "33_fed_private.ipynb", {"WT_F4_NOISE": str(n)}) for n in (0.0, 0.5, 1.0, 2.0, 4.0)]],
    [(f"F3_mu_{m:g}", "32_fedprox.ipynb", {"WT_F3_MU": str(m)}) for m in (0.0, 0.001, 0.01)],
    [(f"F3_mu_{m:g}", "32_fedprox.ipynb", {"WT_F3_MU": str(m)}) for m in (0.1, 1.0, 10.0)],
]


def execute(name, notebook, extra):
    """Child process: run one notebook with nbclient, streaming cell output to a log."""
    import nbformat
    from nbclient import NotebookClient

    log = open(LOGS / f"{name}.log", "a", encoding="utf-8", buffering=1)

    class Client(NotebookClient):
        def output(self, outs, msg, display_id, cell_index):
            content = msg.get("content", {})
            text = content.get("text") or "".join(content.get("traceback", [])) or content.get("data", {}).get("text/plain", "")
            if text:
                log.write(text if text.endswith("\n") else text + "\n")
            return super().output(outs, msg, display_id, cell_index)

    nb = nbformat.read(REPO / "notebooks" / notebook, as_version=4)
    client = Client(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(REPO / "notebooks")}})
    try:
        client.execute()
        log.write("NOTEBOOK OK\n")
    finally:
        nbformat.write(nb, OUT_NB / f"{name}.ipynb")


def lane(jobs):
    for name, notebook, extra in jobs:
        env = {**os.environ, "KERAS_BACKEND": "torch", "WT_REPO_ROOT": str(REPO), "WT_WORK_ROOT": str(REPO),
               "WT_CLIENT_DATASET": str(DATA), "PYTHONUTF8": "1", "FLWR_TELEMETRY_ENABLED": "0", **extra}
        f1 = sorted((REPO / "results" / "runs" / "F1").glob("*/COMPLETE"))
        if f1 and not name.startswith("F1"):
            env["WT_F1_RUN"] = str(f1[-1].parent)
        print(f"[start] {name}", flush=True)
        code = subprocess.call([sys.executable, __file__, "--one", name, notebook], env=env)
        print(f"[{'done' if code == 0 else f'FAILED ({code})'}] {name}  log: {LOGS / (name + '.log')}", flush=True)


if __name__ == "__main__":
    OUT_NB.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:2] == ["--one"]:
        execute(sys.argv[2], sys.argv[3], {})
        sys.exit(0)
    threads = [threading.Thread(target=lane, args=(jobs,)) for jobs in LANES]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("ALL DONE", flush=True)
