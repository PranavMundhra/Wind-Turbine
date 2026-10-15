"""Bundle reviewed source/configs for attaching to a Kaggle notebook as a dataset.

No dependency installation, credentials, raw data, virtual environments or run
outputs are included. The notebook imports this exact source before it is pushed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def package(root: Path, output: Path) -> Path:
    paths = [*root.glob("src/windturbine/**/*.py"), *root.glob("configs/*.yaml"),
             root / "pyproject.toml", root / "README.md",
             root / "notebooks/20_federated_clients.ipynb",
             root / "notebooks/13_arm_local_turbine.ipynb", root / "docs/CLIENT_BRIDGE.md"]
    required = root / "src/windturbine/data/clients.py"
    if required not in paths:
        raise FileNotFoundError("Client bridge source has not been created yet")
    content = {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(paths)}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in content.items()}
    content["bridge_bundle_manifest.json"] = json.dumps(
        {"format_version": 1, "files": hashes}, indent=2, sort_keys=True,
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in sorted(content.items()):
            info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, data)
    return output


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "dist/windturbine-client-bridge.zip")
    args = parser.parse_args()
    print(package(root, args.output))


if __name__ == "__main__":
    main()
