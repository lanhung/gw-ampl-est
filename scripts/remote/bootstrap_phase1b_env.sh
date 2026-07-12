#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/root/autodl-tmp/lensing-4/repo
ENV_ROOT=/root/autodl-tmp/lensing-4/envs/phase1b-smoke
MANIFEST_ROOT=/root/autodl-tmp/lensing-4/manifests/phase1b

mkdir -p "$ENV_ROOT" "$MANIFEST_ROOT"

if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  python3 -m venv "$ENV_ROOT"
fi

"$ENV_ROOT/bin/python" -m pip install \
  pip==24.3.1 \
  setuptools==75.3.0 \
  wheel==0.44.0

"$ENV_ROOT/bin/python" -m pip install \
  --requirement "$REPO_ROOT/configs/environment/phase1b-autodl-requirements.txt"

"$ENV_ROOT/bin/python" -m pip install --no-deps --editable "$REPO_ROOT"

"$ENV_ROOT/bin/python" -m pip freeze \
  > "$MANIFEST_ROOT/environment.freeze.txt"

"$ENV_ROOT/bin/python" - <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path

packages = [
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "zarr",
    "numcodecs",
    "lenstronomy",
    "bilby",
    "lalsuite",
    "astropy",
    "h5py",
]
versions = {name: metadata.version(name) for name in packages}
output = Path("/root/autodl-tmp/lensing-4/manifests/phase1b/environment.json")
output.write_text(json.dumps(versions, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(versions, sort_keys=True))
PY
