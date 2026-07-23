#!/usr/bin/env python3
"""Verify the exact terminal-training wheel in an isolated AutoDL runtime.

The verifier does not open scientific data or start an optimizer.  It proves
that the requested interpreter imports the installed wheel rather than the
repository ``src`` tree, runs the frozen focused and full test commands, and
atomically records the evidence consumed by the terminal release packet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

EXPECTED_DISTRIBUTION = "gwlens-mm"
EXPECTED_GPU_MODEL = "NVIDIA RTX 5000 Ada Generation"
MINIMUM_GPU_COUNT = 3
FOCUSED_TESTS = (
    "tests/test_phase4_training_stack.py",
    "tests/test_phase4_rung65.py",
    "tests/test_phase4_terminal_131k_probe.py",
    "tests/test_phase4_terminal_release.py",
    "tests/test_phase5_architecture_selection.py",
    "tests/test_phase5_terminal_architecture_authorization.py",
    "tests/test_phase5_terminal_downstream.py",
)

_RUNTIME_PROBE = r"""
import importlib.metadata
import json
from pathlib import Path

import gwlens_mm
import torch

distribution = importlib.metadata.distribution("gwlens-mm")
direct_url_text = distribution.read_text("direct_url.json")
direct_url = json.loads(direct_url_text) if direct_url_text is not None else None
print(json.dumps({
    "distribution_name": distribution.metadata["Name"],
    "distribution_version": distribution.version,
    "direct_url": direct_url,
    "module_path": str(Path(gwlens_mm.__file__).resolve()),
    "torch_cuda_available": bool(torch.cuda.is_available()),
}, sort_keys=True))
"""


class WheelVerificationError(RuntimeError):
    """Raised when the exact-wheel contract is not satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _gpu_names() -> tuple[str, ...]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())


def _archive_sha256(direct_url: Mapping[str, Any]) -> str | None:
    archive = direct_url.get("archive_info")
    if not isinstance(archive, Mapping):
        return None
    hashes = archive.get("hashes")
    if isinstance(hashes, Mapping) and isinstance(hashes.get("sha256"), str):
        return str(hashes["sha256"])
    value = archive.get("hash")
    if isinstance(value, str) and value.startswith("sha256="):
        return value.removeprefix("sha256=")
    return None


def validate_runtime_probe(
    root: Path,
    *,
    wheel_sha256: str,
    probe: Mapping[str, Any],
    gpu_names: Sequence[str],
) -> Mapping[str, Any]:
    """Validate an installed-wheel runtime probe without reading data."""

    module_path = Path(str(probe.get("module_path", ""))).resolve()
    source_root = (root / "src").resolve()
    try:
        module_path.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise WheelVerificationError("runtime imported the repository src tree")
    if not any(part in {"site-packages", "dist-packages"} for part in module_path.parts):
        raise WheelVerificationError("runtime module is not installed in site-packages")
    if str(probe.get("distribution_name", "")).lower() != EXPECTED_DISTRIBUTION:
        raise WheelVerificationError("installed distribution identity changed")

    direct_url = probe.get("direct_url")
    if not isinstance(direct_url, Mapping):
        raise WheelVerificationError("installed distribution lacks direct_url evidence")
    directory = direct_url.get("dir_info")
    if isinstance(directory, Mapping) and directory.get("editable") is True:
        raise WheelVerificationError("editable installation is forbidden")
    if _archive_sha256(direct_url) != wheel_sha256:
        raise WheelVerificationError("installed archive hash differs from exact wheel")
    if probe.get("torch_cuda_available") is not True:
        raise WheelVerificationError("CUDA is unavailable in the wheel runtime")

    names = tuple(str(name) for name in gpu_names)
    if len(names) < MINIMUM_GPU_COUNT or any(name != EXPECTED_GPU_MODEL for name in names):
        raise WheelVerificationError("GPU identity does not satisfy the frozen contract")
    return {
        "wheel_import_verified": True,
        "installed_distribution_name": EXPECTED_DISTRIBUTION,
        "installed_distribution_version": str(probe.get("distribution_version", "")),
        "installed_module_path": str(module_path),
        "installed_module_from_repository_source": False,
        "editable_install_used": False,
        "torch_cuda_available": True,
        "gpu_names": list(names),
    }


def _runtime_probe(runtime_python: Path) -> Mapping[str, Any]:
    completed = subprocess.run(
        [str(runtime_python), "-c", _RUNTIME_PROBE],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise WheelVerificationError("runtime probe did not emit one JSON object")
    value = json.loads(lines[0])
    if not isinstance(value, dict):
        raise WheelVerificationError("runtime probe output is not a mapping")
    return value


def pytest_command(
    runtime_python: Path,
    root: Path,
    test_paths: Sequence[str],
) -> list[str]:
    """Build a pytest command that cannot inherit the repository src path."""

    return [
        str(runtime_python),
        "-m",
        "pytest",
        "-q",
        "-c",
        "/dev/null",
        "--noconftest",
        "--rootdir",
        str(root),
        *(str(root / path) for path in test_paths),
    ]


def _run_tests(
    command: Sequence[str],
    *,
    root: Path,
    log_path: Path,
) -> int:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        list(command),
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    root = args.root.resolve()
    runtime_python = args.runtime_python.resolve()
    wheel = args.wheel.resolve()
    if not runtime_python.is_file():
        raise WheelVerificationError("runtime Python does not exist")
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise WheelVerificationError("exact wheel does not exist")
    wheel_sha256 = _sha256(wheel)
    runtime = validate_runtime_probe(
        root,
        wheel_sha256=wheel_sha256,
        probe=_runtime_probe(runtime_python),
        gpu_names=_gpu_names(),
    )

    focused_command = pytest_command(runtime_python, root, FOCUSED_TESTS)
    full_command = pytest_command(runtime_python, root, ("tests",))
    focused_log = args.output.with_name(args.output.stem + ".focused.log")
    full_log = args.output.with_name(args.output.stem + ".full.log")
    focused_exit = _run_tests(focused_command, root=root, log_path=focused_log)
    full_exit = _run_tests(full_command, root=root, log_path=full_log)
    passed = focused_exit == 0 and full_exit == 0
    result = {
        "status": "passed_exact_wheel_on_autodl" if passed else "exact_wheel_test_failed",
        "wheel_path": str(wheel),
        "wheel_filename": wheel.name,
        "wheel_sha256": wheel_sha256,
        "runtime_python": str(runtime_python),
        **runtime,
        "focused_test_exit_code": focused_exit,
        "full_test_exit_code": full_exit,
        "focused_test_command": focused_command,
        "full_test_command": full_command,
        "focused_test_log": str(focused_log),
        "focused_test_log_sha256": _sha256(focused_log),
        "full_test_log": str(full_log),
        "full_test_log_sha256": _sha256(full_log),
        "repository_root_pythonpath_used": True,
        "repository_src_pythonpath_used": False,
        "repository_conftest_used": False,
        "pytest_noconftest_used": True,
        "scientific_data_opened": False,
        "optimizer_started": False,
    }
    _atomic_json(args.output, result)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
