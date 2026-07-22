from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.phase4.verify_terminal_training_wheel import (
    EXPECTED_GPU_MODEL,
    WheelVerificationError,
    pytest_command,
    validate_runtime_probe,
)


def _probe(wheel_hash: str) -> dict[str, object]:
    return {
        "distribution_name": "gwlens-mm",
        "distribution_version": "0.1.0",
        "module_path": "/runtime/lib/python3.11/site-packages/gwlens_mm/__init__.py",
        "torch_cuda_available": True,
        "direct_url": {
            "url": "file:///runtime/gwlens_mm.whl",
            "archive_info": {"hashes": {"sha256": wheel_hash}},
        },
    }


def test_runtime_probe_requires_exact_noneditable_wheel(tmp_path: Path) -> None:
    wheel_hash = "a" * 64
    result = validate_runtime_probe(
        tmp_path,
        wheel_sha256=wheel_hash,
        probe=_probe(wheel_hash),
        gpu_names=[EXPECTED_GPU_MODEL] * 4,
    )
    assert result["wheel_import_verified"] is True
    assert result["installed_module_from_repository_source"] is False
    assert result["editable_install_used"] is False


@pytest.mark.parametrize("failure", ["source", "editable", "hash", "cuda", "gpu"])
def test_runtime_probe_fails_closed(tmp_path: Path, failure: str) -> None:
    wheel_hash = "a" * 64
    probe = _probe(wheel_hash)
    gpu_names = [EXPECTED_GPU_MODEL] * 3
    if failure == "source":
        probe["module_path"] = str(tmp_path / "src/gwlens_mm/__init__.py")
    elif failure == "editable":
        probe["direct_url"] = {
            "url": "file:///checkout",
            "dir_info": {"editable": True},
        }
    elif failure == "hash":
        probe["direct_url"] = {
            "url": "file:///runtime/other.whl",
            "archive_info": {"hashes": {"sha256": "b" * 64}},
        }
    elif failure == "cuda":
        probe["torch_cuda_available"] = False
    else:
        gpu_names = [EXPECTED_GPU_MODEL] * 2
    with pytest.raises(WheelVerificationError):
        validate_runtime_probe(
            tmp_path,
            wheel_sha256=wheel_hash,
            probe=probe,
            gpu_names=gpu_names,
        )


def test_pytest_command_disables_repository_configuration(tmp_path: Path) -> None:
    command = pytest_command(Path("/runtime/bin/python"), tmp_path, ("tests",))
    assert command[:4] == ["/runtime/bin/python", "-m", "pytest", "-q"]
    assert command[4:6] == ["-c", "/dev/null"]
    assert "--noconftest" in command
    assert str(tmp_path / "src") not in command


def test_pytest_command_does_not_load_repository_conftest(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "conftest.py").write_text(
        'raise RuntimeError("repository conftest must remain isolated")\n',
        encoding="utf-8",
    )
    (tests / "test_smoke.py").write_text(
        "def test_smoke():\n    assert True\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        pytest_command(Path(sys.executable), tmp_path, ("tests/test_smoke.py",)),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
