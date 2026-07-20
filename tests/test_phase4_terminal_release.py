from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from gwlens_mm.training.contracts import TrainingGateError
from gwlens_mm.training.terminal_release import (
    ENVIRONMENT_LOCK_HASH,
    EXPECTED_GPU_MODEL,
    prepare_terminal_probe_review_packet,
    validate_terminal_release_checkout_paths,
)
from scripts.phase4.prepare_terminal_probe_release import _verify_release_checkout

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _closeout() -> dict[str, object]:
    return {
        "status": "terminal_131k_independent_closeout_passed",
        "new_train_accepted_count": 65536,
        "new_train_shard_count": 512,
        "development_tail_accepted_count": 512,
        "development_tail_namespace_count": 4,
        "logical_train_accepted_count": 131072,
        "proposal_equals_evaluation": True,
        "all_importance_weights_one": True,
        "combined_manifest_sha256": "a" * 64,
        "train_parent_manifest_sha256": "b" * 64,
        "development_tail_manifest_sha256": "c" * 64,
        "tree_evidence": {"recomputed": True},
        "training_authorized": False,
        "architecture_selection_authorized": False,
        "calibration_authorized": False,
        "sbc_authorized": False,
        "final_evaluation_authorized": False,
        "extension_above_131072_authorized": False,
        "gwosc_gwtc_accessed": False,
    }


def _fixture_root(tmp_path: Path) -> Path:
    for relative in (
        "configs/statistics/terminal_131k_preregistration.yaml",
        "configs/models/phase4_probe_nsf.yaml",
        "results/phase4/final_evaluation_commitment.json",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return tmp_path


def _packet(tmp_path: Path) -> dict[str, object]:
    root = _fixture_root(tmp_path)
    closeout_path = root / "results/phase4/terminal_probe_closeout.json"
    wheel_path = tmp_path / "gwlens_mm.whl"
    wheel_test_path = tmp_path / "wheel-test.json"
    _write_json(closeout_path, _closeout())
    wheel_path.write_bytes(b"exact wheel fixture")
    wheel_hash = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    gpu_names = [EXPECTED_GPU_MODEL] * 3
    _write_json(
        wheel_test_path,
        {
            "status": "passed_exact_wheel_on_autodl",
            "wheel_sha256": wheel_hash,
            "focused_test_exit_code": 0,
            "full_test_exit_code": 0,
            "torch_cuda_available": True,
            "editable_install_used": False,
            "wheel_import_verified": True,
            "installed_distribution_name": "gwlens-mm",
            "installed_module_from_repository_source": False,
            "repository_root_pythonpath_used": True,
            "repository_src_pythonpath_used": False,
            "gpu_names": gpu_names,
        },
    )
    return dict(
        prepare_terminal_probe_review_packet(
            root,
            closeout_result_path=closeout_path,
            training_commit="1" * 40,
            review_checkout_commit="2" * 40,
            wheel_path=wheel_path,
            environment_lock_path=ROOT
            / "configs/environment/phase4-training-freeze.txt",
            wheel_test_result_path=wheel_test_path,
            gpu_names=gpu_names,
        )
    )


def test_terminal_probe_release_packet_is_exact_but_non_authorizing(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path)
    assert packet["status"] == "ready_for_delegated_terminal_probe_authorization_review"
    assert packet["authorization_created"] is False
    assert packet["optimizer_execution_authorized"] is False
    immutable = packet["immutable_training"]
    assert isinstance(immutable, dict)
    assert immutable["environment_lock_sha256"] == ENVIRONMENT_LOCK_HASH
    assert packet["authorized_training_rungs_preview"] == [131072]
    assert packet["authorized_training_seeds_preview"] == [0, 1, 2]
    assert packet["release_review_checkout_commit"] == "2" * 40
    assert packet["closeout_result_path"] == (
        "results/phase4/terminal_probe_closeout.json"
    )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("tree_evidence", {"recomputed": False}),
        ("logical_train_accepted_count", 131071),
        ("all_importance_weights_one", False),
        ("training_authorized", True),
    ],
)
def test_terminal_probe_release_packet_rejects_closeout_drift(
    tmp_path: Path, field: str, invalid: object
) -> None:
    closeout = _closeout()
    closeout[field] = invalid
    root = _fixture_root(tmp_path)
    closeout_path = root / "results/phase4/terminal_probe_closeout.json"
    wheel_path = tmp_path / "fixture.whl"
    wheel_test_path = tmp_path / "wheel-test.json"
    _write_json(closeout_path, closeout)
    wheel_path.write_bytes(b"wheel")
    _write_json(wheel_test_path, {})
    with pytest.raises(TrainingGateError, match="tree replay|closeout contract"):
        prepare_terminal_probe_review_packet(
            root,
            closeout_result_path=closeout_path,
            training_commit="1" * 40,
            review_checkout_commit="2" * 40,
            wheel_path=wheel_path,
            environment_lock_path=ROOT
            / "configs/environment/phase4-training-freeze.txt",
            wheel_test_result_path=wheel_test_path,
            gpu_names=[EXPECTED_GPU_MODEL] * 3,
        )


def test_terminal_probe_release_packet_rejects_gpu_or_wheel_test_drift(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    closeout_path = root / "results/phase4/terminal_probe_closeout.json"
    wheel_path = tmp_path / "fixture.whl"
    wheel_test_path = tmp_path / "wheel-test.json"
    _write_json(closeout_path, _closeout())
    wheel_path.write_bytes(b"wheel")
    _write_json(
        wheel_test_path,
        {
            "status": "passed_exact_wheel_on_autodl",
            "wheel_sha256": hashlib.sha256(b"wheel").hexdigest(),
            "focused_test_exit_code": 0,
            "full_test_exit_code": 0,
            "torch_cuda_available": True,
            "editable_install_used": False,
            "wheel_import_verified": True,
            "installed_distribution_name": "gwlens-mm",
            "installed_module_from_repository_source": False,
            "repository_root_pythonpath_used": True,
            "repository_src_pythonpath_used": False,
            "gpu_names": [EXPECTED_GPU_MODEL] * 3,
        },
    )
    with pytest.raises(TrainingGateError, match="GPU identity"):
        prepare_terminal_probe_review_packet(
            root,
            closeout_result_path=closeout_path,
            training_commit="1" * 40,
            review_checkout_commit="2" * 40,
            wheel_path=wheel_path,
            environment_lock_path=ROOT
            / "configs/environment/phase4-training-freeze.txt",
            wheel_test_result_path=wheel_test_path,
            gpu_names=[EXPECTED_GPU_MODEL] * 2,
        )


def test_terminal_probe_release_packet_rejects_closeout_outside_repository(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path / "repository")
    closeout_path = tmp_path / "outside-closeout.json"
    wheel_path = tmp_path / "fixture.whl"
    wheel_test_path = tmp_path / "wheel-test.json"
    _write_json(closeout_path, _closeout())
    wheel_path.write_bytes(b"wheel")
    _write_json(wheel_test_path, {})
    with pytest.raises(TrainingGateError, match="inside repository"):
        prepare_terminal_probe_review_packet(
            root,
            closeout_result_path=closeout_path,
            training_commit="1" * 40,
            review_checkout_commit="2" * 40,
            wheel_path=wheel_path,
            environment_lock_path=ROOT
            / "configs/environment/phase4-training-freeze.txt",
            wheel_test_result_path=wheel_test_path,
            gpu_names=[EXPECTED_GPU_MODEL] * 3,
        )


def test_terminal_release_checkout_allows_only_exact_closeout_evidence() -> None:
    validate_terminal_release_checkout_paths(
        [
            "AGENTS.md",
            "docs/PROJECT_STATE.md",
            "docs/reports/PHASE4_TERMINAL_131K_CLOSEOUT_REPORT.md",
            "results/experiment_registry.csv",
            "results/phase4/terminal_131k_execution_result.json",
            "results/phase4/terminal_probe_closeout.json",
        ]
    )
    with pytest.raises(TrainingGateError, match="changed frozen software"):
        validate_terminal_release_checkout_paths(
            ["src/gwlens_mm/training/model.py"]
        )


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_release_script_accepts_clean_evidence_only_descendant(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Test")
    source = tmp_path / "src/gwlens_mm/training/model.py"
    source.parent.mkdir(parents=True)
    source.write_text("frozen = True\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "freeze training")
    training_commit = _git(tmp_path, "rev-parse", "HEAD")

    closeout = tmp_path / "results/phase4/terminal_probe_closeout.json"
    closeout.parent.mkdir(parents=True)
    closeout.write_text("{}\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "record closeout")
    assert _verify_release_checkout(tmp_path, training_commit) == _git(
        tmp_path, "rev-parse", "HEAD"
    )


def test_release_script_rejects_descendant_software_change(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Test")
    source = tmp_path / "src/gwlens_mm/training/model.py"
    source.parent.mkdir(parents=True)
    source.write_text("frozen = True\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "freeze training")
    training_commit = _git(tmp_path, "rev-parse", "HEAD")
    source.write_text("frozen = False\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "change training")
    with pytest.raises(TrainingGateError, match="changed frozen software"):
        _verify_release_checkout(tmp_path, training_commit)
