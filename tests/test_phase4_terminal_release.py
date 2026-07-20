from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gwlens_mm.training.contracts import TrainingGateError
from gwlens_mm.training.terminal_release import (
    ENVIRONMENT_LOCK_HASH,
    EXPECTED_GPU_MODEL,
    prepare_terminal_probe_review_packet,
)

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


def _packet(tmp_path: Path) -> dict[str, object]:
    closeout_path = tmp_path / "closeout.json"
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
            ROOT,
            closeout_result_path=closeout_path,
            training_commit="1" * 40,
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
    closeout_path = tmp_path / "closeout.json"
    wheel_path = tmp_path / "fixture.whl"
    wheel_test_path = tmp_path / "wheel-test.json"
    _write_json(closeout_path, closeout)
    wheel_path.write_bytes(b"wheel")
    _write_json(wheel_test_path, {})
    with pytest.raises(TrainingGateError, match="tree replay|closeout contract"):
        prepare_terminal_probe_review_packet(
            ROOT,
            closeout_result_path=closeout_path,
            training_commit="1" * 40,
            wheel_path=wheel_path,
            environment_lock_path=ROOT
            / "configs/environment/phase4-training-freeze.txt",
            wheel_test_result_path=wheel_test_path,
            gpu_names=[EXPECTED_GPU_MODEL] * 3,
        )


def test_terminal_probe_release_packet_rejects_gpu_or_wheel_test_drift(
    tmp_path: Path,
) -> None:
    closeout_path = tmp_path / "closeout.json"
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
            ROOT,
            closeout_result_path=closeout_path,
            training_commit="1" * 40,
            wheel_path=wheel_path,
            environment_lock_path=ROOT
            / "configs/environment/phase4-training-freeze.txt",
            wheel_test_result_path=wheel_test_path,
            gpu_names=[EXPECTED_GPU_MODEL] * 2,
        )
