from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gwlens_mm.training.contracts import TrainingGateError
from gwlens_mm.training.engine import TrainingRunIdentity
from gwlens_mm.training.terminal131 import (
    EXPECTED_TERMINAL_GPU_MODEL,
    TAIL_STRATA,
    validate_terminal_131k_training_gate,
    validate_terminal_probe_release_binding,
)

ROOT = Path(__file__).resolve().parents[1]


def test_training_identity_accepts_only_the_frozen_terminal_rung() -> None:
    identity = TrainingRunIdentity(
        model_configuration_hash="0" * 64,
        training_code_commit="1" * 40,
        training_environment_sha256="2" * 64,
        train_manifest_sha256="3" * 64,
        validation_manifest_sha256="4" * 64,
        final_evaluation_commitment_sha256="5" * 64,
        membership_sha256="6" * 64,
        input_standardizer_sha256="7" * 64,
        target_standardizer_sha256="8" * 64,
        training_rung_count=131072,
        seed=0,
    )
    identity.validate()
    invalid = TrainingRunIdentity(**{**identity.__dict__, "training_rung_count": 262144})
    with pytest.raises(ValueError, match="unregistered rung"):
        invalid.validate()


def test_terminal_probe_execution_remains_closed_without_exact_gate(
    tmp_path: Path,
) -> None:
    with pytest.raises(TrainingGateError, match="authorization is absent"):
        validate_terminal_131k_training_gate(
            ROOT,
            authorization_path=(
                ROOT
                / "configs/execution/phase4_terminal_131k_preregistration_authorization.yaml"
            ),
            stage_a_publication_root=tmp_path / "stage-a",
            stage_b_publication_root=tmp_path / "stage-b",
            combined_base_publication_root=tmp_path / "combined-base",
            correction_publication_root=tmp_path / "correction",
            train_parent_root=tmp_path / "train-increment",
            combined_131k_publication_root=tmp_path / "combined-131k",
            development_tail_parent_root=tmp_path / "tail",
        )


def test_terminal_tail_contract_has_four_distinct_strata() -> None:
    assert len(TAIL_STRATA) == 4
    assert len(set(TAIL_STRATA)) == 4


def test_terminal_runner_dry_plan_is_execution_disabled(tmp_path: Path) -> None:
    from scripts.phase4.run_probe_131k import main

    result_path = tmp_path / "plan.json"
    assert main(["--result", str(result_path)]) == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "implementation_ready_terminal_131k_training_blocked"
    assert result["planned_rung"] == 131072
    assert result["architecture_selection_authorized"] is False
    assert result["extension_above_131072_authorized"] is False


def _release_packet() -> dict[str, object]:
    return {
        "status": "ready_for_delegated_terminal_probe_authorization_review",
        "authorization_created": False,
        "optimizer_execution_authorized": False,
        "authorized_training_rungs_preview": [131072],
        "authorized_training_seeds_preview": [0, 1, 2],
        "architecture_selection_authorized": False,
        "calibration_authorized": False,
        "sbc_authorized": False,
        "final_evaluation_authorized": False,
        "extension_above_131072_authorized": False,
        "gwosc_gwtc_access_authorized": False,
        "publication": {
            "combined_manifest_sha256": "1" * 64,
            "train_parent_manifest_sha256": "2" * 64,
            "development_tail_manifest_sha256": "3" * 64,
            "logical_train_accepted_count": 131072,
            "development_tail_accepted_count": 512,
        },
        "immutable_training": {
            "git_commit": "4" * 40,
            "wheel_path": "/root/autodl-tmp/lensing-4/artifacts/terminal.whl",
            "wheel_filename": "terminal.whl",
            "wheel_sha256": "5" * 64,
            "model_configuration_path": "configs/models/phase4_probe_nsf.yaml",
            "model_configuration_hash": "6" * 64,
            "environment_lock_path": "configs/environment/phase4-training-freeze.txt",
            "environment_lock_sha256": "7" * 64,
            "editable_install_authorized": False,
            "cuda_required": True,
            "observed_gpu_names": [EXPECTED_TERMINAL_GPU_MODEL] * 4,
            "exact_wheel_test_result_path": "/root/autodl-tmp/test.json",
            "exact_wheel_test_result_sha256": "8" * 64,
        },
        "final_evaluation_commitment_sha256": "9" * 64,
    }


def _release_authorization(
    tmp_path: Path, packet: dict[str, object]
) -> dict[str, object]:
    packet_path = tmp_path / "results/phase4/terminal-probe-release.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(json.dumps(packet, sort_keys=True) + "\n", encoding="utf-8")
    training = packet["immutable_training"]
    publication = packet["publication"]
    assert isinstance(training, dict)
    assert isinstance(publication, dict)
    return {
        "terminal_probe_release_review": {
            "path": "results/phase4/terminal-probe-release.json",
            "sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
            "delegated_review_status": (
                "accepted_for_exact_terminal_probe_authorization"
            ),
        },
        "terminal_publication": dict(publication),
        "immutable_training": dict(training),
        "final_evaluation_commitment_sha256": "9" * 64,
    }


def test_terminal_probe_release_packet_is_hash_and_identity_bound(
    tmp_path: Path,
) -> None:
    packet = _release_packet()
    authorization = _release_authorization(tmp_path, packet)
    assert validate_terminal_probe_release_binding(tmp_path, authorization)["status"] == (
        "ready_for_delegated_terminal_probe_authorization_review"
    )


@pytest.mark.parametrize(
    "drift",
    ("packet_hash", "publication", "wheel", "gpu", "optimizer", "review"),
)
def test_terminal_probe_release_binding_fails_closed(
    tmp_path: Path, drift: str
) -> None:
    packet = _release_packet()
    authorization = _release_authorization(tmp_path, packet)
    if drift == "packet_hash":
        authorization["terminal_probe_release_review"]["sha256"] = "0" * 64
    elif drift == "publication":
        authorization["terminal_publication"]["combined_manifest_sha256"] = "a" * 64
    elif drift == "wheel":
        authorization["immutable_training"]["wheel_sha256"] = "b" * 64
    elif drift == "gpu":
        packet["immutable_training"]["observed_gpu_names"] = ["other"] * 4
        authorization = _release_authorization(tmp_path, packet)
    elif drift == "optimizer":
        packet["optimizer_execution_authorized"] = True
        authorization = _release_authorization(tmp_path, packet)
    else:
        authorization["terminal_probe_release_review"][
            "delegated_review_status"
        ] = "pending"
    with pytest.raises(TrainingGateError, match="release|packet|CUDA"):
        validate_terminal_probe_release_binding(tmp_path, authorization)


def test_terminal_probe_release_binding_rejects_host_absolute_packet_path(
    tmp_path: Path,
) -> None:
    packet = _release_packet()
    authorization = _release_authorization(tmp_path, packet)
    authorization["terminal_probe_release_review"]["path"] = str(
        tmp_path / "results/phase4/terminal-probe-release.json"
    )
    with pytest.raises(TrainingGateError, match="repository-relative"):
        validate_terminal_probe_release_binding(tmp_path, authorization)
