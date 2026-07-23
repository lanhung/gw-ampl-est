from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from gwlens_mm.config import load_yaml
from gwlens_mm.training import terminal_architecture_authorization as authorization_module
from gwlens_mm.training.architecture import NEW_ARCHITECTURE_IDS
from gwlens_mm.training.contracts import TrainingGateError, model_configuration_hash
from gwlens_mm.training.terminal_architecture import _validate_terminal_probe_reuse
from gwlens_mm.training.terminal_architecture_authorization import (
    AUTHORIZATION_STATUS,
    CLOSED_BOUNDARIES,
    RELEASE_STATUS,
    REVIEW_STATUS,
    build_terminal_architecture_authorization,
    build_terminal_architecture_release_packet,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path | str]:
    root = tmp_path / "repo"
    for relative in (
        "configs/models/phase4_probe_nsf.yaml",
        "configs/models/phase5_architecture_grid.yaml",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    remote = tmp_path / "remote"
    monkeypatch.setattr(authorization_module, "PROJECT_ROOT", remote)
    (root / "SYNCED_COMMIT").write_text("d" * 40 + "\n", encoding="utf-8")
    model_hash = model_configuration_hash(
        load_yaml(root / "configs/models/phase4_probe_nsf.yaml")
    )
    probe_root = remote / "training" / "terminal-probe"
    shared = {
        "model_configuration_hash": model_hash,
        "training_code_commit": "1" * 40,
        "training_environment_sha256": "2" * 64,
        "train_manifest_sha256": "3" * 64,
        "validation_manifest_sha256": "4" * 64,
        "final_evaluation_commitment_sha256": "5" * 64,
        "membership_sha256": "6" * 64,
        "input_standardizer_sha256": "7" * 64,
        "target_standardizer_sha256": "8" * 64,
        "training_rung_count": 131072,
    }
    for seed in (0, 1, 2):
        directory = probe_root / "rung-131072" / f"seed-{seed}"
        summary = {
            "status": "completed_131k_probe_fit_and_development_validation",
            "identity": {**shared, "seed": seed},
            "development": {
                "status": "completed_development_validation",
                "case_count": 6144,
                "posthoc_calibration_applied": False,
                "final_evaluation_accessed": False,
            },
            "architecture_selection_authorized": False,
            "final_evaluation_accessed": False,
            "extension_above_131072_authorized": False,
        }
        _write_json(directory / "run_summary.json", summary)
        (directory / "best.ckpt").write_bytes(f"checkpoint-{seed}".encode())
    _write_json(
        probe_root / "rung-131072" / "rung_preparation.json",
        {
            "status": "ready_for_authorized_probe_fits",
            "rung_count": 131072,
            "member_count": 131072,
            "optimizer_started": False,
        },
    )
    decision_path = probe_root / "learning_curve_65k_to_131k_decision.json"
    _write_json(
        decision_path,
        {
            "decision": "lock_train_131k_saturated",
            "comparison": "corrected_train_65k_to_train_131k_terminal",
            "selected_training_count": 131072,
            "architecture_selection_review_allowed": True,
            "extension_above_131072_authorized": False,
        },
    )
    terminal_authorization_path = (
        root / "configs/execution/phase4_terminal_131k_probe_authorization.yaml"
    )
    terminal_authorization_path.parent.mkdir(parents=True, exist_ok=True)
    terminal_authorization = {
        "authorization_status": "authorized_terminal_131k_probe_only",
        "authorization": {"architecture_selection_authorized": False},
        "frozen_preregistration": {
            "version": "1.2.0-rc.1",
            "path": "configs/statistics/terminal_131k_preregistration.yaml",
            "canonical_hash": "9" * 64,
        },
        "training_output_root": str(probe_root),
        "learning_curve_output_path": str(decision_path),
        "corrected_65k_publication": {
            "correction_parent_manifest_sha256": "a" * 64,
        },
        "terminal_publication": {
            "combined_manifest_sha256": shared["train_manifest_sha256"],
            "train_parent_manifest_sha256": "b" * 64,
            "development_tail_manifest_sha256": "c" * 64,
        },
        "publication_roots": {
            "stage_a": str(remote / "data/stage-a/published/a"),
            "stage_b": str(remote / "data/stage-b/published/b"),
            "combined_base": str(remote / "data/base/published/base"),
            "correction": str(remote / "data/correction/published/correction"),
            "terminal_train_increment": str(
                remote / "data/increment/published/increment"
            ),
            "terminal_combined_131k": str(
                remote / "data/combined/published/combined"
            ),
            "development_tail": str(remote / "data/tail/published/tail"),
        },
        "final_evaluation_commitment_sha256": shared[
            "final_evaluation_commitment_sha256"
        ],
        "immutable_training": {"model_configuration_hash": model_hash},
    }
    terminal_authorization_path.write_text(
        yaml.safe_dump(terminal_authorization, sort_keys=False), encoding="utf-8"
    )
    wheel = remote / "artifacts/architecture.whl"
    wheel.parent.mkdir(parents=True, exist_ok=True)
    wheel.write_bytes(b"exact wheel")
    environment = remote / "review/environment.lock"
    environment.parent.mkdir(parents=True, exist_ok=True)
    environment.write_text("locked\n", encoding="utf-8")
    wheel_result = remote / "manifests/exact-wheel.json"
    _write_json(
        wheel_result,
        {
            "status": "passed_exact_wheel_on_autodl",
            "wheel_path": str(wheel),
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "focused_test_exit_code": 0,
            "full_test_exit_code": 0,
            "editable_install_used": False,
            "installed_module_from_repository_source": False,
            "torch_cuda_available": True,
            "scientific_data_opened": False,
            "optimizer_started": False,
            "gpu_names": ["NVIDIA RTX 5000 Ada Generation"] * 4,
        },
    )
    return {
        "root": root,
        "remote": remote,
        "probe_root": probe_root,
        "decision": decision_path,
        "terminal_authorization": terminal_authorization_path,
        "wheel": wheel,
        "environment": environment,
        "wheel_result": wheel_result,
        "architecture_output": remote / "training/architecture-grid",
        "packet": root / "results/phase5/terminal_architecture_release_packet.json",
        "review": root / "results/phase5/terminal_architecture_delegated_review.json",
        "authorization": (
            root
            / "configs/execution/phase5_terminal_131k_architecture_authorization.yaml"
        ),
        "training_commit": "d" * 40,
    }


def _packet(paths: dict[str, Path | str]) -> dict[str, object]:
    return dict(
        build_terminal_architecture_release_packet(
            Path(paths["root"]),
            terminal_probe_authorization_path=Path(
                paths["terminal_authorization"]
            ),
            terminal_decision_path=Path(paths["decision"]),
            probe_output_root=Path(paths["probe_root"]),
            training_commit=str(paths["training_commit"]),
            wheel_path=Path(paths["wheel"]),
            exact_wheel_test_result_path=Path(paths["wheel_result"]),
            environment_lock_path=Path(paths["environment"]),
            architecture_output_root=Path(paths["architecture_output"]),
            output_path=Path(paths["packet"]),
        )
    )


def _review(paths: dict[str, Path | str], packet: dict[str, object]) -> None:
    packet_path = Path(paths["packet"])
    _write_json(packet_path, packet)
    _write_json(
        Path(paths["review"]),
        {
            "status": REVIEW_STATUS,
            "reviewed_by": "codex_as_delegated_scientific_and_engineering_reviewer",
            "review_date": "2026-07-23",
            "reviewed_release_packet_sha256": hashlib.sha256(
                packet_path.read_bytes()
            ).hexdigest(),
            "authorization_scope": {
                "locked_training_rung": 131072,
                "authorized_new_architecture_ids": list(NEW_ARCHITECTURE_IDS),
                "authorized_training_seeds": [0, 1, 2],
                "publication_data_access_authorized": True,
                "new_architecture_fit_execution_authorized": True,
                "architecture_selection_execution_authorized": True,
                "probe_output_root": str(paths["probe_root"]),
                "architecture_output_root": str(paths["architecture_output"]),
                "architecture_selection_output_path": str(
                    Path(paths["architecture_output"])
                    / "architecture_selection.json"
                ),
            },
            "closed_boundaries": {key: False for key in CLOSED_BOUNDARIES},
        },
    )


def test_release_and_authorization_bind_exact_probe_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    packet = _packet(paths)
    assert packet["status"] == RELEASE_STATUS
    assert packet["architecture_fit_execution_authorized"] is False
    probe = packet["terminal_probe"]
    assert isinstance(probe, dict)
    assert set(probe["run_summary_sha256"]) == {"0", "1", "2"}
    assert set(probe["best_checkpoint_sha256"]) == {"0", "1", "2"}
    assert packet["architecture_grid"]["new_architecture_ids"] == list(
        NEW_ARCHITECTURE_IDS
    )
    _review(paths, packet)
    authorization = build_terminal_architecture_authorization(
        Path(paths["root"]),
        release_packet_path=Path(paths["packet"]),
        delegated_review_path=Path(paths["review"]),
        output_path=Path(paths["authorization"]),
    )
    assert authorization["authorization_status"] == AUTHORIZATION_STATUS
    assert authorization["locked_training_rung"] == 131072
    assert authorization["authorized_new_architecture_ids"] == list(
        NEW_ARCHITECTURE_IDS
    )
    assert authorization["authorization"][
        "new_architecture_fit_execution_authorized"
    ] is True
    assert authorization["authorization"]["calibration_authorized"] is False
    assert authorization["reused_probe_best_checkpoint_sha256"] == probe[
        "best_checkpoint_sha256"
    ]


def test_release_rejects_incomplete_or_changed_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    checkpoint = Path(paths["probe_root"]) / "rung-131072/seed-1/best.ckpt"
    checkpoint.unlink()
    with pytest.raises(TrainingGateError, match="artifact is incomplete"):
        _packet(paths)


def test_runtime_reuse_validation_hash_binds_checkpoint_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    packet = _packet(paths)
    _review(paths, packet)
    authorization = build_terminal_architecture_authorization(
        Path(paths["root"]),
        release_packet_path=Path(paths["packet"]),
        delegated_review_path=Path(paths["review"]),
        output_path=Path(paths["authorization"]),
    )
    _validate_terminal_probe_reuse(
        probe_output_root=Path(paths["probe_root"]),
        authorization=authorization,
        train_manifest_sha256="3" * 64,
        validation_manifest_sha256="4" * 64,
    )
    checkpoint = Path(paths["probe_root"]) / "rung-131072/seed-2/best.ckpt"
    checkpoint.write_bytes(b"changed")
    with pytest.raises(TrainingGateError, match="checkpoint hash mismatch"):
        _validate_terminal_probe_reuse(
            probe_output_root=Path(paths["probe_root"]),
            authorization=authorization,
            train_manifest_sha256="3" * 64,
            validation_manifest_sha256="4" * 64,
        )


@pytest.mark.parametrize("drift", ("packet", "rung", "output", "downstream"))
def test_authorization_rejects_review_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    packet = _packet(paths)
    _review(paths, packet)
    review_path = Path(paths["review"])
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if drift == "packet":
        review["reviewed_release_packet_sha256"] = "0" * 64
    elif drift == "rung":
        review["authorization_scope"]["locked_training_rung"] = 65536
    elif drift == "output":
        review["authorization_scope"]["architecture_output_root"] = str(
            Path(paths["remote"]) / "other"
        )
    else:
        review["closed_boundaries"]["calibration_authorized"] = True
    _write_json(review_path, review)
    with pytest.raises(TrainingGateError):
        build_terminal_architecture_authorization(
            Path(paths["root"]),
            release_packet_path=Path(paths["packet"]),
            delegated_review_path=review_path,
            output_path=Path(paths["authorization"]),
        )
