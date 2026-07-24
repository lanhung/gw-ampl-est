from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from gwlens_mm.config import load_yaml
from gwlens_mm.production import calibration_sbc_authorization as module
from gwlens_mm.production.calibration_sbc import (
    BASE_GENERATOR_COMMIT,
    COMBINED_BASE_MANIFEST_HASH,
    CONFIG_PATH,
    CORRECTION_GENERATOR_COMMIT,
    CORRECTION_PARENT_MANIFEST_HASH,
    CORRECTION_PUBLICATION_TREE_HASH,
    RC4_HASH,
    validate_future_materialization_authorization,
)
from gwlens_mm.production.calibration_sbc_authorization import (
    AUTHORIZATION_STATUS,
    CLOSED_FLAGS,
    RELEASE_STATUS,
    REVIEW_STATUS,
    build_calibration_sbc_materialization_authorization,
    build_calibration_sbc_materialization_release_packet,
)
from gwlens_mm.production.waveform_correction import (
    CORRECTION_PREREGISTRATION_HASH,
)
from gwlens_mm.training.contracts import TrainingGateError

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path | str | dict[str, object]]:
    root = tmp_path / "repo"
    for relative in (
        "configs/models/phase4_probe_nsf.yaml",
        "configs/models/phase5_architecture_grid.yaml",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    (root / "SYNCED_COMMIT").write_text("a" * 40 + "\n", encoding="utf-8")
    config = load_yaml(ROOT / CONFIG_PATH)
    monkeypatch.setattr(
        module, "load_calibration_sbc_contract", lambda _: (config, {})
    )
    remote = tmp_path / "remote"
    remote.mkdir()
    monkeypatch.setattr(module, "PROJECT_ROOT", remote)
    corrected = {
        "base_generator_commit": BASE_GENERATOR_COMMIT,
        "base_preregistration_hash": RC4_HASH,
        "correction_generator_commit": CORRECTION_GENERATOR_COMMIT,
        "correction_preregistration_hash": CORRECTION_PREREGISTRATION_HASH,
        "correction_parent_manifest_sha256": CORRECTION_PARENT_MANIFEST_HASH,
        "correction_publication_tree_sha256": CORRECTION_PUBLICATION_TREE_HASH,
        "combined_base_manifest_sha256": COMBINED_BASE_MANIFEST_HASH,
    }
    roots = {
        "stage_a": str(remote / "data/stage-a"),
        "stage_b": str(remote / "data/stage-b"),
        "combined_base": str(remote / "data/combined-base"),
        "correction": str(remote / "data/correction"),
        "terminal_train_increment": str(remote / "data/increment"),
        "terminal_combined_131k": str(remote / "data/combined-131k"),
        "development_tail": str(remote / "data/tail"),
    }
    terminal_authorization_path = (
        root / "configs/execution/phase4_terminal_131k_probe_authorization.yaml"
    )
    terminal_authorization_path.parent.mkdir(parents=True, exist_ok=True)
    terminal_authorization_path.write_text(
        yaml.safe_dump(
            {
                "authorization_status": "authorized_terminal_131k_probe_only",
                "corrected_65k_publication": corrected,
                "terminal_publication": {
                    "combined_manifest_sha256": "1" * 64,
                    "train_parent_manifest_sha256": "2" * 64,
                    "development_tail_manifest_sha256": "3" * 64,
                },
                "publication_roots": roots,
                "retained_65k_probe": {
                    "shared_identity": {
                        "validation_manifest_sha256": "4" * 64
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    local_terminal_decision = remote / "training/terminal-decision.json"
    local_architecture_decision = remote / "training/architecture-decision.json"
    _write_json(
        local_terminal_decision,
        {
            "status": "terminal_learning_curve_decision_complete",
            "comparison": "corrected_train_65k_to_train_131k_terminal",
            "decision": "lock_train_131k_saturated",
            "selected_training_count": 131072,
            "architecture_selection_review_allowed": True,
            "extension_above_131072_authorized": False,
            "all_three_probe_seeds_retained": True,
            "best_seed_selected": False,
            "calibration_accessed": False,
            "final_evaluation_accessed": False,
        },
    )
    _write_json(
        local_architecture_decision,
        {
            "status": "architecture_locked_on_development_validation",
            "selected_architecture_id": "nsf-t10-w256",
            "selection_metric": "mean_validation_nlp_across_three_seeds",
            "best_seed_selected": False,
            "total_result_count": 12,
            "new_fit_count": 9,
            "reused_probe_fit_count": 3,
            "calibration_accessed": False,
            "sbc_accessed": False,
            "final_evaluation_accessed": False,
            "opens_later_gate_automatically": False,
        },
    )
    wheel = remote / "artifacts/gwlens.whl"
    wheel.parent.mkdir(parents=True, exist_ok=True)
    wheel.write_bytes(b"exact calibration wheel")
    environment = remote / "environment.lock"
    environment.write_text("locked\n", encoding="utf-8")
    wheel_result = remote / "wheel-result.json"
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
            "scientific_data_opened": False,
            "optimizer_started": False,
        },
    )
    paths: dict[str, Path | str | dict[str, object]] = {
        "root": root,
        "config": config,
        "terminal_authorization": terminal_authorization_path,
        "terminal_decision": local_terminal_decision,
        "architecture_decision": local_architecture_decision,
        "wheel": wheel,
        "environment": environment,
        "wheel_result": wheel_result,
        "packet": root
        / "results/phase6/calibration_sbc_materialization_release_packet.json",
        "review": root
        / "results/phase6/calibration_sbc_materialization_review.json",
        "authorization": root
        / "configs/execution/phase6_calibration_sbc_materialization_authorization.yaml",
        "implementation_commit": "a" * 40,
    }
    return paths


def _packet(paths: dict[str, Path | str | dict[str, object]]) -> dict[str, object]:
    return dict(
        build_calibration_sbc_materialization_release_packet(
            Path(paths["root"]),
            implementation_commit=str(paths["implementation_commit"]),
            terminal_probe_authorization_path=Path(paths["terminal_authorization"]),
            terminal_decision_path=Path(paths["terminal_decision"]),
            architecture_decision_path=Path(paths["architecture_decision"]),
            wheel_path=Path(paths["wheel"]),
            exact_wheel_test_result_path=Path(paths["wheel_result"]),
            environment_lock_path=Path(paths["environment"]),
            output_path=Path(paths["packet"]),
        )
    )


def _review(
    paths: dict[str, Path | str | dict[str, object]], packet: dict[str, object]
) -> None:
    packet_path = Path(paths["packet"])
    _write_json(packet_path, packet)
    architecture = packet["architecture_decision"]
    assert isinstance(architecture, dict)
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
                "training_reference_mode": "terminal_131k",
                "locked_training_rung": 131072,
                "selected_architecture_id": architecture[
                    "selected_architecture_id"
                ],
                "three_model_seeds_retained": True,
                "materialization_execution_authorized": True,
                "calibration_fit_accepted_count": 4096,
                "sbc_diagnostic_accepted_count": 2048,
                "total_accepted_count": 6144,
            },
            "closed_boundaries": {key: False for key in CLOSED_FLAGS},
        },
    )


def test_release_and_authorization_build_exact_terminal_materialization_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    packet = _packet(paths)
    assert packet["status"] == RELEASE_STATUS
    assert packet["materialization_execution_authorized"] is False
    assert packet["prospective_official_identities"]["parent_run_id"].startswith(
        "phase6-stage-c-"
    )
    _review(paths, packet)
    authorization = build_calibration_sbc_materialization_authorization(
        Path(paths["root"]),
        release_packet_path=Path(paths["packet"]),
        delegated_review_path=Path(paths["review"]),
        output_path=Path(paths["authorization"]),
    )
    assert authorization["authorization_status"] == AUTHORIZATION_STATUS
    assert authorization["training_reference_mode"] == "terminal_131k"
    assert len(authorization["published_reference_datasets"]) == 7
    validate_future_materialization_authorization(
        authorization,
        config=paths["config"],
        generator_commit=str(paths["implementation_commit"]),
    )


def test_release_rejects_changed_terminal_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    authorization_path = Path(paths["terminal_authorization"])
    authorization = yaml.safe_load(authorization_path.read_text(encoding="utf-8"))
    authorization["terminal_publication"]["combined_manifest_sha256"] = "bad"
    authorization_path.write_text(
        yaml.safe_dump(authorization, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(TrainingGateError, match="SHA-256"):
        _packet(paths)


def test_release_rejects_checkout_commit_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    (Path(paths["root"]) / "SYNCED_COMMIT").write_text(
        "b" * 40 + "\n", encoding="utf-8"
    )
    with pytest.raises(TrainingGateError, match="synced commit changed"):
        _packet(paths)


def test_delegated_review_cannot_open_downstream_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    packet = _packet(paths)
    _review(paths, packet)
    review_path = Path(paths["review"])
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["closed_boundaries"]["final_evaluation_authorized"] = True
    _write_json(review_path, review)
    with pytest.raises(TrainingGateError, match="changed scope"):
        build_calibration_sbc_materialization_authorization(
            Path(paths["root"]),
            release_packet_path=Path(paths["packet"]),
            delegated_review_path=review_path,
            output_path=Path(paths["authorization"]),
        )
