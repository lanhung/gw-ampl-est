from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from gwlens_mm.training.contracts import TrainingGateError
from gwlens_mm.training.terminal_authorization import (
    build_terminal_probe_authorization,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_root(tmp_path: Path) -> Path:
    for relative in (
        "configs/data/phase4_terminal_131k.yaml",
        "configs/statistics/terminal_131k_preregistration.yaml",
        "results/phase4/final_evaluation_commitment.json",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return tmp_path


def _packet_and_review(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = _fixture_root(tmp_path)
    closeout_path = root / "results/phase4/terminal_probe_closeout.json"
    closeout = {
        "status": "terminal_131k_independent_closeout_passed",
        "execution_mode": "parallel_tail_recovery",
        "new_train_accepted_count": 65536,
        "development_tail_accepted_count": 512,
        "logical_train_accepted_count": 131072,
        "combined_manifest_sha256": "1" * 64,
        "train_parent_manifest_sha256": "2" * 64,
        "development_tail_manifest_sha256": "3" * 64,
        "parent_run_id": "parent",
        "development_tail_parent_id": "tail",
        "combined_train_id": "combined",
        "observed_remaining_free_bytes": 150_000_000_000,
        "tree_evidence": {"recomputed": True},
        "publication_roots": {
            "terminal_train_increment": (
                "/root/autodl-tmp/lensing-4/data/train/published/parent"
            ),
            "terminal_combined_131k": (
                "/root/autodl-tmp/lensing-4/data/combined/published/combined"
            ),
            "development_tail": (
                "/root/autodl-tmp/lensing-4/data/tail/published/tail"
            ),
        },
    }
    _write_json(closeout_path, closeout)
    packet_path = root / "results/phase4/terminal_probe_release_packet.json"
    packet = {
        "status": "ready_for_delegated_terminal_probe_authorization_review",
        "authorization_created": False,
        "optimizer_execution_authorized": False,
        "release_review_checkout_commit": "a" * 40,
        "authorized_training_rungs_preview": [131072],
        "authorized_training_seeds_preview": [0, 1, 2],
        "architecture_selection_authorized": False,
        "calibration_authorized": False,
        "sbc_authorized": False,
        "final_evaluation_authorized": False,
        "extension_above_131072_authorized": False,
        "gwosc_gwtc_access_authorized": False,
        "closeout_result_path": "results/phase4/terminal_probe_closeout.json",
        "closeout_result_sha256": hashlib.sha256(
            closeout_path.read_bytes()
        ).hexdigest(),
        "terminal_preregistration": {
            "path": "configs/statistics/terminal_131k_preregistration.yaml",
            "canonical_hash": (
                "77ff5b6b45b886657e90023c50ae002afffb077db594c80665166d537fd2346a"
            ),
        },
        "publication": {
            "combined_manifest_sha256": "1" * 64,
            "train_parent_manifest_sha256": "2" * 64,
            "development_tail_manifest_sha256": "3" * 64,
            "logical_train_accepted_count": 131072,
            "development_tail_accepted_count": 512,
        },
        "immutable_training": {
            "git_commit": "4" * 40,
            "wheel_path": "/root/autodl-tmp/lensing-4/artifacts/final.whl",
            "wheel_filename": "final.whl",
            "wheel_sha256": "5" * 64,
            "model_configuration_path": "configs/models/phase4_probe_nsf.yaml",
            "model_configuration_hash": "6" * 64,
            "environment_lock_path": "configs/environment/phase4-training-freeze.txt",
            "environment_lock_sha256": "7" * 64,
            "editable_install_authorized": False,
            "cuda_required": True,
            "observed_gpu_names": ["NVIDIA RTX 5000 Ada Generation"] * 4,
            "exact_wheel_test_result_path": str(tmp_path / "wheel-test.json"),
            "exact_wheel_test_result_sha256": "8" * 64,
        },
        "final_evaluation_commitment_sha256": "9" * 64,
    }
    _write_json(packet_path, packet)
    review_path = root / "results/phase4/terminal_probe_delegated_review.json"
    review = {
        "status": "delegated_terminal_probe_authorization_approved",
        "reviewed_by": "codex_as_delegated_scientific_and_engineering_reviewer",
        "review_date": "2026-07-20",
        "reviewed_release_packet_sha256": hashlib.sha256(
            packet_path.read_bytes()
        ).hexdigest(),
        "authorization_scope": {
            "training_rung": 131072,
            "training_seeds": [0, 1, 2],
            "publication_data_access_authorized": True,
            "probe_optimizer_execution_authorized": True,
            "terminal_learning_curve_decision_authorized": True,
            "retained_65k_output_root": (
                "/root/autodl-tmp/lensing-4/training/retained-65k"
            ),
            "training_output_root": (
                "/root/autodl-tmp/lensing-4/training/terminal-131k"
            ),
        },
        "closed_boundaries": {
            "model_tuning_authorized": False,
            "architecture_selection_authorized": False,
            "calibration_authorized": False,
            "sbc_authorized": False,
            "final_evaluation_authorized": False,
            "extension_above_131072_authorized": False,
            "real_noise_authorized": False,
            "gwosc_gwtc_access_authorized": False,
        },
    }
    _write_json(review_path, review)
    return root, packet_path, review_path


def _build(tmp_path: Path) -> dict[str, object]:
    root, packet, review = _packet_and_review(tmp_path)
    return dict(
        build_terminal_probe_authorization(
            root,
            release_packet_path=packet,
            delegated_review_path=review,
            authorization_output_path=(
                root / "configs/execution/future_terminal_probe_fixture.yaml"
            ),
            retained_65k_output_root=Path(
                "/root/autodl-tmp/lensing-4/training/retained-65k"
            ),
            training_output_root=Path(
                "/root/autodl-tmp/lensing-4/training/terminal-131k"
            ),
        )
    )


def test_builder_requires_separate_review_and_closes_downstream(tmp_path: Path) -> None:
    authorization = _build(tmp_path)
    assert authorization["authorization_status"] == "authorized_terminal_131k_probe_only"
    flags = authorization["authorization"]
    assert isinstance(flags, dict)
    assert flags["probe_optimizer_execution_authorized"] is True
    assert flags["architecture_selection_authorized"] is False
    assert authorization["authorized_training_rungs"] == [131072]
    assert authorization["authorized_training_seeds"] == [0, 1, 2]
    assert authorization["terminal_probe_release_review"][
        "delegated_review_status"
    ] == "accepted_for_exact_terminal_probe_authorization"
    assert authorization["publication_roots"]["development_tail"] == (
        "/root/autodl-tmp/lensing-4/data/tail/published/tail"
    )


def test_builder_uses_dynamic_microshard_closeout_roots(tmp_path: Path) -> None:
    root, packet_path, review_path = _packet_and_review(tmp_path)
    closeout_path = root / "results/phase4/terminal_probe_closeout.json"
    closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
    closeout["execution_mode"] = "dynamic_microshard_tail_recovery"
    _write_json(closeout_path, closeout)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["closeout_result_sha256"] = hashlib.sha256(
        closeout_path.read_bytes()
    ).hexdigest()
    _write_json(packet_path, packet)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["reviewed_release_packet_sha256"] = hashlib.sha256(
        packet_path.read_bytes()
    ).hexdigest()
    _write_json(review_path, review)

    authorization = build_terminal_probe_authorization(
        root,
        release_packet_path=packet_path,
        delegated_review_path=review_path,
        authorization_output_path=(
            root / "configs/execution/future_terminal_probe_fixture.yaml"
        ),
        retained_65k_output_root=Path(
            "/root/autodl-tmp/lensing-4/training/retained-65k"
        ),
        training_output_root=Path(
            "/root/autodl-tmp/lensing-4/training/terminal-131k"
        ),
    )

    for role, publication_root in closeout["publication_roots"].items():
        assert authorization["publication_roots"][role] == publication_root


@pytest.mark.parametrize(
    ("root_key", "invalid"),
    [
        ("development_tail", "/tmp/published/tail"),
        (
            "terminal_combined_131k",
            "/root/autodl-tmp/lensing-4/data/combined/published/wrong",
        ),
    ],
)
def test_builder_rejects_parallel_recovery_publication_root_drift(
    tmp_path: Path, root_key: str, invalid: str
) -> None:
    root, packet_path, review_path = _packet_and_review(tmp_path)
    closeout_path = root / "results/phase4/terminal_probe_closeout.json"
    closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
    closeout["publication_roots"][root_key] = invalid
    _write_json(closeout_path, closeout)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["closeout_result_sha256"] = hashlib.sha256(
        closeout_path.read_bytes()
    ).hexdigest()
    _write_json(packet_path, packet)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["reviewed_release_packet_sha256"] = hashlib.sha256(
        packet_path.read_bytes()
    ).hexdigest()
    _write_json(review_path, review)
    with pytest.raises(TrainingGateError, match="publication root"):
        build_terminal_probe_authorization(
            root,
            release_packet_path=packet_path,
            delegated_review_path=review_path,
            authorization_output_path=(
                root / "configs/execution/future_terminal_probe_fixture.yaml"
            ),
            retained_65k_output_root=Path(
                "/root/autodl-tmp/lensing-4/training/retained-65k"
            ),
            training_output_root=Path(
                "/root/autodl-tmp/lensing-4/training/terminal-131k"
            ),
        )


@pytest.mark.parametrize(
    "drift",
    ("packet", "rung", "seed", "downstream", "output", "extra", "closeout"),
)
def test_builder_fails_closed_on_review_or_evidence_drift(
    tmp_path: Path, drift: str
) -> None:
    root, packet_path, review_path = _packet_and_review(tmp_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if drift == "packet":
        review["reviewed_release_packet_sha256"] = "0" * 64
    elif drift == "rung":
        review["authorization_scope"]["training_rung"] = 65536
    elif drift == "seed":
        review["authorization_scope"]["training_seeds"] = [0, 1]
    elif drift == "downstream":
        review["closed_boundaries"]["calibration_authorized"] = True
    elif drift == "output":
        review["authorization_scope"]["training_output_root"] = (
            "/root/autodl-tmp/lensing-4/training/other"
        )
    elif drift == "extra":
        review["closed_boundaries"]["unregistered_future_phase_authorized"] = False
    else:
        closeout_path = root / "results/phase4/terminal_probe_closeout.json"
        closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
        closeout["logical_train_accepted_count"] = 131071
        _write_json(closeout_path, closeout)
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        packet["closeout_result_sha256"] = hashlib.sha256(
            closeout_path.read_bytes()
        ).hexdigest()
        _write_json(packet_path, packet)
        review["reviewed_release_packet_sha256"] = hashlib.sha256(
            packet_path.read_bytes()
        ).hexdigest()
    _write_json(review_path, review)
    with pytest.raises(TrainingGateError):
        build_terminal_probe_authorization(
            root,
            release_packet_path=packet_path,
            delegated_review_path=review_path,
            authorization_output_path=(
                root / "configs/execution/future_terminal_probe_fixture.yaml"
            ),
            retained_65k_output_root=Path(
                "/root/autodl-tmp/lensing-4/training/retained-65k"
            ),
            training_output_root=Path(
                "/root/autodl-tmp/lensing-4/training/terminal-131k"
            ),
        )
