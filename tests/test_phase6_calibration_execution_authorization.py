from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from gwlens_mm.provenance import canonical_json
from gwlens_mm.training import calibration_execution_authorization as module
from gwlens_mm.training.calibration_execution_authorization import (
    SCORE_AUTHORIZATION_STATUS,
    SCORE_CONFIG_HASH,
    SCORE_RELEASE_STATUS,
    SCORE_REVIEW_STATUS,
    STATISTICS_AUTHORIZATION_STATUS,
    STATISTICS_RELEASE_STATUS,
    STATISTICS_REVIEW_STATUS,
    build_score_inference_authorization,
    build_score_inference_release_packet,
    build_statistics_authorization,
    build_statistics_release_packet,
    load_score_contract,
)
from gwlens_mm.training.contracts import TrainingGateError

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_score_inference_contract_is_hash_frozen_and_closed() -> None:
    contract = load_score_contract(ROOT)
    assert SCORE_CONFIG_HASH == (
        "47df45922b8db62970e5b0a7c8315c14d95b5fc1ac7e97b030a975fe31d4f2d8"
    )
    assert contract["locked_training_rung"] == 131072
    assert contract["model_seeds"] == [0, 1, 2]
    assert contract["output_contract"]["total_score_artifact_count"] == 6
    assert all(value is False for value in contract["closed_boundaries"].values())


def test_delegated_score_review_creates_only_six_artifact_authorization(
    tmp_path: Path,
) -> None:
    packet_path = tmp_path / "results/phase6/score_release.json"
    review_path = tmp_path / "results/phase6/score_review.json"
    output_path = tmp_path / "configs/execution/score_authorization.yaml"
    packet = {
        "status": SCORE_RELEASE_STATUS,
        "authorization_created": False,
        "score_inference_authorized": False,
        "score_contract": {
            "path": module.SCORE_CONFIG_PATH,
            "canonical_hash": SCORE_CONFIG_HASH,
            **dict(load_score_contract(ROOT)),
        },
        "selected_architecture": {
            "architecture_id": "nsf-t10-w256",
            "model_configuration_hash": "1" * 64,
            "locked_training_rung": 131072,
        },
        "selected_seed_checkpoints": {
            str(seed): {"path": f"/checkpoint/{seed}", "sha256": "2" * 64}
            for seed in (0, 1, 2)
        },
        "development_publication": {
            "parent_root": "/publication",
            "parent_manifest_sha256": "3" * 64,
        },
        "score_outputs": {
            str(seed): {
                "calibration_fit": f"/scores/{seed}/calibration.npz",
                "sbc_diagnostic": f"/scores/{seed}/sbc.npz",
            }
            for seed in (0, 1, 2)
        },
        "immutable_inference": {"git_commit": "4" * 40},
        "release_packet_repository_path": "results/phase6/score_release.json",
        "future_authorization_path": "configs/execution/score_authorization.yaml",
        "future_review_path": "results/phase6/score_review.json",
    }
    _write_json(packet_path, packet)
    review = {
        "status": SCORE_REVIEW_STATUS,
        "reviewed_release_packet_sha256": _sha256(packet_path),
        "reviewed_by": "delegated-reviewer",
        "review_date": "2026-07-24",
        "authorization_scope": {
            "locked_training_rung": 131072,
            "selected_architecture_id": "nsf-t10-w256",
            "model_seeds": [0, 1, 2],
            "score_artifact_count": 6,
            "calibration_fit_data_access_authorized": True,
            "sbc_diagnostic_data_access_authorized": True,
            "selected_checkpoint_inference_authorized": True,
            "score_artifact_creation_authorized": True,
        },
        "closed_boundaries": {
            key: False for key in module.SCORE_CLOSED_BOUNDARIES
        },
    }
    _write_json(review_path, review)
    authorization = build_score_inference_authorization(
        tmp_path,
        release_packet_path=packet_path,
        delegated_review_path=review_path,
        output_path=output_path,
    )
    assert authorization["authorization_status"] == SCORE_AUTHORIZATION_STATUS
    assert authorization["selected_architecture"]["locked_training_rung"] == 131072
    assert set(authorization["selected_seed_checkpoints"]) == {"0", "1", "2"}
    assert authorization["authorization"]["calibration_map_fitting_authorized"] is False
    changed = json.loads(json.dumps(review))
    changed["authorization_scope"]["score_artifact_count"] = 5
    _write_json(review_path, changed)
    with pytest.raises(TrainingGateError, match="not exact"):
        build_score_inference_authorization(
            tmp_path,
            release_packet_path=packet_path,
            delegated_review_path=review_path,
            output_path=output_path,
        )


def test_score_release_uses_real_materialization_architecture_field_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "_verify_checkout", lambda root, commit: None)
    monkeypatch.setattr(
        module,
        "_selected_checkpoint_artifacts",
        lambda root, **kwargs: {
            "architecture_id": "nsf-t10-w256",
            "model_configuration_hash": "1" * 64,
            "selected_seed_checkpoints": {
                str(seed): {"path": f"/checkpoint/{seed}", "sha256": "2" * 64}
                for seed in (0, 1, 2)
            },
            "shared_identity": {"training_rung_count": 131072},
        },
    )
    monkeypatch.setattr(
        module,
        "_immutable_inference",
        lambda **kwargs: {
            "git_commit": "a" * 40,
            "wheel_sha256": "3" * 64,
            "environment_lock_sha256": "4" * 64,
        },
    )
    score_config = tmp_path / module.SCORE_CONFIG_PATH
    score_config.parent.mkdir(parents=True)
    score_config.write_bytes((ROOT / module.SCORE_CONFIG_PATH).read_bytes())
    architecture_decision = tmp_path / "architecture/selection.json"
    _write_json(
        architecture_decision,
        {"selected_architecture_id": "nsf-t10-w256"},
    )
    architecture_authorization = (
        tmp_path / "configs/execution/architecture_authorization.yaml"
    )
    architecture_authorization.parent.mkdir(parents=True, exist_ok=True)
    architecture_authorization.write_text(
        yaml.safe_dump(
            {
                "authorization_status": module.ARCHITECTURE_AUTHORIZATION_STATUS,
                "architecture_selection_output_path": str(architecture_decision),
                "locked_training_rung": 131072,
            }
        )
    )
    materialization_authorization = (
        tmp_path / "configs/execution/materialization_authorization.yaml"
    )
    materialization_authorization.write_text(
        yaml.safe_dump(
            {
                "authorization_status": module.MATERIALIZATION_AUTHORIZATION_STATUS,
                "architecture_decision": {
                    "selected_architecture_id": "nsf-t10-w256",
                    "locked_training_rung": 131072,
                    "sha256": _sha256(architecture_decision),
                },
                "prospective_official_identities": {
                    "parent_run_id": "parent",
                    "calibration_dataset_id": "calibration",
                    "sbc_dataset_id": "sbc",
                },
            }
        )
    )
    publication = tmp_path / "data/publication"
    _write_json(
        publication / "dataset_manifest.json",
        {
            "status": "passed",
            "calibration_fit_accepted_count": 4096,
            "sbc_diagnostic_accepted_count": 2048,
            "accepted_pair_count": 6144,
            "complete_shard_count": 48,
            "group_disjoint_from_train_validation_and_each_other": True,
            "calibration_fit_statistics_authorized": False,
            "sbc_statistics_authorized": False,
            "checkpoint_access_authorized": False,
            "final_evaluation_authorized": False,
        },
    )
    result_path = tmp_path / "results/materialization_result.json"
    _write_json(
        result_path,
        {
            "status": "passed",
            "publication_path": str(publication),
            "parent_run_id": "parent",
            "calibration_dataset_id": "calibration",
            "sbc_dataset_id": "sbc",
            "accepted_pair_count": 6144,
            "complete_shard_count": 48,
            "publication_tree_sha256": "5" * 64,
            "calibration_fit_statistics_authorized": False,
            "sbc_statistics_authorized": False,
            "final_evaluation_authorized": False,
            "gwosc_gwtc_accessed": False,
        },
    )
    output_path = tmp_path / "results/phase6/score_release.json"
    packet = build_score_inference_release_packet(
        tmp_path,
        implementation_commit="a" * 40,
        architecture_authorization_path=architecture_authorization,
        architecture_decision_path=architecture_decision,
        materialization_authorization_path=materialization_authorization,
        materialization_result_path=result_path,
        publication_root=publication,
        wheel_path=tmp_path / "wheel.whl",
        exact_wheel_test_result_path=tmp_path / "wheel-test.json",
        environment_lock_path=tmp_path / "environment.txt",
        score_output_root=tmp_path / "scores",
        output_path=output_path,
    )
    assert packet["selected_architecture"]["locked_training_rung"] == 131072
    assert packet["development_publication"]["calibration_dataset_id"] == "calibration"
    assert packet["score_inference_authorized"] is False


def _write_score(
    path: Path,
    *,
    split: str,
    seed: int,
    identifiers: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(identifiers)
    ranks = (
        {
            f"rank_{statistic}": np.arange(count, dtype=np.int64)
            for statistic in module.SBC_STATISTICS
        }
        if split == "sbc_diagnostic"
        else {}
    )
    np.savez(
        path,
        marginal_scores=np.zeros((count, 2), dtype=np.float64),
        joint_scores=np.zeros(count, dtype=np.float64),
        em_cells=np.asarray(["cell"] * count),
        physical_system_ids=np.asarray(identifiers),
        split=np.asarray(split),
        model_seed=np.asarray(seed, dtype=np.int64),
        architecture_id=np.asarray("nsf-t10-w256"),
        checkpoint_sha256=np.asarray(str(seed + 1) * 64),
        publication_manifest_sha256=np.asarray("a" * 64),
        inference_commit=np.asarray("b" * 40),
        **ranks,
    )
    expected_draws = 4096 if split == "calibration_fit" else 1024
    summary = {
        "status": "completed_score_extraction_only",
        "split": split,
        "model_seed": seed,
        "architecture_id": "nsf-t10-w256",
        "case_count": count,
        "posterior_draw_count": expected_draws,
        "score_artifact_sha256": _sha256(path),
        "physical_system_ids_sha256": hashlib.sha256(
            canonical_json(sorted(identifiers)).encode()
        ).hexdigest(),
        "calibration_map_fitted": False,
        "sbc_statistical_test_executed": False,
        "model_retrained_or_tuned": False,
        "final_evaluation_accessed": False,
    }
    _write_json(path.with_suffix(".summary.json"), summary)


def test_completed_six_scores_bind_three_independent_statistics_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    calibration_ids = tuple(f"cal-{index:04d}" for index in range(4096))
    sbc_ids = tuple(f"sbc-{index:04d}" for index in range(1024))
    outputs: dict[str, dict[str, str]] = {}
    checkpoints: dict[str, dict[str, str]] = {}
    for seed in (0, 1, 2):
        calibration = tmp_path / f"scores/seed-{seed}/calibration_fit_scores.npz"
        sbc = tmp_path / f"scores/seed-{seed}/sbc_diagnostic_scores.npz"
        _write_score(
            calibration,
            split="calibration_fit",
            seed=seed,
            identifiers=calibration_ids,
        )
        _write_score(
            sbc,
            split="sbc_diagnostic",
            seed=seed,
            identifiers=sbc_ids,
        )
        outputs[str(seed)] = {
            "calibration_fit": str(calibration),
            "sbc_diagnostic": str(sbc),
        }
        checkpoints[str(seed)] = {
            "path": str(tmp_path / f"checkpoint-{seed}"),
            "sha256": str(seed + 1) * 64,
        }
    authorization_path = (
        tmp_path / "configs/execution/score_inference_authorization.yaml"
    )
    authorization_path.parent.mkdir(parents=True)
    authorization_path.write_text(
        yaml.safe_dump(
            {
                "authorization_status": SCORE_AUTHORIZATION_STATUS,
                "selected_architecture": {
                    "architecture_id": "nsf-t10-w256",
                    "locked_training_rung": 131072,
                },
                "selected_seed_checkpoints": checkpoints,
                "score_outputs": outputs,
            }
        )
    )
    output_path = tmp_path / "results/phase6/statistics_release.json"
    packet = build_statistics_release_packet(
        tmp_path,
        score_authorization_path=authorization_path,
        statistics_output_root=tmp_path / "statistics",
        output_path=output_path,
    )
    assert packet["status"] == STATISTICS_RELEASE_STATUS
    assert set(packet["score_artifacts_by_seed"]) == {"0", "1", "2"}
    assert set(packet["score_identities_by_seed"]) == {"0", "1", "2"}
    assert len(packet["statistics_output_roots"]) == 3
    assert packet["statistics_execution_authorized"] is False


def test_statistics_review_authorizes_all_seeds_but_no_checkpoint_or_final_access(
    tmp_path: Path,
) -> None:
    packet_path = tmp_path / "results/phase6/statistics_release.json"
    review_path = tmp_path / "results/phase6/statistics_review.json"
    output_path = tmp_path / "configs/execution/statistics_authorization.yaml"
    packet = {
        "status": STATISTICS_RELEASE_STATUS,
        "authorization_created": False,
        "statistics_execution_authorized": False,
        "selected_architecture": {
            "architecture_id": "nsf-t06-w128",
            "locked_training_rung": 131072,
        },
        "score_artifacts_by_seed": {str(seed): {} for seed in (0, 1, 2)},
        "score_identities_by_seed": {str(seed): {} for seed in (0, 1, 2)},
        "statistics_output_roots": {
            str(seed): f"/statistics/seed-{seed}" for seed in (0, 1, 2)
        },
        "release_packet_repository_path": "results/phase6/statistics_release.json",
        "future_authorization_path": (
            "configs/execution/statistics_authorization.yaml"
        ),
        "future_review_path": "results/phase6/statistics_review.json",
    }
    _write_json(packet_path, packet)
    review = {
        "status": STATISTICS_REVIEW_STATUS,
        "reviewed_release_packet_sha256": _sha256(packet_path),
        "reviewed_by": "delegated-reviewer",
        "review_date": "2026-07-24",
        "authorization_scope": {
            "locked_training_rung": 131072,
            "selected_architecture_id": "nsf-t06-w128",
            "model_seeds": [0, 1, 2],
            "calibration_map_count": 3,
            "independent_sbc_result_count": 3,
            "calibration_fit_authorized": True,
            "sbc_execution_authorized": True,
        },
        "closed_boundaries": {
            key: False for key in module.STATISTICS_CLOSED_BOUNDARIES
        },
    }
    _write_json(review_path, review)
    authorization = build_statistics_authorization(
        tmp_path,
        release_packet_path=packet_path,
        delegated_review_path=review_path,
        output_path=output_path,
    )
    assert authorization["authorization_status"] == STATISTICS_AUTHORIZATION_STATUS
    assert authorization["authorized_model_seeds"] == [0, 1, 2]
    assert authorization["authorization"]["checkpoint_access_authorized"] is False
    assert authorization["authorization"]["final_evaluation_authorized"] is False
