from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from gwlens_mm.training import (
    ablation_evaluation_authorization as release_module,
)
from gwlens_mm.training.ablation_evaluation_authorization import (
    CALIBRATION_AUTHORIZATION_STATUS,
    CALIBRATION_CLOSED_BOUNDARIES,
    CALIBRATION_RELEASE_STATUS,
    CALIBRATION_REVIEW_SCOPE,
    CALIBRATION_REVIEW_STATUS,
    IID_AUTHORIZATION_STATUS,
    IID_CLOSED_BOUNDARIES,
    IID_RELEASE_STATUS,
    IID_REVIEW_SCOPE,
    IID_REVIEW_STATUS,
    build_ablation_calibration_authorization,
    build_ablation_calibration_release_packet,
    build_ablation_iid_authorization,
    build_ablation_iid_release_packet,
    load_ablation_evaluation_release_stack_contract,
)
from gwlens_mm.training.contracts import TrainingGateError

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _calibration_scope() -> dict:
    return {
        "locked_training_rung": 131072,
        "selected_architecture_id": "nsf-t10-w256",
        "ablation_views": ["gw_only", "em_only"],
        "model_seeds": [0, 1, 2],
        "ablation_checkpoint_count": 6,
        "calibration_case_count": 4096,
        "calibration_score_artifact_count": 6,
        "calibration_map_count": 6,
        "scientific_checkpoint_access_authorized": True,
        "calibration_fit_data_access_authorized": True,
        "ablation_calibration_score_execution_authorized": True,
        "ablation_calibration_map_fit_authorized": True,
    }


def _iid_scope() -> dict:
    return {
        "locked_training_rung": 131072,
        "selected_architecture_id": "nsf-t10-w256",
        "ablation_views": ["gw_only", "em_only"],
        "model_seeds": [0, 1, 2],
        "iid_case_count": 8192,
        "ablation_iid_score_artifact_count": 6,
        "paired_comparison_count": 6,
        "final_iid_unsealing_authorized": True,
        "ablation_checkpoint_access_authorized": True,
        "matching_ablation_calibration_map_access_authorized": True,
        "primary_same_seed_iid_score_access_authorized": True,
        "ablation_iid_inference_authorized": True,
        "paired_comparison_execution_authorized": True,
    }


def _checkpoints() -> dict:
    return {
        view: {
            str(seed): {
                "run_root": f"/project/ablations/{view}/seed-{seed}",
                "checkpoint_path": f"/project/ablations/{view}/seed-{seed}/best.ckpt",
                "checkpoint_sha256": f"{seed + (1 if view == 'gw_only' else 4)}" * 64,
            }
            for seed in (0, 1, 2)
        }
        for view in ("gw_only", "em_only")
    }


def _calibration_packet() -> dict:
    output_root = "/root/autodl-tmp/lensing-4/ablation-calibration"
    return {
        "status": CALIBRATION_RELEASE_STATUS,
        "authorization_created": False,
        "ablation_calibration_execution_authorized": False,
        "selected_architecture": {
            "architecture_id": "nsf-t10-w256",
            "model_configuration_hash": "a" * 64,
            "locked_training_rung": 131072,
        },
        "ablation_checkpoints": _checkpoints(),
        "calibration_publication": {
            "parent_root": "/root/autodl-tmp/lensing-4/calibration",
            "parent_manifest_sha256": "b" * 64,
            "calibration_fit_accepted_count": 4096,
        },
        "immutable_inference": {"git_commit": "c" * 40},
        "calibration_score_outputs": {
            view: {
                str(seed): (
                    f"{output_root}/{view}/seed-{seed}/calibration_scores.npz"
                )
                for seed in (0, 1, 2)
            }
            for view in ("gw_only", "em_only")
        },
        "calibration_map_outputs": {
            view: {
                str(seed): (
                    f"{output_root}/{view}/seed-{seed}/"
                    "calibration_region_maps.json"
                )
                for seed in (0, 1, 2)
            }
            for view in ("gw_only", "em_only")
        },
        "review_scope": _calibration_scope(),
        "closed_boundaries": {
            name: False for name in CALIBRATION_CLOSED_BOUNDARIES
        },
        "future_authorization_path": (
            "configs/execution/"
            "phase7_ablation_calibration_execution_authorization.yaml"
        ),
        "future_review_path": (
            "results/phase7/ablation_calibration_execution_review.json"
        ),
        "release_packet_repository_path": (
            "results/phase7/ablation_calibration_release_packet.json"
        ),
    }


def _iid_packet() -> dict:
    output_root = "/root/autodl-tmp/lensing-4/ablation-iid"
    maps = {
        view: {
            str(seed): {
                "calibration_map_path": (
                    f"/root/autodl-tmp/lensing-4/maps/{view}/{seed}.json"
                ),
                "calibration_map_sha256": f"{seed + 1}" * 64,
            }
            for seed in (0, 1, 2)
        }
        for view in ("gw_only", "em_only")
    }
    return {
        "status": IID_RELEASE_STATUS,
        "authorization_created": False,
        "ablation_iid_execution_authorized": False,
        "selected_architecture": {
            "architecture_id": "nsf-t10-w256",
            "model_configuration_hash": "a" * 64,
            "locked_training_rung": 131072,
        },
        "iid_namespace_id": "iid-test",
        "iid_case_count": 8192,
        "ablation_checkpoints": _checkpoints(),
        "ablation_calibration_maps": maps,
        "sealed_publication": {"parent_root": "/project/final"},
        "primary_same_seed_iid_scores": {
            str(seed): {
                "path": f"/project/final/seed-{seed}/iid.npz",
                "sha256": f"{seed + 7}" * 64,
            }
            for seed in (0, 1, 2)
        },
        "immutable_inference": {"git_commit": "c" * 40},
        "ablation_iid_score_outputs": {
            view: {
                str(seed): f"{output_root}/{view}/seed-{seed}/iid_scores.npz"
                for seed in (0, 1, 2)
            }
            for view in ("gw_only", "em_only")
        },
        "paired_comparison_outputs": {
            view: {
                str(seed): (
                    f"{output_root}/{view}/seed-{seed}/paired_comparison.json"
                )
                for seed in (0, 1, 2)
            }
            for view in ("gw_only", "em_only")
        },
        "review_scope": _iid_scope(),
        "closed_boundaries": {name: False for name in IID_CLOSED_BOUNDARIES},
        "future_authorization_path": (
            "configs/execution/phase7_ablation_iid_execution_authorization.yaml"
        ),
        "future_review_path": (
            "results/phase7/ablation_iid_execution_review.json"
        ),
        "release_packet_repository_path": (
            "results/phase7/ablation_iid_release_packet.json"
        ),
    }


def test_release_stack_is_strictly_implementation_only() -> None:
    authorization = load_ablation_evaluation_release_stack_contract(ROOT)
    frozen = authorization["frozen_contracts"]
    assert frozen["ablation_checkpoint_count"] == 6
    assert frozen["calibration_case_count"] == 4096
    assert frozen["iid_case_count"] == 8192
    flags = authorization["authorization"]
    assert (
        flags[
            "nonauthorizing_calibration_release_packet_implementation_authorized"
        ]
        is True
    )
    assert flags["scientific_checkpoint_access_authorized"] is False
    assert flags["final_iid_unsealing_authorized"] is False


def test_calibration_packet_allocates_six_scores_and_six_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    project = tmp_path / "project"
    (root / "results/phase7").mkdir(parents=True)
    (root / "configs/execution").mkdir(parents=True)
    decision = project / "architecture.json"
    _write_json(decision, {"selected_architecture_id": "nsf-t10-w256"})
    training_authorization = root / "configs/execution/ablation.yaml"
    _write_yaml(
        training_authorization,
        {
            "authorization_status": "authorized_terminal_131k_ablation_training_only",
            "selected_primary_model_configuration_hash": "a" * 64,
            "selected_architecture_decision_path": str(decision),
            "selected_architecture_decision_sha256": _sha256(decision),
        },
    )
    publication = project / "calibration"
    publication.mkdir(parents=True)
    manifest = publication / "dataset_manifest.json"
    _write_json(manifest, {"status": "passed"})
    score_authorization = root / "configs/execution/score.yaml"
    _write_yaml(
        score_authorization,
        {
            "authorization_status": (
                "authorized_calibration_sbc_checkpoint_inference_only"
            ),
            "selected_architecture": {
                "architecture_id": "nsf-t10-w256",
                "model_configuration_hash": "a" * 64,
                "locked_training_rung": 131072,
            },
            "development_publication": {
                "parent_root": str(publication),
                "parent_manifest_sha256": _sha256(manifest),
                "calibration_fit_accepted_count": 4096,
            },
        },
    )
    statistics_authorization = root / "configs/execution/statistics.yaml"
    _write_yaml(
        statistics_authorization,
        {
            "authorization_status": "authorized_calibration_sbc_statistics_only"
        },
    )
    wheel = project / "release.whl"
    wheel.write_bytes(b"wheel")
    wheel_result = project / "wheel-result.json"
    _write_json(wheel_result, {})
    environment = project / "environment.txt"
    environment.write_text("environment\n")
    monkeypatch.setattr(release_module, "PROJECT_ROOT", project)
    monkeypatch.setattr(
        release_module,
        "load_ablation_evaluation_release_stack_contract",
        lambda value: {},
    )
    monkeypatch.setattr(release_module, "_verify_checkout", lambda *args: None)
    monkeypatch.setattr(
        release_module,
        "_completed_ablation_checkpoints",
        lambda value: _checkpoints(),
    )
    monkeypatch.setattr(
        release_module, "_primary_statistics_completed", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        release_module,
        "_immutable_inference",
        lambda **kwargs: {"git_commit": kwargs["implementation_commit"]},
    )
    output_root = project / "ablation-calibration"
    packet = build_ablation_calibration_release_packet(
        root,
        implementation_commit="d" * 40,
        ablation_training_authorization_path=training_authorization,
        primary_score_authorization_path=score_authorization,
        primary_statistics_authorization_path=statistics_authorization,
        wheel_path=wheel,
        exact_wheel_test_result_path=wheel_result,
        environment_lock_path=environment,
        calibration_output_root=output_root,
        output_path=(
            root / "results/phase7/ablation_calibration_release_packet.json"
        ),
    )
    assert packet["status"] == CALIBRATION_RELEASE_STATUS
    assert packet["ablation_calibration_execution_authorized"] is False
    assert sum(
        len(values) for values in packet["calibration_score_outputs"].values()
    ) == 6
    assert sum(
        len(values) for values in packet["calibration_map_outputs"].values()
    ) == 6
    assert not output_root.exists()


def test_primary_statistics_completion_requires_exact_runtime_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    roots = {}
    for seed in (0, 1, 2):
        output = project / f"seed-{seed}"
        _write_json(
            output / "run_summary.json",
            {"status": "completed_calibration_fit_and_independent_sbc"},
        )
        roots[str(seed)] = str(output)
    monkeypatch.setattr(release_module, "PROJECT_ROOT", project)
    authorization = {
        "authorization_status": "authorized_calibration_sbc_statistics_only",
        "selected_architecture": {
            "architecture_id": "nsf-t10-w256",
            "locked_training_rung": 131072,
        },
        "authorized_model_seeds": [0, 1, 2],
        "statistics_output_roots": roots,
    }
    release_module._primary_statistics_completed(  # noqa: SLF001
        authorization, selected_architecture_id="nsf-t10-w256"
    )
    changed = deepcopy(authorization)
    _write_json(
        Path(changed["statistics_output_roots"]["2"]) / "run_summary.json",
        {"status": "not-complete"},
    )
    with pytest.raises(TrainingGateError, match="did not complete"):
        release_module._primary_statistics_completed(  # noqa: SLF001
            changed, selected_architecture_id="nsf-t10-w256"
        )


def test_reviewed_calibration_packet_opens_no_iid_boundary(
    tmp_path: Path,
) -> None:
    packet_path = (
        tmp_path / "results/phase7/ablation_calibration_release_packet.json"
    )
    _write_json(packet_path, _calibration_packet())
    review_path = (
        tmp_path / "results/phase7/ablation_calibration_execution_review.json"
    )
    _write_json(
        review_path,
        {
            "status": CALIBRATION_REVIEW_STATUS,
            "reviewed_release_packet_sha256": _sha256(packet_path),
            "reviewed_by": (
                "codex_as_delegated_scientific_and_engineering_reviewer"
            ),
            "review_date": "2026-07-24",
            "authorization_scope": _calibration_scope(),
            "closed_boundaries": {
                name: False for name in CALIBRATION_CLOSED_BOUNDARIES
            },
        },
    )
    output = (
        tmp_path
        / "configs/execution/"
        "phase7_ablation_calibration_execution_authorization.yaml"
    )
    authorization = build_ablation_calibration_authorization(
        tmp_path,
        release_packet_path=packet_path,
        delegated_review_path=review_path,
        output_path=output,
    )
    assert authorization["authorization_status"] == (
        CALIBRATION_AUTHORIZATION_STATUS
    )
    assert authorization["authorization"][
        "ablation_calibration_map_fit_authorized"
    ]
    assert authorization["authorization"]["iid_unsealing_authorized"] is False

    changed = deepcopy(json.loads(review_path.read_text()))
    changed["authorization_scope"]["calibration_map_count"] = 5
    _write_json(review_path, changed)
    with pytest.raises(TrainingGateError, match="review is not exact"):
        build_ablation_calibration_authorization(
            tmp_path,
            release_packet_path=packet_path,
            delegated_review_path=review_path,
            output_path=output,
        )


def test_iid_packet_requires_maps_and_allocates_exact_six_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    project = tmp_path / "project"
    (root / "results/phase7").mkdir(parents=True)
    (root / "configs/execution").mkdir(parents=True)
    calibration = root / "configs/execution/calibration.yaml"
    _write_yaml(
        calibration,
        {
            "authorization_status": CALIBRATION_AUTHORIZATION_STATUS,
            "selected_architecture": {
                "architecture_id": "nsf-t10-w256",
                "model_configuration_hash": "a" * 64,
                "locked_training_rung": 131072,
            },
            "ablation_checkpoints": _checkpoints(),
        },
    )
    primary = root / "configs/execution/final.yaml"
    _write_yaml(
        primary,
        {
            "authorization_status": "authorized_final_evaluation_inference_only",
            "selected_architecture": {
                "architecture_id": "nsf-t10-w256",
                "model_configuration_hash": "a" * 64,
                "locked_training_rung": 131072,
            },
            "sealed_publication": {"parent_root": str(project / "sealed")},
        },
    )
    wheel = project / "release.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"wheel")
    wheel_result = project / "wheel-result.json"
    _write_json(wheel_result, {})
    environment = project / "environment.txt"
    environment.write_text("environment\n")
    maps = _iid_packet()["ablation_calibration_maps"]
    primary_scores = _iid_packet()["primary_same_seed_iid_scores"]
    monkeypatch.setattr(release_module, "PROJECT_ROOT", project)
    monkeypatch.setattr(
        release_module,
        "load_ablation_evaluation_release_stack_contract",
        lambda value: {},
    )
    monkeypatch.setattr(release_module, "_verify_checkout", lambda *args: None)
    monkeypatch.setattr(
        release_module, "_completed_ablation_maps", lambda value: maps
    )
    monkeypatch.setattr(
        release_module,
        "_primary_iid_scores",
        lambda value: ("iid-test", primary_scores),
    )
    monkeypatch.setattr(
        release_module,
        "_immutable_inference",
        lambda **kwargs: {"git_commit": kwargs["implementation_commit"]},
    )
    output_root = project / "ablation-iid"
    packet = build_ablation_iid_release_packet(
        root,
        implementation_commit="d" * 40,
        ablation_calibration_authorization_path=calibration,
        primary_final_inference_authorization_path=primary,
        wheel_path=wheel,
        exact_wheel_test_result_path=wheel_result,
        environment_lock_path=environment,
        iid_output_root=output_root,
        output_path=root / "results/phase7/ablation_iid_release_packet.json",
    )
    assert packet["status"] == IID_RELEASE_STATUS
    assert packet["ablation_iid_execution_authorized"] is False
    assert sum(
        len(values) for values in packet["ablation_iid_score_outputs"].values()
    ) == 6
    assert sum(
        len(values) for values in packet["paired_comparison_outputs"].values()
    ) == 6
    assert not output_root.exists()


def test_reviewed_iid_packet_opens_only_iid_and_paired_comparison(
    tmp_path: Path,
) -> None:
    packet_path = tmp_path / "results/phase7/ablation_iid_release_packet.json"
    _write_json(packet_path, _iid_packet())
    review_path = tmp_path / "results/phase7/ablation_iid_execution_review.json"
    _write_json(
        review_path,
        {
            "status": IID_REVIEW_STATUS,
            "reviewed_release_packet_sha256": _sha256(packet_path),
            "reviewed_by": (
                "codex_as_delegated_scientific_and_engineering_reviewer"
            ),
            "review_date": "2026-07-24",
            "authorization_scope": _iid_scope(),
            "closed_boundaries": {
                name: False for name in IID_CLOSED_BOUNDARIES
            },
        },
    )
    output = (
        tmp_path
        / "configs/execution/phase7_ablation_iid_execution_authorization.yaml"
    )
    authorization = build_ablation_iid_authorization(
        tmp_path,
        release_packet_path=packet_path,
        delegated_review_path=review_path,
        output_path=output,
    )
    assert authorization["authorization_status"] == IID_AUTHORIZATION_STATUS
    flags = authorization["authorization"]
    assert flags["final_iid_unsealing_authorized"] is True
    assert flags["paired_comparison_execution_authorized"] is True
    assert flags["non_iid_ablation_inference_authorized"] is False
    assert flags["result_driven_retraining_authorized"] is False


def test_release_module_does_not_expose_an_execution_entrypoint() -> None:
    source = (
        ROOT
        / "src/gwlens_mm/training/ablation_evaluation_authorization.py"
    ).read_text(encoding="utf-8")
    assert "run_authorized_ablation" not in source
    assert set(CALIBRATION_REVIEW_SCOPE) == set(_calibration_scope())
    assert set(IID_REVIEW_SCOPE) == set(_iid_scope())
