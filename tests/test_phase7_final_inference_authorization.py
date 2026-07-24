from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gwlens_mm.config import load_yaml
from gwlens_mm.training import final_inference_authorization as release_module
from gwlens_mm.training.contracts import TrainingGateError
from gwlens_mm.training.final_inference_authorization import (
    AUTHORIZATION_STATUS,
    RELEASE_STATUS,
    REVIEW_STATUS,
    SCORE_ARTIFACT_COUNT,
    build_final_inference_authorization,
    build_final_inference_release_packet,
    load_final_inference_release_stack_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _review_scope() -> dict:
    return {
        "locked_training_rung": 131072,
        "selected_architecture_id": "nsf-t10-w256",
        "model_seeds": [0, 1, 2],
        "namespace_count": 15,
        "accepted_case_count": 20480,
        "score_artifact_count": 45,
        "final_evaluation_unsealing_authorized": True,
        "final_evaluation_data_access_authorized": True,
        "selected_checkpoint_inference_authorized": True,
        "same_seed_calibration_map_application_authorized": True,
        "immutable_score_artifact_creation_authorized": True,
    }


def _closed_boundaries() -> dict:
    return {
        "model_training_or_tuning_authorized": False,
        "calibration_refit_authorized": False,
        "architecture_or_size_selection_authorized": False,
        "final_result_threshold_change_authorized": False,
        "reference_baseline_execution_authorized": False,
        "ablation_training_authorized": False,
        "gwosc_gwtc_access_authorized": False,
    }


def _packet() -> dict:
    outputs = {
        str(seed): {
            f"namespace-{index:02d}": (
                f"/root/autodl-tmp/lensing-4/final/seed-{seed}/"
                f"namespace-{index:02d}.npz"
            )
            for index in range(15)
        }
        for seed in (0, 1, 2)
    }
    return {
        "status": RELEASE_STATUS,
        "authorization_created": False,
        "final_inference_authorized": False,
        "implementation_commit": "a" * 40,
        "selected_architecture": {
            "decision_path": "/root/autodl-tmp/lensing-4/architecture.json",
            "decision_sha256": "1" * 64,
            "architecture_id": "nsf-t10-w256",
            "model_configuration_hash": "2" * 64,
            "locked_training_rung": 131072,
        },
        "selected_seed_checkpoints": {
            str(seed): {
                "path": f"/root/autodl-tmp/lensing-4/checkpoints/{seed}.ckpt",
                "sha256": str(seed + 3) * 64,
            }
            for seed in (0, 1, 2)
        },
        "same_seed_calibration_sbc_statistics": {
            str(seed): {"run_summary_path": f"/stats/{seed}/run_summary.json"}
            for seed in (0, 1, 2)
        },
        "sealed_publication": {
            "parent_root": "/root/autodl-tmp/lensing-4/final/sealed",
            "manifest_sha256": "6" * 64,
            "generator_commit": "b" * 40,
        },
        "immutable_inference": {
            "git_commit": "a" * 40,
            "wheel_sha256": "7" * 64,
            "environment_lock_sha256": "8" * 64,
        },
        "score_outputs": outputs,
        "inference_contract": {
            "posterior_draws_per_case": 4096,
            "maximum_draw_microbatch": 512,
            "physical_batch_size": 16,
            "draw_microbatch": 256,
            "model_seeds": [0, 1, 2],
            "namespace_count": 15,
            "score_artifact_count": 45,
            "posterior_draws_persisted": False,
        },
        "frozen_contracts": {
            "final_analysis_hash": (
                "7e0e252f0a972e0b0ad2fe8f93f74f1f0172639a6fb258fc7a953be5fb7973e1"
            ),
            "reference_baseline_hash": (
                "1df98c89fc418eddfd9ec766cb04311e0f3d9f40836a0d9ba1dd691d6bc1724e"
            ),
            "final_generation_commitment_sha256": (
                "c13412eced163bac26abc4b22d054f3a6fa967e7e5a4dd7849ebf54f42df6083"
            ),
        },
        "review_scope": _review_scope(),
        "closed_boundaries": _closed_boundaries(),
        "future_authorization_path": (
            "configs/execution/phase7_final_inference_authorization.yaml"
        ),
        "future_review_path": "results/phase7/final_inference_review.json",
        "release_packet_repository_path": (
            "results/phase7/final_inference_release_packet.json"
        ),
    }


def test_release_stack_is_implementation_only() -> None:
    authorization = load_final_inference_release_stack_contract(ROOT)
    flags = authorization["authorization"]
    assert flags["nonauthorizing_release_packet_implementation_authorized"] is True
    assert flags["final_evaluation_data_access_authorized"] is False
    assert flags["checkpoint_access_authorized"] is False
    assert flags["scientific_inference_execution_authorized"] is False
    assert authorization["frozen_contracts"][
        "model_seed_namespace_score_artifact_count"
    ] == SCORE_ARTIFACT_COUNT


def test_release_packet_binds_forty_five_fresh_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    project = tmp_path / "project"
    (root / "results/phase7").mkdir(parents=True)
    (root / "configs/execution").mkdir(parents=True)
    publication = project / "final" / "published" / "parent"
    publication.mkdir(parents=True)
    result_path = project / "results" / "final_materialization.json"
    result_path.parent.mkdir(parents=True)
    decision_path = project / "results" / "architecture.json"
    decision_path.write_text('{"selected_architecture_id":"nsf-t10-w256"}\n')
    result_path.write_text(
        json.dumps(
            {
                "status": "passed_sealed",
                "publication_path": str(publication),
                "accepted_pair_count": 20480,
                "complete_shard_count": 160,
                "namespace_count": 15,
                "unsealing_authorized": False,
                "gwosc_gwtc_accessed": False,
                "publication_tree_sha256": "9" * 64,
            }
        )
    )
    architecture_authorization_path = (
        root / "configs/execution/architecture.yaml"
    )
    _write_yaml(
        architecture_authorization_path,
        {
            "authorization_status": (
                "authorized_terminal_131k_architecture_selection_only"
            ),
            "architecture_selection_output_path": str(decision_path),
        },
    )
    final_authorization_path = root / "configs/execution/final_materialization.yaml"
    _write_yaml(
        final_authorization_path,
        {
            "authorization_status": (
                "authorized_sealed_final_evaluation_materialization_only"
            ),
            "implementation_commit": "b" * 40,
            "architecture_decision": {"sha256": _sha256(decision_path)},
        },
    )
    statistics_authorization_path = root / "configs/execution/statistics.yaml"
    _write_yaml(statistics_authorization_path, {"authorization_status": "fixture"})
    wheel = project / "review" / "final.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"wheel")
    wheel_test = project / "review" / "wheel-test.json"
    wheel_test.write_text("{}\n")
    environment = project / "review" / "environment.txt"
    environment.write_text("environment\n")

    monkeypatch.setattr(release_module, "PROJECT_ROOT", project)
    monkeypatch.setattr(
        release_module,
        "load_final_inference_release_stack_contract",
        lambda value: {},
    )
    monkeypatch.setattr(release_module, "_verify_checkout", lambda *args: None)
    monkeypatch.setattr(
        release_module,
        "_selected_checkpoint_artifacts",
        lambda *args, **kwargs: {
            "architecture_id": "nsf-t10-w256",
            "model_configuration_hash": "2" * 64,
            "selected_seed_checkpoints": {
                str(seed): {"path": f"/checkpoint/{seed}", "sha256": "3" * 64}
                for seed in (0, 1, 2)
            },
            "shared_identity": {"training_rung_count": 131072},
        },
    )
    monkeypatch.setattr(
        release_module,
        "resolve_sealed_final_publication",
        lambda *args, **kwargs: SimpleNamespace(
            manifest_sha256="4" * 64,
            generator_commit="b" * 40,
        ),
    )
    monkeypatch.setattr(
        release_module,
        "_statistics_artifacts",
        lambda *args, **kwargs: {
            str(seed): {"run_summary_path": f"/stats/{seed}.json"}
            for seed in (0, 1, 2)
        },
    )
    monkeypatch.setattr(
        release_module,
        "_immutable_inference",
        lambda **kwargs: {
            "git_commit": kwargs["implementation_commit"],
            "wheel_sha256": "5" * 64,
        },
    )
    monkeypatch.setattr(
        release_module,
        "_expected_namespaces",
        lambda value: tuple(
            SimpleNamespace(namespace_id=f"namespace-{index:02d}")
            for index in range(15)
        ),
    )
    score_root = project / "final" / "scores"
    packet = build_final_inference_release_packet(
        root,
        implementation_commit="a" * 40,
        architecture_authorization_path=architecture_authorization_path,
        architecture_decision_path=decision_path,
        final_materialization_authorization_path=final_authorization_path,
        final_materialization_result_path=result_path,
        publication_root=publication,
        statistics_authorization_path=statistics_authorization_path,
        wheel_path=wheel,
        exact_wheel_test_result_path=wheel_test,
        environment_lock_path=environment,
        score_output_root=score_root,
        output_path=root
        / "results/phase7/final_inference_release_packet.json",
    )
    assert packet["status"] == RELEASE_STATUS
    assert packet["authorization_created"] is False
    assert packet["final_inference_authorized"] is False
    assert sum(len(value) for value in packet["score_outputs"].values()) == 45
    assert len(
        {
            output
            for seed_outputs in packet["score_outputs"].values()
            for output in seed_outputs.values()
        }
    ) == 45
    assert not score_root.exists()


def test_reviewed_packet_builds_only_final_inference_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        release_module,
        "load_final_inference_release_stack_contract",
        lambda value: {},
    )
    root = tmp_path
    packet_path = root / "results/phase7/final_inference_release_packet.json"
    packet_path.parent.mkdir(parents=True)
    packet = _packet()
    packet_path.write_text(json.dumps(packet, sort_keys=True) + "\n")
    review_path = root / "results/phase7/final_inference_review.json"
    review_path.write_text(
        json.dumps(
            {
                "status": REVIEW_STATUS,
                "reviewed_release_packet_sha256": _sha256(packet_path),
                "reviewed_by": (
                    "codex_as_delegated_scientific_and_engineering_reviewer"
                ),
                "review_date": "2026-07-24",
                "authorization_scope": _review_scope(),
                "closed_boundaries": _closed_boundaries(),
            },
            sort_keys=True,
        )
        + "\n"
    )
    output = (
        root / "configs/execution/phase7_final_inference_authorization.yaml"
    )
    authorization = build_final_inference_authorization(
        root,
        release_packet_path=packet_path,
        delegated_review_path=review_path,
        output_path=output,
    )
    assert authorization["authorization_status"] == AUTHORIZATION_STATUS
    flags = authorization["authorization"]
    assert flags["final_evaluation_data_access_authorized"] is True
    assert flags["same_seed_calibration_map_application_authorized"] is True
    assert flags["model_training_or_tuning_authorized"] is False
    assert flags["reference_baseline_execution_authorized"] is False
    assert len(authorization["score_outputs"]) == 3
    assert sum(len(value) for value in authorization["score_outputs"].values()) == 45

    changed = deepcopy(json.loads(review_path.read_text()))
    changed["authorization_scope"]["score_artifact_count"] = 44
    review_path.write_text(json.dumps(changed))
    with pytest.raises(TrainingGateError, match="review is not exact"):
        build_final_inference_authorization(
            root,
            release_packet_path=packet_path,
            delegated_review_path=review_path,
            output_path=output,
        )


def test_runtime_authorization_status_matches_existing_validator() -> None:
    source = (ROOT / "src/gwlens_mm/training/final_inference.py").read_text()
    assert f'"{AUTHORIZATION_STATUS}"' in source
    assert (
        "configs/execution/phase7_final_inference_authorization.yaml"
        in _packet()["future_authorization_path"]
    )
    stack = load_yaml(
        ROOT
        / "configs/execution/phase7_final_inference_release_stack_authorization.yaml"
    )
    assert stack["stop_after_implementation"] is True
