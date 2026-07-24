from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from gwlens_mm.training import final_reporting as reporting
from gwlens_mm.training.final_inference import _expected_namespaces
from gwlens_mm.training.final_reporting import (
    AUTHORIZATION_STATUS,
    RELEASE_STATUS,
    REVIEW_STATUS,
    FinalSummaryGateError,
    build_final_summary_authorization,
    build_final_summary_release_packet,
    dry_run_plan,
    load_final_summary_stack_contract,
    load_validated_score_payload,
    run_authorized_final_summary,
    summarize_final_scores,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _small_namespaces() -> tuple[object, ...]:
    return tuple(
        replace(item, accepted_count=8, shard_count=1)
        for item in _expected_namespaces(ROOT)
    )


def _covered(count: int, level: float) -> np.ndarray:
    result = np.zeros(count, dtype=bool)
    result[: int(round(count * level))] = True
    return result


def _payload(
    specification: object,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    count = int(specification.accepted_count)
    identifiers = np.asarray(
        [f"{specification.namespace_id}-case-{index:03d}" for index in range(count)],
        dtype=np.str_,
    )
    lens_families = np.asarray(
        ["sie_external_shear", "epl_external_shear"] * (count // 2),
        dtype=np.str_,
    )
    if specification.split.value == "iid_test":
        em_cells = np.asarray(
            list(reporting.EM_CELLS),
            dtype=np.str_,
        )
    else:
        em_cells = np.asarray(
            [f"em-cell-{index % 8}" for index in range(count)],
            dtype=np.str_,
        )
    truth = np.column_stack(
        (
            np.linspace(0.1, 0.8, count),
            np.linspace(0.2, 0.9, count),
        )
    )
    payload: dict[str, np.ndarray] = {
        "truth": truth,
        "truth_log_density": np.linspace(-2.0, -1.0, count),
        "nlp_nat_per_target_dimension": -np.linspace(-2.0, -1.0, count) / 2.0,
        "crps": np.full((count, 2), (0.1, 0.2), dtype=np.float64),
        "marginal_region_scores": np.full((count, 2), 0.25),
        "joint_region_scores": np.full(count, 0.25),
        "physical_system_ids": identifiers,
        "lens_families": lens_families,
        "em_cells": em_cells,
        "splits": np.asarray([specification.split.value] * count, dtype=np.str_),
        "diagnostic_context_ids": np.asarray(
            [specification.diagnostic_context_id] * count,
            dtype=np.str_,
        ),
        "model_seed": np.asarray(seed, dtype=np.int64),
        "architecture_id": np.asarray("architecture-1", dtype=np.str_),
        "namespace_id": np.asarray(specification.namespace_id, dtype=np.str_),
        "checkpoint_sha256": np.asarray(str(seed) * 64, dtype=np.str_),
        "publication_manifest_sha256": np.asarray("a" * 64, dtype=np.str_),
        "calibration_map_sha256": np.asarray("b" * 64, dtype=np.str_),
        "inference_commit": np.asarray("c" * 40, dtype=np.str_),
    }
    for level in (0.50, 0.80, 0.90, 0.95):
        suffix = f"{int(round(level * 100)):02d}"
        covered = (
            np.ones(count, dtype=bool)
            if specification.split.value == "iid_test"
            else _covered(count, level)
        )
        payload[f"marginal_covered_{suffix}"] = np.column_stack(
            (covered, covered)
        )
        payload[f"joint_covered_{suffix}"] = covered.copy()
        payload[f"marginal_interval_width_{suffix}"] = np.full(
            (count, 2),
            (1.0, 1.5),
        )
    return payload


def test_final_summary_implementation_gate_remains_closed() -> None:
    authorization = load_final_summary_stack_contract(ROOT)
    flags = authorization["authorization"]
    assert authorization["authorization_status"].endswith("synthetic_fixture_only")
    assert flags["final_score_summary_implementation_authorized"] is True
    assert flags["final_score_artifact_access_authorized"] is False
    assert flags["final_summary_execution_authorized"] is False
    plan = dry_run_plan(ROOT)
    assert plan["status"] == "implementation_ready_final_score_access_closed"
    assert plan["score_artifact_count"] == 45
    assert plan["score_artifacts_opened"] is False


def test_score_artifact_schema_round_trip_and_rejects_posterior_draws(
    tmp_path: Path,
) -> None:
    specification = _small_namespaces()[0]
    path = tmp_path / "score.npz"
    payload = _payload(specification, seed=0)
    np.savez(path, **payload)
    loaded = load_validated_score_payload(
        path,
        specification=specification,
        model_seed=0,
    )
    assert loaded["truth"].shape == (8, 2)
    payload["posterior_draws"] = np.zeros((8, 4, 2))
    np.savez(path, **payload)
    with pytest.raises(FinalSummaryGateError, match="key set"):
        load_validated_score_payload(
            path,
            specification=specification,
            model_seed=0,
        )


def test_three_seed_summary_reports_all_groups_and_never_selects_best_seed() -> None:
    specifications = _small_namespaces()
    payloads = {
        seed: {
            item.namespace_id: _payload(item, seed=seed)
            for item in specifications
        }
        for seed in (0, 1, 2)
    }
    result = summarize_final_scores(payloads, specifications)
    assert result["status"].startswith("completed_preregistered")
    assert result["model_seeds"] == [0, 1, 2]
    assert result["score_artifact_count"] == 45
    assert result["best_seed_selected"] is False
    assert result["all_seed_iid_gate_passed"] is True
    assert result["all_seed_balanced_tail_gate_passed"] is True
    assert result["claim_action"] == "retain_preregistered_claim_domain"
    assert len(
        [
            key
            for key in result["seed_summaries"]["0"]["groups"]
            if key.startswith("balanced_tail/")
            and "/family/" not in key
        ]
    ) == 4
    assert "iid/em_cell/full_precise_spectroscopic" in result["aggregate_groups"]
    assert "waveform_mismatch_test/seobnrv4phm_truth" in result[
        "aggregate_groups"
    ]


def test_cross_seed_case_or_truth_drift_is_fatal() -> None:
    specifications = _small_namespaces()
    payloads = {
        seed: {
            item.namespace_id: _payload(item, seed=seed)
            for item in specifications
        }
        for seed in (0, 1, 2)
    }
    namespace = specifications[0].namespace_id
    payloads[2][namespace]["truth"] = payloads[2][namespace]["truth"].copy()
    payloads[2][namespace]["truth"][0, 0] += 0.5
    with pytest.raises(FinalSummaryGateError, match="differs across seeds"):
        summarize_final_scores(payloads, specifications)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _packet() -> dict[str, object]:
    return {
        "status": RELEASE_STATUS,
        "authorization_created": False,
        "score_artifact_access_authorized": False,
        "release_packet_repository_path": (
            "results/phase7/final_summary_release_packet.json"
        ),
        "future_review_path": "results/phase7/final_summary_delegated_review.json",
        "future_authorization_path": (
            "configs/execution/phase7_final_summary_authorization.yaml"
        ),
        "frozen_contracts": {
            "locked_training_rung": 131072,
            "model_seeds": [0, 1, 2],
            "final_analysis_hash": "7" * 64,
            "final_generation_commitment_sha256": "c" * 64,
            "namespace_count": 15,
            "accepted_case_count": 20480,
            "score_artifact_count": 45,
        },
        "final_inference_authorization_path": "/project/final-inference.yaml",
        "final_inference_authorization_sha256": "d" * 64,
        "score_artifacts": {"synthetic": True},
        "immutable_execution": {"git_commit": "e" * 40},
        "summary_output_path": "/project/final-summary.json",
        "review_scope": reporting._REVIEW_SCOPE,
    }


def test_delegated_review_builds_exact_summary_only_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet_path = tmp_path / "results/phase7/final_summary_release_packet.json"
    packet = _packet()
    _write_json(packet_path, packet)
    review_path = tmp_path / "results/phase7/final_summary_delegated_review.json"
    _write_json(
        review_path,
        {
            "status": REVIEW_STATUS,
            "reviewed_release_packet_sha256": _sha256(packet_path),
            "reviewed_by": (
                "codex_as_delegated_scientific_and_engineering_reviewer"
            ),
            "review_date": "2026-07-24",
            "authorization_scope": reporting._REVIEW_SCOPE,
        },
    )
    monkeypatch.setattr(
        reporting,
        "load_final_summary_stack_contract",
        lambda root: {},
    )
    output = (
        tmp_path / "configs/execution/phase7_final_summary_authorization.yaml"
    )
    authorization = build_final_summary_authorization(
        tmp_path,
        release_packet_path=packet_path,
        delegated_review_path=review_path,
        output_path=output,
    )
    assert authorization["authorization_status"] == AUTHORIZATION_STATUS
    flags = authorization["authorization"]
    assert flags["final_score_artifact_access_authorized"] is True
    assert flags["final_summary_execution_authorized"] is True
    assert flags["checkpoint_access_authorized"] is False
    assert flags["manuscript_claim_finalization_authorized"] is False


def test_release_packet_is_nonauthorizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path
    (root / "results/phase7").mkdir(parents=True)
    inference = tmp_path / "project/final-inference.yaml"
    inference.parent.mkdir(parents=True)
    inference.write_text("authorization_status: synthetic\n")
    wheel = tmp_path / "project/release.whl"
    wheel.write_bytes(b"wheel")
    wheel_result = tmp_path / "project/wheel.json"
    _write_json(wheel_result, {})
    environment = tmp_path / "project/environment.txt"
    environment.write_text("environment\n")
    summary_output = tmp_path / "project/final-summary.json"
    monkeypatch.setattr(
        reporting,
        "load_final_summary_stack_contract",
        lambda value: {},
    )
    monkeypatch.setattr(reporting, "_verify_checkout", lambda *args: None)
    monkeypatch.setattr(
        reporting,
        "_project_path",
        lambda path, **kwargs: Path(path).resolve(),
    )
    monkeypatch.setattr(
        reporting,
        "_immutable_inference",
        lambda **kwargs: {"git_commit": kwargs["implementation_commit"]},
    )
    monkeypatch.setattr(
        reporting,
        "_score_artifact_catalog",
        lambda *args: {"synthetic": True},
    )
    packet = build_final_summary_release_packet(
        root,
        implementation_commit="f" * 40,
        final_inference_authorization_path=inference,
        wheel_path=wheel,
        exact_wheel_test_result_path=wheel_result,
        environment_lock_path=environment,
        summary_output_path=summary_output,
        output_path=root / "results/phase7/final_summary_release_packet.json",
    )
    assert packet["status"] == RELEASE_STATUS
    assert packet["authorization_created"] is False
    assert packet["score_artifact_access_authorized"] is False
    assert not summary_output.exists()


def test_cli_defaults_to_execution_closed() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/phase7/run_final_summary.py"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )
    result = json.loads(completed.stdout)
    assert result["summary_executed"] is False
    assert result["score_artifacts_opened"] is False


def test_authorized_runtime_reads_only_exact_score_catalog_and_writes_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specifications = _small_namespaces()
    project = tmp_path / "project"
    wheel = project / "review/release.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"wheel")
    environment = project / "review/environment.txt"
    environment.write_text("environment\n")
    catalog: dict[str, object] = {}
    for seed in (0, 1, 2):
        seed_catalog: dict[str, object] = {}
        for index, specification in enumerate(specifications):
            path = project / f"scores/seed-{seed}/namespace-{index:02d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **_payload(specification, seed=seed))
            summary_path = path.with_suffix(".summary.json")
            _write_json(summary_path, {"status": "synthetic-score-summary"})
            seed_catalog[specification.namespace_id] = {
                "path": str(path),
                "sha256": _sha256(path),
                "summary_path": str(summary_path),
                "summary_sha256": _sha256(summary_path),
            }
        catalog[str(seed)] = seed_catalog
    output = project / "results/final-summary.json"
    authorization_path = project / "review/final-summary-authorization.yaml"
    authorization_path.write_text(
        yaml.safe_dump(
            {
                "authorization_status": AUTHORIZATION_STATUS,
                "frozen_contracts": {"score_artifact_count": 45},
                "immutable_execution": {
                    "git_commit": "1" * 40,
                    "wheel_path": str(wheel),
                    "wheel_sha256": _sha256(wheel),
                    "environment_lock_path": str(environment),
                    "environment_lock_sha256": _sha256(environment),
                    "editable_install_authorized": False,
                },
                "summary_output_path": str(output),
                "authorization": {
                    key: value
                    for key, value in reporting._REVIEW_SCOPE.items()
                    if key.endswith("_authorized")
                },
                "score_artifacts": catalog,
                "stop_after_final_score_summary": True,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(reporting, "_verify_checkout", lambda *args: None)
    monkeypatch.setattr(
        reporting,
        "_project_path",
        lambda path, **kwargs: Path(path).resolve(),
    )
    monkeypatch.setattr(
        reporting,
        "_expected_namespaces",
        lambda root: specifications,
    )
    result = run_authorized_final_summary(
        tmp_path,
        authorization_path=authorization_path,
        output_path=output,
        execution_commit="1" * 40,
    )
    assert result["status"].startswith("completed_preregistered")
    assert result["score_artifact_count"] == 45
    assert result["final_records_opened"] is False
    assert result["checkpoints_opened"] is False
    assert output.is_file()

    one_artifact = Path(
        catalog["0"][specifications[0].namespace_id]["path"]  # type: ignore[index]
    )
    one_artifact.write_bytes(one_artifact.read_bytes() + b"tamper")
    output.unlink()
    with pytest.raises(FinalSummaryGateError, match="hash changed"):
        run_authorized_final_summary(
            tmp_path,
            authorization_path=authorization_path,
            output_path=output,
            execution_commit="1" * 40,
        )
