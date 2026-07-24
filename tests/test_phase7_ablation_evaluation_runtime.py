from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from gwlens_mm.training import ablation_evaluation_runtime as runtime
from gwlens_mm.training.ablation_evaluation_runtime import (
    AblationEvaluationRuntimeError,
    dry_run_ablation_evaluation_runtime,
    load_ablation_evaluation_runtime_contract,
    validate_ablation_calibration_execution_gate,
    validate_ablation_iid_execution_gate,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _checkpoint_item(project: Path, view: str, seed: int) -> dict:
    return {
        "checkpoint_path": str(project / f"{view}-seed-{seed}.ckpt"),
        "checkpoint_sha256": f"{seed + 1}" * 64,
        "model_configuration_hash": "a" * 64,
    }


def test_runtime_contract_is_unique_seeded_and_execution_closed() -> None:
    contract = load_ablation_evaluation_runtime_contract(ROOT)
    assert contract["locked_training_rung"] == 131072
    seeds = []
    for section in ("calibration", "iid"):
        values = contract[section]["root_seed_by_view_and_model_seed"]
        seeds.extend(
            values[view][str(seed)]
            for view in ("gw_only", "em_only")
            for seed in (0, 1, 2)
        )
    assert len(seeds) == len(set(seeds)) == 12
    assert all(value is False for value in contract["execution"].values())
    plan = dry_run_ablation_evaluation_runtime(ROOT)
    assert plan["calibration_job_count"] == 6
    assert plan["iid_job_count"] == 6
    assert plan["scientific_checkpoint_accessed"] is False
    assert plan["iid_data_unsealed"] is False


def test_calibration_gate_binds_one_fresh_view_seed_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    publication = project / "calibration"
    manifest = publication / "dataset_manifest.json"
    _write_json(manifest, {"status": "passed"})
    checkpoint = project / "gw_only-seed-0.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    environment = project / "environment.txt"
    environment.write_text("environment\n")
    primary_score = project / "primary/seed-0/calibration.npz"
    primary_score.parent.mkdir(parents=True)
    np.savez(
        primary_score,
        physical_system_ids=np.asarray(
            [f"cal-{index:04d}" for index in range(4096)], dtype=np.str_
        ),
    )
    score = project / "outputs/gw_only/seed-0/calibration_scores.npz"
    calibration_map = (
        project / "outputs/gw_only/seed-0/calibration_region_maps.json"
    )
    authorization_path = project / "authorization.yaml"
    item = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "model_configuration_hash": "a" * 64,
    }
    authorization = {
        "authorization_status": (
            "authorized_ablation_calibration_score_and_map_execution_only"
        ),
        "authorization": {
            "scientific_checkpoint_access_authorized": True,
            "calibration_fit_data_access_authorized": True,
            "ablation_calibration_score_execution_authorized": True,
            "ablation_calibration_map_fit_authorized": True,
            "iid_unsealing_authorized": False,
            "iid_inference_or_comparison_authorized": False,
            "model_training_or_tuning_authorized": False,
            "sbc_authorized": False,
            "gwosc_gwtc_access_authorized": False,
        },
        "selected_architecture": {
            "architecture_id": "nsf-t10-w256",
            "locked_training_rung": 131072,
        },
        "ablation_checkpoints": {"gw_only": {"0": item}},
        "calibration_publication": {
            "parent_root": str(publication),
            "parent_manifest_sha256": _sha256(manifest),
            "calibration_fit_accepted_count": 4096,
            "calibration_dataset_id": "calibration-fit",
        },
        "calibration_score_outputs": {"gw_only": {"0": str(score)}},
        "calibration_map_outputs": {
            "gw_only": {"0": str(calibration_map)}
        },
        "primary_same_seed_calibration_scores": {
            "0": {
                "path": str(primary_score),
                "sha256": _sha256(primary_score),
            }
        },
    }
    _write_yaml(authorization_path, authorization)
    monkeypatch.setattr(runtime, "APPROVED_REMOTE_ROOT", project)
    monkeypatch.setattr(
        runtime,
        "load_ablation_evaluation_runtime_contract",
        lambda value: load_ablation_evaluation_runtime_contract(ROOT),
    )
    monkeypatch.setattr(
        runtime,
        "_validate_immutable_runtime",
        lambda *args, **kwargs: "b" * 40,
    )
    gate = validate_ablation_calibration_execution_gate(
        ROOT,
        authorization_path=authorization_path,
        publication_root=publication,
        checkpoint_path=checkpoint,
        environment_lock_path=environment,
        score_output_path=score,
        map_output_path=calibration_map,
        view="gw_only",
        seed=0,
    )
    assert gate["architecture_id"] == "nsf-t10-w256"
    assert gate["inference_commit"] == "b" * 40

    changed = deepcopy(authorization)
    changed["authorization"]["iid_unsealing_authorized"] = True
    _write_yaml(authorization_path, changed)
    with pytest.raises(
        AblationEvaluationRuntimeError, match="crossed its boundary"
    ):
        validate_ablation_calibration_execution_gate(
            ROOT,
            authorization_path=authorization_path,
            publication_root=publication,
            checkpoint_path=checkpoint,
            environment_lock_path=environment,
            score_output_path=score,
            map_output_path=calibration_map,
            view="gw_only",
            seed=0,
        )


def test_iid_gate_requires_matching_map_primary_score_and_fresh_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    publication = project / "sealed"
    publication.mkdir(parents=True)
    checkpoint = project / "em_only-seed-2.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    calibration_map = project / "maps/em_only-seed-2.json"
    _write_json(calibration_map, {"status": "map"})
    primary_score = project / "primary/seed-2/iid.npz"
    primary_score.parent.mkdir(parents=True)
    primary_score.write_bytes(b"primary")
    environment = project / "environment.txt"
    environment.write_text("environment\n")
    output = project / "outputs/em_only/seed-2/iid_scores.npz"
    comparison = project / "outputs/em_only/seed-2/paired_comparison.json"
    authorization_path = project / "authorization.yaml"
    item = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "model_configuration_hash": "a" * 64,
    }
    authorization = {
        "authorization_status": (
            "authorized_ablation_iid_inference_and_paired_comparison_only"
        ),
        "authorization": {
            "final_iid_unsealing_authorized": True,
            "ablation_checkpoint_access_authorized": True,
            "matching_ablation_calibration_map_access_authorized": True,
            "primary_same_seed_iid_score_access_authorized": True,
            "ablation_iid_inference_authorized": True,
            "paired_comparison_execution_authorized": True,
            "calibration_refit_authorized": False,
            "sbc_authorized": False,
            "model_training_or_tuning_authorized": False,
            "non_iid_ablation_inference_authorized": False,
            "result_driven_retraining_authorized": False,
            "gwosc_gwtc_access_authorized": False,
        },
        "selected_architecture": {
            "architecture_id": "nsf-t10-w256",
            "locked_training_rung": 131072,
        },
        "iid_namespace_id": "iid-test",
        "ablation_checkpoints": {"em_only": {"2": item}},
        "ablation_calibration_maps": {
            "em_only": {
                "2": {
                    "calibration_map_path": str(calibration_map),
                    "calibration_map_sha256": _sha256(calibration_map),
                }
            }
        },
        "primary_same_seed_iid_scores": {
            "2": {
                "path": str(primary_score),
                "sha256": _sha256(primary_score),
            }
        },
        "sealed_publication": {"parent_root": str(publication)},
        "ablation_iid_score_outputs": {"em_only": {"2": str(output)}},
        "paired_comparison_outputs": {
            "em_only": {"2": str(comparison)}
        },
    }
    _write_yaml(authorization_path, authorization)
    monkeypatch.setattr(runtime, "APPROVED_REMOTE_ROOT", project)
    monkeypatch.setattr(
        runtime,
        "load_ablation_evaluation_runtime_contract",
        lambda value: load_ablation_evaluation_runtime_contract(ROOT),
    )
    monkeypatch.setattr(
        runtime,
        "_validate_immutable_runtime",
        lambda *args, **kwargs: "b" * 40,
    )
    gate = validate_ablation_iid_execution_gate(
        ROOT,
        authorization_path=authorization_path,
        publication_root=publication,
        environment_lock_path=environment,
        score_output_path=output,
        comparison_output_path=comparison,
        view="em_only",
        seed=2,
    )
    assert gate["namespace_id"] == "iid-test"
    assert gate["primary_path"] == primary_score.resolve()

    changed = deepcopy(authorization)
    changed["authorization"]["non_iid_ablation_inference_authorized"] = True
    _write_yaml(authorization_path, changed)
    with pytest.raises(
        AblationEvaluationRuntimeError, match="crossed its boundary"
    ):
        validate_ablation_iid_execution_gate(
            ROOT,
            authorization_path=authorization_path,
            publication_root=publication,
            environment_lock_path=environment,
            score_output_path=output,
            comparison_output_path=comparison,
            view="em_only",
            seed=2,
        )


def test_calibration_runtime_writes_exact_score_map_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score_path = tmp_path / "calibration_scores.npz"
    map_path = tmp_path / "calibration_region_maps.json"
    identifiers = np.asarray(
        [f"cal-{index:04d}" for index in range(4096)], dtype=np.str_
    )
    payload = {
        "physical_system_ids": identifiers,
        "em_cells": np.asarray(
            [f"cell-{index % 8}" for index in range(4096)], dtype=np.str_
        ),
        "marginal_scores": np.zeros((4096, 2), dtype=np.float64),
        "joint_scores": np.zeros(4096, dtype=np.float64),
    }
    gate = {
        "authorization": {
            "selected_architecture": {"locked_training_rung": 131072}
        },
        "item": {
            "checkpoint_sha256": "a" * 64,
            "model_configuration_hash": "b" * 64,
        },
        "architecture_id": "nsf-t10-w256",
        "inference_commit": "c" * 40,
        "publication": {
            "calibration_dataset_id": "calibration-fit",
            "parent_manifest_sha256": "d" * 64,
        },
        "primary_path": tmp_path / "primary_calibration.npz",
        "contract": {
            "calibration": {
                "root_seed_by_view_and_model_seed": {
                    "gw_only": {"0": 2026072410}
                },
                "physical_batch_size": 16,
                "posterior_draw_chunk_size": 256,
            }
        },
    }
    monkeypatch.setattr(
        runtime,
        "validate_ablation_calibration_execution_gate",
        lambda *args, **kwargs: gate,
    )
    monkeypatch.setattr(
        runtime,
        "_load_ablation_checkpoint",
        lambda *args, **kwargs: (object(), object(), object(), {}),
    )
    monkeypatch.setattr(runtime, "_validate_runtime_versions", lambda value: None)
    monkeypatch.setattr(runtime, "load_input_policy", lambda value: None)
    monkeypatch.setattr(runtime, "_verified_curves", lambda *args: {})
    monkeypatch.setattr(
        runtime, "PublishedStageADataset", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        runtime, "AblatedCalibrationDataset", lambda *args: object()
    )
    monkeypatch.setattr(runtime, "_data_loader", lambda *args, **kwargs: object())
    monkeypatch.setattr(runtime, "_score_batches", lambda *args, **kwargs: payload)
    monkeypatch.setattr(
        runtime,
        "_load_npz",
        lambda value: {"physical_system_ids": identifiers},
    )
    monkeypatch.setattr(
        runtime,
        "fit_ablation_calibration_map",
        lambda *args, **kwargs: {
            "status": "fitted_split_conformal_region_level_maps",
            "ablation_identity": {
                "view": "gw_only",
                "model_seed": 0,
                "checkpoint_sha256": "a" * 64,
            },
        },
    )
    result = runtime.run_authorized_ablation_calibration(
        ROOT,
        authorization_path=tmp_path / "authorization.yaml",
        publication_root=tmp_path / "publication",
        checkpoint_path=tmp_path / "checkpoint.ckpt",
        environment_lock_path=tmp_path / "environment.txt",
        psd_root=tmp_path / "psd",
        score_output_path=score_path,
        map_output_path=map_path,
        view="gw_only",
        seed=0,
        device_name="cpu",
    )
    assert result["status"] == (
        "completed_ablation_calibration_score_and_map"
    )
    assert result["iid_accessed"] is False
    assert score_path.is_file()
    assert map_path.is_file()
    assert map_path.with_name("run_summary.json").is_file()


def test_iid_runtime_writes_score_comparison_and_descriptive_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score_path = tmp_path / "iid_scores.npz"
    comparison_path = tmp_path / "paired_comparison.json"
    map_path = tmp_path / "calibration_region_maps.json"
    _write_json(
        map_path, {"status": "fitted_split_conformal_region_level_maps"}
    )
    primary_path = tmp_path / "primary_iid.npz"
    np.savez(primary_path, placeholder=np.asarray(1))
    identifiers = np.asarray(
        [f"iid-{index:05d}" for index in range(8192)], dtype=np.str_
    )
    payload = {
        "physical_system_ids": identifiers,
        "splits": np.asarray(["iid_test"] * 8192, dtype=np.str_),
    }
    gate = {
        "authorization": {
            "selected_architecture": {"locked_training_rung": 131072},
            "sealed_publication": {
                "manifest_sha256": "d" * 64,
                "generator_commit": "e" * 40,
            },
        },
        "item": {
            "checkpoint_sha256": "a" * 64,
            "model_configuration_hash": "b" * 64,
        },
        "map_item": {"calibration_map_sha256": _sha256(map_path)},
        "map_path": map_path,
        "primary_path": primary_path,
        "architecture_id": "nsf-t10-w256",
        "inference_commit": "c" * 40,
        "namespace_id": "iid-test",
        "contract": {
            "iid": {
                "root_seed_by_view_and_model_seed": {
                    "em_only": {"2": 2026072425}
                },
                "physical_batch_size": 16,
                "posterior_draw_chunk_size": 256,
            }
        },
    }
    namespace = SimpleNamespace(
        specification=SimpleNamespace(
            split=runtime.SplitName.IID_TEST,
            accepted_count=8192,
        )
    )
    publication = SimpleNamespace(
        namespaces={"iid-test": namespace},
        manifest_sha256="d" * 64,
        generator_commit="e" * 40,
    )
    monkeypatch.setattr(
        runtime,
        "validate_ablation_iid_execution_gate",
        lambda *args, **kwargs: gate,
    )
    monkeypatch.setattr(
        runtime,
        "_load_ablation_checkpoint",
        lambda *args, **kwargs: (object(), object(), object(), {}),
    )
    monkeypatch.setattr(runtime, "_validate_runtime_versions", lambda value: None)
    monkeypatch.setattr(runtime, "load_input_policy", lambda value: None)
    monkeypatch.setattr(runtime, "_verified_curves", lambda *args: {})
    monkeypatch.setattr(
        runtime,
        "resolve_sealed_final_publication",
        lambda *args: publication,
    )
    monkeypatch.setattr(
        runtime, "SealedFinalNamespaceDataset", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        runtime,
        "StandardizedFinalNamespaceDataset",
        lambda *args: object(),
    )
    monkeypatch.setattr(runtime, "AblatedIIDDataset", lambda *args: object())
    monkeypatch.setattr(
        runtime, "validate_matching_ablation_map", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runtime, "_final_data_loader", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        runtime, "_score_final_batches", lambda *args, **kwargs: payload
    )
    monkeypatch.setattr(runtime, "_load_npz", lambda value: {"primary": True})
    monkeypatch.setattr(
        runtime,
        "summarize_ablation_iid_scores",
        lambda *args, **kwargs: {
            "status": "completed_descriptive_ablation_iid_summary"
        },
    )
    monkeypatch.setattr(
        runtime,
        "paired_ablation_iid_comparison",
        lambda *args, **kwargs: {
            "status": "completed_descriptive_paired_iid_ablation_comparison"
        },
    )
    result = runtime.run_authorized_ablation_iid(
        ROOT,
        authorization_path=tmp_path / "authorization.yaml",
        publication_root=tmp_path / "publication",
        environment_lock_path=tmp_path / "environment.txt",
        psd_root=tmp_path / "psd",
        score_output_path=score_path,
        comparison_output_path=comparison_path,
        view="em_only",
        seed=2,
        device_name="cpu",
    )
    assert result["status"] == (
        "completed_ablation_iid_inference_and_paired_comparison"
    )
    assert result["best_seed_selected"] is False
    assert result["non_iid_ablation_executed"] is False
    assert score_path.is_file()
    assert comparison_path.is_file()
    assert score_path.with_suffix(".summary.json").is_file()


@pytest.mark.parametrize(
    "script",
    (
        "scripts/phase7/run_ablation_calibration.py",
        "scripts/phase7/run_ablation_iid.py",
    ),
)
def test_cli_defaults_to_dry_run(script: str, tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / script),
        "--authorization",
        str(tmp_path / "absent-authorization.yaml"),
        "--publication-root",
        str(tmp_path / "absent-publication"),
        "--environment-lock",
        str(tmp_path / "absent-environment"),
        "--psd-root",
        str(tmp_path / "absent-psd"),
        "--score-output",
        str(tmp_path / "absent-score.npz"),
        "--view",
        "gw_only",
        "--seed",
        "0",
    ]
    if script.endswith("calibration.py"):
        command.extend(
            [
                "--checkpoint",
                str(tmp_path / "absent.ckpt"),
                "--map-output",
                str(tmp_path / "absent-map.json"),
            ]
        )
    else:
        command.extend(
            [
                "--comparison-output",
                str(tmp_path / "absent-comparison.json"),
            ]
        )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "implementation_ready_execution_closed"
    assert result["scientific_checkpoint_accessed"] is False
    assert result["iid_data_unsealed"] is False


def test_runtime_reuses_primary_score_kernels() -> None:
    source = (
        ROOT / "src/gwlens_mm/training/ablation_evaluation_runtime.py"
    ).read_text(encoding="utf-8")
    assert "from .calibration_inference import" in source
    assert "_score_batches" in source
    assert "_score_final_batches" in source
    assert "posterior_draws_persisted" not in source
