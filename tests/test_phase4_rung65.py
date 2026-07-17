from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from gwlens_mm.training.contracts import TrainingGateError
from gwlens_mm.training.data import resolve_combined_training_publication
from gwlens_mm.training.learning_curve import compare_32k_to_65k
from gwlens_mm.training.rung65 import (
    validate_65k_training_gate,
    validate_immutable_training_artifacts,
)

GENERATOR = "2be777e727ef9d8e1a85f89c68966df5d37932b0"
RC4_HASH = "5aeaac395463bd073c44ead4ff4c5c729b5a2d4b4f1840c0825a53b30ab1bc98"
ROOT = Path(__file__).resolve().parents[1]


def _stage_a_parent(tmp_path: Path) -> Path:
    parent = tmp_path / "stage-a" / "published" / "parent-a"
    parent.mkdir(parents=True)
    validations = {}
    for split, count, shards in (("train", 32768, 256), ("validation", 6144, 48)):
        dataset_id = f"stage-a-{split}"
        child = parent / dataset_id
        child.mkdir()
        (child / "run_manifest.json").write_text(
            json.dumps(
                {
                    "split": split,
                    "dataset_id": dataset_id,
                    "generator_commit": GENERATOR,
                    "accepted_target": count,
                    "all_importance_weights_one": True,
                }
            )
        )
        validations[split] = {
            "status": "passed",
            "split": split,
            "dataset_id": dataset_id,
            "accepted_pair_count": count,
            "complete_shard_count": shards,
            "pairs_per_shard": 128,
            "generator_commit": GENERATOR,
            "proposal_equals_evaluation": True,
            "all_importance_weights_one": True,
        }
    (parent / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "generator_commit": GENERATOR,
                "preregistration_hash": RC4_HASH,
                "accepted_pair_count": 38912,
                "train_accepted_pair_count": 32768,
                "validation_accepted_pair_count": 6144,
                "validations": validations,
                "train_validation_group_disjoint": True,
                "proposal_equals_evaluation": True,
                "all_importance_weights_one": True,
                "model_training_authorized": False,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return parent


def _combined_publication(tmp_path: Path) -> tuple[Path, Path, Path]:
    stage_a = _stage_a_parent(tmp_path)
    stage_a_hash = hashlib.sha256((stage_a / "dataset_manifest.json").read_bytes()).hexdigest()
    stage_b = tmp_path / "stage-b" / "published" / "parent-b"
    stage_b_dataset = stage_b / "stage-b-train"
    stage_b_dataset.mkdir(parents=True)
    stage_b_manifest = {
        "status": "passed",
        "generator_commit": GENERATOR,
        "preregistration_hash": RC4_HASH,
        "accepted_pair_count": 32768,
        "complete_shard_count": 256,
        "proposal_equals_evaluation": True,
        "all_importance_weights_one": True,
        "validation": {
            "status": "passed",
            "split": "train",
            "dataset_id": "stage-b-train",
            "accepted_pair_count": 32768,
            "complete_shard_count": 256,
        },
    }
    (stage_b / "dataset_manifest.json").write_text(
        json.dumps(stage_b_manifest, sort_keys=True) + "\n"
    )
    stage_b_hash = hashlib.sha256(
        (stage_b / "dataset_manifest.json").read_bytes()
    ).hexdigest()
    combined = tmp_path / "train-65k" / "published" / "combined-65k"
    combined.mkdir(parents=True)
    manifest = {
        "status": "passed",
        "accepted_physical_system_count": 65536,
        "validation_physical_system_count": 6144,
        "strict_nested_train_ladder": True,
        "proposal_equals_evaluation": True,
        "all_importance_weights_one": True,
        "training_authorized": False,
        "group_validation": {
            "stage_a_stage_b_group_disjoint": True,
            "stage_b_validation_group_disjoint": True,
        },
        "components": [
            {
                "role": "stage_a_train",
                "dataset_id": "stage-a-train",
                "accepted_count": 32768,
                "parent_manifest_sha256": stage_a_hash,
            },
            {
                "role": "stage_b_train_extension",
                "dataset_id": "stage-b-train",
                "accepted_count": 32768,
                "parent_manifest_sha256": stage_b_hash,
            },
        ],
    }
    (combined / "dataset_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
    )
    return stage_a, stage_b, combined


def test_combined_65k_resolver_binds_both_atomic_parents(tmp_path: Path) -> None:
    stage_a, stage_b, combined = _combined_publication(tmp_path)
    manifest_hash = hashlib.sha256(
        (combined / "dataset_manifest.json").read_bytes()
    ).hexdigest()
    publication = resolve_combined_training_publication(
        combined,
        stage_a_parent_root=stage_a,
        stage_b_parent_root=stage_b,
        expected_generator_commit=GENERATOR,
        expected_preregistration_hash=RC4_HASH,
        expected_combined_manifest_sha256=manifest_hash,
    )
    assert publication.stage_a.train_dataset_id == "stage-a-train"
    assert publication.stage_b_dataset_id == "stage-b-train"
    assert publication.stage_b_train_root == stage_b / "stage-b-train"
    assert len(publication.train_manifest_sha256) == 64
    tampered = json.loads((combined / "dataset_manifest.json").read_text())
    tampered["training_authorized"] = True
    (combined / "dataset_manifest.json").write_text(json.dumps(tampered) + "\n")
    with pytest.raises(TrainingGateError, match="combined manifest contract"):
        resolve_combined_training_publication(
            combined,
            stage_a_parent_root=stage_a,
            stage_b_parent_root=stage_b,
            expected_generator_commit=GENERATOR,
            expected_preregistration_hash=RC4_HASH,
        )


def test_65k_gate_remains_closed_without_future_authorization(tmp_path: Path) -> None:
    with pytest.raises(TrainingGateError, match="authorization is absent"):
        validate_65k_training_gate(
            ROOT,
            authorization_path=(
                ROOT / "configs/execution/phase4_probe_training_stack_authorization.yaml"
            ),
            stage_a_publication_root=tmp_path / "missing-a",
            stage_b_publication_root=tmp_path / "missing-b",
            combined_publication_root=tmp_path / "missing-combined",
        )


def test_65k_gate_accepts_only_exact_atomic_evidence(tmp_path: Path) -> None:
    stage_a, stage_b, combined = _combined_publication(tmp_path)
    decision_path = ROOT / "results/phase4/probe/learning_curve_decision.json"
    commitment_path = ROOT / "results/phase4/final_evaluation_commitment.json"
    authorization = {
        "authorization_status": "authorized_train_65k_probe_only",
        "authorization": {
            "stage_a_data_access_authorized": True,
            "stage_b_data_access_authorized": True,
            "scientific_65k_probe_training_authorized": True,
            "probe_optimizer_execution_authorized": True,
            "learning_curve_decision_authorized": True,
            "model_tuning_authorized": False,
            "architecture_selection_authorized": False,
            "calibration_authorized": False,
            "sbc_authorized": False,
            "final_evaluation_authorized": False,
            "extension_above_65536_authorized": False,
            "gwosc_gwtc_access_authorized": False,
        },
        "authorized_training_rungs": [65536],
        "authorized_training_seeds": [0, 1, 2],
        "prior_learning_curve_decision_sha256": hashlib.sha256(
            decision_path.read_bytes()
        ).hexdigest(),
        "final_evaluation_commitment_sha256": hashlib.sha256(
            commitment_path.read_bytes()
        ).hexdigest(),
        "generator_commit": GENERATOR,
        "preregistration_hash": RC4_HASH,
        "combined_train": {
            "combined_train_id": combined.name,
            "combined_manifest_sha256": hashlib.sha256(
                (combined / "dataset_manifest.json").read_bytes()
            ).hexdigest(),
            "stage_a_parent_manifest_sha256": hashlib.sha256(
                (stage_a / "dataset_manifest.json").read_bytes()
            ).hexdigest(),
            "stage_b_parent_manifest_sha256": hashlib.sha256(
                (stage_b / "dataset_manifest.json").read_bytes()
            ).hexdigest(),
        },
    }
    path = tmp_path / "authorization.yaml"
    path.write_text(yaml.safe_dump(authorization, sort_keys=True))
    result = validate_65k_training_gate(
        ROOT,
        authorization_path=path,
        stage_a_publication_root=stage_a,
        stage_b_publication_root=stage_b,
        combined_publication_root=combined,
    )
    assert result["publication"].combined_manifest_sha256 == authorization[
        "combined_train"
    ]["combined_manifest_sha256"]
    authorization["combined_train"]["stage_b_parent_manifest_sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(authorization, sort_keys=True))
    with pytest.raises(TrainingGateError, match="Stage B manifest hash mismatch"):
        validate_65k_training_gate(
            ROOT,
            authorization_path=path,
            stage_a_publication_root=stage_a,
            stage_b_publication_root=stage_b,
            combined_publication_root=combined,
        )


def test_65k_immutable_artifacts_bind_wheel_and_environment(tmp_path: Path) -> None:
    lock = tmp_path / "environment.lock"
    wheel = tmp_path / "gwlens_mm-0.1.0-py3-none-any.whl"
    lock.write_text("environment\n")
    wheel.write_bytes(b"reviewed wheel")
    immutable = {
        "git_commit": "a" * 40,
        "environment_lock_path": str(lock),
        "environment_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "wheel_path": str(wheel),
        "wheel_filename": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "editable_install_authorized": False,
    }
    result = validate_immutable_training_artifacts(
        ROOT,
        immutable,
        training_commit="a" * 40,
        environment_lock_path=lock,
    )
    assert result["wheel_sha256"] == immutable["wheel_sha256"]
    wheel.write_bytes(b"tampered wheel")
    with pytest.raises(TrainingGateError, match="wheel hash mismatch"):
        validate_immutable_training_artifacts(
            ROOT,
            immutable,
            training_commit="a" * 40,
            environment_lock_path=lock,
        )


def _cases(path: Path, *, nlp: float, crps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = (
        "high_absolute_magnification",
        "extreme_relative_magnification",
        "second_image_near_threshold",
        "extreme_profile_or_environment",
    )
    rows = []
    for index in range(512):
        row = {
            "physical_system_id": f"validation-system-{index:04d}",
            "lens_family": "sie_external_shear" if index % 2 == 0 else "epl_external_shear",
            "em_cell_signature": "all_modalities",
            "tail_view": groups[index // 128],
            "nlp_nat_per_target_dimension": nlp,
            "crps_log_mu_primary": crps,
            "crps_log_mu_secondary": crps,
            "crps_mean": crps,
        }
        for level in (0.50, 0.80, 0.90, 0.95):
            key = f"{level:.2f}"
            covered = index < round(level * 512)
            for target in ("primary", "secondary", "joint"):
                row[f"covered_{target}_{key}"] = covered
            row[f"width_primary_{key}"] = 1.0
            row[f"width_secondary_{key}"] = 1.0
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_terminal_learning_curve_locks_or_stops_without_auto_extension(
    tmp_path: Path,
) -> None:
    smaller = tmp_path / "small"
    larger = tmp_path / "large"
    for seed in (0, 1, 2):
        _cases(
            smaller / "rung-32768" / f"seed-{seed}" / "development_cases.csv",
            nlp=1.0,
            crps=1.0,
        )
        _cases(
            larger / "rung-65536" / f"seed-{seed}" / "development_cases.csv",
            nlp=0.995,
            crps=0.995,
        )
    saturated = compare_32k_to_65k(
        smaller, larger, bootstrap_replicates=100
    )
    assert saturated["decision"] == "lock_train_65k"
    assert saturated["extension_above_65536_authorized"] is False
    for seed in (0, 1, 2):
        _cases(
            larger / "rung-65536" / f"seed-{seed}" / "development_cases.csv",
            nlp=0.90,
            crps=0.90,
        )
    improving = compare_32k_to_65k(smaller, larger, bootstrap_replicates=100)
    assert improving["decision"] == "stop_data_limited_and_new_preregistration"
    assert improving["all_saturation_conditions_passed"] is False
    assert improving["extension_above_65536_authorized"] is False
