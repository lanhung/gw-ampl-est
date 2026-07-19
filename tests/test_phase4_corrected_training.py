from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gwlens_mm.production.storage import tree_checksum
from gwlens_mm.training.contracts import TrainingGateError
from gwlens_mm.training.data import (
    corrected_65k_training_dataset,
    corrected_stage_a_training_dataset,
    resolve_corrected_training_publication,
)
from gwlens_mm.training.whitening import ASDCurve

BASE_GENERATOR = "2be777e727ef9d8e1a85f89c68966df5d37932b0"
CORRECTION_GENERATOR = "499f86b3159af82612e38c134cd81003eedcc4e4"
RC4_HASH = "5aeaac395463bd073c44ead4ff4c5c729b5a2d4b4f1840c0825a53b30ab1bc98"
CORRECTION_HASH = "7fca209de9f06e98da1c5a96ae0f4fc6daec5d2f0c2339a718e1f899bb915b69"


def _shards(root: Path, prefix: str, count: int, pairs: int = 128) -> None:
    for shard_index in range(count // pairs):
        shard = root / "shards" / f"shard-{shard_index:05d}"
        shard.mkdir(parents=True)
        identifiers = [
            f"{prefix}-{shard_index * pairs + row:05d}" for row in range(pairs)
        ]
        (shard / "COMPLETE.json").write_text("{}\n")
        (shard / "shard_manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "accepted_pair_count": pairs,
                    "physical_system_ids": identifiers,
                }
            )
            + "\n"
        )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, str]:
    stage_a = tmp_path / "stage-a" / "published" / "parent-a"
    stage_a.mkdir(parents=True)
    validations = {}
    for split, dataset_id, prefix, count in (
        ("train", "stage-a-train", "a", 32768),
        ("validation", "stage-a-validation", "v", 6144),
    ):
        child = stage_a / dataset_id
        child.mkdir()
        _shards(child, prefix, count)
        (child / "run_manifest.json").write_text(
            json.dumps(
                {
                    "split": split,
                    "dataset_id": dataset_id,
                    "generator_commit": BASE_GENERATOR,
                    "accepted_target": count,
                    "all_importance_weights_one": True,
                }
            )
            + "\n"
        )
        validations[split] = {
            "status": "passed",
            "split": split,
            "dataset_id": dataset_id,
            "accepted_pair_count": count,
            "complete_shard_count": count // 128,
            "pairs_per_shard": 128,
            "generator_commit": BASE_GENERATOR,
            "proposal_equals_evaluation": True,
            "all_importance_weights_one": True,
        }
    (stage_a / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "generator_commit": BASE_GENERATOR,
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
    stage_a_hash = hashlib.sha256((stage_a / "dataset_manifest.json").read_bytes()).hexdigest()

    stage_b = tmp_path / "stage-b" / "published" / "parent-b"
    stage_b_train = stage_b / "stage-b-train"
    stage_b_train.mkdir(parents=True)
    _shards(stage_b_train, "b", 32768)
    stage_b_manifest = {
        "status": "passed",
        "parent_run_id": "parent-b",
        "generator_commit": BASE_GENERATOR,
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
    stage_b_hash = hashlib.sha256((stage_b / "dataset_manifest.json").read_bytes()).hexdigest()

    combined = tmp_path / "combined" / "published" / "combined-65k"
    combined.mkdir(parents=True)
    (combined / "dataset_manifest.json").write_text(
        json.dumps(
            {
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
                        "parent_run_id": "parent-b",
                        "dataset_id": "stage-b-train",
                        "accepted_count": 32768,
                        "parent_manifest_sha256": stage_b_hash,
                    },
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    combined_hash = hashlib.sha256((combined / "dataset_manifest.json").read_bytes()).hexdigest()

    correction = tmp_path / "correction" / "published" / "correction-parent"
    correction.mkdir(parents=True)
    correction_validations = {}
    replacement_specs = (
        ("stage_a_train", "replacement-a", "ra", 2),
        ("stage_b_train", "replacement-b", "rb", 3),
    )
    for component, dataset_id, prefix, count in replacement_specs:
        child = correction / dataset_id
        child.mkdir()
        _shards(child, prefix, count, pairs=count)
        correction_validations[component] = {
            "status": "passed",
            "dataset_id": dataset_id,
            "split": "train",
            "accepted_pair_count": count,
            "complete_shard_count": 1,
            "pairs_per_shard": count,
            "generator_commit": CORRECTION_GENERATOR,
            "proposal_equals_evaluation": True,
            "all_importance_weights_one": True,
        }
    correction_manifest = {
        "status": "passed",
        "generator_commit": CORRECTION_GENERATOR,
        "preregistration_hash": CORRECTION_HASH,
        "corrected_stage_a_train_count": 32768,
        "validation_count": 6144,
        "corrected_stage_b_train_count": 32768,
        "corrected_combined_train_count": 65536,
        "total_excluded_count": 5,
        "total_replacement_count": 5,
        "proposal_equals_evaluation": True,
        "all_importance_weights_one": True,
        "replacement_components_group_disjoint": True,
        "replacement_group_disjoint_from_all_base_components": True,
        "original_publications_modified": False,
        "model_training_authorized": False,
        "gwosc_gwtc_accessed": False,
        "validations": correction_validations,
        "views": {
            "stage_a_train": {
                "base_parent_root": str(stage_a),
                "base_dataset_id": "stage-a-train",
                "base_count": 32768,
                "excluded_physical_system_ids": ["a-00000", "a-00001"],
                "replacement_dataset_id": "replacement-a",
                "replacement_count": 2,
                "corrected_count": 32768,
            },
            "stage_a_validation": {
                "base_parent_root": str(stage_a),
                "base_dataset_id": "stage-a-validation",
                "corrected_count": 6144,
                "unchanged": True,
            },
            "stage_b_train": {
                "base_parent_root": str(stage_b),
                "base_dataset_id": "stage-b-train",
                "base_count": 32768,
                "excluded_physical_system_ids": ["b-00000", "b-00001", "b-00002"],
                "replacement_dataset_id": "replacement-b",
                "replacement_count": 3,
                "corrected_count": 32768,
            },
        },
    }
    (correction / "dataset_manifest.json").write_text(
        json.dumps(correction_manifest, sort_keys=True) + "\n"
    )
    return stage_a, stage_b, combined, correction, combined_hash


def _resolve(tmp_path: Path):  # type: ignore[no-untyped-def]
    stage_a, stage_b, combined, correction, combined_hash = _fixture(tmp_path)
    correction_hash = hashlib.sha256(
        (correction / "dataset_manifest.json").read_bytes()
    ).hexdigest()
    tree_hash, _ = tree_checksum(correction)
    publication = resolve_corrected_training_publication(
        correction,
        stage_a_parent_root=stage_a,
        stage_b_parent_root=stage_b,
        combined_base_root=combined,
        expected_base_generator_commit=BASE_GENERATOR,
        expected_base_preregistration_hash=RC4_HASH,
        expected_correction_generator_commit=CORRECTION_GENERATOR,
        expected_correction_preregistration_hash=CORRECTION_HASH,
        expected_correction_manifest_sha256=correction_hash,
        expected_correction_tree_sha256=tree_hash,
        expected_combined_base_manifest_sha256=combined_hash,
    )
    return publication


def test_corrected_publication_resolves_exact_membership(tmp_path: Path) -> None:
    publication = _resolve(tmp_path)
    assert publication.stage_a_excluded_ids == ("a-00000", "a-00001")
    assert publication.stage_b_excluded_ids == ("b-00000", "b-00001", "b-00002")
    assert publication.stage_a_replacement_ids == ("ra-00000", "ra-00001")
    assert publication.stage_b_replacement_ids == ("rb-00000", "rb-00001", "rb-00002")
    assert len(publication.corrected_stage_a_train_manifest_sha256) == 64
    assert len(publication.corrected_combined_train_manifest_sha256) == 64


def test_corrected_dataset_builders_exclude_and_replace_exactly(tmp_path: Path) -> None:
    publication = _resolve(tmp_path)
    curve = ASDCurve(np.asarray((0.0, 1024.0)), np.asarray((1.0, 1.0)), "test")
    curves = {"H1": curve, "L1": curve, "V1": curve}
    stage_a = corrected_stage_a_training_dataset(publication, curves)
    stage_a_ids = stage_a.physical_system_ids()
    assert len(stage_a_ids) == len(set(stage_a_ids)) == 32768
    assert not {"a-00000", "a-00001"} & set(stage_a_ids)
    assert {"ra-00000", "ra-00001"} <= set(stage_a_ids)
    subset = corrected_stage_a_training_dataset(
        publication, curves, selected_physical_system_ids=("a-00002", "ra-00001")
    )
    assert set(subset.physical_system_ids()) == {"a-00002", "ra-00001"}
    combined = corrected_65k_training_dataset(publication, curves)
    combined_ids = combined.physical_system_ids()
    assert len(combined_ids) == len(set(combined_ids)) == 65536
    assert not {"b-00000", "b-00001", "b-00002"} & set(combined_ids)
    assert {"rb-00000", "rb-00001", "rb-00002"} <= set(combined_ids)


def test_corrected_publication_fails_on_tree_or_manifest_drift(tmp_path: Path) -> None:
    stage_a, stage_b, combined, correction, combined_hash = _fixture(tmp_path)
    manifest_hash = hashlib.sha256(
        (correction / "dataset_manifest.json").read_bytes()
    ).hexdigest()
    tree_hash, _ = tree_checksum(correction)
    (correction / "unexpected.txt").write_text("drift\n")
    with pytest.raises(TrainingGateError, match="tree hash mismatch"):
        resolve_corrected_training_publication(
            correction,
            stage_a_parent_root=stage_a,
            stage_b_parent_root=stage_b,
            combined_base_root=combined,
            expected_base_generator_commit=BASE_GENERATOR,
            expected_base_preregistration_hash=RC4_HASH,
            expected_correction_generator_commit=CORRECTION_GENERATOR,
            expected_correction_preregistration_hash=CORRECTION_HASH,
            expected_correction_manifest_sha256=manifest_hash,
            expected_correction_tree_sha256=tree_hash,
            expected_combined_base_manifest_sha256=combined_hash,
        )
