"""Bounded-memory readers for atomically published Stage A shards."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..production.diagnostic_context import classify_balanced_tail
from ..schema import ArrayProductRole, SplitName, V2Record
from .contracts import TrainingGateError
from .features import (
    InputStandardizer,
    PreparedExample,
    prepare_example,
    prepare_metadata_example,
)
from .whitening import ASDCurve, whiten_detector_grid


@dataclass(frozen=True)
class ShardIndexEntry:
    path: Path
    row_index: int
    physical_system_id: str


@dataclass(frozen=True)
class StageAPublication:
    """Validated identity of the one atomic Stage A parent publication.

    Stage A deliberately publishes the train and validation datasets together by
    renaming their common parent.  The authoritative ``dataset_manifest.json``
    therefore lives at the parent, not inside either child dataset directory.
    """

    parent_root: Path
    manifest_path: Path
    manifest_sha256: str
    generator_commit: str
    preregistration_hash: str
    train_dataset_id: str
    validation_dataset_id: str
    train_root: Path
    validation_root: Path
    namespace_manifest_sha256: Mapping[str, str]


@dataclass(frozen=True)
class CombinedTrainingPublication:
    """Validated two-parent identity for the nested 65k training rung."""

    combined_root: Path
    combined_manifest_path: Path
    combined_manifest_sha256: str
    stage_a: StageAPublication
    stage_b_parent_root: Path
    stage_b_parent_manifest_path: Path
    stage_b_parent_manifest_sha256: str
    stage_b_dataset_id: str
    stage_b_train_root: Path
    train_manifest_sha256: str


@dataclass(frozen=True)
class CorrectedTrainingPublication:
    """Validated immutable overlay for corrected 32k and 65k training views."""

    correction_root: Path
    correction_manifest_path: Path
    correction_manifest_sha256: str
    correction_tree_sha256: str
    stage_a: StageAPublication
    combined_base: CombinedTrainingPublication
    stage_a_excluded_ids: Tuple[str, ...]
    stage_b_excluded_ids: Tuple[str, ...]
    stage_a_replacement_root: Path
    stage_b_replacement_root: Path
    stage_a_replacement_ids: Tuple[str, ...]
    stage_b_replacement_ids: Tuple[str, ...]
    corrected_stage_a_train_manifest_sha256: str
    corrected_combined_train_manifest_sha256: str


@dataclass(frozen=True)
class DevelopmentCase:
    """One validation example plus nondeployable diagnostic group labels."""

    example: PreparedExample
    tail_view: Optional[str]


@dataclass(frozen=True)
class CalibrationSBCCase:
    """One nondeployable development case with its frozen availability cell."""

    example: PreparedExample
    em_cell: str


class StandardizedCalibrationSBCDataset:
    """Apply training-only scales while retaining offline cell labels beside inputs."""

    def __init__(
        self, dataset: "PublishedStageADataset", standardizer: InputStandardizer
    ) -> None:
        if dataset.expected_split not in {
            SplitName.CALIBRATION_FIT,
            SplitName.SBC_DIAGNOSTIC,
        }:
            raise ValueError("calibration/SBC wrapper received another split")
        self.dataset = dataset
        self.standardizer = standardizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> CalibrationSBCCase:
        case = self.dataset.calibration_sbc_case(index)
        return CalibrationSBCCase(
            example=self.standardizer.transform(case.example),
            em_cell=case.em_cell,
        )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def resolve_stage_a_publication(
    parent_root: Path,
    *,
    expected_generator_commit: Optional[str] = None,
    expected_preregistration_hash: Optional[str] = None,
    expected_train_count: int = 32768,
    expected_validation_count: int = 6144,
    expected_pairs_per_shard: int = 128,
) -> StageAPublication:
    """Resolve child datasets only from a passed atomic parent manifest.

    This function reads manifests only.  Array and Parquet access remains lazy and
    happens after the separate scientific-training gate has passed.
    """

    root = parent_root.resolve()
    if "published" not in root.parts or not root.is_dir():
        raise TrainingGateError("Stage A input is not an atomic published parent")
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise TrainingGateError("Stage A parent publication manifest is absent")
    manifest = _load_mapping(manifest_path)
    if manifest.get("status") != "passed":
        raise TrainingGateError("Stage A parent publication did not pass validation")
    if manifest.get("model_training_authorized") is not False:
        raise TrainingGateError("Stage A data manifest must not self-authorize training")
    if manifest.get("train_validation_group_disjoint") is not True:
        raise TrainingGateError("Stage A parent manifest lacks group-disjoint validation")
    if not (
        manifest.get("proposal_equals_evaluation") is True
        and manifest.get("all_importance_weights_one") is True
    ):
        raise TrainingGateError("Stage A parent manifest violates the direct-target contract")
    if (
        int(manifest.get("accepted_pair_count", -1)),
        int(manifest.get("train_accepted_pair_count", -1)),
        int(manifest.get("validation_accepted_pair_count", -1)),
    ) != (
        expected_train_count + expected_validation_count,
        expected_train_count,
        expected_validation_count,
    ):
        raise TrainingGateError("Stage A parent count contract is inconsistent")
    generator_commit = str(manifest.get("generator_commit", ""))
    preregistration_hash = str(manifest.get("preregistration_hash", ""))
    if expected_generator_commit is not None and generator_commit != expected_generator_commit:
        raise TrainingGateError("Stage A generator commit differs from the training gate")
    if (
        expected_preregistration_hash is not None
        and preregistration_hash != expected_preregistration_hash
    ):
        raise TrainingGateError("Stage A preregistration hash differs from the training gate")
    validations = manifest.get("validations")
    if not isinstance(validations, dict) or set(validations) != {"train", "validation"}:
        raise TrainingGateError("Stage A parent manifest has incomplete split validations")
    expected = {
        "train": (expected_train_count, expected_train_count // expected_pairs_per_shard),
        "validation": (
            expected_validation_count,
            expected_validation_count // expected_pairs_per_shard,
        ),
    }
    dataset_ids: Dict[str, str] = {}
    namespace_hashes: Dict[str, str] = {}
    roots: Dict[str, Path] = {}
    for split, (accepted_count, shard_count) in expected.items():
        validation = validations[split]
        if not isinstance(validation, dict):
            raise TrainingGateError(f"Stage A {split} validation is not a mapping")
        if (
            validation.get("status"),
            validation.get("split"),
            int(validation.get("accepted_pair_count", -1)),
            int(validation.get("complete_shard_count", -1)),
            int(validation.get("pairs_per_shard", -1)),
        ) != ("passed", split, accepted_count, shard_count, expected_pairs_per_shard):
            raise TrainingGateError(f"Stage A {split} validation count contract failed")
        if validation.get("generator_commit") != generator_commit:
            raise TrainingGateError(f"Stage A {split} mixes generator commits")
        if not (
            validation.get("proposal_equals_evaluation") is True
            and validation.get("all_importance_weights_one") is True
        ):
            raise TrainingGateError(f"Stage A {split} is not exact direct-target data")
        dataset_id = str(validation.get("dataset_id", ""))
        if not dataset_id or Path(dataset_id).name != dataset_id:
            raise TrainingGateError(f"Stage A {split} dataset identity is invalid")
        dataset_root = root / dataset_id
        if not dataset_root.is_dir():
            raise TrainingGateError(f"Stage A {split} dataset directory is absent")
        run_manifest = _load_mapping(dataset_root / "run_manifest.json")
        if (
            run_manifest.get("split"),
            run_manifest.get("dataset_id"),
            run_manifest.get("generator_commit"),
            int(run_manifest.get("accepted_target", -1)),
            run_manifest.get("all_importance_weights_one"),
        ) != (split, dataset_id, generator_commit, accepted_count, True):
            raise TrainingGateError(f"Stage A {split} run manifest identity mismatch")
        dataset_ids[split] = dataset_id
        roots[split] = dataset_root
        namespace_hashes[split] = _canonical_mapping_sha256(
            {"run_manifest": run_manifest, "validation": validation}
        )
    if dataset_ids["train"] == dataset_ids["validation"]:
        raise TrainingGateError("Stage A train and validation identities collide")
    return StageAPublication(
        parent_root=root,
        manifest_path=manifest_path,
        manifest_sha256=_sha256_file(manifest_path),
        generator_commit=generator_commit,
        preregistration_hash=preregistration_hash,
        train_dataset_id=dataset_ids["train"],
        validation_dataset_id=dataset_ids["validation"],
        train_root=roots["train"],
        validation_root=roots["validation"],
        namespace_manifest_sha256=namespace_hashes,
    )


def resolve_combined_training_publication(
    combined_root: Path,
    *,
    stage_a_parent_root: Path,
    stage_b_parent_root: Path,
    expected_generator_commit: str,
    expected_preregistration_hash: str,
    expected_combined_manifest_sha256: Optional[str] = None,
) -> CombinedTrainingPublication:
    """Resolve Stage A + Stage B only from the passed atomic reference manifest."""

    combined = combined_root.resolve()
    if "published" not in combined.parts or not combined.is_dir():
        raise TrainingGateError("65k input is not an atomic published reference")
    combined_manifest_path = combined / "dataset_manifest.json"
    if not combined_manifest_path.is_file():
        raise TrainingGateError("65k combined manifest is absent")
    combined_manifest_sha256 = _sha256_file(combined_manifest_path)
    if (
        expected_combined_manifest_sha256 is not None
        and combined_manifest_sha256 != expected_combined_manifest_sha256
    ):
        raise TrainingGateError("65k combined manifest hash mismatch")
    manifest = _load_mapping(combined_manifest_path)
    if (
        manifest.get("status"),
        int(manifest.get("accepted_physical_system_count", -1)),
        int(manifest.get("validation_physical_system_count", -1)),
        manifest.get("strict_nested_train_ladder"),
        manifest.get("proposal_equals_evaluation"),
        manifest.get("all_importance_weights_one"),
        manifest.get("training_authorized"),
    ) != ("passed", 65536, 6144, True, True, True, False):
        raise TrainingGateError("65k combined manifest contract failed")
    group_validation = manifest.get("group_validation")
    if not isinstance(group_validation, dict) or not (
        group_validation.get("stage_a_stage_b_group_disjoint") is True
        and group_validation.get("stage_b_validation_group_disjoint") is True
    ):
        raise TrainingGateError("65k combined manifest lacks group-disjoint validation")
    components = manifest.get("components")
    if not isinstance(components, list) or len(components) != 2:
        raise TrainingGateError("65k combined manifest must name exactly two components")
    by_role = {
        str(component.get("role")): component
        for component in components
        if isinstance(component, dict)
    }
    if set(by_role) != {"stage_a_train", "stage_b_train_extension"}:
        raise TrainingGateError("65k combined component roles are invalid")
    if any(int(value.get("accepted_count", -1)) != 32768 for value in by_role.values()):
        raise TrainingGateError("65k combined component counts are invalid")
    stage_a = resolve_stage_a_publication(
        stage_a_parent_root,
        expected_generator_commit=expected_generator_commit,
        expected_preregistration_hash=expected_preregistration_hash,
    )
    if by_role["stage_a_train"].get("dataset_id") != stage_a.train_dataset_id:
        raise TrainingGateError("65k reference names the wrong Stage A train dataset")
    if by_role["stage_a_train"].get("parent_manifest_sha256") != stage_a.manifest_sha256:
        raise TrainingGateError("65k reference names the wrong Stage A parent manifest")
    stage_b_parent = stage_b_parent_root.resolve()
    if "published" not in stage_b_parent.parts or not stage_b_parent.is_dir():
        raise TrainingGateError("Stage B input is not an atomic parent publication")
    stage_b_manifest_path = stage_b_parent / "dataset_manifest.json"
    if not stage_b_manifest_path.is_file():
        raise TrainingGateError("Stage B parent manifest is absent")
    stage_b_manifest_sha256 = _sha256_file(stage_b_manifest_path)
    stage_b_component = by_role["stage_b_train_extension"]
    if stage_b_component.get("parent_manifest_sha256") != stage_b_manifest_sha256:
        raise TrainingGateError("65k reference names the wrong Stage B parent manifest")
    stage_b_manifest = _load_mapping(stage_b_manifest_path)
    validation = stage_b_manifest.get("validation")
    if not isinstance(validation, dict):
        raise TrainingGateError("Stage B parent lacks namespace validation")
    stage_b_dataset_id = str(validation.get("dataset_id", ""))
    if (
        stage_b_manifest.get("status"),
        stage_b_manifest.get("generator_commit"),
        stage_b_manifest.get("preregistration_hash"),
        int(stage_b_manifest.get("accepted_pair_count", -1)),
        int(stage_b_manifest.get("complete_shard_count", -1)),
        stage_b_manifest.get("proposal_equals_evaluation"),
        stage_b_manifest.get("all_importance_weights_one"),
        validation.get("status"),
        validation.get("split"),
        int(validation.get("accepted_pair_count", -1)),
        int(validation.get("complete_shard_count", -1)),
    ) != (
        "passed",
        expected_generator_commit,
        expected_preregistration_hash,
        32768,
        256,
        True,
        True,
        "passed",
        "train",
        32768,
        256,
    ):
        raise TrainingGateError("Stage B parent manifest contract failed")
    if stage_b_component.get("dataset_id") != stage_b_dataset_id:
        raise TrainingGateError("65k reference names the wrong Stage B dataset")
    stage_b_train_root = stage_b_parent / stage_b_dataset_id
    if not stage_b_train_root.is_dir():
        raise TrainingGateError("Stage B train dataset directory is absent")
    train_manifest_sha256 = _canonical_mapping_sha256(
        {
            "combined_manifest_sha256": combined_manifest_sha256,
            "stage_a_train_manifest_sha256": stage_a.namespace_manifest_sha256["train"],
            "stage_b_parent_manifest_sha256": stage_b_manifest_sha256,
        }
    )
    return CombinedTrainingPublication(
        combined_root=combined,
        combined_manifest_path=combined_manifest_path,
        combined_manifest_sha256=combined_manifest_sha256,
        stage_a=stage_a,
        stage_b_parent_root=stage_b_parent,
        stage_b_parent_manifest_path=stage_b_manifest_path,
        stage_b_parent_manifest_sha256=stage_b_manifest_sha256,
        stage_b_dataset_id=stage_b_dataset_id,
        stage_b_train_root=stage_b_train_root,
        train_manifest_sha256=train_manifest_sha256,
    )


def resolve_corrected_training_publication(
    correction_root: Path,
    *,
    stage_a_parent_root: Path,
    stage_b_parent_root: Path,
    combined_base_root: Path,
    expected_base_generator_commit: str,
    expected_base_preregistration_hash: str,
    expected_correction_generator_commit: str,
    expected_correction_preregistration_hash: str,
    expected_correction_manifest_sha256: str,
    expected_correction_tree_sha256: str,
    expected_combined_base_manifest_sha256: str,
) -> CorrectedTrainingPublication:
    """Resolve the immutable five-system overlay without opening strain arrays."""

    from ..production.storage import tree_checksum

    root = correction_root.resolve()
    if "published" not in root.parts or not root.is_dir():
        raise TrainingGateError("waveform correction is not an atomic publication")
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise TrainingGateError("waveform-correction parent manifest is absent")
    manifest_sha256 = _sha256_file(manifest_path)
    if manifest_sha256 != expected_correction_manifest_sha256:
        raise TrainingGateError("waveform-correction parent manifest hash mismatch")
    tree_sha256, _ = tree_checksum(root)
    if tree_sha256 != expected_correction_tree_sha256:
        raise TrainingGateError("waveform-correction publication tree hash mismatch")
    manifest = _load_mapping(manifest_path)
    if (
        manifest.get("status"),
        manifest.get("generator_commit"),
        manifest.get("preregistration_hash"),
        int(manifest.get("corrected_stage_a_train_count", -1)),
        int(manifest.get("validation_count", -1)),
        int(manifest.get("corrected_stage_b_train_count", -1)),
        int(manifest.get("corrected_combined_train_count", -1)),
        int(manifest.get("total_excluded_count", -1)),
        int(manifest.get("total_replacement_count", -1)),
        manifest.get("proposal_equals_evaluation"),
        manifest.get("all_importance_weights_one"),
        manifest.get("replacement_components_group_disjoint"),
        manifest.get("replacement_group_disjoint_from_all_base_components"),
        manifest.get("original_publications_modified"),
        manifest.get("model_training_authorized"),
        manifest.get("gwosc_gwtc_accessed"),
    ) != (
        "passed",
        expected_correction_generator_commit,
        expected_correction_preregistration_hash,
        32768,
        6144,
        32768,
        65536,
        5,
        5,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ):
        raise TrainingGateError("waveform-correction parent contract failed")
    stage_a = resolve_stage_a_publication(
        stage_a_parent_root,
        expected_generator_commit=expected_base_generator_commit,
        expected_preregistration_hash=expected_base_preregistration_hash,
    )
    combined = resolve_combined_training_publication(
        combined_base_root,
        stage_a_parent_root=stage_a_parent_root,
        stage_b_parent_root=stage_b_parent_root,
        expected_generator_commit=expected_base_generator_commit,
        expected_preregistration_hash=expected_base_preregistration_hash,
        expected_combined_manifest_sha256=expected_combined_base_manifest_sha256,
    )
    views = manifest.get("views")
    validations = manifest.get("validations")
    if not isinstance(views, dict) or not isinstance(validations, dict):
        raise TrainingGateError("waveform-correction views or validations are absent")
    if set(views) != {"stage_a_train", "stage_a_validation", "stage_b_train"}:
        raise TrainingGateError("waveform-correction view roles are incomplete")
    if set(validations) != {"stage_a_train", "stage_b_train"}:
        raise TrainingGateError("waveform-correction replacement validations are incomplete")
    if Path(str(views["stage_a_train"].get("base_parent_root", ""))).resolve() != (
        stage_a.parent_root
    ) or Path(str(views["stage_a_validation"].get("base_parent_root", ""))).resolve() != (
        stage_a.parent_root
    ):
        raise TrainingGateError("waveform correction names the wrong Stage A parent")
    if Path(str(views["stage_b_train"].get("base_parent_root", ""))).resolve() != (
        combined.stage_b_parent_root
    ):
        raise TrainingGateError("waveform correction names the wrong Stage B parent")
    if not (
        views["stage_a_train"].get("base_dataset_id") == stage_a.train_dataset_id
        and views["stage_a_validation"].get("base_dataset_id")
        == stage_a.validation_dataset_id
        and views["stage_b_train"].get("base_dataset_id")
        == combined.stage_b_dataset_id
        and views["stage_a_validation"].get("unchanged") is True
    ):
        raise TrainingGateError("waveform correction names the wrong base datasets")

    component_contract = {
        "stage_a_train": (2, views["stage_a_train"], stage_a.train_root),
        "stage_b_train": (3, views["stage_b_train"], combined.stage_b_train_root),
    }
    replacement_roots: Dict[str, Path] = {}
    replacement_ids: Dict[str, Tuple[str, ...]] = {}
    excluded_ids: Dict[str, Tuple[str, ...]] = {}
    replacement_validation_hashes: Dict[str, str] = {}
    for component, (count, view, base_root) in component_contract.items():
        validation = validations[component]
        dataset_id = str(view.get("replacement_dataset_id", ""))
        if (
            validation.get("status"),
            validation.get("dataset_id"),
            validation.get("split"),
            int(validation.get("accepted_pair_count", -1)),
            int(validation.get("complete_shard_count", -1)),
            int(validation.get("pairs_per_shard", -1)),
            validation.get("generator_commit"),
            validation.get("proposal_equals_evaluation"),
            validation.get("all_importance_weights_one"),
            int(view.get("base_count", -1)),
            int(view.get("replacement_count", -1)),
            int(view.get("corrected_count", -1)),
        ) != (
            "passed",
            dataset_id,
            "train",
            count,
            1,
            count,
            expected_correction_generator_commit,
            True,
            True,
            32768,
            count,
            32768,
        ):
            raise TrainingGateError(f"waveform-correction {component} contract failed")
        replacement_root = root / dataset_id
        entries = index_complete_shards(
            replacement_root,
            expected_pairs_per_shard=count,
            expected_total_pairs=count,
        )
        base_entries = index_complete_shards(
            base_root,
            expected_pairs_per_shard=128,
            expected_total_pairs=32768,
            require_published=component == "stage_a_train",
        )
        base_system_ids = {entry.physical_system_id for entry in base_entries}
        excluded = tuple(str(value) for value in view.get("excluded_physical_system_ids", ()))
        replacements = tuple(entry.physical_system_id for entry in entries)
        if (
            len(excluded) != count
            or len(set(excluded)) != count
            or not set(excluded) <= base_system_ids
            or set(replacements) & base_system_ids
        ):
            raise TrainingGateError(f"waveform-correction {component} membership failed")
        replacement_roots[component] = replacement_root
        replacement_ids[component] = replacements
        excluded_ids[component] = excluded
        replacement_validation_hashes[component] = _canonical_mapping_sha256(validation)

    validation_ids = {
        entry.physical_system_id
        for entry in index_complete_shards(
            stage_a.validation_root,
            expected_pairs_per_shard=128,
            expected_total_pairs=6144,
        )
    }
    corrected_stage_a_ids = (
        {
            entry.physical_system_id
            for entry in index_complete_shards(
                stage_a.train_root,
                expected_pairs_per_shard=128,
                expected_total_pairs=32768,
            )
        }
        - set(excluded_ids["stage_a_train"])
    ) | set(replacement_ids["stage_a_train"])
    corrected_stage_b_ids = (
        {
            entry.physical_system_id
            for entry in index_complete_shards(
                combined.stage_b_train_root,
                expected_pairs_per_shard=128,
                expected_total_pairs=32768,
                require_published=False,
            )
        }
        - set(excluded_ids["stage_b_train"])
    ) | set(replacement_ids["stage_b_train"])
    if not (
        len(corrected_stage_a_ids) == len(corrected_stage_b_ids) == 32768
        and len(corrected_stage_a_ids | corrected_stage_b_ids) == 65536
        and not ((corrected_stage_a_ids | corrected_stage_b_ids) & validation_ids)
    ):
        raise TrainingGateError("corrected train/validation membership is invalid")

    corrected_stage_a_hash = _canonical_mapping_sha256(
        {
            "correction_manifest_sha256": manifest_sha256,
            "base_train_manifest_sha256": stage_a.namespace_manifest_sha256["train"],
            "excluded_physical_system_ids": excluded_ids["stage_a_train"],
            "replacement_validation_sha256": replacement_validation_hashes[
                "stage_a_train"
            ],
            "replacement_physical_system_ids": replacement_ids["stage_a_train"],
        }
    )
    corrected_combined_hash = _canonical_mapping_sha256(
        {
            "correction_manifest_sha256": manifest_sha256,
            "combined_base_manifest_sha256": combined.combined_manifest_sha256,
            "corrected_stage_a_train_manifest_sha256": corrected_stage_a_hash,
            "stage_b_parent_manifest_sha256": combined.stage_b_parent_manifest_sha256,
            "stage_b_excluded_physical_system_ids": excluded_ids["stage_b_train"],
            "stage_b_replacement_validation_sha256": replacement_validation_hashes[
                "stage_b_train"
            ],
            "stage_b_replacement_physical_system_ids": replacement_ids["stage_b_train"],
        }
    )
    return CorrectedTrainingPublication(
        correction_root=root,
        correction_manifest_path=manifest_path,
        correction_manifest_sha256=manifest_sha256,
        correction_tree_sha256=tree_sha256,
        stage_a=stage_a,
        combined_base=combined,
        stage_a_excluded_ids=excluded_ids["stage_a_train"],
        stage_b_excluded_ids=excluded_ids["stage_b_train"],
        stage_a_replacement_root=replacement_roots["stage_a_train"],
        stage_b_replacement_root=replacement_roots["stage_b_train"],
        stage_a_replacement_ids=replacement_ids["stage_a_train"],
        stage_b_replacement_ids=replacement_ids["stage_b_train"],
        corrected_stage_a_train_manifest_sha256=corrected_stage_a_hash,
        corrected_combined_train_manifest_sha256=corrected_combined_hash,
    )


def _load_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document is not a mapping: {path}")
    return value


def index_complete_shards(
    dataset_root: Path,
    *,
    expected_pairs_per_shard: int,
    expected_total_pairs: Optional[int] = None,
    require_published: bool = True,
) -> Tuple[ShardIndexEntry, ...]:
    """Index manifest IDs without opening Parquet or Zarr products."""

    if require_published and "published" not in dataset_root.parts:
        raise TrainingGateError("training readers refuse staging or canary roots")
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if require_published:
        parent_manifest = dataset_root.parent / "dataset_manifest.json"
        if not parent_manifest.is_file():
            raise TrainingGateError("atomic parent publication manifest is missing")
        parent = _load_mapping(parent_manifest)
        validation = parent.get("validations", {}).get(dataset_root.name)
        if validation is None:
            validation = next(
                (
                    item
                    for item in parent.get("validations", {}).values()
                    if isinstance(item, dict) and item.get("dataset_id") == dataset_root.name
                ),
                None,
            )
        if parent.get("status") != "passed" or not isinstance(validation, dict):
            raise TrainingGateError("dataset is not named by a passed parent publication")
    shards_root = dataset_root / "shards"
    complete = sorted(
        path
        for path in shards_root.glob("shard-[0-9][0-9][0-9][0-9][0-9]")
        if path.is_dir()
    )
    partial = tuple(shards_root.glob("shard-*.partial"))
    if partial:
        raise TrainingGateError("published dataset contains partial shards")
    entries: List[ShardIndexEntry] = []
    for shard in complete:
        if not (shard / "COMPLETE.json").is_file():
            raise TrainingGateError(f"complete marker missing: {shard}")
        manifest = _load_mapping(shard / "shard_manifest.json")
        if manifest.get("status") != "complete":
            raise TrainingGateError(f"shard is not complete: {shard}")
        count = int(manifest["accepted_pair_count"])
        ids = tuple(str(value) for value in manifest["physical_system_ids"])
        if count != expected_pairs_per_shard or len(ids) != count:
            raise TrainingGateError(f"shard count mismatch: {shard}")
        entries.extend(
            ShardIndexEntry(shard, row_index, physical_id)
            for row_index, physical_id in enumerate(ids)
        )
    if len({entry.physical_system_id for entry in entries}) != len(entries):
        raise TrainingGateError("published dataset contains duplicate physical-system IDs")
    if expected_total_pairs is not None and len(entries) != expected_total_pairs:
        raise TrainingGateError(
            f"published dataset contains {len(entries)}, expected {expected_total_pairs}"
        )
    return tuple(entries)


class PublishedStageADataset:
    """Open one row at a time and never expose clean/noise arrays to the model."""

    def __init__(
        self,
        dataset_root: Path,
        *,
        expected_split: SplitName,
        detector_curves: Mapping[str, ASDCurve],
        expected_pairs_per_shard: int = 128,
        expected_total_pairs: Optional[int] = None,
        selected_physical_system_ids: Optional[Sequence[str]] = None,
        require_published: bool = True,
        minimum_frequency_hz: float = 20.0,
        maximum_frequency_hz: Optional[float] = None,
    ) -> None:
        if set(detector_curves) != {"H1", "L1", "V1"}:
            raise ValueError("detector ASD curves must cover H1/L1/V1 exactly")
        self.dataset_root = dataset_root
        self.expected_split = expected_split
        self.detector_curves: Tuple[ASDCurve, ASDCurve, ASDCurve] = (
            detector_curves["H1"],
            detector_curves["L1"],
            detector_curves["V1"],
        )
        self.minimum_frequency_hz = minimum_frequency_hz
        self.maximum_frequency_hz = maximum_frequency_hz
        entries = index_complete_shards(
            dataset_root,
            expected_pairs_per_shard=expected_pairs_per_shard,
            expected_total_pairs=expected_total_pairs,
            require_published=require_published,
        )
        if selected_physical_system_ids is not None:
            selected = frozenset(selected_physical_system_ids)
            if len(selected) != len(tuple(selected_physical_system_ids)):
                raise ValueError("selected physical-system IDs contain duplicates")
            entries = tuple(entry for entry in entries if entry.physical_system_id in selected)
            if len(entries) != len(selected):
                raise TrainingGateError("selected IDs are not all present in publication")
        self.entries = entries
        self._cached_path: Optional[Path] = None
        self._cached_records: Any = None
        self._cached_noisy: Any = None

    def __len__(self) -> int:
        return len(self.entries)

    def _open_records(self, path: Path) -> Any:
        if self._cached_path != path:
            pandas = importlib.import_module("pandas")
            records = pandas.read_parquet(path / "records.parquet")
            self._cached_path = path
            self._cached_records = records
            self._cached_noisy = None
        return self._cached_records

    def _record(self, entry: ShardIndexEntry) -> V2Record:
        records = self._open_records(entry.path)
        row = records.iloc[entry.row_index]
        record = V2Record.from_json(str(row["record_json"]))
        record.validate()
        if record.pair.physical_system_id != entry.physical_system_id:
            raise TrainingGateError("Parquet row order disagrees with shard manifest")
        if record.pair.split is not self.expected_split:
            raise TrainingGateError("record belongs to the wrong split")
        distribution = record.provenance.distribution
        if (
            distribution.proposal_log_probability
            != distribution.evaluation_prior_log_probability
            or distribution.importance_weight != 1.0
            or not distribution.weight_valid
            or distribution.clipping_applied
        ):
            raise TrainingGateError("Stage A reader requires exact direct-target unit weights")
        roles = {reference.product_role for reference in record.gw_observation.array_products}
        if roles != set(ArrayProductRole):
            raise TrainingGateError("record array-product roles are incomplete")
        return record

    def _open_noisy(self, path: Path, expected_rows: int) -> Any:
        if self._cached_path != path:
            self._open_records(path)
        if self._cached_noisy is None:
            zarr = importlib.import_module("zarr")
            noisy = zarr.open_array(str(path / "noisy.zarr"), mode="r")
            if expected_rows != noisy.shape[0]:
                raise TrainingGateError("Parquet/Zarr row counts disagree")
            self._cached_noisy = noisy
        return self._cached_noisy

    def metadata_example(self, index: int) -> PreparedExample:
        """Read one record for standardizer fitting without opening strain arrays."""

        entry = self.entries[index]
        record = self._record(entry)
        return prepare_metadata_example(record)

    def development_case(self, index: int) -> DevelopmentCase:
        """Read validation-only labels that are never passed into the model."""

        if self.expected_split is not SplitName.VALIDATION:
            raise TrainingGateError("development diagnostics require the validation split")
        entry = self.entries[index]
        record = self._record(entry)
        selection = record.provenance.selection
        if selection is None:
            raise TrainingGateError("development record lacks frozen selection provenance")
        images = {image.image_id: image for image in record.lens_truth.physical_images}
        selected = (
            images[record.pair.primary_image_id],
            images[record.pair.secondary_image_id],
        )
        secondary_snr = float(
            selection.per_image_network_optimal_snr[record.pair.secondary_image_id]
        )
        density_slope = float(record.lens_truth.lens_parameters.get("density_slope", 2.0))
        tail = classify_balanced_tail(
            selected,
            secondary_network_snr=secondary_snr,
            external_convergence=record.lens_truth.external_convergence,
            density_slope=density_slope,
        )
        return DevelopmentCase(
            example=self[index], tail_view=None if tail is None else tail.value
        )

    def calibration_sbc_case(self, index: int) -> CalibrationSBCCase:
        """Return labels beside, never inside, calibration/SBC model inputs."""

        if self.expected_split not in {
            SplitName.CALIBRATION_FIT,
            SplitName.SBC_DIAGNOSTIC,
        }:
            raise TrainingGateError(
                "calibration/SBC labels require a dedicated development split"
            )
        entry = self.entries[index]
        records = self._open_records(entry.path)
        row = records.iloc[entry.row_index]
        cell = str(row["em_cell"])
        if not cell:
            raise TrainingGateError("calibration/SBC record has no EM-cell label")
        return CalibrationSBCCase(example=self[index], em_cell=cell)

    def __getitem__(self, index: int) -> PreparedExample:
        entry = self.entries[index]
        record = self._record(entry)
        records = self._cached_records
        noisy_store = self._open_noisy(entry.path, len(records))
        noisy = np.asarray(noisy_store[entry.row_index], dtype=np.float32)
        detector_mask = np.asarray(
            record.gw_observation.detector_availability_mask, dtype=bool
        )
        whitened = whiten_detector_grid(
            noisy,
            detector_mask,
            sampling_frequency_hz=record.gw_observation.sample_rate_hz,
            detector_curves=self.detector_curves,
            minimum_frequency_hz=self.minimum_frequency_hz,
            maximum_frequency_hz=self.maximum_frequency_hz,
        )
        cell = str(records.iloc[entry.row_index]["em_cell"])
        if not cell:
            raise TrainingGateError("published record has no EM-cell partition label")
        return replace(prepare_example(record, whitened), em_cell=cell)

    def physical_system_ids(self) -> Tuple[str, ...]:
        return tuple(entry.physical_system_id for entry in self.entries)


class ConcatenatedPublishedStageADataset(PublishedStageADataset):
    """Expose immutable direct-target components as one bounded-memory dataset."""

    def __init__(self, components: Sequence[PublishedStageADataset]) -> None:
        datasets = tuple(components)
        if len(datasets) < 2:
            raise ValueError("concatenated training data requires at least two components")
        first = datasets[0]
        if any(dataset.expected_split is not first.expected_split for dataset in datasets):
            raise TrainingGateError("concatenated components belong to different splits")
        curve_ids = tuple((curve.identity, curve.sha256) for curve in first.detector_curves)
        for dataset in datasets[1:]:
            if tuple((curve.identity, curve.sha256) for curve in dataset.detector_curves) != (
                curve_ids
            ):
                raise TrainingGateError("concatenated components use different PSD curves")
            if (
                dataset.minimum_frequency_hz != first.minimum_frequency_hz
                or dataset.maximum_frequency_hz != first.maximum_frequency_hz
            ):
                raise TrainingGateError("concatenated components use different whitening bands")
        entries = tuple(entry for dataset in datasets for entry in dataset.entries)
        identifiers = tuple(entry.physical_system_id for entry in entries)
        if len(identifiers) != len(set(identifiers)):
            raise TrainingGateError("concatenated components contain duplicate systems")
        self.components = datasets
        self.dataset_root = first.dataset_root.parent
        self.expected_split = first.expected_split
        self.detector_curves = first.detector_curves
        self.minimum_frequency_hz = first.minimum_frequency_hz
        self.maximum_frequency_hz = first.maximum_frequency_hz
        self.entries = entries
        self._cached_path = None
        self._cached_records = None
        self._cached_noisy = None


def corrected_stage_a_training_dataset(
    publication: CorrectedTrainingPublication,
    detector_curves: Mapping[str, ASDCurve],
    *,
    selected_physical_system_ids: Optional[Sequence[str]] = None,
) -> ConcatenatedPublishedStageADataset:
    """Build the exact 32k corrected Stage A view, optionally restricted by ID."""

    base_all = PublishedStageADataset(
        publication.stage_a.train_root,
        expected_split=SplitName.TRAIN,
        detector_curves=detector_curves,
        expected_total_pairs=32768,
    )
    base_ids = tuple(
        physical_id
        for physical_id in base_all.physical_system_ids()
        if physical_id not in set(publication.stage_a_excluded_ids)
    )
    replacement_all = PublishedStageADataset(
        publication.stage_a_replacement_root,
        expected_split=SplitName.TRAIN,
        detector_curves=detector_curves,
        expected_pairs_per_shard=2,
        expected_total_pairs=2,
    )
    replacement_ids = replacement_all.physical_system_ids()
    complete_ids = base_ids + replacement_ids
    if len(complete_ids) != 32768 or len(set(complete_ids)) != 32768:
        raise TrainingGateError("corrected Stage A view is not exactly 32k")
    if selected_physical_system_ids is None:
        selected = frozenset(complete_ids)
    else:
        requested = tuple(selected_physical_system_ids)
        selected = frozenset(requested)
        if len(selected) != len(requested) or not selected <= set(complete_ids):
            raise TrainingGateError("corrected Stage A selection is invalid")
    base = PublishedStageADataset(
        publication.stage_a.train_root,
        expected_split=SplitName.TRAIN,
        detector_curves=detector_curves,
        expected_total_pairs=32768,
        selected_physical_system_ids=tuple(
            physical_id for physical_id in base_ids if physical_id in selected
        ),
    )
    replacement = PublishedStageADataset(
        publication.stage_a_replacement_root,
        expected_split=SplitName.TRAIN,
        detector_curves=detector_curves,
        expected_pairs_per_shard=2,
        expected_total_pairs=2,
        selected_physical_system_ids=tuple(
            physical_id for physical_id in replacement_ids if physical_id in selected
        ),
    )
    result = ConcatenatedPublishedStageADataset((base, replacement))
    if len(result) != len(selected):
        raise TrainingGateError("corrected Stage A selection count is inconsistent")
    return result


def corrected_65k_training_dataset(
    publication: CorrectedTrainingPublication,
    detector_curves: Mapping[str, ASDCurve],
) -> ConcatenatedPublishedStageADataset:
    """Build the exact corrected Stage A + Stage B 65k view."""

    stage_a = corrected_stage_a_training_dataset(publication, detector_curves)
    stage_b_base_all = PublishedStageADataset(
        publication.combined_base.stage_b_train_root,
        expected_split=SplitName.TRAIN,
        detector_curves=detector_curves,
        expected_total_pairs=32768,
        require_published=False,
    )
    excluded = set(publication.stage_b_excluded_ids)
    stage_b_base_ids = tuple(
        physical_id
        for physical_id in stage_b_base_all.physical_system_ids()
        if physical_id not in excluded
    )
    stage_b_base = PublishedStageADataset(
        publication.combined_base.stage_b_train_root,
        expected_split=SplitName.TRAIN,
        detector_curves=detector_curves,
        expected_total_pairs=32768,
        selected_physical_system_ids=stage_b_base_ids,
        require_published=False,
    )
    stage_b_replacement = PublishedStageADataset(
        publication.stage_b_replacement_root,
        expected_split=SplitName.TRAIN,
        detector_curves=detector_curves,
        expected_pairs_per_shard=3,
        expected_total_pairs=3,
    )
    result = ConcatenatedPublishedStageADataset(
        (stage_a, stage_b_base, stage_b_replacement)
    )
    if len(result) != 65536:
        raise TrainingGateError("corrected combined training view is not exactly 65k")
    return result


class StandardizedStageADataset:
    """Apply frozen rung statistics lazily without mutating the publication."""

    def __init__(
        self, dataset: PublishedStageADataset, standardizer: InputStandardizer
    ) -> None:
        self.dataset = dataset
        self.standardizer = standardizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> PreparedExample:
        return self.standardizer.transform(self.dataset[index])


class DevelopmentStageADataset:
    """Apply the training-rung standardizer while keeping labels out of inputs."""

    def __init__(
        self, dataset: PublishedStageADataset, standardizer: InputStandardizer
    ) -> None:
        if dataset.expected_split is not SplitName.VALIDATION:
            raise ValueError("development dataset must wrap validation")
        self.dataset = dataset
        self.standardizer = standardizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> DevelopmentCase:
        case = self.dataset.development_case(index)
        return DevelopmentCase(
            example=self.standardizer.transform(case.example), tail_view=case.tail_view
        )


def torch_collate(examples: Sequence[PreparedExample]) -> Mapping[str, Any]:
    """Convert a prepared batch after optional PyTorch is installed."""

    from .features import collate_numpy

    torch = importlib.import_module("torch")
    arrays = collate_numpy(examples)
    return {name: torch.from_numpy(value) for name, value in arrays.items()}


def torch_development_collate(
    cases: Sequence[DevelopmentCase],
) -> Tuple[Mapping[str, Any], Tuple[Mapping[str, Optional[str]], ...]]:
    """Keep IDs and group labels beside, never inside, deployable tensors."""

    tensors = torch_collate([case.example for case in cases])
    metadata = tuple(
        {
            "physical_system_id": case.example.physical_system_id,
            "lens_family": case.example.lens_family,
            "em_cell_signature": case.example.em_cell_signature,
            "tail_view": case.tail_view,
        }
        for case in cases
    )
    return tensors, metadata


def torch_calibration_sbc_collate(
    cases: Sequence[CalibrationSBCCase],
) -> Tuple[Mapping[str, Any], Tuple[Mapping[str, str], ...]]:
    """Keep calibration/SBC identity labels outside the deployable tensor map."""

    tensors = torch_collate([case.example for case in cases])
    metadata = tuple(
        {
            "physical_system_id": case.example.physical_system_id,
            "lens_family": case.example.lens_family,
            "em_cell": case.em_cell,
        }
        for case in cases
    )
    return tensors, metadata
