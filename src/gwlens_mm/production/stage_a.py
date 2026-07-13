"""Fail-closed contracts for direct-target Stage A and its disposable canary."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Set, Tuple

import numpy as np

from ..arrays import validate_strain_array_semantics
from ..config import load_yaml
from ..provenance import canonical_json, configuration_hash, dataset_id
from ..schema import SplitName, V2Record
from .run_control import AttemptRecord
from .storage import tree_checksum, verify_complete_shard

PHASE4_CONFIG = "configs/data/phase4_direct_target_stage_a.yaml"
DIRECT_TARGET_ID = "balanced_literature_informed_benchmark_v1"
CANARY_NAMESPACES = ("train_namespace", "validation_namespace")
OFFICIAL_NAMESPACES = ("train", "validation")


@dataclass(frozen=True)
class CanaryIdentity:
    generator_commit: str
    parent_run_id: str
    train_dataset_id: str
    validation_dataset_id: str
    configuration_hash: str


def load_phase4_contract(
    root: Path, config_path: str = PHASE4_CONFIG
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load the design-only RC.4 delta and reject accidental execution changes."""

    config = load_yaml(root / config_path)
    preregistration = load_yaml(root / config["preregistration"]["path"])
    design = load_yaml(root / config["authorization"]["design_path"])
    if config.get("phase") != "4" or preregistration.get("phase") != "4":
        raise ValueError("Phase 4 contract identity is absent")
    if preregistration.get("preregistration_version") != "1.1.0-rc.4":
        raise ValueError("direct-target preregistration version changed")
    if configuration_hash(preregistration) != config["preregistration"]["canonical_hash"]:
        raise ValueError("direct-target RC.4 canonical hash mismatch")
    parent = load_yaml(root / preregistration["parent_adaptive_preregistration"]["path"])
    if (
        configuration_hash(parent)
        != preregistration["parent_adaptive_preregistration"]["canonical_hash"]
    ):
        raise ValueError("adaptive RC.3 parent hash mismatch")
    if design.get("base_main_commit") != "ce0cf464cf5b56e3df5e1b0c93ffadc12f2e517a":
        raise ValueError("Phase 4 design base commit changed")
    direct = preregistration["direct_target_training_contract"]
    if not (
        direct["q_train_equals_p_eval"] is True
        and direct["training_proposal_id"] == DIRECT_TARGET_ID
        and direct["evaluation_target_id"] == DIRECT_TARGET_ID
        and float(direct["log_importance_weight"]) == 0.0
        and float(direct["importance_weight"]) == 1.0
    ):
        raise ValueError("direct-target unit-weight contract changed")
    stage = config["stage_a"]
    if (
        int(stage["train"]["accepted_pair_count"]),
        int(stage["validation"]["accepted_pair_count"]),
        int(stage["total_accepted_pair_count"]),
        int(stage["total_shard_count"]),
    ) != (32768, 6144, 38912, 304):
        raise ValueError("Stage A accepted-count contract changed")
    authorization = preregistration["authorization"]
    if authorization.get("design_and_implementation_authorized") is not True:
        raise ValueError("RC.4 design authorization is absent")
    for key, value in authorization.items():
        if key != "design_and_implementation_authorized" and value is not False:
            raise ValueError(f"RC.4 requires {key}=false")
    if config["execution"]["scientific_execution_enabled"] is not False:
        raise ValueError("scientific execution was enabled in the design config")
    if config["execution"]["canary_execution_enabled"] is not False:
        raise ValueError("canary execution was enabled in the design config")
    return config, preregistration, design


def build_namespace_config(
    root: Path,
    config: Mapping[str, Any],
    namespace: str,
    *,
    canary: bool,
) -> Dict[str, Any]:
    """Build the exact production-generator config for one split or canary namespace."""

    if canary:
        if namespace not in CANARY_NAMESPACES:
            raise ValueError(f"unknown canary namespace: {namespace}")
        specification = config["disposable_canary"][namespace]
        split = str(config["disposable_canary"]["split"])
        purpose = str(config["disposable_canary"]["dataset_purpose"])
    else:
        if namespace not in OFFICIAL_NAMESPACES:
            raise ValueError(f"unknown official namespace: {namespace}")
        specification = config["stage_a"][namespace]
        split = str(specification["split"])
        purpose = str(config["dataset_purpose"])
    base = load_yaml(root / config["base_data_config"])
    base.update(
        {
            "phase": "4-canary" if canary else "4",
            "root_seed": int(specification["root_seed"]),
            "dataset_purpose": purpose,
            "accepted_pair_count": int(specification["accepted_pair_count"]),
            "shard_count": int(specification["shard_count"]),
            "pairs_per_shard": int(specification["pairs_per_shard"]),
            "production_context": {
                "proposal_mode": "evaluation_target_direct",
                "proposal_distribution_id": DIRECT_TARGET_ID,
                "evaluation_distribution_id": DIRECT_TARGET_ID,
                "id_prefix": str(specification["id_prefix"]),
                "split": split,
                "canary": canary,
            },
        }
    )
    base["execution"] = {
        **base["execution"],
        "qualification_worker_processes": (
            1 if canary else int(config["execution"]["worker_processes"])
        ),
        "attempt_id_stride": int(specification["attempt_id_stride"]),
        "maximum_attempts_per_worker": int(
            config["disposable_canary"]["maximum_attempts_per_namespace"]
            if canary
            else config["execution"]["maximum_attempts_per_worker"]
        ),
        "maximum_active_seconds_per_worker": int(
            config["disposable_canary"]["maximum_active_seconds_per_namespace"]
            if canary
            else config["execution"]["maximum_active_seconds_per_worker"]
        ),
    }
    return base


def derive_canary_identity(
    config: Mapping[str, Any], generator_commit: str
) -> CanaryIdentity:
    if len(generator_commit) != 40:
        raise ValueError("generator commit must be a full Git SHA")
    config_hash = configuration_hash(config)
    canary = config["disposable_canary"]
    parent = f"phase4-canary-{generator_commit[:12]}-{config_hash[:12]}"
    train = dataset_id(
        "2.0.0-alpha.3",
        generator_commit,
        config_hash,
        int(canary["train_namespace"]["root_seed"]),
    ) + "-canary-train"
    validation = dataset_id(
        "2.0.0-alpha.3",
        generator_commit,
        config_hash,
        int(canary["validation_namespace"]["root_seed"]),
    ) + "-canary-validation"
    if train == validation:
        raise ValueError("canary dataset identities collide")
    return CanaryIdentity(generator_commit, parent, train, validation, config_hash)


def verify_generator_commit(
    root: Path,
    generator_commit: str,
    *,
    allowed_postfreeze_paths: Sequence[str] = (),
) -> None:
    """Verify frozen code while allowing an authorization-only descendant."""

    if (root / ".git").exists():
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != generator_commit:
            ancestry = subprocess.run(
                ["git", "merge-base", "--is-ancestor", generator_commit, head],
                cwd=root,
                check=False,
            )
            if ancestry.returncode != 0:
                raise ValueError("generator commit is not an ancestor of checkout HEAD")
            changed = subprocess.run(
                ["git", "diff", "--name-only", f"{generator_commit}..{head}"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            unexpected = sorted(set(changed) - set(allowed_postfreeze_paths))
            if unexpected:
                raise ValueError(
                    "post-freeze checkout changes protected paths: "
                    + ", ".join(unexpected)
                )
        return
    marker = root / "SYNCED_COMMIT"
    if not marker.is_file() or marker.read_text().strip() != generator_commit:
        raise ValueError("disposable checkout lacks exact synchronized commit marker")


def validate_direct_target_record(
    record: V2Record, *, expected_split: SplitName, expected_dataset: str
) -> None:
    """Validate split identity and the exact q=p, unit-weight provenance."""

    record.validate()
    if record.pair.split is not expected_split:
        raise ValueError("record split differs from the direct-target namespace")
    if record.pair.dataset_version != expected_dataset:
        raise ValueError("record dataset identity mismatch")
    distribution = record.provenance.distribution
    values = (
        float(distribution.proposal_log_probability),
        float(distribution.evaluation_prior_log_probability),
        float(distribution.importance_weight),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("direct-target density provenance is nonfinite")
    if values[0] != values[1] or values[2] != 1.0:
        raise ValueError("direct-target record does not have exact unit weight")


def validate_direct_target_shard(
    shard: Path,
    *,
    expected_split: SplitName,
    expected_dataset: str,
    expected_pairs: int,
) -> Dict[str, Any]:
    """Read the real Parquet records and validate every direct-target record."""

    pandas = importlib.import_module("pandas")
    frame = pandas.read_parquet(shard / "records.parquet")
    if len(frame) != expected_pairs:
        raise ValueError("direct-target shard accepted count mismatch")
    pair_ids: set[str] = set()
    for value in frame["record_json"]:
        record = V2Record.from_json(str(value))
        validate_direct_target_record(
            record,
            expected_split=expected_split,
            expected_dataset=expected_dataset,
        )
        if record.pair.pair_id in pair_ids:
            raise ValueError("direct-target shard contains a duplicate pair ID")
        pair_ids.add(record.pair.pair_id)
    return {
        "status": "passed",
        "accepted_pair_count": len(pair_ids),
        "proposal_equals_evaluation": True,
        "all_importance_weights_one": True,
    }


def validate_canary_manifest(manifest: Mapping[str, Any], generator_commit: str) -> None:
    """Validate a completed disposable canary without inspecting efficiency endpoints."""

    if manifest.get("status") != "passed":
        raise ValueError("disposable canary did not pass")
    if manifest.get("generator_commit") != generator_commit:
        raise ValueError("canary generator commit differs from release commit")
    if int(manifest.get("accepted_pair_count", -1)) != 16:
        raise ValueError("canary accepted count must be exactly 16")
    if manifest.get("scientific_use_authorized") is not False:
        raise ValueError("canary scientific-use denial is absent")
    if manifest.get("training_use_authorized") is not False:
        raise ValueError("canary training-use denial is absent")
    if manifest.get("throughput_or_ess_inspected") is not False:
        raise ValueError("canary inspected a forbidden efficiency endpoint")
    if manifest.get("resume_first_namespace_byte_identical") is not True:
        raise ValueError("canary interruption/resume evidence is absent")


def validate_stage_a_namespace(
    stage: Path,
    *,
    namespace_config: Mapping[str, Any],
    expected_split: SplitName,
    expected_dataset: str,
    generator_commit: str,
) -> Tuple[Dict[str, Any], Dict[str, Set[str]]]:
    """Validate every scientific shard, array, journal, ID, and unit weight."""

    pandas = importlib.import_module("pandas")
    zarr = importlib.import_module("zarr")
    expected_shards = int(namespace_config["shard_count"])
    pairs_per_shard = int(namespace_config["pairs_per_shard"])
    expected_total = int(namespace_config["accepted_pair_count"])
    expected_config_hash = configuration_hash(namespace_config)
    stride = int(namespace_config["execution"]["attempt_id_stride"])
    identifiers: Dict[str, Set[str]] = {
        "pair": set(),
        "source": set(),
        "lens": set(),
        "system": set(),
        "noise": set(),
        "attempt_system": set(),
    }
    attempt_ids: set[int] = set()
    accepted_attempt_pairs: set[str] = set()
    accepted_attempt_count = 0
    attempt_count = 0
    family_counts: Counter[str] = Counter()
    em_cell_counts: Counter[str] = Counter()
    shard_artifacts: list[Dict[str, Any]] = []
    for shard_index in range(expected_shards):
        shard = stage / "shards" / f"shard-{shard_index:05d}"
        verify_complete_shard(shard, pairs_per_shard)
        digest, byte_count = tree_checksum(shard)
        shard_artifacts.append(
            {"shard_index": shard_index, "sha256": digest, "bytes": byte_count}
        )
        frame = pandas.read_parquet(shard / "records.parquet")
        arrays = {
            name: zarr.open_array(str(shard / f"{name}.zarr"), mode="r")
            for name in ("noisy", "clean", "noise")
        }
        expected_shape = (
            pairs_per_shard,
            2,
            3,
            int(namespace_config["gw"]["sample_count"]),
        )
        if len(frame) != pairs_per_shard or any(
            array.shape != expected_shape for array in arrays.values()
        ):
            raise ValueError("Stage A shard shape or record count mismatch")
        for row_index, row in frame.iterrows():
            record = V2Record.from_json(str(row["record_json"]))
            validate_direct_target_record(
                record,
                expected_split=expected_split,
                expected_dataset=expected_dataset,
            )
            if record.provenance.generator_git_commit != generator_commit:
                raise ValueError("Stage A record mixes generator commits")
            if record.provenance.configuration_hash != expected_config_hash:
                raise ValueError("Stage A record mixes namespace configurations")
            values = {
                "pair": record.pair.pair_id,
                "source": record.pair.source_id,
                "lens": record.pair.lens_id,
                "system": record.pair.physical_system_id,
            }
            for key, value in values.items():
                if value in identifiers[key]:
                    raise ValueError(f"Stage A duplicate {key} ID")
                if value.startswith(("qualification-", "phase3ca", "phase4-canary")):
                    raise ValueError("engineering ID entered a scientific Stage A split")
                identifiers[key].add(value)
            for noise_id in record.provenance.used_noise_segment_ids:
                if noise_id in identifiers["noise"]:
                    raise ValueError("Stage A duplicate noise-segment ID")
                identifiers["noise"].add(noise_id)
            validate_strain_array_semantics(
                np.asarray(arrays["noisy"][row_index]),
                np.asarray(arrays["clean"][row_index]),
                np.asarray(arrays["noise"][row_index]),
                record.gw_observation.detector_availability_mask,
            )
            family_counts[record.pair.lens_family.value] += 1
            em_cell_counts[str(row["em_cell"])] += 1
        journal = stage / "attempts" / f"shard-{shard_index:05d}.jsonl"
        for line_number, line in enumerate(journal.read_text().splitlines()):
            attempt = AttemptRecord(**json.loads(line))
            attempt.validate()
            expected_attempt_id = shard_index + line_number * stride
            if attempt.attempt_id != expected_attempt_id or attempt.attempt_id in attempt_ids:
                raise ValueError("Stage A attempt ID is duplicated or outside its stream")
            if attempt.physical_system_id in identifiers["attempt_system"]:
                raise ValueError("Stage A attempt physical-system ID is duplicated")
            attempt_ids.add(attempt.attempt_id)
            identifiers["attempt_system"].add(attempt.physical_system_id)
            attempt_count += 1
            if attempt.status == "accepted":
                if attempt.pair_id is None or attempt.pair_id in accepted_attempt_pairs:
                    raise ValueError("Stage A accepted attempt pair ID is invalid")
                accepted_attempt_pairs.add(attempt.pair_id)
                accepted_attempt_count += 1
    if len(identifiers["pair"]) != expected_total or accepted_attempt_count != expected_total:
        raise ValueError("Stage A namespace accepted count mismatch")
    if accepted_attempt_pairs != identifiers["pair"]:
        raise ValueError("Stage A records and accepted journal entries disagree")
    if len(identifiers["noise"]) != expected_total * 6:
        raise ValueError("Stage A detector-noise IDs are incomplete")
    return (
        {
            "status": "passed",
            "split": expected_split.value,
            "dataset_id": expected_dataset,
            "accepted_pair_count": expected_total,
            "complete_shard_count": expected_shards,
            "pairs_per_shard": pairs_per_shard,
            "attempt_count": attempt_count,
            "generator_commit": generator_commit,
            "configuration_hash": expected_config_hash,
            "proposal_equals_evaluation": True,
            "all_importance_weights_one": True,
            "family_counts": dict(family_counts),
            "em_cell_counts": dict(em_cell_counts),
            "pair_ids_sha256": hashlib.sha256(
                canonical_json(sorted(identifiers["pair"])).encode()
            ).hexdigest(),
            "physical_system_ids_sha256": hashlib.sha256(
                canonical_json(sorted(identifiers["system"])).encode()
            ).hexdigest(),
            "shards": shard_artifacts,
        },
        identifiers,
    )
