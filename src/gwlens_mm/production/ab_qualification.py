"""Fail-closed control and analysis for the bounded Phase 3C-A A/B run."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from ..arrays import validate_strain_array_semantics
from ..config import load_yaml
from ..policy import InputPolicy
from ..provenance import canonical_json, configuration_hash, dataset_id
from ..schema import SplitName, V2Record
from .run_control import AttemptRecord, verify_psd_files
from .storage import tree_checksum, verify_complete_shard

ARM_NAMES = ("rc5_control", "proposal_v3_candidate")
FINAL_STATES = {
    "passed_for_future_scientific_production_review",
    "failed_retain_rc5",
    "inconclusive_retain_rc5",
    "invalid_control_or_environment",
    "execution_failed",
}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _false(mapping: Mapping[str, Any], key: str) -> None:
    if mapping.get(key) is not False:
        raise ValueError(f"Phase 3C-A requires {key}=false")


@dataclass(frozen=True)
class ABIdentity:
    authorizing_commit: str
    generator_commit: str
    parent_run_id: str
    control_dataset_id: str
    candidate_dataset_id: str
    configuration_hash: str


def load_and_verify_contract(
    root: Path, generator_commit: str
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], ABIdentity]:
    config = load_yaml(root / "configs/data/phase3ca_proposal_v3_ab.yaml")
    authorization = load_yaml(root / config["authorization"]["path"])
    adaptive = load_yaml(root / config["adaptive_preregistration"]["path"])
    proposal = load_yaml(root / config["proposal"]["path"])
    if (
        authorization.get("phase") != "3C-A"
        or authorization.get("authorization_status") != "authorized_engineering_ab_only"
    ):
        raise ValueError("Phase 3C-A authorization is absent")
    if authorization.get("base_main_commit") != "80367373b92065d049db4d9576201c186ef78623":
        raise ValueError("Phase 3C-A base merge identity changed")
    if configuration_hash(adaptive) != config["adaptive_preregistration"]["canonical_hash"]:
        raise ValueError("adaptive RC.3 canonical hash mismatch")
    if configuration_hash(proposal) != config["proposal"]["canonical_hash"]:
        raise ValueError("proposal-v3 canonical hash mismatch")
    if (
        authorization["candidate"]["implementation_commit"]
        != "9e154addd8634db6f0a91ccdf9f6f95339264405"
    ):
        raise ValueError("proposal-v3 implementation identity changed")
    denied = authorization["authorization"]
    for key in (
        "scientific_data_generation_authorized",
        "model_training_authorized",
        "model_tuning_authorized",
        "calibration_authorized",
        "sbc_authorized",
        "iid_ood_mismatch_evaluation_authorized",
        "real_noise_authorized",
        "gwosc_gwtc_access_authorized",
        "stage_a_authorized",
        "phase3c_b_or_later_authorized",
    ):
        _false(denied, key)
    contract = authorization["ab_contract"]
    if (
        contract["accepted_pairs_per_arm"],
        contract["total_accepted_pairs"],
        contract["blocks_per_arm"],
        contract["accepted_pairs_per_block"],
    ) != (512, 1024, 16, 32):
        raise ValueError("A/B accepted-count contract changed")
    if config["accepted_pairs_per_arm"] * config["arm_count"] != config["total_accepted_pairs"]:
        raise ValueError("A/B configuration count arithmetic failed")
    if len(generator_commit) != 40:
        raise ValueError("generator commit must be a full Git SHA")
    if (root / ".git").exists():
        base_commit = str(authorization["base_main_commit"])
        base_available = subprocess.run(
            ["git", "cat-file", "-e", f"{base_commit}^{{commit}}"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if base_available.returncode == 0:
            ancestry = subprocess.run(
                ["git", "merge-base", "--is-ancestor", base_commit, generator_commit],
                cwd=root,
                check=False,
            )
            if ancestry.returncode:
                raise ValueError("generator commit does not descend from the reviewed base")
    config_hash = configuration_hash(config)
    parent = f"phase3ca-{generator_commit[:12]}-{config_hash[:12]}"
    control = (
        dataset_id(
            "2.0.0-alpha.3",
            generator_commit,
            config_hash,
            int(config["arms"]["rc5_control"]["root_seed"]),
        )
        + "-control"
    )
    candidate = (
        dataset_id(
            "2.0.0-alpha.3",
            generator_commit,
            config_hash,
            int(config["arms"]["proposal_v3_candidate"]["root_seed"]),
        )
        + "-v3"
    )
    if control == candidate:
        raise ValueError("A/B arm dataset identities collide")
    identity = ABIdentity(
        config["authorization"]["authorizing_git_commit"],
        generator_commit,
        parent,
        control,
        candidate,
        config_hash,
    )
    return config, adaptive, proposal, identity


def arm_config(root: Path, config: Mapping[str, Any], arm: str) -> Dict[str, Any]:
    base = load_yaml(root / config["base_data_config"])
    arm_spec = config["arms"][arm]
    base.update(
        {
            "phase": "3C-A",
            "root_seed": int(arm_spec["root_seed"]),
            "dataset_purpose": config["dataset_purpose"],
            "accepted_pair_count": 512,
            "shard_count": 16,
            "pairs_per_shard": 32,
            "engineering_ab": {
                **arm_spec,
                "arm": arm,
                "evaluation_distribution_id": config["evaluation_distribution_id"],
            },
        }
    )
    base["execution"] = {
        **base["execution"],
        "qualification_worker_processes": 1,
        "attempt_id_stride": 16,
        "maximum_attempts_per_worker": int(config["execution"]["maximum_attempts_per_arm"]),
        "maximum_active_seconds_per_worker": int(
            config["execution"]["maximum_active_seconds_per_arm"]
        ),
    }
    return base


def preflight(root: Path, config: Mapping[str, Any], identity: ABIdentity) -> Dict[str, Any]:
    paths = {name: Path(value) for name, value in config["paths"].items()}
    approved = Path("/root/autodl-tmp/lensing-4")
    for path in paths.values():
        if not path.is_absolute() or not path.is_relative_to(approved):
            raise ValueError("Phase 3C-A output path escaped the approved project root")
        path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(paths["staging_root"])
    required = int(config["resource_gates"]["minimum_prelaunch_free_bytes"])
    if usage.free < required:
        raise RuntimeError(f"free-space gate failed: {usage.free} < {required}")
    base = load_yaml(root / config["base_data_config"])
    psd = verify_psd_files(base["gw"]["psd_curves"])
    parent_stage = paths["staging_root"] / identity.parent_run_id
    parent_publication = paths["publication_root"] / identity.parent_run_id
    if parent_publication.exists():
        raise FileExistsError("A/B parent publication identity already exists")
    return {
        "status": "passed",
        "parent_run_id": identity.parent_run_id,
        "generator_commit": identity.generator_commit,
        "configuration_hash": identity.configuration_hash,
        "free_bytes_before": usage.free,
        "required_free_bytes": required,
        "projected_peak_bytes": int(config["resource_gates"]["projected_peak_bytes"]),
        "psd_files": psd,
        "parent_stage": str(parent_stage),
        "parent_publication": str(parent_publication),
        "scientific_use_authorized": False,
        "training_use_authorized": False,
        "gwosc_gwtc_accessed": False,
    }


def relative_ess(weights: Any) -> float:
    values = np.asarray(weights, dtype=np.float64)
    return float(np.sum(values) ** 2 / (len(values) * np.sum(values * values)))


def bootstrap_throughput(
    blocks: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Dict[str, Any]:
    if len(blocks) != 32:
        raise ValueError("throughput analysis requires all 32 arm blocks")
    by_key = {(str(row["arm"]), int(row["block_index"])): row for row in blocks}
    control = np.array(
        [float(by_key[("rc5_control", i)]["active_wall_seconds"]) for i in range(16)]
    )
    candidate = np.array(
        [float(by_key[("proposal_v3_candidate", i)]["active_wall_seconds"]) for i in range(16)]
    )
    count = float(config["accepted_pairs_per_block"])
    point = (16 * count / np.sum(candidate)) / (16 * count / np.sum(control))
    rng = np.random.default_rng(int(config["bootstrap"]["seed"]))
    ratios = np.empty(int(config["bootstrap"]["replicates"]))
    for index in range(len(ratios)):
        selected = rng.integers(0, 16, size=16)
        ratios[index] = np.sum(control[selected]) / np.sum(candidate[selected])
    lower, upper = np.exp(np.quantile(np.log(ratios), (0.025, 0.975)))
    return {
        "replicates": len(ratios),
        "seed": int(config["bootstrap"]["seed"]),
        "seed_domain": config["bootstrap"]["seed_domain"],
        "point_estimate": float(point),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "minimum_required_lower_95": 2.0,
        "passed": bool(lower >= 2.0),
    }


def validate_arm(
    stage: Path,
    arm_cfg: Mapping[str, Any],
    identity: ABIdentity,
    arm: str,
    input_policy: InputPolicy,
) -> Dict[str, Any]:
    pd = importlib.import_module("pandas")
    zarr = importlib.import_module("zarr")

    pair_ids: set[str] = set()
    source_ids: set[str] = set()
    lens_ids: set[str] = set()
    system_ids: set[str] = set()
    noise_ids: set[str] = set()
    attempts = 0
    accepted_attempts = 0
    weights: list[float] = []
    families: list[str] = []
    cells: list[str] = []
    components: list[str] = []
    source_radii: list[float] = []
    theta_values: list[float] = []
    z_lenses: list[float] = []
    multiplicities: Counter[int] = Counter()
    rejection_counts: Counter[str] = Counter()
    artifacts = []
    expected_dataset = (
        identity.control_dataset_id if arm == "rc5_control" else identity.candidate_dataset_id
    )
    for block in range(16):
        shard = stage / "shards" / f"shard-{block:05d}"
        verify_complete_shard(shard, 32)
        digest, size = tree_checksum(shard)
        artifacts.append({"path": f"shards/shard-{block:05d}", "sha256": digest, "bytes": size})
        frame = pd.read_parquet(shard / "records.parquet")
        arrays = {
            name: zarr.open_array(str(shard / f"{name}.zarr"), mode="r")
            for name in ("noisy", "clean", "noise")
        }
        for row_index, row in frame.iterrows():
            record = V2Record.from_json(str(row["record_json"]))
            record.validate()
            if (
                record.pair.split is not SplitName.GENERATOR_QUALIFICATION
                or record.pair.dataset_version != expected_dataset
            ):
                raise ValueError("A/B record identity or non-scientific split mismatch")
            if (
                record.provenance.generator_git_commit != identity.generator_commit
                or record.provenance.configuration_hash != configuration_hash(arm_cfg)
            ):
                raise ValueError("A/B record mixes code or arm configurations")
            validate_strain_array_semantics(
                np.asarray(arrays["noisy"][row_index]),
                np.asarray(arrays["clean"][row_index]),
                np.asarray(arrays["noise"][row_index]),
                record.gw_observation.detector_availability_mask,
            )
            ids = (
                record.pair.pair_id,
                record.pair.source_id,
                record.pair.lens_id,
                record.pair.physical_system_id,
            )
            for value, target in zip(ids, (pair_ids, source_ids, lens_ids, system_ids)):
                if value in target:
                    raise ValueError("duplicate A/B group ID")
                target.add(value)
            for value in record.provenance.used_noise_segment_ids:
                if value in noise_ids:
                    raise ValueError("duplicate A/B noise ID")
                noise_ids.add(value)
            weight = float(record.provenance.distribution.importance_weight)
            if not math.isfinite(weight):
                raise ValueError("nonfinite A/B importance weight")
            weights.append(weight)
            families.append(record.pair.lens_family.value)
            cells.append(str(row["em_cell"]))
            multiplicities[len(record.lens_truth.physical_images)] += 1
            theta = float(record.lens_truth.lens_parameters["einstein_radius_arcsec"])
            source_radii.append(
                math.hypot(
                    float(record.lens_truth.lens_parameters["source_beta_x_arcsec"]),
                    float(record.lens_truth.lens_parameters["source_beta_y_arcsec"]),
                )
                / theta
            )
            theta_values.append(theta)
            z_lenses.append(float(record.lens_truth.lens_parameters["z_lens"]))
        journal = stage / "attempts" / f"shard-{block:05d}.jsonl"
        for line in journal.read_text().splitlines():
            attempt = AttemptRecord(**json.loads(line))
            attempt.validate()
            attempts += 1
            if attempt.proposal_component is None or attempt.log_importance_weight is None:
                raise ValueError("A/B attempt omits exact proposal provenance")
            if attempt.status == "accepted":
                accepted_attempts += 1
                components.append(attempt.proposal_component)
            else:
                rejection_counts[str(attempt.rejection_reason)] += 1
    if len(pair_ids) != 512 or accepted_attempts != 512 or len(noise_ids) != 3072:
        raise ValueError("A/B arm count contract failed")
    input_policy.validate_model_inputs(tuple(sorted(input_policy.allowlist)))
    return {
        "status": "passed",
        "arm": arm,
        "dataset_id": expected_dataset,
        "accepted_pair_count": len(pair_ids),
        "attempt_count": attempts,
        "active_block_count": 16,
        "weights": weights,
        "families": families,
        "em_cells": cells,
        "accepted_components": components,
        "source_radii": source_radii,
        "einstein_radii": theta_values,
        "lens_redshifts": z_lenses,
        "multiplicity_counts": dict(multiplicities),
        "rejection_counts": dict(rejection_counts),
        "artifacts": artifacts,
        "id_hashes": {
            "pair": hashlib.sha256(canonical_json(sorted(pair_ids)).encode()).hexdigest(),
            "source": hashlib.sha256(canonical_json(sorted(source_ids)).encode()).hexdigest(),
            "lens": hashlib.sha256(canonical_json(sorted(lens_ids)).encode()).hexdigest(),
            "system": hashlib.sha256(canonical_json(sorted(system_ids)).encode()).hexdigest(),
            "noise": hashlib.sha256(canonical_json(sorted(noise_ids)).encode()).hexdigest(),
        },
        "scientific_use_authorized": False,
        "training_use_authorized": False,
        "calibration_use_authorized": False,
        "test_use_authorized": False,
    }


def postselection_diagnostics(
    candidate: Mapping[str, Any], config: Mapping[str, Any]
) -> Dict[str, Any]:
    weights = np.asarray(candidate["weights"], dtype=float)
    families = np.asarray(candidate["families"])
    cells = np.asarray(candidate["em_cells"])
    normalized = weights / np.sum(weights)
    gates = config["postselection_gates"]
    family_ess = {name: relative_ess(weights[families == name]) for name in sorted(set(families))}
    cell_ess = {name: relative_ess(weights[cells == name]) for name in sorted(set(cells))}
    tails = gates["tail_support"]
    tail_checks = {
        "rc5_safety_component": candidate["accepted_components"].count("rc5_broad")
        >= int(tails["minimum_rc5_safety_component_accepted"]),
        "normalized_source_radius": max(candidate["source_radii"])
        >= float(tails["maximum_required_normalized_source_radius_minimum"]),
        "einstein_radius_low": min(candidate["einstein_radii"])
        <= float(tails["maximum_required_einstein_radius_arcsec_minimum"]),
        "einstein_radius_high": max(candidate["einstein_radii"])
        >= float(tails["minimum_required_einstein_radius_arcsec_maximum"]),
        "lens_redshift_low": min(candidate["lens_redshifts"])
        <= float(tails["maximum_required_lens_redshift_minimum"]),
        "lens_redshift_high": max(candidate["lens_redshifts"])
        >= float(tails["minimum_required_lens_redshift_maximum"]),
    }
    checks = {
        "finite": bool(np.all(np.isfinite(weights))),
        "overall_ess": relative_ess(weights) >= float(gates["overall_relative_ess_minimum"]),
        "family_ess": len(family_ess) == 2
        and all(
            value >= float(gates["family_relative_ess_minimum"]) for value in family_ess.values()
        ),
        "em_cell_ess": len(cell_ess) == 8
        and all(
            value >= float(gates["em_cell_relative_ess_minimum"]) for value in cell_ess.values()
        ),
        "maximum_weight": float(np.max(normalized))
        <= float(gates["maximum_single_normalized_weight"]),
        "multiplicity": len(candidate["multiplicity_counts"]) >= 2,
        "tail_support": all(tail_checks.values()),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "overall_relative_ess": relative_ess(weights),
        "relative_ess_by_family": family_ess,
        "relative_ess_by_em_cell": cell_ess,
        "maximum_normalized_weight": float(np.max(normalized)),
        "tail_support_checks": tail_checks,
    }


def publish_arm(
    stage: Path, destination: Path, summary: Mapping[str, Any], identity: ABIdentity
) -> Dict[str, Any]:
    small = {
        key: value
        for key, value in summary.items()
        if key
        not in {
            "weights",
            "families",
            "em_cells",
            "accepted_components",
            "source_radii",
            "einstein_radii",
            "lens_redshifts",
        }
    }
    _atomic_json(stage / "validation" / "arm_validation.json", small)
    manifest = {
        "dataset_id": summary["dataset_id"],
        "parent_run_id": identity.parent_run_id,
        "generator_commit": identity.generator_commit,
        "configuration_hash": identity.configuration_hash,
        "purpose": "proposal_efficiency_engineering_qualification",
        "accepted_pair_count": 512,
        "complete_block_count": 16,
        "artifacts": summary["artifacts"],
        "scientific_use_authorized": False,
        "training_use_authorized": False,
        "calibration_use_authorized": False,
        "test_use_authorized": False,
        "permanent_exclusion_from_all_scientific_splits": True,
    }
    _atomic_json(stage / "manifest.json", manifest)
    digest, byte_count = tree_checksum(stage)
    manifest["tree_sha256_before_manifest_update"] = digest
    manifest["published_bytes_before_manifest_update"] = byte_count
    _atomic_json(stage / "manifest.json", manifest)
    return {
        **manifest,
        "staged_tree_sha256": digest,
        "staged_bytes": byte_count,
        "published_path": str(destination),
    }


def strip_large_validation(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key
        not in {
            "weights",
            "families",
            "em_cells",
            "accepted_components",
            "source_radii",
            "einstein_radii",
            "lens_redshifts",
            "artifacts",
        }
    }
