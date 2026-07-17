#!/usr/bin/env python3
"""Release-gate and atomically publish the single authorized Stage B increment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Set

from gwlens_mm.config import load_yaml
from gwlens_mm.production.qualification import generate_qualification_shard
from gwlens_mm.production.run_control import verify_psd_files
from gwlens_mm.production.stage_a import (
    DIRECT_TARGET_ID,
    validate_stage_a_namespace,
)
from gwlens_mm.production.storage import tree_checksum, verify_complete_shard
from gwlens_mm.provenance import canonical_json, configuration_hash, dataset_id
from gwlens_mm.schema import SplitName, V2Record

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = "configs/data/phase4_direct_target_stage_b.yaml"
APPROVED_REMOTE_ROOT = Path("/root/autodl-tmp/lensing-4")
GROUP_KEYS = ("pair", "source", "lens", "system", "noise")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _git_clean_commit(root: Path, expected: str) -> None:
    if (root / ".git").exists():
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != expected:
            raise ValueError("Stage B orchestration checkout commit mismatch")
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if dirty:
            raise ValueError("Stage B orchestration checkout is dirty")
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if branch != "phase4/stage-b-direct-target":
            raise ValueError("Stage B checkout is on the wrong branch")
        return
    marker = root / "SYNCED_COMMIT"
    if not marker.is_file() or marker.read_text().strip() != expected:
        raise ValueError("Stage B disposable checkout marker mismatch")


def _load_contract(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    config = load_yaml(ROOT / config_path)
    if config.get("phase") != "4" or config.get("stage") != (
        "stage_b_direct_target_extension"
    ):
        raise ValueError("Stage B identity is absent")
    preregistration = load_yaml(ROOT / config["preregistration"]["path"])
    adaptive = load_yaml(ROOT / config["adaptive_preregistration"]["path"])
    parent = load_yaml(ROOT / config["parent_scientific_preregistration"]["path"])
    for loaded, specification, label in (
        (preregistration, config["preregistration"], "RC.4"),
        (adaptive, config["adaptive_preregistration"], "RC.3"),
        (parent, config["parent_scientific_preregistration"], "RC.5"),
    ):
        if configuration_hash(loaded) != specification["canonical_hash"]:
            raise ValueError(f"Stage B {label} canonical hash mismatch")
    decision_path = ROOT / config["learning_curve_evidence"]["path"]
    if _sha256(decision_path) != config["learning_curve_evidence"]["sha256"]:
        raise ValueError("Stage B learning-curve evidence hash mismatch")
    decision = json.loads(decision_path.read_text())
    if decision.get("decision") != "continue_to_train_65k":
        raise PermissionError("Stage B lacks the frozen continuation decision")
    direct = config["direct_target_density_implementation"]
    if not (
        direct["evaluation_target_id"] == DIRECT_TARGET_ID
        and direct["mode"] == "evaluation_target_direct"
        and direct["proposal_equals_evaluation"] is True
        and float(direct["log_importance_weight"]) == 0.0
        and float(direct["importance_weight"]) == 1.0
    ):
        raise ValueError("Stage B direct-target contract changed")
    stage = config["stage_b"]
    if (
        int(stage["accepted_pair_count"]),
        int(stage["shard_count"]),
        int(stage["pairs_per_shard"]),
        int(config["cumulative_train_contract"]["accepted_physical_system_count"]),
    ) != (32768, 256, 128, 65536):
        raise ValueError("Stage B exact-count contract changed")
    if int(stage["shard_count"]) * int(stage["pairs_per_shard"]) != int(
        stage["accepted_pair_count"]
    ):
        raise ValueError("Stage B shard arithmetic failed")
    boundaries = config["authorization_boundaries"]
    if any(value is not False for value in boundaries.values()):
        raise ValueError("Stage B design config opened an execution boundary")
    return config


def _load_authorization(config: Mapping[str, Any]) -> Dict[str, Any]:
    authorization = load_yaml(ROOT / str(config["authorization_path"]))
    if authorization.get("authorization_status") != (
        "authorized_exact_stage_b_materialization_only"
    ):
        raise PermissionError("Stage B exact-count authorization is absent")
    implementation_commit = authorization.get("implementation_commit")
    if not isinstance(implementation_commit, str) or len(implementation_commit) != 40:
        raise ValueError("Stage B authorization implementation commit is unresolved")
    if authorization.get("immutable_generator", {}).get("git_commit") != config[
        "release"
    ]["generator_commit"]:
        raise ValueError("Stage B authorization generator commit mismatch")
    if authorization.get("preregistration", {}).get("canonical_hash") != config[
        "preregistration"
    ]["canonical_hash"]:
        raise ValueError("Stage B authorization preregistration hash mismatch")
    counts = authorization.get("stage_b_contract", {})
    if (
        counts.get("additional_train_accepted_count"),
        counts.get("shard_count"),
        counts.get("pairs_per_shard"),
        counts.get("cumulative_train_accepted_count"),
    ) != (32768, 256, 128, 65536):
        raise ValueError("Stage B authorization count mismatch")
    flags = authorization.get("authorization", {})
    for key in (
        "stage_b_scientific_materialization_authorized",
        "accepted_pair_generator_authorized_within_stage_b_only",
    ):
        if flags.get(key) is not True:
            raise PermissionError(f"Stage B requires {key}=true")
    for key in (
        "train_65k_optimizer_authorized",
        "model_tuning_authorized",
        "calibration_authorized",
        "sbc_authorized",
        "final_evaluation_authorized",
        "extension_above_65536_authorized",
        "real_noise_authorized",
        "gwosc_gwtc_access_authorized",
    ):
        if flags.get(key) is not False:
            raise PermissionError(f"Stage B requires {key}=false")
    return authorization


def _verify_release_lineage(
    config: Mapping[str, Any], authorization: Mapping[str, Any], release_commit: str
) -> None:
    """Permit one authorization-only descendant of the frozen implementation."""

    implementation_commit = str(authorization["implementation_commit"])
    if (ROOT / ".git").exists():
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", implementation_commit, release_commit],
            cwd=ROOT,
            check=False,
        )
        if ancestry.returncode != 0:
            raise ValueError("Stage B release does not descend from its implementation")
        changed = set(
            subprocess.run(
                ["git", "diff", "--name-only", f"{implementation_commit}..{release_commit}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        )
        allowed = {str(config["authorization_path"])}
        if changed != allowed:
            raise ValueError("Stage B post-implementation changes are not authorization-only")


def _build_namespace_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    stage = config["stage_b"]
    base = load_yaml(ROOT / str(config["base_data_config"]))
    base.update(
        {
            "phase": "4-stage-b",
            "root_seed": int(stage["root_seed"]),
            "dataset_purpose": str(config["dataset_purpose"]),
            "accepted_pair_count": int(stage["accepted_pair_count"]),
            "shard_count": int(stage["shard_count"]),
            "pairs_per_shard": int(stage["pairs_per_shard"]),
            "production_context": {
                "proposal_mode": "evaluation_target_direct",
                "proposal_distribution_id": DIRECT_TARGET_ID,
                "evaluation_distribution_id": DIRECT_TARGET_ID,
                "attempt_stream_namespace": str(stage["attempt_stream_namespace"]),
                "id_prefix": str(stage["id_prefix"]),
                "split": "train",
                "canary": False,
            },
        }
    )
    base["execution"] = {
        **base["execution"],
        "qualification_worker_processes": int(config["execution"]["worker_processes"]),
        "attempt_id_stride": int(stage["attempt_id_stride"]),
        "maximum_attempts_per_worker": int(
            config["execution"]["maximum_attempts_per_worker"]
        ),
        "maximum_active_seconds_per_worker": int(
            config["execution"]["maximum_active_seconds_per_worker"]
        ),
    }
    return base


def _identities(
    config: Mapping[str, Any], orchestration_commit: str
) -> Dict[str, str]:
    generator_commit = str(config["release"]["generator_commit"])
    config_hash = configuration_hash(config)
    seed = int(config["stage_b"]["root_seed"])
    parent = f"phase4-stage-b-{generator_commit[:12]}-{config_hash[:12]}"
    train = dataset_id("2.0.0-alpha.3", generator_commit, config_hash, seed) + (
        "-train-extension"
    )
    combined = f"phase4-train-65k-{generator_commit[:12]}-{config_hash[:12]}"
    return {
        "parent_run_id": parent,
        "train_dataset_id": train,
        "combined_train_id": combined,
        "orchestration_commit": orchestration_commit,
    }


def _validate_paths(config: Mapping[str, Any]) -> None:
    for value in config["paths"].values():
        path = Path(str(value))
        if not path.is_absolute() or not path.is_relative_to(APPROVED_REMOTE_ROOT):
            raise ValueError("Stage B path escaped the AutoDL project root")


def evaluate_release_gate(
    *, config_path: str, orchestration_commit: str
) -> Dict[str, Any]:
    blockers: list[str] = []
    checks: Dict[str, Any] = {}
    try:
        config = _load_contract(config_path)
        authorization = _load_authorization(config)
        _git_clean_commit(ROOT, orchestration_commit)
        _verify_release_lineage(config, authorization, orchestration_commit)
        _validate_paths(config)
        checks["static_contract"] = "passed"
    except Exception as error:
        return {
            "status": "blocked_preexecution",
            "blockers": [str(error)],
            "checks": checks,
            "official_identities": None,
        }
    release = config["release"]
    wheel_path = Path(str(release["generator_wheel_path"]))
    try:
        actual = _sha256(wheel_path)
        checks["generator_wheel_sha256"] = actual
        if actual != release["generator_wheel_sha256"]:
            blockers.append("Stage B frozen generator wheel hash mismatch")
    except Exception as error:
        blockers.append(f"Stage B frozen generator wheel unavailable: {error}")
    lock_path = ROOT / str(config["environment"]["dependency_lock_path"])
    lock_hash = _sha256(lock_path)
    checks["environment_lock_sha256"] = lock_hash
    if lock_hash != release["environment_lock_sha256"]:
        blockers.append("Stage B environment lock hash mismatch")
    stage_a = config["stage_a_reference"]
    stage_a_manifest = Path(str(stage_a["parent_manifest_path"]))
    try:
        actual = _sha256(stage_a_manifest)
        checks["stage_a_parent_manifest_sha256"] = actual
        if actual != stage_a["parent_manifest_sha256"]:
            blockers.append("Stage A parent manifest hash mismatch")
        manifest = json.loads(stage_a_manifest.read_text())
        if (
            manifest.get("status"),
            manifest.get("train_accepted_pair_count"),
            manifest.get("validation_accepted_pair_count"),
        ) != ("passed", 32768, 6144):
            blockers.append("Stage A parent manifest is not the accepted publication")
    except Exception as error:
        blockers.append(f"Stage A publication unavailable: {error}")
    try:
        base = load_yaml(ROOT / str(config["base_data_config"]))
        checks["psd_files"] = verify_psd_files(base["gw"]["psd_curves"])
    except Exception as error:
        blockers.append(f"Stage B PSD verification failed: {error}")
    staging = Path(str(config["paths"]["staging_root"]))
    publication = Path(str(config["paths"]["publication_root"]))
    staging.parent.mkdir(parents=True, exist_ok=True)
    publication.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(staging.parent).free
    checks["free_bytes"] = free
    if free < int(config["resource_gates"]["minimum_prelaunch_free_bytes"]):
        blockers.append("Stage B free-space gate failed")
    identities = _identities(config, orchestration_commit)
    if (publication / identities["parent_run_id"]).exists():
        blockers.append("Stage B official parent identity already exists")
    combined = Path(str(config["paths"]["combined_publication_root"]))
    if (combined / identities["combined_train_id"]).exists():
        blockers.append("Stage B combined 65k identity already exists")
    checks["authorization_status"] = authorization["authorization_status"]
    status = "ready_for_official_execution" if not blockers else "blocked_preexecution"
    return {
        "status": status,
        "phase": "4-stage-b",
        "generator_commit": release["generator_commit"],
        "orchestration_commit": orchestration_commit,
        "configuration_hash": configuration_hash(config),
        "preregistration_hash": config["preregistration"]["canonical_hash"],
        "checks": checks,
        "blockers": blockers,
        "official_identities": identities if not blockers else None,
        "accepted_target": 32768,
        "cumulative_train_target": 65536,
        "train_65k_optimizer_authorized": False,
        "calibration_authorized": False,
        "final_evaluation_authorized": False,
        "gwosc_gwtc_access_authorized": False,
    }


def _generate_pending(
    *,
    stage: Path,
    namespace_config: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    proposal: Mapping[str, Any],
    generator_commit: str,
    dataset_identity: str,
) -> None:
    pending: list[int] = []
    pairs = int(namespace_config["pairs_per_shard"])
    for shard_index in range(int(namespace_config["shard_count"])):
        shard = stage / "shards" / f"shard-{shard_index:05d}"
        if shard.exists():
            verify_complete_shard(shard, pairs)
        else:
            pending.append(shard_index)
    if not pending:
        return
    workers = min(
        int(namespace_config["execution"]["qualification_worker_processes"]),
        len(pending),
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                generate_qualification_shard,
                shard_index=shard_index,
                stage=stage,
                config=namespace_config,
                preregistration=preregistration,
                generator_git_commit=generator_commit,
                dataset_id=dataset_identity,
                proposal_config=proposal,
            ): shard_index
            for shard_index in pending
        }
        for future in as_completed(futures):
            future.result()


def _published_group_ids(dataset_root: Path, shard_count: int) -> Dict[str, Set[str]]:
    pandas = __import__("pandas")
    identifiers: Dict[str, Set[str]] = {key: set() for key in GROUP_KEYS}
    for shard_index in range(shard_count):
        frame = pandas.read_parquet(
            dataset_root / "shards" / f"shard-{shard_index:05d}" / "records.parquet",
            columns=["record_json"],
        )
        for raw in frame["record_json"]:
            record = V2Record.from_json(str(raw))
            values = {
                "pair": (record.pair.pair_id,),
                "source": (record.pair.source_id,),
                "lens": (record.pair.lens_id,),
                "system": (record.pair.physical_system_id,),
                "noise": tuple(record.provenance.used_noise_segment_ids),
            }
            for key, group in values.items():
                if identifiers[key].intersection(group):
                    raise ValueError(f"published parent contains duplicate {key} ID")
                identifiers[key].update(group)
    return identifiers


def _cross_component_validation(
    config: Mapping[str, Any], stage_b_ids: Mapping[str, Set[str]]
) -> Dict[str, Any]:
    stage_a = config["stage_a_reference"]
    parent = Path(str(stage_a["publication_path"]))
    roots = {
        "stage_a_train": (
            parent / str(stage_a["train_dataset_id"]),
            256,
        ),
        "stage_a_validation": (
            parent / str(stage_a["validation_dataset_id"]),
            48,
        ),
    }
    prior = {
        name: _published_group_ids(root, count) for name, (root, count) in roots.items()
    }
    for name, identifiers in prior.items():
        for key in GROUP_KEYS:
            if identifiers[key] & stage_b_ids[key]:
                raise ValueError(f"Stage B {key} leakage with {name}")
    combined: Dict[str, Any] = {}
    for key in GROUP_KEYS:
        train_union = prior["stage_a_train"][key] | stage_b_ids[key]
        expected = 65536 * (6 if key == "noise" else 1)
        if len(train_union) != expected:
            raise ValueError(f"combined 65k {key} count mismatch")
        combined[f"{key}_count"] = len(train_union)
        combined[f"{key}_ids_sha256"] = hashlib.sha256(
            canonical_json(sorted(train_union)).encode()
        ).hexdigest()
    combined["stage_a_stage_b_group_disjoint"] = True
    combined["stage_b_validation_group_disjoint"] = True
    return combined


def _append_segment(manifest: MutableMapping[str, Any], segment: Mapping[str, Any]) -> None:
    segments = manifest.setdefault("execution_segments", [])
    if not isinstance(segments, list):
        raise ValueError("Stage B execution-segment manifest is invalid")
    segments.append(dict(segment))


def execute(
    *, config_path: str, orchestration_commit: str, certificate_path: Path, output: Path
) -> Dict[str, Any]:
    config = _load_contract(config_path)
    authorization = _load_authorization(config)
    _git_clean_commit(ROOT, orchestration_commit)
    _verify_release_lineage(config, authorization, orchestration_commit)
    certificate = json.loads(certificate_path.read_text())
    if certificate.get("status") != "ready_for_official_execution":
        raise PermissionError("Stage B release certificate is not ready")
    if certificate.get("orchestration_commit") != orchestration_commit:
        raise ValueError("Stage B certificate orchestration commit mismatch")
    if certificate.get("configuration_hash") != configuration_hash(config):
        raise ValueError("Stage B certificate configuration hash mismatch")
    identities = certificate.get("official_identities")
    if not isinstance(identities, dict):
        raise ValueError("Stage B certificate lacks official identities")
    stage_root = Path(str(config["paths"]["staging_root"])) / str(
        identities["parent_run_id"]
    )
    publication = Path(str(config["paths"]["publication_root"])) / str(
        identities["parent_run_id"]
    )
    dataset_root = stage_root / str(identities["train_dataset_id"])
    published_dataset = publication / str(identities["train_dataset_id"])
    if publication.exists():
        work_root = published_dataset
    else:
        for child in ("attempts", "shards", "validation", "environment"):
            (dataset_root / child).mkdir(parents=True, exist_ok=True)
        work_root = dataset_root
    namespace_config = _build_namespace_config(config)
    run_manifest_path = (publication if publication.exists() else stage_root) / (
        "run_manifest.json"
    )
    if run_manifest_path.exists():
        run_manifest: MutableMapping[str, Any] = json.loads(run_manifest_path.read_text())
    else:
        run_manifest = {
            "status": "generating_or_resuming",
            "parent_run_id": identities["parent_run_id"],
            "dataset_id": identities["train_dataset_id"],
            "generator_commit": config["release"]["generator_commit"],
            "orchestration_commit": orchestration_commit,
            "configuration_hash": configuration_hash(config),
            "namespace_configuration_hash": configuration_hash(namespace_config),
            "authorization_basis_commit": authorization["implementation_commit"],
            "accepted_target": 32768,
            "cumulative_train_target": 65536,
            "resume_strategy": "verify and preserve every complete atomic shard",
            "train_65k_optimizer_authorized": False,
            "calibration_authorized": False,
            "final_evaluation_authorized": False,
            "gwosc_gwtc_access_authorized": False,
            "execution_segments": [],
        }
    segment: Dict[str, Any] = {"started_at": _utc_now(), "status": "running"}
    _append_segment(run_manifest, segment)
    _atomic_json(run_manifest_path, run_manifest)
    started = time.monotonic()
    parent_preregistration = load_yaml(
        ROOT / str(config["parent_scientific_preregistration"]["path"])
    )
    proposal = load_yaml(ROOT / str(config["direct_target_density_implementation"]["path"]))
    if not publication.exists():
        _generate_pending(
            stage=dataset_root,
            namespace_config=namespace_config,
            preregistration=parent_preregistration,
            proposal=proposal,
            generator_commit=str(config["release"]["generator_commit"]),
            dataset_identity=str(identities["train_dataset_id"]),
        )
    validation, stage_b_ids = validate_stage_a_namespace(
        work_root,
        namespace_config=namespace_config,
        expected_split=SplitName.TRAIN,
        expected_dataset=str(identities["train_dataset_id"]),
        generator_commit=str(config["release"]["generator_commit"]),
    )
    cross = _cross_component_validation(config, stage_b_ids)
    if publication.exists():
        digest, byte_count = tree_checksum(publication)
    else:
        parent_manifest = {
            "status": "passed",
            "parent_run_id": identities["parent_run_id"],
            "dataset_id": identities["train_dataset_id"],
            "generator_commit": config["release"]["generator_commit"],
            "orchestration_commit": orchestration_commit,
            "preregistration_version": config["preregistration"]["version"],
            "preregistration_hash": config["preregistration"]["canonical_hash"],
            "configuration_hash": configuration_hash(config),
            "accepted_pair_count": 32768,
            "complete_shard_count": 256,
            "validation": validation,
            "cross_component_validation": cross,
            "proposal_equals_evaluation": True,
            "all_importance_weights_one": True,
            "train_65k_optimizer_authorized": False,
            "calibration_authorized": False,
            "final_evaluation_authorized": False,
            "gwosc_gwtc_accessed": False,
        }
        _atomic_json(stage_root / "dataset_manifest.json", parent_manifest)
        run_manifest["status"] = "validated_ready_for_atomic_publication"
        segment.update(
            {
                "finished_at": _utc_now(),
                "active_wall_seconds": time.monotonic() - started,
                "status": "completed",
            }
        )
        _atomic_json(stage_root / "run_manifest.json", run_manifest)
        digest, byte_count = tree_checksum(stage_root)
        if byte_count > int(config["resource_gates"]["maximum_stage_b_output_bytes"]):
            raise RuntimeError("Stage B output exceeded the frozen byte cap")
        Path(str(config["paths"]["publication_root"])).mkdir(parents=True, exist_ok=True)
        os.replace(stage_root, publication)
    post_free = shutil.disk_usage(publication).free
    if post_free < int(config["resource_gates"]["minimum_post_peak_free_bytes"]):
        raise RuntimeError("Stage B post-publication free-space gate failed")
    stage_b_manifest_path = publication / "dataset_manifest.json"
    combined_manifest = {
        "status": "passed",
        "combined_train_id": identities["combined_train_id"],
        "accepted_physical_system_count": 65536,
        "validation_physical_system_count": 6144,
        "components": [
            {
                "role": "stage_a_train",
                "parent_run_id": config["stage_a_reference"]["parent_run_id"],
                "dataset_id": config["stage_a_reference"]["train_dataset_id"],
                "accepted_count": 32768,
                "parent_manifest_sha256": config["stage_a_reference"][
                    "parent_manifest_sha256"
                ],
            },
            {
                "role": "stage_b_train_extension",
                "parent_run_id": identities["parent_run_id"],
                "dataset_id": identities["train_dataset_id"],
                "accepted_count": 32768,
                "parent_manifest_sha256": _sha256(stage_b_manifest_path),
                "publication_tree_sha256": digest,
                "publication_bytes": byte_count,
            },
        ],
        "group_validation": cross,
        "strict_nested_train_ladder": True,
        "proposal_equals_evaluation": True,
        "all_importance_weights_one": True,
        "training_authorized": False,
        "calibration_authorized": False,
        "sbc_authorized": False,
        "final_evaluation_authorized": False,
        "extension_above_65536_authorized": False,
        "gwosc_gwtc_accessed": False,
    }
    combined_partial = Path(str(config["paths"]["combined_staging_root"])) / str(
        identities["combined_train_id"]
    )
    combined_published = Path(str(config["paths"]["combined_publication_root"])) / str(
        identities["combined_train_id"]
    )
    if not combined_published.exists():
        combined_partial.mkdir(parents=True, exist_ok=False)
        _atomic_json(combined_partial / "dataset_manifest.json", combined_manifest)
        combined_published.parent.mkdir(parents=True, exist_ok=True)
        os.replace(combined_partial, combined_published)
    elif _sha256(combined_published / "dataset_manifest.json") != hashlib.sha256(
        (json.dumps(combined_manifest, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest():
        raise ValueError("existing combined 65k manifest differs from deterministic result")
    result = {
        "status": "passed",
        "parent_run_id": identities["parent_run_id"],
        "train_dataset_id": identities["train_dataset_id"],
        "combined_train_id": identities["combined_train_id"],
        "generator_commit": config["release"]["generator_commit"],
        "orchestration_commit": orchestration_commit,
        "accepted_pair_count": 32768,
        "complete_shard_count": 256,
        "cumulative_train_accepted_pair_count": 65536,
        "stage_b_publication_path": str(publication),
        "stage_b_publication_tree_sha256": digest,
        "stage_b_publication_bytes": byte_count,
        "combined_manifest_path": str(combined_published / "dataset_manifest.json"),
        "combined_manifest_sha256": _sha256(
            combined_published / "dataset_manifest.json"
        ),
        "remaining_free_bytes": post_free,
        "proposal_equals_evaluation": True,
        "all_importance_weights_one": True,
        "cross_component_validation": cross,
        "train_65k_optimizer_authorized": False,
        "calibration_authorized": False,
        "sbc_authorized": False,
        "final_evaluation_authorized": False,
        "gwosc_gwtc_accessed": False,
    }
    _atomic_json(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "execute"))
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--orchestration-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--release-certificate", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.mode == "preflight":
            result = evaluate_release_gate(
                config_path=args.config,
                orchestration_commit=args.orchestration_commit,
            )
            _atomic_json(args.output, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] == "ready_for_official_execution" else 2
        if args.release_certificate is None:
            raise ValueError("execute mode requires --release-certificate")
        result = execute(
            config_path=args.config,
            orchestration_commit=args.orchestration_commit,
            certificate_path=args.release_certificate,
            output=args.output,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        _atomic_json(
            args.output,
            {
                "status": "execution_failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "train_65k_optimizer_authorized": False,
                "calibration_authorized": False,
                "final_evaluation_authorized": False,
                "gwosc_gwtc_accessed": False,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
