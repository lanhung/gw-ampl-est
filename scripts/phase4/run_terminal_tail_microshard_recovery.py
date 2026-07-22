#!/usr/bin/env python3
"""Complete the frozen terminal development-tail pool with dynamic microshards.

The scientific contract remains four independently generated, direct-target,
128-case conditional strata.  This runner changes only the physical work
partition from fixed four-case worker shards to 128 one-case atomic
microshards per stratum.  A process pool dynamically schedules those shards,
so a rare attempt stream cannot strand three otherwise idle workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Set, Tuple

import gwlens_mm
from gwlens_mm.config import load_yaml
from gwlens_mm.production.storage import tree_checksum
from gwlens_mm.production.terminal131 import (
    TerminalNamespace,
    build_terminal_namespace_config,
    load_terminal_131k_contract,
    terminal_namespaces,
)
from gwlens_mm.provenance import canonical_json, configuration_hash, dataset_id
from scripts.phase4 import run_terminal_131k as base

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = (
    "configs/execution/"
    "phase4_terminal_131k_tail_microshard_recovery_authorization.yaml"
)
GENERATOR_COMMIT = "a4e6bac014ccd521d510c97593cb1368e826d5eb"
BASE_CONFIG_HASH = "abd07c5b8031a5cc9564531d29d9349f65b0918fafc494767fc912b7e7444ed7"
TRAIN_PARENT_ID = "phase4-terminal-131k-a4e6bac014cc-abd07c5b8031"
TRAIN_DATASET_ID = (
    "gwlens-v2-2.0.0-alpha.3-e592848e725db2c3-train-increment"
)
PARALLEL32_STOP_TREE = (
    "2866e66739aa26f70e560bc8bacb196baccc2406acbcb64719d5b4a2338a253a"
)
_POSTFREEZE_PATHS = {
    "AGENTS.md",
    AUTHORIZATION,
    "docs/DECISIONS.md",
    "docs/FAILURES.md",
    "docs/PROJECT_STATE.md",
    "docs/reports/PHASE4_TERMINAL_TAIL_MICROSHARD_RECOVERY_REPORT.md",
    "results/experiment_registry.csv",
    "results/phase4/terminal_tail_parallel32_resource_stop.json",
}


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
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify_orchestration_identity(root: Path, expected: str) -> None:
    if len(expected) != 40:
        raise ValueError("microshard orchestration commit must be a full SHA")
    if (root / ".git").is_dir():
        head = _git_head(root)
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if dirty:
            raise ValueError("microshard checkout is dirty")
        if head == expected:
            return
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", expected, head],
            cwd=root,
            check=False,
        ).returncode
        if ancestor != 0:
            raise ValueError("microshard orchestration commit is not an ancestor")
        changed = set(
            subprocess.run(
                ["git", "diff", "--name-only", f"{expected}..{head}"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        )
        unexpected = sorted(changed - _POSTFREEZE_PATHS)
        if unexpected:
            raise ValueError(
                "microshard post-freeze protected paths changed: "
                + ", ".join(unexpected)
            )
        return
    marker = root / "SYNCED_ORCHESTRATION_COMMIT"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != expected:
        raise ValueError("microshard checkout lacks its exact commit marker")


def _verify_generator_core_unchanged(original_wheel: Path, recovery_wheel: Path) -> str:
    """Require byte identity for the generator and all production physics code."""

    excluded = {
        "gwlens_mm/production/calibration_sbc.py",
        "gwlens_mm/production/final_evaluation.py",
        "gwlens_mm/production/terminal131.py",
    }
    with zipfile.ZipFile(original_wheel) as original, zipfile.ZipFile(
        recovery_wheel
    ) as recovery:
        frozen = {
            name
            for name in original.namelist()
            if name.startswith("gwlens_mm/")
            and not name.startswith("gwlens_mm/training/")
            and name not in excluded
        }
        recovery_names = set(recovery.namelist())
        missing = sorted(frozen - recovery_names)
        changed = sorted(
            name
            for name in frozen & recovery_names
            if original.read(name) != recovery.read(name)
        )
        payload = {
            name: hashlib.sha256(recovery.read(name)).hexdigest()
            for name in sorted(frozen & recovery_names)
        }
    if missing or changed:
        raise ValueError(
            "microshard recovery changed frozen generator code: "
            + ", ".join((*missing, *changed))
        )
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _microshard_namespace(namespace: TerminalNamespace) -> TerminalNamespace:
    if namespace.balanced_tail_stratum is None:
        raise ValueError("microshard recovery accepts tail namespaces only")
    return replace(
        namespace,
        shard_count=128,
        pairs_per_shard=1,
        id_prefix=f"{namespace.id_prefix}-micro128",
        attempt_namespace=f"{namespace.attempt_namespace}-micro128",
    )


def _tail_namespaces(config: Mapping[str, Any]) -> Tuple[TerminalNamespace, ...]:
    result = tuple(
        _microshard_namespace(namespace)
        for namespace in terminal_namespaces(config)[1:]
    )
    if len(result) != 4:
        raise ValueError("microshard recovery requires four tail namespaces")
    if any(item.shard_count != 128 or item.pairs_per_shard != 1 for item in result):
        raise ValueError("microshard layout drifted")
    if sum(item.accepted_count for item in result) != 512:
        raise ValueError("microshard tail count drifted")
    return result


def _namespace_config(
    config: Mapping[str, Any],
    namespace: TerminalNamespace,
    authorization: Mapping[str, Any],
) -> Dict[str, Any]:
    result = build_terminal_namespace_config(ROOT, config, namespace)
    execution = authorization["microshard_execution_contract"]
    result["execution"] = {
        **result["execution"],
        "qualification_worker_processes": int(execution["worker_processes"]),
        "maximum_active_seconds_per_worker": int(
            execution["maximum_active_seconds_per_microshard"]
        ),
        "maximum_attempts_per_worker": int(
            execution["maximum_attempts_per_microshard"]
        ),
    }
    return result


def _recovery_identities(
    config: Mapping[str, Any], orchestration_commit: str, authorization: Mapping[str, Any]
) -> Dict[str, Any]:
    namespaces = _tail_namespaces(config)
    configs = {
        str(item.balanced_tail_stratum): _namespace_config(
            config, item, authorization
        )
        for item in namespaces
    }
    tail_ids = {
        str(item.balanced_tail_stratum): dataset_id(
            "2.0.0-alpha.3",
            GENERATOR_COMMIT,
            configuration_hash(configs[str(item.balanced_tail_stratum)]),
            item.root_seed,
        )
        + f"-development-tail-{item.balanced_tail_stratum}-micro128"
        for item in namespaces
    }
    identity_payload = {
        "base_configuration_hash": BASE_CONFIG_HASH,
        "generator_commit": GENERATOR_COMMIT,
        "orchestration_commit": orchestration_commit,
        "tail_dataset_ids": tail_ids,
        "tail_shards_per_namespace": 128,
        "tail_pairs_per_shard": 1,
    }
    suffix = hashlib.sha256(canonical_json(identity_payload).encode()).hexdigest()[:12]
    return {
        "tail_parent_id": (
            f"phase4-terminal-tail-micro128-{orchestration_commit[:12]}-{suffix}"
        ),
        "tail_dataset_ids": tail_ids,
        "combined_train_id": (
            f"phase4-train-131k-{orchestration_commit[:12]}-{suffix}"
        ),
        "identity_payload_sha256": hashlib.sha256(
            canonical_json(identity_payload).encode()
        ).hexdigest(),
    }


def _authorization(config: Mapping[str, Any]) -> Dict[str, Any]:
    authorization = load_yaml(ROOT / AUTHORIZATION)
    if authorization.get("authorization_status") not in {
        "implementation_only",
        "authorized_engineering_microshard_recovery_only",
    }:
        raise PermissionError("microshard recovery authorization is absent")
    frozen = authorization.get("frozen_scientific_contract", {})
    if (
        frozen.get("generator_commit") != GENERATOR_COMMIT
        or frozen.get("base_configuration_hash") != BASE_CONFIG_HASH
        or configuration_hash(config) != BASE_CONFIG_HASH
        or frozen.get("preregistration_hash")
        != config["preregistration"]["canonical_hash"]
        or frozen.get("train_parent_id") != TRAIN_PARENT_ID
        or frozen.get("train_dataset_id") != TRAIN_DATASET_ID
    ):
        raise ValueError("microshard recovery changed the frozen scientific contract")
    execution = authorization.get("microshard_execution_contract", {})
    observed = (
        int(execution.get("worker_processes", -1)),
        int(execution.get("namespace_count", -1)),
        int(execution.get("accepted_pairs_per_namespace", -1)),
        int(execution.get("shards_per_namespace", -1)),
        int(execution.get("accepted_pairs_per_shard", -1)),
        int(execution.get("total_accepted_pairs", -1)),
    )
    if observed != (32, 4, 128, 128, 1, 512):
        raise ValueError("microshard count or scheduler contract drifted")
    if int(execution.get("maximum_active_seconds_per_microshard", -1)) != 345600:
        raise ValueError("microshard active-time cap drifted")
    if int(execution.get("maximum_attempts_per_microshard", -1)) != 2000000:
        raise ValueError("microshard attempt cap drifted")
    flags = authorization.get("authorization", {})
    if authorization["authorization_status"] == (
        "authorized_engineering_microshard_recovery_only"
    ):
        if flags.get("tail_microshard_recovery_authorized") is not True:
            raise PermissionError("microshard execution flag is closed")
    for key in (
        "scientific_contract_change_authorized",
        "training_authorized",
        "architecture_selection_authorized",
        "calibration_authorized",
        "sbc_authorized",
        "final_evaluation_authorized",
        "extension_above_131072_authorized",
        "real_noise_authorized",
        "gwosc_gwtc_access_authorized",
    ):
        if flags.get(key) is not False:
            raise PermissionError(f"microshard recovery requires {key}=false")
    return authorization


def evaluate_release_gate(*, orchestration_commit: str) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    blockers: list[str] = []
    try:
        config = load_terminal_131k_contract(ROOT)
        authorization = _authorization(config)
        if authorization.get("authorization_status") != (
            "authorized_engineering_microshard_recovery_only"
        ):
            raise PermissionError("microshard execution is not authorized")
        immutable = authorization["immutable_orchestration"]
        if immutable.get("git_commit") != orchestration_commit:
            raise ValueError("microshard orchestration commit mismatch")
        _verify_orchestration_identity(ROOT, orchestration_commit)
        checks["static_contract"] = "passed"
    except Exception as error:
        return {
            "status": "blocked_preexecution",
            "checks": checks,
            "blockers": [str(error)],
            "official_identities": None,
        }
    train_parent = Path(str(authorization["published_train_increment"]["parent_root"]))
    try:
        checks["train_parent_manifest_sha256"] = _sha256(
            train_parent / "dataset_manifest.json"
        )
        if checks["train_parent_manifest_sha256"] != authorization[
            "published_train_increment"
        ]["parent_manifest_sha256"]:
            blockers.append("published train parent manifest hash mismatch")
        if not (train_parent / TRAIN_DATASET_ID).is_dir():
            blockers.append("published train increment is absent")
    except Exception as error:
        blockers.append(f"published train check failed: {error}")
    try:
        failed = authorization["failed_parallel32_evidence"]
        evidence = Path(str(failed["path"]))
        observed_tree, observed_bytes = tree_checksum(evidence)
        checks["failed_parallel32_tree_sha256"] = observed_tree
        checks["failed_parallel32_bytes"] = observed_bytes
        if (
            observed_tree != failed["tree_sha256"]
            or observed_tree != PARALLEL32_STOP_TREE
            or observed_bytes != int(failed["bytes"])
        ):
            blockers.append("failed parallel32 evidence changed")
    except Exception as error:
        blockers.append(f"failed parallel32 evidence check failed: {error}")
    identities = _recovery_identities(config, orchestration_commit, authorization)
    paths = authorization["paths"]
    for value in paths.values():
        path = Path(str(value))
        if not path.is_absolute() or not path.is_relative_to(base.APPROVED_REMOTE_ROOT):
            blockers.append("microshard path escaped the AutoDL project root")
            break
    try:
        immutable = authorization["immutable_orchestration"]
        generator_wheel = Path(str(immutable["generator_wheel_path"]))
        recovery_wheel = Path(str(immutable["recovery_wheel_path"]))
        checks["generator_wheel_sha256"] = _sha256(generator_wheel)
        checks["recovery_wheel_sha256"] = _sha256(recovery_wheel)
        if checks["generator_wheel_sha256"] != immutable["generator_wheel_sha256"]:
            blockers.append("microshard generator wheel hash mismatch")
        if checks["recovery_wheel_sha256"] != immutable["recovery_wheel_sha256"]:
            blockers.append("microshard recovery wheel hash mismatch")
        checks["generator_core_manifest_sha256"] = (
            _verify_generator_core_unchanged(generator_wheel, recovery_wheel)
        )
        if checks["generator_core_manifest_sha256"] != immutable[
            "generator_core_manifest_sha256"
        ]:
            blockers.append("microshard frozen generator-core manifest mismatch")
        imported = Path(str(gwlens_mm.__file__)).resolve()
        runtime = Path(str(immutable["runtime_site_packages"])).resolve()
        checks["imported_package_path"] = str(imported)
        if not imported.is_relative_to(runtime):
            blockers.append("microshard package import escaped immutable runtime")
        lock = ROOT / str(config["environment"]["dependency_lock_path"])
        checks["environment_lock_sha256"] = _sha256(lock)
        if checks["environment_lock_sha256"] != immutable["environment_lock_sha256"]:
            blockers.append("microshard environment lock mismatch")
    except Exception as error:
        blockers.append(f"microshard immutable environment check failed: {error}")
    tail_publication = Path(str(paths["tail_publication_root"])) / str(
        identities["tail_parent_id"]
    )
    combined_publication = Path(str(paths["combined_publication_root"])) / str(
        identities["combined_train_id"]
    )
    if tail_publication.exists() or combined_publication.exists():
        blockers.append("microshard publication identity collides")
    free = shutil.disk_usage(Path(str(paths["tail_staging_root"]))).free
    checks["free_bytes"] = free
    if free < int(authorization["resource_gates"]["minimum_prelaunch_free_bytes"]):
        blockers.append("microshard prelaunch free-space gate failed")
    checks.update(
        {
            "worker_processes": 32,
            "physical_cpu_count": 32,
            "tail_shard_count": 512,
            "tail_accepted_target": 512,
        }
    )
    return {
        "status": "ready_for_official_execution" if not blockers else "blocked_preexecution",
        "phase": "4-terminal-tail-microshard-recovery",
        "generator_commit": GENERATOR_COMMIT,
        "orchestration_commit": orchestration_commit,
        "configuration_hash": configuration_hash(config),
        "preregistration_hash": config["preregistration"]["canonical_hash"],
        "checks": checks,
        "blockers": blockers,
        "official_identities": identities if not blockers else None,
        "development_tail_accepted_target": 512,
        "terminal_train_target": 131072,
        "training_authorized": False,
        "final_evaluation_authorized": False,
        "gwosc_gwtc_access_authorized": False,
    }


def _publish_namespace(
    *,
    config: Mapping[str, Any],
    authorization: Mapping[str, Any],
    namespace: TerminalNamespace,
    parent_stage: Path,
    parent_publication: Path,
    dataset_identity: str,
    preregistration: Mapping[str, Any],
    proposal: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Set[str]]]:
    namespace_config = _namespace_config(config, namespace, authorization)
    dataset_stage = parent_stage / dataset_identity
    dataset_published = parent_publication / dataset_identity
    work_root = dataset_published if parent_publication.exists() else dataset_stage
    if not parent_publication.exists():
        for child in ("attempts", "shards", "validation", "environment"):
            (dataset_stage / child).mkdir(parents=True, exist_ok=True)
        base._generate_pending(
            stage=dataset_stage,
            namespace_config=namespace_config,
            preregistration=preregistration,
            proposal=proposal,
            generator_commit=GENERATOR_COMMIT,
            dataset_identity=dataset_identity,
            scheduler_workers=32,
        )
    validation, identifiers = base.validate_stage_a_namespace(
        work_root,
        namespace_config=namespace_config,
        expected_split=namespace.split,
        expected_dataset=dataset_identity,
        generator_commit=GENERATOR_COMMIT,
    )
    base._validate_tail_records(work_root, namespace, dataset_identity)
    return validation, identifiers


def execute(
    *, orchestration_commit: str, certificate_path: Path, output: Path
) -> Dict[str, Any]:
    config = load_terminal_131k_contract(ROOT)
    authorization = _authorization(config)
    if authorization.get("authorization_status") != (
        "authorized_engineering_microshard_recovery_only"
    ):
        raise PermissionError("microshard execution is not authorized")
    _verify_orchestration_identity(ROOT, orchestration_commit)
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    expected = evaluate_release_gate(orchestration_commit=orchestration_commit)
    if expected["status"] != "ready_for_official_execution":
        raise PermissionError("microshard release gate is no longer ready")
    for key in (
        "status",
        "generator_commit",
        "orchestration_commit",
        "configuration_hash",
        "preregistration_hash",
        "official_identities",
    ):
        if certificate.get(key) != expected.get(key):
            raise ValueError(f"microshard certificate drifted at {key}")
    identities = dict(expected["official_identities"])
    corrected = base._resolve_corrected(config)
    preregistration = load_yaml(
        ROOT / str(config["parent_scientific_preregistration"]["path"])
    )
    proposal = load_yaml(
        ROOT / str(config["direct_target_density_implementation"]["path"])
    )
    train_parent = Path(str(authorization["published_train_increment"]["parent_root"]))
    train_dataset = train_parent / TRAIN_DATASET_ID
    train_ids = base._collect_group_ids((train_dataset,))
    if len(train_ids["system"]) != 65536:
        raise ValueError("published train increment count changed")
    train_tree, train_bytes = tree_checksum(train_parent)
    paths = authorization["paths"]
    tail_stage = Path(str(paths["tail_staging_root"])) / str(
        identities["tail_parent_id"]
    )
    tail_publication = Path(str(paths["tail_publication_root"])) / str(
        identities["tail_parent_id"]
    )
    if tail_publication.exists():
        raise FileExistsError("microshard publication identity already exists")
    if tail_stage.exists():
        run_manifest_path = tail_stage / "run_manifest.json"
        if not run_manifest_path.is_file():
            raise FileExistsError("microshard staging lacks a run manifest")
        run_manifest: MutableMapping[str, Any] = json.loads(
            run_manifest_path.read_text(encoding="utf-8")
        )
        if (
            run_manifest.get("tail_parent_id") != identities["tail_parent_id"]
            or run_manifest.get("orchestration_commit") != orchestration_commit
        ):
            raise ValueError("microshard resume identity drifted")
    else:
        tail_stage.mkdir(parents=True, exist_ok=False)
        run_manifest = {
            "status": "generating_or_resuming",
            "tail_parent_id": identities["tail_parent_id"],
            "tail_dataset_ids": identities["tail_dataset_ids"],
            "generator_commit": GENERATOR_COMMIT,
            "orchestration_commit": orchestration_commit,
            "base_configuration_hash": configuration_hash(config),
            "accepted_target": 512,
            "worker_processes": 32,
            "shards_per_namespace": 128,
            "accepted_pairs_per_shard": 1,
            "failed_parallel32_evidence_reused": False,
            "training_authorized": False,
            "final_evaluation_authorized": False,
            "gwosc_gwtc_access_authorized": False,
            "started_at": _utc_now(),
        }
        _atomic_json(tail_stage / "run_manifest.json", run_manifest)
    started = time.monotonic()
    validations: Dict[str, Any] = {}
    identifier_sets: Dict[str, Mapping[str, Set[str]]] = {}
    for namespace in _tail_namespaces(config):
        stratum = str(namespace.balanced_tail_stratum)
        validation, identifiers_value = _publish_namespace(
            config=config,
            authorization=authorization,
            namespace=namespace,
            parent_stage=tail_stage,
            parent_publication=tail_publication,
            dataset_identity=str(identities["tail_dataset_ids"][stratum]),
            preregistration=preregistration,
            proposal=proposal,
        )
        validations[stratum] = validation
        identifier_sets[stratum] = identifiers_value
    active_seconds = time.monotonic() - started
    if active_seconds > float(
        authorization["resource_gates"]["maximum_tail_active_hours"]
    ) * 3600:
        raise RuntimeError("microshard recovery exceeded its active-time cap")
    cross = base._cross_validation(corrected, train_ids, identifier_sets)
    manifest = {
        "status": "passed",
        "parent_run_id": identities["tail_parent_id"],
        "generator_commit": GENERATOR_COMMIT,
        "orchestration_commit": orchestration_commit,
        "preregistration_hash": config["preregistration"]["canonical_hash"],
        "base_configuration_hash": configuration_hash(config),
        "accepted_pair_count": 512,
        "namespace_count": 4,
        "cases_per_stratum": 128,
        "shards_per_namespace": 128,
        "accepted_pairs_per_shard": 1,
        "dataset_ids": identities["tail_dataset_ids"],
        "validations": validations,
        "cross_component_validation": cross,
        "dataset_role": "development_only_tail_diagnostic",
        "failed_parallel32_evidence_reused": False,
        "training_use_authorized": False,
        "architecture_selection_use_authorized": False,
        "calibration_use_authorized": False,
        "final_claim_use_authorized": False,
        "final_evaluation_unsealed": False,
        "gwosc_gwtc_accessed": False,
    }
    run_manifest["status"] = "validated_ready_for_atomic_publication"
    run_manifest["active_wall_seconds"] = active_seconds
    run_manifest["finished_at"] = _utc_now()
    _atomic_json(tail_stage / "run_manifest.json", run_manifest)
    _atomic_json(tail_stage / "dataset_manifest.json", manifest)
    tail_publication.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tail_stage, tail_publication)
    tail_tree, tail_bytes = tree_checksum(tail_publication)
    if tail_bytes > int(authorization["resource_gates"]["maximum_tail_output_bytes"]):
        raise RuntimeError("microshard publication exceeded its byte cap")
    remaining = shutil.disk_usage(tail_publication).free
    if remaining < int(authorization["resource_gates"]["minimum_post_run_free_bytes"]):
        raise RuntimeError("microshard post-run free-space gate failed")
    combined_partial = Path(str(paths["combined_staging_root"])) / str(
        identities["combined_train_id"]
    )
    combined_published = Path(str(paths["combined_publication_root"])) / str(
        identities["combined_train_id"]
    )
    if combined_partial.exists() or combined_published.exists():
        raise FileExistsError("microshard combined identity already exists")
    combined_manifest = {
        "status": "passed",
        "combined_train_id": identities["combined_train_id"],
        "accepted_physical_system_count": 131072,
        "validation_physical_system_count": 6144,
        "components": [
            {
                "role": "corrected_train_65k",
                "accepted_count": 65536,
                "logical_manifest_sha256": config["corrected_65k_reference"][
                    "corrected_combined_train_manifest_sha256"
                ],
            },
            {
                "role": "terminal_131k_train_increment",
                "accepted_count": 65536,
                "parent_run_id": TRAIN_PARENT_ID,
                "dataset_id": TRAIN_DATASET_ID,
                "parent_manifest_sha256": _sha256(
                    train_parent / "dataset_manifest.json"
                ),
                "publication_tree_sha256": train_tree,
                "publication_bytes": train_bytes,
            },
        ],
        "development_tail_parent_id": identities["tail_parent_id"],
        "development_tail_manifest_sha256": _sha256(
            tail_publication / "dataset_manifest.json"
        ),
        "development_tail_publication_tree_sha256": tail_tree,
        "development_tail_accepted_count": 512,
        "development_tail_shards_per_namespace": 128,
        "development_tail_pairs_per_shard": 1,
        "group_validation": cross,
        "strict_nested_train_ladder": True,
        "proposal_equals_evaluation": True,
        "all_importance_weights_one": True,
        "training_authorized": False,
        "architecture_selection_authorized": False,
        "calibration_authorized": False,
        "sbc_authorized": False,
        "final_evaluation_authorized": False,
        "extension_above_131072_authorized": False,
        "gwosc_gwtc_accessed": False,
    }
    combined_partial.mkdir(parents=True, exist_ok=False)
    _atomic_json(combined_partial / "dataset_manifest.json", combined_manifest)
    combined_published.parent.mkdir(parents=True, exist_ok=True)
    os.replace(combined_partial, combined_published)
    result = {
        "status": "passed",
        "phase": "4-terminal-tail-microshard-recovery",
        "generator_commit": GENERATOR_COMMIT,
        "orchestration_commit": orchestration_commit,
        "configuration_hash": configuration_hash(config),
        "preregistration_hash": config["preregistration"]["canonical_hash"],
        "train_parent_id": TRAIN_PARENT_ID,
        "train_dataset_id": TRAIN_DATASET_ID,
        **identities,
        "new_train_accepted_count": 65536,
        "development_tail_accepted_count": 512,
        "development_tail_namespace_count": 4,
        "development_tail_shard_count": 512,
        "development_tail_shards_per_namespace": 128,
        "development_tail_pairs_per_shard": 1,
        "terminal_train_accepted_count": 131072,
        "train_publication_tree_sha256": train_tree,
        "train_publication_bytes": train_bytes,
        "development_tail_publication_tree_sha256": tail_tree,
        "development_tail_publication_bytes": tail_bytes,
        "combined_manifest_path": str(combined_published / "dataset_manifest.json"),
        "combined_manifest_sha256": _sha256(
            combined_published / "dataset_manifest.json"
        ),
        "active_wall_seconds": active_seconds,
        "remaining_free_bytes": remaining,
        "proposal_equals_evaluation": True,
        "all_importance_weights_one": True,
        "failed_parallel32_evidence_reused": False,
        "train_131k_probe_authorized": False,
        "architecture_selection_authorized": False,
        "calibration_authorized": False,
        "sbc_authorized": False,
        "final_evaluation_authorized": False,
        "extension_above_131072_authorized": False,
        "gwosc_gwtc_accessed": False,
        "completed_at": _utc_now(),
    }
    _atomic_json(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "execute"))
    parser.add_argument("--orchestration-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--release-certificate", type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        if arguments.mode == "preflight":
            result = evaluate_release_gate(
                orchestration_commit=arguments.orchestration_commit
            )
            _atomic_json(arguments.output, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] == "ready_for_official_execution" else 2
        if arguments.release_certificate is None:
            raise ValueError("execute mode requires a release certificate")
        result = execute(
            orchestration_commit=arguments.orchestration_commit,
            certificate_path=arguments.release_certificate,
            output=arguments.output,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        _atomic_json(
            arguments.output,
            {
                "status": "execution_failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "train_131k_probe_authorized": False,
                "architecture_selection_authorized": False,
                "calibration_authorized": False,
                "sbc_authorized": False,
                "final_evaluation_authorized": False,
                "gwosc_gwtc_accessed": False,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
