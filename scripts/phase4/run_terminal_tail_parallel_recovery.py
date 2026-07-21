#!/usr/bin/env python3
"""Recover the terminal development-tail pool with parallel atomic shards.

This is an engineering-only recovery.  It reuses the already published and
validated 65,536-system train increment, leaves the stopped one-shard tail
attempt immutable, and changes only the physical shard layout of each frozen
128-case conditional tail namespace from 1x128 to 32x4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Set

from gwlens_mm.config import load_yaml
from gwlens_mm.production.storage import tree_checksum
from gwlens_mm.production.terminal131 import (
    TerminalNamespace,
    build_terminal_namespace_config,
    load_terminal_131k_contract,
    terminal_namespaces,
)
from gwlens_mm.provenance import canonical_json, configuration_hash, dataset_id
from scripts.phase4.run_terminal_131k import (
    APPROVED_REMOTE_ROOT,
    _atomic_json,
    _collect_group_ids,
    _cross_validation,
    _publish_namespace,
    _resolve_corrected,
    _sha256,
)

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = "configs/execution/phase4_terminal_131k_tail_parallel_recovery_authorization.yaml"
GENERATOR_COMMIT = "a4e6bac014ccd521d510c97593cb1368e826d5eb"
BASE_CONFIG_HASH = "abd07c5b8031a5cc9564531d29d9349f65b0918fafc494767fc912b7e7444ed7"
TRAIN_PARENT_ID = "phase4-terminal-131k-a4e6bac014cc-abd07c5b8031"
TRAIN_DATASET_ID = "gwlens-v2-2.0.0-alpha.3-e592848e725db2c3-train-increment"
_POSTFREEZE_PATHS = {
    "AGENTS.md",
    AUTHORIZATION,
    "docs/DECISIONS.md",
    "docs/FAILURES.md",
    "docs/PROJECT_STATE.md",
    "docs/reports/PHASE4_TERMINAL_TAIL_PARALLEL_RECOVERY_REPORT.md",
    "results/experiment_registry.csv",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        raise ValueError("parallel-tail orchestration commit must be a full SHA")
    if (root / ".git").is_dir():
        head = _git_head(root)
        if subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip():
            raise ValueError("parallel-tail checkout is dirty")
        if head == expected:
            return
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", expected, head],
            cwd=root,
            check=False,
        ).returncode != 0:
            raise ValueError("parallel-tail commit is not an ancestor of checkout")
        changed = subprocess.run(
            ["git", "diff", "--name-only", f"{expected}..{head}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        unexpected = sorted(set(changed) - _POSTFREEZE_PATHS)
        if unexpected:
            raise ValueError(
                "parallel-tail post-freeze protected paths changed: "
                + ", ".join(unexpected)
            )
        return
    marker = root / "SYNCED_ORCHESTRATION_COMMIT"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != expected:
        raise ValueError("parallel-tail checkout lacks its exact commit marker")


def _parallel_namespace(namespace: TerminalNamespace) -> TerminalNamespace:
    if namespace.balanced_tail_stratum is None:
        raise ValueError("parallel recovery accepts tail namespaces only")
    return replace(
        namespace,
        shard_count=32,
        pairs_per_shard=4,
        id_prefix=f"{namespace.id_prefix}-parallel32",
        attempt_namespace=f"{namespace.attempt_namespace}-parallel32",
    )


def _tail_namespaces(config: Mapping[str, Any]) -> tuple[TerminalNamespace, ...]:
    namespaces = tuple(_parallel_namespace(item) for item in terminal_namespaces(config)[1:])
    if len(namespaces) != 4:
        raise ValueError("parallel recovery requires exactly four tail namespaces")
    if any(item.shard_count * item.pairs_per_shard != 128 for item in namespaces):
        raise ValueError("parallel tail shard arithmetic changed the 128-case strata")
    if sum(item.accepted_count for item in namespaces) != 512:
        raise ValueError("parallel tail total changed")
    return namespaces


def _recovery_identities(config: Mapping[str, Any], orchestration_commit: str) -> Dict[str, Any]:
    namespaces = _tail_namespaces(config)
    namespace_configs = {
        str(item.balanced_tail_stratum): build_terminal_namespace_config(ROOT, config, item)
        for item in namespaces
    }
    tail_ids = {
        str(item.balanced_tail_stratum): dataset_id(
            "2.0.0-alpha.3",
            GENERATOR_COMMIT,
            configuration_hash(namespace_configs[str(item.balanced_tail_stratum)]),
            item.root_seed,
        )
        + f"-development-tail-{item.balanced_tail_stratum}-parallel32"
        for item in namespaces
    }
    identity_payload = {
        "base_configuration_hash": BASE_CONFIG_HASH,
        "generator_commit": GENERATOR_COMMIT,
        "orchestration_commit": orchestration_commit,
        "tail_dataset_ids": tail_ids,
        "tail_shards_per_namespace": 32,
        "tail_pairs_per_shard": 4,
    }
    suffix = hashlib.sha256(canonical_json(identity_payload).encode()).hexdigest()[:12]
    return {
        "tail_parent_id": f"phase4-terminal-tail-parallel32-{orchestration_commit[:12]}-{suffix}",
        "tail_dataset_ids": tail_ids,
        "combined_train_id": f"phase4-train-131k-{orchestration_commit[:12]}-{suffix}",
        "identity_payload_sha256": hashlib.sha256(
            canonical_json(identity_payload).encode()
        ).hexdigest(),
    }


def _authorization(config: Mapping[str, Any]) -> Dict[str, Any]:
    authorization = load_yaml(ROOT / AUTHORIZATION)
    if authorization.get("authorization_status") != (
        "authorized_engineering_tail_parallel_recovery_only"
    ):
        raise PermissionError("parallel tail recovery authorization is absent")
    frozen = authorization.get("frozen_scientific_contract", {})
    if (
        frozen.get("generator_commit") != GENERATOR_COMMIT
        or frozen.get("base_configuration_hash") != BASE_CONFIG_HASH
        or frozen.get("base_configuration_hash") != configuration_hash(config)
        or frozen.get("preregistration_hash") != config["preregistration"]["canonical_hash"]
        or frozen.get("train_parent_id") != TRAIN_PARENT_ID
        or frozen.get("train_dataset_id") != TRAIN_DATASET_ID
    ):
        raise ValueError("parallel tail recovery changed the frozen contract")
    execution = authorization.get("parallel_execution_contract", {})
    if (
        int(execution.get("worker_processes", -1)),
        int(execution.get("namespace_count", -1)),
        int(execution.get("accepted_pairs_per_namespace", -1)),
        int(execution.get("shards_per_namespace", -1)),
        int(execution.get("accepted_pairs_per_shard", -1)),
        int(execution.get("total_accepted_pairs", -1)),
    ) != (32, 4, 128, 32, 4, 512):
        raise ValueError("parallel tail recovery count or scheduler mismatch")
    interruption = authorization.get("stopped_tail_evidence", {})
    evidence = Path(str(interruption.get("path", "")))
    if not (
        evidence.is_absolute()
        and evidence.is_relative_to(APPROVED_REMOTE_ROOT)
        and evidence.is_dir()
        and int(interruption.get("complete_shard_count", -1)) == 0
        and int(interruption.get("partial_shard_count", -1)) == 1
        and int(interruption.get("accepted_partial_count", -1)) == 15
        and int(interruption.get("attempt_count", -1)) == 91839
        and interruption.get("reuse_authorized") is False
        and interruption.get("deletion_authorized") is False
    ):
        raise ValueError("stopped tail evidence is missing or mutable")
    partials = tuple(evidence.rglob("shard-*.partial"))
    completes = tuple(
        path
        for path in evidence.rglob("shard-*")
        if path.is_dir() and not path.name.endswith(".partial")
    )
    journals = tuple(evidence.rglob("*.jsonl"))
    attempt_lines = sum(
        sum(1 for _ in path.open("r", encoding="utf-8")) for path in journals
    )
    accepted_chunks = sum(
        1
        for path in evidence.rglob("noisy.zarr/[0-9]*.0.0.0")
        if path.is_file()
    )
    evidence_bytes = sum(path.stat().st_size for path in evidence.rglob("*") if path.is_file())
    if (
        len(partials),
        len(completes),
        attempt_lines,
        accepted_chunks,
        evidence_bytes,
    ) != (
        1,
        0,
        int(interruption["attempt_count"]),
        int(interruption["accepted_partial_count"]),
        int(interruption["retained_bytes"]),
    ):
        raise ValueError("stopped tail evidence contents changed")
    for key, expected in {
        "tail_recovery_authorized": True,
        "reuse_published_train_increment_authorized": True,
        "scientific_contract_change_authorized": False,
        "training_authorized": False,
        "architecture_selection_authorized": False,
        "calibration_authorized": False,
        "sbc_authorized": False,
        "final_evaluation_authorized": False,
        "extension_above_131072_authorized": False,
        "real_noise_authorized": False,
        "gwosc_gwtc_access_authorized": False,
    }.items():
        if authorization.get("authorization", {}).get(key) is not expected:
            raise PermissionError(f"parallel tail authorization requires {key}={expected}")
    return authorization


def evaluate_release_gate(*, orchestration_commit: str) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    blockers: list[str] = []
    try:
        config = load_terminal_131k_contract(ROOT)
        authorization = _authorization(config)
        immutable = authorization["immutable_orchestration"]
        if immutable.get("git_commit") != orchestration_commit:
            raise ValueError("parallel tail orchestration commit mismatch")
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
    train_dataset = train_parent / TRAIN_DATASET_ID
    try:
        manifest = train_parent / "dataset_manifest.json"
        observed_manifest_hash = _sha256(manifest)
        checks["train_parent_manifest_sha256"] = observed_manifest_hash
        if (
            observed_manifest_hash
            != authorization["published_train_increment"]["parent_manifest_sha256"]
        ):
            blockers.append("published train parent manifest hash mismatch")
        if not train_dataset.is_dir():
            blockers.append("published train dataset is absent")
    except Exception as error:
        blockers.append(f"published train increment check failed: {error}")
    identities = _recovery_identities(config, orchestration_commit)
    paths = authorization["paths"]
    for value in paths.values():
        path = Path(str(value))
        if not path.is_absolute() or not path.is_relative_to(APPROVED_REMOTE_ROOT):
            blockers.append("parallel tail path escaped the AutoDL project root")
            break
    immutable = authorization["immutable_orchestration"]
    try:
        wheel = Path(str(immutable["generator_wheel_path"]))
        observed_wheel_hash = _sha256(wheel)
        checks["generator_wheel_sha256"] = observed_wheel_hash
        if observed_wheel_hash != immutable["generator_wheel_sha256"]:
            blockers.append("parallel tail generator wheel hash mismatch")
        environment_lock = ROOT / str(config["environment"]["dependency_lock_path"])
        observed_lock_hash = _sha256(environment_lock)
        checks["environment_lock_sha256"] = observed_lock_hash
        if observed_lock_hash != immutable["environment_lock_sha256"]:
            blockers.append("parallel tail environment lock hash mismatch")
        if not Path(str(immutable["python_executable"])).is_file():
            blockers.append("parallel tail Python executable is absent")
    except Exception as error:
        blockers.append(f"parallel tail immutable environment check failed: {error}")
    tail_publication = Path(str(paths["tail_publication_root"])) / identities["tail_parent_id"]
    combined_publication = (
        Path(str(paths["combined_publication_root"])) / identities["combined_train_id"]
    )
    if tail_publication.exists() or combined_publication.exists():
        blockers.append("parallel tail recovery publication identity collides")
    free = shutil.disk_usage(Path(str(paths["tail_staging_root"]))).free
    checks["free_bytes"] = free
    if free < int(authorization["resource_gates"]["minimum_prelaunch_free_bytes"]):
        blockers.append("parallel tail recovery free-space gate failed")
    checks["worker_processes"] = 32
    checks["tail_shard_count"] = 128
    checks["tail_accepted_target"] = 512
    return {
        "status": "ready_for_official_execution" if not blockers else "blocked_preexecution",
        "phase": "4-terminal-tail-parallel-recovery",
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


def _validate_certificate(
    certificate: Mapping[str, Any], orchestration_commit: str, config: Mapping[str, Any]
) -> Dict[str, Any]:
    expected = evaluate_release_gate(orchestration_commit=orchestration_commit)
    if expected["status"] != "ready_for_official_execution":
        raise PermissionError("parallel tail release gate is no longer ready")
    for key in (
        "status",
        "generator_commit",
        "orchestration_commit",
        "configuration_hash",
        "preregistration_hash",
        "official_identities",
    ):
        if certificate.get(key) != expected.get(key):
            raise ValueError(f"parallel tail certificate drifted at {key}")
    if certificate.get("configuration_hash") != configuration_hash(config):
        raise ValueError("parallel tail certificate base configuration mismatch")
    return dict(expected["official_identities"])


def execute(*, orchestration_commit: str, certificate_path: Path, output: Path) -> Dict[str, Any]:
    config = load_terminal_131k_contract(ROOT)
    authorization = _authorization(config)
    _verify_orchestration_identity(ROOT, orchestration_commit)
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    identities = _validate_certificate(certificate, orchestration_commit, config)
    corrected = _resolve_corrected(config)
    preregistration = load_yaml(ROOT / str(config["parent_scientific_preregistration"]["path"]))
    proposal = load_yaml(ROOT / str(config["direct_target_density_implementation"]["path"]))
    train_parent = Path(str(authorization["published_train_increment"]["parent_root"]))
    train_dataset = train_parent / TRAIN_DATASET_ID
    train_ids = _collect_group_ids((train_dataset,))
    if len(train_ids["system"]) != 65536:
        raise ValueError("published train increment no longer contains exactly 65,536 systems")
    train_tree, train_bytes = tree_checksum(train_parent)
    if train_tree != authorization["published_train_increment"]["publication_tree_sha256"]:
        raise ValueError("published train increment tree hash mismatch")

    tail_stage = Path(str(authorization["paths"]["tail_staging_root"])) / str(
        identities["tail_parent_id"]
    )
    tail_publication = Path(str(authorization["paths"]["tail_publication_root"])) / str(
        identities["tail_parent_id"]
    )
    if tail_stage.exists() or tail_publication.exists():
        if tail_publication.exists():
            raise FileExistsError("parallel tail publication identity already exists")
        run_manifest_path = tail_stage / "run_manifest.json"
        if not run_manifest_path.is_file():
            raise FileExistsError("parallel tail staging lacks its run manifest")
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        expected_manifest_fields = {
            "status": "generating_or_resuming",
            "tail_parent_id": identities["tail_parent_id"],
            "generator_commit": GENERATOR_COMMIT,
            "orchestration_commit": orchestration_commit,
            "base_configuration_hash": configuration_hash(config),
            "accepted_target": 512,
            "worker_processes": 32,
            "shards_per_namespace": 32,
            "accepted_pairs_per_shard": 4,
        }
        if any(run_manifest.get(key) != value for key, value in expected_manifest_fields.items()):
            raise ValueError("parallel tail resume manifest drifted")
    else:
        tail_stage.mkdir(parents=True, exist_ok=False)
        run_manifest = {
            "status": "generating_or_resuming",
            "tail_parent_id": identities["tail_parent_id"],
            "tail_dataset_ids": identities["tail_dataset_ids"],
            "generator_commit": GENERATOR_COMMIT,
            "orchestration_commit": orchestration_commit,
            "base_configuration_hash": configuration_hash(config),
            "preregistration_hash": config["preregistration"]["canonical_hash"],
            "accepted_target": 512,
            "worker_processes": 32,
            "shards_per_namespace": 32,
            "accepted_pairs_per_shard": 4,
            "resume_strategy": "verify complete shards; retain and exclude partial evidence",
            "stopped_tail_evidence_reused": False,
            "training_authorized": False,
            "final_evaluation_authorized": False,
            "gwosc_gwtc_access_authorized": False,
            "started_at": _utc_now(),
        }
        _atomic_json(tail_stage / "run_manifest.json", run_manifest)
    started = time.monotonic()
    tail_validations: Dict[str, Any] = {}
    tail_identifiers: Dict[str, Mapping[str, Set[str]]] = {}
    for namespace in _tail_namespaces(config):
        stratum = str(namespace.balanced_tail_stratum)
        validation, identifiers_value = _publish_namespace(
            config=config,
            namespace=namespace,
            parent_stage=tail_stage,
            parent_publication=tail_publication,
            dataset_identity=str(identities["tail_dataset_ids"][stratum]),
            generator_commit=GENERATOR_COMMIT,
            preregistration=preregistration,
            proposal=proposal,
            scheduler_workers=32,
        )
        tail_validations[stratum] = validation
        tail_identifiers[stratum] = identifiers_value
    active_seconds = time.monotonic() - started
    if active_seconds > float(authorization["resource_gates"]["maximum_tail_active_hours"]) * 3600:
        raise RuntimeError("parallel tail recovery exceeded its active-time cap")
    cross = _cross_validation(corrected, train_ids, tail_identifiers)
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
        "shards_per_namespace": 32,
        "accepted_pairs_per_shard": 4,
        "dataset_ids": identities["tail_dataset_ids"],
        "validations": tail_validations,
        "cross_component_validation": cross,
        "dataset_role": "development_only_tail_diagnostic",
        "stopped_tail_evidence_reused": False,
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
        raise RuntimeError("parallel tail recovery exceeded its byte cap")
    remaining = shutil.disk_usage(tail_publication).free
    if remaining < int(authorization["resource_gates"]["minimum_post_run_free_bytes"]):
        raise RuntimeError("parallel tail recovery post-run free-space gate failed")

    combined_partial = Path(str(authorization["paths"]["combined_staging_root"])) / str(
        identities["combined_train_id"]
    )
    combined_published = Path(str(authorization["paths"]["combined_publication_root"])) / str(
        identities["combined_train_id"]
    )
    if combined_partial.exists() or combined_published.exists():
        raise FileExistsError("parallel recovery combined identity already exists")
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
                "parent_manifest_sha256": _sha256(train_parent / "dataset_manifest.json"),
                "publication_tree_sha256": train_tree,
                "publication_bytes": train_bytes,
            },
        ],
        "development_tail_parent_id": identities["tail_parent_id"],
        "development_tail_manifest_sha256": _sha256(tail_publication / "dataset_manifest.json"),
        "development_tail_publication_tree_sha256": tail_tree,
        "development_tail_accepted_count": 512,
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
        "phase": "4-terminal-tail-parallel-recovery",
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
        "development_tail_shard_count": 128,
        "terminal_train_accepted_count": 131072,
        "train_publication_tree_sha256": train_tree,
        "train_publication_bytes": train_bytes,
        "development_tail_publication_tree_sha256": tail_tree,
        "development_tail_publication_bytes": tail_bytes,
        "combined_manifest_path": str(combined_published / "dataset_manifest.json"),
        "combined_manifest_sha256": _sha256(combined_published / "dataset_manifest.json"),
        "active_wall_seconds": active_seconds,
        "remaining_free_bytes": remaining,
        "proposal_equals_evaluation": True,
        "all_importance_weights_one": True,
        "stopped_tail_evidence_reused": False,
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
    args = _parser().parse_args()
    try:
        if args.mode == "preflight":
            result = evaluate_release_gate(orchestration_commit=args.orchestration_commit)
            _atomic_json(args.output, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] == "ready_for_official_execution" else 2
        if args.release_certificate is None:
            raise ValueError("execute mode requires --release-certificate")
        result = execute(
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
