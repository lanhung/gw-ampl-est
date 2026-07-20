#!/usr/bin/env python3
"""Release-gate and publish the terminal 131k train increment and tail pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Set, Tuple

from gwlens_mm.config import load_yaml
from gwlens_mm.production.qualification import generate_qualification_shard
from gwlens_mm.production.run_control import verify_psd_files
from gwlens_mm.production.stage_a import validate_stage_a_namespace, verify_generator_commit
from gwlens_mm.production.storage import tree_checksum, verify_complete_shard
from gwlens_mm.production.terminal131 import (
    TERMINAL_131K_CONFIG,
    TerminalIdentities,
    TerminalNamespace,
    build_terminal_namespace_config,
    derive_terminal_identities,
    load_terminal_131k_contract,
    terminal_namespaces,
    validate_terminal_record,
)
from gwlens_mm.provenance import canonical_json, configuration_hash
from gwlens_mm.schema import V2Record
from gwlens_mm.training.data import resolve_corrected_training_publication

ROOT = Path(__file__).resolve().parents[2]
APPROVED_REMOTE_ROOT = Path("/root/autodl-tmp/lensing-4")
GROUP_KEYS = ("pair", "source", "lens", "system", "noise")
_POSTFREEZE_PATHS = (
    "AGENTS.md",
    "configs/execution/phase4_terminal_131k_execution_authorization.yaml",
    "docs/DECISIONS.md",
    "docs/FAILURES.md",
    "docs/PROJECT_STATE.md",
    "docs/reports/PHASE4_TERMINAL_131K_EXECUTION_REPORT.md",
    "results/experiment_registry.csv",
)


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


def _authorization(config: Mapping[str, Any]) -> Dict[str, Any]:
    path = ROOT / str(config["future_execution_authorization_path"])
    authorization = load_yaml(path)
    if authorization.get("authorization_status") != (
        "authorized_exact_terminal_131k_materialization_only"
    ):
        raise PermissionError("terminal 131k exact execution authorization is absent")
    frozen = authorization.get("frozen_contract", {})
    if (
        frozen.get("preregistration_hash") != config["preregistration"]["canonical_hash"]
        or frozen.get("configuration_hash") != configuration_hash(config)
        or frozen.get("terminal_65k_decision_sha256")
        != config["triggering_evidence"]["sha256"]
        or frozen.get("corrected_65k_manifest_sha256")
        != config["corrected_65k_reference"][
            "corrected_combined_train_manifest_sha256"
        ]
    ):
        raise ValueError("terminal 131k authorization frozen contract mismatch")
    counts = authorization.get("materialization_contract", {})
    if (
        int(counts.get("new_train_accepted_count", -1)),
        int(counts.get("new_train_shard_count", -1)),
        int(counts.get("development_tail_accepted_count", -1)),
        int(counts.get("development_tail_namespace_count", -1)),
        int(counts.get("terminal_train_count", -1)),
    ) != (65536, 512, 512, 4, 131072):
        raise ValueError("terminal 131k authorization count mismatch")
    flags = authorization.get("authorization", {})
    for key in (
        "train_increment_materialization_authorized",
        "development_tail_materialization_authorized",
        "accepted_pair_generator_authorized_within_terminal_release_only",
    ):
        if flags.get(key) is not True:
            raise PermissionError(f"terminal 131k requires {key}=true")
    for key in (
        "train_131k_probe_authorized",
        "architecture_selection_authorized",
        "calibration_authorized",
        "sbc_authorized",
        "final_evaluation_authorized",
        "extension_above_131072_authorized",
        "real_noise_authorized",
        "gwosc_gwtc_access_authorized",
    ):
        if flags.get(key) is not False:
            raise PermissionError(f"terminal 131k requires {key}=false")
    return authorization


def _validate_paths(config: Mapping[str, Any]) -> None:
    for value in config["paths"].values():
        path = Path(str(value))
        if not path.is_absolute() or not path.is_relative_to(APPROVED_REMOTE_ROOT):
            raise ValueError("terminal 131k path escaped the AutoDL project root")


def _resolve_corrected(config: Mapping[str, Any]) -> Any:
    reference = config["corrected_65k_reference"]
    roots = reference["publication_roots"]
    result = resolve_corrected_training_publication(
        Path(str(roots["correction"])),
        stage_a_parent_root=Path(str(roots["stage_a"])),
        stage_b_parent_root=Path(str(roots["stage_b"])),
        combined_base_root=Path(str(roots["combined_base"])),
        expected_base_generator_commit=str(reference["base_generator_commit"]),
        expected_base_preregistration_hash=str(reference["base_preregistration_hash"]),
        expected_correction_generator_commit=str(
            reference["correction_generator_commit"]
        ),
        expected_correction_preregistration_hash=str(
            reference["correction_preregistration_hash"]
        ),
        expected_correction_manifest_sha256=str(
            reference["correction_parent_manifest_sha256"]
        ),
        expected_correction_tree_sha256=str(
            reference["correction_publication_tree_sha256"]
        ),
        expected_combined_base_manifest_sha256=str(
            reference["combined_base_manifest_sha256"]
        ),
    )
    if result.corrected_combined_train_manifest_sha256 != reference[
        "corrected_combined_train_manifest_sha256"
    ]:
        raise ValueError("resolved corrected 65k view hash mismatch")
    return result


def _identity_mapping(identities: TerminalIdentities) -> Dict[str, Any]:
    return {
        "parent_run_id": identities.parent_run_id,
        "train_dataset_id": identities.train_dataset_id,
        "development_tail_parent_id": identities.development_tail_parent_id,
        "development_tail_dataset_ids": dict(
            identities.development_tail_dataset_ids
        ),
        "combined_train_id": identities.combined_train_id,
        "configuration_hash": identities.configuration_hash,
    }


def evaluate_release_gate(
    *, config_path: str, generator_commit: str
) -> Dict[str, Any]:
    """Return identities only after every exact immutable/resource check passes."""

    checks: Dict[str, Any] = {}
    blockers: list[str] = []
    try:
        config = load_terminal_131k_contract(ROOT, config_path)
        authorization = _authorization(config)
        _validate_paths(config)
        immutable = authorization["immutable_generator"]
        if immutable.get("git_commit") != generator_commit:
            raise ValueError("terminal 131k generator commit mismatch")
        verify_generator_commit(
            ROOT, generator_commit, allowed_postfreeze_paths=_POSTFREEZE_PATHS
        )
        checks["static_contract"] = "passed"
    except Exception as error:
        return {
            "status": "blocked_preexecution",
            "checks": checks,
            "blockers": [str(error)],
            "official_identities": None,
        }
    wheel = Path(str(immutable["wheel_path"]))
    try:
        checks["generator_wheel_sha256"] = _sha256(wheel)
        if checks["generator_wheel_sha256"] != immutable["wheel_sha256"]:
            blockers.append("terminal 131k generator wheel hash mismatch")
    except Exception as error:
        blockers.append(f"terminal 131k generator wheel unavailable: {error}")
    lock = ROOT / str(config["environment"]["dependency_lock_path"])
    checks["environment_lock_sha256"] = _sha256(lock)
    if checks["environment_lock_sha256"] != immutable["environment_lock_sha256"]:
        blockers.append("terminal 131k environment lock hash mismatch")
    try:
        resolved = _resolve_corrected(config)
        checks["corrected_65k_manifest_sha256"] = (
            resolved.corrected_combined_train_manifest_sha256
        )
    except Exception as error:
        blockers.append(f"corrected 65k publication resolution failed: {error}")
    try:
        base = load_yaml(ROOT / str(config["base_data_config"]))
        checks["psd_files"] = verify_psd_files(base["gw"]["psd_curves"])
    except Exception as error:
        blockers.append(f"terminal 131k PSD verification failed: {error}")
    train_staging = Path(str(config["paths"]["train_staging_root"]))
    train_staging.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(train_staging.parent).free
    checks["free_bytes"] = free
    if free < int(config["resource_gates"]["minimum_prelaunch_free_bytes"]):
        blockers.append("terminal 131k free-space gate failed")
    identities = derive_terminal_identities(ROOT, config, generator_commit)
    collisions = (
        Path(str(config["paths"]["train_publication_root"]))
        / identities.parent_run_id,
        Path(str(config["paths"]["tail_publication_root"]))
        / identities.development_tail_parent_id,
        Path(str(config["paths"]["combined_publication_root"]))
        / identities.combined_train_id,
    )
    if any(path.exists() for path in collisions):
        blockers.append("terminal 131k official publication identity already exists")
    status = "ready_for_official_execution" if not blockers else "blocked_preexecution"
    return {
        "status": status,
        "phase": "4-terminal-131k",
        "generator_commit": generator_commit,
        "configuration_hash": configuration_hash(config),
        "preregistration_hash": config["preregistration"]["canonical_hash"],
        "checks": checks,
        "blockers": blockers,
        "official_identities": _identity_mapping(identities) if not blockers else None,
        "new_train_accepted_target": 65536,
        "development_tail_accepted_target": 512,
        "terminal_train_target": 131072,
        "train_131k_probe_authorized": False,
        "architecture_selection_authorized": False,
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


def _validate_tail_records(
    root: Path, namespace: TerminalNamespace, dataset_identity: str
) -> None:
    pandas = __import__("pandas")
    observed = 0
    for parquet in sorted(root.glob("shards/shard-*/records.parquet")):
        frame = pandas.read_parquet(parquet, columns=["record_json"])
        for raw in frame["record_json"]:
            validate_terminal_record(
                V2Record.from_json(str(raw)), namespace, expected_dataset=dataset_identity
            )
            observed += 1
    if observed != namespace.accepted_count:
        raise ValueError("development-tail record count mismatch")


def _collect_group_ids(
    roots: Tuple[Path, ...], *, excluded_system_ids: Tuple[str, ...] = ()
) -> Dict[str, Set[str]]:
    pandas = __import__("pandas")
    excluded = set(excluded_system_ids)
    observed_exclusions: set[str] = set()
    result: Dict[str, Set[str]] = {key: set() for key in GROUP_KEYS}
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"group-reference root is absent: {root}")
        for parquet in sorted(root.glob("shards/shard-*/records.parquet")):
            frame = pandas.read_parquet(parquet, columns=["record_json"])
            for raw in frame["record_json"]:
                record = V2Record.from_json(str(raw))
                system = record.pair.physical_system_id
                if system in excluded:
                    observed_exclusions.add(system)
                    continue
                values = {
                    "pair": (record.pair.pair_id,),
                    "source": (record.pair.source_id,),
                    "lens": (record.pair.lens_id,),
                    "system": (system,),
                    "noise": tuple(record.provenance.used_noise_segment_ids),
                }
                for key, items in values.items():
                    if result[key] & set(items):
                        raise ValueError(f"group references duplicate {key} ID")
                    result[key].update(items)
    if observed_exclusions != excluded:
        raise ValueError("corrected-view exclusions do not match base records")
    return result


def _cross_validation(
    corrected: Any,
    train_ids: Mapping[str, Set[str]],
    tail_ids: Mapping[str, Mapping[str, Set[str]]],
) -> Dict[str, Any]:
    corrected_train = _collect_group_ids(
        (
            corrected.stage_a.train_root,
            corrected.combined_base.stage_b_train_root,
            corrected.stage_a_replacement_root,
            corrected.stage_b_replacement_root,
        ),
        excluded_system_ids=(
            *corrected.stage_a_excluded_ids,
            *corrected.stage_b_excluded_ids,
        ),
    )
    validation = _collect_group_ids((corrected.stage_a.validation_root,))
    pools: Dict[str, Mapping[str, Set[str]]] = {
        "corrected_train_65k": corrected_train,
        "core_validation": validation,
        "train_increment": train_ids,
        **{f"tail_{key}": value for key, value in tail_ids.items()},
    }
    names = tuple(pools)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            for key in GROUP_KEYS:
                if pools[left_name][key] & pools[right_name][key]:
                    raise ValueError(
                        f"terminal 131k {key} leakage: {left_name} vs {right_name}"
                    )
    combined: Dict[str, Any] = {}
    for key in GROUP_KEYS:
        values = corrected_train[key] | train_ids[key]
        expected = 131072 * (6 if key == "noise" else 1)
        if len(values) != expected:
            raise ValueError(f"terminal 131k combined {key} count mismatch")
        combined[f"{key}_count"] = len(values)
        combined[f"{key}_ids_sha256"] = hashlib.sha256(
            canonical_json(sorted(values)).encode()
        ).hexdigest()
    combined["all_train_validation_tail_groups_disjoint"] = True
    combined["corrected_65k_membership_preserved"] = True
    return combined


def _append_segment(manifest: MutableMapping[str, Any]) -> Dict[str, Any]:
    segments = manifest.setdefault("execution_segments", [])
    if not isinstance(segments, list):
        raise ValueError("terminal 131k execution segments are invalid")
    segment: Dict[str, Any] = {"started_at": _utc_now(), "status": "running"}
    segments.append(segment)
    return segment


def _publish_namespace(
    *,
    config: Mapping[str, Any],
    namespace: TerminalNamespace,
    parent_stage: Path,
    parent_publication: Path,
    dataset_identity: str,
    generator_commit: str,
    preregistration: Mapping[str, Any],
    proposal: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Set[str]]]:
    namespace_config = build_terminal_namespace_config(ROOT, config, namespace)
    dataset_stage = parent_stage / dataset_identity
    dataset_published = parent_publication / dataset_identity
    work_root = dataset_published if parent_publication.exists() else dataset_stage
    if not parent_publication.exists():
        for child in ("attempts", "shards", "validation", "environment"):
            (dataset_stage / child).mkdir(parents=True, exist_ok=True)
        _generate_pending(
            stage=dataset_stage,
            namespace_config=namespace_config,
            preregistration=preregistration,
            proposal=proposal,
            generator_commit=generator_commit,
            dataset_identity=dataset_identity,
        )
    validation, identifiers = validate_stage_a_namespace(
        work_root,
        namespace_config=namespace_config,
        expected_split=namespace.split,
        expected_dataset=dataset_identity,
        generator_commit=generator_commit,
    )
    if namespace.balanced_tail_stratum is not None:
        _validate_tail_records(work_root, namespace, dataset_identity)
    return validation, identifiers


def execute(
    *, config_path: str, generator_commit: str, certificate_path: Path, output: Path
) -> Dict[str, Any]:
    """Generate, validate, and atomically publish the exact terminal release."""

    config = load_terminal_131k_contract(ROOT, config_path)
    _authorization(config)
    verify_generator_commit(
        ROOT, generator_commit, allowed_postfreeze_paths=_POSTFREEZE_PATHS
    )
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    if (
        certificate.get("status") != "ready_for_official_execution"
        or certificate.get("generator_commit") != generator_commit
        or certificate.get("configuration_hash") != configuration_hash(config)
    ):
        raise PermissionError("terminal 131k release certificate is not ready")
    identities_value = certificate.get("official_identities")
    expected = _identity_mapping(derive_terminal_identities(ROOT, config, generator_commit))
    if identities_value != expected:
        raise ValueError("terminal 131k release identities mismatch")
    corrected = _resolve_corrected(config)
    preregistration = load_yaml(
        ROOT / str(config["parent_scientific_preregistration"]["path"])
    )
    proposal = load_yaml(
        ROOT / str(config["direct_target_density_implementation"]["path"])
    )
    namespaces = terminal_namespaces(config)
    train_namespace = namespaces[0]
    train_stage = Path(str(config["paths"]["train_staging_root"])) / str(
        expected["parent_run_id"]
    )
    train_publication = Path(str(config["paths"]["train_publication_root"])) / str(
        expected["parent_run_id"]
    )
    run_manifest_path = (train_publication if train_publication.exists() else train_stage) / (
        "run_manifest.json"
    )
    if run_manifest_path.exists():
        run_manifest: MutableMapping[str, Any] = json.loads(
            run_manifest_path.read_text(encoding="utf-8")
        )
    else:
        run_manifest = {
            "status": "generating_or_resuming",
            "parent_run_id": expected["parent_run_id"],
            "generator_commit": generator_commit,
            "configuration_hash": configuration_hash(config),
            "accepted_target": 65536,
            "terminal_train_target": 131072,
            "resume_strategy": "verify and preserve every complete atomic shard",
            "training_authorized": False,
            "architecture_selection_authorized": False,
            "final_evaluation_authorized": False,
            "gwosc_gwtc_access_authorized": False,
            "execution_segments": [],
        }
    segment = _append_segment(run_manifest)
    _atomic_json(run_manifest_path, run_manifest)
    started = time.monotonic()
    train_validation, train_ids = _publish_namespace(
        config=config,
        namespace=train_namespace,
        parent_stage=train_stage,
        parent_publication=train_publication,
        dataset_identity=str(expected["train_dataset_id"]),
        generator_commit=generator_commit,
        preregistration=preregistration,
        proposal=proposal,
    )
    if not train_publication.exists():
        train_manifest = {
            "status": "passed",
            "parent_run_id": expected["parent_run_id"],
            "dataset_id": expected["train_dataset_id"],
            "generator_commit": generator_commit,
            "preregistration_version": config["preregistration"]["version"],
            "preregistration_hash": config["preregistration"]["canonical_hash"],
            "configuration_hash": configuration_hash(config),
            "accepted_pair_count": 65536,
            "complete_shard_count": 512,
            "validation": train_validation,
            "proposal_equals_evaluation": True,
            "all_importance_weights_one": True,
            "training_authorized": False,
            "architecture_selection_authorized": False,
            "final_evaluation_authorized": False,
            "gwosc_gwtc_accessed": False,
        }
        _atomic_json(train_stage / "dataset_manifest.json", train_manifest)
        segment.update(
            {
                "finished_at": _utc_now(),
                "active_wall_seconds": time.monotonic() - started,
                "status": "completed",
            }
        )
        run_manifest["status"] = "validated_ready_for_atomic_publication"
        _atomic_json(train_stage / "run_manifest.json", run_manifest)
        train_publication.parent.mkdir(parents=True, exist_ok=True)
        os.replace(train_stage, train_publication)
    train_tree, train_bytes = tree_checksum(train_publication)

    tail_stage = Path(str(config["paths"]["tail_staging_root"])) / str(
        expected["development_tail_parent_id"]
    )
    tail_publication = Path(str(config["paths"]["tail_publication_root"])) / str(
        expected["development_tail_parent_id"]
    )
    tail_validations: Dict[str, Any] = {}
    tail_identifiers: Dict[str, Mapping[str, Set[str]]] = {}
    tail_dataset_ids = expected["development_tail_dataset_ids"]
    for namespace in namespaces[1:]:
        stratum = str(namespace.balanced_tail_stratum)
        validation, identifiers = _publish_namespace(
            config=config,
            namespace=namespace,
            parent_stage=tail_stage,
            parent_publication=tail_publication,
            dataset_identity=str(tail_dataset_ids[stratum]),
            generator_commit=generator_commit,
            preregistration=preregistration,
            proposal=proposal,
        )
        tail_validations[stratum] = validation
        tail_identifiers[stratum] = identifiers
    cross = _cross_validation(corrected, train_ids, tail_identifiers)
    if not tail_publication.exists():
        tail_manifest = {
            "status": "passed",
            "parent_run_id": expected["development_tail_parent_id"],
            "generator_commit": generator_commit,
            "preregistration_hash": config["preregistration"]["canonical_hash"],
            "configuration_hash": configuration_hash(config),
            "accepted_pair_count": 512,
            "namespace_count": 4,
            "cases_per_stratum": 128,
            "dataset_ids": tail_dataset_ids,
            "validations": tail_validations,
            "cross_component_validation": cross,
            "dataset_role": "development_only_tail_diagnostic",
            "training_use_authorized": False,
            "architecture_selection_use_authorized": False,
            "calibration_use_authorized": False,
            "final_claim_use_authorized": False,
            "final_evaluation_unsealed": False,
            "gwosc_gwtc_accessed": False,
        }
        _atomic_json(tail_stage / "dataset_manifest.json", tail_manifest)
        tail_publication.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tail_stage, tail_publication)
    tail_tree, tail_bytes = tree_checksum(tail_publication)
    total_bytes = train_bytes + tail_bytes
    if total_bytes > int(config["resource_gates"]["maximum_new_publication_bytes"]):
        raise RuntimeError("terminal 131k publication exceeded its frozen byte cap")
    remaining = shutil.disk_usage(train_publication).free
    if remaining < int(config["resource_gates"]["minimum_post_peak_free_bytes"]):
        raise RuntimeError("terminal 131k post-publication free-space gate failed")

    combined_partial = Path(str(config["paths"]["combined_staging_root"])) / str(
        expected["combined_train_id"]
    )
    combined_published = Path(str(config["paths"]["combined_publication_root"])) / str(
        expected["combined_train_id"]
    )
    combined_manifest = {
        "status": "passed",
        "combined_train_id": expected["combined_train_id"],
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
                "parent_run_id": expected["parent_run_id"],
                "dataset_id": expected["train_dataset_id"],
                "parent_manifest_sha256": _sha256(
                    train_publication / "dataset_manifest.json"
                ),
                "publication_tree_sha256": train_tree,
                "publication_bytes": train_bytes,
            },
        ],
        "development_tail_parent_id": expected["development_tail_parent_id"],
        "development_tail_manifest_sha256": _sha256(
            tail_publication / "dataset_manifest.json"
        ),
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
    deterministic = json.dumps(combined_manifest, indent=2, sort_keys=True) + "\n"
    if not combined_published.exists():
        combined_partial.mkdir(parents=True, exist_ok=False)
        _atomic_json(combined_partial / "dataset_manifest.json", combined_manifest)
        combined_published.parent.mkdir(parents=True, exist_ok=True)
        os.replace(combined_partial, combined_published)
    elif _sha256(combined_published / "dataset_manifest.json") != hashlib.sha256(
        deterministic.encode()
    ).hexdigest():
        raise ValueError("existing terminal 131k manifest differs from deterministic result")
    result = {
        "status": "passed",
        **expected,
        "generator_commit": generator_commit,
        "new_train_accepted_count": 65536,
        "new_train_shard_count": 512,
        "development_tail_accepted_count": 512,
        "development_tail_namespace_count": 4,
        "terminal_train_accepted_count": 131072,
        "train_publication_tree_sha256": train_tree,
        "train_publication_bytes": train_bytes,
        "development_tail_publication_tree_sha256": tail_tree,
        "development_tail_publication_bytes": tail_bytes,
        "combined_manifest_path": str(combined_published / "dataset_manifest.json"),
        "combined_manifest_sha256": _sha256(
            combined_published / "dataset_manifest.json"
        ),
        "remaining_free_bytes": remaining,
        "proposal_equals_evaluation": True,
        "all_importance_weights_one": True,
        "train_131k_probe_authorized": False,
        "architecture_selection_authorized": False,
        "calibration_authorized": False,
        "sbc_authorized": False,
        "final_evaluation_authorized": False,
        "extension_above_131072_authorized": False,
        "gwosc_gwtc_accessed": False,
    }
    _atomic_json(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "execute"))
    parser.add_argument("--config", default=TERMINAL_131K_CONFIG)
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--release-certificate", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.mode == "preflight":
            result = evaluate_release_gate(
                config_path=args.config, generator_commit=args.generator_commit
            )
            _atomic_json(args.output, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] == "ready_for_official_execution" else 2
        if args.release_certificate is None:
            raise ValueError("execute mode requires --release-certificate")
        result = execute(
            config_path=args.config,
            generator_commit=args.generator_commit,
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
                "final_evaluation_authorized": False,
                "gwosc_gwtc_accessed": False,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
