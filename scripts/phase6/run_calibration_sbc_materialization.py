#!/usr/bin/env python3
"""Future fail-closed release gate and atomic calibration/SBC materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Mapping, Set

from gwlens_mm.config import load_yaml
from gwlens_mm.production.calibration_sbc import (
    CONFIG_PATH,
    CalibrationSBCNamespace,
    build_calibration_sbc_namespace_config,
    calibration_sbc_namespaces,
    derive_calibration_sbc_identities,
    load_calibration_sbc_contract,
    validate_future_materialization_authorization,
)
from gwlens_mm.production.qualification import generate_qualification_shard
from gwlens_mm.production.run_control import verify_psd_files
from gwlens_mm.production.stage_a import validate_stage_a_namespace
from gwlens_mm.production.storage import tree_checksum, verify_complete_shard
from gwlens_mm.provenance import configuration_hash
from gwlens_mm.schema import V2Record
from gwlens_mm.training.data import resolve_stage_a_publication

ROOT = Path(__file__).resolve().parents[2]
APPROVED_REMOTE_ROOT = Path("/root/autodl-tmp/lensing-4")
GROUP_KEYS = ("pair", "source", "lens", "system", "noise")


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
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _future_authorization(path: Path, config: Mapping[str, Any]) -> Dict[str, Any]:
    authorization = load_yaml(path)
    generator_commit = str(authorization.get("immutable_generator", {}).get("git_commit", ""))
    validate_future_materialization_authorization(
        authorization,
        config=config,
        generator_commit=generator_commit,
    )
    implementation_commit = authorization.get("implementation_commit")
    if not isinstance(implementation_commit, str) or len(implementation_commit) != 40:
        raise ValueError("calibration/SBC implementation commit is unresolved")
    references = authorization.get("published_reference_datasets")
    if not isinstance(references, list) or len(references) != 2:
        raise ValueError("calibration/SBC gate requires Stage A and Stage B references")
    roles = {item.get("role") for item in references if isinstance(item, dict)}
    if roles != {"stage_a_train_and_validation", "stage_b_train_extension"}:
        raise ValueError("calibration/SBC published reference roles are incomplete")
    for evidence_name in ("training_size_decision", "architecture_decision"):
        evidence = authorization.get(evidence_name, {})
        evidence_path = Path(str(evidence.get("path", "")))
        if not evidence_path.is_absolute() or _sha256(evidence_path) != evidence.get("sha256"):
            raise ValueError(f"calibration/SBC {evidence_name} evidence mismatch")
    return authorization


def _validate_reference_parent(
    reference: Mapping[str, Any], *, expected_generator_commit: str
) -> Path:
    root = Path(str(reference["root"])).resolve()
    manifest = Path(str(reference["manifest_path"])).resolve()
    if (
        not root.is_relative_to(APPROVED_REMOTE_ROOT)
        or "staging" in root.parts
        or not root.is_dir()
        or manifest != root / "dataset_manifest.json"
        or _sha256(manifest) != reference["manifest_sha256"]
    ):
        raise ValueError("reference is not the authorization-bound atomic parent")
    role = str(reference["role"])
    if role == "stage_a_train_and_validation":
        resolved = resolve_stage_a_publication(
            root,
            expected_generator_commit=expected_generator_commit,
            expected_preregistration_hash=(
                "5aeaac395463bd073c44ead4ff4c5c729b5a2d4b4f1840c0825a53b30ab1bc98"
            ),
        )
        if resolved.parent_root != root:
            raise ValueError("Stage A reference resolver changed the parent identity")
        return root
    if role != "stage_b_train_extension":
        raise ValueError("unknown calibration/SBC published-reference role")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    validation = value.get("validation", {})
    cross = value.get("cross_component_validation", {})
    if (
        value.get("status"),
        value.get("generator_commit"),
        value.get("preregistration_hash"),
        int(value.get("accepted_pair_count", -1)),
        int(value.get("complete_shard_count", -1)),
        value.get("proposal_equals_evaluation"),
        value.get("all_importance_weights_one"),
        validation.get("status"),
        validation.get("split"),
        int(validation.get("accepted_pair_count", -1)),
        cross.get("stage_a_stage_b_group_disjoint"),
        cross.get("stage_b_validation_group_disjoint"),
    ) != (
        "passed",
        expected_generator_commit,
        "5aeaac395463bd073c44ead4ff4c5c729b5a2d4b4f1840c0825a53b30ab1bc98",
        32768,
        256,
        True,
        True,
        "passed",
        "train",
        32768,
        True,
        True,
    ):
        raise ValueError("Stage B reference manifest violates the frozen contract")
    dataset_id = str(validation.get("dataset_id", ""))
    if not dataset_id or not (root / dataset_id).is_dir():
        raise ValueError("Stage B reference dataset is absent")
    return root


def _verify_checkout(
    orchestration_commit: str,
    *,
    authorization: Mapping[str, Any],
    authorization_path: Path,
) -> None:
    if len(orchestration_commit) != 40:
        raise ValueError("orchestration commit must be a full Git SHA")
    if (ROOT / ".git").is_dir():
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != orchestration_commit or dirty:
            raise ValueError("calibration/SBC orchestration checkout is not exact and clean")
        implementation_commit = str(authorization["implementation_commit"])
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", implementation_commit, head],
            cwd=ROOT,
            check=False,
        )
        if ancestry.returncode != 0:
            raise ValueError("calibration/SBC release does not descend from implementation")
        changed = set(
            subprocess.run(
                ["git", "diff", "--name-only", f"{implementation_commit}..{head}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        )
        allowed = set(str(value) for value in authorization.get("allowed_release_changes", []))
        allowed.add(str(authorization_path.resolve().relative_to(ROOT.resolve())))
        if not changed <= allowed:
            raise ValueError("post-implementation release changed protected code")
        return
    marker = ROOT / "SYNCED_COMMIT"
    if not marker.is_file() or marker.read_text().strip() != orchestration_commit:
        raise ValueError("disposable checkout lacks exact orchestration marker")


def _validate_paths(config: Mapping[str, Any]) -> None:
    for value in config["paths"].values():
        path = Path(str(value))
        if not path.is_absolute() or not path.is_relative_to(APPROVED_REMOTE_ROOT):
            raise ValueError("calibration/SBC path escaped the AutoDL project root")


def evaluate_release_gate(
    *,
    authorization_path: Path,
    orchestration_commit: str,
    config_path: str = CONFIG_PATH,
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    blockers: list[str] = []
    try:
        config, _ = load_calibration_sbc_contract(ROOT, config_path)
        authorization = _future_authorization(authorization_path, config)
        _verify_checkout(
            orchestration_commit,
            authorization=authorization,
            authorization_path=authorization_path,
        )
        _validate_paths(config)
        checks["static_contract"] = "passed"
    except Exception as error:
        return {
            "status": "blocked_preexecution",
            "blockers": [str(error)],
            "checks": checks,
            "official_identities": None,
        }
    immutable = authorization["immutable_generator"]
    generator_commit = str(immutable["git_commit"])
    try:
        wheel = Path(str(immutable["wheel_path"]))
        checks["generator_wheel_sha256"] = _sha256(wheel)
        if checks["generator_wheel_sha256"] != immutable["wheel_sha256"]:
            blockers.append("calibration/SBC generator wheel hash mismatch")
    except Exception as error:
        blockers.append(f"calibration/SBC generator wheel unavailable: {error}")
    try:
        environment = Path(str(immutable["environment_lock_path"]))
        checks["environment_lock_sha256"] = _sha256(environment)
        if checks["environment_lock_sha256"] != immutable["environment_lock_sha256"]:
            blockers.append("calibration/SBC environment lock hash mismatch")
    except Exception as error:
        blockers.append(f"calibration/SBC environment lock unavailable: {error}")
    try:
        base = load_yaml(ROOT / str(config["base_data_config"]))
        checks["psd_files"] = verify_psd_files(base["gw"]["psd_curves"])
    except Exception as error:
        blockers.append(f"calibration/SBC PSD verification failed: {error}")
    reference_roots = []
    for reference in authorization["published_reference_datasets"]:
        try:
            reference_roots.append(
                _validate_reference_parent(
                    reference,
                    expected_generator_commit=generator_commit,
                )
            )
        except Exception as error:
            blockers.append(f"calibration/SBC reference {reference.get('role')} failed: {error}")
    staging = Path(str(config["paths"]["staging_root"]))
    publication = Path(str(config["paths"]["publication_root"]))
    staging.parent.mkdir(parents=True, exist_ok=True)
    publication.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(staging.parent).free
    checks["free_bytes"] = free
    if free < int(config["resource_gates"]["minimum_prelaunch_free_bytes"]):
        blockers.append("calibration/SBC free-space gate failed")
    identities = derive_calibration_sbc_identities(config, generator_commit)
    if (publication / identities.parent_run_id).exists():
        blockers.append("calibration/SBC official parent identity already exists")
    checks["published_reference_roots"] = [str(value) for value in reference_roots]
    status = "ready_for_official_execution" if not blockers else "blocked_preexecution"
    return {
        "status": status,
        "phase": "6-stage-c-development-materialization",
        "generator_commit": generator_commit,
        "orchestration_commit": orchestration_commit,
        "configuration_hash": configuration_hash(config),
        "calibration_sbc_preregistration_hash": config[
            "calibration_sbc_preregistration"
        ]["canonical_hash"],
        "checks": checks,
        "blockers": blockers,
        "official_identities": identities.as_dict() if not blockers else None,
        "accepted_target": 6144,
        "calibration_fit_statistics_authorized": False,
        "sbc_statistics_authorized": False,
        "checkpoint_access_authorized": False,
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
    pairs = int(namespace_config["pairs_per_shard"])
    pending = []
    for shard_index in range(int(namespace_config["shard_count"])):
        shard = stage / "shards" / f"shard-{shard_index:05d}"
        if shard.exists():
            verify_complete_shard(shard, pairs)
        else:
            pending.append(shard_index)
    workers = min(
        int(namespace_config["execution"]["qualification_worker_processes"]),
        len(pending),
    )
    if workers == 0:
        return
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


def _dataset_id_for_namespace(
    identities: Mapping[str, Any], namespace: CalibrationSBCNamespace
) -> str:
    key = (
        "calibration_dataset_id"
        if namespace.namespace_id == "calibration_fit"
        else "sbc_dataset_id"
    )
    return str(identities[key])


def _published_reference_ids(roots: tuple[Path, ...]) -> Dict[str, Set[str]]:
    """Stream group IDs from the two authorization-bound atomic parents."""

    pandas = __import__("pandas")
    identifiers: Dict[str, Set[str]] = {key: set() for key in GROUP_KEYS}
    for root in roots:
        records = tuple(sorted(root.rglob("records.parquet")))
        if not records:
            raise ValueError("published calibration/SBC reference has no records")
        for path in records:
            frame = pandas.read_parquet(path, columns=["record_json"])
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
                        raise ValueError(f"published references duplicate {key} ID")
                    identifiers[key].update(group)
    return identifiers


def execute(
    *,
    authorization_path: Path,
    orchestration_commit: str,
    certificate_path: Path,
    output: Path,
    config_path: str = CONFIG_PATH,
) -> Dict[str, Any]:
    config, _ = load_calibration_sbc_contract(ROOT, config_path)
    authorization = _future_authorization(authorization_path, config)
    _verify_checkout(
        orchestration_commit,
        authorization=authorization,
        authorization_path=authorization_path,
    )
    certificate = json.loads(certificate_path.read_text())
    expected = evaluate_release_gate(
        authorization_path=authorization_path,
        orchestration_commit=orchestration_commit,
        config_path=config_path,
    )
    if expected["status"] != "ready_for_official_execution":
        raise PermissionError("calibration/SBC release gate is no longer ready")
    for key in (
        "status",
        "generator_commit",
        "orchestration_commit",
        "configuration_hash",
        "official_identities",
    ):
        if certificate.get(key) != expected.get(key):
            raise ValueError(f"calibration/SBC release certificate {key} mismatch")
    identities = certificate["official_identities"]
    generator_commit = str(certificate["generator_commit"])
    parent_stage = Path(str(config["paths"]["staging_root"])) / str(
        identities["parent_run_id"]
    )
    publication = Path(str(config["paths"]["publication_root"])) / str(
        identities["parent_run_id"]
    )
    if publication.exists():
        raise FileExistsError("calibration/SBC publication identity already exists")
    parent_stage.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        parent_stage / "run_manifest.json",
        {
            "status": "generating_or_resuming",
            "parent_run_id": identities["parent_run_id"],
            "generator_commit": generator_commit,
            "orchestration_commit": orchestration_commit,
            "configuration_hash": configuration_hash(config),
            "accepted_target": 6144,
            "calibration_fit_statistics_authorized": False,
            "sbc_statistics_authorized": False,
            "checkpoint_access_authorized": False,
            "final_evaluation_authorized": False,
        },
    )
    preregistration = load_yaml(
        ROOT / str(config["parent_scientific_preregistration"]["path"])
    )
    proposal = load_yaml(ROOT / str(config["target_proposal_config"]))
    namespaces = calibration_sbc_namespaces(config)
    namespace_configs: Dict[str, Mapping[str, Any]] = {}
    for namespace in namespaces:
        namespace_config = build_calibration_sbc_namespace_config(
            ROOT, config, namespace
        )
        namespace_configs[namespace.namespace_id] = namespace_config
        dataset_identity = _dataset_id_for_namespace(identities, namespace)
        stage = parent_stage / dataset_identity
        for child in ("attempts", "shards", "validation", "environment"):
            (stage / child).mkdir(parents=True, exist_ok=True)
        _generate_pending(
            stage=stage,
            namespace_config=namespace_config,
            preregistration=preregistration,
            proposal=proposal,
            generator_commit=generator_commit,
            dataset_identity=dataset_identity,
        )
    validations: Dict[str, Any] = {}
    all_ids: Dict[str, Set[str]] = {key: set() for key in GROUP_KEYS}
    for namespace in namespaces:
        dataset_identity = _dataset_id_for_namespace(identities, namespace)
        summary, identifiers = validate_stage_a_namespace(
            parent_stage / dataset_identity,
            namespace_config=namespace_configs[namespace.namespace_id],
            expected_split=namespace.split,
            expected_dataset=dataset_identity,
            generator_commit=generator_commit,
        )
        counts = summary["em_cell_counts"]
        if len(counts) != 8 or set(counts.values()) != {namespace.expected_em_cell_count}:
            raise ValueError(f"{namespace.namespace_id} EM cells are not exactly balanced")
        validations[namespace.namespace_id] = summary
        for key in GROUP_KEYS:
            if all_ids[key] & identifiers[key]:
                raise ValueError(f"cross-namespace calibration/SBC {key} leakage")
            all_ids[key].update(identifiers[key])
    reference_roots = tuple(
        Path(str(value["root"])) for value in authorization["published_reference_datasets"]
    )
    reference_ids = _published_reference_ids(reference_roots)
    for key in GROUP_KEYS:
        if all_ids[key] & reference_ids[key]:
            raise ValueError(f"calibration/SBC {key} leaks into train or validation")
    manifest = {
        "status": "passed",
        "parent_run_id": identities["parent_run_id"],
        "generator_commit": generator_commit,
        "orchestration_commit": orchestration_commit,
        "configuration_hash": configuration_hash(config),
        "calibration_sbc_preregistration_hash": config[
            "calibration_sbc_preregistration"
        ]["canonical_hash"],
        "calibration_fit_accepted_count": 4096,
        "sbc_diagnostic_accepted_count": 2048,
        "accepted_pair_count": 6144,
        "complete_shard_count": 48,
        "validations": validations,
        "group_disjoint_from_train_validation_and_each_other": True,
        "proposal_equals_evaluation": True,
        "all_importance_weights_one": True,
        "calibration_fit_statistics_authorized": False,
        "sbc_statistics_authorized": False,
        "checkpoint_access_authorized": False,
        "final_evaluation_authorized": False,
        "gwosc_gwtc_accessed": False,
    }
    _atomic_json(parent_stage / "dataset_manifest.json", manifest)
    _atomic_json(
        parent_stage / "run_manifest.json",
        {**manifest, "status": "validated_ready_for_atomic_publication"},
    )
    digest, byte_count = tree_checksum(parent_stage)
    if byte_count > int(config["resource_gates"]["maximum_output_bytes"]):
        raise RuntimeError("calibration/SBC output exceeded the frozen byte cap")
    prepublication_free = shutil.disk_usage(parent_stage).free
    if prepublication_free < int(
        config["resource_gates"]["minimum_post_peak_free_bytes"]
    ):
        raise RuntimeError("calibration/SBC pre-publication free-space gate failed")
    publication.parent.mkdir(parents=True, exist_ok=True)
    os.replace(parent_stage, publication)
    remaining = shutil.disk_usage(publication).free
    if remaining < int(config["resource_gates"]["minimum_post_peak_free_bytes"]):
        raise RuntimeError("calibration/SBC post-publication free-space gate failed")
    result = {
        "status": "passed",
        "parent_run_id": identities["parent_run_id"],
        "calibration_dataset_id": identities["calibration_dataset_id"],
        "sbc_dataset_id": identities["sbc_dataset_id"],
        "accepted_pair_count": 6144,
        "complete_shard_count": 48,
        "publication_path": str(publication),
        "publication_tree_sha256": digest,
        "publication_bytes": byte_count,
        "remaining_free_bytes": remaining,
        "proposal_equals_evaluation": True,
        "all_importance_weights_one": True,
        "calibration_fit_statistics_authorized": False,
        "sbc_statistics_authorized": False,
        "final_evaluation_authorized": False,
        "gwosc_gwtc_accessed": False,
    }
    _atomic_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "execute"))
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--orchestration-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--release-certificate", type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.mode == "preflight":
            result = evaluate_release_gate(
                authorization_path=arguments.authorization,
                orchestration_commit=arguments.orchestration_commit,
                config_path=arguments.config,
            )
            _atomic_json(arguments.output, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] == "ready_for_official_execution" else 2
        if arguments.release_certificate is None:
            raise ValueError("execute mode requires --release-certificate")
        result = execute(
            authorization_path=arguments.authorization,
            orchestration_commit=arguments.orchestration_commit,
            certificate_path=arguments.release_certificate,
            output=arguments.output,
            config_path=arguments.config,
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
                "calibration_fit_statistics_authorized": False,
                "sbc_statistics_authorized": False,
                "final_evaluation_authorized": False,
                "gwosc_gwtc_accessed": False,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
