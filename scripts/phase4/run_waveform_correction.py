#!/usr/bin/env python3
"""Preflight and atomically publish the five-system waveform correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Set

from gwlens_mm.config import load_yaml
from gwlens_mm.production.qualification import generate_qualification_shard
from gwlens_mm.production.stage_a import validate_stage_a_namespace
from gwlens_mm.production.storage import tree_checksum
from gwlens_mm.production.waveform_correction import (
    CORRECTION_COMPONENTS,
    CORRECTION_CONFIG,
    GROUP_KEYS,
    build_replacement_namespace_config,
    derive_waveform_correction_identity,
    load_waveform_correction_contract,
    published_group_ids,
)
from gwlens_mm.provenance import configuration_hash
from gwlens_mm.schema import SplitName

ROOT = Path(__file__).resolve().parents[2]
APPROVED_REMOTE_ROOT = Path("/root/autodl-tmp/lensing-4")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_exact_commit(expected: str) -> None:
    if (ROOT / ".git").exists():
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != expected or dirty:
            raise ValueError("waveform-correction checkout is not the clean frozen commit")
    else:
        marker = ROOT / "SYNCED_COMMIT"
        if not marker.is_file() or marker.read_text().strip() != expected:
            raise ValueError("waveform-correction synced commit marker mismatch")


def _load_authorization(
    config: Mapping[str, Any], implementation_commit: str, *, execution: bool
) -> Dict[str, Any]:
    authorization = load_yaml(ROOT / str(config["authorization_path"]))
    expected_status = (
        "authorized_exact_replacement_and_corrected_view_publication"
        if execution
        else "authorized_implementation_only_execution_unresolved"
    )
    if authorization.get("authorization_status") != expected_status:
        raise PermissionError("waveform-correction authorization status mismatch")
    if authorization.get("implementation_commit") != implementation_commit:
        raise ValueError("waveform-correction implementation commit mismatch")
    if authorization.get("preregistration", {}).get("canonical_hash") != config[
        "preregistration"
    ]["canonical_hash"]:
        raise ValueError("waveform-correction authorization hash mismatch")
    flags = authorization.get("authorization", {})
    immutable = authorization.get("immutable_implementation", {})
    if immutable.get("git_commit") != implementation_commit:
        raise ValueError("waveform-correction immutable commit mismatch")
    wheel_path = Path(str(immutable.get("wheel_path", "")))
    if (
        immutable.get("editable_install_authorized") is not False
        or not wheel_path.is_absolute()
        or not wheel_path.is_relative_to(APPROVED_REMOTE_ROOT)
        or _sha256(wheel_path) != immutable.get("wheel_sha256")
    ):
        raise ValueError("waveform-correction immutable wheel contract failed")
    evidence = authorization.get("preexecution_evidence", {})
    regression_path = ROOT / str(evidence.get("real_record_regression_path", ""))
    if _sha256(regression_path) != evidence.get("real_record_regression_sha256"):
        raise ValueError("waveform-correction regression evidence hash mismatch")
    if execution:
        for key in (
            "replacement_materialization_authorized",
            "corrected_view_publication_authorized",
        ):
            if flags.get(key) is not True:
                raise PermissionError(f"waveform correction requires {key}=true")
    for key in (
        "model_training_authorized",
        "architecture_selection_authorized",
        "calibration_authorized",
        "sbc_authorized",
        "final_evaluation_authorized",
        "extension_above_65536_authorized",
        "real_noise_authorized",
        "gwosc_gwtc_access_authorized",
    ):
        if flags.get(key) is not False:
            raise PermissionError(f"waveform correction requires {key}=false")
    return authorization


def _base_paths(config: Mapping[str, Any]) -> Dict[str, Path]:
    base = config["immutable_base_publications"]
    stage_a_root = Path(str(base["stage_a"]["parent_root"]))
    stage_b_root = Path(str(base["stage_b"]["parent_root"]))
    return {
        "stage_a_parent": stage_a_root,
        "stage_a_train": stage_a_root / str(base["stage_a"]["train_dataset_id"]),
        "stage_a_validation": stage_a_root
        / str(base["stage_a"]["validation_dataset_id"]),
        "stage_b_parent": stage_b_root,
        "stage_b_train": stage_b_root / str(base["stage_b"]["train_dataset_id"]),
        "combined_manifest": Path(str(base["combined_65k"]["manifest_path"])),
    }


def evaluate_release_gate(
    *, config_path: str, implementation_commit: str
) -> Dict[str, Any]:
    config = load_waveform_correction_contract(ROOT, config_path)
    _clean_exact_commit(implementation_commit)
    authorization = _load_authorization(config, implementation_commit, execution=True)
    blockers = []
    checks: Dict[str, Any] = {}
    paths = _base_paths(config)
    expected = config["immutable_base_publications"]
    for label, path, digest in (
        (
            "stage_a_parent",
            paths["stage_a_parent"] / "dataset_manifest.json",
            expected["stage_a"]["parent_manifest_sha256"],
        ),
        (
            "stage_b_parent",
            paths["stage_b_parent"] / "dataset_manifest.json",
            expected["stage_b"]["parent_manifest_sha256"],
        ),
        (
            "combined_65k",
            paths["combined_manifest"],
            expected["combined_65k"]["manifest_sha256"],
        ),
    ):
        try:
            actual = _sha256(path)
            checks[f"{label}_manifest_sha256"] = actual
            if actual != digest:
                blockers.append(f"{label} manifest hash mismatch")
        except Exception as error:
            blockers.append(f"{label} manifest unavailable: {error}")
    identity = derive_waveform_correction_identity(config, implementation_commit)
    publication = Path(str(config["paths"]["publication_root"])) / identity.parent_run_id
    staging = Path(str(config["paths"]["staging_root"])) / identity.parent_run_id
    for path in (publication, staging):
        if not path.is_absolute() or not path.is_relative_to(APPROVED_REMOTE_ROOT):
            blockers.append("waveform-correction path escaped the project root")
    if publication.exists() or staging.exists():
        blockers.append("waveform-correction identity already exists")
    free = shutil.disk_usage(Path(str(config["paths"]["staging_root"])).parent).free
    checks["free_bytes"] = free
    if free < int(config["resource_gates"]["minimum_prelaunch_free_bytes"]):
        blockers.append("waveform-correction free-space gate failed")
    status = "ready_for_official_execution" if not blockers else "blocked_preexecution"
    return {
        "status": status,
        "phase": "4-waveform-correction",
        "implementation_commit": implementation_commit,
        "configuration_hash": configuration_hash(config),
        "preregistration_hash": config["preregistration"]["canonical_hash"],
        "authorization_status": authorization["authorization_status"],
        "official_identities": identity.__dict__ if not blockers else None,
        "checks": checks,
        "blockers": blockers,
        "replacement_count": 5,
        "model_training_authorized": False,
        "gwosc_gwtc_access_authorized": False,
    }


def _assert_disjoint(
    left: Mapping[str, Set[str]], right: Mapping[str, Set[str]], label: str
) -> None:
    for key in GROUP_KEYS:
        if left[key] & right[key]:
            raise ValueError(f"waveform-correction {label} {key} leakage")


def execute(
    *,
    config_path: str,
    implementation_commit: str,
    certificate_path: Path,
    output: Path,
) -> Dict[str, Any]:
    config = load_waveform_correction_contract(ROOT, config_path)
    _clean_exact_commit(implementation_commit)
    _load_authorization(config, implementation_commit, execution=True)
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    if (
        certificate.get("status") != "ready_for_official_execution"
        or certificate.get("implementation_commit") != implementation_commit
        or certificate.get("configuration_hash") != configuration_hash(config)
    ):
        raise PermissionError("waveform-correction release certificate is invalid")
    raw_identity = certificate.get("official_identities")
    if not isinstance(raw_identity, dict):
        raise ValueError("waveform-correction release certificate lacks identities")
    identity = derive_waveform_correction_identity(config, implementation_commit)
    if raw_identity != identity.__dict__:
        raise ValueError("waveform-correction identities differ from derivation")

    stage = Path(str(config["paths"]["staging_root"])) / identity.parent_run_id
    publication = (
        Path(str(config["paths"]["publication_root"])) / identity.parent_run_id
    )
    if stage.exists() or publication.exists():
        raise FileExistsError("waveform-correction official identity already exists")
    stage.mkdir(parents=True, exist_ok=False)
    parent_preregistration = load_yaml(
        ROOT / str(config["parent_scientific_preregistration"]["path"])
    )
    proposal = load_yaml(
        ROOT / str(config["direct_target_density_implementation"]["path"])
    )
    child_ids = {
        "stage_a_train": identity.stage_a_replacement_dataset_id,
        "stage_b_train": identity.stage_b_replacement_dataset_id,
    }
    namespace_configs: Dict[str, Mapping[str, Any]] = {}
    validations: Dict[str, Any] = {}
    replacement_groups: Dict[str, Dict[str, Set[str]]] = {}
    for component in CORRECTION_COMPONENTS:
        namespace_config = build_replacement_namespace_config(ROOT, config, component)
        namespace_configs[component] = namespace_config
        child = stage / child_ids[component]
        for name in ("attempts", "shards", "validation", "environment"):
            (child / name).mkdir(parents=True, exist_ok=True)
        _atomic_json(
            child / "run_manifest.json",
            {
                "status": "generating",
                "component": component,
                "split": "train",
                "dataset_id": child_ids[component],
                "generator_commit": implementation_commit,
                "configuration_hash": configuration_hash(namespace_config),
                "accepted_target": namespace_config["accepted_pair_count"],
                "all_importance_weights_one": True,
                "waveform_numerical_validity_enabled": True,
            },
        )
        generate_qualification_shard(
            shard_index=0,
            stage=child,
            config=namespace_config,
            preregistration=parent_preregistration,
            generator_git_commit=implementation_commit,
            dataset_id=child_ids[component],
            proposal_config=proposal,
        )
        validation, groups = validate_stage_a_namespace(
            child,
            namespace_config=namespace_config,
            expected_split=SplitName.TRAIN,
            expected_dataset=child_ids[component],
            generator_commit=implementation_commit,
        )
        validations[component] = validation
        replacement_groups[component] = {
            key: set(values) for key, values in groups.items() if key in GROUP_KEYS
        }

    base_paths = _base_paths(config)
    base_groups = {
        "stage_a_train": published_group_ids(base_paths["stage_a_train"]),
        "stage_a_validation": published_group_ids(base_paths["stage_a_validation"]),
        "stage_b_train": published_group_ids(base_paths["stage_b_train"]),
    }
    for component in CORRECTION_COMPONENTS:
        excluded = set(
            config["replacement_namespaces"][component][
                "excluded_physical_system_ids"
            ]
        )
        if not excluded <= base_groups[component]["system"]:
            raise ValueError(f"waveform-correction exclusions absent from {component}")
        for base_name, groups in base_groups.items():
            _assert_disjoint(
                replacement_groups[component], groups, f"{component}/{base_name}"
            )
    _assert_disjoint(
        replacement_groups["stage_a_train"],
        replacement_groups["stage_b_train"],
        "replacement cross-component",
    )

    views = {
        "stage_a_train": {
            "base_parent_root": str(base_paths["stage_a_parent"]),
            "base_dataset_id": config["immutable_base_publications"]["stage_a"][
                "train_dataset_id"
            ],
            "base_count": 32768,
            "excluded_physical_system_ids": config["replacement_namespaces"][
                "stage_a_train"
            ]["excluded_physical_system_ids"],
            "replacement_dataset_id": child_ids["stage_a_train"],
            "replacement_count": 2,
            "corrected_count": 32768,
        },
        "stage_a_validation": {
            "base_parent_root": str(base_paths["stage_a_parent"]),
            "base_dataset_id": config["immutable_base_publications"]["stage_a"][
                "validation_dataset_id"
            ],
            "corrected_count": 6144,
            "unchanged": True,
        },
        "stage_b_train": {
            "base_parent_root": str(base_paths["stage_b_parent"]),
            "base_dataset_id": config["immutable_base_publications"]["stage_b"][
                "train_dataset_id"
            ],
            "base_count": 32768,
            "excluded_physical_system_ids": config["replacement_namespaces"][
                "stage_b_train"
            ]["excluded_physical_system_ids"],
            "replacement_dataset_id": child_ids["stage_b_train"],
            "replacement_count": 3,
            "corrected_count": 32768,
        },
    }
    manifest = {
        "status": "passed",
        "parent_run_id": identity.parent_run_id,
        "generator_commit": implementation_commit,
        "configuration_hash": configuration_hash(config),
        "preregistration_version": config["preregistration"]["version"],
        "preregistration_hash": config["preregistration"]["canonical_hash"],
        "audit_sha256": config["audit"]["sha256"],
        "validations": validations,
        "views": views,
        "corrected_stage_a_train_count": 32768,
        "validation_count": 6144,
        "corrected_stage_b_train_count": 32768,
        "corrected_combined_train_count": 65536,
        "total_excluded_count": 5,
        "total_replacement_count": 5,
        "replacement_group_disjoint_from_all_base_components": True,
        "replacement_components_group_disjoint": True,
        "proposal_equals_evaluation": True,
        "all_importance_weights_one": True,
        "original_publications_modified": False,
        "model_training_authorized": False,
        "architecture_selection_authorized": False,
        "calibration_authorized": False,
        "sbc_authorized": False,
        "final_evaluation_authorized": False,
        "gwosc_gwtc_accessed": False,
    }
    _atomic_json(stage / "dataset_manifest.json", manifest)
    digest, byte_count = tree_checksum(stage)
    if byte_count > int(config["resource_gates"]["maximum_correction_output_bytes"]):
        raise RuntimeError("waveform-correction output exceeded frozen byte cap")
    publication.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, publication)
    remaining = shutil.disk_usage(publication).free
    if remaining < int(
        config["resource_gates"]["minimum_post_publication_free_bytes"]
    ):
        raise RuntimeError("waveform-correction post-publication free-space gate failed")
    result = {
        **manifest,
        "publication_path": str(publication),
        "publication_tree_sha256": digest,
        "publication_bytes": byte_count,
        "parent_manifest_sha256": _sha256(publication / "dataset_manifest.json"),
        "remaining_free_bytes": remaining,
    }
    _atomic_json(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "execute"))
    parser.add_argument("--config", default=CORRECTION_CONFIG)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--release-certificate", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.mode == "preflight":
        result = evaluate_release_gate(
            config_path=args.config, implementation_commit=args.implementation_commit
        )
        _atomic_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "ready_for_official_execution" else 2
    if args.release_certificate is None:
        raise ValueError("waveform-correction execute requires a release certificate")
    result = execute(
        config_path=args.config,
        implementation_commit=args.implementation_commit,
        certificate_path=args.release_certificate,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if "--output" in sys.argv:
            index = sys.argv.index("--output") + 1
            if index < len(sys.argv):
                _atomic_json(
                    Path(sys.argv[index]),
                    {
                        "status": "execution_failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "model_training_authorized": False,
                        "gwosc_gwtc_accessed": False,
                    },
                )
        raise
