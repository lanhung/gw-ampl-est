"""Single fail-closed release gate for Phase 4 official execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .config import load_yaml
from .production.run_control import verify_psd_files
from .production.stage_a import (
    PHASE4_CONFIG,
    load_phase4_contract,
    validate_canary_manifest,
    verify_generator_commit,
)
from .provenance import configuration_hash, dataset_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_branch(root: Path) -> str:
    return subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_clean(root: Path) -> bool:
    return not subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _official_identities(config: Mapping[str, Any], commit: str) -> Dict[str, str]:
    config_hash = configuration_hash(config)
    train_seed = int(config["stage_a"]["train"]["root_seed"])
    validation_seed = int(config["stage_a"]["validation"]["root_seed"])
    parent = f"phase4-stage-a-{commit[:12]}-{config_hash[:12]}"
    train = dataset_id("2.0.0-alpha.3", commit, config_hash, train_seed) + "-train"
    validation = (
        dataset_id("2.0.0-alpha.3", commit, config_hash, validation_seed) + "-validation"
    )
    if train == validation:
        raise ValueError("official Stage A dataset identities collide")
    return {
        "parent_run_id": parent,
        "train_dataset_id": train,
        "validation_dataset_id": validation,
    }


def evaluate_phase4_release_gate(
    root: Path,
    *,
    generator_commit: str,
    config_path: str = PHASE4_CONFIG,
) -> Dict[str, Any]:
    """Evaluate every pre-execution condition without creating official identities early."""

    blockers: list[str] = []
    checks: Dict[str, Any] = {}
    try:
        config, preregistration, design = load_phase4_contract(root, config_path)
        checks["static_contract"] = "passed"
    except Exception as error:
        return {
            "status": "blocked_preexecution",
            "generator_commit": generator_commit,
            "checks": {"static_contract": f"failed:{type(error).__name__}"},
            "blockers": [str(error)],
            "official_identities": None,
        }
    try:
        verify_generator_commit(root, generator_commit)
        checks["generator_commit"] = "passed"
    except Exception as error:
        checks["generator_commit"] = "failed"
        blockers.append(str(error))
    if (root / ".git").exists():
        branch = _git_branch(root)
        checks["branch"] = branch
        if branch != "phase4/direct-target-stage-a":
            blockers.append("checkout is not on the Phase 4 release branch")
        clean = _git_clean(root)
        checks["clean_worktree"] = clean
        if not clean:
            blockers.append("working tree is not clean")
    environment = load_yaml(root / config["environment"]["lock_path"])
    dependency_lock = root / config["environment"]["dependency_lock_path"]
    dependency_hash = _sha256(dependency_lock)
    checks["dependency_lock_sha256"] = dependency_hash
    if dependency_hash != environment.get("dependency_lock_sha256"):
        blockers.append("dependency lock hash mismatch")
    release = config["release"]
    if release.get("final_generator_commit") != generator_commit:
        blockers.append("final generator commit is unresolved or mismatched")
    wheel_hash = release.get("generator_wheel_sha256")
    if not isinstance(wheel_hash, str) or len(wheel_hash) != 64:
        blockers.append("generator wheel SHA-256 is unresolved")
    future_path = config["authorization"].get("future_execution_path")
    execution_authorization: Mapping[str, Any] | None = None
    if not future_path:
        blockers.append("future Stage A execution authorization is absent")
    else:
        execution_authorization = load_yaml(root / str(future_path))
        flags = execution_authorization.get("authorization", {})
        for key in (
            "disposable_canary_accepted",
            "scientific_data_generation_authorized",
            "stage_a_materialization_authorized",
        ):
            if flags.get(key) is not True:
                blockers.append(f"execution authorization requires {key}=true")
        for key in (
            "model_training_authorized",
            "calibration_authorized",
            "sbc_authorized",
            "iid_ood_mismatch_evaluation_authorized",
            "gwosc_gwtc_access_authorized",
        ):
            if flags.get(key) is not False:
                blockers.append(f"execution authorization requires {key}=false")
        counts = execution_authorization.get("stage_a_contract", {})
        if (
            counts.get("train_accepted_count"),
            counts.get("validation_accepted_count"),
            counts.get("total_accepted_count"),
        ) != (32768, 6144, 38912):
            blockers.append("execution authorization Stage A counts mismatch")
    canary_path = release.get("canary_manifest_path")
    if not canary_path:
        blockers.append("disposable canary manifest is unresolved")
    else:
        manifest_path = Path(str(canary_path))
        if not manifest_path.is_file():
            blockers.append("disposable canary manifest does not exist")
        else:
            expected_hash = release.get("canary_manifest_sha256")
            actual_hash = _sha256(manifest_path)
            checks["canary_manifest_sha256"] = actual_hash
            if actual_hash != expected_hash:
                blockers.append("disposable canary manifest hash mismatch")
            try:
                validate_canary_manifest(json.loads(manifest_path.read_text()), generator_commit)
                checks["disposable_canary"] = "passed"
            except Exception as error:
                blockers.append(str(error))
    base = load_yaml(root / config["base_data_config"])
    try:
        checks["psd_files"] = verify_psd_files(base["gw"]["psd_curves"])
    except Exception as error:
        blockers.append(f"PSD verification failed: {error}")
    staging = Path(config["paths"]["stage_a_staging_root"])
    publication = Path(config["paths"]["stage_a_publication_root"])
    try:
        staging_parent_exists = staging.parent.exists()
    except OSError as error:
        staging_parent_exists = False
        blockers.append(f"Stage A filesystem inspection failed: {error}")
    if staging_parent_exists:
        free = shutil.disk_usage(staging.parent).free
        checks["free_bytes"] = free
        if free < int(config["resource_gates"]["minimum_prelaunch_free_bytes"]):
            blockers.append("Stage A free-space gate failed")
    elif not any(item.startswith("Stage A filesystem inspection failed:") for item in blockers):
        blockers.append("Stage A filesystem root does not exist")
    try:
        if publication.exists() and any(publication.iterdir()):
            blockers.append("Stage A publication root is not empty")
    except OSError as error:
        blockers.append(f"Stage A publication inspection failed: {error}")
    status = "ready_for_official_execution" if not blockers else "blocked_preexecution"
    identities = _official_identities(config, generator_commit) if not blockers else None
    return {
        "status": status,
        "phase": "4",
        "generator_commit": generator_commit,
        "preregistration_version": preregistration["preregistration_version"],
        "preregistration_hash": config["preregistration"]["canonical_hash"],
        "checks": checks,
        "blockers": blockers,
        "official_identities": identities,
        "scientific_data_generation_authorized": status == "ready_for_official_execution",
        "model_training_authorized": False,
        "gwosc_gwtc_access_authorized": False,
        "design_authorization_status": design["authorization_status"],
        "execution_authorization_loaded": execution_authorization is not None,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("phase4",))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", default=PHASE4_CONFIG)
    parser.add_argument("--generator-commit")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    commit = args.generator_commit
    if commit is None:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=args.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    result = evaluate_phase4_release_gate(
        args.root,
        generator_commit=commit,
        config_path=args.config,
    )
    if args.output is not None:
        _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ready_for_official_execution" else 2


if __name__ == "__main__":
    raise SystemExit(main())
