#!/usr/bin/env python3
"""Run, intentionally interrupt, resume, validate, and publish Phase 3A."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Mapping

from gwlens_mm.config import load_yaml
from gwlens_mm.policy import InputPolicy
from gwlens_mm.production.authorization import load_qualification_authorization
from gwlens_mm.production.qualification import (
    complete_shard_hash_lines,
    generate_qualification_shard,
    load_completed_shard_summary,
    publish_qualification_dataset,
    validate_qualification_dataset,
)
from gwlens_mm.production.run_control import build_preflight_manifest
from gwlens_mm.production.storage import sha256_file, verify_complete_shard
from gwlens_mm.provenance import configuration_hash, dataset_id

ROOT = Path(__file__).resolve().parents[2]


def _atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.partial")
    temporary.write_text(
        json.dumps(dict(data), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _environment() -> Dict[str, Any]:
    packages = {}
    for name in (
        "numpy",
        "scipy",
        "pandas",
        "zarr",
        "numcodecs",
        "pyarrow",
        "bilby",
        "lalsuite",
        "lenstronomy",
        "astropy",
    ):
        packages[name] = importlib.metadata.version(name)
    memory = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            memory[key] = value.strip()
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.SubprocessError):
        gpu = []
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory": memory,
        "gpu_provenance_only": gpu,
        "packages": packages,
    }


def _merge_counter(target: Counter[Any], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        target[key] += int(value)


def _aggregate_summaries(summaries: list[Mapping[str, Any]]) -> Dict[str, Any]:
    rejection_counts: Counter[str] = Counter()
    family_attempts: Counter[str] = Counter()
    family_accepted: Counter[str] = Counter()
    multiplicities: Counter[str] = Counter()
    em_cells: Counter[str] = Counter()
    timing_totals: Dict[str, float] = defaultdict(float)
    attempt_count = 0
    accepted_count = 0
    shard_bytes = 0
    for summary in summaries:
        attempt_count += int(summary["attempt_count"])
        accepted_count += int(summary["accepted_pair_count"])
        shard_bytes += int(summary["shard_bytes"])
        _merge_counter(rejection_counts, summary["rejection_counts"])
        _merge_counter(family_attempts, summary["family_attempts"])
        _merge_counter(family_accepted, summary["family_accepted"])
        _merge_counter(multiplicities, summary["image_multiplicity_counts"])
        _merge_counter(em_cells, summary["em_cell_counts"])
        for key, value in summary["timing_totals_seconds"].items():
            timing_totals[key] += float(value)
    return {
        "attempt_count": attempt_count,
        "accepted_pair_count": accepted_count,
        "acceptance_rate": accepted_count / attempt_count,
        "shard_bytes": shard_bytes,
        "rejection_counts": dict(rejection_counts),
        "family_attempts": dict(family_attempts),
        "family_accepted": dict(family_accepted),
        "image_multiplicity_counts": dict(multiplicities),
        "em_cell_counts": dict(em_cells),
        "timing_totals_seconds": dict(timing_totals),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-complete-shards", type=int)
    args = parser.parse_args()
    if len(args.generator_commit) != 40:
        raise ValueError("generator commit must be a full Git hash")
    config = load_yaml(ROOT / "configs/data/phase3a_qualification.yaml")
    preregistration = load_yaml(ROOT / config["preregistration"]["path"])
    authorization = load_qualification_authorization(
        ROOT / config["authorization"]["path"],
        repository_root=ROOT,
        authorizing_git_commit=config["authorization"]["authorizing_git_commit"],
    )
    config_hash = configuration_hash(config)
    identifier = dataset_id(
        str(config["schema_version"]),
        args.generator_commit,
        config_hash,
        int(config["root_seed"]),
    )
    run_id = f"phase3a-{identifier}"
    stage = authorization.staging_root / identifier
    publication = authorization.publication_root / identifier
    for root in (
        authorization.staging_root,
        authorization.publication_root,
        authorization.manifest_root,
        authorization.log_root,
    ):
        root.mkdir(parents=True, exist_ok=True)
    if publication.exists():
        raise FileExistsError(f"qualification publication already exists: {publication}")
    run_manifest_path = stage / "run_manifest.json"
    active_started = time.perf_counter()
    if args.resume:
        if not stage.is_dir() or not run_manifest_path.is_file():
            raise FileNotFoundError("resume requires an existing staged run manifest")
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        expected_identity = {
            "run_id": run_id,
            "dataset_id": identifier,
            "generator_git_commit": args.generator_commit,
            "configuration_hash": config_hash,
            "authorizing_git_commit": authorization.authorizing_git_commit,
            "preregistration_hash": authorization.preregistration_hash,
        }
        for key, value in expected_identity.items():
            if run_manifest.get(key) != value:
                raise ValueError(f"resume identity mismatch: {key}")
    else:
        if stage.exists():
            raise FileExistsError(f"new run staging already exists: {stage}")
        preflight = build_preflight_manifest(
            authorization,
            config,
            run_id=run_id,
            dataset_id=identifier,
            generator_git_commit=args.generator_commit,
            configuration_hash=config_hash,
        )
        stage.mkdir(parents=True)
        (stage / "attempts").mkdir()
        (stage / "shards").mkdir()
        (stage / "validation").mkdir()
        (stage / "environment").mkdir()
        environment_path = stage / "environment" / "environment.json"
        _atomic_json(environment_path, _environment())
        run_manifest = {
            **preflight,
            "base_main_commit": authorization.base_main_commit,
            "active_elapsed_seconds": 0.0,
            "complete_shard_count": 0,
            "environment_sha256": sha256_file(environment_path),
            "state": "ready_for_generation",
        }
        _atomic_json(run_manifest_path, run_manifest)
    environment_path = stage / "environment" / "environment.json"
    if run_manifest.get("environment_sha256") != sha256_file(environment_path):
        raise ValueError("resume environment evidence changed")
    partials = sorted((stage / "shards").glob("shard-*.partial"))
    if partials:
        raise RuntimeError(
            "retained partial shard requires human-reviewed new run identity: "
            + ", ".join(path.name for path in partials)
        )
    completed = []
    total_shards = int(config["shard_count"])
    for index in range(total_shards):
        path = stage / "shards" / f"shard-{index:05d}"
        if path.exists():
            verify_complete_shard(path, int(config["pairs_per_shard"]))
            completed.append(index)
    if completed != list(range(len(completed))):
        raise ValueError("completed shards are not a contiguous immutable prefix")
    interruption_count = int(config["interruption_test"]["stop_after_complete_shards"])
    pre_hash_path = authorization.manifest_root / f"{identifier}.pre_resume_hashes.sha256"
    post_hash_path = authorization.manifest_root / f"{identifier}.post_resume_hashes.sha256"
    if args.resume:
        if len(completed) < interruption_count or not pre_hash_path.is_file():
            raise ValueError("resume requires the reviewed three-shard interruption state")
        expected_lines = tuple(pre_hash_path.read_text(encoding="utf-8").splitlines())
        actual_lines = complete_shard_hash_lines(
            stage, tuple(range(interruption_count))
        )
        if actual_lines != expected_lines:
            raise ValueError("pre-resume shard hashes changed before resume")
    target = total_shards
    if args.stop_after_complete_shards is not None:
        if args.stop_after_complete_shards != interruption_count:
            raise ValueError("only the reviewed three-shard interruption is allowed")
        target = interruption_count
    if len(completed) > target:
        raise ValueError("requested stop target is behind existing complete shards")
    pending = list(range(len(completed), target))
    if pending:
        workers = min(
            int(config["execution"]["qualification_worker_processes"]), len(pending)
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    generate_qualification_shard,
                    shard_index=index,
                    stage=stage,
                    config=config,
                    preregistration=preregistration,
                    generator_git_commit=args.generator_commit,
                    dataset_id=identifier,
                )
                for index in pending
            ]
            for future in futures:
                future.result()
    completed = list(range(target))
    summaries = [load_completed_shard_summary(stage, index) for index in completed]
    run_manifest["active_elapsed_seconds"] = float(
        run_manifest.get("active_elapsed_seconds", 0.0)
    ) + (time.perf_counter() - active_started)
    run_manifest["complete_shard_count"] = len(completed)
    aggregate = _aggregate_summaries(summaries)
    if target < total_shards:
        lines = complete_shard_hash_lines(stage, tuple(range(interruption_count)))
        pre_hash_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        run_manifest["state"] = "intentional_stop_after_three_complete_shards"
        _atomic_json(run_manifest_path, run_manifest)
        result = {
            "status": "intentional_stop",
            "dataset_id": identifier,
            "run_id": run_id,
            "generator_git_commit": args.generator_commit,
            "configuration_hash": config_hash,
            "complete_shard_count": len(completed),
            "pre_resume_hash_file": str(pre_hash_path),
            **aggregate,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    post_lines = complete_shard_hash_lines(stage, tuple(range(interruption_count)))
    post_hash_path.write_text("\n".join(post_lines) + "\n", encoding="utf-8")
    if tuple(pre_hash_path.read_text(encoding="utf-8").splitlines()) != post_lines:
        raise ValueError("first three shards are not byte-identical after resume")
    input_policy = InputPolicy.from_files(
        ROOT / "configs/policy/deployable_input_allowlist.json",
        ROOT / "configs/policy/privileged_input_denylist.json",
    )
    validation, identities, artifacts = validate_qualification_dataset(
        stage=stage,
        config=config,
        authorization=authorization,
        generator_git_commit=args.generator_commit,
        configuration_hash=config_hash,
        dataset_id=identifier,
        input_policy=input_policy,
    )
    run_manifest["state"] = "validated_ready_for_atomic_publication"
    _atomic_json(run_manifest_path, run_manifest)
    publication_result = publish_qualification_dataset(
        stage=stage,
        publication=publication,
        config=config,
        authorization=authorization,
        generator_git_commit=args.generator_commit,
        configuration_hash=config_hash,
        dataset_id=identifier,
        validation=validation,
        identities=identities,
        shard_artifacts=artifacts,
        environment_file=environment_path,
    )
    result = {
        **publication_result,
        "run_id": run_id,
        "active_elapsed_seconds": run_manifest["active_elapsed_seconds"],
        "accepted_pairs_per_hour": int(config["accepted_pair_count"])
        / float(run_manifest["active_elapsed_seconds"])
        * 3600.0,
        "pre_resume_hash_file": str(pre_hash_path),
        "post_resume_hash_file": str(post_hash_path),
        "resume_byte_identical": True,
        "validation": validation,
        **aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
