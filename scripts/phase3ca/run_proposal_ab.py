#!/usr/bin/env python3
"""Execute the bounded, sequential Phase 3C-A engineering A/B qualification."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from gwlens_mm.config import load_yaml
from gwlens_mm.policy import InputPolicy
from gwlens_mm.production.ab_qualification import (
    ARM_NAMES,
    _atomic_json,
    arm_config,
    bootstrap_throughput,
    load_and_verify_contract,
    postselection_diagnostics,
    preflight,
    publish_arm,
    strip_large_validation,
    validate_arm,
    validate_first_block_health,
)
from gwlens_mm.production.qualification import generate_qualification_shard
from gwlens_mm.production.storage import tree_checksum, verify_complete_shard
from gwlens_mm.provenance import configuration_hash

ROOT = Path(__file__).resolve().parents[2]


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
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            memory[key] = value.strip()
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory": memory,
        "packages": packages,
    }


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _summary_path(stage: Path, block: int) -> Path:
    return stage / "attempts" / f"shard-{block:05d}.summary.json"


def _load_summary(stage: Path, block: int) -> Dict[str, Any]:
    return json.loads(_summary_path(stage, block).read_text())


def _write_telemetry(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _atomic_list(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--config", default="configs/data/phase3ca_proposal_v3_ab.yaml"
    )
    args = parser.parse_args()
    config, adaptive, proposal, identity = load_and_verify_contract(
        ROOT, args.generator_commit, args.config
    )
    synced = ROOT / "SYNCED_COMMIT"
    if not (ROOT / ".git").exists() and (
        not synced.is_file() or synced.read_text().strip() != args.generator_commit
    ):
        raise ValueError("remote disposable checkout lacks the exact synchronized commit marker")
    pre = preflight(ROOT, config, identity)
    paths = {name: Path(value) for name, value in config["paths"].items()}
    parent_stage = paths["staging_root"] / identity.parent_run_id
    parent_publication = paths["publication_root"] / identity.parent_run_id
    parent_stage.mkdir(parents=True, exist_ok=True)
    environment_path = parent_stage / "environment.json"
    if environment_path.exists():
        environment = json.loads(environment_path.read_text())
        if environment != _environment():
            raise ValueError("resume environment identity changed")
    else:
        environment = _environment()
        _atomic_json(environment_path, environment)
    _atomic_json(
        parent_stage / "preflight.json",
        {
            **pre,
            "environment": environment,
            "adaptive_preregistration_hash": config["adaptive_preregistration"]["canonical_hash"],
            "proposal_hash": config["proposal"]["canonical_hash"],
        },
    )
    arm_ids = {
        "rc5_control": identity.control_dataset_id,
        "proposal_v3_candidate": identity.candidate_dataset_id,
    }
    arm_cfgs = {arm: arm_config(ROOT, config, arm) for arm in ARM_NAMES}
    for arm in ARM_NAMES:
        stage = parent_stage / arm_ids[arm]
        for child in ("attempts", "shards", "validation", "environment"):
            (stage / child).mkdir(parents=True, exist_ok=True)
        _atomic_json(stage / "environment" / "environment.json", environment)
        _atomic_json(
            stage / "run_manifest.json",
            {
                "parent_run_id": identity.parent_run_id,
                "dataset_id": arm_ids[arm],
                "arm": arm,
                "generator_commit": identity.generator_commit,
                "arm_configuration_hash": configuration_hash(arm_cfgs[arm]),
                "authorizing_commit": identity.authorizing_commit,
                "state": "ready_or_resuming",
                "accepted_target": 512,
                "block_target": 16,
                "scientific_use_authorized": False,
                "training_use_authorized": False,
            },
        )
    proposal_config = load_yaml(ROOT / config["proposal"]["path"])
    telemetry: list[Dict[str, Any]] = []
    existing_telemetry = parent_stage / "block_telemetry.json"
    if existing_telemetry.exists():
        telemetry = json.loads(existing_telemetry.read_text())
    health_results: Dict[str, Any] = {}
    pre_hash_path = paths["manifest_root"] / f"{identity.parent_run_id}.pre_resume_hashes.sha256"
    post_hash_path = paths["manifest_root"] / f"{identity.parent_run_id}.post_resume_hashes.sha256"
    for block in range(16):
        order = (
            config["execution"]["even_block_order"]
            if block % 2 == 0
            else config["execution"]["odd_block_order"]
        )
        for arm in order:
            stage = parent_stage / arm_ids[arm]
            complete = stage / "shards" / f"shard-{block:05d}"
            if complete.exists():
                verify_complete_shard(complete, 32)
                if not any(
                    row["arm"] == arm and int(row["block_index"]) == block for row in telemetry
                ):
                    summary = _load_summary(stage, block)
                    telemetry.append(
                        {
                            "arm": arm,
                            "block_index": block,
                            "accepted_pairs": 32,
                            "attempts": int(summary["attempt_count"]),
                            "active_wall_seconds": float(summary["elapsed_seconds"]),
                            "cpu_seconds": None,
                            "peak_rss_bytes": None,
                            "peak_staging_bytes": _tree_bytes(parent_stage),
                            **{
                                f"timing_{key}": value
                                for key, value in summary["timing_totals_seconds"].items()
                            },
                        }
                    )
                continue
            arm_rows = [row for row in telemetry if row["arm"] == arm]
            attempts_used = sum(int(row["attempts"]) for row in arm_rows)
            active_used = sum(float(row["active_wall_seconds"]) for row in arm_rows)
            remaining_attempts = (
                int(config["execution"]["maximum_attempts_per_arm"]) - attempts_used
            )
            remaining_seconds = (
                float(config["execution"]["maximum_active_seconds_per_arm"]) - active_used
            )
            if remaining_attempts <= 0 or remaining_seconds <= 0:
                raise RuntimeError(f"{arm} reached its execution cap before block {block}")
            block_cfg = arm_cfgs[arm]
            before_usage = resource.getrusage(resource.RUSAGE_SELF)
            before_cpu = before_usage.ru_utime + before_usage.ru_stime
            wall_started = time.perf_counter()
            summary = generate_qualification_shard(
                shard_index=block,
                stage=stage,
                config=block_cfg,
                preregistration=load_yaml(ROOT / block_cfg["preregistration"]["path"]),
                generator_git_commit=identity.generator_commit,
                dataset_id=arm_ids[arm],
                proposal_config=proposal_config,
                maximum_attempts_override=remaining_attempts,
                maximum_active_seconds_override=remaining_seconds,
            )
            active = time.perf_counter() - wall_started
            after_usage = resource.getrusage(resource.RUSAGE_SELF)
            after_cpu = after_usage.ru_utime + after_usage.ru_stime
            telemetry.append(
                {
                    "arm": arm,
                    "block_index": block,
                    "execution_order_position": len(telemetry),
                    "accepted_pairs": 32,
                    "attempts": int(summary["attempt_count"]),
                    "active_wall_seconds": active,
                    "cpu_seconds": after_cpu - before_cpu,
                    "time_integrated_cpu_utilization": (after_cpu - before_cpu) / active,
                    "peak_rss_bytes": int(after_usage.ru_maxrss) * 1024,
                    "peak_staging_bytes": _tree_bytes(parent_stage),
                    "operator_pause_seconds": 0.0,
                    **{
                        f"timing_{key}": value
                        for key, value in summary["timing_totals_seconds"].items()
                    },
                }
            )
            _atomic_list(existing_telemetry, telemetry)
        if block == 0:
            health_results = {
                arm: validate_first_block_health(
                    parent_stage / arm_ids[arm], arm_ids[arm]
                )
                for arm in ARM_NAMES
            }
            lines = [
                f"{health_results[arm]['tree_sha256']}  {arm}/{arm_ids[arm]}/shards/shard-00000"
                for arm in ARM_NAMES
            ]
            pre_hash_path.parent.mkdir(parents=True, exist_ok=True)
            pre_hash_path.write_text("\n".join(lines) + "\n")
            _atomic_json(
                parent_stage / "first_matched_block_health.json",
                {"status": "passed", "arms": health_results, "interim_throughput_inspected": False},
            )
    if not health_results:
        health_results = {
            arm: validate_first_block_health(
                parent_stage / arm_ids[arm], arm_ids[arm]
            )
            for arm in ARM_NAMES
        }
    final_lines = []
    for arm in ARM_NAMES:
        digest, _ = tree_checksum(parent_stage / arm_ids[arm] / "shards" / "shard-00000")
        final_lines.append(f"{digest}  {arm}/{arm_ids[arm]}/shards/shard-00000")
    post_hash_path.write_text("\n".join(final_lines) + "\n")
    if pre_hash_path.read_text() != post_hash_path.read_text():
        raise ValueError("completed first matched blocks changed")
    policy = InputPolicy.from_files(
        ROOT / "configs/policy/deployable_input_allowlist.json",
        ROOT / "configs/policy/privileged_input_denylist.json",
    )
    validations = {
        arm: validate_arm(parent_stage / arm_ids[arm], arm_cfgs[arm], identity, arm, policy)
        for arm in ARM_NAMES
    }
    postselection = postselection_diagnostics(validations["proposal_v3_candidate"], config)
    throughput = bootstrap_throughput(telemetry, config)
    if postselection["status"] != "passed":
        outcome = "failed_retain_rc5"
    elif throughput["passed"]:
        outcome = "passed_for_future_scientific_production_review"
    elif throughput["point_estimate"] >= 2.0:
        outcome = "inconclusive_retain_rc5"
    else:
        outcome = "failed_retain_rc5"
    publications = {
        arm: publish_arm(
            parent_stage / arm_ids[arm],
            parent_publication / arm_ids[arm],
            validations[arm],
            identity,
            str(config["dataset_purpose"]),
        )
        for arm in ARM_NAMES
    }
    comparison = {
        "status": outcome,
        "parent_run_id": identity.parent_run_id,
        "generator_commit": identity.generator_commit,
        "configuration_hash": identity.configuration_hash,
        "arm_publications": publications,
        "throughput": throughput,
        "postselection": postselection,
        "resume_byte_identical": True,
        "accepted_pair_count": 1024,
        "scientific_use_authorized": False,
        "training_use_authorized": False,
        "stage_a_authorized": False,
        "gwosc_gwtc_accessed": False,
    }
    _atomic_json(parent_stage / "comparison_manifest.json", comparison)
    staged_digest, staged_bytes = tree_checksum(parent_stage)
    comparison["staged_parent_tree_sha256"] = staged_digest
    comparison["staged_parent_bytes"] = staged_bytes
    _atomic_json(parent_stage / "comparison_manifest.json", comparison)
    if parent_publication.exists():
        raise FileExistsError("A/B parent publication identity already exists")
    parent_publication.parent.mkdir(parents=True, exist_ok=True)
    os.replace(parent_stage, parent_publication)
    comparison_digest, comparison_bytes = tree_checksum(parent_publication)
    comparison["comparison_tree_sha256"] = comparison_digest
    comparison["comparison_published_bytes"] = comparison_bytes
    comparison["remaining_free_bytes"] = (
        os.statvfs(parent_publication).f_bavail * os.statvfs(parent_publication).f_frsize
    )
    _atomic_json(
        paths["manifest_root"] / f"{identity.parent_run_id}.comparison_manifest.json", comparison
    )
    _write_telemetry(
        paths["manifest_root"] / f"{identity.parent_run_id}.block_telemetry.csv", telemetry
    )
    result = {
        **comparison,
        "preflight": pre,
        "arm_validation": {
            arm: strip_large_validation(value) for arm, value in validations.items()
        },
        "health_gate": health_results,
        "pre_resume_hashes": str(pre_hash_path),
        "post_resume_hashes": str(post_hash_path),
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output" in sys.argv:
            output_index = sys.argv.index("--output") + 1
            if output_index < len(sys.argv):
                _atomic_json(
                    Path(sys.argv[output_index]),
                    {
                        "status": "execution_failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "scientific_use_authorized": False,
                        "training_use_authorized": False,
                        "stage_a_authorized": False,
                    },
                )
        raise
