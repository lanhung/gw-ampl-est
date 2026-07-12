#!/usr/bin/env python3
"""Run the exact 32-accepted-pair Phase 3A microbenchmark gate."""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

from gwlens_mm.config import load_yaml
from gwlens_mm.production.authorization import load_qualification_authorization
from gwlens_mm.production.generator import QualificationGenerator
from gwlens_mm.production.run_control import AttemptJournal
from gwlens_mm.production.storage import ShardWriter, tree_checksum
from gwlens_mm.provenance import configuration_hash, dataset_id

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(ROOT / "configs/data/phase3a_qualification.yaml")
    preregistration = load_yaml(ROOT / config["preregistration"]["path"])
    authorization = load_qualification_authorization(
        ROOT / config["authorization"]["path"],
        repository_root=ROOT,
        authorizing_git_commit=config["authorization"]["authorizing_git_commit"],
    )
    config_hash = configuration_hash(config)
    base_id = dataset_id(
        config["schema_version"],
        args.generator_commit,
        config_hash,
        int(config["root_seed"]),
    )
    micro_id = f"{base_id}-microbenchmark"
    stage = authorization.staging_root / micro_id
    if stage.exists():
        raise FileExistsError(f"microbenchmark staging already exists: {stage}")
    free_before = shutil.disk_usage(authorization.staging_root.parent).free
    if free_before < authorization.minimum_prelaunch_free_bytes:
        raise RuntimeError("microbenchmark free-space gate failed")
    (stage / "attempts").mkdir(parents=True)
    shards_root = stage / "shards"
    writer = ShardWriter(
        shards_root,
        0,
        expected_pairs=int(config["microbenchmark"]["accepted_pair_count"]),
        sample_count=int(config["gw"]["sample_count"]),
    )
    generator = QualificationGenerator(config, preregistration, args.generator_commit)
    accepted = 0
    attempt_id = 0
    rejection_counts: Counter[str] = Counter()
    family_attempts: Counter[str] = Counter()
    family_accepted: Counter[str] = Counter()
    multiplicities: Counter[int] = Counter()
    em_cells: Counter[str] = Counter()
    timing_totals: Dict[str, float] = defaultdict(float)
    whitened_by_detector: Dict[str, list[np.ndarray]] = defaultdict(list)
    started = time.perf_counter()
    with AttemptJournal(stage / "attempts" / "attempts.jsonl") as journal:
        while accepted < int(config["microbenchmark"]["accepted_pair_count"]):
            outcome = generator.generate_attempt(
                attempt_id=attempt_id,
                accepted_index=accepted,
                dataset_id=micro_id,
            )
            journal.append(outcome.attempt)
            family_attempts[outcome.attempt.lens_family] += 1
            for name, value in outcome.timings.items():
                if name.endswith("_seconds"):
                    timing_totals[name] += float(value)
            if outcome.generated is None:
                rejection_counts[str(outcome.attempt.rejection_reason)] += 1
            else:
                generated = outcome.generated
                writer.append(
                    generated.record,
                    generated.products.noisy,
                    generated.products.clean,
                    generated.products.noise,
                    attempt_id=attempt_id,
                    partition_metadata={"em_cell": generated.em_cell},
                )
                family_accepted[outcome.attempt.lens_family] += 1
                multiplicities[generated.image_multiplicity] += 1
                em_cells[generated.em_cell] += 1
                for detector_index, detector in enumerate(("H1", "L1", "V1")):
                    whitened_by_detector[detector].append(
                        generated.products.whitened_noise[:, detector_index].reshape(-1)
                    )
                accepted += 1
            attempt_id += 1
    checksum_started = time.perf_counter()
    writer.finalize()
    shard_hash, staged_bytes = tree_checksum(shards_root / "shard-00000")
    checksum_seconds = time.perf_counter() - checksum_started
    elapsed = time.perf_counter() - started
    free_after = shutil.disk_usage(authorization.staging_root.parent).free
    projected_walltime_hours = elapsed / accepted * int(config["accepted_pair_count"]) / 3600.0
    projected_output_bytes = int(staged_bytes / accepted * int(config["accepted_pair_count"]))
    whitening: Dict[str, Dict[str, Any]] = {}
    for detector, pieces in whitened_by_detector.items():
        values = np.concatenate(pieces)
        whitening[detector] = {
            "finite_fraction": float(np.mean(np.isfinite(values))),
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
            "quantiles": np.quantile(values, [0.001, 0.01, 0.5, 0.99, 0.999]).tolist(),
            "outlier_count_abs_gt_8": int(np.sum(np.abs(values) > 8.0)),
        }
    gates = {
        "walltime": projected_walltime_hours
        <= float(config["resource_gates"]["maximum_projected_walltime_hours"]),
        "output": projected_output_bytes
        < int(config["resource_gates"]["maximum_output_bytes"]),
        "remaining_free": free_after - projected_output_bytes
        >= int(config["resource_gates"]["minimum_post_run_free_bytes"]),
        "whitening": all(
            summary["finite_fraction"] == float(config["whitening"]["finite_fraction_required"])
            and float(config["whitening"]["aggregate_standard_deviation_minimum"])
            <= summary["standard_deviation"]
            <= float(config["whitening"]["aggregate_standard_deviation_maximum"])
            for summary in whitening.values()
        ),
        "em_balance": set(em_cells.values()) == {4} and len(em_cells) == 8,
    }
    result = {
        "status": "passed" if all(gates.values()) else "failed",
        "dataset_id": micro_id,
        "generator_git_commit": args.generator_commit,
        "configuration_hash": config_hash,
        "accepted_pair_count": accepted,
        "attempt_count": attempt_id,
        "acceptance_rate": accepted / attempt_id,
        "elapsed_seconds": elapsed,
        "accepted_pairs_per_hour": accepted / elapsed * 3600.0,
        "attempts_per_hour": attempt_id / elapsed * 3600.0,
        "projected_qualification_walltime_hours": projected_walltime_hours,
        "staged_bytes": staged_bytes,
        "projected_output_bytes": projected_output_bytes,
        "free_bytes_before": free_before,
        "free_bytes_after": free_after,
        "projected_remaining_free_bytes": free_after - projected_output_bytes,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "checksum_seconds": checksum_seconds,
        "shard_tree_sha256": shard_hash,
        "rejection_counts": dict(rejection_counts),
        "family_attempts": dict(family_attempts),
        "family_accepted": dict(family_accepted),
        "image_multiplicity_counts": dict(multiplicities),
        "em_cell_counts": dict(em_cells),
        "timing_totals_seconds": dict(timing_totals),
        "whitening": whitening,
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not all(gates.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
