#!/usr/bin/env python3
"""Run the exact parallel 32-accepted-pair Phase 3A microbenchmark gate."""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from gwlens_mm.config import load_yaml
from gwlens_mm.production.authorization import load_qualification_authorization
from gwlens_mm.production.generator import (
    GeneratedQualificationPair,
    QualificationGenerator,
)
from gwlens_mm.production.run_control import AttemptJournal, build_preflight_manifest
from gwlens_mm.production.storage import ShardWriter, tree_checksum
from gwlens_mm.provenance import configuration_hash, dataset_id

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class MicroWorkerResult:
    worker_index: int
    generated: List[Tuple[int, int, GeneratedQualificationPair]]
    attempt_count: int
    rejection_counts: Dict[str, int]
    family_attempts: Dict[str, int]
    timing_totals: Dict[str, float]
    elapsed_seconds: float


def _worker(
    worker_index: int,
    worker_count: int,
    accepted_count: int,
    maximum_attempts: int,
    stage: Path,
    micro_id: str,
    generator_commit: str,
) -> MicroWorkerResult:
    config = load_yaml(ROOT / "configs/data/phase3a_qualification.yaml")
    preregistration = load_yaml(ROOT / config["preregistration"]["path"])
    generator = QualificationGenerator(config, preregistration, generator_commit)
    accepted_indices = list(range(worker_index, accepted_count, worker_count))
    generated: List[Tuple[int, int, GeneratedQualificationPair]] = []
    rejection_counts: Counter[str] = Counter()
    family_attempts: Counter[str] = Counter()
    timing_totals: Dict[str, float] = defaultdict(float)
    local_attempt = 0
    started = time.perf_counter()
    journal_path = stage / "attempts" / f"worker-{worker_index:02d}.jsonl"
    with AttemptJournal(
        journal_path,
        first_attempt_id=worker_index,
        attempt_stride=worker_count,
    ) as journal:
        while len(generated) < len(accepted_indices):
            if local_attempt >= maximum_attempts:
                raise RuntimeError(f"worker {worker_index} exceeded maximum attempts")
            attempt_id = worker_index + local_attempt * worker_count
            accepted_index = accepted_indices[len(generated)]
            outcome = generator.generate_attempt(
                attempt_id=attempt_id,
                accepted_index=accepted_index,
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
                generated.append((accepted_index, attempt_id, outcome.generated))
            local_attempt += 1
    return MicroWorkerResult(
        worker_index,
        generated,
        local_attempt,
        dict(rejection_counts),
        dict(family_attempts),
        dict(timing_totals),
        time.perf_counter() - started,
    )


def _merge_counts(target: Counter[Any], values: Dict[Any, int]) -> None:
    for key, value in values.items():
        target[key] += value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(ROOT / "configs/data/phase3a_qualification.yaml")
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
    worker_count = int(config["microbenchmark"]["worker_processes"])
    accepted_target = int(config["microbenchmark"]["accepted_pair_count"])
    authorization.staging_root.mkdir(parents=True, exist_ok=True)
    stage = authorization.staging_root / micro_id
    if stage.exists():
        raise FileExistsError(f"microbenchmark staging already exists: {stage}")
    free_before = shutil.disk_usage(authorization.staging_root.parent).free
    if free_before < authorization.minimum_prelaunch_free_bytes:
        raise RuntimeError("microbenchmark free-space gate failed")
    preflight = build_preflight_manifest(
        authorization,
        config,
        run_id=f"phase3a-{micro_id}",
        dataset_id=micro_id,
        generator_git_commit=args.generator_commit,
        configuration_hash=config_hash,
    )
    (stage / "attempts").mkdir(parents=True)
    (stage / "run_manifest.json").write_text(
        json.dumps(
            {
                **preflight,
                "state": "microbenchmark_preflight_passed",
                "included_in_final_dataset": False,
                "accepted_pair_target": accepted_target,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    shards_root = stage / "shards"
    maximum_attempts = int(config["execution"]["maximum_attempts_per_worker"])
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _worker,
                worker,
                worker_count,
                accepted_target,
                maximum_attempts,
                stage,
                micro_id,
                args.generator_commit,
            )
            for worker in range(worker_count)
        ]
        worker_results = [future.result() for future in futures]
    generation_elapsed = time.perf_counter() - started
    accepted_rows = sorted(
        (item for result in worker_results for item in result.generated),
        key=lambda item: item[0],
    )
    if [item[0] for item in accepted_rows] != list(range(accepted_target)):
        raise RuntimeError("microbenchmark workers did not produce exact accepted indexes")
    writer = ShardWriter(
        shards_root,
        0,
        expected_pairs=accepted_target,
        sample_count=int(config["gw"]["sample_count"]),
    )
    rejection_counts: Counter[str] = Counter()
    family_attempts: Counter[str] = Counter()
    family_accepted: Counter[str] = Counter()
    multiplicities: Counter[int] = Counter()
    em_cells: Counter[str] = Counter()
    timing_totals: Dict[str, float] = defaultdict(float)
    whitened_by_detector: Dict[str, list[np.ndarray]] = defaultdict(list)
    for result in worker_results:
        _merge_counts(rejection_counts, result.rejection_counts)
        _merge_counts(family_attempts, result.family_attempts)
        for name, value in result.timing_totals.items():
            timing_totals[name] += value
    storage_started = time.perf_counter()
    for _, attempt_id, generated in accepted_rows:
        writer.append(
            generated.record,
            generated.products.noisy,
            generated.products.clean,
            generated.products.noise,
            attempt_id=attempt_id,
            partition_metadata={"em_cell": generated.em_cell},
        )
        family_accepted[generated.record.pair.lens_family.value] += 1
        multiplicities[generated.image_multiplicity] += 1
        em_cells[generated.em_cell] += 1
        for detector_index, detector in enumerate(("H1", "L1", "V1")):
            whitened_by_detector[detector].append(
                generated.products.whitened_noise[:, detector_index].reshape(-1)
            )
    storage_write_seconds = time.perf_counter() - storage_started
    checksum_started = time.perf_counter()
    writer.finalize()
    shard_hash, staged_bytes = tree_checksum(shards_root / "shard-00000")
    checksum_seconds = time.perf_counter() - checksum_started
    elapsed = time.perf_counter() - started
    free_after = shutil.disk_usage(authorization.staging_root.parent).free
    qualification_workers = int(config["execution"]["qualification_worker_processes"])
    projected_walltime_hours = (
        generation_elapsed
        / accepted_target
        * int(config["accepted_pair_count"])
        * worker_count
        / qualification_workers
        / 3600.0
    )
    projected_output_bytes = int(
        staged_bytes / accepted_target * int(config["accepted_pair_count"])
    )
    whitening: Dict[str, Dict[str, Any]] = {}
    for detector, pieces in whitened_by_detector.items():
        values = np.concatenate(pieces)
        whitening[detector] = {
            "finite_fraction": float(np.mean(np.isfinite(values))),
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
            "quantiles": np.quantile(
                values, [0.001, 0.01, 0.5, 0.99, 0.999]
            ).tolist(),
            "outlier_count_abs_gt_8": int(np.sum(np.abs(values) > 8.0)),
        }
    expected_families = {"sie_external_shear", "epl_external_shear"}
    gates = {
        "walltime": projected_walltime_hours
        <= float(config["resource_gates"]["maximum_projected_walltime_hours"]),
        "output": projected_output_bytes
        < int(config["resource_gates"]["maximum_output_bytes"]),
        "remaining_free": free_after - projected_output_bytes
        >= int(config["resource_gates"]["minimum_post_run_free_bytes"]),
        "whitening": all(
            summary["finite_fraction"]
            == float(config["whitening"]["finite_fraction_required"])
            and abs(summary["mean"])
            <= float(config["whitening"]["aggregate_absolute_mean_maximum"])
            and float(config["whitening"]["aggregate_standard_deviation_minimum"])
            <= summary["standard_deviation"]
            <= float(config["whitening"]["aggregate_standard_deviation_maximum"])
            for summary in whitening.values()
        ),
        "em_balance": set(em_cells.values()) == {4} and len(em_cells) == 8,
        "both_lens_families": set(family_accepted) == expected_families,
        "double_and_quad_multiplicity": 2 in multiplicities
        and any(value >= 4 for value in multiplicities),
        "exact_accepted_count": len(accepted_rows) == accepted_target,
    }
    attempt_count = sum(result.attempt_count for result in worker_results)
    child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    result = {
        "status": "passed" if all(gates.values()) else "failed",
        "dataset_id": micro_id,
        "dataset_purpose": "generator_qualification_microbenchmark",
        "included_in_final_dataset": False,
        "generator_git_commit": args.generator_commit,
        "authorizing_git_commit": authorization.authorizing_git_commit,
        "preregistration_hash": authorization.preregistration_hash,
        "configuration_hash": config_hash,
        "accepted_pair_count": len(accepted_rows),
        "attempt_count": attempt_count,
        "acceptance_rate": len(accepted_rows) / attempt_count,
        "elapsed_seconds": elapsed,
        "generation_elapsed_seconds": generation_elapsed,
        "storage_write_seconds": storage_write_seconds,
        "accepted_pairs_per_hour": len(accepted_rows) / generation_elapsed * 3600.0,
        "attempts_per_hour": attempt_count / generation_elapsed * 3600.0,
        "projected_qualification_walltime_hours": projected_walltime_hours,
        "microbenchmark_worker_processes": worker_count,
        "qualification_worker_processes": qualification_workers,
        "staged_bytes": staged_bytes,
        "projected_output_bytes": projected_output_bytes,
        "disk_amplification": staged_bytes
        / (accepted_target * 3 * 2 * 3 * int(config["gw"]["sample_count"]) * 4),
        "free_bytes_before": free_before,
        "free_bytes_after": free_after,
        "projected_remaining_free_bytes": free_after - projected_output_bytes,
        "peak_child_rss_kib": child_rss,
        "projected_worker_peak_rss_kib": child_rss * qualification_workers,
        "checksum_seconds": checksum_seconds,
        "shard_tree_sha256": shard_hash,
        "rejection_counts": dict(rejection_counts),
        "family_attempts": dict(family_attempts),
        "family_accepted": dict(family_accepted),
        "image_multiplicity_counts": dict(multiplicities),
        "em_cell_counts": dict(em_cells),
        "timing_totals_seconds": dict(timing_totals),
        "worker_summaries": [
            {
                "worker_index": item.worker_index,
                "attempt_count": item.attempt_count,
                "accepted_count": len(item.generated),
                "elapsed_seconds": item.elapsed_seconds,
            }
            for item in worker_results
        ],
        "whitening": whitening,
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not all(gates.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
