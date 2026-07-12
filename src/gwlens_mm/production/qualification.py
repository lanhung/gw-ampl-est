"""Parallel atomic-shard generation and streaming Phase 3A validation."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from ..arrays import validate_strain_array_semantics
from ..policy import InputPolicy
from ..provenance import ArtifactChecksum, DatasetManifest, canonical_json
from ..schema import ArrayProductRole, SplitName, V2Record
from .authorization import QualificationAuthorization
from .generator import QualificationGenerator
from .run_control import AttemptJournal, AttemptRecord
from .storage import ShardWriter, sha256_file, tree_checksum, verify_complete_shard


def _atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.partial")
    temporary.write_text(
        json.dumps(dict(data), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def generate_qualification_shard(
    *,
    shard_index: int,
    stage: Path,
    config: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    generator_git_commit: str,
    dataset_id: str,
) -> Dict[str, Any]:
    """Generate one deterministic shard in one owned worker process."""

    pairs_per_shard = int(config["pairs_per_shard"])
    stride = int(config["execution"]["attempt_id_stride"])
    maximum_attempts = int(config["execution"]["maximum_attempts_per_worker"])
    if stride < int(config["shard_count"]) or not 0 <= shard_index < stride:
        raise ValueError("attempt stride does not provide disjoint shard sequences")
    complete = stage / "shards" / f"shard-{shard_index:05d}"
    partial = stage / "shards" / f"shard-{shard_index:05d}.partial"
    journal_path = stage / "attempts" / f"shard-{shard_index:05d}.jsonl"
    summary_path = stage / "attempts" / f"shard-{shard_index:05d}.summary.json"
    if complete.exists():
        raise FileExistsError(f"completed shard cannot be rewritten: {complete}")
    if partial.exists() or journal_path.exists() or summary_path.exists():
        raise RuntimeError(
            f"inconsistent retained partial evidence prevents shard resume: {shard_index}"
        )
    generator = QualificationGenerator(config, preregistration, generator_git_commit)
    writer = ShardWriter(
        stage / "shards",
        shard_index,
        expected_pairs=pairs_per_shard,
        sample_count=int(config["gw"]["sample_count"]),
    )
    accepted = 0
    local_attempt = 0
    rejection_counts: Counter[str] = Counter()
    family_attempts: Counter[str] = Counter()
    family_accepted: Counter[str] = Counter()
    multiplicities: Counter[int] = Counter()
    em_cells: Counter[str] = Counter()
    timing_totals: Dict[str, float] = defaultdict(float)
    whitening_moments = {
        detector: {"count": 0, "finite_count": 0, "sum": 0.0, "sum_squares": 0.0}
        for detector in ("H1", "L1", "V1")
    }
    started = time.perf_counter()
    with AttemptJournal(
        journal_path,
        first_attempt_id=shard_index,
        attempt_stride=stride,
    ) as journal:
        while accepted < pairs_per_shard:
            if local_attempt >= maximum_attempts:
                raise RuntimeError(f"shard {shard_index} exceeded maximum attempts")
            attempt_id = shard_index + local_attempt * stride
            accepted_index = shard_index * pairs_per_shard + accepted
            outcome = generator.generate_attempt(
                attempt_id=attempt_id,
                accepted_index=accepted_index,
                dataset_id=dataset_id,
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
                    values = np.asarray(
                        generated.products.whitened_noise[:, detector_index],
                        dtype=np.float64,
                    )
                    moments = whitening_moments[detector]
                    moments["count"] += int(values.size)
                    moments["finite_count"] += int(np.sum(np.isfinite(values)))
                    moments["sum"] += float(np.sum(values))
                    moments["sum_squares"] += float(np.sum(values**2))
                accepted += 1
            local_attempt += 1
    manifest = writer.finalize()
    shard_hash, shard_bytes = tree_checksum(complete)
    summary: Dict[str, Any] = {
        "status": "complete",
        "shard_index": shard_index,
        "accepted_pair_count": accepted,
        "attempt_count": local_attempt,
        "first_attempt_id": shard_index,
        "last_attempt_id": shard_index + (local_attempt - 1) * stride,
        "elapsed_seconds": time.perf_counter() - started,
        "shard_tree_sha256": shard_hash,
        "shard_bytes": shard_bytes,
        "manifest_pair_ids_sha256": hashlib.sha256(
            canonical_json(manifest.pair_ids).encode()
        ).hexdigest(),
        "rejection_counts": dict(rejection_counts),
        "family_attempts": dict(family_attempts),
        "family_accepted": dict(family_accepted),
        "image_multiplicity_counts": dict(multiplicities),
        "em_cell_counts": dict(em_cells),
        "timing_totals_seconds": dict(timing_totals),
        "whitening_moments": whitening_moments,
    }
    _atomic_json(summary_path, summary)
    return summary


def load_completed_shard_summary(stage: Path, shard_index: int) -> Dict[str, Any]:
    path = stage / "attempts" / f"shard-{shard_index:05d}.summary.json"
    if not path.is_file():
        raise ValueError(f"completed shard summary is missing: {shard_index}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "complete" or data.get("shard_index") != shard_index:
        raise ValueError(f"invalid completed shard summary: {shard_index}")
    return data


def complete_shard_hash_lines(
    stage: Path, shard_indices: Sequence[int]
) -> Tuple[str, ...]:
    lines = []
    for index in shard_indices:
        path = stage / "shards" / f"shard-{index:05d}"
        digest, _ = tree_checksum(path)
        lines.append(f"{digest}  shards/shard-{index:05d}")
    return tuple(lines)


def _merge_counter(target: Counter[Any], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        target[key] += int(value)


def validate_qualification_dataset(
    *,
    stage: Path,
    config: Mapping[str, Any],
    authorization: QualificationAuthorization,
    generator_git_commit: str,
    configuration_hash: str,
    dataset_id: str,
    input_policy: InputPolicy,
) -> Tuple[Dict[str, Any], Dict[str, Tuple[str, ...]], Tuple[ArtifactChecksum, ...]]:
    """Stream every shard and array while collecting only small record metadata."""

    zarr = importlib.import_module("zarr")
    pandas = importlib.import_module("pandas")
    pair_ids: set[str] = set()
    source_ids: set[str] = set()
    lens_ids: set[str] = set()
    system_ids: set[str] = set()
    noise_ids: set[str] = set()
    attempt_ids: set[int] = set()
    attempt_system_ids: set[str] = set()
    accepted_attempt_pairs: set[str] = set()
    ordered_pair_ids: list[str] = []
    ordered_source_ids: list[str] = []
    ordered_lens_ids: list[str] = []
    ordered_noise_ids: list[str] = []
    artifacts = []
    family_counts: Counter[str] = Counter()
    multiplicities: Counter[int] = Counter()
    em_cells: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    attempted_count = 0
    accepted_attempt_count = 0
    stride = int(config["execution"]["attempt_id_stride"])
    expected_pairs = int(config["pairs_per_shard"])
    for shard_index in range(int(config["shard_count"])):
        shard_path = stage / "shards" / f"shard-{shard_index:05d}"
        verify_complete_shard(shard_path, expected_pairs)
        shard_digest, shard_bytes = tree_checksum(shard_path)
        artifacts.append(
            ArtifactChecksum(
                f"shards/shard-{shard_index:05d}",
                shard_digest,
                shard_bytes,
                "complete",
            )
        )
        frame = pandas.read_parquet(shard_path / "records.parquet")
        if len(frame) != expected_pairs:
            raise ValueError("Parquet record count disagrees with shard contract")
        arrays = {
            name: zarr.open_array(str(shard_path / f"{name}.zarr"), mode="r")
            for name in ("noisy", "clean", "noise")
        }
        expected_shape = (
            expected_pairs,
            2,
            3,
            int(config["gw"]["sample_count"]),
        )
        if any(array.shape != expected_shape for array in arrays.values()):
            raise ValueError("published Zarr shape disagrees with shard contract")
        for row_index, row in frame.iterrows():
            record = V2Record.from_json(str(row["record_json"]))
            if record.schema_version != str(config["schema_version"]):
                raise ValueError("qualification record has the wrong schema")
            if record.pair.split is not SplitName.GENERATOR_QUALIFICATION:
                raise ValueError("qualification record entered a scientific split")
            if record.pair.dataset_version != dataset_id:
                raise ValueError("record dataset ID mismatch")
            if record.provenance.generator_git_commit != generator_git_commit:
                raise ValueError("mixed generator commits across records")
            if record.provenance.configuration_hash != configuration_hash:
                raise ValueError("mixed configuration hashes across records")
            if record.gw_observation.preprocessing_version != str(
                config["gw"]["preprocessing_version"]
            ):
                raise ValueError("record preprocessing version mismatch")
            required_source = {
                "mass_1_source",
                "mass_2_source",
                "mass_ratio",
                "a_1",
                "a_2",
                "tilt_1",
                "tilt_2",
                "phi_12",
                "phi_jl",
            }
            if not required_source <= set(record.source_truth.intrinsic_parameters):
                raise ValueError("record omits frozen intrinsic source parameters")
            if not {"z_lens", "z_source"} <= set(record.lens_truth.lens_parameters):
                raise ValueError("record omits latent lens/source redshift truth")
            required_seed_keys = {
                "lens",
                "source",
                "pair_selection",
                "em_measurement_noise",
                "missing_modalities",
                "stellar_kinematics",
                "augmentation",
            }
            detector_seed_keys = {
                f"detector_noise:{image_id}:{detector}"
                for image_id in (
                    record.pair.primary_image_id,
                    record.pair.secondary_image_id,
                )
                for detector in ("H1", "L1", "V1")
            }
            if set(record.provenance.seed_hierarchy) != (
                required_seed_keys | detector_seed_keys
            ):
                raise ValueError("record seed hierarchy is incomplete or ambiguous")
            expected_paths = {
                role: f"shards/shard-{shard_index:05d}/{role.value}.zarr"
                for role in ArrayProductRole
            }
            if any(
                reference.uri != expected_paths[reference.product_role]
                for reference in record.gw_observation.array_products
            ):
                raise ValueError("record array reference points to the wrong shard")
            if record.pair.pair_id in pair_ids or record.pair.source_id in source_ids:
                raise ValueError("cross-shard pair/source duplicate")
            if record.pair.lens_id in lens_ids or record.pair.physical_system_id in system_ids:
                raise ValueError("cross-shard lens/system duplicate")
            pair_ids.add(record.pair.pair_id)
            source_ids.add(record.pair.source_id)
            lens_ids.add(record.pair.lens_id)
            system_ids.add(record.pair.physical_system_id)
            for noise_id in record.provenance.used_noise_segment_ids:
                if noise_id in noise_ids:
                    raise ValueError("cross-shard noise-segment duplicate")
                noise_ids.add(noise_id)
            noisy = np.asarray(arrays["noisy"][row_index])
            clean = np.asarray(arrays["clean"][row_index])
            noise = np.asarray(arrays["noise"][row_index])
            validate_strain_array_semantics(
                noisy,
                clean,
                noise,
                record.gw_observation.detector_availability_mask,
            )
            selected = {
                record.pair.primary_image_id,
                record.pair.secondary_image_id,
            }
            physical = {image.image_id for image in record.lens_truth.physical_images}
            if not selected <= physical:
                raise ValueError("selected pair is absent from retained physical images")
            if not math.isfinite(record.provenance.distribution.importance_weight):
                raise ValueError("importance weight is nonfinite")
            if record.provenance.selection is None:
                raise ValueError("privileged selection provenance is missing")
            family_counts[record.pair.lens_family.value] += 1
            multiplicities[len(record.lens_truth.physical_images)] += 1
            em_cells[str(row["em_cell"])] += 1
            ordered_pair_ids.append(record.pair.pair_id)
            ordered_source_ids.append(record.pair.source_id)
            ordered_lens_ids.append(record.pair.lens_id)
            ordered_noise_ids.extend(record.provenance.used_noise_segment_ids)
        journal_path = stage / "attempts" / f"shard-{shard_index:05d}.jsonl"
        lines = journal_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines):
            attempt = AttemptRecord(**json.loads(line))
            attempt.validate()
            expected_attempt_id = shard_index + line_number * stride
            if attempt.attempt_id != expected_attempt_id or attempt.attempt_id in attempt_ids:
                raise ValueError("attempt IDs are duplicate or outside their shard sequence")
            if attempt.physical_system_id in attempt_system_ids:
                raise ValueError("attempt physical-system ID is duplicated")
            attempt_ids.add(attempt.attempt_id)
            attempt_system_ids.add(attempt.physical_system_id)
            attempted_count += 1
            if attempt.status == "accepted":
                accepted_attempt_count += 1
                if attempt.pair_id is None or attempt.pair_id in accepted_attempt_pairs:
                    raise ValueError("accepted attempt pair ID is missing or duplicated")
                accepted_attempt_pairs.add(attempt.pair_id)
            else:
                rejection_counts[str(attempt.rejection_reason)] += 1
    expected_total = int(config["accepted_pair_count"])
    if len(ordered_pair_ids) != expected_total or accepted_attempt_count != expected_total:
        raise ValueError("final accepted count is not exactly 4,096")
    if accepted_attempt_pairs != pair_ids:
        raise ValueError("accepted attempt journal and records disagree")
    if len(noise_ids) != expected_total * 6:
        raise ValueError("detector-specific noise IDs are incomplete")
    input_policy.validate_model_inputs(tuple(sorted(input_policy.allowlist)))
    validation: Dict[str, Any] = {
        "status": "passed",
        "accepted_pair_count": len(ordered_pair_ids),
        "attempt_count": attempted_count,
        "complete_shard_count": int(config["shard_count"]),
        "pairs_per_shard": expected_pairs,
        "schema_version": str(config["schema_version"]),
        "generator_git_commit": generator_git_commit,
        "authorizing_git_commit": authorization.authorizing_git_commit,
        "preregistration_version": authorization.preregistration_version,
        "preregistration_hash": authorization.preregistration_hash,
        "configuration_hash": configuration_hash,
        "family_counts": dict(family_counts),
        "image_multiplicity_counts": dict(multiplicities),
        "em_cell_counts": dict(em_cells),
        "rejection_counts": dict(rejection_counts),
        "pair_ids_sha256": hashlib.sha256(canonical_json(sorted(pair_ids)).encode()).hexdigest(),
        "source_ids_sha256": hashlib.sha256(
            canonical_json(sorted(source_ids)).encode()
        ).hexdigest(),
        "lens_ids_sha256": hashlib.sha256(canonical_json(sorted(lens_ids)).encode()).hexdigest(),
        "physical_system_ids_sha256": hashlib.sha256(
            canonical_json(sorted(system_ids)).encode()
        ).hexdigest(),
        "noise_segment_ids_sha256": hashlib.sha256(
            canonical_json(sorted(noise_ids)).encode()
        ).hexdigest(),
        "all_ids_unique": True,
        "all_groups_confined_to_generator_qualification": True,
        "all_physical_images_retained": True,
        "selected_pair_identity_valid": True,
        "array_decomposition_exact_float32": True,
        "deployable_input_policy": "passed",
        "scientific_use_authorized": False,
        "training_use_authorized": False,
        "calibration_use_authorized": False,
        "test_use_authorized": False,
    }
    identities = {
        "pair_ids": tuple(ordered_pair_ids),
        "source_ids": tuple(ordered_source_ids),
        "lens_ids": tuple(ordered_lens_ids),
        "noise_segment_ids": tuple(ordered_noise_ids),
    }
    return validation, identities, tuple(artifacts)


def publish_qualification_dataset(
    *,
    stage: Path,
    publication: Path,
    config: Mapping[str, Any],
    authorization: QualificationAuthorization,
    generator_git_commit: str,
    configuration_hash: str,
    dataset_id: str,
    validation: Mapping[str, Any],
    identities: Mapping[str, Sequence[str]],
    shard_artifacts: Sequence[ArtifactChecksum],
    environment_file: Path,
) -> Dict[str, Any]:
    if stage.parent != authorization.staging_root:
        raise ValueError("qualification staging path is outside the authorized root")
    if publication.parent != authorization.publication_root:
        raise ValueError("qualification publication path is outside the authorized root")
    validation_root = stage / "validation"
    validation_root.mkdir(parents=True, exist_ok=True)
    validation_path = validation_root / "qualification_validation.json"
    _atomic_json(validation_path, validation)
    artifacts = tuple(shard_artifacts) + (
        ArtifactChecksum(
            "validation/qualification_validation.json",
            sha256_file(validation_path),
            validation_path.stat().st_size,
            "complete",
        ),
        ArtifactChecksum(
            "environment/environment.json",
            sha256_file(environment_file),
            environment_file.stat().st_size,
            "complete",
        ),
    )
    attempted_count = int(validation["attempt_count"])
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        schema_version=str(config["schema_version"]),
        generator_git_commit=generator_git_commit,
        configuration_hash=configuration_hash,
        root_seed=int(config["root_seed"]),
        planned_pair_count=int(config["accepted_pair_count"]),
        accepted_pair_count=len(identities["pair_ids"]),
        attempted_pair_count=attempted_count,
        pair_ids=tuple(identities["pair_ids"]),
        source_ids=tuple(identities["source_ids"]),
        lens_ids=tuple(identities["lens_ids"]),
        noise_segment_ids=tuple(identities["noise_segment_ids"]),
        artifacts=artifacts,
        generation_status="complete",
        dataset_purpose="generator_qualification",
        scientific_use_authorized=False,
        authorizing_git_commit=authorization.authorizing_git_commit,
        training_use_authorized=False,
        calibration_use_authorized=False,
        test_use_authorized=False,
    )
    manifest.validate()
    (stage / "dataset_manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    enriched = {
        **manifest.to_dict(),
        "preregistration_version": authorization.preregistration_version,
        "preregistration_hash": authorization.preregistration_hash,
        "base_main_commit": authorization.base_main_commit,
        "shard_count": int(config["shard_count"]),
        "pairs_per_shard": int(config["pairs_per_shard"]),
        "physical_system_ids_sha256": validation["physical_system_ids_sha256"],
        "environment_sha256": sha256_file(environment_file),
        "psd_identities": {
            detector: f"{item['file']}:{item['sha256']}"
            for detector, item in config["gw"]["psd_curves"].items()
        },
        "full_production_authorized": False,
        "gwosc_gwtc_accessed": False,
    }
    (stage / "manifest.json").write_text(
        json.dumps(enriched, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    stage_hash, published_bytes = tree_checksum(stage)
    free_before = os.statvfs(stage).f_bavail * os.statvfs(stage).f_frsize
    if published_bytes >= authorization.maximum_output_bytes:
        raise RuntimeError("final dataset exceeds the 10 GB output gate")
    if free_before < authorization.minimum_post_run_free_bytes:
        raise RuntimeError("final dataset violates the post-run free-space gate")
    if publication.exists():
        raise FileExistsError(f"publication already exists: {publication}")
    publication.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, publication)
    return {
        "status": "published",
        "dataset_id": manifest.dataset_id,
        "published_path": str(publication),
        "published_bytes": published_bytes,
        "published_tree_sha256": stage_hash,
        "remaining_free_bytes": os.statvfs(publication).f_bavail
        * os.statvfs(publication).f_frsize,
    }
