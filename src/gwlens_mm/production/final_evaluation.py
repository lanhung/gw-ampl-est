"""Fail-closed deterministic contexts for the sealed final evaluation."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Set, Tuple

import numpy as np

from ..arrays import validate_strain_array_semantics
from ..config import load_yaml
from ..provenance import canonical_json, configuration_hash
from ..schema import SplitName, V2Record
from .diagnostic_context import BalancedTailStratum, classify_balanced_tail
from .run_control import AttemptRecord
from .storage import tree_checksum, verify_complete_shard

FINAL_EVALUATION_CONFIG = "configs/data/phase4_final_evaluation.yaml"
RC4_HASH = "5aeaac395463bd073c44ead4ff4c5c729b5a2d4b4f1840c0825a53b30ab1bc98"
RC3_HASH = "6082475631539d3069edacc52f41b37fb8fe725ccd7c6bc9980cc3008795a927"
RC5_HASH = "4dde279cf1bea78d1ddbd4fab99d88e88e334c80c180dc7850679736c5e53edb"


@dataclass(frozen=True)
class FinalEvaluationNamespace:
    namespace_id: str
    split: SplitName
    accepted_count: int
    shard_count: int
    root_seed: int
    seed_domain: str
    attempt_namespace: str
    diagnostic_context_id: str
    proposal_mode: str
    proposal_distribution_id: str
    evaluation_distribution_id: str
    id_prefix: str
    assumed_lens_family: str | None = None
    balanced_tail_stratum: str | None = None
    parameter_ood_stratum: str | None = None
    truth_waveform: str | None = None
    truth_psd_curves: Mapping[str, Mapping[str, str]] | None = None


def _namespace_seed(root_seed: int, seed_domain: str, context: str) -> int:
    payload = canonical_json(
        {"root_seed": root_seed, "seed_domain": seed_domain, "context": context}
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def load_final_evaluation_contract(
    root: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load the frozen design and reject any accidental execution opening."""

    config = load_yaml(root / FINAL_EVALUATION_CONFIG)
    authorization = load_yaml(root / config["authorization_path"])
    if authorization.get("authorization_status") != (
        "authorized_implementation_and_commitment_only"
    ):
        raise PermissionError("final-evaluation implementation authorization is absent")
    if config.get("status") != "implementation_only_execution_disabled":
        raise PermissionError("final-evaluation config is not implementation-only")
    frozen = authorization["frozen_scientific_contracts"]
    if (
        frozen["direct_target_preregistration_hash"],
        frozen["adaptive_preregistration_hash"],
        frozen["parent_preregistration_hash"],
    ) != (RC4_HASH, RC3_HASH, RC5_HASH):
        raise ValueError("final-evaluation authorization changed a frozen contract")
    for key in (
        "stage_a_staging_access_authorized",
        "waveform_pair_generation_authorized",
        "final_evaluation_materialization_authorized",
        "final_evaluation_unsealing_authorized",
        "scientific_probe_training_authorized",
        "model_selection_authorized",
        "calibration_authorized",
        "sbc_authorized",
        "gwosc_gwtc_access_authorized",
        "real_noise_authorized",
    ):
        if authorization["authorization"].get(key) is not False:
            raise PermissionError(f"implementation gate requires {key}=false")
    if any(config["execution"].get(key) is not False for key in (
        "enabled",
        "materialization_authorized",
        "unsealing_authorized",
        "model_training_authorized",
        "gwosc_gwtc_access_authorized",
    )):
        raise PermissionError("final-evaluation execution flag opened")
    for key, expected in (
        ("preregistration", RC4_HASH),
        ("parent_adaptive_preregistration", RC3_HASH),
        ("parent_scientific_preregistration", RC5_HASH),
    ):
        item = config[key]
        loaded = load_yaml(root / item["path"])
        if item["canonical_hash"] != expected or configuration_hash(loaded) != expected:
            raise ValueError(f"{key} canonical hash mismatch")
    if config["totals"] != {"accepted_count": 20480, "shard_count": 160}:
        raise ValueError("final-evaluation total count contract changed")
    return config, authorization


def final_evaluation_namespaces(config: Mapping[str, Any]) -> Tuple[FinalEvaluationNamespace, ...]:
    """Expand the six frozen splits into deterministic materialization namespaces."""

    root_seed = int(config["root_seed"])
    result = []

    def add(
        split_name: str,
        context: str,
        count: int,
        shards: int,
        mode: str,
        *,
        assumed: str | None = None,
        tail: str | None = None,
        ood: str | None = None,
        waveform: str | None = None,
        psd: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        specification = config["splits"][split_name]
        result.append(
            FinalEvaluationNamespace(
                namespace_id=f"{split_name}/{context}",
                split=SplitName(split_name),
                accepted_count=count,
                shard_count=shards,
                root_seed=_namespace_seed(
                    root_seed, str(specification["seed_domain"]), context
                ),
                seed_domain=str(specification["seed_domain"]),
                attempt_namespace=str(specification["attempt_namespace"]),
                diagnostic_context_id=context,
                proposal_mode=mode,
                proposal_distribution_id=str(specification["proposal_distribution_id"]),
                evaluation_distribution_id=str(specification["evaluation_distribution_id"]),
                id_prefix=f"{specification['id_prefix']}-{context}",
                assumed_lens_family=assumed,
                balanced_tail_stratum=tail,
                parameter_ood_stratum=ood,
                truth_waveform=waveform,
                truth_psd_curves=psd,
            )
        )

    iid = config["splits"]["iid_test"]
    add(
        "iid_test",
        "iid",
        int(iid["accepted_count"]),
        int(iid["shard_count"]),
        str(iid["proposal_mode"]),
    )
    tail = config["splits"]["balanced_tail_diagnostic"]
    for value in tail["strata"]:
        add(
            "balanced_tail_diagnostic",
            str(value),
            int(tail["cases_per_stratum"]),
            int(tail["cases_per_stratum"]) // int(config["pairs_per_shard"]),
            str(tail["proposal_mode"]),
            tail=str(value),
        )
    cross = config["splits"]["cross_family_misspecification_test"]
    for cell in cross["cells"]:
        truth = str(cell["truth"])
        mode = (
            "evaluation_target_family_sie"
            if truth == "sie_external_shear"
            else "evaluation_target_family_epl"
        )
        add(
            "cross_family_misspecification_test",
            str(cell["id"]),
            int(cross["cases_per_cell"]),
            int(cross["cases_per_cell"]) // int(config["pairs_per_shard"]),
            mode,
            assumed=str(cell["assumed"]),
        )
    ood = config["splits"]["parameter_region_ood"]
    for value in ood["strata"]:
        add(
            "parameter_region_ood",
            str(value),
            int(ood["cases_per_stratum"]),
            int(ood["cases_per_stratum"]) // int(config["pairs_per_shard"]),
            f"parameter_ood_{value}",
            ood=str(value),
        )
    waveform = config["splits"]["waveform_mismatch_test"]
    add(
        "waveform_mismatch_test",
        "seobnrv4phm_truth",
        int(waveform["accepted_count"]),
        int(waveform["shard_count"]),
        str(waveform["proposal_mode"]),
        waveform=str(waveform["truth_waveform"]),
    )
    psd = config["splits"]["psd_mismatch_test"]
    add(
        "psd_mismatch_test",
        "zero_det_high_p_truth",
        int(psd["accepted_count"]),
        int(psd["shard_count"]),
        str(psd["proposal_mode"]),
        psd=psd["truth_psd_curves"],
    )
    namespaces = tuple(result)
    if sum(item.accepted_count for item in namespaces) != int(config["totals"]["accepted_count"]):
        raise ValueError("expanded final-evaluation accepted count mismatch")
    if sum(item.shard_count for item in namespaces) != int(config["totals"]["shard_count"]):
        raise ValueError("expanded final-evaluation shard count mismatch")
    if len({item.root_seed for item in namespaces}) != len(namespaces):
        raise ValueError("final-evaluation namespace seeds collide")
    return namespaces


def build_final_evaluation_namespace_config(
    root: Path, config: Mapping[str, Any], namespace: FinalEvaluationNamespace
) -> Dict[str, Any]:
    """Build a generator-ready config without creating data or identities."""

    base = load_yaml(root / config["base_data_config"])
    base.update(
        {
            "phase": "4-final-evaluation",
            "root_seed": namespace.root_seed,
            "dataset_purpose": str(config["dataset_purpose"]),
            "accepted_pair_count": namespace.accepted_count,
            "shard_count": namespace.shard_count,
            "pairs_per_shard": int(config["pairs_per_shard"]),
            "production_context": {
                "proposal_mode": namespace.proposal_mode,
                "proposal_distribution_id": namespace.proposal_distribution_id,
                "evaluation_distribution_id": namespace.evaluation_distribution_id,
                "id_prefix": namespace.id_prefix,
                "split": namespace.split.value,
                "diagnostic_context_id": namespace.diagnostic_context_id,
                "assumed_lens_family": namespace.assumed_lens_family,
                "balanced_tail_stratum": namespace.balanced_tail_stratum,
                "parameter_ood_stratum": namespace.parameter_ood_stratum,
                "seed_domain": namespace.seed_domain,
                "attempt_namespace": namespace.attempt_namespace,
            },
        }
    )
    base["execution"] = {
        **base["execution"],
        "qualification_worker_processes": int(config["execution"]["worker_processes"]),
        "attempt_id_stride": int(config["attempt_id_stride"]),
        "maximum_attempts_per_worker": int(
            config["execution"]["maximum_attempts_per_worker"]
        ),
    }
    if namespace.truth_waveform is not None:
        base["gw"] = {**base["gw"], "waveform": namespace.truth_waveform}
    if namespace.truth_psd_curves is not None:
        base["gw"] = {**base["gw"], "psd_curves": dict(namespace.truth_psd_curves)}
    return base


def validate_final_evaluation_record(
    record: V2Record,
    namespace: FinalEvaluationNamespace,
    *,
    expected_dataset: str,
) -> None:
    """Validate one record against its sealed diagnostic context."""

    record.validate()
    if record.pair.split is not namespace.split:
        raise ValueError("final-evaluation record split mismatch")
    if record.pair.dataset_version != expected_dataset:
        raise ValueError("final-evaluation record dataset mismatch")
    if (
        record.pair.proposal_distribution_id != namespace.proposal_distribution_id
        or record.pair.evaluation_prior_id != namespace.evaluation_distribution_id
    ):
        raise ValueError("final-evaluation record distribution identity mismatch")
    distribution = record.provenance.distribution
    if (
        distribution.proposal_log_probability
        != distribution.evaluation_prior_log_probability
        or distribution.importance_weight != 1.0
        or not distribution.weight_valid
        or distribution.clipping_applied
    ):
        raise ValueError("final-evaluation diagnostics require exact unit weights")
    if namespace.truth_waveform is not None and (
        record.source_truth.waveform_model != namespace.truth_waveform
        or record.provenance.waveform_model != namespace.truth_waveform
    ):
        raise ValueError("waveform-mismatch truth identity mismatch")
    if namespace.truth_psd_curves is not None:
        expected = {
            detector: f"{item['file']}:{item['sha256']}"
            for detector, item in namespace.truth_psd_curves.items()
        }
        if record.gw_observation.detector_psd_references != expected:
            raise ValueError("PSD-mismatch truth identity mismatch")
    if namespace.proposal_mode == "evaluation_target_family_sie" and (
        record.pair.lens_family.value != "sie_external_shear"
    ):
        raise ValueError("cross-family SIE truth cell contains another family")
    if namespace.proposal_mode == "evaluation_target_family_epl" and (
        record.pair.lens_family.value != "epl_external_shear"
    ):
        raise ValueError("cross-family EPL truth cell contains another family")
    if namespace.parameter_ood_stratum is not None:
        lens = record.lens_truth.lens_parameters
        value = namespace.parameter_ood_stratum
        if value == "slope_outside_training":
            slope = float(lens["density_slope"])
            valid = record.pair.lens_family.value == "epl_external_shear" and (
                1.4 <= slope < 1.6 or 2.5 < slope <= 2.7
            )
        elif value == "extreme_flattening":
            valid = 0.25 <= float(lens["axis_ratio"]) < 0.4
        elif value == "high_external_shear":
            valid = 0.15 < float(lens["shear_amplitude"]) <= 0.25
        elif value == "extreme_external_convergence":
            valid = 0.15 < abs(record.lens_truth.external_convergence) <= 0.25
        else:  # pragma: no cover - configuration validation owns this path
            raise ValueError("unknown parameter-OOD stratum")
        if not valid:
            raise ValueError("parameter-OOD record lies outside its frozen stratum")
    if namespace.balanced_tail_stratum is not None:
        selection = record.provenance.selection
        if selection is None:
            raise ValueError("tail diagnostic lacks selection provenance")
        images = {image.image_id: image for image in record.lens_truth.physical_images}
        selected = (images[record.pair.primary_image_id], images[record.pair.secondary_image_id])
        actual = classify_balanced_tail(
            selected,
            secondary_network_snr=selection.per_image_network_optimal_snr[
                record.pair.secondary_image_id
            ],
            external_convergence=record.lens_truth.external_convergence,
            density_slope=float(record.lens_truth.lens_parameters["density_slope"]),
        )
        if actual != BalancedTailStratum(namespace.balanced_tail_stratum):
            raise ValueError("balanced-tail record entered the wrong priority stratum")


def validate_final_evaluation_namespace(
    stage: Path,
    *,
    namespace_config: Mapping[str, Any],
    namespace: FinalEvaluationNamespace,
    expected_dataset: str,
    generator_commit: str,
) -> Tuple[Dict[str, Any], Dict[str, Set[str]]]:
    """Stream and validate one complete, still-sealed evaluation namespace."""

    pandas = importlib.import_module("pandas")
    zarr = importlib.import_module("zarr")
    expected_shards = int(namespace_config["shard_count"])
    pairs_per_shard = int(namespace_config["pairs_per_shard"])
    expected_count = int(namespace_config["accepted_pair_count"])
    expected_config_hash = configuration_hash(namespace_config)
    stride = int(namespace_config["execution"]["attempt_id_stride"])
    identifiers: Dict[str, Set[str]] = {
        "pair": set(),
        "source": set(),
        "lens": set(),
        "system": set(),
        "noise": set(),
        "augmentation_parent": set(),
        "attempt_system": set(),
    }
    attempt_ids: set[int] = set()
    accepted_attempt_pairs: set[str] = set()
    accepted_attempt_count = 0
    attempt_count = 0
    artifacts: list[Dict[str, Any]] = []
    for shard_index in range(expected_shards):
        shard = stage / "shards" / f"shard-{shard_index:05d}"
        verify_complete_shard(shard, pairs_per_shard)
        digest, byte_count = tree_checksum(shard)
        artifacts.append(
            {"shard_index": shard_index, "sha256": digest, "bytes": byte_count}
        )
        frame = pandas.read_parquet(shard / "records.parquet")
        arrays = {
            role: zarr.open_array(str(shard / f"{role}.zarr"), mode="r")
            for role in ("noisy", "clean", "noise")
        }
        expected_shape = (
            pairs_per_shard,
            2,
            3,
            int(namespace_config["gw"]["sample_count"]),
        )
        if len(frame) != pairs_per_shard or any(
            array.shape != expected_shape for array in arrays.values()
        ):
            raise ValueError("final-evaluation shard shape or count mismatch")
        for row_index, row in frame.iterrows():
            record = V2Record.from_json(str(row["record_json"]))
            validate_final_evaluation_record(
                record, namespace, expected_dataset=expected_dataset
            )
            if record.provenance.generator_git_commit != generator_commit:
                raise ValueError("final-evaluation records mix generator commits")
            if record.provenance.configuration_hash != expected_config_hash:
                raise ValueError("final-evaluation records mix namespace configs")
            if str(row.get("diagnostic_context")) != namespace.diagnostic_context_id:
                raise ValueError("final-evaluation partition context mismatch")
            values = {
                "pair": record.pair.pair_id,
                "source": record.pair.source_id,
                "lens": record.pair.lens_id,
                "system": record.pair.physical_system_id,
            }
            for key, value in values.items():
                if value in identifiers[key]:
                    raise ValueError(f"final-evaluation duplicate {key} ID")
                if value.startswith(("qualification-", "phase3ca", "phase4-canary")):
                    raise ValueError("engineering ID entered final evaluation")
                identifiers[key].add(value)
            parent = record.pair.augmentation_parent_id
            if parent is not None:
                if parent in identifiers["augmentation_parent"]:
                    raise ValueError("final-evaluation duplicate augmentation parent")
                identifiers["augmentation_parent"].add(parent)
            for noise_id in record.provenance.used_noise_segment_ids:
                if noise_id in identifiers["noise"]:
                    raise ValueError("final-evaluation duplicate noise ID")
                identifiers["noise"].add(noise_id)
            validate_strain_array_semantics(
                np.asarray(arrays["noisy"][row_index]),
                np.asarray(arrays["clean"][row_index]),
                np.asarray(arrays["noise"][row_index]),
                record.gw_observation.detector_availability_mask,
            )
        journal = stage / "attempts" / f"shard-{shard_index:05d}.jsonl"
        for line_number, line in enumerate(journal.read_text().splitlines()):
            attempt = AttemptRecord(**json.loads(line))
            attempt.validate()
            expected_attempt_id = shard_index + line_number * stride
            if attempt.attempt_id != expected_attempt_id or attempt.attempt_id in attempt_ids:
                raise ValueError("final-evaluation attempt stream is inconsistent")
            if attempt.physical_system_id in identifiers["attempt_system"]:
                raise ValueError("final-evaluation attempt system ID is duplicated")
            attempt_ids.add(attempt.attempt_id)
            identifiers["attempt_system"].add(attempt.physical_system_id)
            attempt_count += 1
            if attempt.status == "accepted":
                if attempt.pair_id is None or attempt.pair_id in accepted_attempt_pairs:
                    raise ValueError("final-evaluation accepted attempt is invalid")
                accepted_attempt_pairs.add(attempt.pair_id)
                accepted_attempt_count += 1
    if len(identifiers["pair"]) != expected_count:
        raise ValueError("final-evaluation namespace accepted count mismatch")
    if accepted_attempt_count != expected_count:
        raise ValueError("final-evaluation journal accepted count mismatch")
    if accepted_attempt_pairs != identifiers["pair"]:
        raise ValueError("final-evaluation records and journal disagree")
    if len(identifiers["noise"]) != expected_count * 6:
        raise ValueError("final-evaluation detector-noise IDs are incomplete")
    published_bytes = sum(int(item["bytes"]) for item in artifacts)
    if not math.isfinite(float(published_bytes)):
        raise ValueError("final-evaluation byte count is invalid")
    return (
        {
            "status": "passed_sealed",
            "namespace_id": namespace.namespace_id,
            "split": namespace.split.value,
            "diagnostic_context_id": namespace.diagnostic_context_id,
            "dataset_id": expected_dataset,
            "accepted_pair_count": expected_count,
            "complete_shard_count": expected_shards,
            "attempt_count": attempt_count,
            "generator_commit": generator_commit,
            "configuration_hash": expected_config_hash,
            "proposal_distribution_id": namespace.proposal_distribution_id,
            "evaluation_distribution_id": namespace.evaluation_distribution_id,
            "published_bytes": published_bytes,
            "shards": artifacts,
        },
        identifiers,
    )


def collect_published_group_identifiers(
    roots: Tuple[Path, ...],
) -> Dict[str, Set[str]]:
    """Stream group IDs from already published scientific reference datasets."""

    pandas = importlib.import_module("pandas")
    identifiers: Dict[str, Set[str]] = {
        "pair": set(),
        "source": set(),
        "lens": set(),
        "system": set(),
        "noise": set(),
        "augmentation_parent": set(),
    }
    for root in roots:
        if not root.is_dir() or not (root / "dataset_manifest.json").is_file():
            raise ValueError(f"published reference manifest is absent: {root}")
        records = tuple(sorted(root.rglob("records.parquet")))
        if not records:
            raise ValueError(f"published reference has no Parquet records: {root}")
        for path in records:
            frame = pandas.read_parquet(path, columns=["record_json"])
            for value in frame["record_json"]:
                record = V2Record.from_json(str(value))
                values = {
                    "pair": record.pair.pair_id,
                    "source": record.pair.source_id,
                    "lens": record.pair.lens_id,
                    "system": record.pair.physical_system_id,
                }
                for key, identifier in values.items():
                    if identifier in identifiers[key]:
                        raise ValueError(f"published references duplicate {key} ID")
                    identifiers[key].add(identifier)
                parent = record.pair.augmentation_parent_id
                if parent is not None:
                    if parent in identifiers["augmentation_parent"]:
                        raise ValueError("published references duplicate augmentation parent")
                    identifiers["augmentation_parent"].add(parent)
                for noise_id in record.provenance.used_noise_segment_ids:
                    if noise_id in identifiers["noise"]:
                        raise ValueError("published references duplicate noise ID")
                    identifiers["noise"].add(noise_id)
    return identifiers


def dry_run_plan(root: Path) -> Mapping[str, Any]:
    config, _ = load_final_evaluation_contract(root)
    namespaces = final_evaluation_namespaces(config)
    return {
        "status": "implementation_ready_execution_blocked",
        "configuration_hash": configuration_hash(config),
        "accepted_count": sum(item.accepted_count for item in namespaces),
        "shard_count": sum(item.shard_count for item in namespaces),
        "namespace_count": len(namespaces),
        "namespaces": [
            {
                "namespace_id": item.namespace_id,
                "split": item.split.value,
                "accepted_count": item.accepted_count,
                "shard_count": item.shard_count,
                "root_seed": item.root_seed,
                "proposal_mode": item.proposal_mode,
            }
            for item in namespaces
        ],
        "waveform_pairs_generated": 0,
        "materialization_authorized": False,
        "training_authorized": False,
        "final_evaluation_unsealed": False,
    }
