"""Fail-closed direct-target calibration-fit and SBC pool contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from ..config import load_yaml
from ..provenance import canonical_json, configuration_hash, dataset_id
from ..schema import SplitName, V2Record
from .stage_a import DIRECT_TARGET_ID, validate_direct_target_record
from .waveform_correction import (
    CORRECTION_PREREGISTRATION_HASH,
    apply_frozen_source_waveform_numerical_validity,
    load_waveform_correction_contract,
)

CONFIG_PATH = "configs/data/phase6_calibration_sbc_direct_target.yaml"
CALIBRATION_SBC_HASH = "033b996930c93e7e4a9881fc3de49bb85cf4be96fcbd890bf2543b46368c9d8e"
RC4_HASH = "5aeaac395463bd073c44ead4ff4c5c729b5a2d4b4f1840c0825a53b30ab1bc98"
RC3_HASH = "6082475631539d3069edacc52f41b37fb8fe725ccd7c6bc9980cc3008795a927"
RC5_HASH = "4dde279cf1bea78d1ddbd4fab99d88e88e334c80c180dc7850679736c5e53edb"
NUMERICAL_VALIDITY_COMMITMENT_PATH = (
    "results/phase6/calibration_sbc_numerical_validity_commitment.json"
)
NUMERICAL_VALIDITY_COMMITMENT_HASH = (
    "af87affbaf56695fe0a6c7f422a70fed154dd2df2255df819348ad204dd0ccd4"
)


@dataclass(frozen=True)
class CalibrationSBCNamespace:
    namespace_id: str
    split: SplitName
    accepted_count: int
    shard_count: int
    root_seed: int
    seed_context: str
    attempt_stream_namespace: str
    id_prefix: str
    expected_em_cell_count: int


@dataclass(frozen=True)
class CalibrationSBCIdentities:
    parent_run_id: str
    calibration_dataset_id: str
    sbc_dataset_id: str

    def as_dict(self) -> Mapping[str, str]:
        return {
            "parent_run_id": self.parent_run_id,
            "calibration_dataset_id": self.calibration_dataset_id,
            "sbc_dataset_id": self.sbc_dataset_id,
        }


def _derived_seed(root_seed: int, seed_domain: str, context: str) -> int:
    payload = canonical_json(
        {"root_seed": root_seed, "seed_domain": seed_domain, "context": context}
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def load_calibration_sbc_numerical_validity_commitment(
    root: Path, config: Mapping[str, Any]
) -> Dict[str, Any]:
    """Validate the prospective RC.1 rejection overlay without opening data."""

    path = root / NUMERICAL_VALIDITY_COMMITMENT_PATH
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    recorded = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if digest != NUMERICAL_VALIDITY_COMMITMENT_HASH or recorded != digest:
        raise ValueError("calibration/SBC numerical-validity commitment hash mismatch")
    commitment = json.loads(path.read_text(encoding="utf-8"))
    correction = load_waveform_correction_contract(root)
    namespaces = commitment.get("namespace_contract", {})
    if (
        commitment.get("commitment_status") != "finalized_before_materialization"
        or commitment.get("future_materialization_generator_commit") is not None
        or commitment.get("counts_and_seed_namespaces_changed") is not False
        or commitment.get("base_generator_configuration", {}).get("canonical_hash")
        != configuration_hash(config)
        or commitment.get("waveform_numerical_validity_preregistration", {}).get(
            "canonical_hash"
        )
        != CORRECTION_PREREGISTRATION_HASH
        or correction["preregistration"]["canonical_hash"]
        != CORRECTION_PREREGISTRATION_HASH
        or {
            key: int(value.get("accepted_count", -1))
            for key, value in namespaces.items()
        }
        != {"calibration_fit": 4096, "sbc_diagnostic": 2048}
        or any(value is not False for value in commitment.get("use_policy", {}).values())
    ):
        raise ValueError("calibration/SBC numerical-validity commitment changed")
    return commitment


def load_calibration_sbc_contract(
    root: Path, config_path: str = CONFIG_PATH
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load the frozen design and prove that every execution boundary is closed."""

    config = load_yaml(root / config_path)
    authorization = load_yaml(root / config["implementation_authorization_path"])
    if config.get("stage") != "calibration_sbc_direct_target_pools":
        raise ValueError("calibration/SBC data-pool identity is absent")
    if config.get("status") != "implementation_only_execution_disabled":
        raise PermissionError("calibration/SBC data-pool implementation gate changed")
    if authorization.get("authorization_status") != "authorized_implementation_only":
        raise PermissionError("calibration/SBC materialization-stack gate is absent")
    flags = authorization.get("authorization", {})
    for key in (
        "waveform_pair_generation_authorized",
        "calibration_sbc_materialization_authorized",
        "scientific_data_access_authorized",
        "model_checkpoint_access_authorized",
        "calibration_fit_authorized",
        "sbc_execution_authorized",
        "final_evaluation_authorized",
        "model_retraining_or_tuning_authorized",
        "gwosc_gwtc_access_authorized",
    ):
        if flags.get(key) is not False:
            raise PermissionError(f"implementation gate requires {key}=false")
    expected_hashes = (
        ("calibration_sbc_preregistration", CALIBRATION_SBC_HASH),
        ("direct_target_preregistration", RC4_HASH),
        ("adaptive_preregistration", RC3_HASH),
        ("parent_scientific_preregistration", RC5_HASH),
    )
    for key, expected_hash in expected_hashes:
        specification = config[key]
        loaded = load_yaml(root / specification["path"])
        if (
            specification["canonical_hash"] != expected_hash
            or configuration_hash(loaded) != expected_hash
        ):
            raise ValueError(f"calibration/SBC {key} canonical hash mismatch")
    direct = config["direct_target_density"]
    if not (
        direct["mode"] == "evaluation_target_direct"
        and direct["proposal_distribution_id"] == DIRECT_TARGET_ID
        and direct["evaluation_distribution_id"] == DIRECT_TARGET_ID
        and direct["proposal_equals_evaluation"] is True
        and float(direct["log_importance_weight"]) == 0.0
        and float(direct["importance_weight"]) == 1.0
    ):
        raise ValueError("calibration/SBC direct-target density contract changed")
    if config["totals"] != {
        "accepted_pair_count": 6144,
        "shard_count": 48,
        "namespace_count": 2,
    }:
        raise ValueError("calibration/SBC total count contract changed")
    resources = config["resource_gates"]
    if not (
        int(resources["projected_publication_bytes"])
        <= int(resources["maximum_output_bytes"])
        and int(resources["minimum_prelaunch_free_bytes"])
        >= int(resources["minimum_post_peak_free_bytes"])
        + int(resources["maximum_output_bytes"])
    ):
        raise ValueError("calibration/SBC resource arithmetic is not fail-closed")
    execution = config["execution"]
    for key in (
        "enabled",
        "materialization_authorized",
        "calibration_fit_authorized",
        "sbc_authorized",
        "checkpoint_access_authorized",
        "final_evaluation_authorized",
        "gwosc_gwtc_access_authorized",
    ):
        if execution.get(key) is not False:
            raise PermissionError(f"design config requires execution.{key}=false")
    calibration, sbc = calibration_sbc_namespaces(config)
    if (
        calibration.accepted_count,
        calibration.shard_count,
        calibration.expected_em_cell_count,
        sbc.accepted_count,
        sbc.shard_count,
        sbc.expected_em_cell_count,
    ) != (4096, 32, 512, 2048, 16, 256):
        raise ValueError("calibration/SBC namespace count contract changed")
    load_calibration_sbc_numerical_validity_commitment(root, config)
    return config, authorization


def calibration_sbc_namespaces(
    config: Mapping[str, Any]
) -> Tuple[CalibrationSBCNamespace, ...]:
    result = []
    for name in ("calibration_fit", "sbc_diagnostic"):
        item = config["splits"][name]
        namespace = CalibrationSBCNamespace(
            namespace_id=name,
            split=SplitName(str(item["split"])),
            accepted_count=int(item["accepted_pair_count"]),
            shard_count=int(item["shard_count"]),
            root_seed=int(item["root_seed"]),
            seed_context=str(item["seed_context"]),
            attempt_stream_namespace=str(item["attempt_stream_namespace"]),
            id_prefix=str(item["id_prefix"]),
            expected_em_cell_count=int(item["expected_em_cell_count"]),
        )
        expected_seed = _derived_seed(
            int(config["root_seed"]),
            str(config["split_assignment_seed_domain"]),
            namespace.seed_context,
        )
        if namespace.root_seed != expected_seed:
            raise ValueError(f"{name} root seed differs from its frozen derivation")
        if namespace.shard_count * int(config["pairs_per_shard"]) != (
            namespace.accepted_count
        ):
            raise ValueError(f"{name} shard arithmetic failed")
        result.append(namespace)
    namespaces = tuple(result)
    if len({item.root_seed for item in namespaces}) != len(namespaces):
        raise ValueError("calibration/SBC root seeds collide")
    if len({item.id_prefix for item in namespaces}) != len(namespaces):
        raise ValueError("calibration/SBC ID prefixes collide")
    return namespaces


def build_calibration_sbc_namespace_config(
    root: Path,
    config: Mapping[str, Any],
    namespace: CalibrationSBCNamespace,
) -> Dict[str, Any]:
    """Build one generator config without creating an identity or a pair."""

    base = load_yaml(root / str(config["base_data_config"]))
    direct = config["direct_target_density"]
    base.update(
        {
            "phase": "6-stage-c-development",
            "root_seed": namespace.root_seed,
            "dataset_purpose": str(config["dataset_purpose"]),
            "accepted_pair_count": namespace.accepted_count,
            "shard_count": namespace.shard_count,
            "pairs_per_shard": int(config["pairs_per_shard"]),
            "production_context": {
                "proposal_mode": str(direct["mode"]),
                "proposal_distribution_id": str(
                    direct["proposal_distribution_id"]
                ),
                "evaluation_distribution_id": str(
                    direct["evaluation_distribution_id"]
                ),
                "attempt_stream_namespace": namespace.attempt_stream_namespace,
                "id_prefix": namespace.id_prefix,
                "split": namespace.split.value,
                "canary": False,
            },
        }
    )
    base["execution"] = {
        **base["execution"],
        "qualification_worker_processes": int(config["execution"]["worker_processes"]),
        "attempt_id_stride": int(config["execution"]["attempt_id_stride"]),
        "maximum_attempts_per_worker": int(
            config["execution"]["maximum_attempts_per_worker"]
        ),
        "maximum_active_seconds_per_worker": int(
            config["execution"]["maximum_active_seconds_per_worker"]
        ),
    }
    return apply_frozen_source_waveform_numerical_validity(root, base)


def derive_calibration_sbc_identities(
    config: Mapping[str, Any], generator_commit: str
) -> CalibrationSBCIdentities:
    if len(generator_commit) != 40 or any(
        value not in "0123456789abcdef" for value in generator_commit.lower()
    ):
        raise ValueError("calibration/SBC generator commit must be a full Git SHA")
    namespaces = calibration_sbc_namespaces(config)
    config_hash = configuration_hash(config)
    parent = f"phase6-stage-c-{generator_commit[:12]}-{config_hash[:12]}"
    values = {
        namespace.namespace_id: dataset_id(
            "2.0.0-alpha.3",
            generator_commit,
            config_hash,
            namespace.root_seed,
        )
        + f"-{namespace.namespace_id.replace('_', '-')}"
        for namespace in namespaces
    }
    if len(set(values.values())) != 2:
        raise ValueError("calibration/SBC dataset identities collide")
    return CalibrationSBCIdentities(
        parent,
        values["calibration_fit"],
        values["sbc_diagnostic"],
    )


def validate_future_materialization_authorization(
    authorization: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    generator_commit: str,
) -> None:
    """Validate a later exact gate without making that gate exist now."""

    if authorization.get("authorization_status") != (
        "authorized_exact_calibration_sbc_materialization_only"
    ):
        raise PermissionError("exact calibration/SBC materialization is not authorized")
    if authorization.get("immutable_generator", {}).get("git_commit") != generator_commit:
        raise ValueError("calibration/SBC generator identity mismatch")
    frozen = authorization.get("frozen_contract", {})
    if frozen.get("configuration_hash") != configuration_hash(config):
        raise ValueError("calibration/SBC configuration hash mismatch")
    if frozen.get("calibration_sbc_preregistration_hash") != CALIBRATION_SBC_HASH:
        raise ValueError("calibration/SBC preregistration identity mismatch")
    if (
        frozen.get("waveform_numerical_validity_preregistration_hash")
        != CORRECTION_PREREGISTRATION_HASH
        or frozen.get("waveform_numerical_validity_commitment_sha256")
        != NUMERICAL_VALIDITY_COMMITMENT_HASH
    ):
        raise ValueError("calibration/SBC numerical-validity identity mismatch")
    entry = authorization.get("entry_gate", {})
    if not (
        entry.get("training_size_locked") is True
        and entry.get("architecture_locked") is True
        and entry.get("three_model_seeds_retained") is True
    ):
        raise PermissionError("calibration/SBC entry gate is not locked")
    counts = authorization.get("materialization_contract", {})
    if (
        counts.get("calibration_fit_accepted_count"),
        counts.get("sbc_diagnostic_accepted_count"),
        counts.get("total_accepted_count"),
        counts.get("total_shard_count"),
    ) != (4096, 2048, 6144, 48):
        raise ValueError("calibration/SBC authorization count mismatch")
    flags = authorization.get("authorization", {})
    for key in (
        "calibration_sbc_materialization_authorized",
        "accepted_pair_generator_authorized_within_stage_c_only",
    ):
        if flags.get(key) is not True:
            raise PermissionError(f"calibration/SBC materialization requires {key}=true")
    for key in (
        "calibration_fit_statistics_authorized",
        "sbc_statistics_authorized",
        "checkpoint_access_authorized",
        "final_evaluation_authorized",
        "model_retraining_or_tuning_authorized",
        "gwosc_gwtc_access_authorized",
    ):
        if flags.get(key) is not False:
            raise PermissionError(f"materialization-only gate requires {key}=false")


def validate_calibration_sbc_record(
    record: V2Record,
    namespace: CalibrationSBCNamespace,
    *,
    expected_dataset: str,
) -> None:
    validate_direct_target_record(
        record,
        expected_split=namespace.split,
        expected_dataset=expected_dataset,
    )
    if record.pair.proposal_distribution_id != DIRECT_TARGET_ID:
        raise ValueError("calibration/SBC proposal identity mismatch")
    if record.pair.evaluation_prior_id != DIRECT_TARGET_ID:
        raise ValueError("calibration/SBC evaluation identity mismatch")


def dry_run_plan(root: Path) -> Mapping[str, Any]:
    config, _ = load_calibration_sbc_contract(root)
    namespaces = calibration_sbc_namespaces(config)
    built = [
        build_calibration_sbc_namespace_config(root, config, namespace)
        for namespace in namespaces
    ]
    return {
        "status": "implementation_ready_execution_closed",
        "configuration_hash": configuration_hash(config),
        "waveform_numerical_validity_preregistration_hash": (
            CORRECTION_PREREGISTRATION_HASH
        ),
        "waveform_numerical_validity_commitment_sha256": (
            NUMERICAL_VALIDITY_COMMITMENT_HASH
        ),
        "accepted_pair_count": sum(item.accepted_count for item in namespaces),
        "shard_count": sum(item.shard_count for item in namespaces),
        "namespace_count": len(namespaces),
        "namespaces": [
            {
                "namespace_id": item.namespace_id,
                "split": item.split.value,
                "accepted_pair_count": item.accepted_count,
                "shard_count": item.shard_count,
                "root_seed": item.root_seed,
                "namespace_config_hash": configuration_hash(namespace_config),
            }
            for item, namespace_config in zip(namespaces, built)
        ],
        "official_identities": None,
        "pair_generated": False,
        "calibration_fitted": False,
        "sbc_executed": False,
        "final_evaluation_accessed": False,
    }
