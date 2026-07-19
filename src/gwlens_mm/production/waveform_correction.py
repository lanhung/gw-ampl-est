"""Versioned numerical-waveform correction and replacement-view contracts."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Set, Tuple

from ..config import load_yaml
from ..provenance import configuration_hash, dataset_id
from ..schema import V2Record
from .stage_a import DIRECT_TARGET_ID

CORRECTION_CONFIG = "configs/data/phase4_waveform_numerical_correction.yaml"
CORRECTION_PREREGISTRATION_HASH = (
    "7fca209de9f06e98da1c5a96ae0f4fc6daec5d2f0c2339a718e1f899bb915b69"
)
CORRECTION_COMPONENTS = ("stage_a_train", "stage_b_train")
GROUP_KEYS = ("pair", "source", "lens", "system", "noise")


@dataclass(frozen=True)
class WaveformCorrectionIdentity:
    parent_run_id: str
    stage_a_replacement_dataset_id: str
    stage_b_replacement_dataset_id: str
    configuration_hash: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_waveform_correction_contract(
    root: Path, config_path: str = CORRECTION_CONFIG
) -> Dict[str, Any]:
    """Load and validate the non-executing correction contract."""

    config = load_yaml(root / config_path)
    if (config.get("phase"), config.get("stage")) != (
        "4",
        "waveform_numerical_correction",
    ):
        raise ValueError("waveform-correction identity is absent")
    preregistration = load_yaml(root / config["preregistration"]["path"])
    if (
        preregistration.get("preregistration_version") != "1.1.1-rc.1"
        or configuration_hash(preregistration) != CORRECTION_PREREGISTRATION_HASH
        or config["preregistration"]["canonical_hash"]
        != CORRECTION_PREREGISTRATION_HASH
    ):
        raise ValueError("waveform-correction preregistration hash mismatch")
    parent = load_yaml(root / config["parent_scientific_preregistration"]["path"])
    if configuration_hash(parent) != config["parent_scientific_preregistration"][
        "canonical_hash"
    ]:
        raise ValueError("waveform-correction RC.5 parent hash mismatch")
    audit_path = root / config["audit"]["path"]
    if _sha256(audit_path) != config["audit"]["sha256"]:
        raise ValueError("waveform-correction audit hash mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("status")
        != "completed_exhaustive_source_polarization_spectral_audit"
        or int(audit.get("record_count", -1)) != 71680
        or len(audit.get("pathologies", ())) != 5
        or audit.get("published_data_modified") is not False
        or audit.get("waveform_pairs_generated") != 0
    ):
        raise ValueError("waveform-correction audit contract is incomplete")
    numerical = config["numerical_validity"]
    if (
        numerical.get("enabled") is not True
        or float(numerical["minimum_frequency_hz"]) != 20.0
        or float(numerical["positive_amplitude_quantile"]) != 0.999
        or float(numerical["maximum_peak_to_quantile_ratio"]) != 10.0
        or numerical.get("quantile_method") != "numpy_linear"
    ):
        raise ValueError("waveform-correction numerical threshold changed")
    if config["execution"]["replacement_materialization_enabled"] is not False:
        raise ValueError("design config enabled replacement materialization")
    if config["execution"]["corrected_view_publication_enabled"] is not False:
        raise ValueError("design config enabled corrected-view publication")
    if any(value is not False for value in config["authorization_boundaries"].values()):
        raise ValueError("waveform-correction design opened a downstream boundary")
    return config


def build_replacement_namespace_config(
    root: Path, config: Mapping[str, Any], component: str
) -> Dict[str, Any]:
    """Build one exact direct-target replacement namespace."""

    if component not in CORRECTION_COMPONENTS:
        raise ValueError(f"unknown waveform-correction component: {component}")
    specification = config["replacement_namespaces"][component]
    base = deepcopy(load_yaml(root / str(config["base_data_config"])))
    base.update(
        {
            "phase": "4-waveform-correction",
            "root_seed": int(specification["root_seed"]),
            "dataset_purpose": "scientific_waveform_numerical_replacement",
            "accepted_pair_count": int(specification["accepted_pair_count"]),
            "shard_count": int(specification["shard_count"]),
            "pairs_per_shard": int(specification["pairs_per_shard"]),
            "production_context": {
                "proposal_mode": "evaluation_target_direct",
                "proposal_distribution_id": DIRECT_TARGET_ID,
                "evaluation_distribution_id": DIRECT_TARGET_ID,
                "id_prefix": str(specification["id_prefix"]),
                "split": str(specification["split"]),
                "waveform_numerical_correction": True,
            },
        }
    )
    base["gw"] = deepcopy(base["gw"])
    base["gw"]["source_polarization_numerical_validity"] = {
        "enabled": True,
        "minimum_frequency_hz": float(
            config["numerical_validity"]["minimum_frequency_hz"]
        ),
        "positive_amplitude_quantile": float(
            config["numerical_validity"]["positive_amplitude_quantile"]
        ),
        "maximum_peak_to_quantile_ratio": float(
            config["numerical_validity"]["maximum_peak_to_quantile_ratio"]
        ),
    }
    base["execution"] = {
        **base["execution"],
        "qualification_worker_processes": 1,
        "attempt_id_stride": int(specification["attempt_id_stride"]),
        "maximum_attempts_per_worker": int(
            config["execution"]["maximum_attempts_per_worker"]
        ),
        "maximum_active_seconds_per_worker": int(
            config["execution"]["maximum_active_seconds_per_worker"]
        ),
    }
    return base


def derive_waveform_correction_identity(
    config: Mapping[str, Any], implementation_commit: str
) -> WaveformCorrectionIdentity:
    """Derive fresh parent and child identities from the frozen implementation."""

    if len(implementation_commit) != 40:
        raise ValueError("waveform-correction commit must be a full Git SHA")
    config_hash = configuration_hash(config)
    parent = (
        f"phase4-waveform-correction-{implementation_commit[:12]}-{config_hash[:12]}"
    )
    identities: Dict[str, str] = {}
    for component in CORRECTION_COMPONENTS:
        specification = config["replacement_namespaces"][component]
        identities[component] = dataset_id(
            "2.0.0-alpha.3",
            implementation_commit,
            config_hash,
            int(specification["root_seed"]),
        ) + f"-{component.replace('_', '-')}"
    if len(set(identities.values())) != 2:
        raise ValueError("waveform-correction child identities collide")
    return WaveformCorrectionIdentity(
        parent,
        identities["stage_a_train"],
        identities["stage_b_train"],
        config_hash,
    )


def published_group_ids(dataset_root: Path) -> Dict[str, Set[str]]:
    """Read group identities without loading strain arrays."""

    pandas = __import__("pandas")
    identifiers: Dict[str, Set[str]] = {key: set() for key in GROUP_KEYS}
    for parquet in sorted(dataset_root.glob("shards/shard-*/records.parquet")):
        frame = pandas.read_parquet(parquet, columns=["record_json"])
        for raw in frame["record_json"]:
            record = V2Record.from_json(str(raw))
            values: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
                ("pair", (record.pair.pair_id,)),
                ("source", (record.pair.source_id,)),
                ("lens", (record.pair.lens_id,)),
                ("system", (record.pair.physical_system_id,)),
                ("noise", record.provenance.used_noise_segment_ids),
            )
            for key, group in values:
                overlap = identifiers[key] & set(group)
                if overlap:
                    raise ValueError(f"duplicate {key} identity in {dataset_root}")
                identifiers[key].update(group)
    return identifiers
