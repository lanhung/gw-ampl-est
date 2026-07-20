"""Fail-closed contracts for the terminal 131k direct-target extension."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from ..config import load_yaml
from ..provenance import canonical_json, configuration_hash, dataset_id
from ..schema import SplitName, V2Record
from .diagnostic_context import BalancedTailStratum, classify_balanced_tail
from .stage_a import DIRECT_TARGET_ID, validate_direct_target_record
from .waveform_correction import apply_frozen_source_waveform_numerical_validity

TERMINAL_131K_CONFIG = "configs/data/phase4_terminal_131k.yaml"
TERMINAL_131K_PREREGISTRATION_HASH = (
    "77ff5b6b45b886657e90023c50ae002afffb077db594c80665166d537fd2346a"
)
TAIL_STRATA = tuple(item.value for item in BalancedTailStratum)


@dataclass(frozen=True)
class TerminalNamespace:
    namespace_id: str
    split: SplitName
    root_seed: int
    accepted_count: int
    shard_count: int
    pairs_per_shard: int
    id_prefix: str
    attempt_namespace: str
    proposal_distribution_id: str
    evaluation_distribution_id: str
    balanced_tail_stratum: str | None = None


@dataclass(frozen=True)
class TerminalIdentities:
    parent_run_id: str
    train_dataset_id: str
    development_tail_parent_id: str
    development_tail_dataset_ids: Mapping[str, str]
    combined_train_id: str
    configuration_hash: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _derived_seed(master: int, domain: str, context: str) -> int:
    payload = canonical_json(
        {"master_root_seed": master, "seed_domain": domain, "context": context}
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def load_terminal_131k_contract(
    root: Path, config_path: str = TERMINAL_131K_CONFIG
) -> Dict[str, Any]:
    """Load the frozen design and reject every accidental execution opening."""

    config = load_yaml(root / config_path)
    if (config.get("phase"), config.get("stage")) != (
        "4",
        "terminal_131k_direct_target_extension",
    ):
        raise ValueError("terminal 131k configuration identity is absent")
    preregistration = load_yaml(root / str(config["preregistration"]["path"]))
    if (
        preregistration.get("preregistration_version") != "1.2.0-rc.1"
        or configuration_hash(preregistration) != TERMINAL_131K_PREREGISTRATION_HASH
        or config["preregistration"]["canonical_hash"]
        != TERMINAL_131K_PREREGISTRATION_HASH
    ):
        raise ValueError("terminal 131k preregistration hash mismatch")
    parent = load_yaml(root / str(config["parent_scientific_preregistration"]["path"]))
    if configuration_hash(parent) != config["parent_scientific_preregistration"][
        "canonical_hash"
    ]:
        raise ValueError("terminal 131k RC.5 parent hash mismatch")
    decision_path = root / str(config["triggering_evidence"]["path"])
    if _sha256(decision_path) != config["triggering_evidence"]["sha256"]:
        raise ValueError("terminal 65k decision hash mismatch")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("decision") != "stop_data_limited_and_new_preregistration":
        raise PermissionError("terminal 131k extension lacks its frozen trigger")
    direct = config["direct_target_density_implementation"]
    if not (
        direct["evaluation_target_id"] == DIRECT_TARGET_ID
        and direct["mode"] == "evaluation_target_direct"
        and direct["proposal_equals_evaluation"] is True
        and float(direct["log_importance_weight"]) == 0.0
        and float(direct["importance_weight"]) == 1.0
    ):
        raise ValueError("terminal 131k direct-target contract changed")
    train = config["train_increment"]
    tail = config["development_tail"]
    terminal = config["terminal_reference"]
    if (
        int(train["accepted_pair_count"]),
        int(train["shard_count"]),
        int(train["pairs_per_shard"]),
        int(tail["accepted_pair_count"]),
        int(tail["namespace_count"]),
        int(tail["accepted_pairs_per_namespace"]),
        int(terminal["accepted_train_count"]),
    ) != (65536, 512, 128, 512, 4, 128, 131072):
        raise ValueError("terminal 131k count contract changed")
    if (
        int(train["shard_count"]) * int(train["pairs_per_shard"])
        != int(train["accepted_pair_count"])
        or int(tail["namespace_count"])
        * int(tail["accepted_pairs_per_namespace"])
        != int(tail["accepted_pair_count"])
        or tuple(tail["strata"]) != TAIL_STRATA
    ):
        raise ValueError("terminal 131k namespace arithmetic changed")
    if int(train["root_seed"]) == int(tail["master_root_seed"]):
        raise ValueError("train and development-tail seed domains collide")
    reference = config["corrected_65k_reference"]
    if (
        int(reference["accepted_train_count"]),
        int(reference["validation_count"]),
        reference["corrected_combined_train_manifest_sha256"],
    ) != (
        65536,
        6144,
        "da8aaa8d86afb4d93156191976b420bfc7bbc7dfe0fdc6c6f627515d804a7379",
    ):
        raise ValueError("corrected 65k reference changed")
    if config["execution"]["enabled"] is not False or any(
        value is not False for value in config["authorization_boundaries"].values()
    ):
        raise PermissionError("terminal 131k design config opened execution")
    return config


def terminal_namespaces(config: Mapping[str, Any]) -> Tuple[TerminalNamespace, ...]:
    """Expand the one train and four development-tail namespaces deterministically."""

    train = config["train_increment"]
    result = [
        TerminalNamespace(
            "train_increment",
            SplitName.TRAIN,
            int(train["root_seed"]),
            int(train["accepted_pair_count"]),
            int(train["shard_count"]),
            int(train["pairs_per_shard"]),
            str(train["id_prefix"]),
            str(train["attempt_stream_namespace"]),
            DIRECT_TARGET_ID,
            DIRECT_TARGET_ID,
        )
    ]
    tail = config["development_tail"]
    for stratum in tail["strata"]:
        value = str(stratum)
        result.append(
            TerminalNamespace(
                f"development_tail/{value}",
                SplitName.BALANCED_TAIL_DIAGNOSTIC,
                _derived_seed(
                    int(tail["master_root_seed"]), str(tail["seed_domain"]), value
                ),
                int(tail["accepted_pairs_per_namespace"]),
                int(tail["shards_per_namespace"]),
                int(tail["pairs_per_shard"]),
                f"{tail['id_prefix']}-{value}",
                f"{tail['attempt_stream_namespace']}-{value}",
                str(tail["proposal_distribution_id"]),
                str(tail["evaluation_distribution_id"]),
                value,
            )
        )
    namespaces = tuple(result)
    if len({item.root_seed for item in namespaces}) != len(namespaces):
        raise ValueError("terminal 131k namespace root seeds collide")
    return namespaces


def build_terminal_namespace_config(
    root: Path, config: Mapping[str, Any], namespace: TerminalNamespace
) -> Dict[str, Any]:
    """Build a generator-ready namespace without creating an official identity."""

    base = deepcopy(load_yaml(root / str(config["base_data_config"])))
    purpose = (
        config["train_increment"]["dataset_purpose"]
        if namespace.split is SplitName.TRAIN
        else config["development_tail"]["dataset_purpose"]
    )
    base.update(
        {
            "phase": "4-terminal-131k",
            "root_seed": namespace.root_seed,
            "dataset_purpose": str(purpose),
            "accepted_pair_count": namespace.accepted_count,
            "shard_count": namespace.shard_count,
            "pairs_per_shard": namespace.pairs_per_shard,
            "production_context": {
                "proposal_mode": "evaluation_target_direct",
                "proposal_distribution_id": namespace.proposal_distribution_id,
                "evaluation_distribution_id": namespace.evaluation_distribution_id,
                "id_prefix": namespace.id_prefix,
                "split": namespace.split.value,
                "attempt_stream_namespace": namespace.attempt_namespace,
                "diagnostic_context_id": namespace.namespace_id,
                "balanced_tail_stratum": namespace.balanced_tail_stratum,
                "terminal_131k_preregistration_hash": (
                    TERMINAL_131K_PREREGISTRATION_HASH
                ),
            },
        }
    )
    stride = (
        config["train_increment"]["attempt_id_stride"]
        if namespace.split is SplitName.TRAIN
        else config["development_tail"]["attempt_id_stride"]
    )
    base["execution"] = {
        **base["execution"],
        "qualification_worker_processes": int(config["execution"]["worker_processes"]),
        "attempt_id_stride": int(stride),
        "maximum_attempts_per_worker": int(
            config["execution"]["maximum_attempts_per_worker"]
        ),
        "maximum_active_seconds_per_worker": int(
            config["execution"]["maximum_active_seconds_per_worker"]
        ),
    }
    return apply_frozen_source_waveform_numerical_validity(root, base)


def derive_terminal_identities(
    root: Path, config: Mapping[str, Any], generator_commit: str
) -> TerminalIdentities:
    """Derive identities only for a later ready release certificate."""

    if len(generator_commit) != 40 or any(
        value not in "0123456789abcdef" for value in generator_commit.lower()
    ):
        raise ValueError("terminal 131k generator commit must be a full Git SHA")
    config_hash = configuration_hash(config)
    train = terminal_namespaces(config)[0]
    parent = f"phase4-terminal-131k-{generator_commit[:12]}-{config_hash[:12]}"
    train_id = dataset_id(
        "2.0.0-alpha.3", generator_commit, config_hash, train.root_seed
    ) + "-train-increment"
    tail_ids = {
        namespace.balanced_tail_stratum: dataset_id(
            "2.0.0-alpha.3",
            generator_commit,
            configuration_hash(build_terminal_namespace_config(root, config, namespace)),
            namespace.root_seed,
        )
        + f"-development-tail-{namespace.balanced_tail_stratum}"
        for namespace in terminal_namespaces(config)[1:]
        if namespace.balanced_tail_stratum is not None
    }
    if len(tail_ids) != 4 or len(set(tail_ids.values())) != 4:
        raise ValueError("development-tail dataset identities collide")
    return TerminalIdentities(
        parent,
        train_id,
        f"{parent}-development-tail",
        tail_ids,
        f"phase4-train-131k-{generator_commit[:12]}-{config_hash[:12]}",
        config_hash,
    )


def validate_terminal_record(
    record: V2Record, namespace: TerminalNamespace, *, expected_dataset: str
) -> None:
    """Validate direct-target identity and the optional priority-tail condition."""

    validate_direct_target_record(
        record, expected_split=namespace.split, expected_dataset=expected_dataset
    )
    if (
        record.pair.proposal_distribution_id != namespace.proposal_distribution_id
        or record.pair.evaluation_prior_id != namespace.evaluation_distribution_id
    ):
        raise ValueError("terminal namespace distribution identity mismatch")
    if namespace.balanced_tail_stratum is None:
        return
    selection = record.provenance.selection
    if selection is None:
        raise ValueError("development-tail record lacks selection provenance")
    images = {image.image_id: image for image in record.lens_truth.physical_images}
    selected = (
        images[record.pair.primary_image_id],
        images[record.pair.secondary_image_id],
    )
    actual = classify_balanced_tail(
        selected,
        secondary_network_snr=selection.per_image_network_optimal_snr[
            record.pair.secondary_image_id
        ],
        external_convergence=record.lens_truth.external_convergence,
        density_slope=float(record.lens_truth.lens_parameters["density_slope"]),
    )
    if actual is not BalancedTailStratum(namespace.balanced_tail_stratum):
        raise ValueError("development-tail record entered the wrong priority stratum")
