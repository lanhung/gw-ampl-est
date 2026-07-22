"""Fail-closed training path for the terminal direct-target 131k rung."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from ..config import load_yaml
from ..provenance import configuration_hash
from ..schema import SplitName
from .contracts import TrainingGateError, model_configuration_hash
from .data import (
    ConcatenatedPublishedStageADataset,
    CorrectedTrainingPublication,
    DevelopmentCase,
    DevelopmentStageADataset,
    PublishedStageADataset,
    StandardizedStageADataset,
    corrected_65k_training_dataset,
    resolve_corrected_training_publication,
)
from .engine import (
    TrainingRunIdentity,
    authorized_probe_execution_evidence,
    evaluate_development_validation,
    membership_hash,
    optimization_batch_geometry,
    standardizer_hash,
    train_probe,
)
from .features import InputStandardizer, load_input_policy
from .model import build_probe_model
from .rung65 import _load_standardizers, validate_immutable_training_artifacts
from .runner import (
    _atomic_json,
    _data_loader,
    _development_loader,
    _sha256_file,
    _validate_runtime_versions,
    _verified_curves,
    _verify_training_checkout,
    fit_rung_standardizers,
)

TRAIN_131K_COUNT = 131072
TRAIN_INCREMENT_COUNT = 65536
VALIDATION_COUNT = 6144
TAIL_COUNT = 512
TAIL_STRATA = (
    "high_absolute_magnification",
    "extreme_relative_magnification",
    "second_image_near_threshold",
    "extreme_profile_or_environment",
)
SEEDS = (0, 1, 2)
TERMINAL_PREREGISTRATION_PATH = (
    "configs/statistics/terminal_131k_preregistration.yaml"
)
TERMINAL_RELEASE_REVIEW_STATUS = (
    "ready_for_delegated_terminal_probe_authorization_review"
)
TERMINAL_RELEASE_REVIEW_ACCEPTANCE = (
    "accepted_for_exact_terminal_probe_authorization"
)
EXPECTED_TERMINAL_GPU_MODEL = "NVIDIA RTX 5000 Ada Generation"


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TrainingGateError(f"expected a JSON mapping: {path}")
    return value


def validate_terminal_probe_release_binding(
    root: Path,
    authorization: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Bind one separately reviewed packet to the future execution gate."""

    review = authorization.get("terminal_probe_release_review", {})
    if not isinstance(review, Mapping):
        raise TrainingGateError("terminal probe gate lacks release review metadata")
    path_value = str(review.get("path", ""))
    packet_relative = Path(path_value)
    if (
        packet_relative.is_absolute()
        or ".." in packet_relative.parts
        or packet_relative.parts[:2] != ("results", "phase4")
    ):
        raise TrainingGateError(
            "terminal probe release-review packet path must be repository-relative"
        )
    root_resolved = root.resolve()
    packet_path = (root_resolved / packet_relative).resolve()
    if not packet_path.is_relative_to(root_resolved):
        raise TrainingGateError(
            "terminal probe release-review packet escaped repository root"
        )
    expected_hash = str(review.get("sha256", ""))
    if (
        not packet_path.is_absolute()
        or not packet_path.is_file()
        or len(expected_hash) != 64
        or _sha256_file(packet_path) != expected_hash
        or review.get("delegated_review_status")
        != TERMINAL_RELEASE_REVIEW_ACCEPTANCE
    ):
        raise TrainingGateError("terminal probe release-review packet is not accepted")
    packet = _load_json(packet_path)
    review_checkout_commit = str(packet.get("release_review_checkout_commit", ""))
    if (
        packet.get("status") != TERMINAL_RELEASE_REVIEW_STATUS
        or packet.get("authorization_created") is not False
        or packet.get("optimizer_execution_authorized") is not False
        or packet.get("authorized_training_rungs_preview") != [TRAIN_131K_COUNT]
        or packet.get("authorized_training_seeds_preview") != list(SEEDS)
        or packet.get("architecture_selection_authorized") is not False
        or packet.get("calibration_authorized") is not False
        or packet.get("sbc_authorized") is not False
        or packet.get("final_evaluation_authorized") is not False
        or packet.get("extension_above_131072_authorized") is not False
        or packet.get("gwosc_gwtc_access_authorized") is not False
        or len(review_checkout_commit) != 40
        or any(char not in "0123456789abcdef" for char in review_checkout_commit)
    ):
        raise TrainingGateError("terminal probe release-review packet contract failed")

    packet_publication = packet.get("publication", {})
    authorized_publication = authorization.get("terminal_publication", {})
    if not isinstance(packet_publication, Mapping) or not isinstance(
        authorized_publication, Mapping
    ):
        raise TrainingGateError("terminal release packet lacks publication identities")
    for field in (
        "combined_manifest_sha256",
        "train_parent_manifest_sha256",
        "development_tail_manifest_sha256",
    ):
        if packet_publication.get(field) != authorized_publication.get(field):
            raise TrainingGateError("terminal release packet publication identity drifted")
    if (
        int(packet_publication.get("logical_train_accepted_count", -1))
        != TRAIN_131K_COUNT
        or int(packet_publication.get("development_tail_accepted_count", -1))
        != TAIL_COUNT
    ):
        raise TrainingGateError("terminal release packet publication counts changed")

    packet_training = packet.get("immutable_training", {})
    authorized_training = authorization.get("immutable_training", {})
    if not isinstance(packet_training, Mapping) or not isinstance(
        authorized_training, Mapping
    ):
        raise TrainingGateError("terminal release packet lacks training identities")
    for field in (
        "git_commit",
        "wheel_path",
        "wheel_filename",
        "wheel_sha256",
        "model_configuration_path",
        "model_configuration_hash",
        "environment_lock_path",
        "environment_lock_sha256",
    ):
        if packet_training.get(field) != authorized_training.get(field):
            raise TrainingGateError("terminal release packet training identity drifted")
    gpu_names = packet_training.get("observed_gpu_names")
    if not isinstance(gpu_names, list) or not (
        len(gpu_names) >= 3
        and all(str(name) == EXPECTED_TERMINAL_GPU_MODEL for name in gpu_names)
        and packet_training.get("editable_install_authorized") is False
        and packet_training.get("cuda_required") is True
    ):
        raise TrainingGateError("terminal release packet CUDA identity drifted")
    for evidence_field in (
        "exact_wheel_test_result_sha256",
        "exact_wheel_test_result_path",
    ):
        if not packet_training.get(evidence_field):
            raise TrainingGateError("terminal release packet lacks exact-wheel evidence")
    if (
        packet.get("final_evaluation_commitment_sha256")
        != authorization.get("final_evaluation_commitment_sha256")
    ):
        raise TrainingGateError("terminal release packet commitment identity drifted")
    packet_retained = packet.get("retained_65k_probe", {})
    authorized_retained = authorization.get("retained_65k_probe", {})
    if (
        not isinstance(packet_retained, Mapping)
        or not isinstance(authorized_retained, Mapping)
        or dict(packet_retained) != dict(authorized_retained)
    ):
        raise TrainingGateError("terminal retained 65k probe identity drifted")
    return packet


@dataclass(frozen=True)
class Terminal131TrainingPublication:
    """One validated logical 131k train reference and development-tail parent."""

    combined_root: Path
    combined_manifest_path: Path
    combined_manifest_sha256: str
    corrected_65k: CorrectedTrainingPublication
    train_parent_root: Path
    train_parent_manifest_sha256: str
    train_dataset_id: str
    train_increment_root: Path
    train_increment_manifest_sha256: str
    development_tail_parent_root: Path
    development_tail_manifest_sha256: str
    development_tail_dataset_ids: Mapping[str, str]
    development_tail_shards_per_namespace: int
    development_tail_pairs_per_shard: int


def _validated_tail_shard_layout(manifest: Mapping[str, Any]) -> Tuple[int, int]:
    """Accept only reviewed physical layouts that preserve 128 cases."""

    fields_present = (
        "shards_per_namespace" in manifest,
        "accepted_pairs_per_shard" in manifest,
    )
    if any(fields_present) and not all(fields_present):
        raise TrainingGateError("terminal tail shard layout is only partially declared")
    if not any(fields_present):
        return 1, 128
    layout = (
        int(manifest["shards_per_namespace"]),
        int(manifest["accepted_pairs_per_shard"]),
    )
    if layout not in ((1, 128), (32, 4), (128, 1)) or layout[0] * layout[1] != 128:
        raise TrainingGateError("terminal tail physical shard layout is unauthorized")
    return layout


def _resolve_corrected_from_authorization(
    authorization: Mapping[str, Any],
    *,
    stage_a_publication_root: Path,
    stage_b_publication_root: Path,
    combined_base_publication_root: Path,
    correction_publication_root: Path,
) -> CorrectedTrainingPublication:
    corrected = authorization.get("corrected_65k_publication", {})
    return resolve_corrected_training_publication(
        correction_publication_root,
        stage_a_parent_root=stage_a_publication_root,
        stage_b_parent_root=stage_b_publication_root,
        combined_base_root=combined_base_publication_root,
        expected_base_generator_commit=str(corrected.get("base_generator_commit", "")),
        expected_base_preregistration_hash=str(
            corrected.get("base_preregistration_hash", "")
        ),
        expected_correction_generator_commit=str(
            corrected.get("correction_generator_commit", "")
        ),
        expected_correction_preregistration_hash=str(
            corrected.get("correction_preregistration_hash", "")
        ),
        expected_correction_manifest_sha256=str(
            corrected.get("correction_parent_manifest_sha256", "")
        ),
        expected_correction_tree_sha256=str(
            corrected.get("correction_publication_tree_sha256", "")
        ),
        expected_combined_base_manifest_sha256=str(
            corrected.get("combined_base_manifest_sha256", "")
        ),
    )


def resolve_terminal_131k_training_publication(
    authorization: Mapping[str, Any],
    *,
    stage_a_publication_root: Path,
    stage_b_publication_root: Path,
    combined_base_publication_root: Path,
    correction_publication_root: Path,
    train_parent_root: Path,
    combined_131k_publication_root: Path,
    development_tail_parent_root: Path,
) -> Terminal131TrainingPublication:
    """Resolve all terminal parents from manifests without opening strain arrays."""

    corrected = _resolve_corrected_from_authorization(
        authorization,
        stage_a_publication_root=stage_a_publication_root,
        stage_b_publication_root=stage_b_publication_root,
        combined_base_publication_root=combined_base_publication_root,
        correction_publication_root=correction_publication_root,
    )
    terminal = authorization.get("terminal_publication", {})
    combined_root = combined_131k_publication_root.resolve()
    combined_manifest_path = combined_root / "dataset_manifest.json"
    if "published" not in combined_root.parts or not combined_manifest_path.is_file():
        raise TrainingGateError("terminal 131k reference is not atomically published")
    combined_hash = _sha256_file(combined_manifest_path)
    if combined_hash != terminal.get("combined_manifest_sha256"):
        raise TrainingGateError("terminal 131k combined manifest hash mismatch")
    combined = _load_json(combined_manifest_path)
    components = {
        str(item.get("role")): item
        for item in combined.get("components", ())
        if isinstance(item, dict)
    }
    corrected_component = components.get("corrected_train_65k")
    increment_component = components.get("terminal_131k_train_increment")
    if not isinstance(corrected_component, dict) or not isinstance(
        increment_component, dict
    ):
        raise TrainingGateError("terminal 131k manifest lacks its two train components")
    if (
        combined.get("status"),
        int(combined.get("accepted_physical_system_count", -1)),
        int(combined.get("validation_physical_system_count", -1)),
        combined.get("strict_nested_train_ladder"),
        combined.get("proposal_equals_evaluation"),
        combined.get("all_importance_weights_one"),
        combined.get("training_authorized"),
        combined.get("architecture_selection_authorized"),
        combined.get("calibration_authorized"),
        combined.get("sbc_authorized"),
        combined.get("final_evaluation_authorized"),
        combined.get("extension_above_131072_authorized"),
        combined.get("gwosc_gwtc_accessed"),
        int(corrected_component.get("accepted_count", -1)),
        corrected_component.get("logical_manifest_sha256"),
        int(increment_component.get("accepted_count", -1)),
    ) != (
        "passed",
        TRAIN_131K_COUNT,
        VALIDATION_COUNT,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        65536,
        corrected.corrected_combined_train_manifest_sha256,
        TRAIN_INCREMENT_COUNT,
    ):
        raise TrainingGateError("terminal 131k combined manifest contract failed")

    train_parent = train_parent_root.resolve()
    train_manifest_path = train_parent / "dataset_manifest.json"
    if "published" not in train_parent.parts or not train_manifest_path.is_file():
        raise TrainingGateError("terminal train increment parent is not published")
    train_parent_hash = _sha256_file(train_manifest_path)
    if train_parent_hash != increment_component.get("parent_manifest_sha256"):
        raise TrainingGateError("terminal train parent hash differs from combined manifest")
    if train_parent_hash != terminal.get("train_parent_manifest_sha256"):
        raise TrainingGateError("terminal train parent hash differs from authorization")
    train_manifest = _load_json(train_manifest_path)
    train_dataset_id = str(increment_component.get("dataset_id", ""))
    if (
        train_manifest.get("status"),
        int(train_manifest.get("accepted_pair_count", -1)),
        int(train_manifest.get("complete_shard_count", -1)),
        train_manifest.get("dataset_id"),
        train_manifest.get("proposal_equals_evaluation"),
        train_manifest.get("all_importance_weights_one"),
        train_manifest.get("training_authorized"),
    ) != ("passed", TRAIN_INCREMENT_COUNT, 512, train_dataset_id, True, True, False):
        raise TrainingGateError("terminal train increment parent contract failed")
    increment_root = train_parent / train_dataset_id
    if not increment_root.is_dir():
        raise TrainingGateError("terminal train increment dataset is absent")

    tail_parent = development_tail_parent_root.resolve()
    tail_manifest_path = tail_parent / "dataset_manifest.json"
    if "published" not in tail_parent.parts or not tail_manifest_path.is_file():
        raise TrainingGateError("terminal development-tail parent is not published")
    tail_hash = _sha256_file(tail_manifest_path)
    if tail_hash != combined.get("development_tail_manifest_sha256"):
        raise TrainingGateError("terminal tail hash differs from combined manifest")
    if tail_hash != terminal.get("development_tail_manifest_sha256"):
        raise TrainingGateError("terminal tail hash differs from authorization")
    tail_manifest = _load_json(tail_manifest_path)
    dataset_ids = tail_manifest.get("dataset_ids")
    if not isinstance(dataset_ids, dict) or set(dataset_ids) != set(TAIL_STRATA):
        raise TrainingGateError("terminal tail dataset identities are incomplete")
    if (
        tail_manifest.get("status"),
        int(tail_manifest.get("accepted_pair_count", -1)),
        int(tail_manifest.get("namespace_count", -1)),
        int(tail_manifest.get("cases_per_stratum", -1)),
        tail_manifest.get("dataset_role"),
        tail_manifest.get("training_use_authorized"),
        tail_manifest.get("architecture_selection_use_authorized"),
        tail_manifest.get("calibration_use_authorized"),
        tail_manifest.get("final_claim_use_authorized"),
        tail_manifest.get("final_evaluation_unsealed"),
        tail_manifest.get("gwosc_gwtc_accessed"),
    ) != (
        "passed",
        TAIL_COUNT,
        4,
        128,
        "development_only_tail_diagnostic",
        False,
        False,
        False,
        False,
        False,
        False,
    ):
        raise TrainingGateError("terminal development-tail parent contract failed")
    if any(not (tail_parent / str(dataset_id)).is_dir() for dataset_id in dataset_ids.values()):
        raise TrainingGateError("terminal development-tail child dataset is absent")
    tail_shards_per_namespace, tail_pairs_per_shard = _validated_tail_shard_layout(
        tail_manifest
    )
    return Terminal131TrainingPublication(
        combined_root=combined_root,
        combined_manifest_path=combined_manifest_path,
        combined_manifest_sha256=combined_hash,
        corrected_65k=corrected,
        train_parent_root=train_parent,
        train_parent_manifest_sha256=train_parent_hash,
        train_dataset_id=train_dataset_id,
        train_increment_root=increment_root,
        train_increment_manifest_sha256=_sha256_file(train_manifest_path),
        development_tail_parent_root=tail_parent,
        development_tail_manifest_sha256=tail_hash,
        development_tail_dataset_ids={
            str(key): str(value) for key, value in dataset_ids.items()
        },
        development_tail_shards_per_namespace=tail_shards_per_namespace,
        development_tail_pairs_per_shard=tail_pairs_per_shard,
    )


def terminal_131k_training_dataset(
    publication: Terminal131TrainingPublication,
    detector_curves: Mapping[str, Any],
) -> ConcatenatedPublishedStageADataset:
    retained = corrected_65k_training_dataset(publication.corrected_65k, detector_curves)
    increment = PublishedStageADataset(
        publication.train_increment_root,
        expected_split=SplitName.TRAIN,
        detector_curves=detector_curves,
        expected_total_pairs=TRAIN_INCREMENT_COUNT,
    )
    result = ConcatenatedPublishedStageADataset((retained, increment))
    if len(result) != TRAIN_131K_COUNT:
        raise TrainingGateError("terminal training reader is not exactly 131,072 systems")
    return result


class TerminalDevelopmentTailDataset:
    """Expose the four frozen tail namespaces with labels outside model tensors."""

    def __init__(
        self,
        publication: Terminal131TrainingPublication,
        detector_curves: Mapping[str, Any],
        standardizer: InputStandardizer,
    ) -> None:
        datasets = []
        for stratum in TAIL_STRATA:
            dataset_id = publication.development_tail_dataset_ids[stratum]
            dataset = PublishedStageADataset(
                publication.development_tail_parent_root / dataset_id,
                expected_split=SplitName.BALANCED_TAIL_DIAGNOSTIC,
                detector_curves=detector_curves,
                expected_pairs_per_shard=(
                    publication.development_tail_pairs_per_shard
                ),
                expected_total_pairs=128,
            )
            datasets.append((stratum, dataset))
        identifiers = tuple(
            identifier
            for _, dataset in datasets
            for identifier in dataset.physical_system_ids()
        )
        if len(identifiers) != TAIL_COUNT or len(set(identifiers)) != TAIL_COUNT:
            raise TrainingGateError("terminal development-tail membership is invalid")
        self.datasets = tuple(datasets)
        self.standardizer = standardizer

    def __len__(self) -> int:
        return TAIL_COUNT

    def __getitem__(self, index: int) -> DevelopmentCase:
        if not 0 <= index < TAIL_COUNT:
            raise IndexError(index)
        dataset_index, row_index = divmod(index, 128)
        stratum, dataset = self.datasets[dataset_index]
        return DevelopmentCase(
            example=self.standardizer.transform(dataset[row_index]),
            tail_view=stratum,
        )

    def physical_system_ids(self) -> Tuple[str, ...]:
        return tuple(
            identifier
            for _, dataset in self.datasets
            for identifier in dataset.physical_system_ids()
        )


def validate_terminal_131k_training_gate(
    root: Path,
    *,
    authorization_path: Path,
    stage_a_publication_root: Path,
    stage_b_publication_root: Path,
    combined_base_publication_root: Path,
    correction_publication_root: Path,
    train_parent_root: Path,
    combined_131k_publication_root: Path,
    development_tail_parent_root: Path,
) -> Mapping[str, Any]:
    """Require an exact post-publication gate before indexing terminal data."""

    authorization = load_yaml(authorization_path)
    if authorization.get("authorization_status") != "authorized_terminal_131k_probe_only":
        raise TrainingGateError("terminal 131k probe-training authorization is absent")
    flags = authorization.get("authorization", {})
    for required in (
        "corrected_65k_data_access_authorized",
        "terminal_train_increment_data_access_authorized",
        "development_tail_data_access_authorized",
        "scientific_131k_probe_training_authorized",
        "probe_optimizer_execution_authorized",
        "terminal_learning_curve_decision_authorized",
    ):
        if flags.get(required) is not True:
            raise TrainingGateError(f"terminal 131k gate requires {required}=true")
    for forbidden in (
        "model_tuning_authorized",
        "architecture_selection_authorized",
        "calibration_authorized",
        "sbc_authorized",
        "final_evaluation_authorized",
        "extension_above_131072_authorized",
        "real_noise_authorized",
        "gwosc_gwtc_access_authorized",
    ):
        if flags.get(forbidden) is not False:
            raise TrainingGateError(f"terminal 131k gate must keep {forbidden}=false")
    if authorization.get("authorized_training_rungs") != [131072]:
        raise TrainingGateError("terminal gate must authorize only the 131k rung")
    if authorization.get("authorized_training_seeds") != [0, 1, 2]:
        raise TrainingGateError("terminal gate must authorize exactly seeds 0/1/2")

    preregistration_path = root / TERMINAL_PREREGISTRATION_PATH
    preregistration = load_yaml(preregistration_path)
    frozen = authorization.get("frozen_preregistration", {})
    if (
        frozen.get("version") != "1.2.0-rc.1"
        or frozen.get("path") != TERMINAL_PREREGISTRATION_PATH
        or frozen.get("canonical_hash") != configuration_hash(preregistration)
    ):
        raise TrainingGateError("terminal 131k preregistration identity mismatch")
    terminal_decision = preregistration.get("terminal_outcomes", {})
    if not (
        terminal_decision.get("both_outcomes", {}).get("selected_training_count")
        == TRAIN_131K_COUNT
        and terminal_decision.get("both_outcomes", {}).get(
            "extension_above_131072_authorized"
        )
        is False
    ):
        raise TrainingGateError("terminal 131k resource-cap outcome changed")
    commitment_path = root / "results/phase4/final_evaluation_commitment.json"
    commitment = _load_json(commitment_path)
    commitment_hash = _sha256_file(commitment_path)
    if (
        commitment.get("commitment_status") != "finalized_before_training"
        or authorization.get("final_evaluation_commitment_sha256") != commitment_hash
    ):
        raise TrainingGateError("terminal gate lacks the finalized evaluation commitment")
    release_packet = validate_terminal_probe_release_binding(root, authorization)
    _validate_authorized_publication_roots(
        authorization,
        stage_a=stage_a_publication_root,
        stage_b=stage_b_publication_root,
        combined_base=combined_base_publication_root,
        correction=correction_publication_root,
        terminal_train_increment=train_parent_root,
        terminal_combined_131k=combined_131k_publication_root,
        development_tail=development_tail_parent_root,
    )
    publication = resolve_terminal_131k_training_publication(
        authorization,
        stage_a_publication_root=stage_a_publication_root,
        stage_b_publication_root=stage_b_publication_root,
        combined_base_publication_root=combined_base_publication_root,
        correction_publication_root=correction_publication_root,
        train_parent_root=train_parent_root,
        combined_131k_publication_root=combined_131k_publication_root,
        development_tail_parent_root=development_tail_parent_root,
    )
    return {
        "authorization": authorization,
        "publication": publication,
        "final_evaluation_commitment_sha256": commitment_hash,
        "terminal_preregistration_hash": frozen["canonical_hash"],
        "terminal_probe_release_packet": release_packet,
        "terminal_probe_release_packet_sha256": authorization[
            "terminal_probe_release_review"
        ]["sha256"],
    }


def _validate_authorized_publication_roots(
    authorization: Mapping[str, Any], **observed: Path
) -> None:
    """Bind every CLI data root to the separately reviewed authorization."""

    configured = authorization.get("publication_roots")
    if not isinstance(configured, Mapping) or set(configured) != set(observed):
        raise TrainingGateError("terminal gate publication-root set changed")
    for name, path in observed.items():
        if Path(str(configured[name])).resolve() != path.resolve():
            raise TrainingGateError(
                f"terminal gate publication root changed for {name}"
            )


def _terminal_datasets(
    publication: Terminal131TrainingPublication,
    curves: Mapping[str, Any],
) -> Tuple[
    ConcatenatedPublishedStageADataset,
    PublishedStageADataset,
]:
    training = terminal_131k_training_dataset(publication, curves)
    validation = PublishedStageADataset(
        publication.corrected_65k.stage_a.validation_root,
        expected_split=SplitName.VALIDATION,
        detector_curves=curves,
        expected_total_pairs=VALIDATION_COUNT,
    )
    train_ids = set(training.physical_system_ids())
    validation_ids = set(validation.physical_system_ids())
    if len(train_ids) != TRAIN_131K_COUNT or train_ids & validation_ids:
        raise TrainingGateError("terminal train and core validation membership is invalid")
    return training, validation


def _evaluation_seed(
    evaluation: Mapping[str, Any], *, membership_sha256: str, seed: int, suffix: str
) -> int:
    payload = (
        f"{evaluation['posterior_draw_seed_domain']}\0{membership_sha256}\0{seed}\0{suffix}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def run_authorized_131k_probe(
    root: Path,
    *,
    authorization_path: Path,
    stage_a_publication_root: Path,
    stage_b_publication_root: Path,
    combined_base_publication_root: Path,
    correction_publication_root: Path,
    train_parent_root: Path,
    combined_131k_publication_root: Path,
    development_tail_parent_root: Path,
    environment_lock_path: Path,
    psd_root: Path,
    output_root: Path,
    training_commit: str,
    seed: int,
    device_name: str,
    resume_checkpoint: Optional[Path] = None,
    execute_optimizer: bool = True,
) -> Mapping[str, Any]:
    """Prepare or fit one frozen terminal 131k seed from scratch."""

    if seed not in SEEDS:
        raise TrainingGateError("terminal probe seed is outside 0/1/2")
    gate = validate_terminal_131k_training_gate(
        root,
        authorization_path=authorization_path,
        stage_a_publication_root=stage_a_publication_root,
        stage_b_publication_root=stage_b_publication_root,
        combined_base_publication_root=combined_base_publication_root,
        correction_publication_root=correction_publication_root,
        train_parent_root=train_parent_root,
        combined_131k_publication_root=combined_131k_publication_root,
        development_tail_parent_root=development_tail_parent_root,
    )
    authorization = gate["authorization"]
    artifacts = validate_immutable_training_artifacts(
        root,
        authorization.get("immutable_training", {}),
        training_commit=training_commit,
        environment_lock_path=environment_lock_path,
    )
    _verify_training_checkout(root, training_commit, authorization_path)
    configured_output = Path(str(authorization.get("training_output_root", ""))).resolve()
    if configured_output != output_root.resolve():
        raise TrainingGateError("terminal probe output differs from authorization")
    if not output_root.is_absolute() or not output_root.is_relative_to(
        Path("/root/autodl-tmp/lensing-4")
    ):
        raise TrainingGateError("terminal probe output escaped the AutoDL project root")
    if int(authorization.get("maximum_concurrent_fits", -1)) != 3:
        raise TrainingGateError("terminal gate must permit exactly three concurrent fits")
    model = load_yaml(root / "configs/models/phase4_probe_nsf.yaml")
    immutable = authorization.get("immutable_training", {})
    if immutable.get("model_configuration_hash") != model_configuration_hash(model):
        raise TrainingGateError("terminal probe model configuration hash mismatch")
    _validate_runtime_versions(model)
    load_input_policy(root)
    publication = gate["publication"]
    if not isinstance(publication, Terminal131TrainingPublication):
        raise TrainingGateError("terminal gate returned the wrong publication type")
    curves = _verified_curves(model, psd_root)
    train_dataset, validation_dataset = _terminal_datasets(publication, curves)
    train_ids = train_dataset.physical_system_ids()
    validation_manifest_sha256 = publication.corrected_65k.stage_a.namespace_manifest_sha256[
        "validation"
    ]
    rung_directory = output_root / "rung-131072"
    preparation_path = rung_directory / "rung_preparation.json"
    if preparation_path.is_file():
        preparation = _load_json(preparation_path)
        if not (
            preparation.get("status") == "ready_for_authorized_probe_fits"
            and tuple(preparation.get("member_ids", ())) == train_ids
            and preparation.get("combined_manifest_sha256")
            == publication.combined_manifest_sha256
            and preparation.get("model_configuration_hash")
            == model_configuration_hash(model)
            and preparation.get("terminal_probe_release_packet_sha256")
            == gate["terminal_probe_release_packet_sha256"]
        ):
            raise TrainingGateError("terminal rung preparation identity mismatch")
        input_standardizer, target_standardizer = _load_standardizers(preparation)
    else:
        if execute_optimizer:
            raise TrainingGateError("terminal shared preprocessing must complete before fits")
        input_standardizer, target_standardizer = fit_rung_standardizers(train_dataset)
        preparation = {
            "status": "ready_for_authorized_probe_fits",
            "rung_count": TRAIN_131K_COUNT,
            "member_count": len(train_ids),
            "member_ids": list(train_ids),
            "member_ids_sha256": membership_hash(train_ids),
            "combined_manifest_sha256": publication.combined_manifest_sha256,
            "corrected_65k_manifest_sha256": (
                publication.corrected_65k.corrected_combined_train_manifest_sha256
            ),
            "train_increment_manifest_sha256": (
                publication.train_increment_manifest_sha256
            ),
            "validation_manifest_sha256": validation_manifest_sha256,
            "development_tail_manifest_sha256": (
                publication.development_tail_manifest_sha256
            ),
            "final_evaluation_commitment_sha256": gate[
                "final_evaluation_commitment_sha256"
            ],
            "terminal_probe_release_packet_sha256": gate[
                "terminal_probe_release_packet_sha256"
            ],
            "model_configuration_hash": model_configuration_hash(model),
            "input_standardizer": asdict(input_standardizer),
            "input_standardizer_sha256": standardizer_hash(input_standardizer),
            "target_standardizer": asdict(target_standardizer),
            "target_standardizer_sha256": standardizer_hash(target_standardizer),
            "strain_products_opened": False,
            "optimizer_started": False,
            "calibration_accessed": False,
            "final_evaluation_accessed": False,
        }
        _atomic_json(preparation_path, preparation)
    if not execute_optimizer:
        return preparation

    identity = TrainingRunIdentity(
        model_configuration_hash=model_configuration_hash(model),
        training_code_commit=training_commit,
        training_environment_sha256=artifacts["environment_lock_sha256"],
        train_manifest_sha256=publication.combined_manifest_sha256,
        validation_manifest_sha256=validation_manifest_sha256,
        final_evaluation_commitment_sha256=str(
            gate["final_evaluation_commitment_sha256"]
        ),
        membership_sha256=membership_hash(train_ids),
        input_standardizer_sha256=standardizer_hash(input_standardizer),
        target_standardizer_sha256=standardizer_hash(target_standardizer),
        training_rung_count=TRAIN_131K_COUNT,
        seed=seed,
    )
    identity.validate()
    run_directory = rung_directory / f"seed-{seed}"
    if run_directory.exists() and resume_checkpoint is None:
        raise FileExistsError("terminal probe identity already has an output directory")
    evidence = {
        **authorized_probe_execution_evidence(identity),
        "terminal_publication_validated": True,
        "combined_manifest_sha256": publication.combined_manifest_sha256,
        "immutable_wheel_sha256": artifacts["wheel_sha256"],
        "terminal_probe_release_packet_sha256": gate[
            "terminal_probe_release_packet_sha256"
        ],
        "member_count": len(train_ids),
        "member_ids_sha256": membership_hash(train_ids),
        "member_ids": list(train_ids),
        "input_standardizer": asdict(input_standardizer),
        "target_standardizer": asdict(target_standardizer),
        "architecture_selection_authorized": False,
        "calibration_authorized": False,
        "final_evaluation_accessed": False,
        "extension_above_131072_authorized": False,
    }
    _atomic_json(run_directory / "run_preparation.json", evidence)
    workers = int(authorization.get("data_loader_worker_processes", 0))
    if not 0 <= workers <= 16:
        raise TrainingGateError("terminal data-loader worker count is outside review")
    standardized_train = StandardizedStageADataset(train_dataset, input_standardizer)
    standardized_validation = StandardizedStageADataset(
        validation_dataset, input_standardizer
    )
    _, physical_microbatch_size, _ = optimization_batch_geometry(model["optimization"])
    train_loader = _data_loader(
        standardized_train,
        batch_size=physical_microbatch_size,
        seed=seed,
        training=True,
        worker_processes=workers,
        device_name=device_name,
    )
    validation_loader = _data_loader(
        standardized_validation,
        batch_size=physical_microbatch_size,
        seed=seed,
        training=False,
        worker_processes=workers,
        device_name=device_name,
    )
    probe_model = build_probe_model(model, seed=seed)
    training_summary = train_probe(
        probe_model,
        train_loader,
        validation_loader,
        config=model,
        identity=identity,
        input_standardizer=input_standardizer,
        standardizer=target_standardizer,
        execution_evidence=evidence,
        output_directory=run_directory,
        device_name=device_name,
        resume_checkpoint=resume_checkpoint,
    )
    evaluation = model["development_evaluation"]
    development_loader = _development_loader(
        DevelopmentStageADataset(validation_dataset, input_standardizer),
        batch_size=int(evaluation["batch_size"]),
        seed=seed,
        worker_processes=workers,
        device_name=device_name,
    )
    core_summary = evaluate_development_validation(
        probe_model,
        development_loader,
        standardizer=target_standardizer,
        device_name=device_name,
        posterior_draws_per_case=int(evaluation["posterior_draws_per_case"]),
        evaluation_seed=_evaluation_seed(
            evaluation,
            membership_sha256=identity.membership_sha256,
            seed=seed,
            suffix="core_validation",
        ),
        output_directory=run_directory,
        levels=tuple(float(value) for value in evaluation["coverage_levels"]),
    )
    tail_dataset = TerminalDevelopmentTailDataset(
        publication, curves, input_standardizer
    )
    tail_loader = _development_loader(
        tail_dataset,
        batch_size=int(evaluation["batch_size"]),
        seed=seed,
        worker_processes=workers,
        device_name=device_name,
    )
    tail_directory = output_root / "terminal-tail" / "rung-131072" / f"seed-{seed}"
    tail_summary = evaluate_development_validation(
        probe_model,
        tail_loader,
        standardizer=target_standardizer,
        device_name=device_name,
        posterior_draws_per_case=int(evaluation["posterior_draws_per_case"]),
        evaluation_seed=_evaluation_seed(
            evaluation,
            membership_sha256=identity.membership_sha256,
            seed=seed,
            suffix="terminal_development_tail",
        ),
        output_directory=tail_directory,
        levels=tuple(float(value) for value in evaluation["coverage_levels"]),
    )
    result = {
        "status": "completed_131k_probe_fit_and_development_validation",
        "identity": asdict(identity),
        "training": training_summary,
        "development": core_summary,
        "development_tail": tail_summary,
        "terminal_probe_release_packet_sha256": gate[
            "terminal_probe_release_packet_sha256"
        ],
        "architecture_selection_authorized": False,
        "calibration_accessed": False,
        "final_evaluation_accessed": False,
        "extension_above_131072_authorized": False,
    }
    _atomic_json(run_directory / "run_summary.json", result)
    return result


def evaluate_retained_65k_on_terminal_tail(
    root: Path,
    *,
    authorization_path: Path,
    stage_a_publication_root: Path,
    stage_b_publication_root: Path,
    combined_base_publication_root: Path,
    correction_publication_root: Path,
    train_parent_root: Path,
    combined_131k_publication_root: Path,
    development_tail_parent_root: Path,
    environment_lock_path: Path,
    psd_root: Path,
    retained_65k_output_root: Path,
    output_root: Path,
    training_commit: str,
    seed: int,
    device_name: str,
) -> Mapping[str, Any]:
    """Evaluate one retained corrected-65k checkpoint on the new sealed tail pool."""

    if seed not in SEEDS:
        raise TrainingGateError("retained terminal-tail seed is outside 0/1/2")
    gate = validate_terminal_131k_training_gate(
        root,
        authorization_path=authorization_path,
        stage_a_publication_root=stage_a_publication_root,
        stage_b_publication_root=stage_b_publication_root,
        combined_base_publication_root=combined_base_publication_root,
        correction_publication_root=correction_publication_root,
        train_parent_root=train_parent_root,
        combined_131k_publication_root=combined_131k_publication_root,
        development_tail_parent_root=development_tail_parent_root,
    )
    authorization = gate["authorization"]
    validate_immutable_training_artifacts(
        root,
        authorization.get("immutable_training", {}),
        training_commit=training_commit,
        environment_lock_path=environment_lock_path,
    )
    configured_retained = Path(
        str(authorization.get("retained_65k_output_root", ""))
    ).resolve()
    configured_output = Path(str(authorization.get("training_output_root", ""))).resolve()
    if retained_65k_output_root.resolve() != configured_retained:
        raise TrainingGateError("retained 65k output differs from authorization")
    if output_root.resolve() != configured_output:
        raise TrainingGateError("terminal tail output differs from authorization")
    model_config = load_yaml(root / "configs/models/phase4_probe_nsf.yaml")
    publication = gate["publication"]
    if not isinstance(publication, Terminal131TrainingPublication):
        raise TrainingGateError("terminal gate returned the wrong publication type")
    curves = _verified_curves(model_config, psd_root)
    checkpoint_path = (
        retained_65k_output_root / "rung-65536" / f"seed-{seed}" / "best.ckpt"
    )
    summary_path = checkpoint_path.parent / "run_summary.json"
    retained_binding = authorization.get("retained_65k_probe", {})
    retained_artifacts = (
        retained_binding.get("artifacts", {})
        if isinstance(retained_binding, Mapping)
        else {}
    )
    expected_artifact = (
        retained_artifacts.get(str(seed), {})
        if isinstance(retained_artifacts, Mapping)
        else {}
    )
    if (
        not isinstance(retained_binding, Mapping)
        or Path(str(retained_binding.get("output_root", ""))).resolve()
        != retained_65k_output_root.resolve()
        or int(retained_binding.get("training_rung_count", -1)) != 65536
        or not isinstance(expected_artifact, Mapping)
        or not checkpoint_path.is_file()
        or not summary_path.is_file()
        or _sha256_file(checkpoint_path)
        != expected_artifact.get("best_checkpoint_sha256")
        or _sha256_file(summary_path) != expected_artifact.get("run_summary_sha256")
    ):
        raise TrainingGateError("retained corrected-65k artifact hash mismatch")
    retained_summary = _load_json(summary_path)
    retained_identity = retained_summary.get("identity", {})
    shared_identity = retained_binding.get("shared_identity", {})
    expected_identity = {
        **(dict(shared_identity) if isinstance(shared_identity, Mapping) else {}),
        "seed": seed,
        "training_rung_count": 65536,
    }
    if (
        retained_summary.get("status")
        != "completed_65k_probe_fit_and_development_validation"
        or not isinstance(retained_identity, Mapping)
        or dict(retained_identity) != expected_identity
    ):
        raise TrainingGateError("retained corrected-65k run summary changed")
    torch = importlib.import_module("torch")
    device = torch.device(device_name)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    identity = state.get("identity", {})
    if (
        int(identity.get("training_rung_count", -1)) != 65536
        or int(identity.get("seed", -1)) != seed
        or identity.get("model_configuration_hash")
        != model_configuration_hash(model_config)
        or dict(identity) != dict(retained_identity)
    ):
        raise TrainingGateError("retained corrected-65k checkpoint identity mismatch")
    input_standardizer, target_standardizer = _load_standardizers(
        {
            "input_standardizer": state["input_standardizer"],
            "target_standardizer": state["target_standardizer"],
            "input_standardizer_sha256": identity["input_standardizer_sha256"],
            "target_standardizer_sha256": identity["target_standardizer_sha256"],
        }
    )
    model = build_probe_model(model_config, seed=seed)
    model.load_state_dict(state["model"])
    model.to(device)
    tail_dataset = TerminalDevelopmentTailDataset(
        publication, curves, input_standardizer
    )
    evaluation = model_config["development_evaluation"]
    workers = int(authorization.get("data_loader_worker_processes", 0))
    loader = _development_loader(
        tail_dataset,
        batch_size=int(evaluation["batch_size"]),
        seed=seed,
        worker_processes=workers,
        device_name=device_name,
    )
    output_directory = output_root / "terminal-tail" / "rung-65536" / f"seed-{seed}"
    if output_directory.exists():
        raise FileExistsError("retained terminal-tail result identity already exists")
    summary = evaluate_development_validation(
        model,
        loader,
        standardizer=target_standardizer,
        device_name=device_name,
        posterior_draws_per_case=int(evaluation["posterior_draws_per_case"]),
        evaluation_seed=_evaluation_seed(
            evaluation,
            membership_sha256=str(identity["membership_sha256"]),
            seed=seed,
            suffix="terminal_development_tail",
        ),
        output_directory=output_directory,
        levels=tuple(float(value) for value in evaluation["coverage_levels"]),
    )
    result = {
        "status": "completed_retained_65k_terminal_tail_evaluation",
        "seed": seed,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_identity": identity,
        "terminal_probe_release_packet_sha256": gate[
            "terminal_probe_release_packet_sha256"
        ],
        "development_tail": summary,
        "checkpoint_retrained": False,
        "architecture_selection_authorized": False,
        "calibration_accessed": False,
        "final_evaluation_accessed": False,
    }
    _atomic_json(output_directory / "run_summary.json", result)
    return result
