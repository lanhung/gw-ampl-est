"""Terminal-131k adapter for the frozen post-lock architecture grid."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from ..config import load_yaml
from ..schema import SplitName
from .architecture import (
    GRID_PATH,
    NEW_ARCHITECTURE_IDS,
    PROBE_ARCHITECTURE_ID,
    candidate_model_configuration,
    load_architecture_specs,
)
from .contracts import TrainingGateError, model_configuration_hash
from .data import DevelopmentStageADataset, PublishedStageADataset, StandardizedStageADataset
from .engine import (
    TrainingRunIdentity,
    evaluate_development_validation,
    membership_hash,
    optimization_batch_geometry,
    standardizer_hash,
    train_probe,
)
from .features import load_input_policy
from .model import build_probe_model
from .rung65 import (
    SEEDS,
    VALIDATION_COUNT,
    _load_standardizers,
    validate_immutable_training_artifacts,
)
from .runner import (
    _atomic_json,
    _data_loader,
    _development_loader,
    _validate_runtime_versions,
    _verified_curves,
    _verify_training_checkout,
)
from .terminal131 import (
    TRAIN_131K_COUNT,
    Terminal131TrainingPublication,
    resolve_terminal_131k_training_publication,
    terminal_131k_training_dataset,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TrainingGateError(f"expected JSON mapping: {path}")
    return value


def _validate_terminal_probe_reuse(
    *,
    probe_output_root: Path,
    authorization: Mapping[str, Any],
    train_manifest_sha256: str,
    validation_manifest_sha256: str,
) -> Tuple[Mapping[str, Any], ...]:
    expected_hashes = authorization.get("reused_probe_run_summary_sha256", {})
    expected_checkpoint_hashes = authorization.get(
        "reused_probe_best_checkpoint_sha256", {}
    )
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != {
        str(seed) for seed in SEEDS
    }:
        raise TrainingGateError("terminal architecture gate must bind three probe fits")
    if not isinstance(expected_checkpoint_hashes, dict) or set(
        expected_checkpoint_hashes
    ) != {str(seed) for seed in SEEDS}:
        raise TrainingGateError(
            "terminal architecture gate must bind three probe checkpoints"
        )
    expected_model_hash = str(
        authorization.get("reused_probe_model_configuration_hash", "")
    )
    summaries = []
    for seed in SEEDS:
        path = probe_output_root / "rung-131072" / f"seed-{seed}" / "run_summary.json"
        checkpoint_path = path.parent / "best.ckpt"
        if not path.is_file() or _sha256(path) != expected_hashes[str(seed)]:
            raise TrainingGateError("terminal reused probe summary hash mismatch")
        if (
            not checkpoint_path.is_file()
            or _sha256(checkpoint_path) != expected_checkpoint_hashes[str(seed)]
        ):
            raise TrainingGateError("terminal reused probe checkpoint hash mismatch")
        summary = _load_mapping(path)
        identity = summary.get("identity", {})
        development = summary.get("development", {})
        if (
            summary.get("status")
            != "completed_131k_probe_fit_and_development_validation"
            or int(identity.get("training_rung_count", -1)) != TRAIN_131K_COUNT
            or int(identity.get("seed", -1)) != seed
            or identity.get("model_configuration_hash") != expected_model_hash
            or identity.get("train_manifest_sha256") != train_manifest_sha256
            or identity.get("validation_manifest_sha256") != validation_manifest_sha256
            or identity.get("final_evaluation_commitment_sha256")
            != authorization.get("final_evaluation_commitment_sha256")
            or development.get("status") != "completed_development_validation"
            or int(development.get("case_count", -1)) != VALIDATION_COUNT
            or summary.get("architecture_selection_authorized") is not False
            or development.get("posthoc_calibration_applied") is not False
            or development.get("final_evaluation_accessed") is not False
        ):
            raise TrainingGateError("terminal reused probe violates architecture contract")
        summaries.append(summary)
    for key in (
        "membership_sha256",
        "input_standardizer_sha256",
        "target_standardizer_sha256",
    ):
        if len({summary["identity"][key] for summary in summaries}) != 1:
            raise TrainingGateError("terminal probe fits do not share one rung identity")
    if any(
        summary["identity"]["training_environment_sha256"]
        != authorization.get("reused_probe_training_environment_sha256")
        for summary in summaries
    ):
        raise TrainingGateError("terminal probe environment differs from authorization")
    return tuple(summaries)


def validate_terminal_architecture_execution_gate(
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
    terminal_decision_path: Path,
    probe_output_root: Path,
) -> Mapping[str, Any]:
    """Require one terminal lock and exact probe reuse before any grid fit."""

    authorization = load_yaml(authorization_path)
    if authorization.get("authorization_status") != (
        "authorized_terminal_131k_architecture_selection_only"
    ):
        raise TrainingGateError("terminal architecture execution authorization is absent")
    flags = authorization.get("authorization", {})
    for required in (
        "corrected_65k_data_access_authorized",
        "terminal_train_increment_data_access_authorized",
        "new_architecture_fit_execution_authorized",
        "architecture_selection_execution_authorized",
    ):
        if flags.get(required) is not True:
            raise TrainingGateError(f"terminal architecture gate requires {required}=true")
    for forbidden in (
        "probe_architecture_retraining_authorized",
        "best_seed_selection_authorized",
        "model_tuning_authorized",
        "calibration_authorized",
        "sbc_authorized",
        "final_evaluation_authorized",
        "extension_above_131072_authorized",
        "real_noise_authorized",
        "gwosc_gwtc_access_authorized",
    ):
        if flags.get(forbidden) is not False:
            raise TrainingGateError(f"terminal architecture gate requires {forbidden}=false")
    if authorization.get("locked_training_rung") != TRAIN_131K_COUNT:
        raise TrainingGateError("terminal architecture gate requires the 131k lock")
    if authorization.get("authorized_new_architecture_ids") != list(
        NEW_ARCHITECTURE_IDS
    ) or authorization.get("authorized_training_seeds") != list(SEEDS):
        raise TrainingGateError("terminal architecture grid or seeds changed")
    configured_decision = Path(str(authorization.get("terminal_decision_path", ""))).resolve()
    if terminal_decision_path.resolve() != configured_decision:
        raise TrainingGateError("terminal architecture decision path changed")
    decision = _load_mapping(terminal_decision_path)
    if (
        decision.get("decision")
        not in {
            "lock_train_131k_saturated",
            "lock_train_131k_resource_capped_data_limited",
        }
        or decision.get("comparison")
        != "corrected_train_65k_to_train_131k_terminal"
        or int(decision.get("selected_training_count", -1)) != TRAIN_131K_COUNT
        or decision.get("architecture_selection_review_allowed") is not True
        or decision.get("extension_above_131072_authorized") is not False
        or _sha256(terminal_decision_path)
        != authorization.get("terminal_decision_sha256")
    ):
        raise TrainingGateError("terminal architecture gate lacks an exact size lock")
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
    grid_hash = _sha256(root / GRID_PATH)
    if authorization.get("architecture_grid_sha256") != grid_hash:
        raise TrainingGateError("terminal architecture grid hash mismatch")
    base = load_yaml(root / "configs/models/phase4_probe_nsf.yaml")
    base_hash = model_configuration_hash(base)
    if authorization.get("reused_probe_model_configuration_hash") != base_hash:
        raise TrainingGateError("terminal reused probe model hash mismatch")
    configured_probe = Path(
        str(authorization.get("reused_probe_output_root", ""))
    ).resolve()
    if probe_output_root.resolve() != configured_probe:
        raise TrainingGateError("terminal reused probe root changed")
    reused = _validate_terminal_probe_reuse(
        probe_output_root=probe_output_root,
        authorization=authorization,
        train_manifest_sha256=publication.combined_manifest_sha256,
        validation_manifest_sha256=(
            publication.corrected_65k.stage_a.namespace_manifest_sha256["validation"]
        ),
    )
    expected_candidate_hashes = authorization.get("candidate_model_hashes", {})
    observed_candidate_hashes = {
        specification.architecture_id: model_configuration_hash(
            candidate_model_configuration(root, specification)
        )
        for specification in load_architecture_specs(root)
        if not specification.reused_probe
    }
    if expected_candidate_hashes != observed_candidate_hashes:
        raise TrainingGateError("terminal candidate model hashes changed")
    return {
        "authorization": authorization,
        "publication": publication,
        "reused_probe_summaries": reused,
        "architecture_grid_sha256": grid_hash,
    }


def run_authorized_terminal_architecture_fit(
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
    terminal_decision_path: Path,
    probe_output_root: Path,
    environment_lock_path: Path,
    psd_root: Path,
    output_root: Path,
    training_commit: str,
    architecture: str,
    seed: int,
    device_name: str,
    resume_checkpoint: Optional[Path] = None,
) -> Mapping[str, Any]:
    """Fit one of nine new terminal-rung results without retraining the probe."""

    specifications = {spec.architecture_id: spec for spec in load_architecture_specs(root)}
    if architecture not in NEW_ARCHITECTURE_IDS or architecture not in specifications:
        raise TrainingGateError("terminal fit is outside the frozen new architecture grid")
    if seed not in SEEDS:
        raise TrainingGateError("terminal architecture seed is outside 0/1/2")
    gate = validate_terminal_architecture_execution_gate(
        root,
        authorization_path=authorization_path,
        stage_a_publication_root=stage_a_publication_root,
        stage_b_publication_root=stage_b_publication_root,
        combined_base_publication_root=combined_base_publication_root,
        correction_publication_root=correction_publication_root,
        train_parent_root=train_parent_root,
        combined_131k_publication_root=combined_131k_publication_root,
        development_tail_parent_root=development_tail_parent_root,
        terminal_decision_path=terminal_decision_path,
        probe_output_root=probe_output_root,
    )
    authorization = gate["authorization"]
    artifacts = validate_immutable_training_artifacts(
        root,
        authorization.get("immutable_training", {}),
        training_commit=training_commit,
        environment_lock_path=environment_lock_path,
    )
    _verify_training_checkout(root, training_commit, authorization_path)
    configured_output = Path(str(authorization.get("architecture_output_root", ""))).resolve()
    if output_root.resolve() != configured_output or not output_root.is_relative_to(
        Path("/root/autodl-tmp/lensing-4")
    ):
        raise TrainingGateError("terminal architecture output root is not authorized")
    model = candidate_model_configuration(root, specifications[architecture])
    expected_model_hash = authorization["candidate_model_hashes"][architecture]
    if model_configuration_hash(model) != expected_model_hash:
        raise TrainingGateError("terminal architecture model hash mismatch")
    _validate_runtime_versions(model)
    load_input_policy(root)
    publication = gate["publication"]
    if not isinstance(publication, Terminal131TrainingPublication):
        raise TrainingGateError("terminal architecture gate returned wrong publication")
    curves = _verified_curves(model, psd_root)
    train_dataset = terminal_131k_training_dataset(publication, curves)
    validation_dataset = PublishedStageADataset(
        publication.corrected_65k.stage_a.validation_root,
        expected_split=SplitName.VALIDATION,
        detector_curves=curves,
        expected_total_pairs=VALIDATION_COUNT,
    )
    train_ids = train_dataset.physical_system_ids()
    preparation = _load_mapping(
        probe_output_root / "rung-131072" / "rung_preparation.json"
    )
    if not (
        tuple(preparation.get("member_ids", ())) == train_ids
        and preparation.get("combined_manifest_sha256")
        == publication.combined_manifest_sha256
    ):
        raise TrainingGateError("terminal architecture membership differs from probe")
    input_standardizer, target_standardizer = _load_standardizers(preparation)
    validation_hash = publication.corrected_65k.stage_a.namespace_manifest_sha256[
        "validation"
    ]
    identity = TrainingRunIdentity(
        model_configuration_hash=expected_model_hash,
        training_code_commit=training_commit,
        training_environment_sha256=artifacts["environment_lock_sha256"],
        train_manifest_sha256=publication.combined_manifest_sha256,
        validation_manifest_sha256=validation_hash,
        final_evaluation_commitment_sha256=str(
            authorization.get("final_evaluation_commitment_sha256", "")
        ),
        membership_sha256=membership_hash(train_ids),
        input_standardizer_sha256=standardizer_hash(input_standardizer),
        target_standardizer_sha256=standardizer_hash(target_standardizer),
        training_rung_count=TRAIN_131K_COUNT,
        seed=seed,
    )
    identity.validate()
    run_directory = output_root / architecture / f"seed-{seed}"
    if run_directory.exists() and resume_checkpoint is None:
        raise FileExistsError("terminal architecture result identity already exists")
    from .architecture import architecture_execution_evidence

    evidence = architecture_execution_evidence(
        identity,
        architecture=architecture,
        immutable_wheel_sha256=artifacts["wheel_sha256"],
    )
    _atomic_json(run_directory / "run_preparation.json", evidence)
    workers = int(authorization.get("data_loader_worker_processes", 0))
    if not 0 <= workers <= 16:
        raise TrainingGateError("terminal architecture loader count is outside review")
    standardized_train = StandardizedStageADataset(train_dataset, input_standardizer)
    standardized_validation = StandardizedStageADataset(
        validation_dataset, input_standardizer
    )
    _, physical_batch, _ = optimization_batch_geometry(model["optimization"])
    train_loader = _data_loader(
        standardized_train,
        batch_size=physical_batch,
        seed=seed,
        training=True,
        worker_processes=workers,
        device_name=device_name,
    )
    validation_loader = _data_loader(
        standardized_validation,
        batch_size=physical_batch,
        seed=seed,
        training=False,
        worker_processes=workers,
        device_name=device_name,
    )
    model_instance = build_probe_model(model, seed=seed)
    parameter_count = sum(int(parameter.numel()) for parameter in model_instance.parameters())
    training = train_probe(
        model_instance,
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
    seed_payload = (
        f"{evaluation['posterior_draw_seed_domain']}\0{identity.membership_sha256}\0"
        f"{architecture}\0{seed}"
    ).encode()
    evaluation_seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
    development = evaluate_development_validation(
        model_instance,
        _development_loader(
            DevelopmentStageADataset(validation_dataset, input_standardizer),
            batch_size=int(evaluation["batch_size"]),
            seed=seed,
            worker_processes=workers,
            device_name=device_name,
        ),
        standardizer=target_standardizer,
        device_name=device_name,
        posterior_draws_per_case=int(evaluation["posterior_draws_per_case"]),
        evaluation_seed=evaluation_seed,
        output_directory=run_directory,
        levels=tuple(float(value) for value in evaluation["coverage_levels"]),
    )
    result = {
        "status": "completed_terminal_architecture_fit_and_development_validation",
        "architecture_id": architecture,
        "transforms": specifications[architecture].transforms,
        "conditioner_width": specifications[architecture].conditioner_width,
        "seed": seed,
        "trainable_parameter_count": parameter_count,
        "identity": asdict(identity),
        "training": training,
        "development": development,
        "calibration_accessed": False,
        "final_evaluation_accessed": False,
        "extension_above_131072_authorized": False,
    }
    _atomic_json(run_directory / "run_summary.json", result)
    return result


def collect_terminal_architecture_results(
    root: Path,
    *,
    gate: Mapping[str, Any],
    architecture_output_root: Path,
) -> Tuple[Mapping[str, Any], ...]:
    """Collect exactly three terminal probe reuses plus nine new results."""

    authorization = gate["authorization"]
    publication = gate["publication"]
    if not isinstance(publication, Terminal131TrainingPublication):
        raise TrainingGateError("terminal result collector received wrong publication")
    reused_identity = gate["reused_probe_summaries"][0]["identity"]
    probe_model = build_probe_model(
        load_yaml(root / "configs/models/phase4_probe_nsf.yaml"), seed=0
    )
    probe_parameters = sum(int(parameter.numel()) for parameter in probe_model.parameters())
    rows = []
    for summary in gate["reused_probe_summaries"]:
        rows.append(
            {
                "architecture_id": PROBE_ARCHITECTURE_ID,
                "seed": int(summary["identity"]["seed"]),
                "trainable_parameter_count": probe_parameters,
                "development_mean_nlp_nat_per_target_dimension": float(
                    summary["development"]["mean_nlp_nat_per_target_dimension"]
                ),
                "reused_probe_fit": True,
            }
        )
    for architecture in NEW_ARCHITECTURE_IDS:
        expected_hash = authorization["candidate_model_hashes"][architecture]
        for seed in SEEDS:
            path = architecture_output_root / architecture / f"seed-{seed}" / "run_summary.json"
            if not path.is_file():
                raise TrainingGateError("terminal architecture result summary is absent")
            summary = _load_mapping(path)
            identity = summary.get("identity", {})
            development = summary.get("development", {})
            if (
                summary.get("status")
                != "completed_terminal_architecture_fit_and_development_validation"
                or summary.get("architecture_id") != architecture
                or int(summary.get("seed", -1)) != seed
                or identity.get("model_configuration_hash") != expected_hash
                or int(identity.get("training_rung_count", -1)) != TRAIN_131K_COUNT
                or identity.get("train_manifest_sha256")
                != publication.combined_manifest_sha256
                or identity.get("validation_manifest_sha256")
                != publication.corrected_65k.stage_a.namespace_manifest_sha256[
                    "validation"
                ]
                or identity.get("membership_sha256")
                != reused_identity.get("membership_sha256")
                or identity.get("input_standardizer_sha256")
                != reused_identity.get("input_standardizer_sha256")
                or identity.get("target_standardizer_sha256")
                != reused_identity.get("target_standardizer_sha256")
                or identity.get("final_evaluation_commitment_sha256")
                != authorization.get("final_evaluation_commitment_sha256")
                or development.get("status") != "completed_development_validation"
                or int(development.get("case_count", -1)) != VALIDATION_COUNT
                or summary.get("calibration_accessed") is not False
                or summary.get("final_evaluation_accessed") is not False
                or not (path.parent / "best.ckpt").is_file()
            ):
                raise TrainingGateError("terminal architecture result violates grid")
            rows.append(
                {
                    "architecture_id": architecture,
                    "seed": seed,
                    "trainable_parameter_count": int(
                        summary.get("trainable_parameter_count", -1)
                    ),
                    "development_mean_nlp_nat_per_target_dimension": float(
                        development["mean_nlp_nat_per_target_dimension"]
                    ),
                    "reused_probe_fit": False,
                }
            )
    return tuple(rows)
