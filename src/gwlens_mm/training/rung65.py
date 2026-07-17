"""Fail-closed execution path for the terminal 65k learning-curve rung."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from ..config import load_yaml
from ..schema import SplitName
from .contracts import (
    TrainingGateError,
    load_training_stack_contract,
    model_configuration_hash,
)
from .data import (
    CombinedTrainingPublication,
    ConcatenatedPublishedStageADataset,
    DevelopmentStageADataset,
    PublishedStageADataset,
    StandardizedStageADataset,
    resolve_combined_training_publication,
)
from .engine import (
    TargetStandardizer,
    TrainingRunIdentity,
    evaluate_development_validation,
    membership_hash,
    optimization_batch_geometry,
    standardizer_hash,
    train_probe,
)
from .features import InputStandardizer, load_input_policy
from .model import build_probe_model
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

TRAIN_65K_COUNT = 65536
VALIDATION_COUNT = 6144
SEEDS = (0, 1, 2)


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TrainingGateError(f"expected a JSON mapping: {path}")
    return value


def validate_immutable_training_artifacts(
    root: Path,
    immutable: Mapping[str, Any],
    *,
    training_commit: str,
    environment_lock_path: Path,
) -> Mapping[str, str]:
    """Bind a future 65k execution to its reviewed wheel and environment lock."""

    if immutable.get("git_commit") != training_commit:
        raise TrainingGateError("65k authorization training commit mismatch")
    expected_environment_hash = str(immutable.get("environment_lock_sha256", ""))
    configured_lock = Path(str(immutable.get("environment_lock_path", "")))
    expected_lock_path = (
        configured_lock if configured_lock.is_absolute() else root / configured_lock
    ).resolve()
    if environment_lock_path.resolve() != expected_lock_path:
        raise TrainingGateError("65k training environment lock path mismatch")
    if not environment_lock_path.is_file():
        raise TrainingGateError("65k training environment lock is absent")
    if _sha256_file(environment_lock_path) != expected_environment_hash:
        raise TrainingGateError("65k training environment lock hash mismatch")
    wheel_path = Path(str(immutable.get("wheel_path", ""))).resolve()
    wheel_filename = str(immutable.get("wheel_filename", ""))
    if not wheel_path.is_file() or wheel_path.name != wheel_filename:
        raise TrainingGateError("65k immutable training wheel is absent or misnamed")
    wheel_hash = str(immutable.get("wheel_sha256", ""))
    if _sha256_file(wheel_path) != wheel_hash:
        raise TrainingGateError("65k immutable training wheel hash mismatch")
    if immutable.get("editable_install_authorized") is not False:
        raise TrainingGateError("65k gate must forbid editable installation")
    return {
        "environment_lock_sha256": expected_environment_hash,
        "wheel_path": str(wheel_path),
        "wheel_sha256": wheel_hash,
    }


def validate_65k_training_gate(
    root: Path,
    *,
    authorization_path: Path,
    stage_a_publication_root: Path,
    stage_b_publication_root: Path,
    combined_publication_root: Path,
) -> Mapping[str, Any]:
    """Require the exact post-publication gate before indexing either component."""

    load_training_stack_contract(root)
    authorization = load_yaml(authorization_path)
    if authorization.get("authorization_status") != "authorized_train_65k_probe_only":
        raise TrainingGateError("65k probe-training authorization is absent")
    flags = authorization.get("authorization", {})
    for required in (
        "stage_a_data_access_authorized",
        "stage_b_data_access_authorized",
        "scientific_65k_probe_training_authorized",
        "probe_optimizer_execution_authorized",
        "learning_curve_decision_authorized",
    ):
        if flags.get(required) is not True:
            raise TrainingGateError(f"65k gate requires {required}=true")
    for forbidden in (
        "model_tuning_authorized",
        "architecture_selection_authorized",
        "calibration_authorized",
        "sbc_authorized",
        "final_evaluation_authorized",
        "extension_above_65536_authorized",
        "gwosc_gwtc_access_authorized",
    ):
        if flags.get(forbidden) is not False:
            raise TrainingGateError(f"65k gate must keep {forbidden}=false")
    if authorization.get("authorized_training_rungs") != [65536]:
        raise TrainingGateError("65k gate must authorize only the terminal rung")
    if authorization.get("authorized_training_seeds") != [0, 1, 2]:
        raise TrainingGateError("65k gate must authorize exactly seeds 0, 1 and 2")
    decision_path = root / "results/phase4/probe/learning_curve_decision.json"
    decision = _load_json(decision_path)
    if decision.get("decision") != "continue_to_train_65k":
        raise TrainingGateError("65k gate lacks the frozen 32k continuation decision")
    if authorization.get("prior_learning_curve_decision_sha256") != hashlib.sha256(
        decision_path.read_bytes()
    ).hexdigest():
        raise TrainingGateError("65k gate learning-curve evidence hash mismatch")
    commitment_path = root / "results/phase4/final_evaluation_commitment.json"
    commitment = _load_json(commitment_path)
    if commitment.get("commitment_status") != "finalized_before_training":
        raise TrainingGateError("final-evaluation commitment is not finalized")
    commitment_hash = hashlib.sha256(commitment_path.read_bytes()).hexdigest()
    if authorization.get("final_evaluation_commitment_sha256") != commitment_hash:
        raise TrainingGateError("65k gate final-evaluation commitment hash mismatch")
    combined = authorization.get("combined_train", {})
    publication = resolve_combined_training_publication(
        combined_publication_root,
        stage_a_parent_root=stage_a_publication_root,
        stage_b_parent_root=stage_b_publication_root,
        expected_generator_commit=str(authorization.get("generator_commit", "")),
        expected_preregistration_hash=str(authorization.get("preregistration_hash", "")),
        expected_combined_manifest_sha256=str(
            combined.get("combined_manifest_sha256", "")
        ),
    )
    if combined.get("combined_train_id") != publication.combined_root.name:
        raise TrainingGateError("65k gate combined dataset identity mismatch")
    if combined.get("stage_a_parent_manifest_sha256") != publication.stage_a.manifest_sha256:
        raise TrainingGateError("65k gate Stage A manifest hash mismatch")
    if (
        combined.get("stage_b_parent_manifest_sha256")
        != publication.stage_b_parent_manifest_sha256
    ):
        raise TrainingGateError("65k gate Stage B manifest hash mismatch")
    return {
        "authorization": authorization,
        "publication": publication,
        "final_evaluation_commitment_sha256": commitment_hash,
    }


def _load_standardizers(value: Mapping[str, Any]) -> Tuple[InputStandardizer, TargetStandardizer]:
    input_value = value["input_standardizer"]
    target_value = value["target_standardizer"]
    astrometry_mean = [float(item) for item in input_value["astrometry_mean"]]
    astrometry_scale = [
        float(item) for item in input_value["astrometry_standard_deviation"]
    ]
    target_mean = [float(item) for item in target_value["mean"]]
    target_scale = [float(item) for item in target_value["standard_deviation"]]
    if not (
        len(astrometry_mean) == len(astrometry_scale) == 5
        and len(target_mean) == len(target_scale) == 2
    ):
        raise TrainingGateError("65k rung standardizer dimensions are invalid")
    input_standardizer = InputStandardizer(
        tuple(float(item) for item in input_value["scalar_mean"]),
        tuple(float(item) for item in input_value["scalar_standard_deviation"]),
        (
            astrometry_mean[0],
            astrometry_mean[1],
            astrometry_mean[2],
            astrometry_mean[3],
            astrometry_mean[4],
        ),
        (
            astrometry_scale[0],
            astrometry_scale[1],
            astrometry_scale[2],
            astrometry_scale[3],
            astrometry_scale[4],
        ),
    )
    target_standardizer = TargetStandardizer(
        (target_mean[0], target_mean[1]),
        (target_scale[0], target_scale[1]),
    )
    if not (
        value.get("input_standardizer_sha256")
        == standardizer_hash(input_standardizer)
        and value.get("target_standardizer_sha256")
        == standardizer_hash(target_standardizer)
    ):
        raise TrainingGateError("65k rung standardizer hash mismatch")
    return input_standardizer, target_standardizer


def _training_dataset(
    publication: CombinedTrainingPublication,
    curves: Mapping[str, Any],
) -> ConcatenatedPublishedStageADataset:
    stage_a = PublishedStageADataset(
        publication.stage_a.train_root,
        expected_split=SplitName.TRAIN,
        detector_curves=curves,
        expected_total_pairs=32768,
    )
    # The specialized resolver already validates the Stage B atomic parent and
    # combined reference. The generic Stage A reader cannot interpret its
    # single-namespace parent shape, so only that redundant parent lookup is disabled.
    stage_b = PublishedStageADataset(
        publication.stage_b_train_root,
        expected_split=SplitName.TRAIN,
        detector_curves=curves,
        expected_total_pairs=32768,
        require_published=False,
    )
    combined = ConcatenatedPublishedStageADataset((stage_a, stage_b))
    if len(combined) != TRAIN_65K_COUNT:
        raise TrainingGateError("combined training reader does not contain exactly 65k")
    return combined


def run_authorized_65k_probe(
    root: Path,
    *,
    authorization_path: Path,
    stage_a_publication_root: Path,
    stage_b_publication_root: Path,
    combined_publication_root: Path,
    environment_lock_path: Path,
    psd_root: Path,
    output_root: Path,
    training_commit: str,
    seed: int,
    device_name: str,
    resume_checkpoint: Optional[Path] = None,
    execute_optimizer: bool = True,
) -> Mapping[str, Any]:
    """Prepare or fit one frozen 65k probe seed from scratch."""

    if seed not in SEEDS:
        raise TrainingGateError("65k probe seed is outside 0/1/2")
    gate = validate_65k_training_gate(
        root,
        authorization_path=authorization_path,
        stage_a_publication_root=stage_a_publication_root,
        stage_b_publication_root=stage_b_publication_root,
        combined_publication_root=combined_publication_root,
    )
    authorization = gate["authorization"]
    immutable = authorization.get("immutable_training", {})
    artifacts = validate_immutable_training_artifacts(
        root,
        immutable,
        training_commit=training_commit,
        environment_lock_path=environment_lock_path,
    )
    expected_environment_hash = artifacts["environment_lock_sha256"]
    _verify_training_checkout(root, training_commit, authorization_path)
    configured_output = Path(str(authorization.get("training_output_root", ""))).resolve()
    if configured_output != output_root.resolve():
        raise TrainingGateError("65k output differs from the authorized root")
    if not output_root.is_absolute() or not output_root.is_relative_to(
        Path("/root/autodl-tmp/lensing-4")
    ):
        raise TrainingGateError("65k output escaped the AutoDL project root")
    if int(authorization.get("maximum_concurrent_fits", -1)) != 3:
        raise TrainingGateError("65k authorization must permit exactly three fits")
    model = load_yaml(root / "configs/models/phase4_probe_nsf.yaml")
    if immutable.get("model_configuration_hash") != model_configuration_hash(model):
        raise TrainingGateError("65k authorized model configuration hash mismatch")
    _validate_runtime_versions(model)
    load_input_policy(root)
    publication = gate["publication"]
    curves = _verified_curves(model, psd_root)
    train_dataset = _training_dataset(publication, curves)
    validation_dataset = PublishedStageADataset(
        publication.stage_a.validation_root,
        expected_split=SplitName.VALIDATION,
        detector_curves=curves,
        expected_total_pairs=VALIDATION_COUNT,
    )
    train_ids = train_dataset.physical_system_ids()
    validation_ids = validation_dataset.physical_system_ids()
    if len(train_ids) != TRAIN_65K_COUNT or set(train_ids) & set(validation_ids):
        raise TrainingGateError("65k train/validation membership is invalid")
    rung_directory = output_root / "rung-65536"
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
        ):
            raise TrainingGateError("65k rung preparation identity mismatch")
        input_standardizer, target_standardizer = _load_standardizers(preparation)
    else:
        if execute_optimizer:
            raise TrainingGateError("65k shared preprocessing must complete before fits")
        input_standardizer, target_standardizer = fit_rung_standardizers(train_dataset)
        preparation = {
            "status": "ready_for_authorized_probe_fits",
            "rung_count": TRAIN_65K_COUNT,
            "member_count": len(train_ids),
            "member_ids": list(train_ids),
            "member_ids_sha256": membership_hash(train_ids),
            "combined_manifest_sha256": publication.combined_manifest_sha256,
            "stage_a_parent_manifest_sha256": publication.stage_a.manifest_sha256,
            "stage_b_parent_manifest_sha256": publication.stage_b_parent_manifest_sha256,
            "train_manifest_sha256": publication.train_manifest_sha256,
            "validation_manifest_sha256": publication.stage_a.namespace_manifest_sha256[
                "validation"
            ],
            "final_evaluation_commitment_sha256": gate[
                "final_evaluation_commitment_sha256"
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
        training_environment_sha256=expected_environment_hash,
        train_manifest_sha256=publication.train_manifest_sha256,
        validation_manifest_sha256=publication.stage_a.namespace_manifest_sha256[
            "validation"
        ],
        final_evaluation_commitment_sha256=str(
            gate["final_evaluation_commitment_sha256"]
        ),
        membership_sha256=membership_hash(train_ids),
        input_standardizer_sha256=standardizer_hash(input_standardizer),
        target_standardizer_sha256=standardizer_hash(target_standardizer),
        training_rung_count=TRAIN_65K_COUNT,
        seed=seed,
    )
    identity.validate()
    run_directory = rung_directory / f"seed-{seed}"
    if run_directory.exists() and resume_checkpoint is None:
        raise FileExistsError("65k run identity already has an output directory")
    execution_evidence = {
        "status": "authorized_65k_probe_training",
        "run_identity": asdict(identity),
        "combined_publication_validated": True,
        "combined_manifest_sha256": publication.combined_manifest_sha256,
        "immutable_wheel_sha256": artifacts["wheel_sha256"],
        "member_count": len(train_ids),
        "member_ids_sha256": membership_hash(train_ids),
        "member_ids": list(train_ids),
        "input_standardizer": asdict(input_standardizer),
        "target_standardizer": asdict(target_standardizer),
        "architecture_selection_authorized": False,
        "calibration_authorized": False,
        "final_evaluation_accessed": False,
    }
    _atomic_json(run_directory / "run_preparation.json", execution_evidence)
    workers = int(authorization.get("data_loader_worker_processes", 0))
    if not 0 <= workers <= 16:
        raise TrainingGateError("65k data-loader worker count is outside review")
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
        execution_evidence=execution_evidence,
        output_directory=run_directory,
        device_name=device_name,
        resume_checkpoint=resume_checkpoint,
    )
    evaluation = model["development_evaluation"]
    seed_payload = (
        f"{evaluation['posterior_draw_seed_domain']}\0{identity.membership_sha256}\0{seed}"
    ).encode()
    evaluation_seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
    development_loader = _development_loader(
        DevelopmentStageADataset(validation_dataset, input_standardizer),
        batch_size=int(evaluation["batch_size"]),
        seed=seed,
        worker_processes=workers,
        device_name=device_name,
    )
    development_summary = evaluate_development_validation(
        probe_model,
        development_loader,
        standardizer=target_standardizer,
        device_name=device_name,
        posterior_draws_per_case=int(evaluation["posterior_draws_per_case"]),
        evaluation_seed=evaluation_seed,
        output_directory=run_directory,
        levels=tuple(float(value) for value in evaluation["coverage_levels"]),
    )
    result = {
        "status": "completed_65k_probe_fit_and_development_validation",
        "identity": asdict(identity),
        "training": training_summary,
        "development": development_summary,
        "architecture_selection_authorized": False,
        "calibration_accessed": False,
        "final_evaluation_accessed": False,
        "extension_above_65536_authorized": False,
    }
    _atomic_json(run_directory / "run_summary.json", result)
    return result
