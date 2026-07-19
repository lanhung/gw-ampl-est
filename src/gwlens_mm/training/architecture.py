"""Fail-closed implementation for post-size-lock architecture selection."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from ..config import load_yaml
from ..schema import SplitName
from .contracts import (
    WAVEFORM_CORRECTION_HASH,
    TrainingGateError,
    model_configuration_hash,
)
from .data import (
    CombinedTrainingPublication,
    ConcatenatedPublishedStageADataset,
    CorrectedTrainingPublication,
    DevelopmentStageADataset,
    PublishedStageADataset,
    StandardizedStageADataset,
    corrected_65k_training_dataset,
    resolve_combined_training_publication,
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
from .features import load_input_policy
from .model import build_probe_model
from .rung65 import (
    SEEDS,
    TRAIN_65K_COUNT,
    VALIDATION_COUNT,
    _load_standardizers,
    _training_dataset,
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
from .whitening import ASDCurve

GRID_PATH = "configs/models/phase5_architecture_grid.yaml"
BASE_MODEL_PATH = "configs/models/phase4_probe_nsf.yaml"
PROBE_ARCHITECTURE_ID = "nsf-t10-w256"
NEW_ARCHITECTURE_IDS = ("nsf-t06-w128", "nsf-t06-w256", "nsf-t10-w128")
ALL_ARCHITECTURE_IDS = NEW_ARCHITECTURE_IDS + (PROBE_ARCHITECTURE_ID,)

LockedTrainingPublication = Union[
    CombinedTrainingPublication, CorrectedTrainingPublication
]


@dataclass(frozen=True)
class ArchitectureSpec:
    architecture_id: str
    transforms: int
    conditioner_width: int
    reused_probe: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def architecture_id(transforms: int, width: int) -> str:
    return f"nsf-t{transforms:02d}-w{width}"


def architecture_execution_evidence(
    identity: TrainingRunIdentity,
    *,
    architecture: str,
    immutable_wheel_sha256: str,
) -> Mapping[str, Any]:
    """Add architecture identity to the common pre-optimizer envelope."""

    if architecture not in NEW_ARCHITECTURE_IDS:
        raise TrainingGateError("execution evidence requires a new grid architecture")
    if len(immutable_wheel_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in immutable_wheel_sha256
    ):
        raise TrainingGateError("architecture wheel identity must be SHA-256")
    return {
        **authorized_probe_execution_evidence(identity),
        "fit_role": "authorized_architecture_fit",
        "architecture_id": architecture,
        "reused_probe_retraining": False,
        "immutable_wheel_sha256": immutable_wheel_sha256,
        "calibration_authorized": False,
        "final_evaluation_accessed": False,
    }


def load_architecture_specs(root: Path) -> Tuple[ArchitectureSpec, ...]:
    """Load the exact preregistered 2x2 grid without accepting extra candidates."""

    grid = load_yaml(root / GRID_PATH)
    parent = grid.get("parent_preregistration", {})
    if (
        parent.get("version") != "1.1.0-rc.4"
        or parent.get("path")
        != "configs/statistics/direct_target_stage_a_preregistration.yaml"
        or parent.get("canonical_hash")
        != "5aeaac395463bd073c44ead4ff4c5c729b5a2d4b4f1840c0825a53b30ab1bc98"
        or grid.get("base_model_configuration") != BASE_MODEL_PATH
    ):
        raise TrainingGateError("architecture grid parent scientific identity changed")
    configured = grid.get("architecture_grid", {})
    transforms = tuple(int(value) for value in configured.get("flow_transforms", ()))
    widths = tuple(int(value) for value in configured.get("conditioner_widths", ()))
    seeds = tuple(int(value) for value in configured.get("seeds", ()))
    identifiers = tuple(str(value) for value in configured.get("architecture_ids", ()))
    expected = tuple(architecture_id(t, w) for t in (6, 10) for w in (128, 256))
    if transforms != (6, 10) or widths != (128, 256) or seeds != SEEDS:
        raise TrainingGateError("architecture grid differs from the frozen 2x2x3 design")
    if identifiers != expected or set(identifiers) != set(ALL_ARCHITECTURE_IDS):
        raise TrainingGateError("architecture identifiers differ from the frozen grid")
    fit = grid.get("fit_contract", {})
    if not (
        fit.get("reuse_probe_architecture_id") == PROBE_ARCHITECTURE_ID
        and fit.get("reuse_probe_fits_at_locked_rung") is True
        and fit.get("retrain_reused_probe_without_declared_failure") is False
        and int(fit.get("maximum_architecture_results", -1)) == 12
        and int(fit.get("maximum_new_architecture_fits_after_lock", -1)) == 9
    ):
        raise TrainingGateError("architecture fit/reuse contract is inconsistent")
    selection = grid.get("selection_contract", {})
    if not (
        selection.get("primary_metric")
        == "mean_validation_nlp_nat_per_target_dimension_across_three_seeds"
        and selection.get("select_best_seed") is False
        and selection.get("tie_break") == "lower_trainable_parameter_count"
        and selection.get("final_evaluation_access") == "forbidden"
        and selection.get("calibration_fit_access") == "forbidden"
        and selection.get("sbc_access") == "forbidden"
    ):
        raise TrainingGateError("architecture selection contract is inconsistent")
    execution = grid.get("execution", {})
    if execution != {
        "architecture_fit_execution_enabled": False,
        "architecture_selection_execution_enabled": False,
        "scientific_data_access_enabled": False,
        "engineering_fixture_tests_only": True,
    }:
        raise TrainingGateError("architecture grid must remain execution-disabled")
    return tuple(
        ArchitectureSpec(
            architecture_id=architecture_id(t, w),
            transforms=t,
            conditioner_width=w,
            reused_probe=(t, w) == (10, 256),
        )
        for t in transforms
        for w in widths
    )


def candidate_model_configuration(
    root: Path, specification: ArchitectureSpec
) -> Mapping[str, Any]:
    """Derive one full model config by changing only the two frozen grid axes."""

    if specification not in load_architecture_specs(root):
        raise TrainingGateError("requested architecture is outside the frozen grid")
    model = copy.deepcopy(load_yaml(root / BASE_MODEL_PATH))
    model["implementation_id"] = f"phase5_{specification.architecture_id}_v1"
    model["architecture"]["flow"]["transforms"] = specification.transforms
    model["architecture"]["flow"][
        "conditioner_width"
    ] = specification.conditioner_width
    model["execution"]["scientific_training_enabled"] = False
    model["execution"]["model_selection_enabled"] = False
    model["execution"]["engineering_smoke_only"] = True
    return model


def selected_model_configuration(root: Path, architecture: str) -> Mapping[str, Any]:
    """Return the exact fitted configuration, preserving the reused probe identity."""

    specifications = {item.architecture_id: item for item in load_architecture_specs(root)}
    if architecture not in specifications:
        raise TrainingGateError("selected architecture is outside the frozen grid")
    specification = specifications[architecture]
    if specification.reused_probe:
        return load_yaml(root / BASE_MODEL_PATH)
    return candidate_model_configuration(root, specification)


def _load_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TrainingGateError(f"expected JSON mapping: {path}")
    return value


def _locked_train_manifest_sha256(
    publication: LockedTrainingPublication,
) -> str:
    if isinstance(publication, CorrectedTrainingPublication):
        return publication.corrected_combined_train_manifest_sha256
    return publication.train_manifest_sha256


def _locked_preparation_manifest_sha256(
    publication: LockedTrainingPublication,
) -> str:
    if isinstance(publication, CorrectedTrainingPublication):
        return publication.correction_manifest_sha256
    return publication.combined_manifest_sha256


def _locked_training_dataset(
    publication: LockedTrainingPublication,
    detector_curves: Mapping[str, ASDCurve],
) -> ConcatenatedPublishedStageADataset:
    """Use the immutable correction overlay whenever the size-lock probe did."""

    if isinstance(publication, CorrectedTrainingPublication):
        return corrected_65k_training_dataset(publication, detector_curves)
    return _training_dataset(publication, detector_curves)


def _validate_probe_reuse(
    *,
    probe_output_root: Path,
    authorization: Mapping[str, Any],
    train_manifest_sha256: str,
    validation_manifest_sha256: str,
) -> Tuple[Mapping[str, Any], ...]:
    expected_hashes = authorization.get("reused_probe_run_summary_sha256", {})
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != {
        str(seed) for seed in SEEDS
    }:
        raise TrainingGateError("architecture gate must bind all three reused probe fits")
    expected_model_hash = str(
        authorization.get("reused_probe_model_configuration_hash", "")
    )
    summaries = []
    for seed in SEEDS:
        path = probe_output_root / "rung-65536" / f"seed-{seed}" / "run_summary.json"
        if not path.is_file() or _sha256(path) != expected_hashes[str(seed)]:
            raise TrainingGateError("reused probe run summary hash mismatch")
        summary = _load_mapping(path)
        identity = summary.get("identity", {})
        development = summary.get("development", {})
        if (
            summary.get("status")
            != "completed_65k_probe_fit_and_development_validation"
            or int(identity.get("training_rung_count", -1)) != TRAIN_65K_COUNT
            or int(identity.get("seed", -1)) != seed
            or identity.get("model_configuration_hash") != expected_model_hash
            or identity.get("train_manifest_sha256") != train_manifest_sha256
            or identity.get("validation_manifest_sha256") != validation_manifest_sha256
            or identity.get("final_evaluation_commitment_sha256")
            != authorization.get("final_evaluation_commitment_sha256")
            or development.get("status") != "completed_development_validation"
            or int(development.get("case_count", -1)) != VALIDATION_COUNT
            or development.get("posthoc_calibration_applied") is not False
            or development.get("final_evaluation_accessed") is not False
        ):
            raise TrainingGateError("reused probe fit violates the architecture contract")
        if not (path.parent / "best.ckpt").is_file():
            raise TrainingGateError("reused probe fit lacks its immutable best checkpoint")
        summaries.append(summary)
    for key in (
        "membership_sha256",
        "input_standardizer_sha256",
        "target_standardizer_sha256",
    ):
        if len({summary["identity"][key] for summary in summaries}) != 1:
            raise TrainingGateError("reused probe fits do not share one locked-rung identity")
    if any(
        summary["identity"]["training_environment_sha256"]
        != authorization.get("reused_probe_training_environment_sha256")
        for summary in summaries
    ):
        raise TrainingGateError("reused probe environment differs from authorization")
    return tuple(summaries)


def validate_architecture_execution_gate(
    root: Path,
    *,
    authorization_path: Path,
    stage_a_publication_root: Path,
    stage_b_publication_root: Path,
    combined_publication_root: Path,
    correction_publication_root: Optional[Path] = None,
    terminal_decision_path: Path,
    probe_output_root: Path,
) -> Mapping[str, Any]:
    """Require a size lock and exact reuse evidence before any new fit."""

    authorization = load_yaml(authorization_path)
    authorization_status = authorization.get("authorization_status")
    corrected = authorization_status == "authorized_corrected_architecture_selection_only"
    if authorization_status not in {
        "authorized_architecture_selection_only",
        "authorized_corrected_architecture_selection_only",
    }:
        raise TrainingGateError("architecture-selection execution authorization is absent")
    flags = authorization.get("authorization", {})
    for required in (
        "stage_a_data_access_authorized",
        "stage_b_data_access_authorized",
        "new_architecture_fit_execution_authorized",
        "architecture_selection_execution_authorized",
    ):
        if flags.get(required) is not True:
            raise TrainingGateError(f"architecture gate requires {required}=true")
    if corrected and flags.get("replacement_data_access_authorized") is not True:
        raise TrainingGateError(
            "corrected architecture gate requires replacement_data_access_authorized=true"
        )
    for forbidden in (
        "probe_architecture_retraining_authorized",
        "best_seed_selection_authorized",
        "model_tuning_authorized",
        "calibration_authorized",
        "sbc_authorized",
        "final_evaluation_authorized",
        "extension_above_65536_authorized",
        "gwosc_gwtc_access_authorized",
    ):
        if flags.get(forbidden) is not False:
            raise TrainingGateError(f"architecture gate must keep {forbidden}=false")
    if authorization.get("locked_training_rung") != TRAIN_65K_COUNT:
        raise TrainingGateError("architecture selection requires the locked 65k rung")
    if authorization.get("authorized_new_architecture_ids") != list(
        NEW_ARCHITECTURE_IDS
    ):
        raise TrainingGateError("architecture gate must authorize exactly three new models")
    if authorization.get("authorized_training_seeds") != list(SEEDS):
        raise TrainingGateError("architecture gate must authorize exactly seeds 0/1/2")
    configured_decision = Path(str(authorization.get("terminal_decision_path", ""))).resolve()
    if terminal_decision_path.resolve() != configured_decision:
        raise TrainingGateError("terminal decision path differs from authorization")
    decision = _load_mapping(terminal_decision_path)
    if (
        decision.get("decision") != "lock_train_65k"
        or decision.get("comparison") != "train_32k_to_train_65k"
        or decision.get("extension_above_65536_authorized") is not False
        or _sha256(terminal_decision_path)
        != authorization.get("terminal_decision_sha256")
    ):
        raise TrainingGateError("architecture gate lacks the exact terminal size lock")
    if corrected:
        if correction_publication_root is None:
            raise TrainingGateError(
                "corrected architecture gate requires the correction publication"
            )
        if authorization.get("preregistration_hash") != WAVEFORM_CORRECTION_HASH:
            raise TrainingGateError(
                "corrected architecture gate changed the numerical preregistration"
            )
        correction = authorization.get("correction_publication", {})
        corrected_publication = resolve_corrected_training_publication(
            correction_publication_root,
            stage_a_parent_root=stage_a_publication_root,
            stage_b_parent_root=stage_b_publication_root,
            combined_base_root=combined_publication_root,
            expected_base_generator_commit=str(
                authorization.get("base_generator_commit", "")
            ),
            expected_base_preregistration_hash=str(
                authorization.get("base_preregistration_hash", "")
            ),
            expected_correction_generator_commit=str(
                correction.get("generator_commit", "")
            ),
            expected_correction_preregistration_hash=WAVEFORM_CORRECTION_HASH,
            expected_correction_manifest_sha256=str(
                correction.get("parent_manifest_sha256", "")
            ),
            expected_correction_tree_sha256=str(
                correction.get("publication_tree_sha256", "")
            ),
            expected_combined_base_manifest_sha256=str(
                authorization.get("combined_base_manifest_sha256", "")
            ),
        )
        if authorization.get("corrected_combined_train_manifest_sha256") != (
            corrected_publication.corrected_combined_train_manifest_sha256
        ):
            raise TrainingGateError("corrected architecture training-view hash mismatch")
        publication: LockedTrainingPublication = corrected_publication
    else:
        combined = authorization.get("combined_train", {})
        publication = resolve_combined_training_publication(
            combined_publication_root,
            stage_a_parent_root=stage_a_publication_root,
            stage_b_parent_root=stage_b_publication_root,
            expected_generator_commit=str(authorization.get("generator_commit", "")),
            expected_preregistration_hash=str(
                authorization.get("preregistration_hash", "")
            ),
            expected_combined_manifest_sha256=str(
                combined.get("combined_manifest_sha256", "")
            ),
        )
    grid_hash = _sha256(root / GRID_PATH)
    if authorization.get("architecture_grid_sha256") != grid_hash:
        raise TrainingGateError("architecture grid hash mismatch")
    base_hash = model_configuration_hash(load_yaml(root / BASE_MODEL_PATH))
    if authorization.get("reused_probe_model_configuration_hash") != base_hash:
        raise TrainingGateError("reused probe model hash mismatch")
    probe_root = Path(str(authorization.get("reused_probe_output_root", ""))).resolve()
    if probe_output_root.resolve() != probe_root:
        raise TrainingGateError("reused probe root differs from authorization")
    reused = _validate_probe_reuse(
        probe_output_root=probe_output_root,
        authorization=authorization,
        train_manifest_sha256=_locked_train_manifest_sha256(publication),
        validation_manifest_sha256=publication.stage_a.namespace_manifest_sha256[
            "validation"
        ],
    )
    expected_candidate_hashes = authorization.get("candidate_model_hashes", {})
    observed_candidate_hashes = {
        spec.architecture_id: model_configuration_hash(
            candidate_model_configuration(root, spec)
        )
        for spec in load_architecture_specs(root)
        if not spec.reused_probe
    }
    if expected_candidate_hashes != observed_candidate_hashes:
        raise TrainingGateError("candidate model hashes differ from authorization")
    return {
        "authorization": authorization,
        "publication": publication,
        "reused_probe_summaries": reused,
        "architecture_grid_sha256": grid_hash,
    }


def run_authorized_architecture_fit(
    root: Path,
    *,
    authorization_path: Path,
    stage_a_publication_root: Path,
    stage_b_publication_root: Path,
    combined_publication_root: Path,
    correction_publication_root: Optional[Path] = None,
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
    """Fit one of the nine new results; the reused probe is always rejected."""

    specifications = {spec.architecture_id: spec for spec in load_architecture_specs(root)}
    if architecture not in NEW_ARCHITECTURE_IDS or architecture not in specifications:
        raise TrainingGateError("fit is not one of the nine authorized new results")
    if seed not in SEEDS:
        raise TrainingGateError("architecture fit seed is outside 0/1/2")
    gate = validate_architecture_execution_gate(
        root,
        authorization_path=authorization_path,
        stage_a_publication_root=stage_a_publication_root,
        stage_b_publication_root=stage_b_publication_root,
        combined_publication_root=combined_publication_root,
        correction_publication_root=correction_publication_root,
        terminal_decision_path=terminal_decision_path,
        probe_output_root=probe_output_root,
    )
    authorization = gate["authorization"]
    immutable = authorization.get("immutable_training", {})
    artifacts = validate_immutable_training_artifacts(
        root,
        immutable,
        training_commit=training_commit,
        environment_lock_path=environment_lock_path,
    )
    _verify_training_checkout(root, training_commit, authorization_path)
    configured_output = Path(str(authorization.get("architecture_output_root", ""))).resolve()
    if output_root.resolve() != configured_output or not output_root.is_relative_to(
        Path("/root/autodl-tmp/lensing-4")
    ):
        raise TrainingGateError("architecture output root is not authorized")
    model = candidate_model_configuration(root, specifications[architecture])
    expected_model_hash = authorization["candidate_model_hashes"][architecture]
    if model_configuration_hash(model) != expected_model_hash:
        raise TrainingGateError("architecture model configuration hash mismatch")
    _validate_runtime_versions(model)
    load_input_policy(root)
    publication = gate["publication"]
    curves = _verified_curves(model, psd_root)
    train_dataset = _locked_training_dataset(publication, curves)
    validation_dataset = PublishedStageADataset(
        publication.stage_a.validation_root,
        expected_split=SplitName.VALIDATION,
        detector_curves=curves,
        expected_total_pairs=VALIDATION_COUNT,
    )
    train_ids = train_dataset.physical_system_ids()
    preparation = _load_mapping(
        probe_output_root / "rung-65536" / "rung_preparation.json"
    )
    if not (
        tuple(preparation.get("member_ids", ())) == train_ids
        and preparation.get("combined_manifest_sha256")
        == _locked_preparation_manifest_sha256(publication)
    ):
        raise TrainingGateError("architecture fits differ from the locked-rung membership")
    input_standardizer, target_standardizer = _load_standardizers(preparation)
    identity = TrainingRunIdentity(
        model_configuration_hash=expected_model_hash,
        training_code_commit=training_commit,
        training_environment_sha256=artifacts["environment_lock_sha256"],
        train_manifest_sha256=_locked_train_manifest_sha256(publication),
        validation_manifest_sha256=publication.stage_a.namespace_manifest_sha256[
            "validation"
        ],
        final_evaluation_commitment_sha256=str(
            authorization.get("final_evaluation_commitment_sha256", "")
        ),
        membership_sha256=membership_hash(train_ids),
        input_standardizer_sha256=standardizer_hash(input_standardizer),
        target_standardizer_sha256=standardizer_hash(target_standardizer),
        training_rung_count=TRAIN_65K_COUNT,
        seed=seed,
    )
    identity.validate()
    run_directory = output_root / architecture / f"seed-{seed}"
    if run_directory.exists() and resume_checkpoint is None:
        raise FileExistsError("architecture result identity already exists")
    evidence = architecture_execution_evidence(
        identity,
        architecture=architecture,
        immutable_wheel_sha256=artifacts["wheel_sha256"],
    )
    _atomic_json(run_directory / "run_preparation.json", evidence)
    workers = int(authorization.get("data_loader_worker_processes", 0))
    if not 0 <= workers <= 16:
        raise TrainingGateError("architecture data-loader worker count is outside review")
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
        "status": "completed_architecture_fit_and_development_validation",
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
    }
    _atomic_json(run_directory / "run_summary.json", result)
    return result


def select_architecture(
    results: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Select one architecture by three-seed mean NLP, never by best seed."""

    if len(results) != 12:
        raise TrainingGateError("architecture selection requires exactly twelve results")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        architecture = str(result.get("architecture_id", ""))
        grouped.setdefault(architecture, []).append(result)
    if set(grouped) != set(ALL_ARCHITECTURE_IDS):
        raise TrainingGateError("architecture result grid is incomplete")
    rows: list[dict[str, Any]] = []
    for architecture in ALL_ARCHITECTURE_IDS:
        values = grouped[architecture]
        seeds = tuple(sorted(int(value.get("seed", -1)) for value in values))
        if seeds != SEEDS:
            raise TrainingGateError("architecture result lacks the exact three seeds")
        parameter_counts = {
            int(value.get("trainable_parameter_count", -1)) for value in values
        }
        if len(parameter_counts) != 1 or min(parameter_counts) <= 0:
            raise TrainingGateError("architecture parameter-count identity is inconsistent")
        nlps = [
            float(
                value.get(
                    "development_mean_nlp_nat_per_target_dimension", float("nan")
                )
            )
            for value in values
        ]
        if not all(value == value and abs(value) != float("inf") for value in nlps):
            raise TrainingGateError("architecture NLP metric is nonfinite")
        rows.append(
            {
                "architecture_id": architecture,
                "mean_validation_nlp_nat_per_target_dimension": sum(nlps) / 3.0,
                "seed_validation_nlp": {
                    str(int(value["seed"])): float(
                        value["development_mean_nlp_nat_per_target_dimension"]
                    )
                    for value in values
                },
                "trainable_parameter_count": parameter_counts.pop(),
                "reused_probe_fits": architecture == PROBE_ARCHITECTURE_ID,
            }
        )
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["mean_validation_nlp_nat_per_target_dimension"]),
            int(row["trainable_parameter_count"]),
            str(row["architecture_id"]),
        ),
    )
    return {
        "status": "architecture_locked_on_development_validation",
        "selected_architecture_id": ordered[0]["architecture_id"],
        "selection_metric": "mean_validation_nlp_across_three_seeds",
        "best_seed_selected": False,
        "architecture_results": rows,
        "total_result_count": 12,
        "new_fit_count": 9,
        "reused_probe_fit_count": 3,
        "calibration_accessed": False,
        "sbc_accessed": False,
        "final_evaluation_accessed": False,
        "opens_later_gate_automatically": False,
    }


def collect_architecture_results(
    root: Path,
    *,
    gate: Mapping[str, Any],
    architecture_output_root: Path,
) -> Tuple[Mapping[str, Any], ...]:
    """Collect exactly 3 reused plus 9 new development-only result summaries."""

    authorization = gate["authorization"]
    publication = gate["publication"]
    reused_identity = gate["reused_probe_summaries"][0]["identity"]
    base_model = load_yaml(root / BASE_MODEL_PATH)
    probe_model = build_probe_model(base_model, seed=0)
    probe_parameter_count = sum(
        int(parameter.numel()) for parameter in probe_model.parameters()
    )
    rows = []
    for summary in gate["reused_probe_summaries"]:
        rows.append(
            {
                "architecture_id": PROBE_ARCHITECTURE_ID,
                "seed": int(summary["identity"]["seed"]),
                "trainable_parameter_count": probe_parameter_count,
                "development_mean_nlp_nat_per_target_dimension": float(
                    summary["development"][
                        "mean_nlp_nat_per_target_dimension"
                    ]
                ),
                "reused_probe_fit": True,
            }
        )
    candidate_hashes = authorization["candidate_model_hashes"]
    for architecture in NEW_ARCHITECTURE_IDS:
        expected_model_hash = candidate_hashes[architecture]
        for seed in SEEDS:
            path = architecture_output_root / architecture / f"seed-{seed}" / "run_summary.json"
            if not path.is_file():
                raise TrainingGateError("new architecture result summary is absent")
            summary = _load_mapping(path)
            identity = summary.get("identity", {})
            development = summary.get("development", {})
            if (
                summary.get("status")
                != "completed_architecture_fit_and_development_validation"
                or summary.get("architecture_id") != architecture
                or int(summary.get("seed", -1)) != seed
                or identity.get("model_configuration_hash") != expected_model_hash
                or int(identity.get("training_rung_count", -1)) != TRAIN_65K_COUNT
                or identity.get("train_manifest_sha256")
                != _locked_train_manifest_sha256(publication)
                or identity.get("validation_manifest_sha256")
                != publication.stage_a.namespace_manifest_sha256["validation"]
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
                raise TrainingGateError("new architecture result violates the frozen grid")
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
