"""Fail-closed training stack for the two preregistered input ablations."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..config import load_yaml
from ..provenance import configuration_hash
from ..schema import SplitName
from .architecture import selected_model_configuration
from .contracts import WAVEFORM_CORRECTION_HASH, TrainingGateError, model_configuration_hash
from .data import (
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
from .features import InputStandardizer, PreparedExample, load_input_policy
from .final_evaluation import em_only_example, gw_only_example
from .model import build_probe_model
from .rung65 import (
    SEEDS,
    TRAIN_65K_COUNT,
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

FINAL_ANALYSIS_PATH = "configs/statistics/final_evaluation_analysis_preregistration.yaml"
FINAL_ANALYSIS_HASH = "7e0e252f0a972e0b0ad2fe8f93f74f1f0172639a6fb258fc7a953be5fb7973e1"
IMPLEMENTATION_AUTHORIZATION_PATH = (
    "configs/execution/phase7_ablation_training_stack_authorization.yaml"
)
ABLATION_VIEWS = ("gw_only", "em_only")
MAXIMUM_FITS = 6


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TrainingGateError(f"expected JSON mapping: {path}")
    return value


def validate_ablation_stack_contract(root: Path) -> Mapping[str, Any]:
    """Validate the frozen RC.6 views and the implementation-only boundary."""

    addendum = load_yaml(root / FINAL_ANALYSIS_PATH)
    if (
        addendum.get("preregistration_version") != "1.1.0-rc.6"
        or configuration_hash(addendum) != FINAL_ANALYSIS_HASH
        or addendum.get("execution", {}).get("ablation_training_enabled") is not False
    ):
        raise TrainingGateError("final-analysis RC.6 ablation contract changed")
    ablations = addendum.get("ablations", {})
    for key in ("gw_only_same_backbone", "em_only_same_backbone"):
        specification = ablations.get(key, {})
        if not (
            specification.get("separately_trained") is True
            and specification.get("locked_architecture_and_training_size") is True
            and specification.get("seeds") == list(SEEDS)
            and specification.get("retain_lens_family_as_model_condition") is True
            and specification.get("optimizer_and_budget_identical_to_primary") is True
        ):
            raise TrainingGateError(f"RC.6 {key} contract changed")
    if not (
        ablations.get("additional_architecture_tuning") == "forbidden"
        and ablations.get("final_evaluation_use_for_selection") == "forbidden"
    ):
        raise TrainingGateError("RC.6 ablation-selection boundary changed")

    authorization = load_yaml(root / IMPLEMENTATION_AUTHORIZATION_PATH)
    if authorization.get("authorization_status") != "authorized_implementation_only":
        raise TrainingGateError("ablation stack lacks its implementation-only gate")
    contract = authorization.get("implementation_contract", {})
    if not (
        contract.get("ablation_views") == list(ABLATION_VIEWS)
        and contract.get("seeds") == list(SEEDS)
        and int(contract.get("maximum_future_fits", -1)) == MAXIMUM_FITS
        and contract.get("locked_training_size_required") is True
        and contract.get("locked_architecture_required") is True
        and contract.get("identical_optimizer_and_budget_required") is True
        and contract.get("final_evaluation_access_forbidden") is True
        and contract.get("additional_architecture_tuning_forbidden") is True
    ):
        raise TrainingGateError("ablation implementation contract is inconsistent")
    flags = authorization.get("authorization", {})
    for required in (
        "ablation_view_implementation_authorized",
        "fail_closed_runner_implementation_authorized",
        "synthetic_fixture_tests_authorized",
    ):
        if flags.get(required) is not True:
            raise TrainingGateError(f"ablation implementation requires {required}=true")
    for forbidden in (
        "scientific_data_access_authorized",
        "scientific_checkpoint_access_authorized",
        "ablation_fit_execution_authorized",
        "primary_model_retraining_authorized",
        "architecture_or_size_selection_authorized",
        "model_tuning_authorized",
        "calibration_or_sbc_access_authorized",
        "final_evaluation_materialization_authorized",
        "final_evaluation_unsealing_authorized",
        "final_evaluation_inference_authorized",
        "gwosc_gwtc_access_authorized",
    ):
        if flags.get(forbidden) is not False:
            raise TrainingGateError(f"implementation gate must keep {forbidden}=false")
    return {"addendum": addendum, "authorization": authorization}


def apply_ablation_view(example: PreparedExample, view: str) -> PreparedExample:
    """Apply exactly one RC.6 input view without modifying labels or targets."""

    if view == "gw_only":
        return gw_only_example(example)
    if view == "em_only":
        return em_only_example(example)
    raise TrainingGateError("unknown ablation input view")


class AblatedStandardizedDataset(StandardizedStageADataset):
    """Apply the primary-rung standardizer and then the frozen input ablation."""

    def __init__(
        self,
        dataset: PublishedStageADataset,
        standardizer: InputStandardizer,
        view: str,
    ) -> None:
        if view not in ABLATION_VIEWS:
            raise TrainingGateError("ablation dataset view is not preregistered")
        super().__init__(dataset, standardizer)
        self.view = view

    def __getitem__(self, index: int) -> PreparedExample:
        standardized = self.standardizer.transform(self.dataset[index])
        return apply_ablation_view(standardized, self.view)


class AblatedDevelopmentDataset(DevelopmentStageADataset):
    """Apply one frozen view while preserving labels outside model tensors."""

    def __init__(
        self,
        dataset: PublishedStageADataset,
        standardizer: InputStandardizer,
        view: str,
    ) -> None:
        if dataset.expected_split is not SplitName.VALIDATION:
            raise TrainingGateError("ablation development data must be validation")
        if view not in ABLATION_VIEWS:
            raise TrainingGateError("ablation development view is not preregistered")
        super().__init__(dataset, standardizer)
        self.view = view

    def __getitem__(self, index: int) -> DevelopmentCase:
        case = self.dataset.development_case(index)
        standardized = self.standardizer.transform(case.example)
        return DevelopmentCase(
            example=apply_ablation_view(standardized, self.view),
            tail_view=case.tail_view,
        )


def ablation_model_configuration(
    root: Path, *, architecture_id: str, view: str
) -> Mapping[str, Any]:
    """Derive a distinct identity while preserving architecture and optimization."""

    if view not in ABLATION_VIEWS:
        raise TrainingGateError("ablation model view is not preregistered")
    primary = selected_model_configuration(root, architecture_id)
    model: dict[str, Any] = copy.deepcopy(dict(primary))
    model["implementation_id"] = f"phase7-{architecture_id}-{view}-ablation-v1"
    model["ablation"] = {
        "view": view,
        "base_architecture_id": architecture_id,
        "base_model_configuration_hash": model_configuration_hash(primary),
        "transform_order": "primary_rung_standardization_then_input_masking",
        "target_changed": False,
        "optimizer_or_budget_changed": False,
    }
    execution = dict(model["execution"])
    execution["scientific_training_enabled"] = False
    execution["model_selection_enabled"] = False
    execution["engineering_smoke_only"] = True
    model["execution"] = execution
    return model


def _resolve_selected_architecture(
    root: Path,
    authorization: Mapping[str, Any],
    decision_path: Path,
) -> Tuple[str, Mapping[str, Any], Mapping[str, Any]]:
    configured = Path(str(authorization.get("selected_architecture_decision_path", "")))
    if decision_path.resolve() != configured.resolve():
        raise TrainingGateError("selected-architecture decision path changed")
    decision = _load_mapping(decision_path)
    architecture_id = str(decision.get("selected_architecture_id", ""))
    if (
        decision.get("status") != "architecture_locked_on_development_validation"
        or decision.get("best_seed_selected") is not False
        or int(decision.get("total_result_count", -1)) != 12
        or int(decision.get("new_fit_count", -1)) != 9
        or int(decision.get("reused_probe_fit_count", -1)) != 3
        or decision.get("calibration_accessed") is not False
        or decision.get("sbc_accessed") is not False
        or decision.get("final_evaluation_accessed") is not False
        or decision.get("opens_later_gate_automatically") is not False
        or _sha256(decision_path)
        != authorization.get("selected_architecture_decision_sha256")
    ):
        raise TrainingGateError("ablation gate lacks an exact development-only lock")
    primary = selected_model_configuration(root, architecture_id)
    if model_configuration_hash(primary) != authorization.get(
        "selected_primary_model_configuration_hash"
    ):
        raise TrainingGateError("selected primary model configuration changed")
    expected = authorization.get("ablation_model_configuration_hashes", {})
    observed = {
        view: model_configuration_hash(
            ablation_model_configuration(root, architecture_id=architecture_id, view=view)
        )
        for view in ABLATION_VIEWS
    }
    if expected != observed:
        raise TrainingGateError("ablation model identities differ from authorization")
    return architecture_id, primary, decision


def _validate_terminal_size_lock(
    authorization: Mapping[str, Any], decision_path: Path
) -> Mapping[str, Any]:
    configured = Path(str(authorization.get("terminal_size_decision_path", "")))
    if decision_path.resolve() != configured.resolve():
        raise TrainingGateError("terminal size-decision path changed")
    decision = _load_mapping(decision_path)
    if (
        decision.get("decision") != "lock_train_65k"
        or decision.get("comparison") != "train_32k_to_train_65k"
        or decision.get("extension_above_65536_authorized") is not False
        or _sha256(decision_path) != authorization.get("terminal_size_decision_sha256")
    ):
        raise TrainingGateError("ablation gate lacks the exact terminal 65k lock")
    return decision


def validate_ablation_training_execution_gate(
    root: Path,
    *,
    authorization_path: Path,
    stage_a_publication_root: Path,
    stage_b_publication_root: Path,
    combined_publication_root: Path,
    correction_publication_root: Path,
    terminal_size_decision_path: Path,
    selected_architecture_decision_path: Path,
    primary_rung_preparation_path: Path,
) -> Mapping[str, Any]:
    """Bind a future six-fit gate before any published strain or checkpoint opens."""

    validate_ablation_stack_contract(root)
    authorization = load_yaml(authorization_path)
    if authorization.get("authorization_status") != "authorized_ablation_training_only":
        raise TrainingGateError("scientific ablation-training authorization is absent")
    flags = authorization.get("authorization", {})
    for required in (
        "corrected_65k_data_access_authorized",
        "ablation_fit_execution_authorized",
        "development_validation_authorized",
    ):
        if flags.get(required) is not True:
            raise TrainingGateError(f"ablation execution requires {required}=true")
    for forbidden in (
        "primary_model_retraining_authorized",
        "architecture_or_size_selection_authorized",
        "additional_architecture_tuning_authorized",
        "calibration_or_sbc_access_authorized",
        "final_evaluation_materialization_authorized",
        "final_evaluation_unsealing_authorized",
        "final_evaluation_inference_authorized",
        "extension_above_65536_authorized",
        "gwosc_gwtc_access_authorized",
    ):
        if flags.get(forbidden) is not False:
            raise TrainingGateError(f"ablation gate must keep {forbidden}=false")
    if not (
        authorization.get("locked_training_rung") == TRAIN_65K_COUNT
        and authorization.get("authorized_ablation_views") == list(ABLATION_VIEWS)
        and authorization.get("authorized_training_seeds") == list(SEEDS)
        and int(authorization.get("maximum_fit_count", -1)) == MAXIMUM_FITS
        and int(authorization.get("maximum_concurrent_fits", -1)) == 3
    ):
        raise TrainingGateError("ablation count, seed, or locked-rung contract changed")

    terminal_decision = _validate_terminal_size_lock(
        authorization, terminal_size_decision_path
    )
    architecture_id, primary_model, decision = _resolve_selected_architecture(
        root, authorization, selected_architecture_decision_path
    )
    commitment_path = root / "results/phase4/final_evaluation_commitment.json"
    commitment = _load_mapping(commitment_path)
    commitment_hash = _sha256(commitment_path)
    if (
        commitment.get("commitment_status") != "finalized_before_training"
        or authorization.get("final_evaluation_commitment_sha256") != commitment_hash
    ):
        raise TrainingGateError("ablation gate lacks the sealed final commitment")

    correction = authorization.get("correction_publication", {})
    publication = resolve_corrected_training_publication(
        correction_publication_root,
        stage_a_parent_root=stage_a_publication_root,
        stage_b_parent_root=stage_b_publication_root,
        combined_base_root=combined_publication_root,
        expected_base_generator_commit=str(authorization.get("base_generator_commit", "")),
        expected_base_preregistration_hash=str(
            authorization.get("base_preregistration_hash", "")
        ),
        expected_correction_generator_commit=str(correction.get("generator_commit", "")),
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
        publication.corrected_combined_train_manifest_sha256
    ):
        raise TrainingGateError("ablation gate changed the corrected 65k view")

    configured_preparation = Path(
        str(authorization.get("primary_rung_preparation_path", ""))
    ).resolve()
    if primary_rung_preparation_path.resolve() != configured_preparation:
        raise TrainingGateError("primary-rung preparation path changed")
    preparation = _load_mapping(primary_rung_preparation_path)
    train_ids = tuple(str(value) for value in preparation.get("member_ids", ()))
    if not (
        preparation.get("status") == "ready_for_authorized_probe_fits"
        and int(preparation.get("rung_count", -1)) == TRAIN_65K_COUNT
        and len(train_ids) == TRAIN_65K_COUNT
        and len(set(train_ids)) == TRAIN_65K_COUNT
        and preparation.get("train_manifest_sha256")
        == publication.corrected_combined_train_manifest_sha256
        and preparation.get("validation_manifest_sha256")
        == publication.stage_a.namespace_manifest_sha256["validation"]
        and preparation.get("final_evaluation_commitment_sha256") == commitment_hash
        and _sha256(primary_rung_preparation_path)
        == authorization.get("primary_rung_preparation_sha256")
    ):
        raise TrainingGateError("ablation gate lacks the exact locked-rung preparation")
    input_standardizer, target_standardizer = _load_standardizers(preparation)
    if not (
        standardizer_hash(input_standardizer)
        == preparation.get("input_standardizer_sha256")
        and standardizer_hash(target_standardizer)
        == preparation.get("target_standardizer_sha256")
    ):
        raise TrainingGateError("locked-rung standardizer identity changed")
    return {
        "authorization": authorization,
        "publication": publication,
        "selected_architecture_id": architecture_id,
        "terminal_size_decision": terminal_decision,
        "selected_primary_model": primary_model,
        "selected_architecture_decision": decision,
        "preparation": preparation,
        "input_standardizer": input_standardizer,
        "target_standardizer": target_standardizer,
        "train_ids": train_ids,
        "final_evaluation_commitment_sha256": commitment_hash,
    }


def run_authorized_ablation_fit(
    root: Path,
    *,
    authorization_path: Path,
    stage_a_publication_root: Path,
    stage_b_publication_root: Path,
    combined_publication_root: Path,
    correction_publication_root: Path,
    terminal_size_decision_path: Path,
    selected_architecture_decision_path: Path,
    primary_rung_preparation_path: Path,
    environment_lock_path: Path,
    psd_root: Path,
    output_root: Path,
    training_commit: str,
    view: str,
    seed: int,
    device_name: str,
    resume_checkpoint: Optional[Path] = None,
) -> Mapping[str, Any]:
    """Execute one future authorized ablation fit and development evaluation."""

    if view not in ABLATION_VIEWS or seed not in SEEDS:
        raise TrainingGateError("ablation execution is limited to two views and seeds 0/1/2")
    gate = validate_ablation_training_execution_gate(
        root,
        authorization_path=authorization_path,
        stage_a_publication_root=stage_a_publication_root,
        stage_b_publication_root=stage_b_publication_root,
        combined_publication_root=combined_publication_root,
        correction_publication_root=correction_publication_root,
        terminal_size_decision_path=terminal_size_decision_path,
        selected_architecture_decision_path=selected_architecture_decision_path,
        primary_rung_preparation_path=primary_rung_preparation_path,
    )
    authorization = gate["authorization"]
    artifacts = validate_immutable_training_artifacts(
        root,
        authorization.get("immutable_training", {}),
        training_commit=training_commit,
        environment_lock_path=environment_lock_path,
    )
    _verify_training_checkout(root, training_commit, authorization_path)
    configured_output = Path(str(authorization.get("ablation_output_root", ""))).resolve()
    if output_root.resolve() != configured_output or not output_root.is_relative_to(
        Path("/root/autodl-tmp/lensing-4")
    ):
        raise TrainingGateError("ablation output root is not authorized")
    architecture_id = str(gate["selected_architecture_id"])
    model = ablation_model_configuration(
        root, architecture_id=architecture_id, view=view
    )
    _validate_runtime_versions(model)
    load_input_policy(root)
    curves = _verified_curves(model, psd_root)
    publication = gate["publication"]
    if not isinstance(publication, CorrectedTrainingPublication):
        raise TrainingGateError("ablation gate returned a non-corrected publication")
    train_dataset = corrected_65k_training_dataset(publication, curves)
    validation_dataset = PublishedStageADataset(
        publication.stage_a.validation_root,
        expected_split=SplitName.VALIDATION,
        detector_curves=curves,
        expected_total_pairs=VALIDATION_COUNT,
    )
    train_ids = train_dataset.physical_system_ids()
    if train_ids != gate["train_ids"] or set(train_ids) & set(
        validation_dataset.physical_system_ids()
    ):
        raise TrainingGateError("ablation train/validation membership changed")
    input_standardizer = gate["input_standardizer"]
    target_standardizer = gate["target_standardizer"]
    identity = TrainingRunIdentity(
        model_configuration_hash=model_configuration_hash(model),
        training_code_commit=training_commit,
        training_environment_sha256=artifacts["environment_lock_sha256"],
        train_manifest_sha256=publication.corrected_combined_train_manifest_sha256,
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
    run_directory = output_root / view / f"seed-{seed}"
    if run_directory.exists() and resume_checkpoint is None:
        raise FileExistsError("ablation fit identity already exists")
    evidence = {
        **authorized_probe_execution_evidence(identity),
        "fit_role": "authorized_input_ablation_fit",
        "ablation_view": view,
        "selected_architecture_id": architecture_id,
        "selected_architecture_decision_sha256": _sha256(
            selected_architecture_decision_path
        ),
        "terminal_size_decision_sha256": _sha256(terminal_size_decision_path),
        "selected_primary_model_configuration_hash": model_configuration_hash(
            gate["selected_primary_model"]
        ),
        "immutable_wheel_sha256": artifacts["wheel_sha256"],
        "optimizer_or_budget_changed": False,
        "architecture_or_size_selection_authorized": False,
        "calibration_or_sbc_accessed": False,
        "final_evaluation_accessed": False,
    }
    _atomic_json(run_directory / "run_preparation.json", evidence)
    workers = int(authorization.get("data_loader_worker_processes", 0))
    if not 0 <= workers <= 16:
        raise TrainingGateError("ablation data-loader worker count is outside review")
    train_view = AblatedStandardizedDataset(
        train_dataset, input_standardizer, view
    )
    validation_view = AblatedStandardizedDataset(
        validation_dataset, input_standardizer, view
    )
    _, physical_batch, _ = optimization_batch_geometry(model["optimization"])
    train_loader = _data_loader(
        train_view,
        batch_size=physical_batch,
        seed=seed,
        training=True,
        worker_processes=workers,
        device_name=device_name,
    )
    validation_loader = _data_loader(
        validation_view,
        batch_size=physical_batch,
        seed=seed,
        training=False,
        worker_processes=workers,
        device_name=device_name,
    )
    model_instance = build_probe_model(model, seed=seed)
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
        f"{architecture_id}\0{view}\0{seed}"
    ).encode()
    evaluation_seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
    development = evaluate_development_validation(
        model_instance,
        _development_loader(
            AblatedDevelopmentDataset(
                validation_dataset, input_standardizer, view
            ),
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
        "status": "completed_ablation_fit_and_development_validation",
        "ablation_view": view,
        "selected_architecture_id": architecture_id,
        "seed": seed,
        "identity": asdict(identity),
        "training": training,
        "development": development,
        "optimizer_or_budget_changed": False,
        "architecture_or_size_selection_authorized": False,
        "calibration_or_sbc_accessed": False,
        "final_evaluation_accessed": False,
    }
    _atomic_json(run_directory / "run_summary.json", result)
    return result


def summarize_ablation_results(
    results: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Require all six fits and report each three-seed development mean."""

    if len(results) != MAXIMUM_FITS:
        raise TrainingGateError("ablation summary requires exactly six results")
    architectures = {str(result.get("selected_architecture_id", "")) for result in results}
    if len(architectures) != 1 or not next(iter(architectures)):
        raise TrainingGateError("ablation results do not share one selected architecture")
    common_identity_keys = (
        "training_code_commit",
        "training_environment_sha256",
        "train_manifest_sha256",
        "validation_manifest_sha256",
        "final_evaluation_commitment_sha256",
        "membership_sha256",
        "input_standardizer_sha256",
        "target_standardizer_sha256",
        "training_rung_count",
    )
    for key in common_identity_keys:
        if len({result.get("identity", {}).get(key) for result in results}) != 1:
            raise TrainingGateError(f"ablation results differ in frozen identity field {key}")
    rows = []
    view_model_hashes = set()
    for view in ABLATION_VIEWS:
        matches = [result for result in results if result.get("ablation_view") == view]
        if tuple(sorted(int(result.get("seed", -1)) for result in matches)) != SEEDS:
            raise TrainingGateError("ablation summary lacks exact seeds 0/1/2")
        nlps = []
        for result in matches:
            development = result.get("development", {})
            value = float(development.get("mean_nlp_nat_per_target_dimension", np.nan))
            if (
                result.get("status")
                != "completed_ablation_fit_and_development_validation"
                or not np.isfinite(value)
                or result.get("optimizer_or_budget_changed") is not False
                or result.get("architecture_or_size_selection_authorized") is not False
                or result.get("calibration_or_sbc_accessed") is not False
                or result.get("final_evaluation_accessed") is not False
            ):
                raise TrainingGateError("ablation result is invalid or contaminated")
            nlps.append(value)
        model_hashes = {
            result.get("identity", {}).get("model_configuration_hash")
            for result in matches
        }
        if len(model_hashes) != 1 or not next(iter(model_hashes)):
            raise TrainingGateError("one ablation view does not share one model identity")
        view_model_hashes.update(model_hashes)
        rows.append(
            {
                "ablation_view": view,
                "mean_validation_nlp_nat_per_target_dimension": float(np.mean(nlps)),
                "seed_validation_nlp": {
                    str(int(result["seed"])): float(
                        result["development"][
                            "mean_nlp_nat_per_target_dimension"
                        ]
                    )
                    for result in matches
                },
            }
        )
    if len(view_model_hashes) != len(ABLATION_VIEWS):
        raise TrainingGateError("the two ablation views do not have distinct identities")
    return {
        "status": "completed_six_preregistered_ablation_fits",
        "selected_architecture_id": next(iter(architectures)),
        "fit_count": MAXIMUM_FITS,
        "views": rows,
        "best_seed_selected": False,
        "additional_architecture_tuning_performed": False,
        "calibration_or_sbc_accessed": False,
        "final_evaluation_accessed": False,
        "opens_later_gate_automatically": False,
    }
