"""Fail-closed final-evaluation publication and inference stack.

The current gate permits implementation and synthetic fixtures only.  Real
sealed data, selected checkpoints and calibration maps remain inaccessible
until a later identity-bound authorization is supplied.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..config import load_yaml
from ..production.final_evaluation import (
    FinalEvaluationNamespace,
    final_evaluation_namespaces,
    validate_final_evaluation_record,
)
from ..provenance import configuration_hash
from ..schema import V2Record
from .calibration import LEVELS, TARGETS, empirical_pit_scores, joint_hpd_scores
from .contracts import TrainingGateError, model_configuration_hash
from .data import PublishedStageADataset, torch_collate
from .engine import TargetStandardizer, standardizer_hash
from .features import PreparedExample, load_input_policy
from .final_evaluation import (
    FINAL_ANALYSIS_HASH,
    condition_on_lens_family,
    cross_family_analysis_context,
)
from .metrics import empirical_crps
from .reference_baseline import REFERENCE_CONFIG_HASH
from .rung65 import _load_standardizers
from .runner import (
    _validate_runtime_versions,
    _verified_curves,
    _verify_training_checkout,
)
from .terminal_downstream import checkpoint_training_rung_is_authorized

STACK_AUTHORIZATION = (
    "configs/execution/phase7_final_inference_stack_authorization.yaml"
)
FINAL_GENERATOR_CONFIG = "configs/data/phase4_final_evaluation.yaml"
FINAL_GENERATOR_CONFIG_HASH = (
    "11277a2a4c5d233e6f525b3ab5d6ece90c115d818d752849076f3a136e574d66"
)
FINAL_COMMITMENT = "results/phase4/final_evaluation_commitment.json"
FINAL_COMMITMENT_SHA256 = (
    "c13412eced163bac26abc4b22d054f3a6fa967e7e5a4dd7849ebf54f42df6083"
)
FINAL_CASE_COUNT = 20480
FINAL_SHARD_COUNT = 160
FINAL_NAMESPACE_COUNT = 15
POSTERIOR_DRAW_COUNT = 4096
MAXIMUM_DRAW_MICROBATCH = 512
MODEL_SEEDS = (0, 1, 2)
INFERENCE_SEED_DOMAIN = "final_evaluation_posterior_sampling_v1"
APPROVED_REMOTE_ROOT = Path("/root/autodl-tmp/lensing-4")


@dataclass(frozen=True)
class FinalNamespacePublication:
    """One sealed namespace named by a passed parent manifest."""

    specification: FinalEvaluationNamespace
    dataset_id: str
    dataset_root: Path
    validation: Mapping[str, Any]


@dataclass(frozen=True)
class FinalEvaluationPublication:
    """Validated immutable identity of the complete sealed final pool."""

    parent_root: Path
    manifest_path: Path
    manifest_sha256: str
    generator_commit: str
    configuration_hash: str
    commitment_sha256: str
    namespaces: Mapping[str, FinalNamespacePublication]


@dataclass(frozen=True)
class FinalEvaluationCase:
    """One prepared case plus offline split/context labels."""

    example: PreparedExample
    split: str
    diagnostic_context_id: str


class SealedFinalNamespaceDataset(PublishedStageADataset):
    """Lazy reader that additionally enforces one frozen diagnostic namespace."""

    def __init__(
        self,
        publication: FinalNamespacePublication,
        *,
        detector_curves: Mapping[str, Any],
    ) -> None:
        self.publication = publication
        super().__init__(
            publication.dataset_root,
            expected_split=publication.specification.split,
            detector_curves=detector_curves,
            expected_total_pairs=publication.specification.accepted_count,
            require_published=False,
        )

    def _record(self, entry: Any) -> V2Record:
        record = super()._record(entry)
        validate_final_evaluation_record(
            record,
            self.publication.specification,
            expected_dataset=self.publication.dataset_id,
        )
        return record

    def final_case(self, index: int) -> FinalEvaluationCase:
        records = self._open_records(self.entries[index].path)
        cell = str(records.iloc[self.entries[index].row_index]["em_cell"])
        if not cell:
            raise FinalInferenceGateError("final record has no EM-cell partition")
        example = self[index]
        return FinalEvaluationCase(
            example=example,
            split=self.publication.specification.split.value,
            diagnostic_context_id=(
                self.publication.specification.diagnostic_context_id
            ),
        )


class StandardizedFinalNamespaceDataset:
    """Apply training-only scales and frozen family conditions lazily."""

    def __init__(self, dataset: SealedFinalNamespaceDataset, standardizer: Any) -> None:
        self.dataset = dataset
        self.standardizer = standardizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> FinalEvaluationCase:
        case = self.dataset.final_case(index)
        example = self.standardizer.transform(case.example)
        namespace = self.dataset.publication.specification
        if namespace.split.value == "cross_family_misspecification_test":
            context = cross_family_analysis_context(namespace.diagnostic_context_id)
            if context.inference_mode == "sie_family_condition":
                example = condition_on_lens_family(example, "sie_external_shear")
            elif context.inference_mode == (
                "epl_family_condition_with_frozen_training_slope_prior_marginalized"
            ):
                example = condition_on_lens_family(example, "epl_external_shear")
        return FinalEvaluationCase(
            example=example,
            split=case.split,
            diagnostic_context_id=case.diagnostic_context_id,
        )


class FinalInferenceGateError(TrainingGateError):
    """Raised when final inference would cross a frozen gate."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalInferenceGateError(f"expected a JSON mapping: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, value: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **value)
    os.replace(temporary, path)


def load_final_inference_stack_contract(root: Path) -> Mapping[str, Any]:
    """Validate the implementation-only gate and every frozen parent identity."""

    authorization = load_yaml(root / STACK_AUTHORIZATION)
    if authorization.get("authorization_status") != (
        "authorized_implementation_and_synthetic_fixture_only"
    ):
        raise FinalInferenceGateError("final-inference implementation gate is absent")
    frozen = authorization.get("frozen_contracts", {})
    if (
        configuration_hash(
            load_yaml(root / "configs/statistics/final_evaluation_analysis_preregistration.yaml")
        )
        != FINAL_ANALYSIS_HASH
        or configuration_hash(
            load_yaml(root / "configs/statistics/reference_baseline_preregistration.yaml")
        )
        != REFERENCE_CONFIG_HASH
        or configuration_hash(load_yaml(root / FINAL_GENERATOR_CONFIG))
        != FINAL_GENERATOR_CONFIG_HASH
        or _sha256(root / FINAL_COMMITMENT) != FINAL_COMMITMENT_SHA256
        or frozen.get("final_analysis_hash") != FINAL_ANALYSIS_HASH
        or frozen.get("reference_baseline_hash") != REFERENCE_CONFIG_HASH
        or frozen.get("final_generator_configuration_hash")
        != FINAL_GENERATOR_CONFIG_HASH
        or frozen.get("final_generation_commitment_sha256")
        != FINAL_COMMITMENT_SHA256
    ):
        raise FinalInferenceGateError("final-inference parent contract drifted")
    flags = authorization.get("authorization", {})
    allowed_true = {
        "sealed_publication_resolver_implementation_authorized",
        "checkpoint_inference_runner_implementation_authorized",
        "calibrated_metric_artifact_implementation_authorized",
        "cross_family_executor_implementation_authorized",
        "synthetic_fixture_tests_authorized",
    }
    if any(flags.get(name) is not True for name in allowed_true):
        raise FinalInferenceGateError("final-inference implementation gate is incomplete")
    if any(value is not False for name, value in flags.items() if name not in allowed_true):
        raise FinalInferenceGateError("implementation gate opened scientific execution")
    return authorization


def _expected_namespaces(root: Path) -> Tuple[FinalEvaluationNamespace, ...]:
    config = load_yaml(root / FINAL_GENERATOR_CONFIG)
    if configuration_hash(config) != FINAL_GENERATOR_CONFIG_HASH:
        raise FinalInferenceGateError("final generator configuration hash mismatch")
    namespaces = final_evaluation_namespaces(config)
    if (
        len(namespaces) != FINAL_NAMESPACE_COUNT
        or sum(item.accepted_count for item in namespaces) != FINAL_CASE_COUNT
        or sum(item.shard_count for item in namespaces) != FINAL_SHARD_COUNT
    ):
        raise FinalInferenceGateError("final namespace arithmetic drifted")
    return namespaces


def resolve_sealed_final_publication(
    root: Path,
    parent_root: Path,
    *,
    expected_manifest_sha256: Optional[str] = None,
) -> FinalEvaluationPublication:
    """Resolve manifests only; no Parquet, Zarr or checkpoint is opened."""

    parent = parent_root.resolve()
    if "sealed" not in parent.parts or not parent.is_dir():
        raise FinalInferenceGateError("final pool is not an atomic sealed publication")
    manifest_path = parent / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FinalInferenceGateError("sealed final parent manifest is absent")
    manifest_sha256 = _sha256(manifest_path)
    if (
        expected_manifest_sha256 is not None
        and manifest_sha256 != expected_manifest_sha256
    ):
        raise FinalInferenceGateError("sealed final parent manifest hash mismatch")
    manifest = _load_mapping(manifest_path)
    if (
        manifest.get("status"),
        manifest.get("sealed"),
        manifest.get("unsealing_authorized"),
        int(manifest.get("accepted_pair_count", -1)),
        int(manifest.get("complete_shard_count", -1)),
        int(manifest.get("namespace_count", -1)),
        manifest.get("all_namespaces_group_disjoint"),
        manifest.get("learning_curve_use_authorized"),
        manifest.get("architecture_selection_use_authorized"),
        manifest.get("calibration_fit_use_authorized"),
        manifest.get("gwosc_gwtc_accessed"),
    ) != (
        "passed_sealed",
        True,
        False,
        FINAL_CASE_COUNT,
        FINAL_SHARD_COUNT,
        FINAL_NAMESPACE_COUNT,
        True,
        False,
        False,
        False,
        False,
    ):
        raise FinalInferenceGateError("sealed final parent manifest contract failed")
    if (
        manifest.get("configuration_hash") != FINAL_GENERATOR_CONFIG_HASH
        or manifest.get("commitment_sha256") != FINAL_COMMITMENT_SHA256
    ):
        raise FinalInferenceGateError("sealed final parent identity drifted")
    validations = manifest.get("validations")
    expected = {item.namespace_id: item for item in _expected_namespaces(root)}
    if not isinstance(validations, dict) or set(validations) != set(expected):
        raise FinalInferenceGateError("sealed final namespace set is incomplete")
    resolved: Dict[str, FinalNamespacePublication] = {}
    dataset_ids: set[str] = set()
    for namespace_id, specification in expected.items():
        validation = validations[namespace_id]
        if not isinstance(validation, dict):
            raise FinalInferenceGateError("sealed namespace validation is not a mapping")
        dataset_id = str(validation.get("dataset_id", ""))
        if (
            validation.get("status"),
            validation.get("namespace_id"),
            validation.get("split"),
            validation.get("diagnostic_context_id"),
            int(validation.get("accepted_pair_count", -1)),
            int(validation.get("complete_shard_count", -1)),
        ) != (
            "passed_sealed",
            namespace_id,
            specification.split.value,
            specification.diagnostic_context_id,
            specification.accepted_count,
            specification.shard_count,
        ):
            raise FinalInferenceGateError("sealed namespace count or context mismatch")
        if (
            not dataset_id
            or Path(dataset_id).name != dataset_id
            or dataset_id in dataset_ids
        ):
            raise FinalInferenceGateError("sealed namespace dataset identity is invalid")
        dataset_root = parent / dataset_id
        run_manifest = _load_mapping(dataset_root / "run_manifest.json")
        if (
            run_manifest.get("status"),
            run_manifest.get("namespace_id"),
            run_manifest.get("split"),
            run_manifest.get("diagnostic_context_id"),
            run_manifest.get("dataset_id"),
            int(run_manifest.get("accepted_target", -1)),
            run_manifest.get("unsealing_authorized"),
        ) != (
            "generating_or_resuming_sealed",
            namespace_id,
            specification.split.value,
            specification.diagnostic_context_id,
            dataset_id,
            specification.accepted_count,
            False,
        ):
            raise FinalInferenceGateError("sealed namespace run manifest mismatch")
        dataset_ids.add(dataset_id)
        resolved[namespace_id] = FinalNamespacePublication(
            specification=specification,
            dataset_id=dataset_id,
            dataset_root=dataset_root,
            validation=validation,
        )
    return FinalEvaluationPublication(
        parent_root=parent,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        generator_commit=str(manifest.get("generator_commit", "")),
        configuration_hash=str(manifest["configuration_hash"]),
        commitment_sha256=str(manifest["commitment_sha256"]),
        namespaces=resolved,
    )


def final_inference_seed(namespace_id: str, model_seed: int) -> int:
    """Derive one deterministic, disjoint namespace/model sampling seed."""

    if not namespace_id or model_seed not in MODEL_SEEDS:
        raise FinalInferenceGateError("final inference seed inputs are invalid")
    payload = f"{INFERENCE_SEED_DOMAIN}\0{namespace_id}\0{model_seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def calibrated_final_batch_metrics(
    posterior_draws: np.ndarray,
    truth: np.ndarray,
    posterior_draw_log_density: np.ndarray,
    truth_log_density: np.ndarray,
    em_cells: Sequence[str],
    calibration_map: Mapping[str, Any],
) -> Mapping[str, np.ndarray]:
    """Compute frozen per-case scores without retaining posterior draws."""

    draws = np.asarray(posterior_draws, dtype=np.float64)
    targets = np.asarray(truth, dtype=np.float64)
    draw_density = np.asarray(posterior_draw_log_density, dtype=np.float64)
    truth_density = np.asarray(truth_log_density, dtype=np.float64)
    cells = tuple(str(value) for value in em_cells)
    if (
        draws.ndim != 3
        or draws.shape[1:] != (POSTERIOR_DRAW_COUNT, 2)
        or targets.shape != (len(draws), 2)
        or draw_density.shape != (len(draws), POSTERIOR_DRAW_COUNT)
        or truth_density.shape != (len(draws),)
        or len(cells) != len(draws)
        or any(not value for value in cells)
        or not all(
            np.all(np.isfinite(value))
            for value in (draws, targets, draw_density, truth_density)
        )
    ):
        raise FinalInferenceGateError("final posterior batch has invalid shape or values")
    marginal_scores = empirical_pit_scores(draws, targets)
    joint_scores = joint_hpd_scores(draw_density, truth_density)
    result: Dict[str, np.ndarray] = {
        "truth": targets,
        "truth_log_density": truth_density,
        "nlp_nat_per_target_dimension": -truth_density / 2.0,
        "crps": empirical_crps(draws, targets),
        "marginal_region_scores": marginal_scores,
        "joint_region_scores": joint_scores,
    }
    cell_array = np.asarray(cells, dtype=np.str_)
    for level in LEVELS:
        key = f"{level:.2f}"
        suffix = f"{int(round(level * 100)):02d}"
        marginal_covered = np.empty((len(draws), 2), dtype=bool)
        marginal_width = np.empty((len(draws), 2), dtype=np.float64)
        joint_covered = np.empty(len(draws), dtype=bool)
        for cell in sorted(set(cells)):
            selected = np.flatnonzero(cell_array == cell)
            try:
                mapping = calibration_map["em_cells"][cell]
            except KeyError as error:
                raise FinalInferenceGateError(
                    f"calibration map has no final EM cell: {cell}"
                ) from error
            for target_index, target_name in enumerate(TARGETS):
                threshold = float(
                    mapping["marginal"][target_name][key][
                        "raw_region_mass_threshold"
                    ]
                )
                if not 0.0 <= threshold <= 1.0:
                    raise FinalInferenceGateError("calibration threshold lies outside [0,1]")
                marginal_covered[selected, target_index] = (
                    marginal_scores[selected, target_index] <= threshold
                )
                alpha = (1.0 - threshold) / 2.0
                values = draws[selected, :, target_index]
                # NumPy's default is the frozen linear quantile. Omitting the
                # version-renamed keyword keeps both supported type stubs valid.
                lower = np.quantile(values, alpha, axis=1)
                upper = np.quantile(values, 1.0 - alpha, axis=1)
                marginal_width[selected, target_index] = upper - lower
            joint_threshold = float(
                mapping["joint"][key]["raw_region_mass_threshold"]
            )
            if not 0.0 <= joint_threshold <= 1.0:
                raise FinalInferenceGateError("joint calibration threshold is invalid")
            joint_covered[selected] = joint_scores[selected] <= joint_threshold
        result[f"marginal_covered_{suffix}"] = marginal_covered
        result[f"joint_covered_{suffix}"] = joint_covered
        result[f"marginal_interval_width_{suffix}"] = marginal_width
    if not all(np.all(np.isfinite(value)) for value in result.values()):
        raise FinalInferenceGateError("final metric payload contains NaN or Inf")
    return result


def torch_final_collate(
    cases: Sequence[FinalEvaluationCase],
) -> Tuple[Mapping[str, Any], Tuple[Mapping[str, str], ...]]:
    """Keep every diagnostic label outside the deployable tensor mapping."""

    tensors = torch_collate([case.example for case in cases])
    metadata = tuple(
        {
            "physical_system_id": case.example.physical_system_id,
            "lens_family": case.example.lens_family,
            "em_cell": str(case.example.em_cell or ""),
            "split": case.split,
            "diagnostic_context_id": case.diagnostic_context_id,
        }
        for case in cases
    )
    if any(not value["em_cell"] for value in metadata):
        raise FinalInferenceGateError("final batch has no EM-cell label")
    return tensors, metadata


def _final_data_loader(
    dataset: StandardizedFinalNamespaceDataset,
    *,
    batch_size: int,
    seed: int,
    device_name: str,
) -> Any:
    torch = importlib.import_module("torch")
    data = importlib.import_module("torch.utils.data")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=torch_final_collate,
        pin_memory=device_name.startswith("cuda"),
        persistent_workers=False,
        generator=generator,
    )


def _move_inputs(batch: Mapping[str, Any], device: Any) -> Dict[str, Any]:
    return {name: value.to(device) for name, value in batch.items()}


def _context_with_family(model: Any, moved: Mapping[str, Any], family: str) -> Any:
    torch = importlib.import_module("torch")
    condition = (
        (1.0, 0.0) if family == "sie_external_shear" else (0.0, 1.0)
    )
    replaced = dict(moved)
    replaced["lens_family_condition"] = torch.as_tensor(
        condition,
        dtype=moved["lens_family_condition"].dtype,
        device=moved["lens_family_condition"].device,
    )[None].expand(len(moved["lens_family_condition"]), -1)
    return model.encode_context(replaced)


def _mixture_density(model: Any, samples: Any, sie_context: Any, epl_context: Any) -> Any:
    torch = importlib.import_module("torch")
    sie = model.sample_log_prob_from_context(samples, sie_context)
    epl = model.sample_log_prob_from_context(samples, epl_context)
    return torch.logaddexp(sie, epl) - float(np.log(2.0))


def _score_final_batches(
    model: Any,
    loader: Any,
    *,
    target_standardizer: TargetStandardizer,
    calibration_map: Mapping[str, Any],
    inference_seed: int,
    draw_microbatch: int,
    device_name: str,
) -> Mapping[str, np.ndarray]:
    """Create bounded per-case score arrays; posterior draws are never persisted."""

    if not 1 <= draw_microbatch <= MAXIMUM_DRAW_MICROBATCH:
        raise FinalInferenceGateError("final draw microbatch exceeds the frozen cap")
    torch = importlib.import_module("torch")
    device = torch.device(device_name)
    torch.manual_seed(inference_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(inference_seed)
    mean = np.asarray(target_standardizer.mean, dtype=np.float64)
    scale = np.asarray(target_standardizer.standard_deviation, dtype=np.float64)
    jacobian = target_standardizer.log_absolute_jacobian
    parts: Dict[str, list[np.ndarray]] = {}
    identifiers: list[str] = []
    lens_families: list[str] = []
    em_cells: list[str] = []
    splits: list[str] = []
    contexts: list[str] = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch, metadata in loader:
            moved = _move_inputs(batch, device)
            target_tensor = moved.pop("target")
            target = target_tensor.detach().cpu().numpy().astype(np.float64)
            standardized_truth = (
                target_tensor
                - torch.as_tensor(mean, dtype=target_tensor.dtype, device=device)
            ) / torch.as_tensor(scale, dtype=target_tensor.dtype, device=device)
            context_ids = {str(value["diagnostic_context_id"]) for value in metadata}
            if len(context_ids) != 1:
                raise FinalInferenceGateError("one batch mixed final diagnostic contexts")
            context_id = next(iter(context_ids))
            equal_mixture = False
            if str(metadata[0]["split"]) == "cross_family_misspecification_test":
                equal_mixture = (
                    cross_family_analysis_context(context_id).inference_mode
                    == "equal_density_mixture_of_sie_and_epl_family_conditions"
                )
            base_context = model.encode_context(moved)
            if equal_mixture:
                sie_context = _context_with_family(
                    model, moved, "sie_external_shear"
                )
                epl_context = _context_with_family(
                    model, moved, "epl_external_shear"
                )
                truth_sie = model.flow.log_prob(
                    standardized_truth, context=sie_context
                )
                truth_epl = model.flow.log_prob(
                    standardized_truth, context=epl_context
                )
                truth_density_tensor = (
                    torch.logaddexp(truth_sie, truth_epl)
                    - float(np.log(2.0))
                    - jacobian
                )
            else:
                truth_density_tensor = (
                    model.flow.log_prob(standardized_truth, context=base_context)
                    - jacobian
                )
            draw_parts: list[np.ndarray] = []
            density_parts: list[np.ndarray] = []
            remaining = POSTERIOR_DRAW_COUNT
            while remaining:
                chunk = min(draw_microbatch, remaining)
                if equal_mixture:
                    if chunk % 2:
                        chunk -= 1
                    if chunk == 0:
                        chunk = 2
                    per_family = chunk // 2
                    sie_sample = model.sample_from_context(per_family, sie_context)
                    epl_sample = model.sample_from_context(per_family, epl_context)
                    sampled = torch.empty(
                        (
                            len(target_tensor),
                            chunk,
                            int(sie_sample.shape[-1]),
                        ),
                        dtype=sie_sample.dtype,
                        device=sie_sample.device,
                    )
                    sampled[:, 0::2, :] = sie_sample
                    sampled[:, 1::2, :] = epl_sample
                    sampled_density = _mixture_density(
                        model, sampled, sie_context, epl_context
                    )
                else:
                    sampled = model.sample_from_context(chunk, base_context)
                    sampled_density = model.sample_log_prob_from_context(
                        sampled, base_context
                    )
                standardized = sampled.detach().cpu().numpy().astype(np.float64)
                draw_parts.append(
                    standardized * scale[None, None, :] + mean[None, None, :]
                )
                density_parts.append(
                    sampled_density.detach().cpu().numpy().astype(np.float64)
                    - jacobian
                )
                remaining -= chunk
            draws = np.concatenate(draw_parts, axis=1)
            densities = np.concatenate(density_parts, axis=1)
            metrics = calibrated_final_batch_metrics(
                draws,
                target,
                densities,
                truth_density_tensor.detach().cpu().numpy().astype(np.float64),
                tuple(str(value["em_cell"]) for value in metadata),
                calibration_map,
            )
            for name, values in metrics.items():
                parts.setdefault(name, []).append(np.asarray(values))
            identifiers.extend(str(value["physical_system_id"]) for value in metadata)
            lens_families.extend(str(value["lens_family"]) for value in metadata)
            em_cells.extend(str(value["em_cell"]) for value in metadata)
            splits.extend(str(value["split"]) for value in metadata)
            contexts.extend(str(value["diagnostic_context_id"]) for value in metadata)
    result = {name: np.concatenate(values, axis=0) for name, values in parts.items()}
    result.update(
        {
            "physical_system_ids": np.asarray(identifiers, dtype=np.str_),
            "lens_families": np.asarray(lens_families, dtype=np.str_),
            "em_cells": np.asarray(em_cells, dtype=np.str_),
            "splits": np.asarray(splits, dtype=np.str_),
            "diagnostic_context_ids": np.asarray(contexts, dtype=np.str_),
        }
    )
    return result


def _validate_artifact_map(
    value: Mapping[str, Any], *, label: str
) -> Mapping[int, Mapping[str, Any]]:
    if set(value) != {"0", "1", "2"}:
        raise FinalInferenceGateError(f"{label} must bind seeds 0, 1 and 2")
    result: Dict[int, Mapping[str, Any]] = {}
    for seed in MODEL_SEEDS:
        item = value[str(seed)]
        if not isinstance(item, dict):
            raise FinalInferenceGateError(f"{label} seed identity is not a mapping")
        path = Path(str(item.get("path", ""))).resolve()
        if (
            not path.is_absolute()
            or not path.is_relative_to(APPROVED_REMOTE_ROOT)
            or not path.is_file()
            or _sha256(path) != item.get("sha256")
        ):
            raise FinalInferenceGateError(f"{label} seed {seed} identity mismatch")
        result[seed] = item
    return result


def _validate_calibration_statistics(
    value: Mapping[str, Any],
) -> Mapping[int, Mapping[str, Any]]:
    """Bind each seed's map, independent SBC and common statistics summary."""

    if set(value) != {"0", "1", "2"}:
        raise FinalInferenceGateError(
            "calibration statistics must bind seeds 0, 1 and 2"
        )
    resolved: Dict[int, Mapping[str, Any]] = {}
    for seed in MODEL_SEEDS:
        item = value[str(seed)]
        if not isinstance(item, dict):
            raise FinalInferenceGateError("calibration statistic identity is invalid")
        paths: Dict[str, Path] = {}
        for label in (
            "run_summary",
            "calibration_map",
            "sbc_summary",
            "independent_coverage",
        ):
            path = Path(str(item.get(f"{label}_path", ""))).resolve()
            if (
                not path.is_relative_to(APPROVED_REMOTE_ROOT)
                or not path.is_file()
                or _sha256(path) != item.get(f"{label}_sha256")
            ):
                raise FinalInferenceGateError(
                    f"seed {seed} {label} identity mismatch"
                )
            paths[label] = path
        summary = _load_mapping(paths["run_summary"])
        score_identity = summary.get("score_identity", {})
        if (
            summary.get("status")
            != "completed_calibration_fit_and_independent_sbc"
            or int(score_identity.get("model_seed", -1)) != seed
            or summary.get("calibration_map_fitted_from_calibration_fit_only")
            is not True
            or summary.get("sbc_used_to_fit_calibration_map") is not False
            or summary.get("final_evaluation_accessed") is not False
            or summary.get("calibration_map_sha256")
            != item["calibration_map_sha256"]
            or summary.get("sbc_summary_sha256") != item["sbc_summary_sha256"]
            or summary.get("independent_coverage_sha256")
            != item["independent_coverage_sha256"]
        ):
            raise FinalInferenceGateError(
                "calibration/SBC run summary is incomplete or seed-mixed"
            )
        calibration = _load_mapping(paths["calibration_map"])
        sbc = _load_mapping(paths["sbc_summary"])
        if (
            calibration.get("status")
            != "fitted_split_conformal_region_level_maps"
            or sbc.get("status") != "completed_independent_sbc_rank_tests"
            or sbc.get("calibration_map_fitted_from_sbc") is not False
        ):
            raise FinalInferenceGateError(
                "same-seed calibration or independent SBC status is invalid"
            )
        resolved[seed] = {**item, **paths}
    return resolved


def validate_final_inference_authorization(
    root: Path,
    *,
    authorization_path: Path,
    publication_root: Path,
) -> Mapping[str, Any]:
    """Require all post-lock identities before reading one sealed record."""

    authorization = load_yaml(authorization_path)
    if authorization.get("authorization_status") != (
        "authorized_final_evaluation_inference_only"
    ):
        raise FinalInferenceGateError("final scientific inference is not authorized")
    flags = authorization.get("authorization", {})
    for key in (
        "final_evaluation_unsealing_authorized",
        "final_evaluation_data_access_authorized",
        "selected_checkpoint_inference_authorized",
        "same_seed_calibration_map_application_authorized",
        "immutable_score_artifact_creation_authorized",
    ):
        if flags.get(key) is not True:
            raise FinalInferenceGateError(f"final inference requires {key}=true")
    for key in (
        "model_training_or_tuning_authorized",
        "calibration_refit_authorized",
        "architecture_or_size_selection_authorized",
        "final_result_threshold_change_authorized",
        "gwosc_gwtc_access_authorized",
    ):
        if flags.get(key) is not False:
            raise FinalInferenceGateError(f"final inference requires {key}=false")
    frozen = authorization.get("frozen_contracts", {})
    if (
        frozen.get("final_analysis_hash") != FINAL_ANALYSIS_HASH
        or frozen.get("reference_baseline_hash") != REFERENCE_CONFIG_HASH
        or frozen.get("final_generation_commitment_sha256")
        != FINAL_COMMITMENT_SHA256
    ):
        raise FinalInferenceGateError("final inference authorization contract drifted")
    publication = resolve_sealed_final_publication(
        root,
        publication_root,
        expected_manifest_sha256=str(
            authorization.get("sealed_publication", {}).get("manifest_sha256", "")
        ),
    )
    checkpoints = _validate_artifact_map(
        authorization.get("selected_seed_checkpoints", {}),
        label="selected checkpoint",
    )
    statistics = _validate_calibration_statistics(
        authorization.get("same_seed_calibration_sbc_statistics", {})
    )
    decision = authorization.get("selected_architecture", {})
    decision_path = Path(str(decision.get("decision_path", ""))).resolve()
    if (
        not decision_path.is_file()
        or _sha256(decision_path) != decision.get("decision_sha256")
    ):
        raise FinalInferenceGateError("selected architecture decision identity mismatch")
    decision_value = _load_mapping(decision_path)
    architecture_id = str(decision_value.get("selected_architecture_id", ""))
    if architecture_id != decision.get("architecture_id"):
        raise FinalInferenceGateError("selected architecture differs from authorization")
    model = importlib.import_module(
        "gwlens_mm.training.architecture"
    ).selected_model_configuration(root, architecture_id)
    if model_configuration_hash(model) != decision.get("model_configuration_hash"):
        raise FinalInferenceGateError("selected model configuration hash mismatch")
    return {
        "authorization": authorization,
        "publication": publication,
        "checkpoints": checkpoints,
        "calibration_statistics": statistics,
        "model": model,
        "architecture_id": architecture_id,
    }


def _require_remote_path(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(APPROVED_REMOTE_ROOT):
        raise FinalInferenceGateError(f"{label} escaped the AutoDL project root")
    return resolved


def run_authorized_final_inference(
    root: Path,
    *,
    authorization_path: Path,
    publication_root: Path,
    namespace_id: str,
    seed: int,
    environment_lock_path: Path,
    psd_root: Path,
    output_path: Path,
    device_name: str = "cuda",
) -> Mapping[str, Any]:
    """Run one immutable seed/namespace score job after a future exact gate."""

    if seed not in MODEL_SEEDS:
        raise FinalInferenceGateError("final inference seed is outside 0/1/2")
    for path, label in (
        (authorization_path, "final inference authorization"),
        (publication_root, "sealed final publication"),
        (environment_lock_path, "final inference environment"),
        (psd_root, "PSD root"),
        (output_path, "final score output"),
    ):
        _require_remote_path(path, label=label)
    gate = validate_final_inference_authorization(
        root,
        authorization_path=authorization_path,
        publication_root=publication_root,
    )
    authorization = gate["authorization"]
    publication: FinalEvaluationPublication = gate["publication"]
    if namespace_id not in publication.namespaces:
        raise FinalInferenceGateError("final inference namespace is not committed")
    immutable = authorization.get("immutable_inference", {})
    inference_commit = str(immutable.get("git_commit", ""))
    if (
        environment_lock_path.resolve()
        != Path(str(immutable.get("environment_lock_path", ""))).resolve()
        or _sha256(environment_lock_path)
        != immutable.get("environment_lock_sha256")
    ):
        raise FinalInferenceGateError("final inference environment identity mismatch")
    _verify_training_checkout(root, inference_commit, authorization_path)
    model_config = gate["model"]
    _validate_runtime_versions(model_config)
    load_input_policy(root)
    curves = _verified_curves(model_config, psd_root)
    outputs = authorization.get("score_outputs", {})
    output_value = outputs.get(str(seed), {}) if isinstance(outputs, dict) else {}
    expected_output = Path(str(output_value.get(namespace_id, ""))).resolve()
    if (
        output_path.resolve() != expected_output
        or output_path.exists()
        or output_path.with_suffix(".summary.json").exists()
    ):
        raise FinalInferenceGateError("final score output identity is invalid or reused")
    checkpoint_path = Path(str(gate["checkpoints"][seed]["path"]))
    torch = importlib.import_module("torch")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    identity = state.get("identity", {})
    if (
        int(identity.get("seed", -1)) != seed
        or not checkpoint_training_rung_is_authorized(identity, authorization)
        or identity.get("model_configuration_hash")
        != model_configuration_hash(model_config)
    ):
        raise FinalInferenceGateError("selected checkpoint run identity mismatch")
    standardizer_value = {
        "input_standardizer": state["input_standardizer"],
        "target_standardizer": state["target_standardizer"],
        "input_standardizer_sha256": identity["input_standardizer_sha256"],
        "target_standardizer_sha256": identity["target_standardizer_sha256"],
    }
    input_standardizer, target_standardizer = _load_standardizers(
        standardizer_value
    )
    if (
        standardizer_hash(input_standardizer)
        != identity["input_standardizer_sha256"]
        or standardizer_hash(target_standardizer)
        != identity["target_standardizer_sha256"]
    ):
        raise FinalInferenceGateError("selected checkpoint standardizer mismatch")
    model = importlib.import_module("gwlens_mm.training.model").build_probe_model(
        model_config, seed=seed
    )
    model.load_state_dict(state["model"])
    namespace_publication = publication.namespaces[namespace_id]
    dataset = SealedFinalNamespaceDataset(
        namespace_publication,
        detector_curves=curves,
    )
    standardized = StandardizedFinalNamespaceDataset(dataset, input_standardizer)
    contract = authorization.get("inference_contract", {})
    if (
        int(contract.get("posterior_draws_per_case", -1))
        != POSTERIOR_DRAW_COUNT
        or int(contract.get("maximum_draw_microbatch", -1))
        != MAXIMUM_DRAW_MICROBATCH
    ):
        raise FinalInferenceGateError("final posterior inference contract drifted")
    physical_batch = int(contract.get("physical_batch_size", 0))
    draw_microbatch = int(contract.get("draw_microbatch", 0))
    if not 1 <= physical_batch <= 32 or not 1 <= draw_microbatch <= 512:
        raise FinalInferenceGateError("final inference batching contract is unsafe")
    inference_seed = final_inference_seed(namespace_id, seed)
    loader = _final_data_loader(
        standardized,
        batch_size=physical_batch,
        seed=inference_seed,
        device_name=device_name,
    )
    statistics = gate["calibration_statistics"][seed]
    calibration_map = _load_mapping(Path(statistics["calibration_map_path"]))
    payload = dict(
        _score_final_batches(
            model,
            loader,
            target_standardizer=target_standardizer,
            calibration_map=calibration_map,
            inference_seed=inference_seed,
            draw_microbatch=draw_microbatch,
            device_name=device_name,
        )
    )
    expected_count = namespace_publication.specification.accepted_count
    identifiers = tuple(str(value) for value in payload["physical_system_ids"])
    if (
        len(identifiers) != expected_count
        or len(set(identifiers)) != expected_count
        or payload["truth"].shape != (expected_count, 2)
    ):
        raise FinalInferenceGateError("final score payload count or IDs are invalid")
    payload.update(
        {
            "model_seed": np.asarray(seed, dtype=np.int64),
            "architecture_id": np.asarray(gate["architecture_id"], dtype=np.str_),
            "namespace_id": np.asarray(namespace_id, dtype=np.str_),
            "checkpoint_sha256": np.asarray(_sha256(checkpoint_path), dtype=np.str_),
            "publication_manifest_sha256": np.asarray(
                publication.manifest_sha256, dtype=np.str_
            ),
            "calibration_map_sha256": np.asarray(
                statistics["calibration_map_sha256"], dtype=np.str_
            ),
            "inference_commit": np.asarray(inference_commit, dtype=np.str_),
        }
    )
    _atomic_npz(output_path, payload)
    summary = {
        "status": "completed_final_score_extraction_only",
        "namespace_id": namespace_id,
        "split": namespace_publication.specification.split.value,
        "diagnostic_context_id": (
            namespace_publication.specification.diagnostic_context_id
        ),
        "model_seed": seed,
        "architecture_id": gate["architecture_id"],
        "case_count": expected_count,
        "posterior_draw_count": POSTERIOR_DRAW_COUNT,
        "score_artifact_path": str(output_path),
        "score_artifact_sha256": _sha256(output_path),
        "physical_system_ids_sha256": hashlib.sha256(
            json.dumps(sorted(identifiers), separators=(",", ":")).encode()
        ).hexdigest(),
        "posterior_draws_persisted": False,
        "calibration_refit": False,
        "model_retrained_or_tuned": False,
        "final_result_threshold_changed": False,
        "gwosc_gwtc_accessed": False,
    }
    _atomic_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def dry_run_plan(root: Path) -> Mapping[str, Any]:
    """Describe the future score stack without resolving scientific identities."""

    load_final_inference_stack_contract(root)
    namespaces = _expected_namespaces(root)
    return {
        "status": "implementation_ready_execution_closed",
        "final_analysis_hash": FINAL_ANALYSIS_HASH,
        "reference_baseline_hash": REFERENCE_CONFIG_HASH,
        "final_generation_commitment_sha256": FINAL_COMMITMENT_SHA256,
        "accepted_case_count": sum(item.accepted_count for item in namespaces),
        "shard_count": sum(item.shard_count for item in namespaces),
        "namespace_count": len(namespaces),
        "model_seeds": list(MODEL_SEEDS),
        "posterior_draws_per_case": POSTERIOR_DRAW_COUNT,
        "maximum_draw_microbatch": MAXIMUM_DRAW_MICROBATCH,
        "sealed_publication_identity": None,
        "selected_checkpoint_identities": None,
        "calibration_map_identities": None,
        "sbc_summary_identities": None,
        "final_data_accessed": False,
        "checkpoint_accessed": False,
        "calibration_map_accessed": False,
        "score_artifact_created": False,
        "model_trained_or_tuned": False,
        "gwosc_gwtc_accessed": False,
    }


def write_dry_run_plan(root: Path, output: Path) -> None:
    _atomic_json(output, dry_run_plan(root))
