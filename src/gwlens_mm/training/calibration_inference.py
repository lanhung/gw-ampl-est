"""Fail-closed checkpoint inference for calibration-fit and independent SBC scores."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from ..config import load_yaml
from ..provenance import canonical_json
from ..schema import SplitName
from .architecture import selected_model_configuration
from .calibration import (
    SBC_STATISTICS,
    deterministic_sbc_subset,
    randomized_rank_from_counts,
)
from .contracts import TrainingGateError, model_configuration_hash
from .data import (
    PublishedStageADataset,
    StandardizedCalibrationSBCDataset,
    torch_calibration_sbc_collate,
)
from .engine import TargetStandardizer, standardizer_hash
from .features import load_input_policy
from .rung65 import _load_standardizers
from .runner import (
    _validate_runtime_versions,
    _verified_curves,
    _verify_training_checkout,
)
from .terminal_downstream import checkpoint_training_rung_is_authorized

APPROVED_REMOTE_ROOT = Path("/root/autodl-tmp/lensing-4")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TrainingGateError(f"expected a JSON mapping: {path}")
    return value


def _atomic_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.savez(handle, **payload)
    os.replace(partial, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _require_remote_path(path: Path, *, name: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(APPROVED_REMOTE_ROOT):
        raise TrainingGateError(f"{name} escaped the AutoDL project root")
    return resolved


def score_inference_seed(
    contract: Mapping[str, Any], *, split: SplitName, model_seed: int
) -> int:
    """Resolve one split-specific seed and reject correlated/missing namespaces."""

    roots = contract.get("root_seed_by_split_and_model_seed", {})
    if not isinstance(roots, dict) or set(roots) != {
        SplitName.CALIBRATION_FIT.value,
        SplitName.SBC_DIAGNOSTIC.value,
    }:
        raise TrainingGateError("score inference requires exactly two seed namespaces")
    expected_keys = {"0", "1", "2"}
    values: Dict[str, Dict[str, int]] = {}
    for split_name, mapping in roots.items():
        if not isinstance(mapping, dict) or set(mapping) != expected_keys:
            raise TrainingGateError("score inference seed map must bind seeds 0/1/2")
        values[split_name] = {str(key): int(value) for key, value in mapping.items()}
    flattened = [value for mapping in values.values() for value in mapping.values()]
    if any(value < 0 for value in flattened) or len(set(flattened)) != 6:
        raise TrainingGateError("score inference seed namespaces collide or are invalid")
    return values[split.value][str(model_seed)]


def validate_score_inference_authorization(
    root: Path,
    *,
    authorization_path: Path,
    split: SplitName,
    seed: int,
    checkpoint_path: Path,
    publication_root: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Bind one score artifact to a locked architecture, seed, split and publication."""

    authorization = load_yaml(authorization_path)
    if authorization.get("authorization_status") != (
        "authorized_calibration_sbc_checkpoint_inference_only"
    ):
        raise TrainingGateError("calibration/SBC checkpoint inference is not authorized")
    if split not in {SplitName.CALIBRATION_FIT, SplitName.SBC_DIAGNOSTIC}:
        raise TrainingGateError("score inference received an unauthorized split")
    if seed not in (0, 1, 2):
        raise TrainingGateError("score inference seed is outside 0/1/2")
    from .calibration_execution_authorization import load_score_contract

    if dict(authorization.get("inference_contract", {})) != dict(
        load_score_contract(root)
    ):
        raise TrainingGateError("score inference contract differs from frozen config")
    _require_remote_path(authorization_path, name="score authorization")
    _require_remote_path(checkpoint_path, name="selected checkpoint")
    _require_remote_path(publication_root, name="development publication")
    _require_remote_path(output_path, name="score output")
    flags = authorization.get("authorization", {})
    for key in (
        "selected_checkpoint_inference_authorized",
        "calibration_fit_data_access_authorized",
        "sbc_diagnostic_data_access_authorized",
        "score_artifact_creation_authorized",
    ):
        if flags.get(key) is not True:
            raise TrainingGateError(f"score inference requires {key}=true")
    for key in (
        "calibration_map_fitting_authorized",
        "sbc_statistical_test_authorized",
        "model_retraining_or_tuning_authorized",
        "final_evaluation_authorized",
        "gwosc_gwtc_access_authorized",
    ):
        if flags.get(key) is not False:
            raise TrainingGateError(f"score-only gate requires {key}=false")
    decision = authorization.get("selected_architecture", {})
    decision_path = Path(str(decision.get("decision_path", "")))
    _require_remote_path(decision_path, name="selected-architecture decision")
    if (
        not decision_path.is_absolute()
        or _sha256(decision_path) != decision.get("decision_sha256")
    ):
        raise TrainingGateError("selected-architecture decision identity mismatch")
    selected = _load_mapping(decision_path)
    architecture_id = str(selected.get("selected_architecture_id", ""))
    if architecture_id != decision.get("architecture_id"):
        raise TrainingGateError("selected architecture differs from authorization")
    model = selected_model_configuration(root, architecture_id)
    if model_configuration_hash(model) != decision.get("model_configuration_hash"):
        raise TrainingGateError("selected model configuration hash mismatch")
    checkpoints = authorization.get("selected_seed_checkpoints", {})
    item = checkpoints.get(str(seed), {}) if isinstance(checkpoints, dict) else {}
    if (
        checkpoint_path.resolve() != Path(str(item.get("path", ""))).resolve()
        or _sha256(checkpoint_path) != item.get("sha256")
    ):
        raise TrainingGateError("selected checkpoint identity mismatch")
    publication = authorization.get("development_publication", {})
    if publication_root.resolve() != Path(str(publication.get("parent_root", ""))).resolve():
        raise TrainingGateError("calibration/SBC publication root mismatch")
    manifest = publication_root / "dataset_manifest.json"
    if _sha256(manifest) != publication.get("parent_manifest_sha256"):
        raise TrainingGateError("calibration/SBC publication manifest hash mismatch")
    outputs = authorization.get("score_outputs", {})
    expected_output = Path(str(outputs.get(str(seed), {}).get(split.value, ""))).resolve()
    summary_path = output_path.with_suffix(".summary.json")
    if (
        output_path.resolve() != expected_output
        or output_path.exists()
        or summary_path.exists()
    ):
        raise TrainingGateError("score output identity is unauthorized or already exists")
    immutable = authorization.get("immutable_inference", {})
    commit = str(immutable.get("git_commit", ""))
    _verify_training_checkout(root, commit, authorization_path)
    environment_path = Path(str(immutable.get("environment_lock_path", "")))
    _require_remote_path(environment_path, name="score-inference environment lock")
    if _sha256(environment_path) != immutable.get("environment_lock_sha256"):
        raise TrainingGateError("score-inference environment identity mismatch")
    return {
        "authorization": authorization,
        "model": model,
        "architecture_id": architecture_id,
        "inference_commit": commit,
    }


def _data_loader(dataset: Any, *, batch_size: int, seed: int, device_name: str) -> Any:
    torch = importlib.import_module("torch")
    data = importlib.import_module("torch.utils.data")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=torch_calibration_sbc_collate,
        pin_memory=device_name.startswith("cuda"),
        persistent_workers=False,
        generator=generator,
    )


def _move_inputs(batch: Mapping[str, Any], device: Any) -> Dict[str, Any]:
    return {name: value.to(device) for name, value in batch.items()}


def _score_batches(
    model: Any,
    loader: Any,
    *,
    split: SplitName,
    target_standardizer: TargetStandardizer,
    posterior_draw_count: int,
    posterior_draw_chunk_size: int,
    inference_seed: int,
    device_name: str,
) -> Dict[str, np.ndarray]:
    torch = importlib.import_module("torch")
    device = torch.device(device_name)
    torch.manual_seed(inference_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(inference_seed)
    mean = np.asarray(target_standardizer.mean, dtype=np.float64)
    scale = np.asarray(target_standardizer.standard_deviation, dtype=np.float64)
    marginal_parts = []
    joint_parts = []
    rank_parts: Dict[str, list[np.ndarray]] = {name: [] for name in SBC_STATISTICS}
    identifiers: list[str] = []
    cells: list[str] = []
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
            context = model.encode_context(moved)
            truth_log_density = model.flow.log_prob(
                standardized_truth, context=context
            )
            truth_density = truth_log_density.detach().cpu().numpy().astype(np.float64)
            batch_ids = tuple(str(value["physical_system_id"]) for value in metadata)
            case_count = len(target)
            marginal_less = np.zeros((case_count, 2), dtype=np.int64)
            marginal_equal = np.zeros((case_count, 2), dtype=np.int64)
            joint_greater_equal = np.zeros(case_count, dtype=np.int64)
            rank_less = {
                name: np.zeros(case_count, dtype=np.int64) for name in SBC_STATISTICS
            }
            rank_equal = {
                name: np.zeros(case_count, dtype=np.int64) for name in SBC_STATISTICS
            }
            draws_remaining = posterior_draw_count
            while draws_remaining:
                chunk = min(posterior_draw_chunk_size, draws_remaining)
                sampled = model.sample_from_context(chunk, context)
                sampled_log_density = model.sample_log_prob_from_context(
                    sampled, context
                )
                standardized_draws = (
                    sampled.detach().cpu().numpy().astype(np.float64)
                )
                draws = standardized_draws * scale[None, None, :] + mean[None, None, :]
                draw_density = (
                    sampled_log_density.detach().cpu().numpy().astype(np.float64)
                )
                if not (
                    np.all(np.isfinite(draws))
                    and np.all(np.isfinite(draw_density))
                    and np.all(np.isfinite(truth_density))
                ):
                    raise FloatingPointError("score extraction produced NaN or Inf")
                marginal_less += np.sum(draws < target[:, None, :], axis=1)
                marginal_equal += np.sum(draws == target[:, None, :], axis=1)
                joint_greater_equal += np.sum(
                    draw_density >= truth_density[:, None], axis=1
                )
                if split is SplitName.SBC_DIAGNOSTIC:
                    transformed_draws = {
                        "log_abs_mu_primary": draws[:, :, 0],
                        "log_abs_mu_secondary": draws[:, :, 1],
                        "log_abs_mu_sum": draws[:, :, 0] + draws[:, :, 1],
                        "log_abs_mu_difference": draws[:, :, 0] - draws[:, :, 1],
                        "joint_log_density_rank": draw_density,
                    }
                    transformed_truth = {
                        "log_abs_mu_primary": target[:, 0],
                        "log_abs_mu_secondary": target[:, 1],
                        "log_abs_mu_sum": target[:, 0] + target[:, 1],
                        "log_abs_mu_difference": target[:, 0] - target[:, 1],
                        "joint_log_density_rank": truth_density,
                    }
                    for name in SBC_STATISTICS:
                        rank_less[name] += np.sum(
                            transformed_draws[name]
                            < transformed_truth[name][:, None],
                            axis=1,
                        )
                        rank_equal[name] += np.sum(
                            transformed_draws[name]
                            == transformed_truth[name][:, None],
                            axis=1,
                        )
                draws_remaining -= chunk
            pit = (
                marginal_less + 0.5 * marginal_equal
            ) / posterior_draw_count
            marginal_parts.append(2.0 * np.abs(pit - 0.5))
            joint_parts.append(joint_greater_equal / posterior_draw_count)
            if split is SplitName.SBC_DIAGNOSTIC:
                for name in SBC_STATISTICS:
                    rank_parts[name].append(
                        np.asarray(
                            [
                                randomized_rank_from_counts(
                                    int(rank_less[name][index]),
                                    int(rank_equal[name][index]),
                                    physical_system_id=identifier,
                                    statistic=name,
                                )
                                for index, identifier in enumerate(batch_ids)
                            ],
                            dtype=np.int64,
                        )
                    )
            identifiers.extend(batch_ids)
            cells.extend(str(value["em_cell"]) for value in metadata)
    result = {
        "marginal_scores": np.concatenate(marginal_parts, axis=0),
        "joint_scores": np.concatenate(joint_parts, axis=0),
        "physical_system_ids": np.asarray(identifiers, dtype=np.str_),
        "em_cells": np.asarray(cells, dtype=np.str_),
    }
    if split is SplitName.SBC_DIAGNOSTIC:
        result.update(
            {
                f"rank_{name}": np.concatenate(rank_parts[name], axis=0)
                for name in SBC_STATISTICS
            }
        )
    return result


def run_authorized_score_inference(
    root: Path,
    *,
    authorization_path: Path,
    publication_root: Path,
    checkpoint_path: Path,
    environment_lock_path: Path,
    psd_root: Path,
    output_path: Path,
    split: SplitName,
    seed: int,
    device_name: str,
) -> Mapping[str, Any]:
    """Produce one immutable score artifact; never fit a map or run an SBC test."""

    gate = validate_score_inference_authorization(
        root,
        authorization_path=authorization_path,
        split=split,
        seed=seed,
        checkpoint_path=checkpoint_path,
        publication_root=publication_root,
        output_path=output_path,
    )
    authorization = gate["authorization"]
    immutable = authorization["immutable_inference"]
    if environment_lock_path.resolve() != Path(
        str(immutable["environment_lock_path"])
    ).resolve() or _sha256(environment_lock_path) != immutable["environment_lock_sha256"]:
        raise TrainingGateError("score-inference environment path or hash mismatch")
    model_config = gate["model"]
    _validate_runtime_versions(model_config)
    load_input_policy(root)
    curves = _verified_curves(model_config, psd_root)
    publication = authorization["development_publication"]
    dataset_key = (
        "calibration_dataset_id"
        if split is SplitName.CALIBRATION_FIT
        else "sbc_dataset_id"
    )
    expected_count = 4096 if split is SplitName.CALIBRATION_FIT else 2048
    full_dataset = PublishedStageADataset(
        publication_root / str(publication[dataset_key]),
        expected_split=split,
        detector_curves=curves,
        expected_total_pairs=expected_count,
    )
    selected_ids: Sequence[str] | None = None
    if split is SplitName.SBC_DIAGNOSTIC:
        selected_ids = deterministic_sbc_subset(
            full_dataset.physical_system_ids(),
            root_seed=int(authorization["inference_contract"]["sbc_subset_seed"]),
            count=1024,
        )
        full_dataset = PublishedStageADataset(
            publication_root / str(publication[dataset_key]),
            expected_split=split,
            detector_curves=curves,
            expected_total_pairs=expected_count,
            selected_physical_system_ids=selected_ids,
        )
    torch = importlib.import_module("torch")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    identity = state.get("identity", {})
    if (
        int(identity.get("seed", -1)) != seed
        or not checkpoint_training_rung_is_authorized(identity, authorization)
        or identity.get("model_configuration_hash")
        != model_configuration_hash(model_config)
    ):
        raise TrainingGateError("selected checkpoint run identity mismatch")
    standardizer_value = {
        "input_standardizer": state["input_standardizer"],
        "target_standardizer": state["target_standardizer"],
        "input_standardizer_sha256": identity["input_standardizer_sha256"],
        "target_standardizer_sha256": identity["target_standardizer_sha256"],
    }
    input_standardizer, target_standardizer = _load_standardizers(standardizer_value)
    if (
        standardizer_hash(input_standardizer) != identity["input_standardizer_sha256"]
        or standardizer_hash(target_standardizer)
        != identity["target_standardizer_sha256"]
    ):
        raise TrainingGateError("checkpoint standardizer identity mismatch")
    model = importlib.import_module("gwlens_mm.training.model").build_probe_model(
        model_config, seed=seed
    )
    model.load_state_dict(state["model"])
    standardized = StandardizedCalibrationSBCDataset(
        full_dataset, input_standardizer
    )
    contract = authorization["inference_contract"]
    draw_count = (
        int(contract["calibration_posterior_draws_per_case"])
        if split is SplitName.CALIBRATION_FIT
        else int(contract["sbc_posterior_draws_per_replicate"])
    )
    draw_chunk_size = int(contract["posterior_draw_chunk_size"])
    if not 1 <= draw_chunk_size <= min(512, draw_count):
        raise TrainingGateError("posterior draw chunk size violates the memory cap")
    inference_seed = score_inference_seed(contract, split=split, model_seed=seed)
    loader = _data_loader(
        standardized,
        batch_size=int(contract["physical_batch_size"]),
        seed=inference_seed,
        device_name=device_name,
    )
    payload = _score_batches(
        model,
        loader,
        split=split,
        target_standardizer=target_standardizer,
        posterior_draw_count=draw_count,
        posterior_draw_chunk_size=draw_chunk_size,
        inference_seed=inference_seed,
        device_name=device_name,
    )
    required_count = 4096 if split is SplitName.CALIBRATION_FIT else 1024
    if payload["marginal_scores"].shape != (required_count, 2):
        raise TrainingGateError("score artifact has the wrong frozen case count")
    identifiers = tuple(str(value) for value in payload["physical_system_ids"])
    if len(identifiers) != required_count or len(set(identifiers)) != required_count:
        raise TrainingGateError("score artifact physical-system IDs are invalid")
    if split is SplitName.CALIBRATION_FIT:
        cells = tuple(str(value) for value in payload["em_cells"])
        if len(set(cells)) != 8 or any(cells.count(value) != 512 for value in set(cells)):
            raise TrainingGateError("calibration score artifact is not balanced by EM cell")
    payload.update(
        {
            "split": np.asarray(split.value, dtype=np.str_),
            "model_seed": np.asarray(seed, dtype=np.int64),
            "architecture_id": np.asarray(gate["architecture_id"], dtype=np.str_),
            "checkpoint_sha256": np.asarray(_sha256(checkpoint_path), dtype=np.str_),
            "publication_manifest_sha256": np.asarray(
                _sha256(publication_root / "dataset_manifest.json"), dtype=np.str_
            ),
            "inference_commit": np.asarray(gate["inference_commit"], dtype=np.str_),
        }
    )
    _atomic_npz(output_path, payload)
    summary = {
        "status": "completed_score_extraction_only",
        "split": split.value,
        "model_seed": seed,
        "architecture_id": gate["architecture_id"],
        "case_count": required_count,
        "posterior_draw_count": draw_count,
        "score_artifact_path": str(output_path),
        "score_artifact_sha256": _sha256(output_path),
        "physical_system_ids_sha256": hashlib.sha256(
            canonical_json(sorted(identifiers)).encode()
        ).hexdigest(),
        "calibration_map_fitted": False,
        "sbc_statistical_test_executed": False,
        "model_retrained_or_tuned": False,
        "final_evaluation_accessed": False,
    }
    _atomic_json(output_path.with_suffix(".summary.json"), summary)
    return summary
