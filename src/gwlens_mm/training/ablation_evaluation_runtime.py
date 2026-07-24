"""Fail-closed runtime for separately calibrated IID input ablations.

This module shares the primary calibration and final-inference kernels.  It
does not duplicate schema access or posterior-score mathematics.  Scientific
execution is possible only with a future reviewed authorization produced by
``ablation_evaluation_authorization``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np

from ..config import load_yaml
from ..provenance import configuration_hash
from ..schema import SplitName
from .ablation_evaluation import (
    ABLATION_EVALUATION_CONFIG_HASH,
    ABLATION_VIEWS,
    CALIBRATION_COUNT,
    IID_COUNT,
    MODEL_SEEDS,
    AblatedCalibrationDataset,
    AblatedIIDDataset,
    fit_ablation_calibration_map,
    paired_ablation_iid_comparison,
    summarize_ablation_iid_scores,
    validate_matching_ablation_map,
)
from .ablation_evaluation_authorization import (
    CALIBRATION_AUTHORIZATION_STATUS,
    IID_AUTHORIZATION_STATUS,
)
from .ablations import ablation_model_configuration
from .calibration_inference import (
    _atomic_json,
    _atomic_npz,
    _data_loader,
    _score_batches,
)
from .contracts import TrainingGateError, model_configuration_hash
from .data import PublishedStageADataset
from .engine import standardizer_hash
from .features import load_input_policy
from .final_inference import (
    POSTERIOR_DRAW_COUNT,
    SealedFinalNamespaceDataset,
    StandardizedFinalNamespaceDataset,
    _final_data_loader,
    _score_final_batches,
    resolve_sealed_final_publication,
)
from .rung65 import _load_standardizers
from .runner import (
    _validate_runtime_versions,
    _verified_curves,
    _verify_training_checkout,
)
from .terminal131 import TRAIN_131K_COUNT
from .terminal_downstream import checkpoint_training_rung_is_authorized

RUNTIME_CONFIG_PATH = "configs/inference/phase7_ablation_evaluation.yaml"
RUNTIME_CONFIG_HASH = (
    "1fb19fe9bfcf451919196b0510fad471c507ad5220bbc1410ebd196d00b20dcd"
)
RUNTIME_STACK_AUTHORIZATION = (
    "configs/execution/"
    "phase7_ablation_evaluation_runtime_stack_authorization.yaml"
)
APPROVED_REMOTE_ROOT = Path("/root/autodl-tmp/lensing-4")


class AblationEvaluationRuntimeError(TrainingGateError):
    """Raised when future ablation execution crosses its exact release."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AblationEvaluationRuntimeError(f"expected JSON mapping: {path}")
    return value


def _load_npz(path: Path) -> Mapping[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _remote_path(path: Path, *, name: str, require_file: bool = False) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(APPROVED_REMOTE_ROOT):
        raise AblationEvaluationRuntimeError(
            f"{name} escaped the AutoDL project root"
        )
    if require_file and not resolved.is_file():
        raise AblationEvaluationRuntimeError(f"{name} is absent")
    return resolved


def load_ablation_evaluation_runtime_contract(
    root: Path,
) -> Mapping[str, Any]:
    """Validate the frozen inference settings and implementation-only gate."""

    config = load_yaml(root / RUNTIME_CONFIG_PATH)
    if (
        configuration_hash(config) != RUNTIME_CONFIG_HASH
        or config.get("status") != "frozen_execution_disabled"
        or int(config.get("locked_training_rung", -1)) != TRAIN_131K_COUNT
        or config.get("ablation_views") != list(ABLATION_VIEWS)
        or config.get("model_seeds") != list(MODEL_SEEDS)
        or config.get("best_seed_selection_authorized") is not False
        or config.get("parent_contract", {}).get("canonical_hash")
        != ABLATION_EVALUATION_CONFIG_HASH
    ):
        raise AblationEvaluationRuntimeError(
            "ablation runtime configuration drifted"
        )
    observed_seeds: list[int] = []
    for section, count in (("calibration", CALIBRATION_COUNT), ("iid", IID_COUNT)):
        contract = config.get(section, {})
        if (
            int(contract.get("accepted_case_count", -1)) != count
            or int(contract.get("posterior_draws_per_case", -1))
            != POSTERIOR_DRAW_COUNT
            or not 1 <= int(contract.get("physical_batch_size", 0)) <= 32
            or not 1
            <= int(contract.get("posterior_draw_chunk_size", 0))
            <= 512
        ):
            raise AblationEvaluationRuntimeError(
                f"ablation {section} execution contract drifted"
            )
        roots = contract.get("root_seed_by_view_and_model_seed", {})
        if set(roots) != set(ABLATION_VIEWS):
            raise AblationEvaluationRuntimeError(
                f"ablation {section} view seed domains drifted"
            )
        for view in ABLATION_VIEWS:
            values = roots.get(view, {})
            if set(values) != {str(seed) for seed in MODEL_SEEDS}:
                raise AblationEvaluationRuntimeError(
                    f"ablation {section} model seed domains drifted"
                )
            observed_seeds.extend(int(values[str(seed)]) for seed in MODEL_SEEDS)
    if len(observed_seeds) != 12 or len(set(observed_seeds)) != 12:
        raise AblationEvaluationRuntimeError(
            "ablation calibration and IID seed domains collide"
        )
    if any(value is not False for value in config.get("execution", {}).values()):
        raise AblationEvaluationRuntimeError(
            "ablation runtime configuration opened execution"
        )
    stack = load_yaml(root / RUNTIME_STACK_AUTHORIZATION)
    if stack.get("authorization_status") != (
        "authorized_implementation_and_synthetic_fixture_only"
    ):
        raise AblationEvaluationRuntimeError(
            "ablation runtime implementation gate is absent"
        )
    frozen = stack.get("frozen_execution_config", {})
    if (
        frozen.get("path") != RUNTIME_CONFIG_PATH
        or frozen.get("canonical_hash") != RUNTIME_CONFIG_HASH
        or frozen.get("parent_rc8_hash") != ABLATION_EVALUATION_CONFIG_HASH
    ):
        raise AblationEvaluationRuntimeError(
            "ablation runtime implementation gate drifted"
        )
    flags = stack.get("authorization", {})
    allowed = {
        "typed_calibration_runtime_implementation_authorized",
        "typed_iid_runtime_implementation_authorized",
        "fail_closed_cli_implementation_authorized",
        "synthetic_fixture_tests_authorized",
    }
    if any(flags.get(name) is not True for name in allowed) or any(
        value is not False for name, value in flags.items() if name not in allowed
    ):
        raise AblationEvaluationRuntimeError(
            "ablation runtime implementation opened scientific execution"
        )
    return config


def _identity_item(
    authorization: Mapping[str, Any],
    *,
    view: str,
    seed: int,
) -> Mapping[str, Any]:
    if view not in ABLATION_VIEWS or seed not in MODEL_SEEDS:
        raise AblationEvaluationRuntimeError("ablation view or seed is invalid")
    checkpoints = authorization.get("ablation_checkpoints", {})
    item = checkpoints.get(view, {}).get(str(seed), {})
    if not isinstance(item, dict):
        raise AblationEvaluationRuntimeError("ablation checkpoint identity is absent")
    return item


def _validate_immutable_runtime(
    root: Path,
    authorization_path: Path,
    authorization: Mapping[str, Any],
    environment_lock_path: Path,
) -> str:
    immutable = authorization.get("immutable_inference", {})
    commit = str(immutable.get("git_commit", ""))
    environment = _remote_path(
        environment_lock_path, name="ablation environment", require_file=True
    )
    if (
        environment
        != Path(str(immutable.get("environment_lock_path", ""))).resolve()
        or _sha256(environment) != immutable.get("environment_lock_sha256")
    ):
        raise AblationEvaluationRuntimeError(
            "ablation inference environment identity changed"
        )
    _verify_training_checkout(root, commit, authorization_path)
    return commit


def _load_ablation_checkpoint(
    root: Path,
    *,
    authorization: Mapping[str, Any],
    item: Mapping[str, Any],
    architecture_id: str,
    view: str,
    seed: int,
) -> tuple[Any, Any, Any, Mapping[str, Any]]:
    checkpoint_path = _remote_path(
        Path(str(item.get("checkpoint_path", ""))),
        name="ablation checkpoint",
        require_file=True,
    )
    if _sha256(checkpoint_path) != item.get("checkpoint_sha256"):
        raise AblationEvaluationRuntimeError("ablation checkpoint hash changed")
    model_config = ablation_model_configuration(
        root, architecture_id=architecture_id, view=view
    )
    if model_configuration_hash(model_config) != item.get(
        "model_configuration_hash"
    ):
        raise AblationEvaluationRuntimeError(
            "ablation model configuration hash changed"
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
        raise AblationEvaluationRuntimeError(
            "ablation checkpoint run identity changed"
        )
    standardizers = {
        "input_standardizer": state["input_standardizer"],
        "target_standardizer": state["target_standardizer"],
        "input_standardizer_sha256": identity["input_standardizer_sha256"],
        "target_standardizer_sha256": identity["target_standardizer_sha256"],
    }
    input_standardizer, target_standardizer = _load_standardizers(standardizers)
    if (
        standardizer_hash(input_standardizer)
        != identity["input_standardizer_sha256"]
        or standardizer_hash(target_standardizer)
        != identity["target_standardizer_sha256"]
    ):
        raise AblationEvaluationRuntimeError(
            "ablation checkpoint standardizer identity changed"
        )
    model = importlib.import_module(
        "gwlens_mm.training.model"
    ).build_probe_model(model_config, seed=seed)
    model.load_state_dict(state["model"])
    return model, input_standardizer, target_standardizer, model_config


def validate_ablation_calibration_execution_gate(
    root: Path,
    *,
    authorization_path: Path,
    publication_root: Path,
    checkpoint_path: Path,
    environment_lock_path: Path,
    score_output_path: Path,
    map_output_path: Path,
    view: str,
    seed: int,
) -> Mapping[str, Any]:
    """Validate one future view/seed calibration job without opening arrays."""

    contract = load_ablation_evaluation_runtime_contract(root)
    authorization = load_yaml(authorization_path)
    if authorization.get("authorization_status") != (
        CALIBRATION_AUTHORIZATION_STATUS
    ):
        raise AblationEvaluationRuntimeError(
            "ablation calibration execution is not authorized"
        )
    flags = authorization.get("authorization", {})
    required_true = {
        "scientific_checkpoint_access_authorized",
        "calibration_fit_data_access_authorized",
        "ablation_calibration_score_execution_authorized",
        "ablation_calibration_map_fit_authorized",
    }
    if any(flags.get(name) is not True for name in required_true) or any(
        flags.get(name) is not False
        for name in (
            "iid_unsealing_authorized",
            "iid_inference_or_comparison_authorized",
            "model_training_or_tuning_authorized",
            "sbc_authorized",
            "gwosc_gwtc_access_authorized",
        )
    ):
        raise AblationEvaluationRuntimeError(
            "ablation calibration authorization crossed its boundary"
        )
    item = _identity_item(authorization, view=view, seed=seed)
    if checkpoint_path.resolve() != Path(str(item.get("checkpoint_path", ""))).resolve():
        raise AblationEvaluationRuntimeError("ablation checkpoint path changed")
    publication = authorization.get("calibration_publication", {})
    publication_root = _remote_path(
        publication_root, name="calibration publication"
    )
    manifest = publication_root / "dataset_manifest.json"
    if (
        publication_root
        != Path(str(publication.get("parent_root", ""))).resolve()
        or not manifest.is_file()
        or _sha256(manifest) != publication.get("parent_manifest_sha256")
        or int(publication.get("calibration_fit_accepted_count", -1))
        != CALIBRATION_COUNT
    ):
        raise AblationEvaluationRuntimeError(
            "ablation calibration publication changed"
        )
    primary = authorization.get("primary_same_seed_calibration_scores", {}).get(
        str(seed), {}
    )
    primary_path = _remote_path(
        Path(str(primary.get("path", ""))),
        name="same-seed primary calibration score",
        require_file=True,
    )
    if _sha256(primary_path) != primary.get("sha256"):
        raise AblationEvaluationRuntimeError(
            "primary calibration score identity changed"
        )
    expected_score = Path(
        str(
            authorization.get("calibration_score_outputs", {})
            .get(view, {})
            .get(str(seed), "")
        )
    ).resolve()
    expected_map = Path(
        str(
            authorization.get("calibration_map_outputs", {})
            .get(view, {})
            .get(str(seed), "")
        )
    ).resolve()
    for path, expected, name in (
        (score_output_path, expected_score, "ablation calibration score"),
        (map_output_path, expected_map, "ablation calibration map"),
    ):
        output = _remote_path(path, name=name)
        if output != expected or output.exists():
            raise AblationEvaluationRuntimeError(
                f"{name} identity is invalid or reused"
            )
    summary_path = expected_map.with_name("run_summary.json")
    if summary_path.exists():
        raise AblationEvaluationRuntimeError(
            "ablation calibration summary identity exists"
        )
    commit = _validate_immutable_runtime(
        root, authorization_path, authorization, environment_lock_path
    )
    selected = authorization.get("selected_architecture", {})
    if int(selected.get("locked_training_rung", -1)) != TRAIN_131K_COUNT:
        raise AblationEvaluationRuntimeError(
            "ablation calibration rung is not terminal"
        )
    return {
        "authorization": authorization,
        "contract": contract,
        "item": item,
        "architecture_id": str(selected.get("architecture_id", "")),
        "inference_commit": commit,
        "publication": publication,
        "primary_path": primary_path,
    }


def run_authorized_ablation_calibration(
    root: Path,
    *,
    authorization_path: Path,
    publication_root: Path,
    checkpoint_path: Path,
    environment_lock_path: Path,
    psd_root: Path,
    score_output_path: Path,
    map_output_path: Path,
    view: str,
    seed: int,
    device_name: str = "cuda",
) -> Mapping[str, Any]:
    """Score and fit one independent ablation calibration map."""

    gate = validate_ablation_calibration_execution_gate(
        root,
        authorization_path=authorization_path,
        publication_root=publication_root,
        checkpoint_path=checkpoint_path,
        environment_lock_path=environment_lock_path,
        score_output_path=score_output_path,
        map_output_path=map_output_path,
        view=view,
        seed=seed,
    )
    authorization = gate["authorization"]
    model, input_standardizer, target_standardizer, model_config = (
        _load_ablation_checkpoint(
            root,
            authorization=authorization,
            item=gate["item"],
            architecture_id=gate["architecture_id"],
            view=view,
            seed=seed,
        )
    )
    _validate_runtime_versions(model_config)
    load_input_policy(root)
    curves = _verified_curves(model_config, psd_root)
    publication = gate["publication"]
    dataset = PublishedStageADataset(
        publication_root / str(publication["calibration_dataset_id"]),
        expected_split=SplitName.CALIBRATION_FIT,
        detector_curves=curves,
        expected_total_pairs=CALIBRATION_COUNT,
    )
    standardized = AblatedCalibrationDataset(
        dataset, input_standardizer, view
    )
    contract = gate["contract"]["calibration"]
    inference_seed = int(
        contract["root_seed_by_view_and_model_seed"][view][str(seed)]
    )
    loader = _data_loader(
        standardized,
        batch_size=int(contract["physical_batch_size"]),
        seed=inference_seed,
        device_name=device_name,
    )
    payload = dict(
        _score_batches(
            model,
            loader,
            split=SplitName.CALIBRATION_FIT,
            target_standardizer=target_standardizer,
            posterior_draw_count=POSTERIOR_DRAW_COUNT,
            posterior_draw_chunk_size=int(
                contract["posterior_draw_chunk_size"]
            ),
            inference_seed=inference_seed,
            device_name=device_name,
        )
    )
    identifiers = tuple(str(value) for value in payload["physical_system_ids"])
    if len(identifiers) != CALIBRATION_COUNT or len(set(identifiers)) != (
        CALIBRATION_COUNT
    ):
        raise AblationEvaluationRuntimeError(
            "ablation calibration case identity is invalid"
        )
    checkpoint_hash = str(gate["item"]["checkpoint_sha256"])
    payload.update(
        {
            "split": np.asarray(SplitName.CALIBRATION_FIT.value, dtype=np.str_),
            "ablation_view": np.asarray(view, dtype=np.str_),
            "model_seed": np.asarray(seed, dtype=np.int64),
            "architecture_id": np.asarray(
                gate["architecture_id"], dtype=np.str_
            ),
            "checkpoint_sha256": np.asarray(
                checkpoint_hash, dtype=np.str_
            ),
            "publication_manifest_sha256": np.asarray(
                gate["publication"]["parent_manifest_sha256"], dtype=np.str_
            ),
            "inference_commit": np.asarray(
                gate["inference_commit"], dtype=np.str_
            ),
        }
    )
    _atomic_npz(score_output_path, payload)
    primary_score = _load_npz(gate["primary_path"])
    primary_identifiers = tuple(
        str(value) for value in primary_score.get("physical_system_ids", ())
    )
    calibration_map = fit_ablation_calibration_map(
        payload,
        view=view,
        model_seed=seed,
        checkpoint_sha256=checkpoint_hash,
        primary_calibration_case_ids=primary_identifiers,
    )
    _atomic_json(map_output_path, calibration_map)
    summary = {
        "status": "completed_ablation_calibration_score_and_map",
        "view": view,
        "model_seed": seed,
        "architecture_id": gate["architecture_id"],
        "calibration_case_count": CALIBRATION_COUNT,
        "posterior_draw_count": POSTERIOR_DRAW_COUNT,
        "checkpoint_sha256": checkpoint_hash,
        "calibration_score_path": str(score_output_path.resolve()),
        "calibration_score_sha256": _sha256(score_output_path),
        "calibration_map_path": str(map_output_path.resolve()),
        "calibration_map_sha256": _sha256(map_output_path),
        "same_cases_as_primary_calibration": True,
        "primary_calibration_map_reused": False,
        "map_shared_across_views_or_seeds": False,
        "iid_accessed": False,
        "model_retrained_or_tuned": False,
        "sbc_executed": False,
        "gwosc_gwtc_accessed": False,
    }
    _atomic_json(map_output_path.with_name("run_summary.json"), summary)
    return summary


def validate_ablation_iid_execution_gate(
    root: Path,
    *,
    authorization_path: Path,
    publication_root: Path,
    environment_lock_path: Path,
    score_output_path: Path,
    comparison_output_path: Path,
    view: str,
    seed: int,
) -> Mapping[str, Any]:
    """Validate one future ablation IID job without opening its inputs."""

    contract = load_ablation_evaluation_runtime_contract(root)
    authorization = load_yaml(authorization_path)
    if authorization.get("authorization_status") != IID_AUTHORIZATION_STATUS:
        raise AblationEvaluationRuntimeError(
            "ablation IID execution is not authorized"
        )
    flags = authorization.get("authorization", {})
    required_true = {
        "final_iid_unsealing_authorized",
        "ablation_checkpoint_access_authorized",
        "matching_ablation_calibration_map_access_authorized",
        "primary_same_seed_iid_score_access_authorized",
        "ablation_iid_inference_authorized",
        "paired_comparison_execution_authorized",
    }
    if any(flags.get(name) is not True for name in required_true) or any(
        flags.get(name) is not False
        for name in (
            "calibration_refit_authorized",
            "sbc_authorized",
            "model_training_or_tuning_authorized",
            "non_iid_ablation_inference_authorized",
            "result_driven_retraining_authorized",
            "gwosc_gwtc_access_authorized",
        )
    ):
        raise AblationEvaluationRuntimeError(
            "ablation IID authorization crossed its boundary"
        )
    item = _identity_item(authorization, view=view, seed=seed)
    maps = authorization.get("ablation_calibration_maps", {})
    map_item = maps.get(view, {}).get(str(seed), {})
    map_path = _remote_path(
        Path(str(map_item.get("calibration_map_path", ""))),
        name="matching ablation calibration map",
        require_file=True,
    )
    if _sha256(map_path) != map_item.get("calibration_map_sha256"):
        raise AblationEvaluationRuntimeError("ablation calibration map changed")
    primary = authorization.get("primary_same_seed_iid_scores", {}).get(
        str(seed), {}
    )
    primary_path = _remote_path(
        Path(str(primary.get("path", ""))),
        name="same-seed primary IID score",
        require_file=True,
    )
    if _sha256(primary_path) != primary.get("sha256"):
        raise AblationEvaluationRuntimeError("primary IID score changed")
    publication = authorization.get("sealed_publication", {})
    publication_root = _remote_path(
        publication_root, name="sealed final publication"
    )
    if publication_root != Path(str(publication.get("parent_root", ""))).resolve():
        raise AblationEvaluationRuntimeError(
            "sealed final publication path changed"
        )
    expected_score = Path(
        str(
            authorization.get("ablation_iid_score_outputs", {})
            .get(view, {})
            .get(str(seed), "")
        )
    ).resolve()
    expected_comparison = Path(
        str(
            authorization.get("paired_comparison_outputs", {})
            .get(view, {})
            .get(str(seed), "")
        )
    ).resolve()
    for path, expected, name in (
        (score_output_path, expected_score, "ablation IID score"),
        (
            comparison_output_path,
            expected_comparison,
            "ablation IID comparison",
        ),
    ):
        output = _remote_path(path, name=name)
        if output != expected or output.exists():
            raise AblationEvaluationRuntimeError(
                f"{name} identity is invalid or reused"
            )
    if score_output_path.with_suffix(".summary.json").exists():
        raise AblationEvaluationRuntimeError(
            "ablation IID summary identity exists"
        )
    commit = _validate_immutable_runtime(
        root, authorization_path, authorization, environment_lock_path
    )
    selected = authorization.get("selected_architecture", {})
    if int(selected.get("locked_training_rung", -1)) != TRAIN_131K_COUNT:
        raise AblationEvaluationRuntimeError("ablation IID rung is not terminal")
    return {
        "authorization": authorization,
        "contract": contract,
        "item": item,
        "map_item": map_item,
        "map_path": map_path,
        "primary_path": primary_path,
        "architecture_id": str(selected.get("architecture_id", "")),
        "inference_commit": commit,
        "namespace_id": str(authorization.get("iid_namespace_id", "")),
    }


def run_authorized_ablation_iid(
    root: Path,
    *,
    authorization_path: Path,
    publication_root: Path,
    environment_lock_path: Path,
    psd_root: Path,
    score_output_path: Path,
    comparison_output_path: Path,
    view: str,
    seed: int,
    device_name: str = "cuda",
) -> Mapping[str, Any]:
    """Run one calibrated IID ablation and same-seed paired comparison."""

    gate = validate_ablation_iid_execution_gate(
        root,
        authorization_path=authorization_path,
        publication_root=publication_root,
        environment_lock_path=environment_lock_path,
        score_output_path=score_output_path,
        comparison_output_path=comparison_output_path,
        view=view,
        seed=seed,
    )
    authorization = gate["authorization"]
    model, input_standardizer, target_standardizer, model_config = (
        _load_ablation_checkpoint(
            root,
            authorization=authorization,
            item=gate["item"],
            architecture_id=gate["architecture_id"],
            view=view,
            seed=seed,
        )
    )
    _validate_runtime_versions(model_config)
    load_input_policy(root)
    curves = _verified_curves(model_config, psd_root)
    publication = resolve_sealed_final_publication(root, publication_root)
    sealed_identity = authorization.get("sealed_publication", {})
    if (
        publication.manifest_sha256
        != sealed_identity.get("manifest_sha256")
        or publication.generator_commit
        != sealed_identity.get("generator_commit")
    ):
        raise AblationEvaluationRuntimeError(
            "sealed final publication identity changed"
        )
    namespace_id = gate["namespace_id"]
    if namespace_id not in publication.namespaces:
        raise AblationEvaluationRuntimeError(
            "authorized IID namespace is absent"
        )
    namespace = publication.namespaces[namespace_id]
    if (
        namespace.specification.split is not SplitName.IID_TEST
        or namespace.specification.accepted_count != IID_COUNT
    ):
        raise AblationEvaluationRuntimeError(
            "authorized final namespace is not exact IID"
        )
    dataset = AblatedIIDDataset(
        StandardizedFinalNamespaceDataset(
            SealedFinalNamespaceDataset(namespace, detector_curves=curves),
            input_standardizer,
        ),
        view,
    )
    calibration_map = _load_json(gate["map_path"])
    checkpoint_hash = str(gate["item"]["checkpoint_sha256"])
    validate_matching_ablation_map(
        calibration_map,
        view=view,
        model_seed=seed,
        checkpoint_sha256=checkpoint_hash,
    )
    contract = gate["contract"]["iid"]
    inference_seed = int(
        contract["root_seed_by_view_and_model_seed"][view][str(seed)]
    )
    loader = _final_data_loader(
        cast(StandardizedFinalNamespaceDataset, dataset),
        batch_size=int(contract["physical_batch_size"]),
        seed=inference_seed,
        device_name=device_name,
    )
    payload = dict(
        _score_final_batches(
            model,
            loader,
            target_standardizer=target_standardizer,
            calibration_map=calibration_map,
            inference_seed=inference_seed,
            draw_microbatch=int(contract["posterior_draw_chunk_size"]),
            device_name=device_name,
        )
    )
    identifiers = tuple(str(value) for value in payload["physical_system_ids"])
    if (
        len(identifiers) != IID_COUNT
        or len(set(identifiers)) != IID_COUNT
        or set(str(value) for value in payload["splits"])
        != {SplitName.IID_TEST.value}
    ):
        raise AblationEvaluationRuntimeError(
            "ablation IID score identity is invalid"
        )
    payload.update(
        {
            "ablation_view": np.asarray(view, dtype=np.str_),
            "model_seed": np.asarray(seed, dtype=np.int64),
            "architecture_id": np.asarray(
                gate["architecture_id"], dtype=np.str_
            ),
            "checkpoint_sha256": np.asarray(
                checkpoint_hash, dtype=np.str_
            ),
            "publication_manifest_sha256": np.asarray(
                publication.manifest_sha256, dtype=np.str_
            ),
            "calibration_map_sha256": np.asarray(
                gate["map_item"]["calibration_map_sha256"], dtype=np.str_
            ),
            "inference_commit": np.asarray(
                gate["inference_commit"], dtype=np.str_
            ),
        }
    )
    _atomic_npz(score_output_path, payload)
    primary = _load_npz(gate["primary_path"])
    score_summary = summarize_ablation_iid_scores(
        payload, view=view, model_seed=seed
    )
    comparison = paired_ablation_iid_comparison(
        primary,
        payload,
        view=view,
        model_seed=seed,
    )
    _atomic_json(comparison_output_path, comparison)
    summary = {
        "status": "completed_ablation_iid_inference_and_paired_comparison",
        "view": view,
        "model_seed": seed,
        "architecture_id": gate["architecture_id"],
        "iid_namespace_id": namespace_id,
        "iid_case_count": IID_COUNT,
        "posterior_draw_count": POSTERIOR_DRAW_COUNT,
        "checkpoint_sha256": checkpoint_hash,
        "calibration_map_sha256": gate["map_item"][
            "calibration_map_sha256"
        ],
        "ablation_iid_score_path": str(score_output_path.resolve()),
        "ablation_iid_score_sha256": _sha256(score_output_path),
        "paired_comparison_path": str(comparison_output_path.resolve()),
        "paired_comparison_sha256": _sha256(comparison_output_path),
        "descriptive_iid_summary": score_summary,
        "best_seed_selected": False,
        "calibration_refit": False,
        "model_retrained_or_tuned": False,
        "non_iid_ablation_executed": False,
        "result_can_trigger_retraining_or_tuning": False,
        "gwosc_gwtc_accessed": False,
    }
    _atomic_json(score_output_path.with_suffix(".summary.json"), summary)
    return summary


def dry_run_ablation_evaluation_runtime(root: Path) -> Mapping[str, Any]:
    """Expose the exact workload while proving all scientific access is off."""

    contract = load_ablation_evaluation_runtime_contract(root)
    return {
        "status": "implementation_ready_execution_closed",
        "runtime_config_hash": RUNTIME_CONFIG_HASH,
        "views": list(ABLATION_VIEWS),
        "model_seeds": list(MODEL_SEEDS),
        "calibration_job_count": 6,
        "calibration_case_count_per_job": CALIBRATION_COUNT,
        "iid_job_count": 6,
        "iid_case_count_per_job": IID_COUNT,
        "posterior_draws_per_case": POSTERIOR_DRAW_COUNT,
        "calibration_physical_batch_size": contract["calibration"][
            "physical_batch_size"
        ],
        "iid_physical_batch_size": contract["iid"]["physical_batch_size"],
        "scientific_checkpoint_accessed": False,
        "calibration_fit_data_accessed": False,
        "iid_data_unsealed": False,
        "primary_iid_score_accessed": False,
        "scientific_artifact_created": False,
        "model_trained_or_tuned": False,
        "gwosc_gwtc_accessed": False,
    }
