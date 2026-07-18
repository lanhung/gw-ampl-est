"""Fail-closed orchestration for the future authorized 16k/32k probe fits."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Tuple

from ..config import load_yaml
from ..schema import SplitName
from .contracts import (
    TrainingGateError,
    deterministic_probe_subset,
    model_configuration_hash,
    validate_scientific_training_gate,
)
from .data import (
    DevelopmentStageADataset,
    PublishedStageADataset,
    StageAPublication,
    StandardizedStageADataset,
    torch_collate,
    torch_development_collate,
)
from .engine import (
    DeterministicShardEpochSampler,
    TargetStandardizer,
    TrainingRunIdentity,
    authorized_probe_execution_evidence,
    evaluate_development_validation,
    membership_hash,
    optimization_batch_geometry,
    standardizer_hash,
    train_probe,
)
from .features import InputStandardizer, PreparedExample, load_input_policy
from .model import build_probe_model
from .whitening import ASDCurve


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _verify_training_checkout(
    root: Path, training_commit: str, authorization_path: Path
) -> None:
    if len(training_commit) != 40:
        raise TrainingGateError("training commit must be a full Git SHA")
    if not (root / ".git").is_dir():
        marker = root / "SYNCED_COMMIT"
        if not marker.is_file() or marker.read_text().strip() != training_commit:
            raise TrainingGateError("training checkout lacks the frozen commit marker")
        return
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head == training_commit:
        return
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", training_commit, head],
        cwd=root,
        check=False,
    )
    if ancestry.returncode != 0:
        raise TrainingGateError("frozen training commit is not an ancestor of checkout HEAD")
    relative_authorization = str(authorization_path.resolve().relative_to(root.resolve()))
    changed = subprocess.run(
        ["git", "diff", "--name-only", f"{training_commit}..{head}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    allowed = {relative_authorization, "AGENTS.md"}
    unexpected = sorted(set(changed) - allowed)
    if unexpected:
        raise TrainingGateError(
            "post-freeze checkout changed training code: " + ", ".join(unexpected)
        )


def _verified_curves(model: Mapping[str, Any], psd_root: Path) -> Mapping[str, ASDCurve]:
    specifications = model["inputs"]["gw"]["whitening"]["psd_curves"]
    return {
        detector: ASDCurve.from_file(
            psd_root / str(specification["file"]),
            expected_sha256=str(specification["sha256"]),
        )
        for detector, specification in specifications.items()
    }


def _validate_runtime_versions(model: Mapping[str, Any]) -> Mapping[str, str]:
    expected = model["software_candidate"]
    observed = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": importlib.metadata.version("torch"),
        "nflows": importlib.metadata.version("nflows"),
        "numpy": importlib.metadata.version("numpy"),
        "pandas": importlib.metadata.version("pandas"),
        "pyarrow": importlib.metadata.version("pyarrow"),
        "zarr": importlib.metadata.version("zarr"),
        "numcodecs": importlib.metadata.version("numcodecs"),
        "scipy": importlib.metadata.version("scipy"),
        "pyyaml": importlib.metadata.version("PyYAML"),
    }
    if observed != {name: str(value) for name, value in expected.items()}:
        raise TrainingGateError(
            f"training runtime versions differ from the frozen candidate: {observed}"
        )
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise TrainingGateError("authorized scientific probe training requires CUDA")
    return observed


def _metadata_examples(dataset: PublishedStageADataset) -> Iterable[PreparedExample]:
    for index in range(len(dataset)):
        yield dataset.metadata_example(index)


def fit_rung_standardizers(
    dataset: PublishedStageADataset,
) -> Tuple[InputStandardizer, TargetStandardizer]:
    """Fit rung-only statistics without loading any strain product."""

    input_standardizer = InputStandardizer.fit_iterable(_metadata_examples(dataset))
    target_standardizer = TargetStandardizer.fit_iterable(
        example.target for example in _metadata_examples(dataset)
    )
    return input_standardizer, target_standardizer


def _data_loader(
    dataset: StandardizedStageADataset,
    *,
    batch_size: int,
    seed: int,
    training: bool,
    worker_processes: int,
    device_name: str,
) -> Any:
    torch = importlib.import_module("torch")
    data = importlib.import_module("torch.utils.data")
    generator = torch.Generator()
    generator.manual_seed(seed)
    if training:
        entries = dataset.dataset.entries
        sampler: Optional[Any] = DeterministicShardEpochSampler(
            tuple(str(entry.path) for entry in entries), seed=seed
        )
    else:
        sampler = None
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=worker_processes,
        collate_fn=torch_collate,
        pin_memory=device_name.startswith("cuda"),
        persistent_workers=False,
        generator=generator,
    )


def _development_loader(
    dataset: DevelopmentStageADataset,
    *,
    batch_size: int,
    seed: int,
    worker_processes: int,
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
        num_workers=worker_processes,
        collate_fn=torch_development_collate,
        pin_memory=device_name.startswith("cuda"),
        persistent_workers=False,
        generator=generator,
    )


def _build_run_identity(
    *,
    publication: StageAPublication,
    model: Mapping[str, Any],
    training_commit: str,
    environment_sha256: str,
    final_evaluation_commitment_sha256: str,
    member_ids: Tuple[str, ...],
    input_standardizer: InputStandardizer,
    target_standardizer: TargetStandardizer,
    rung_count: int,
    seed: int,
) -> TrainingRunIdentity:
    return TrainingRunIdentity(
        model_configuration_hash=model_configuration_hash(model),
        training_code_commit=training_commit,
        training_environment_sha256=environment_sha256,
        train_manifest_sha256=publication.namespace_manifest_sha256["train"],
        validation_manifest_sha256=publication.namespace_manifest_sha256["validation"],
        final_evaluation_commitment_sha256=final_evaluation_commitment_sha256,
        membership_sha256=membership_hash(member_ids),
        input_standardizer_sha256=standardizer_hash(input_standardizer),
        target_standardizer_sha256=standardizer_hash(target_standardizer),
        training_rung_count=rung_count,
        seed=seed,
    )


def run_authorized_probe(
    root: Path,
    *,
    authorization_path: Path,
    stage_a_publication_root: Path,
    environment_lock_path: Path,
    psd_root: Path,
    output_root: Path,
    training_commit: str,
    rung_count: int,
    seed: int,
    device_name: str,
    resume_checkpoint: Optional[Path] = None,
    execute_optimizer: bool = True,
) -> Mapping[str, Any]:
    """Run one fit only after the future exact authorization passes every gate."""

    if rung_count not in (16384, 32768) or seed not in (0, 1, 2):
        raise TrainingGateError("probe execution is limited to 16k/32k and seeds 0/1/2")
    gate = validate_scientific_training_gate(
        root,
        authorization_path=authorization_path,
        stage_a_publication_root=stage_a_publication_root,
    )
    authorization = gate["authorization"]
    immutable = authorization.get("immutable_training", {})
    expected_environment_hash = str(immutable.get("environment_lock_sha256", ""))
    if immutable.get("git_commit") != training_commit:
        raise TrainingGateError("training authorization commit mismatch")
    expected_lock_path = (
        root / str(immutable.get("environment_lock_path", ""))
    ).resolve()
    if environment_lock_path.resolve() != expected_lock_path:
        raise TrainingGateError("training environment lock path mismatch")
    if _sha256_file(environment_lock_path) != expected_environment_hash:
        raise TrainingGateError("training environment lock hash mismatch")
    _verify_training_checkout(root, training_commit, authorization_path)
    configured_output = Path(str(authorization.get("training_output_root", ""))).resolve()
    if configured_output != output_root.resolve():
        raise TrainingGateError("training output differs from the authorized root")
    if not output_root.is_absolute() or not output_root.is_relative_to(
        Path("/root/autodl-tmp/lensing-4")
    ):
        raise TrainingGateError("training output escaped the AutoDL project root")
    if int(authorization.get("maximum_concurrent_fits", -1)) != 3:
        raise TrainingGateError("probe authorization must permit exactly three seed fits")
    model = load_yaml(root / "configs/models/phase4_probe_nsf.yaml")
    if immutable.get("model_configuration_hash") != model_configuration_hash(model):
        raise TrainingGateError("authorized model configuration hash mismatch")
    _validate_runtime_versions(model)
    load_input_policy(root)
    publication = gate["publication"]
    curves = _verified_curves(model, psd_root)
    full_train = PublishedStageADataset(
        publication.train_root,
        expected_split=SplitName.TRAIN,
        detector_curves=curves,
        expected_total_pairs=32768,
    )
    full_ids = full_train.physical_system_ids()
    if set(full_ids) & set(
        PublishedStageADataset(
            publication.validation_root,
            expected_split=SplitName.VALIDATION,
            detector_curves=curves,
            expected_total_pairs=6144,
        ).physical_system_ids()
    ):
        raise TrainingGateError("published train and validation groups overlap")
    root_seed = int(authorization.get("probe_membership_root_seed", -1))
    member_ids = (
        deterministic_probe_subset(full_ids, root_seed=root_seed)
        if rung_count == 16384
        else full_ids
    )
    train_dataset = PublishedStageADataset(
        publication.train_root,
        expected_split=SplitName.TRAIN,
        detector_curves=curves,
        expected_total_pairs=32768,
        selected_physical_system_ids=member_ids,
    )
    validation_dataset = PublishedStageADataset(
        publication.validation_root,
        expected_split=SplitName.VALIDATION,
        detector_curves=curves,
        expected_total_pairs=6144,
    )
    rung_directory = output_root / f"rung-{rung_count}"
    rung_preparation_path = rung_directory / "rung_preparation.json"
    if rung_preparation_path.is_file():
        rung_preparation = json.loads(rung_preparation_path.read_text(encoding="utf-8"))
        if (
            rung_preparation.get("status") != "ready_for_authorized_probe_fits"
            or tuple(rung_preparation.get("member_ids", ())) != member_ids
            or rung_preparation.get("stage_a_parent_manifest_sha256")
            != publication.manifest_sha256
            or rung_preparation.get("model_configuration_hash")
            != model_configuration_hash(model)
        ):
            raise TrainingGateError("rung preparation identity mismatch")
        input_value = rung_preparation["input_standardizer"]
        target_value = rung_preparation["target_standardizer"]
        astrometry_mean = [float(value) for value in input_value["astrometry_mean"]]
        astrometry_scale = [
            float(value) for value in input_value["astrometry_standard_deviation"]
        ]
        target_mean = [float(value) for value in target_value["mean"]]
        target_scale = [float(value) for value in target_value["standard_deviation"]]
        if not (
            len(astrometry_mean) == len(astrometry_scale) == 5
            and len(target_mean) == len(target_scale) == 2
        ):
            raise TrainingGateError("rung standardizer dimensions are invalid")
        input_standardizer = InputStandardizer(
            tuple(float(value) for value in input_value["scalar_mean"]),
            tuple(float(value) for value in input_value["scalar_standard_deviation"]),
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
            rung_preparation.get("input_standardizer_sha256")
            == standardizer_hash(input_standardizer)
            and rung_preparation.get("target_standardizer_sha256")
            == standardizer_hash(target_standardizer)
        ):
            raise TrainingGateError("rung standardizer hash mismatch")
    else:
        if execute_optimizer:
            raise TrainingGateError(
                "shared rung preprocessing must pass before concurrent seed fits"
            )
        input_standardizer, target_standardizer = fit_rung_standardizers(train_dataset)
        rung_preparation = {
            "status": "ready_for_authorized_probe_fits",
            "rung_count": rung_count,
            "member_count": len(member_ids),
            "member_ids": list(member_ids),
            "member_ids_sha256": membership_hash(member_ids),
            "stage_a_parent_manifest_sha256": publication.manifest_sha256,
            "train_namespace_manifest_sha256": publication.namespace_manifest_sha256[
                "train"
            ],
            "validation_namespace_manifest_sha256": publication.namespace_manifest_sha256[
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
        _atomic_json(rung_preparation_path, rung_preparation)
    if not execute_optimizer:
        return rung_preparation
    identity = _build_run_identity(
        publication=publication,
        model=model,
        training_commit=training_commit,
        environment_sha256=expected_environment_hash,
        final_evaluation_commitment_sha256=str(
            gate["final_evaluation_commitment_sha256"]
        ),
        member_ids=member_ids,
        input_standardizer=input_standardizer,
        target_standardizer=target_standardizer,
        rung_count=rung_count,
        seed=seed,
    )
    identity.validate()
    run_directory = rung_directory / f"seed-{seed}"
    if run_directory.exists() and resume_checkpoint is None:
        raise FileExistsError("probe run identity already has an output directory")
    preparation = {
        **authorized_probe_execution_evidence(identity),
        "stage_a_parent_manifest_sha256": publication.manifest_sha256,
        "member_count": len(member_ids),
        "member_ids_sha256": membership_hash(member_ids),
        "member_ids": list(member_ids),
        "input_standardizer": asdict(input_standardizer),
        "target_standardizer": asdict(target_standardizer),
        "scientific_probe_training_authorized": True,
        "model_selection_authorized": False,
        "calibration_authorized": False,
        "final_evaluation_accessed": False,
    }
    _atomic_json(run_directory / "run_preparation.json", preparation)
    worker_processes = int(authorization.get("data_loader_worker_processes", 0))
    if not 0 <= worker_processes <= 16:
        raise TrainingGateError("data-loader worker count is outside the reviewed range")
    standardized_train = StandardizedStageADataset(train_dataset, input_standardizer)
    standardized_validation = StandardizedStageADataset(
        validation_dataset, input_standardizer
    )
    _, physical_microbatch_size, _ = optimization_batch_geometry(
        model["optimization"]
    )
    train_loader = _data_loader(
        standardized_train,
        batch_size=physical_microbatch_size,
        seed=seed,
        training=True,
        worker_processes=worker_processes,
        device_name=device_name,
    )
    validation_loader = _data_loader(
        standardized_validation,
        batch_size=physical_microbatch_size,
        seed=seed,
        training=False,
        worker_processes=worker_processes,
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
        execution_evidence=preparation,
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
        worker_processes=worker_processes,
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
        "status": "completed_probe_fit_and_development_validation",
        "identity": asdict(identity),
        "training": training_summary,
        "development": development_summary,
        "calibration_accessed": False,
        "final_evaluation_accessed": False,
    }
    _atomic_json(run_directory / "run_summary.json", result)
    return result
