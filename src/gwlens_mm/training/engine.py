"""Deterministic optimization, checkpoint, and resume primitives."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np

from .contracts import TrainingGateError
from .features import InputStandardizer


@dataclass(frozen=True)
class TargetStandardizer:
    mean: Tuple[float, float]
    standard_deviation: Tuple[float, float]

    @classmethod
    def fit(cls, targets: np.ndarray) -> "TargetStandardizer":
        values = np.asarray(targets, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
            raise ValueError("target standardization requires at least two two-dimensional cases")
        if not np.all(np.isfinite(values)):
            raise ValueError("target standardization received NaN or Inf")
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0, ddof=0)
        if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
            raise ValueError("target standard deviations must be finite and positive")
        return cls(
            (float(mean[0]), float(mean[1])),
            (float(scale[0]), float(scale[1])),
        )

    @property
    def log_absolute_jacobian(self) -> float:
        return float(np.log(np.asarray(self.standard_deviation)).sum())

    def transform_numpy(self, targets: np.ndarray) -> np.ndarray:
        return (
            (np.asarray(targets) - np.asarray(self.mean))
            / np.asarray(self.standard_deviation)
        ).astype(np.float32)


@dataclass(frozen=True)
class TrainingRunIdentity:
    model_configuration_hash: str
    training_code_commit: str
    training_environment_sha256: str
    train_manifest_sha256: str
    validation_manifest_sha256: str
    final_evaluation_commitment_sha256: str
    membership_sha256: str
    input_standardizer_sha256: str
    target_standardizer_sha256: str
    training_rung_count: int
    seed: int

    def validate(self) -> None:
        for name, value in (
            ("model configuration", self.model_configuration_hash),
            ("training environment", self.training_environment_sha256),
            ("train manifest", self.train_manifest_sha256),
            ("validation manifest", self.validation_manifest_sha256),
            ("final-evaluation commitment", self.final_evaluation_commitment_sha256),
            ("membership", self.membership_sha256),
            ("input standardizer", self.input_standardizer_sha256),
            ("target standardizer", self.target_standardizer_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} identity must be lowercase SHA-256")
        if len(self.training_code_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.training_code_commit
        ):
            raise ValueError("training code identity must be a lowercase Git commit")
        if self.training_rung_count not in (16384, 32768, 65536):
            raise ValueError("training run uses an unregistered rung")
        if self.seed not in (0, 1, 2):
            raise ValueError("training seed is outside the preregistered set")


def membership_hash(physical_system_ids: Sequence[str]) -> str:
    if len(set(physical_system_ids)) != len(physical_system_ids):
        raise ValueError("membership contains duplicate IDs")
    digest = hashlib.sha256()
    for identifier in sorted(physical_system_ids):
        digest.update(identifier.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def standardizer_hash(standardizer: Any) -> str:
    """Hash an immutable input or target standardizer as canonical JSON."""

    payload = json.dumps(
        asdict(standardizer), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class DeterministicEpochSampler:
    """Epoch-addressed shuffle order that is independent of prior iteration."""

    def __init__(self, size: int, *, seed: int) -> None:
        if size <= 0:
            raise ValueError("sampler size must be positive")
        if seed not in (0, 1, 2):
            raise ValueError("sampler seed is outside the preregistered set")
        self.size = int(size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("sampler epoch must be nonnegative")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = np.random.default_rng(np.random.SeedSequence((self.seed, self.epoch)))
        return iter(int(value) for value in generator.permutation(self.size))

    def __len__(self) -> int:
        return self.size


def set_deterministic_training_seed(seed: int) -> None:
    if seed not in (0, 1, 2):
        raise ValueError("training seed is outside the preregistered set")
    torch = importlib.import_module("torch")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _move_batch(batch: Mapping[str, Any], device: Any) -> Dict[str, Any]:
    return {name: value.to(device) for name, value in batch.items()}


def _standardize_target(target: Any, standardizer: TargetStandardizer, torch: Any) -> Any:
    mean = torch.as_tensor(standardizer.mean, dtype=target.dtype, device=target.device)
    scale = torch.as_tensor(
        standardizer.standard_deviation, dtype=target.dtype, device=target.device
    )
    return (target - mean) / scale


def _atomic_torch_save(torch: Any, value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(dict(value), partial)
    os.replace(partial, path)


def _atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def _capture_rng_state(torch: Any) -> Mapping[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(torch: Any, state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise TrainingGateError("CUDA checkpoint cannot resume without CUDA")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _set_training_epoch(train_loader: Any, epoch: int) -> None:
    sampler = getattr(train_loader, "sampler", None)
    if not isinstance(sampler, DeterministicEpochSampler):
        raise TrainingGateError(
            "scientific train loader requires DeterministicEpochSampler"
        )
    sampler.set_epoch(epoch)


def _validate_execution_evidence(
    evidence: Mapping[str, Any], identity: TrainingRunIdentity
) -> None:
    if evidence.get("status") != "authorized_probe_training":
        raise TrainingGateError("training engine received no scientific authorization evidence")
    if evidence.get("final_evaluation_commitment_finalized") is not True:
        raise TrainingGateError("training engine requires a finalized evaluation commitment")
    if evidence.get("stage_a_publication_validated") is not True:
        raise TrainingGateError("training engine requires validated Stage A publication")
    if evidence.get("run_identity") != asdict(identity):
        raise TrainingGateError("authorization evidence does not bind the full run identity")


def train_probe(
    model: Any,
    train_loader: Iterable[Mapping[str, Any]],
    validation_loader: Iterable[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    identity: TrainingRunIdentity,
    input_standardizer: InputStandardizer,
    standardizer: TargetStandardizer,
    execution_evidence: Mapping[str, Any],
    output_directory: Path,
    device_name: str,
    resume_checkpoint: Optional[Path] = None,
) -> Mapping[str, Any]:
    """Fit one preregistered seed; unavailable until the later scientific gate."""

    identity.validate()
    if standardizer_hash(input_standardizer) != identity.input_standardizer_sha256:
        raise TrainingGateError("input standardizer does not match run identity")
    if standardizer_hash(standardizer) != identity.target_standardizer_sha256:
        raise TrainingGateError("target standardizer does not match run identity")
    _validate_execution_evidence(execution_evidence, identity)
    torch = importlib.import_module("torch")
    set_deterministic_training_seed(identity.seed)
    device = torch.device(device_name)
    model.to(device)
    optimization = config["optimization"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        betas=tuple(float(value) for value in optimization["betas"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    start_epoch = 0
    best_value = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    checkpoint_path = output_directory / "last.ckpt"
    best_path = output_directory / "best.ckpt"
    if resume_checkpoint is not None:
        state = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if state["identity"] != asdict(identity):
            raise TrainingGateError("checkpoint identity does not match this training run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_value = float(state["best_validation_nlp_per_dimension"])
        best_epoch = int(state["best_epoch"])
        epochs_without_improvement = int(state["epochs_without_improvement"])
        if state["input_standardizer"] != asdict(input_standardizer):
            raise TrainingGateError("checkpoint input standardizer mismatch")
        if state["target_standardizer"] != asdict(standardizer):
            raise TrainingGateError("checkpoint target standardizer mismatch")
        _restore_rng_state(torch, state["rng_state"])
    maximum_epochs = int(optimization["maximum_epochs"])
    patience = int(optimization["early_stopping_patience_epochs"])
    minimum_delta = float(optimization["early_stopping_minimum_delta"])
    history = []
    for epoch in range(start_epoch, maximum_epochs):
        _set_training_epoch(train_loader, epoch)
        model.train()
        training_sum = 0.0
        training_count = 0
        for batch in train_loader:
            moved = _move_batch(batch, device)
            target = moved.pop("target")
            optimizer.zero_grad(set_to_none=True)
            standardized = _standardize_target(target, standardizer, torch)
            log_probability = model.log_prob(standardized, moved)
            loss = -log_probability.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("training loss became nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(optimization["gradient_clip_norm"])
            )
            optimizer.step()
            training_sum += float(loss.detach()) * len(target)
            training_count += len(target)
        model.eval()
        validation_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for batch in validation_loader:
                moved = _move_batch(batch, device)
                target = moved.pop("target")
                standardized = _standardize_target(target, standardizer, torch)
                log_probability_z = model.log_prob(standardized, moved)
                log_probability_target = (
                    log_probability_z - standardizer.log_absolute_jacobian
                )
                validation_sum += float((-log_probability_target).sum())
                validation_count += len(target)
        if training_count == 0 or validation_count == 0:
            raise TrainingGateError("training and validation loaders must be nonempty")
        validation_nlp = validation_sum / validation_count / 2.0
        row = {
            "epoch": epoch,
            "training_nlp_standardized": training_sum / training_count,
            "validation_nlp_nat_per_target_dimension": validation_nlp,
        }
        history.append(row)
        improved = validation_nlp < best_value - minimum_delta
        if improved:
            best_value = validation_nlp
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        state = {
            "identity": asdict(identity),
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "input_standardizer": asdict(input_standardizer),
            "target_standardizer": asdict(standardizer),
            "rng_state": _capture_rng_state(torch),
            "best_validation_nlp_per_dimension": best_value,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
        }
        _atomic_torch_save(torch, state, checkpoint_path)
        if improved:
            _atomic_torch_save(torch, state, best_path)
        if epochs_without_improvement >= patience:
            break
    if best_epoch < 0:
        raise RuntimeError("training produced no finite best epoch")
    if bool(optimization.get("restore_best_epoch")):
        best_state = torch.load(best_path, map_location=device, weights_only=False)
        if best_state["identity"] != asdict(identity):
            raise TrainingGateError("best checkpoint identity mismatch")
        model.load_state_dict(best_state["model"])
    summary = {
        "status": "completed",
        "identity": asdict(identity),
        "best_epoch": best_epoch,
        "best_validation_nlp_nat_per_target_dimension": best_value,
        "epochs_completed": len(history),
        "posthoc_calibration_applied": False,
        "final_evaluation_accessed": False,
    }
    summary_path = output_directory / "training_summary.json"
    _atomic_text_write(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def validate_engineering_smoke_limits(
    *, examples: int, optimizer_steps: int, authorization: Mapping[str, Any]
) -> None:
    contract = authorization["engineering_smoke_contract"]
    if examples > int(contract["maximum_in_memory_examples"]):
        raise TrainingGateError("engineering smoke exceeds its example cap")
    if optimizer_steps > int(contract["maximum_optimizer_steps"]):
        raise TrainingGateError("engineering smoke exceeds its optimizer-step cap")
    if contract["scientific_data_reads_authorized"] is not False:
        raise TrainingGateError("engineering smoke may not read scientific data")
