"""Deterministic optimization, checkpoint, and resume primitives."""

from __future__ import annotations

import csv
import hashlib
import importlib
import io
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np

from .contracts import TrainingGateError
from .features import InputStandardizer
from .metrics import empirical_crps


@dataclass(frozen=True)
class TargetStandardizer:
    mean: Tuple[float, float]
    standard_deviation: Tuple[float, float]

    @classmethod
    def fit(cls, targets: np.ndarray) -> "TargetStandardizer":
        values = np.asarray(targets, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
            raise ValueError("target standardization requires at least two two-dimensional cases")
        return cls.fit_iterable(values)

    @classmethod
    def fit_iterable(cls, targets: Iterable[np.ndarray]) -> "TargetStandardizer":
        """Fit target moments in a deterministic bounded-memory pass."""

        count = 0
        mean = np.zeros(2, dtype=np.float64)
        m2 = np.zeros(2, dtype=np.float64)
        for target in targets:
            value = np.asarray(target, dtype=np.float64)
            if value.shape != (2,) or not np.all(np.isfinite(value)):
                raise ValueError("target standardization received NaN, Inf, or wrong shape")
            count += 1
            delta = value - mean
            mean += delta / count
            m2 += delta * (value - mean)
        if count < 2:
            raise ValueError("target standardization requires at least two two-dimensional cases")
        scale = np.sqrt(m2 / count)
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


class DeterministicShardEpochSampler:
    """Epoch-addressed shuffle that preserves shard-local I/O.

    A global element permutation makes a lazy sharded reader reopen Parquet and
    Zarr stores for almost every example.  Randomizing both the shard order and
    the row order within every shard preserves an unbiased epoch permutation
    while reducing store opens to roughly the number of shards.
    """

    def __init__(self, shard_keys: Sequence[str], *, seed: int) -> None:
        if not shard_keys:
            raise ValueError("shard-aware sampler requires at least one example")
        if seed not in (0, 1, 2):
            raise ValueError("sampler seed is outside the preregistered set")
        groups: Dict[str, list[int]] = {}
        for index, key in enumerate(shard_keys):
            if not key:
                raise ValueError("shard-aware sampler received an empty shard key")
            groups.setdefault(str(key), []).append(index)
        self.groups = tuple(
            (key, tuple(indices)) for key, indices in sorted(groups.items())
        )
        self.size = len(shard_keys)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("sampler epoch must be nonnegative")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = np.random.default_rng(np.random.SeedSequence((self.seed, self.epoch)))
        group_order = generator.permutation(len(self.groups))
        ordered: list[int] = []
        for group_index in group_order:
            indices = np.asarray(self.groups[int(group_index)][1], dtype=np.int64)
            ordered.extend(int(value) for value in generator.permutation(indices))
        if len(ordered) != self.size or len(set(ordered)) != self.size:
            raise RuntimeError("shard-aware sampler did not construct an epoch permutation")
        return iter(ordered)

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
    if not isinstance(
        sampler, (DeterministicEpochSampler, DeterministicShardEpochSampler)
    ):
        raise TrainingGateError(
            "scientific train loader requires an epoch-addressed deterministic sampler"
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
    history: list[Mapping[str, Any]] = []
    if resume_checkpoint is not None:
        if resume_checkpoint.resolve() != checkpoint_path.resolve():
            raise TrainingGateError("probe resume must use this run identity's last checkpoint")
        state = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if state["identity"] != asdict(identity):
            raise TrainingGateError("checkpoint identity does not match this training run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_value = float(state["best_validation_nlp_per_dimension"])
        best_epoch = int(state["best_epoch"])
        epochs_without_improvement = int(state["epochs_without_improvement"])
        checkpoint_history = state.get("history")
        if not isinstance(checkpoint_history, list) or len(checkpoint_history) != start_epoch:
            raise TrainingGateError("checkpoint history does not match its completed epoch")
        history = [dict(row) for row in checkpoint_history]
        if state["input_standardizer"] != asdict(input_standardizer):
            raise TrainingGateError("checkpoint input standardizer mismatch")
        if state["target_standardizer"] != asdict(standardizer):
            raise TrainingGateError("checkpoint target standardizer mismatch")
        _restore_rng_state(torch, state["rng_state"])
        if best_epoch >= 0 and not best_path.is_file():
            raise TrainingGateError("resume checkpoint has no matching best checkpoint")
    maximum_epochs = int(optimization["maximum_epochs"])
    patience = int(optimization["early_stopping_patience_epochs"])
    minimum_delta = float(optimization["early_stopping_minimum_delta"])
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


def evaluate_development_validation(
    model: Any,
    validation_loader: Iterable[
        Tuple[Mapping[str, Any], Sequence[Mapping[str, Optional[str]]]]
    ],
    *,
    standardizer: TargetStandardizer,
    device_name: str,
    posterior_draws_per_case: int,
    evaluation_seed: int,
    output_directory: Path,
    levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
) -> Mapping[str, Any]:
    """Evaluate only the authorized development validation cases."""

    if posterior_draws_per_case < 128:
        raise TrainingGateError("development posterior draw count is too small")
    torch = importlib.import_module("torch")
    device = torch.device(device_name)
    torch.manual_seed(evaluation_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(evaluation_seed)
    model.eval()
    mean = np.asarray(standardizer.mean, dtype=np.float64)
    scale = np.asarray(standardizer.standard_deviation, dtype=np.float64)
    rows: list[Dict[str, Any]] = []
    with torch.no_grad():
        for batch, metadata in validation_loader:
            moved = _move_batch(batch, device)
            target_tensor = moved.pop("target")
            standardized = _standardize_target(target_tensor, standardizer, torch)
            log_probability = (
                model.log_prob(standardized, moved) - standardizer.log_absolute_jacobian
            )
            sampled = model.sample(posterior_draws_per_case, moved)
            sampled_log_probability = model.sample_log_prob(sampled, moved)
            samples = sampled.detach().cpu().numpy().astype(np.float64)
            sampled_log_probability_numpy = (
                sampled_log_probability.detach().cpu().numpy().astype(np.float64)
            )
            target = target_tensor.detach().cpu().numpy().astype(np.float64)
            if samples.shape != (len(target), posterior_draws_per_case, 2):
                raise TrainingGateError("development posterior samples have the wrong shape")
            samples = samples * scale[None, None, :] + mean[None, None, :]
            crps = empirical_crps(samples, target)
            log_probability_numpy = log_probability.detach().cpu().numpy().astype(np.float64)
            if not (
                np.all(np.isfinite(log_probability_numpy))
                and np.all(np.isfinite(sampled_log_probability_numpy))
                and np.all(np.isfinite(samples))
                and np.all(np.isfinite(crps))
            ):
                raise FloatingPointError("development evaluation produced NaN or Inf")
            for case_index, labels in enumerate(metadata):
                row: Dict[str, Any] = {
                    "physical_system_id": labels["physical_system_id"],
                    "lens_family": labels["lens_family"],
                    "em_cell_signature": labels["em_cell_signature"],
                    "tail_view": labels["tail_view"] or "none",
                    "nlp_nat_per_target_dimension": -log_probability_numpy[case_index] / 2.0,
                    "crps_log_mu_primary": crps[case_index, 0],
                    "crps_log_mu_secondary": crps[case_index, 1],
                    "crps_mean": float(np.mean(crps[case_index])),
                }
                for level in levels:
                    key = f"{level:.2f}"
                    alpha = (1.0 - level) / 2.0
                    lower = np.quantile(samples[case_index], alpha, axis=0)
                    upper = np.quantile(samples[case_index], 1.0 - alpha, axis=0)
                    contained = (target[case_index] >= lower) & (
                        target[case_index] <= upper
                    )
                    joint_threshold = np.quantile(
                        sampled_log_probability_numpy[case_index], 1.0 - level
                    )
                    row[f"covered_primary_{key}"] = bool(contained[0])
                    row[f"covered_secondary_{key}"] = bool(contained[1])
                    row[f"covered_joint_{key}"] = bool(
                        log_probability_numpy[case_index]
                        + standardizer.log_absolute_jacobian
                        >= joint_threshold
                    )
                    row[f"width_primary_{key}"] = float(upper[0] - lower[0])
                    row[f"width_secondary_{key}"] = float(upper[1] - lower[1])
                rows.append(row)
    if not rows:
        raise TrainingGateError("development validation loader was empty")
    identifiers = [str(row["physical_system_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise TrainingGateError("development validation contains duplicate system IDs")
    nlp = np.asarray([row["nlp_nat_per_target_dimension"] for row in rows])
    crps_mean = np.asarray([row["crps_mean"] for row in rows])
    aggregate_coverage: Dict[str, Any] = {}
    for level in levels:
        key = f"{level:.2f}"
        primary = np.mean([row[f"covered_primary_{key}"] for row in rows])
        secondary = np.mean([row[f"covered_secondary_{key}"] for row in rows])
        joint = np.mean([row[f"covered_joint_{key}"] for row in rows])
        aggregate_coverage[key] = {
            "marginal": [float(primary), float(secondary)],
            "maximum_marginal_absolute_error": float(
                max(abs(primary - level), abs(secondary - level))
            ),
            "joint": float(joint),
            "joint_absolute_error": float(abs(joint - level)),
            "mean_width": [
                float(np.mean([row[f"width_primary_{key}"] for row in rows])),
                float(np.mean([row[f"width_secondary_{key}"] for row in rows])),
            ],
        }
    em_cell_coverage: Dict[str, Any] = {}
    for group in sorted({str(row["em_cell_signature"]) for row in rows}):
        selected = [row for row in rows if row["em_cell_signature"] == group]
        em_cell_coverage[group] = {
            "case_count": len(selected),
            "marginal_0.90": [
                float(np.mean([row["covered_primary_0.90"] for row in selected])),
                float(np.mean([row["covered_secondary_0.90"] for row in selected])),
            ],
            "joint_0.90": float(
                np.mean([row["covered_joint_0.90"] for row in selected])
            ),
        }
    tail_views: Dict[str, Any] = {}
    for group in sorted({str(row["tail_view"]) for row in rows} - {"none"}):
        selected = [row for row in rows if row["tail_view"] == group]
        tail_views[group] = {
            "case_count": len(selected),
            "median_nlp_nat_per_target_dimension": float(
                np.median([row["nlp_nat_per_target_dimension"] for row in selected])
            ),
            "median_crps": float(np.median([row["crps_mean"] for row in selected])),
            "minimum_case_requirement_met": len(selected) >= 128,
        }
    summary = {
        "status": "completed_development_validation",
        "case_count": len(rows),
        "posterior_draws_per_case": posterior_draws_per_case,
        "evaluation_seed": evaluation_seed,
        "mean_nlp_nat_per_target_dimension": float(np.mean(nlp)),
        "median_nlp_nat_per_target_dimension": float(np.median(nlp)),
        "median_crps": float(np.median(crps_mean)),
        "coverage": aggregate_coverage,
        "em_cell_coverage": em_cell_coverage,
        "validation_internal_tail_views": tail_views,
        "posthoc_calibration_applied": False,
        "final_evaluation_accessed": False,
    }
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text_write(output_directory / "development_cases.csv", csv_buffer.getvalue())
    _atomic_text_write(
        output_directory / "development_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
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
