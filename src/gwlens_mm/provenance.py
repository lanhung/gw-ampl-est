"""Deterministic configuration hashes, seed hierarchy, and dataset manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

SEED_DOMAINS = (
    "source",
    "lens",
    "pair_selection",
    "detector_noise",
    "em_measurement_noise",
    "missing_modalities",
    "stellar_kinematics",
    "augmentation",
)


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def configuration_hash(configuration: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(configuration).encode("utf-8")).hexdigest()


def derive_seed(root_seed: int, domain: str, *stable_identifiers: str) -> int:
    """Derive a stable unsigned 64-bit child seed without Python's hash()."""

    if root_seed < 0:
        raise ValueError("root_seed must be nonnegative")
    if domain not in SEED_DOMAINS:
        raise ValueError(f"unknown seed domain: {domain}")
    if not stable_identifiers or any(not identifier for identifier in stable_identifiers):
        raise ValueError("one or more nonempty stable identifiers are required")
    payload = canonical_json(
        {"root_seed": root_seed, "domain": domain, "identifiers": stable_identifiers}
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def dataset_id(
    schema_version: str, generator_git_commit: str, config_hash: str, root_seed: int
) -> str:
    if len(generator_git_commit) != 40 or len(config_hash) != 64:
        raise ValueError("dataset IDs require a full Git hash and SHA-256 config hash")
    digest = hashlib.sha256(
        canonical_json(
            {
                "schema_version": schema_version,
                "generator_git_commit": generator_git_commit,
                "configuration_hash": config_hash,
                "root_seed": root_seed,
            }
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"gwlens-v2-{schema_version}-{digest}"


@dataclass(frozen=True)
class ArtifactChecksum:
    relative_path: str
    sha256: Optional[str]
    bytes: Optional[int]
    status: str = "pending"

    def validate(self) -> None:
        if (
            not self.relative_path
            or self.relative_path.startswith("/")
            or ".." in Path(self.relative_path).parts
        ):
            raise ValueError("artifact paths must be safe and dataset-relative")
        if self.status not in {"pending", "complete", "failed"}:
            raise ValueError("unsupported checksum status")
        if self.status == "complete":
            if (
                self.sha256 is None
                or len(self.sha256) != 64
                or any(character not in "0123456789abcdef" for character in self.sha256.lower())
                or self.bytes is None
                or self.bytes < 0
            ):
                raise ValueError("complete artifacts require SHA-256 and nonnegative byte size")


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    schema_version: str
    generator_git_commit: str
    configuration_hash: str
    root_seed: int
    planned_pair_count: int
    accepted_pair_count: int
    attempted_pair_count: int
    pair_ids: Tuple[str, ...]
    source_ids: Tuple[str, ...]
    lens_ids: Tuple[str, ...]
    noise_segment_ids: Tuple[str, ...]
    artifacts: Tuple[ArtifactChecksum, ...]
    generation_status: str
    dataset_purpose: str
    scientific_use_authorized: bool
    authorizing_git_commit: str
    training_use_authorized: bool = False
    calibration_use_authorized: bool = False
    test_use_authorized: bool = False

    def validate(self) -> None:
        if not self.dataset_id or self.root_seed < 0:
            raise ValueError("dataset identity and nonnegative root seed are required")
        if self.dataset_purpose not in {
            "engineering_smoke",
            "generator_qualification",
            "proposal_efficiency_engineering_qualification",
        }:
            raise ValueError("unsupported non-scientific dataset purpose")
        if self.scientific_use_authorized:
            raise ValueError("engineering data cannot be authorized for scientific use")
        if any(
            (
                self.training_use_authorized,
                self.calibration_use_authorized,
                self.test_use_authorized,
            )
        ):
            raise ValueError("engineering data cannot be authorized for downstream use")
        if len(self.authorizing_git_commit) != 40 or any(
            character not in "0123456789abcdef"
            for character in self.authorizing_git_commit.lower()
        ):
            raise ValueError("authorizing_git_commit must be a full hexadecimal Git hash")
        if self.planned_pair_count <= 0:
            raise ValueError("planned pair count must be positive")
        if self.accepted_pair_count < 0 or self.accepted_pair_count > self.planned_pair_count:
            raise ValueError("accepted pair count cannot exceed planned count")
        if self.attempted_pair_count < self.accepted_pair_count:
            raise ValueError("attempt/accepted counts are inconsistent")
        if len(self.pair_ids) != self.accepted_pair_count:
            raise ValueError("pair IDs must account for every accepted pair")
        for label, identifiers in (
            ("pair", self.pair_ids),
            ("source", self.source_ids),
            ("lens", self.lens_ids),
            ("noise segment", self.noise_segment_ids),
        ):
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"duplicate {label} IDs in manifest")
        if self.generation_status not in {"planned", "running", "complete", "failed"}:
            raise ValueError("unsupported generation status")
        for artifact in self.artifacts:
            artifact.validate()
        if self.generation_status == "complete":
            if self.accepted_pair_count != self.planned_pair_count:
                raise ValueError(
                    "complete generation requires accepted count to equal planned count"
                )
            if not self.artifacts or any(
                artifact.status != "complete" for artifact in self.artifacts
            ):
                raise ValueError("complete generation requires every published artifact complete")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "DatasetManifest":
        data = json.loads(text)
        data["pair_ids"] = tuple(data["pair_ids"])
        data["source_ids"] = tuple(data["source_ids"])
        data["lens_ids"] = tuple(data["lens_ids"])
        data["noise_segment_ids"] = tuple(data["noise_segment_ids"])
        data["artifacts"] = tuple(ArtifactChecksum(**item) for item in data["artifacts"])
        manifest = cls(**data)
        manifest.validate()
        return manifest


def reject_duplicate_dataset_ids(manifests: Iterable[DatasetManifest]) -> None:
    identifiers = [manifest.dataset_id for manifest in manifests]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate dataset IDs")
