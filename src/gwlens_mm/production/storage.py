"""Bounded-memory Zarr/Parquet shard writing and immutable resume checks."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from ..arrays import validate_strain_array_semantics
from ..schema import V2Record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_checksum(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix()
        child_size = child.stat().st_size
        child_digest = sha256_file(child)
        digest.update(f"{relative}\0{child_digest}\0{child_size}\n".encode())
        size += child_size
    return digest.hexdigest(), size


@dataclass(frozen=True)
class ShardArtifact:
    relative_path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class ShardManifest:
    shard_index: int
    accepted_pair_count: int
    first_attempt_id: int
    last_attempt_id: int
    pair_ids: Tuple[str, ...]
    source_ids: Tuple[str, ...]
    lens_ids: Tuple[str, ...]
    physical_system_ids: Tuple[str, ...]
    noise_segment_ids: Tuple[str, ...]
    artifacts: Tuple[ShardArtifact, ...]
    status: str = "complete"

    def validate(self, expected_pairs: int) -> None:
        if self.status != "complete" or self.accepted_pair_count != expected_pairs:
            raise ValueError("complete shard has the wrong accepted-pair count")
        if self.last_attempt_id < self.first_attempt_id:
            raise ValueError("shard attempt range is invalid")
        for name, values in (
            ("pair", self.pair_ids),
            ("source", self.source_ids),
            ("lens", self.lens_ids),
            ("physical system", self.physical_system_ids),
            ("noise segment", self.noise_segment_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {name} ID within shard")
        if len(self.pair_ids) != expected_pairs:
            raise ValueError("shard pair IDs do not account for accepted pairs")
        required = {"noisy.zarr", "clean.zarr", "noise.zarr", "records.parquet"}
        if not required <= {artifact.relative_path for artifact in self.artifacts}:
            raise ValueError("shard manifest is missing required artifacts")
        for artifact in self.artifacts:
            if len(artifact.sha256) != 64 or artifact.bytes < 0:
                raise ValueError("shard artifact checksum is incomplete")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "ShardManifest":
        data = json.loads(text)
        for name in (
            "pair_ids",
            "source_ids",
            "lens_ids",
            "physical_system_ids",
            "noise_segment_ids",
        ):
            data[name] = tuple(data[name])
        data["artifacts"] = tuple(ShardArtifact(**item) for item in data["artifacts"])
        return cls(**data)


class ShardWriter:
    """Write one pair at a time; never retain a dataset-sized array."""

    def __init__(
        self,
        shards_root: Path,
        shard_index: int,
        *,
        expected_pairs: int,
        sample_count: int,
    ) -> None:
        self.shard_index = shard_index
        self.expected_pairs = expected_pairs
        self.sample_count = sample_count
        self.partial = shards_root / f"shard-{shard_index:05d}.partial"
        self.complete = shards_root / f"shard-{shard_index:05d}"
        if self.complete.exists():
            raise FileExistsError(f"complete shard is immutable: {self.complete}")
        self.partial.mkdir(parents=True, exist_ok=False)
        zarr = importlib.import_module("zarr")
        compressor = importlib.import_module("numcodecs").Blosc(
            cname="zstd", clevel=5, shuffle=2
        )
        shape = (expected_pairs, 2, 3, sample_count)
        chunks = (1, 2, 3, sample_count)
        self._arrays = {
            name: zarr.open_array(
                str(self.partial / f"{name}.zarr"),
                mode="w",
                shape=shape,
                chunks=chunks,
                dtype="float32",
                compressor=compressor,
            )
            for name in ("noisy", "clean", "noise")
        }
        self._records: list[Dict[str, str]] = []
        self._pair_ids: list[str] = []
        self._source_ids: list[str] = []
        self._lens_ids: list[str] = []
        self._system_ids: list[str] = []
        self._noise_ids: list[str] = []
        self._first_attempt: Optional[int] = None
        self._last_attempt: Optional[int] = None

    def append(
        self,
        record: V2Record,
        noisy: np.ndarray,
        clean: np.ndarray,
        noise: np.ndarray,
        *,
        attempt_id: int,
        partition_metadata: Mapping[str, str],
    ) -> None:
        index = len(self._records)
        if index >= self.expected_pairs:
            raise ValueError("cannot append beyond the reviewed shard size")
        expected_shape = (2, 3, self.sample_count)
        for name, array in (("noisy", noisy), ("clean", clean), ("noise", noise)):
            if array.shape != expected_shape or array.dtype != np.float32:
                raise ValueError(f"{name} array has the wrong shape or dtype")
        record.validate()
        validate_strain_array_semantics(
            noisy,
            clean,
            noise,
            record.gw_observation.detector_availability_mask,
        )
        self._arrays["noisy"][index] = noisy
        self._arrays["clean"][index] = clean
        self._arrays["noise"][index] = noise
        self._records.append(
            {
                "pair_id": record.pair.pair_id,
                "lens_family": record.pair.lens_family.value,
                "record_json": record.to_json(indent=None),
                **dict(partition_metadata),
            }
        )
        self._pair_ids.append(record.pair.pair_id)
        self._source_ids.append(record.pair.source_id)
        self._lens_ids.append(record.pair.lens_id)
        self._system_ids.append(record.pair.physical_system_id)
        self._noise_ids.extend(record.provenance.used_noise_segment_ids)
        self._first_attempt = attempt_id if self._first_attempt is None else self._first_attempt
        self._last_attempt = attempt_id

    def finalize(self) -> ShardManifest:
        if len(self._records) != self.expected_pairs:
            raise ValueError("partial shard cannot be finalized")
        pandas = importlib.import_module("pandas")
        pandas.DataFrame(self._records).to_parquet(self.partial / "records.parquet", index=False)
        artifacts = []
        for name in ("noisy.zarr", "clean.zarr", "noise.zarr", "records.parquet"):
            path = self.partial / name
            digest, size = (
                tree_checksum(path) if path.is_dir() else (sha256_file(path), path.stat().st_size)
            )
            artifacts.append(ShardArtifact(name, digest, size))
        if self._first_attempt is None or self._last_attempt is None:
            raise RuntimeError("finalized shard has no attempt range")
        manifest = ShardManifest(
            shard_index=self.shard_index,
            accepted_pair_count=len(self._records),
            first_attempt_id=self._first_attempt,
            last_attempt_id=self._last_attempt,
            pair_ids=tuple(self._pair_ids),
            source_ids=tuple(self._source_ids),
            lens_ids=tuple(self._lens_ids),
            physical_system_ids=tuple(self._system_ids),
            noise_segment_ids=tuple(self._noise_ids),
            artifacts=tuple(artifacts),
        )
        manifest.validate(self.expected_pairs)
        manifest_path = self.partial / "shard_manifest.json"
        manifest_path.write_text(manifest.to_json(), encoding="utf-8")
        marker = self.partial / "COMPLETE.json"
        marker.write_text(
            json.dumps(
                {"shard_manifest_sha256": sha256_file(manifest_path)}, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(self.partial, self.complete)
        return manifest


def verify_complete_shard(path: Path, expected_pairs: int) -> ShardManifest:
    manifest_path = path / "shard_manifest.json"
    marker_path = path / "COMPLETE.json"
    if not path.is_dir() or not manifest_path.is_file() or not marker_path.is_file():
        raise ValueError(f"incomplete shard: {path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("shard_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("shard manifest hash mismatch")
    manifest = ShardManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    manifest.validate(expected_pairs)
    for artifact in manifest.artifacts:
        artifact_path = path / artifact.relative_path
        digest, size = (
            tree_checksum(artifact_path)
            if artifact_path.is_dir()
            else (sha256_file(artifact_path), artifact_path.stat().st_size)
        )
        if digest != artifact.sha256 or size != artifact.bytes:
            raise ValueError(f"shard artifact mismatch: {artifact.relative_path}")
    return manifest
