"""Qualification run identity, preflight gates, and append-only attempts."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import socket
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, TextIO

from ..provenance import canonical_json
from .authorization import QualificationAuthorization


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: int
    proposal_seed: int
    lens_family: str
    em_cell: str
    status: str
    rejection_reason: Optional[str]
    pair_id: Optional[str]
    source_id: str
    lens_id: str
    physical_system_id: str

    def validate(self) -> None:
        if self.attempt_id < 0 or self.proposal_seed < 0:
            raise ValueError("attempt identifiers must be nonnegative")
        if self.status not in {"accepted", "rejected"}:
            raise ValueError("attempt status must be accepted or rejected")
        if (self.status == "accepted") != (self.pair_id is not None):
            raise ValueError("only accepted attempts have pair IDs")
        if (self.status == "rejected") != (self.rejection_reason is not None):
            raise ValueError("only rejected attempts require a rejection reason")
        identifiers = (
            self.lens_family,
            self.em_cell,
            self.source_id,
            self.lens_id,
            self.physical_system_id,
        )
        if not all(identifiers):
            raise ValueError("attempt provenance identifiers are required")


class AttemptJournal:
    """Durable JSONL journal that refuses non-contiguous or changed resume."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._last_id = -1
        self._seen_systems: set[str] = set()
        if path.exists():
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                data = json.loads(line)
                record = AttemptRecord(**data)
                record.validate()
                if record.attempt_id != line_number:
                    raise ValueError("attempt journal IDs are not contiguous")
                if record.physical_system_id in self._seen_systems:
                    raise ValueError("attempt journal contains duplicate physical-system IDs")
                self._seen_systems.add(record.physical_system_id)
                self._last_id = record.attempt_id
        self._handle: TextIO = path.open("a", encoding="utf-8")

    @property
    def next_attempt_id(self) -> int:
        return self._last_id + 1

    def append(self, record: AttemptRecord) -> None:
        record.validate()
        if record.attempt_id != self.next_attempt_id:
            raise ValueError("attempt IDs must be appended contiguously")
        if record.physical_system_id in self._seen_systems:
            raise ValueError("physical-system ID already exists in attempt journal")
        self._handle.write(canonical_json(asdict(record)) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._last_id = record.attempt_id
        self._seen_systems.add(record.physical_system_id)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "AttemptJournal":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_psd_files(psd_specification: Mapping[str, Mapping[str, str]]) -> Dict[str, str]:
    bilby = importlib.import_module("bilby")
    bilby_file = getattr(bilby, "__file__", None)
    if bilby_file is None:
        raise RuntimeError("Bilby package path is unavailable")
    root = Path(bilby_file).resolve().parent / "gw" / "detector" / "noise_curves"
    verified: Dict[str, str] = {}
    if set(psd_specification) != {"H1", "L1", "V1"}:
        raise ValueError("PSD specification must cover H1/L1/V1")
    for detector, specification in psd_specification.items():
        path = root / specification["file"]
        actual = sha256_path(path)
        if actual != specification["sha256"]:
            raise ValueError(f"PSD hash mismatch for {detector}: {path}")
        verified[detector] = f"{path}:{actual}"
    return verified


def build_preflight_manifest(
    authorization: QualificationAuthorization,
    config: Mapping[str, Any],
    *,
    run_id: str,
    dataset_id: str,
    generator_git_commit: str,
    configuration_hash: str,
) -> Dict[str, Any]:
    """Validate resource/identity gates and return the pre-write manifest."""

    if not run_id or not dataset_id:
        raise ValueError("run and dataset IDs are required")
    if len(generator_git_commit) != 40 or len(configuration_hash) != 64:
        raise ValueError("full generator commit and configuration hash are required")
    usage = shutil.disk_usage(authorization.staging_root.parent)
    if usage.free < authorization.minimum_prelaunch_free_bytes:
        raise RuntimeError(
            f"free-space gate failed: {usage.free} < {authorization.minimum_prelaunch_free_bytes}"
        )
    psd = verify_psd_files(config["gw"]["psd_curves"])
    expected = [
        f"shards/shard-{index:05d}" for index in range(authorization.shard_count)
    ]
    return {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "state": "preflight_passed",
        "dataset_purpose": "generator_qualification",
        "scientific_use_authorized": False,
        "training_use_authorized": False,
        "calibration_use_authorized": False,
        "test_use_authorized": False,
        "generator_git_commit": generator_git_commit,
        "authorizing_git_commit": authorization.authorizing_git_commit,
        "configuration_hash": configuration_hash,
        "preregistration_version": authorization.preregistration_version,
        "preregistration_hash": authorization.preregistration_hash,
        "hostname": socket.gethostname(),
        "python": sys.version,
        "free_bytes_before_run": usage.free,
        "projected_peak_bytes": int(
            config["resource_gates"].get("qualification_projected_peak_bytes", 5807608883)
        ),
        "psd_files": psd,
        "staging_path": str(authorization.staging_root / dataset_id),
        "publication_path": str(authorization.publication_root / dataset_id),
        "log_path": str(authorization.log_root / f"{run_id}.log"),
        "resume_plan": "verify immutable complete shards; retain and fail on inconsistent partial",
        "expected_artifacts": expected,
    }
