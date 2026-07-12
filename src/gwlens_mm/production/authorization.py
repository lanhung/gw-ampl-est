"""Fail-closed loader for the separate human Phase 3A execution gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..config import load_yaml
from ..provenance import configuration_hash


@dataclass(frozen=True)
class QualificationAuthorization:
    authorization_path: Path
    authorizing_git_commit: str
    base_main_commit: str
    preregistration_path: Path
    preregistration_version: str
    preregistration_hash: str
    schema_version: str
    accepted_pair_count: int
    shard_pair_count: int
    staging_root: Path
    publication_root: Path
    manifest_root: Path
    log_root: Path
    minimum_prelaunch_free_bytes: int
    minimum_post_run_free_bytes: int
    maximum_output_bytes: int
    maximum_walltime_hours: float

    @property
    def shard_count(self) -> int:
        count, remainder = divmod(self.accepted_pair_count, self.shard_pair_count)
        if remainder:
            raise ValueError("accepted count must be divisible by shard size")
        return count


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"authorization {name} must be a mapping")
    return value


def _false(mapping: Mapping[str, Any], key: str) -> None:
    if mapping.get(key) is not False:
        raise ValueError(f"authorization requires {key}=false")


def load_qualification_authorization(
    path: Path,
    *,
    repository_root: Path,
    authorizing_git_commit: str,
) -> QualificationAuthorization:
    """Load and verify every human gate, including the frozen config hash."""

    data = load_yaml(path)
    if data.get("phase") != "3A" or data.get("authorization_status") != "authorized":
        raise ValueError("Phase 3A human authorization is absent")
    if data.get("authorized_by") != "human_project_owner" or not data.get(
        "authorization_date"
    ):
        raise ValueError("Phase 3A authorization identity is incomplete")
    if len(authorizing_git_commit) != 40 or any(
        character not in "0123456789abcdef" for character in authorizing_git_commit.lower()
    ):
        raise ValueError("authorizing commit must be a full hexadecimal Git hash")

    frozen = _mapping(data.get("frozen_preregistration"), "frozen_preregistration")
    relative_preregistration = Path(str(frozen.get("path", "")))
    if relative_preregistration.is_absolute() or ".." in relative_preregistration.parts:
        raise ValueError("preregistration path must be repository-relative")
    preregistration_path = repository_root / relative_preregistration
    preregistration = load_yaml(preregistration_path)
    expected_hash = str(frozen.get("canonical_hash", ""))
    actual_hash = configuration_hash(preregistration)
    if actual_hash != expected_hash:
        raise ValueError(
            f"frozen preregistration hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    if preregistration.get("preregistration_version") != frozen.get("version"):
        raise ValueError("frozen preregistration version mismatch")
    if preregistration.get("scientific_schema_version") != frozen.get(
        "scientific_schema_version"
    ):
        raise ValueError("frozen scientific schema mismatch")
    for key in (
        "execution_enabled",
        "scientific_data_generation_authorized",
        "training_authorized",
        "gwtc_gwosc_download_authorized",
    ):
        _false(preregistration, key)

    authorization = _mapping(data.get("authorization"), "authorization")
    if authorization.get("generator_qualification_authorized") is not True:
        raise ValueError("generator qualification is not authorized")
    for key in (
        "full_production_authorized",
        "model_training_authorized",
        "posthoc_calibration_authorized",
        "sbc_authorized",
        "scientific_testing_authorized",
        "gwosc_gwtc_download_authorized",
        "real_noise_generation_authorized",
        "manuscript_authorized",
    ):
        _false(authorization, key)
    policy = _mapping(data.get("dataset_policy"), "dataset_policy")
    if policy.get("dataset_purpose") != "generator_qualification":
        raise ValueError("authorization has the wrong dataset purpose")
    for key in (
        "scientific_use_authorized",
        "training_use_authorized",
        "calibration_use_authorized",
        "test_use_authorized",
    ):
        _false(policy, key)
    if data.get("stop_after_phase3a") is not True:
        raise ValueError("Phase 3A stop gate is absent")

    paths = _mapping(data.get("remote_paths"), "remote_paths")
    resources = _mapping(data.get("resource_gates"), "resource_gates")
    result = QualificationAuthorization(
        authorization_path=path,
        authorizing_git_commit=authorizing_git_commit,
        base_main_commit=str(data.get("base_main_commit", "")),
        preregistration_path=preregistration_path,
        preregistration_version=str(frozen["version"]),
        preregistration_hash=actual_hash,
        schema_version=str(frozen["scientific_schema_version"]),
        accepted_pair_count=int(authorization["exact_accepted_pair_count"]),
        shard_pair_count=int(authorization["shard_pair_count"]),
        staging_root=Path(str(paths["staging_root"])),
        publication_root=Path(str(paths["publication_root"])),
        manifest_root=Path(str(paths["manifest_root"])),
        log_root=Path(str(paths["log_root"])),
        minimum_prelaunch_free_bytes=int(resources["minimum_prelaunch_free_bytes"]),
        minimum_post_run_free_bytes=int(resources["minimum_post_run_free_bytes"]),
        maximum_output_bytes=int(resources["maximum_phase3a_output_bytes"]),
        maximum_walltime_hours=float(
            resources["maximum_projected_walltime_hours_without_new_review"]
        ),
    )
    if result.accepted_pair_count != 4096 or result.shard_pair_count != 128:
        raise ValueError("authorization must remain exactly 4096 pairs in 128-pair shards")
    if result.shard_count != 32:
        raise ValueError("authorization must produce exactly 32 shards")
    approved_root = Path("/root/autodl-tmp/lensing-4")
    for remote_path in (
        result.staging_root,
        result.publication_root,
        result.manifest_root,
        result.log_root,
    ):
        if not remote_path.is_absolute() or not remote_path.is_relative_to(approved_root):
            raise ValueError(f"remote output path is outside the approved root: {remote_path}")
    return result
