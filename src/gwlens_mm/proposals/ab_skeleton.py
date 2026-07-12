"""Dry-run-only validation for a future proposal-v2 A/B qualification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class ABBlock:
    block_index: int
    first_arm: str
    second_arm: str
    accepted_pairs_per_arm: int


@dataclass(frozen=True)
class ABRunPlan:
    parent_run_id: str
    control_dataset_id: str
    candidate_dataset_id: str
    parent_comparison_manifest: str
    control_manifest: str
    candidate_manifest: str
    control_checksum_manifest: str
    candidate_checksum_manifest: str
    blocks: Tuple[ABBlock, ...]
    maximum_attempts_per_arm: int
    maximum_active_hours_per_arm: float
    maximum_accepted_pairs_per_arm: int
    maximum_accepted_pairs_total: int
    dry_run_only: bool


def build_dry_run_plan(config: Mapping[str, Any], parent_run_id: str) -> ABRunPlan:
    skeleton = config["future_ab_skeleton"]
    caps = skeleton["caps"]
    if not parent_run_id or "/" in parent_run_id or ".." in parent_run_id:
        raise ValueError("parent A/B run ID must be a safe nonempty identifier")
    if skeleton["dry_run_only"] is not True:
        raise ValueError("Phase 3C-0 A/B runner must remain dry-run-only")
    arm_count = int(skeleton["arm_count"])
    per_arm = int(skeleton["accepted_pairs_per_arm"])
    total = int(skeleton["total_accepted_pairs"])
    block_count = int(skeleton["blocks_per_arm"])
    per_block = int(skeleton["accepted_pairs_per_block"])
    if arm_count != 2 or block_count * per_block != per_arm or arm_count * per_arm != total:
        raise ValueError("A/B count arithmetic is inconsistent")
    if str(skeleton["execution_order"]) != "sequential_arm_blocks":
        raise ValueError("future arm blocks must execute sequentially")
    if not all(
        bool(skeleton[key])
        for key in (
            "distinct_dataset_ids",
            "distinct_arm_manifests_and_checksums",
            "one_parent_comparison_manifest",
            "identical_environment_workers_and_telemetry",
        )
    ):
        raise ValueError("future A/B identity and environment contracts are incomplete")
    if int(caps["maximum_accepted_pairs_per_arm"]) != per_arm:
        raise ValueError("per-arm acceptance cap differs from the design")
    if int(caps["maximum_accepted_pairs_total"]) != total:
        raise ValueError("total acceptance cap differs from the design")
    if bool(caps["partial_or_capped_block_omission_authorized"]):
        raise ValueError("partial or capped blocks may not be omitted")
    required_telemetry = {
        "active_wall_seconds_per_block",
        "attempts_per_block",
        "accepted_pairs_per_block",
        "cpu_seconds",
        "process_tree_or_cgroup_peak_rss",
        "time_integrated_cpu_utilization",
        "proposal_sample_time",
        "proposal_log_density_time",
        "peak_staging_bytes",
        "environment_identity",
    }
    if not required_telemetry <= set(skeleton["telemetry"]):
        raise ValueError("future A/B telemetry contract is incomplete")
    control_id = f"{parent_run_id}-rc5-control"
    candidate_id = f"{parent_run_id}-proposal-v2-candidate"
    if control_id == candidate_id:
        raise ValueError("A/B arm dataset IDs must differ")
    blocks = tuple(
        ABBlock(
            block_index=index,
            first_arm="rc5_control" if index % 2 == 0 else "proposal_v2_candidate",
            second_arm="proposal_v2_candidate" if index % 2 == 0 else "rc5_control",
            accepted_pairs_per_arm=per_block,
        )
        for index in range(block_count)
    )
    return ABRunPlan(
        parent_run_id=parent_run_id,
        control_dataset_id=control_id,
        candidate_dataset_id=candidate_id,
        parent_comparison_manifest=f"manifests/proposal_v2/{parent_run_id}/comparison.json",
        control_manifest=f"manifests/proposal_v2/{parent_run_id}/rc5-control/manifest.json",
        candidate_manifest=(
            f"manifests/proposal_v2/{parent_run_id}/proposal-v2-candidate/manifest.json"
        ),
        control_checksum_manifest=(
            f"manifests/proposal_v2/{parent_run_id}/rc5-control/checksums.sha256"
        ),
        candidate_checksum_manifest=(
            f"manifests/proposal_v2/{parent_run_id}/proposal-v2-candidate/checksums.sha256"
        ),
        blocks=blocks,
        maximum_attempts_per_arm=int(caps["maximum_attempts_per_arm"]),
        maximum_active_hours_per_arm=float(caps["maximum_active_hours_per_arm"]),
        maximum_accepted_pairs_per_arm=per_arm,
        maximum_accepted_pairs_total=total,
        dry_run_only=True,
    )
