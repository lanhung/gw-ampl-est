from __future__ import annotations

import hashlib
from pathlib import Path

from gwlens_mm.config import load_yaml
from gwlens_mm.provenance import configuration_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/statistics/terminal_131k_preregistration.yaml"
AUTHORIZATION_PATH = (
    ROOT / "configs/execution/phase4_terminal_131k_preregistration_authorization.yaml"
)
DECISION_PATH = ROOT / "results/phase4/corrected_probe_65k/learning_curve_decision.json"
EXPECTED_HASH = "77ff5b6b45b886657e90023c50ae002afffb077db594c80665166d537fd2346a"


def test_terminal_preregistration_identity_and_trigger_are_exact() -> None:
    config = load_yaml(CONFIG_PATH)
    authorization = load_yaml(AUTHORIZATION_PATH)
    decision_sha = hashlib.sha256(DECISION_PATH.read_bytes()).hexdigest()

    assert config["preregistration_version"] == "1.2.0-rc.1"
    assert config["status"] == "design_frozen_execution_disabled"
    assert configuration_hash(config) == EXPECTED_HASH
    assert authorization["authorized_scientific_contract"]["preregistration_version"] == (
        "1.2.0-rc.1"
    )
    assert config["triggering_terminal_evidence"]["decision"] == (
        "stop_data_limited_and_new_preregistration"
    )
    assert decision_sha == config["triggering_terminal_evidence"]["decision_sha256"]
    assert config["triggering_terminal_evidence"]["final_evaluation_accessed"] is False
    assert config["triggering_terminal_evidence"]["calibration_accessed"] is False


def test_parent_contracts_remain_immutable() -> None:
    config = load_yaml(CONFIG_PATH)
    for item in config["parent_contracts"].values():
        loaded = load_yaml(ROOT / item["path"])
        assert configuration_hash(loaded) == item["canonical_hash"]


def test_terminal_train_ladder_is_strict_and_exact() -> None:
    contract = load_yaml(CONFIG_PATH)["terminal_training_ladder"]
    existing = contract["corrected_train_65k"]["accepted_count"]
    increment = contract["train_131k_increment"]
    terminal = contract["train_131k_terminal"]

    assert existing == 65_536
    assert increment["accepted_count"] == 65_536
    assert increment["shard_count"] == 512
    assert increment["accepted_pairs_per_shard"] == 128
    assert increment["shard_count"] * increment["accepted_pairs_per_shard"] == (
        increment["accepted_count"]
    )
    assert terminal["accepted_count"] == existing + increment["accepted_count"] == 131_072
    assert terminal["strict_nested_membership_required"] is True
    assert terminal["corrected_train_65k_membership_must_not_change"] is True
    assert contract["extension_above_131072"]["authorized"] is False
    assert contract["extension_above_131072"]["automatic"] is False


def test_development_tail_pool_is_balanced_disjoint_and_non_scientific() -> None:
    tail = load_yaml(CONFIG_PATH)["development_tail_pool"]
    strata = tail["strata"]

    assert tail["accepted_count"] == 512
    assert tail["shard_count"] == 4
    assert tail["accepted_pairs_per_shard"] == 128
    assert len(strata) == 4
    assert {item["accepted_count"] for item in strata} == {128}
    assert sum(item["accepted_count"] for item in strata) == tail["accepted_count"]
    assert [item["id"] for item in strata] == [
        "high_absolute_magnification",
        "extreme_relative_magnification",
        "second_image_near_threshold",
        "extreme_profile_or_environment",
    ]
    assert tail["proposal_equals_evaluation_within_declared_conditional_stratum"] is True
    assert tail["log_importance_weight"] == 0.0
    assert tail["importance_weight"] == 1.0
    assert "every_final_evaluation_split" in tail["group_disjoint_from"]
    assert tail["final_evaluation_ids_or_seeds_reused"] is False
    assert tail["training_use_authorized"] is False
    assert tail["calibration_use_authorized"] is False
    assert tail["architecture_selection_use_authorized"] is False


def test_terminal_decision_always_locks_at_resource_cap_without_false_saturation() -> None:
    config = load_yaml(CONFIG_PATH)
    comparison = config["terminal_comparison"]
    outcomes = config["terminal_outcomes"]

    assert comparison["saturation_criteria"] == {
        "nlp_improvement_ci_upper_max_nat_per_target_dimension": 0.01,
        "median_crps_relative_improvement_max": 0.01,
        "all_three_seed_nlp_and_crps_point_improvements_below_threshold": True,
        "require_all": True,
    }
    assert comparison["raw_uncalibrated_coverage_may_block_terminal_size_lock"] is False
    assert comparison["development_tail_minimum_cases_per_stratum"] == 128
    assert comparison["development_tail_completion_required_before_decision"] is True
    assert comparison["final_evaluation_may_affect_decision"] is False
    assert comparison["calibration_or_sbc_may_affect_decision"] is False
    assert outcomes["saturation_passes"]["decision"] == "lock_train_131k_saturated"
    assert outcomes["saturation_does_not_pass"]["decision"] == (
        "lock_train_131k_resource_capped_data_limited"
    )
    assert outcomes["saturation_does_not_pass"]["mandatory_reporting_label"] == (
        "terminal_training_scale_remains_data_limited"
    )
    assert outcomes["both_outcomes"]["selected_training_count"] == 131_072
    assert outcomes["both_outcomes"]["extension_above_131072_authorized"] is False


def test_resource_arithmetic_and_execution_denials_are_fail_closed() -> None:
    config = load_yaml(CONFIG_PATH)
    resource = config["resource_projection"]
    execution = config["execution"]
    authorization = load_yaml(AUTHORIZATION_PATH)

    assert resource["projected_train_increment_publication_bytes"] == (
        2 * resource["measured_stage_b_publication_bytes"]
    )
    assert resource["projected_train_increment_elapsed_hours"] == (
        2 * resource["measured_stage_b_elapsed_hours"]
    )
    assert resource["projected_train_increment_peak_bytes"] == (
        resource["projected_train_increment_publication_bytes"]
        + resource["inherited_active_staging_and_reserve_bytes"]
    )
    assert resource["projected_post_peak_free_bytes_before_tail"] == (
        resource["free_bytes_at_65k_closeout"]
        - resource["projected_train_increment_peak_bytes"]
    )
    assert resource["projected_post_peak_free_bytes_before_tail"] >= (
        resource["minimum_post_peak_free_bytes"]
    )
    assert all(value is False for value in execution.values())
    denied = authorization["authorization"]
    assert denied["additional_pair_generation_authorized"] is False
    assert denied["development_tail_materialization_authorized"] is False
    assert denied["train_131k_probe_execution_authorized"] is False
    assert denied["architecture_selection_authorized"] is False
    assert denied["calibration_authorized"] is False
    assert denied["sbc_authorized"] is False
    assert denied["final_evaluation_authorized"] is False
    assert denied["gwosc_gwtc_access_authorized"] is False


def test_probe_reuse_and_architecture_fit_cap_are_exact() -> None:
    architecture = load_yaml(CONFIG_PATH)["post_lock_architecture_contract"]
    assert architecture["flow_transforms"] == [6, 10]
    assert architecture["conditioner_widths"] == [128, 256]
    assert architecture["seeds"] == [0, 1, 2]
    assert architecture["maximum_architecture_results"] == 12
    assert architecture["reuse_131k_probe_fits_for_10_transform_width_256"] is True
    assert architecture["maximum_additional_fits_after_terminal_lock"] == 9
    assert architecture["separate_execution_authorization_required"] is True
