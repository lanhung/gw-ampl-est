from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from gwlens_mm.config import load_yaml
from gwlens_mm.provenance import configuration_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/statistics/adaptive_scientific_production_preregistration.yaml"
EXPECTED_HASH = "ba5dae2aa769331b917d3f622bfc967c607700f9908521576301841cb71d804b"


def adaptive_config() -> dict[str, Any]:
    return load_yaml(CONFIG_PATH)


def test_phase3b_version_hash_and_parent_are_frozen() -> None:
    config = adaptive_config()
    assert config["preregistration_version"] == "1.1.0-rc.1"
    assert configuration_hash(config) == EXPECTED_HASH
    assert config["parent_preregistration"]["canonical_hash"] == (
        "4dde279cf1bea78d1ddbd4fab99d88e88e334c80c180dc7850679736c5e53edb"
    )


def test_phase3b_is_design_only_and_every_execution_gate_is_closed() -> None:
    authorization = adaptive_config()["authorization"]
    assert authorization["design_work_authorized"] is True
    for key, value in authorization.items():
        if key != "design_work_authorized":
            assert value is False, key


def test_phase3a_artifact_is_permanently_excluded() -> None:
    config = adaptive_config()
    evidence = config["phase3a_qualification_evidence"]
    grouping = config["group_assignment"]
    assert evidence["accepted_pair_count"] == 4096
    assert evidence["scientific_use_authorized"] is False
    assert evidence["permanent_exclusion_from_all_scientific_splits"] is True
    assert evidence["dataset_id"] in grouping["phase3a_dataset_ids_forbidden"]
    assert len(evidence["forbidden_id_prefixes"]) == 5


def test_training_ladder_is_strictly_nested_and_capped() -> None:
    ladder = adaptive_config()["nested_training_ladder"]
    rungs = ladder["rungs"]
    assert list(rungs.values()) == [16384, 32768, 65536]
    assert rungs["train_16k"] < rungs["train_32k"] < rungs["train_65k"]
    assert ladder["membership_rule"] == "stable_rank_less_than_rung_count"
    assert ladder["automatic_extension_above_65536"] is False
    assert ladder["above_65536_action"] == "hard_stop_and_new_preregistration"


def test_development_and_final_pool_arithmetic_and_roles() -> None:
    config = adaptive_config()
    development = config["development_pool"]
    final = config["final_evaluation_pool"]
    development_total = sum(item["count"] for item in development["splits"].values())
    final_total = sum(item["count"] for item in final["splits"].values())
    assert development_total == development["total"] == 12288
    assert final_total == final["total"] == 20480
    assert development["group_disjoint"] is True
    assert final["group_disjoint"] is True
    assert final["group_disjoint_from_training_and_development"] is True
    assert "learning_curve_stopping" in development["splits"]["validation"][
        "allowed_uses"
    ]
    assert "learning_curve_stopping" in development["splits"]["calibration_fit"][
        "forbidden_uses"
    ]
    assert "calibration_fit" in development["splits"]["sbc_diagnostic"][
        "forbidden_uses"
    ]


def test_total_counts_exclude_phase3a_and_are_exact() -> None:
    config = adaptive_config()
    totals = config["total_scientific_counts_by_stop"]
    development = config["development_pool"]["total"]
    final = config["final_evaluation_pool"]["total"]
    for rung, count in config["nested_training_ladder"]["rungs"].items():
        assert totals[rung] == count + development + final
    assert totals["phase3a_qualification_excluded_from_totals"] is True


def test_final_evaluation_cannot_affect_scale_selection() -> None:
    config = adaptive_config()
    final = config["final_evaluation_pool"]
    probe = config["learning_curve_probe"]
    final_names = set(final["splits"])
    assert final["sealed_during_scale_and_architecture_selection"] is True
    assert final["early_materialization_authorized"] is False
    assert final_names <= set(probe["forbidden_metric_sources"])
    assert config["stopping_rule"]["final_evaluation_may_affect_stopping"] is False


def test_stopping_rule_is_fail_closed_at_32k_and_65k() -> None:
    rule = adaptive_config()["stopping_rule"]
    first = rule["comparison_16k_to_32k"]
    second = rule["comparison_32k_to_65k"]
    assert first["require_all_conditions"] is True
    assert first["gray_zone_action"] == "continue_to_train_65k"
    assert first["nlp_improvement_ci_upper_max_nat_per_target_dimension"] == 0.01
    assert first["median_crps_relative_improvement_max"] == 0.01
    assert first["maximum_marginal_coverage_error_improvement_max"] == 0.005
    assert first["maximum_em_cell_coverage_degradation"] == 0.02
    assert second["meaningful_improvement_action"] == (
        "stop_data_limited_and_new_preregistration"
    )
    assert second["gray_zone_action"] == "stop_inconclusive_and_new_preregistration"


def test_calibration_sbc_and_architecture_gates_remain_separate() -> None:
    config = adaptive_config()
    development = config["development_pool"]["splits"]
    architecture = config["final_architecture_selection"]
    assert development["calibration_fit"]["count"] == 4096
    assert development["sbc_diagnostic"]["count"] == 2048
    assert architecture["begins_only_after_training_size_lock"] is True
    assert architecture["best_seed_selection_forbidden"] is True
    assert architecture["maximum_fits"] == 12
    assert architecture["open_calibration_sbc_or_final_evaluation_automatically"] is False


def test_proposal_v2_gate_is_support_preserving_and_unauthorized() -> None:
    gate = adaptive_config()["proposal_efficiency_future_gate"]
    mixture = gate["candidate_mixture"]
    adoption = gate["adoption_gates"]
    assert gate["accepted_pair_count"] == 512
    assert gate["authorized_in_phase3b"] is False
    assert gate["scientific_use_authorized"] is False
    assert gate["evaluation_target_unchanged"] is True
    assert math.isclose(
        mixture["efficient_component_weight"]
        + mixture["rc5_broad_support_safety_component_weight"],
        1.0,
    )
    assert mixture["rc5_broad_support_safety_component_weight"] > 0
    assert adoption["minimum_acceptance_or_throughput_ratio_vs_rc5"] == 2.0
    assert adoption["all_importance_weights_finite"] is True


def test_resource_projections_reproduce_phase3a_linear_model() -> None:
    resource = adaptive_config()["resource_projection"]
    baseline = resource["baseline"]
    peak_model = resource["peak_storage_model"]
    for total in (49152, 65536, 98304):
        projection = resource["rc5_baseline_projections"][f"total_{total}"]
        expected_attempts = math.ceil(total * baseline["attempts"] / 4096)
        expected_published = math.ceil(total * baseline["published_bytes"] / 4096)
        expected_peak = math.ceil(
            expected_published
            * (1 + peak_model["retained_failed_evidence_fraction"])
            + peak_model["run_checkpoint_and_cache_reserve_bytes"]
            + peak_model["active_shard_bytes"]
        )
        assert projection["projected_attempts"] == expected_attempts
        assert projection["projected_published_bytes"] == expected_published
        assert projection["projected_peak_bytes"] == expected_peak
        assert projection["projected_remaining_free_bytes"] == (
            baseline["free_bytes_after_phase3a"] - expected_peak
        )
        assert projection["projected_remaining_free_bytes"] > 100_000_000_000
    hypothetical = resource["proposal_v2_hypothetical_two_x_scenario"]
    assert hypothetical["measured"] is False
    assert hypothetical["may_be_reported_as_completed_measurement"] is False


def test_real_noise_catalog_and_later_phase_remain_closed() -> None:
    config = adaptive_config()
    protocol = config["real_noise_and_catalog_future_protocol"]
    authorization = config["authorization"]
    assert protocol["authorized_in_phase3b"] is False
    assert protocol["separate_authorization_required"] is True
    assert protocol["proposed_91_event_count_is_fixed_fact"] is False
    assert authorization["gwosc_gwtc_access_authorized"] is False
    assert authorization["phase3c_or_later_authorized"] is False
