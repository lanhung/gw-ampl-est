from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from gwlens_mm.config import load_yaml
from gwlens_mm.provenance import configuration_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/statistics/adaptive_scientific_production_preregistration.yaml"
COMMITMENT_PATH = ROOT / "results/phase3b/final_evaluation_commitment.json"
EXPECTED_HASH = "b94e7733d7fbb6f4c9dc4d5842b6a87f29e0515b4047b7b1604bca1438d15805"
EXPECTED_COMMITMENT_HASH = (
    "29a1b8487679e7f6a671395e47288c6bace45eb21b141f0ec94940391d14f272"
)


def adaptive_config() -> dict[str, Any]:
    return load_yaml(CONFIG_PATH)


def commitment() -> dict[str, Any]:
    return json.loads(COMMITMENT_PATH.read_text(encoding="utf-8"))


def test_phase3b1_version_hash_parent_and_review_status_are_frozen() -> None:
    config = adaptive_config()
    assert config["preregistration_version"] == "1.1.0-rc.2"
    assert config["status"] == "awaiting_human_review"
    assert configuration_hash(config) == EXPECTED_HASH
    assert config["parent_preregistration"]["canonical_hash"] == (
        "4dde279cf1bea78d1ddbd4fab99d88e88e334c80c180dc7850679736c5e53edb"
    )


def test_every_execution_gate_remains_closed() -> None:
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


def test_16k_is_probe_only_and_only_32k_or_65k_can_lock() -> None:
    ladder = adaptive_config()["nested_training_ladder"]
    rungs = ladder["rungs"]
    assert rungs == {
        "train_16k_probe_subset": 16384,
        "train_32k": 32768,
        "train_65k": 65536,
    }
    assert ladder["independently_lockable_final_rungs"] == [
        "train_32k",
        "train_65k",
    ]
    assert ladder["train_16k_probe_subset_is_final_lock"] is False
    assert ladder["automatic_extension_above_65536"] is False


def test_only_achievable_final_totals_are_65536_and_98304() -> None:
    totals = adaptive_config()["total_scientific_counts_by_stop"]
    assert totals["achievable_final_totals"] == [65536, 98304]
    assert totals["train_32k"] == 32768 + 12288 + 20480 == 65536
    assert totals["train_65k"] == 65536 + 12288 + 20480 == 98304
    assert totals["train_16k_probe_subset_final_total"] is None
    assert 49152 not in totals.values()


def test_staged_materialization_arithmetic_and_order_are_exact() -> None:
    sequence = adaptive_config()["materialization_sequence"]
    stage_a = sequence["stage_a_scale_selection"]
    stage_b = sequence["stage_b_conditional_extension"]
    stage_c = sequence["stage_c_post_lock"]
    assert stage_a["train_32k"] + stage_a["validation"] == stage_a["total"] == 38912
    assert stage_a["generation_continues_to_train_32k_regardless_of_16k_probe_result"]
    assert stage_b["additional_training_systems"] == 32768
    assert stage_b["automatic"] is False
    assert (
        stage_c["calibration_fit"]
        + stage_c["sbc_diagnostic"]
        + stage_c["final_evaluation"]
        == stage_c["total"]
        == 26624
    )
    assert stage_c["requires_training_size_and_architecture_lock"] is True


def test_changed_training_proposal_requires_weighted_target_correction() -> None:
    targets = adaptive_config()["scientific_sampling_targets"]
    correction = targets["training_target_correction"]
    assert targets["evaluation_target_id"] == "balanced_literature_informed_benchmark_v1"
    assert correction["required_when_training_proposal_differs_from_evaluation_target"]
    assert correction["objective"] == "importance_weighted_conditional_negative_log_probability"
    assert "log_p_eval" in correction["log_weight_formula"]
    assert "log_q_train" in correction["log_weight_formula"]
    assert correction["full_latent_proposal_and_evaluation_variables_required"]
    assert correction["normalized_globally_within_each_rung_to_mean"] == 1.0
    assert correction["clipping_authorized"] is False
    assert correction["deployable_model_input"] is False
    assert correction["identical_semantics_across_rungs"] is True


def test_validation_calibration_sbc_and_iid_are_direct_target_draws() -> None:
    targets = adaptive_config()["scientific_sampling_targets"]
    direct = targets["direct_target_splits"]
    assert set(direct) == {"validation", "calibration_fit", "sbc_diagnostic", "iid_test"}
    assert all(value.startswith("direct_evaluation") for value in direct.values())
    assert targets["sbc_uncorrected_proposal_draws_forbidden"] is True
    probe_forbidden = set(adaptive_config()["learning_curve_probe"]["forbidden_metric_sources"])
    assert {"calibration_fit", "sbc_diagnostic", "iid_test"} <= probe_forbidden


def test_final_commitment_freezes_generation_rules_not_unknown_ids() -> None:
    config = adaptive_config()
    final = config["final_evaluation_pool"]
    item = commitment()
    assert final["concrete_accepted_ids_known_before_materialization"] is False
    assert final["deterministic_generation_commitment_required_before_training"] is True
    assert item["accepted_ids_known_before_materialization"] is False
    assert item["future_scientific_generator_commit"]["value"] is None
    assert item["must_be_finalized_and_hashed_before_training"] is True
    assert item["preregistration"]["canonical_hash"] == EXPECTED_HASH
    assert sum(
        count for name, count in item["split_counts"].items() if name != "total"
    ) == item["split_counts"]["total"] == 20480
    assert "attempt_id_allocation_rule" in item
    assert "accepted_rank_allocation_rule" in item
    assert hashlib.sha256(COMMITMENT_PATH.read_bytes()).hexdigest() == EXPECTED_COMMITMENT_HASH


def test_proposal_adoption_requires_two_x_throughput_lower_bound() -> None:
    gate = adaptive_config()["proposal_efficiency_future_gate"]
    adoption = gate["adoption_gates"]
    assert gate["authorized_in_phase3b"] is False
    assert adoption["primary_mandatory_endpoint"] == (
        "accepted_pairs_per_active_hour_ratio_vs_rc5"
    )
    assert adoption["confidence_level"] == 0.95
    assert adoption["minimum_lower_confidence_bound"] == 2.0
    assert adoption["acceptance_gain_without_throughput_gain_authorizes_adoption"] is False
    assert "acceptance_rate_ratio" in adoption["secondary_endpoints"]
    ab_design = adoption["ab_design"]
    assert ab_design["paired_blocks_per_arm"] == 16
    assert ab_design["accepted_pairs_per_block"] == 32
    assert ab_design["paired_blocks_per_arm"] * ab_design["accepted_pairs_per_block"] == 512
    assert ab_design["bootstrap"]["replicates"] == 10000
    assert ab_design["bootstrap"]["resampling_unit"] == "matched_block_index"
    assert ab_design["endpoint_switching_after_results_forbidden"] is True


def test_executable_proposal_specification_precedes_any_512_pair_authorization() -> None:
    gate = adaptive_config()["proposal_efficiency_future_gate"]
    specification = gate["executable_specification_required_before_authorization"]
    mixture = gate["candidate_mixture"]
    assert specification["separately_reviewed"] is True
    assert len(specification["requirements"]) == 10
    assert "exact_normalized_log_density_evaluator" in specification["requirements"]
    assert specification["conceptual_component_names_are_not_executable_specification"]
    assert math.isclose(
        mixture["efficient_component_weight"]
        + mixture["rc5_broad_support_safety_component_weight"],
        1.0,
    )
    assert mixture["rc5_broad_support_safety_component_weight"] == 0.2


def test_probe_fits_are_reused_and_only_nine_new_fits_are_allowed() -> None:
    architecture = adaptive_config()["final_architecture_selection"]
    assert architecture["maximum_architecture_results"] == 12
    assert architecture["reuse_probe_fits_at_locked_rung"] is True
    assert architecture["maximum_new_architecture_fits_after_lock"] == 9
    assert architecture["identical_probe_retraining_without_declared_failure_forbidden"]
    assert architecture["best_seed_selection_forbidden"] is True


def test_physical_system_and_noise_realization_semantics_are_explicit() -> None:
    semantics = adaptive_config()["physical_system_and_noise_semantics"]
    future = semantics["future_augmentation_requirements"]
    assert semantics["independent_sample_unit"] == "accepted_physical_system"
    assert semantics["stored_gaussian_noise_realizations_per_accepted_physical_system"] == 1
    assert semantics["additional_training_time_noise_augmentation_authorized"] is False
    assert future["inherit_parent_physical_system_split"] is True
    assert future["count_as_additional_physical_system"] is False
    assert future["identical_policy_across_training_rungs"] is True


def test_resource_projections_cover_stages_and_only_achievable_totals() -> None:
    resource = adaptive_config()["resource_projection"]
    baseline = resource["baseline"]
    stages = resource["staged_rc5_baseline_projections"]
    for row in stages.values():
        count = row["incremental_accepted_systems"]
        assert row["projected_attempts"] == math.ceil(count * baseline["attempts"] / 4096)
        assert row["projected_published_bytes"] == math.ceil(
            count * baseline["published_bytes"] / 4096
        )
    finals = resource["achievable_final_rc5_baseline_projections"]
    assert set(finals) == {"total_65536", "total_98304"}
    hypothetical = resource["proposal_v2_hypothetical_training_only_two_x_scenario"]
    assert hypothetical["measured"] is False
    assert "direct_target_nontraining_splits_unchanged" in hypothetical["assumption"]
    assert hypothetical["may_be_reported_as_completed_measurement"] is False


def test_final_evaluation_and_external_work_remain_closed() -> None:
    config = adaptive_config()
    final = config["final_evaluation_pool"]
    protocol = config["real_noise_and_catalog_future_protocol"]
    assert final["sealed_during_scale_and_architecture_selection"] is True
    assert final["early_materialization_authorized"] is False
    assert config["stopping_rule"]["final_evaluation_may_affect_stopping"] is False
    assert protocol["authorized_in_phase3b"] is False
    assert config["authorization"]["gwosc_gwtc_access_authorized"] is False
    assert config["authorization"]["phase3c_or_later_authorized"] is False
