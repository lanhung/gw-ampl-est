import json
from pathlib import Path

import pytest

from gwlens_mm.config import load_yaml, validate_smoke_configuration
from gwlens_mm.provenance import (
    ArtifactChecksum,
    DatasetManifest,
    configuration_hash,
    dataset_id,
    derive_seed,
    reject_duplicate_dataset_ids,
)
from gwlens_mm.schema import FROZEN_SMOKE_SCHEMA_VERSION, SCHEMA_VERSION
from gwlens_mm.storage import strain_storage_estimate

ROOT = Path(__file__).resolve().parents[1]


def test_config_hash_and_seed_derivation_are_deterministic():
    assert configuration_hash({"b": 2, "a": 1}) == configuration_hash({"a": 1, "b": 2})
    seed = derive_seed(1234, "detector_noise", "pair-1", "H1")
    assert seed == derive_seed(1234, "detector_noise", "pair-1", "H1")
    assert seed != derive_seed(1234, "detector_noise", "pair-1", "L1")


def example_manifest(identifier="gwlens-v2-example"):
    return DatasetManifest(
        dataset_id=identifier,
        schema_version="2.0.0-alpha.2",
        generator_git_commit="a" * 40,
        configuration_hash="b" * 64,
        root_seed=1,
        planned_pair_count=1,
        accepted_pair_count=1,
        attempted_pair_count=2,
        pair_ids=("pair-1",),
        source_ids=("source-1",),
        lens_ids=("lens-1",),
        noise_segment_ids=("noise-1", "noise-2"),
        artifacts=(ArtifactChecksum("arrays/strain.zarr", None, None),),
        generation_status="planned",
        dataset_purpose="engineering_smoke",
        scientific_use_authorized=False,
        authorizing_git_commit="c" * 40,
    )


def test_manifest_round_trip_and_duplicate_detection():
    manifest = example_manifest()
    manifest.validate()
    assert DatasetManifest.from_json(manifest.to_json()) == manifest
    with pytest.raises(ValueError, match="duplicate dataset IDs"):
        reject_duplicate_dataset_ids([manifest, example_manifest()])


def test_dataset_id_is_deterministic():
    identifier = dataset_id("2.0.0-alpha.2", "a" * 40, "b" * 64, 10)
    assert identifier == dataset_id("2.0.0-alpha.2", "a" * 40, "b" * 64, 10)


def test_complete_manifest_requires_complete_artifacts_and_target_count():
    pending_artifact = example_manifest()
    data = pending_artifact.to_dict()
    data["generation_status"] = "complete"
    with pytest.raises(ValueError, match="artifact"):
        DatasetManifest.from_json(json.dumps(data))

    data["artifacts"] = [
        {
            "relative_path": "arrays/strain.zarr",
            "sha256": "c" * 64,
            "bytes": 123,
            "status": "complete",
        }
    ]
    data["planned_pair_count"] = 2
    with pytest.raises(ValueError, match="accepted count"):
        DatasetManifest.from_json(json.dumps(data))

    data["planned_pair_count"] = 1
    complete = DatasetManifest.from_json(json.dumps(data))
    assert complete.generation_status == "complete"


@pytest.mark.parametrize(
    ("pairs", "expected_bytes"),
    [
        (48, 48 * 2 * 3 * 4096 * 4 * 3),
        (10_000, 10_000 * 2 * 3 * 4096 * 4 * 3),
        (100_000, 100_000 * 2 * 3 * 4096 * 4 * 3),
    ],
)
def test_storage_size_calculation(pairs, expected_bytes):
    assert strain_storage_estimate(pairs).bytes == expected_bytes


def test_phase1b_smoke_configuration_is_explicitly_authorized():
    config = load_yaml(ROOT / "configs/data/v2_smoke.yaml")
    validate_smoke_configuration(config, expected_execution_authorized=True)
    assert config["execution_authorized"] is True
    assert sum(config["accepted_pairs"].values()) == 48
    assert config["schema_version"] == FROZEN_SMOKE_SCHEMA_VERSION == "2.0.0-alpha.2"
    assert SCHEMA_VERSION == "2.0.0-alpha.3"


def test_engineering_manifest_rejects_scientific_authorization():
    data = example_manifest().to_dict()
    data["scientific_use_authorized"] = True
    with pytest.raises(ValueError, match="cannot be authorized"):
        DatasetManifest.from_json(json.dumps(data))


def phase2_config():
    return load_yaml(ROOT / "configs/statistics/phase2_preregistration.yaml")


def test_phase2_preregistration_is_fail_closed_and_hash_frozen():
    config = phase2_config()
    assert config["status"] == "human_finite_source_support_approved"
    for field in (
        "execution_enabled",
        "scientific_data_generation_authorized",
        "training_authorized",
        "gwtc_gwosc_download_authorized",
        "engineering_smoke_scientific_use_authorized",
    ):
        assert config[field] is False
    assert config["splits"]["real_noise_test"] == 0
    assert configuration_hash(config) == (
        "1403f1f8cf96fbc34c2cfd99928bd7c24b5fde5495e54689d2a5ee7ec250c418"
    )


def test_rc3_source_plane_density_is_exact_and_shared() -> None:
    config = phase2_config()
    measure = config["source_plane_measure"]
    assert measure["support"]["area_dimensionless"] == 25.0
    assert measure["normalized_log_density_formula"] == (
        "-log(25.0)-2*log(einstein_radius_arcsec)"
    )
    theta_e = 1.7
    angular_area = 25.0 * theta_e**2
    angular_density = 1.0 / (25.0 * theta_e**2)
    assert angular_area * angular_density == pytest.approx(1.0)
    source_id = measure["id"]
    assert config["proposal_distribution"]["source_plane_sampling"] == source_id
    assert config["evaluation_distribution"]["source_plane_sampling"] == source_id
    assert measure["density_evaluation_stage"] == (
        "before_lens_multiplicity_and_detection_selection"
    )
    assert measure["support_interpretation"] == "deliberately_truncated_finite_benchmark"
    assert measure["full_multiply_imaged_cross_section_claim"] is False


def test_rc3_solver_and_support_audit_are_frozen() -> None:
    contract = phase2_config()["lens_solver"]["numerical_contract"]
    assert contract["solver"] == "deterministic_union"
    grid = contract["primary_components"][1]
    assert grid["search_window_over_einstein_radius"] == 8.0
    assert grid["minimum_image_separation_over_einstein_radius"] == 0.02
    assert grid["initial_guess_cut"] is False
    assert contract["precision_limit_arcsec"] == 1.0e-10
    assert contract["random_initial_guesses"] == 0
    audit = contract["support_audit"]
    reference_grid = audit["reference_components"][1]
    assert reference_grid["search_window_over_einstein_radius"] == 12.0
    assert reference_grid["minimum_image_separation_over_einstein_radius"] == 0.005
    assert audit["boundary_requirement"] == (
        "identical_primary_reference_classification_and_images"
    )
    assert audit["require_no_multiple_images_on_source_support_boundary"] is False
    assert audit["failure_action"] == "hard_stop_before_microbenchmark"


def test_phase2_counts_and_storage_arithmetic_are_exact():
    config = phase2_config()
    counts = {
        key: value
        for key, value in config["splits"].items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    storage = config["storage_plan"]
    assert sum(counts.values()) == storage["planned_pairs_including_qualification"]
    assert (
        storage["bytes_per_pair_uncompressed"]
        == storage["products_per_pair"] * 2 * 3 * 16384 * 4
    )
    assert (
        storage["raw_array_bytes"]
        == storage["planned_pairs_including_qualification"]
        * storage["bytes_per_pair_uncompressed"]
    )
    assert storage["qualification_raw_bytes"] < 10_000_000_000
    assert storage["projected_remaining_bytes_at_peak"] > 100_000_000_000
    expected_peak = (
        storage["raw_array_bytes"]
        + storage["metadata_and_chunk_overhead_bytes"]
        + storage["retained_failed_shard_cap_bytes"]
        + storage["run_checkpoint_and_cache_reserve_bytes"]
        + storage["active_temporary_shard_bytes"]
    )
    assert storage["projected_peak_bytes"] == expected_peak
    assert (
        storage["minimum_prelaunch_free_bytes"]
        == expected_peak + storage["minimum_post_peak_free_bytes"]
    )


def test_phase2_evaluation_support_is_contained_in_proposal():
    config = phase2_config()
    proposal = config["proposal_distribution"]
    evaluation = config["evaluation_distribution"]
    for field in (
        "lens_redshift",
        "source_redshift",
        "einstein_radius_arcsec",
        "axis_ratio",
        "shear_amplitude",
        "epl_density_slope",
        "external_convergence",
    ):
        assert proposal[field]["minimum"] <= evaluation[field]["minimum"]
        assert proposal[field]["maximum"] >= evaluation[field]["maximum"]


def test_phase2_calibration_and_sbc_splits_are_disjoint_by_design():
    config = phase2_config()
    assert config["splits"]["calibration_fit"] == 6144
    assert config["splits"]["sbc_diagnostic"] == 2048
    assert config["estimator_design"]["posthoc_calibration_source_split"] == (
        "calibration_fit"
    )
    assert config["calibration_protocol"]["simulation_based_calibration"][
        "source_split"
    ] == "sbc_diagnostic"
    assert config["estimator_design"]["sbc_never_fits_calibration_map"] is True


def test_phase2_gold_subsets_separate_development_from_final_evidence():
    gold = phase2_config()["calibration_protocol"]
    development = gold["gold_development_subset"]
    final = gold["gold_final_subset"]
    assert development["source_split"] == "validation"
    assert development["may_trigger_method_revision"] is True
    assert final["source_split"] == "iid_test"
    assert final["may_trigger_method_revision"] is False
    assert final["report_once_after_freeze"] is True


def test_phase2_architecture_selection_never_picks_a_best_seed():
    design = phase2_config()["estimator_design"]
    selection = design["architecture_selection"]
    assert design["training_seeds"] == [0, 1, 2]
    assert selection["aggregate_across_seeds"] == (
        "mean_validation_negative_log_probability"
    )
    assert selection["do_not_select_best_seed"] is True
    assert selection["report_all_seed_results"] is True


def test_phase2_diagnostic_sets_are_fully_counted_and_named():
    config = phase2_config()
    definitions = config["diagnostic_set_definitions"]
    assert len(definitions["balanced_tail_diagnostic"]["strata"]) == 4
    assert len(definitions["cross_family_misspecification_test"]["cells"]) == 4
    assert len(definitions["parameter_region_ood"]["strata"]) == 4
    assert definitions["waveform_mismatch_test"]["cases"] == 2048
    assert definitions["psd_mismatch_test"]["cases"] == 2048
    assert "lens_family_ood" not in config["splits"]
    assert "waveform_psd_mismatch_test" not in config["splits"]


def test_phase2_em_cells_are_explicit_and_environment_aware():
    model = phase2_config()["em_observation_model"]
    cells = model["availability_cells"]
    assert len(cells) == 8
    for cell in cells.values():
        assert isinstance(cell["modalities"], list)
        assert "timing_std_seconds" in cell
        assert cell["environment_state"] in {"informative", "weak", "unavailable"}
        assert "velocity_dispersion_fractional_std" in cell
        assert cell["uncertainty_covariance"].startswith("diagonal")


def test_phase2_source_mass_support_prevents_invalid_secondaries():
    source = phase2_config()["source_population"]
    assert source["mass_ratio"]["distribution"] == "conditional_uniform"
    assert source["secondary_mass_source_frame_minimum_msun"] == 10.0
    assert source["invalid_secondary_mass_policy"] == (
        "impossible_by_conditional_support_not_rejected"
    )
