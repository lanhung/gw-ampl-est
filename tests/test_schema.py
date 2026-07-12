import json
from pathlib import Path

import pytest

from gwlens_mm.schema import V2Record, v2_json_schema, validate_covariance

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/v2_metadata_example.json"


def test_metadata_example_round_trip_without_field_loss():
    original = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    record = V2Record.from_dict(original)
    assert record.to_dict() == original
    assert V2Record.from_json(record.to_json()).to_dict() == original


def test_schema_artifact_matches_generated_schema():
    artifact = json.loads((ROOT / "examples/v2_metadata_schema.json").read_text(encoding="utf-8"))
    assert artifact == v2_json_schema()


def test_multi_image_truth_keeps_selected_pair_separate():
    record = V2Record.from_json(EXAMPLE.read_text(encoding="utf-8"))
    assert len(record.lens_truth.physical_images) == 3
    assert {record.pair.primary_image_id, record.pair.secondary_image_id} == {"image_0", "image_2"}
    assert record.lens_truth.censored_image_ids == ("image_3",)
    assert {item.image_id for item in record.em_observation.observed_image_astrometry} == {
        "image_0",
        "image_2",
        "image_3",
    }


@pytest.mark.parametrize(
    "matrix",
    [
        [[1.0, 0.1], [0.2, 1.0]],
        [[1.0, 2.0], [2.0, 1.0]],
        [[0.0, 0.0], [0.0, 1.0]],
    ],
)
def test_invalid_covariances_are_rejected(matrix):
    with pytest.raises(ValueError):
        validate_covariance(matrix, 2)


def test_missing_modality_mask_mismatch_is_rejected():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["em_observation"]["modality_availability_mask"]["velocity_dispersion"] = True
    with pytest.raises(ValueError, match="modality mask"):
        V2Record.from_dict(data)


def test_detector_mask_rejects_an_unobserved_selected_image():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["gw_observation"]["detector_availability_mask"][1] = [False, False, False]
    with pytest.raises(ValueError, match="at least one available detector"):
        V2Record.from_dict(data)


def test_nonfinite_lens_parameter_is_rejected():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["lens_truth"]["lens_parameters"]["axis_ratio"] = float("nan")
    with pytest.raises(ValueError, match="lens parameter"):
        V2Record.from_dict(data)


def test_astrometry_rejects_unknown_image_id():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["em_observation"]["observed_image_astrometry"][0]["image_id"] = "unknown"
    with pytest.raises(ValueError, match="unknown physical images"):
        V2Record.from_dict(data)


def test_astrometry_rejects_duplicate_image_id():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["em_observation"]["observed_image_astrometry"][1]["image_id"] = "image_0"
    with pytest.raises(ValueError, match="must be unique"):
        V2Record.from_dict(data)


def test_missing_astrometry_stays_null_and_masked_without_truth_imputation():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["em_observation"]["observed_image_astrometry"] = None
    data["em_observation"]["modality_availability_mask"]["image_astrometry"] = False
    record = V2Record.from_dict(data)
    assert record.em_observation.observed_image_astrometry is None


def test_astrometry_rejects_invalid_covariance_shape():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["em_observation"]["observed_image_astrometry"][0]["covariance_arcsec2"] = [[1.0]]
    with pytest.raises(ValueError, match="covariance must have shape"):
        V2Record.from_dict(data)


def test_implicit_position_only_astrometry_is_rejected():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["em_observation"]["observed_image_positions_arcsec"] = [[1.0, 0.0]]
    with pytest.raises(ValueError, match="implicit positional astrometry"):
        V2Record.from_dict(data)


def test_every_extra_physical_image_requires_explicit_status():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["lens_truth"]["censored_image_ids"] = []
    with pytest.raises(ValueError, match="every non-selected physical image"):
        V2Record.from_dict(data)


def test_censored_image_requires_reason():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["lens_truth"]["physical_images"][2]["censoring_reason"] = None
    with pytest.raises(ValueError, match="censoring reason"):
        V2Record.from_dict(data)


def test_noise_references_align_with_detector_mask():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["provenance"]["detector_noise_references"][0]["available"] = False
    data["provenance"]["detector_noise_references"][0]["segment_id"] = None
    data["provenance"]["detector_noise_references"][0]["noise_source"] = "unavailable"
    with pytest.raises(ValueError, match="availability disagrees"):
        V2Record.from_dict(data)


def test_unavailable_noise_reference_cannot_have_segment_id():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["provenance"]["detector_noise_references"][2]["segment_id"] = "not-allowed"
    with pytest.raises(ValueError, match="must not have a segment ID"):
        V2Record.from_dict(data)


def test_detector_noise_grid_order_is_explicit():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    references = data["provenance"]["detector_noise_references"]
    references[0], references[1] = references[1], references[0]
    with pytest.raises(ValueError, match="image-major"):
        V2Record.from_dict(data)


def test_bare_time_difference_is_rejected():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["gw_observation"]["observed_event_time_difference_seconds"] = 1.0
    with pytest.raises(ValueError, match="bare observed time differences"):
        V2Record.from_dict(data)


def test_zero_timing_uncertainty_requires_explicit_control():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    timing = data["gw_observation"]["observed_time_difference"]
    timing["standard_deviation_seconds"] = 0.0
    with pytest.raises(ValueError, match="deterministic_control"):
        V2Record.from_dict(data)
    timing["deterministic_control"] = True
    V2Record.from_dict(data)


@pytest.mark.parametrize(
    ("primary_definition", "expected_message"),
    [
        ("earliest_arriving", "earliest-arriving primary"),
        ("brightest", "brightest primary"),
        ("minimum_image", "minimum-image primary"),
    ],
)
def test_inconsistent_primary_definition_is_rejected(primary_definition, expected_message):
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["pair"]["primary_definition"] = primary_definition
    if primary_definition == "earliest_arriving":
        data["lens_truth"]["physical_images"][0]["arrival_time_seconds"] = 100000.0
    elif primary_definition == "brightest":
        secondary = data["lens_truth"]["physical_images"][1]
        secondary.update(mu_signed=-4.0, mu_abs=4.0, amplitude_factor=2.0)
    else:
        data["lens_truth"]["physical_images"][0]["morse_class"] = "maximum"
    with pytest.raises(ValueError, match=expected_message):
        V2Record.from_dict(data)


def test_catalog_anchor_imposes_no_brightness_or_arrival_constraint():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["pair"]["primary_definition"] = "catalog_anchor"
    primary = data["lens_truth"]["physical_images"][0]
    primary.update(
        mu_signed=1.0,
        mu_abs=1.0,
        amplitude_factor=1.0,
        arrival_time_seconds=100000.0,
    )
    V2Record.from_dict(data)


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("einstein_radius_arcsec", 0.0, "Einstein radius"),
        ("lens_redshift", -0.1, "lens redshift"),
        ("source_redshift", -0.1, "source redshift"),
    ],
)
def test_em_scalar_physical_domains(field, invalid_value, message):
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["em_observation"][field]["value"] = invalid_value
    with pytest.raises(ValueError, match=message):
        V2Record.from_dict(data)


def test_velocity_dispersion_must_be_positive():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["em_observation"]["velocity_dispersion_km_s"] = {
        "value": 0.0,
        "standard_deviation": 10.0,
    }
    data["em_observation"]["modality_availability_mask"]["velocity_dispersion"] = True
    with pytest.raises(ValueError, match="velocity dispersion"):
        V2Record.from_dict(data)


def test_redshift_ordering_is_a_quality_flag_not_hard_rejection():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["em_observation"]["lens_redshift"]["value"] = 1.5
    data["em_observation"]["redshift_ordering_valid"] = False
    record = V2Record.from_dict(data)
    assert record.em_observation.redshift_ordering_valid is False
