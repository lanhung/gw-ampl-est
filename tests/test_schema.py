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
