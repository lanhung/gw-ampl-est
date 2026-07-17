from __future__ import annotations

from pathlib import Path

import pytest

from gwlens_mm.config import load_yaml
from gwlens_mm.provenance import configuration_hash
from gwlens_mm.training.architecture import (
    ALL_ARCHITECTURE_IDS,
    NEW_ARCHITECTURE_IDS,
    PROBE_ARCHITECTURE_ID,
    candidate_model_configuration,
    load_architecture_specs,
    select_architecture,
    validate_architecture_execution_gate,
)
from gwlens_mm.training.contracts import TrainingGateError, model_configuration_hash

ROOT = Path(__file__).resolve().parents[1]
GRID_HASH = "abb3ef575e0f37a8f0150169391efb350b1c53893508bf8ba2505f9219075355"


def test_architecture_grid_is_exact_and_candidate_hashes_are_distinct() -> None:
    assert configuration_hash(
        load_yaml(ROOT / "configs/models/phase5_architecture_grid.yaml")
    ) == GRID_HASH
    specifications = load_architecture_specs(ROOT)
    assert tuple(spec.architecture_id for spec in specifications) == (
        "nsf-t06-w128",
        "nsf-t06-w256",
        "nsf-t10-w128",
        "nsf-t10-w256",
    )
    assert sum(spec.reused_probe for spec in specifications) == 1
    assert next(spec for spec in specifications if spec.reused_probe).architecture_id == (
        PROBE_ARCHITECTURE_ID
    )
    hashes = {
        model_configuration_hash(candidate_model_configuration(ROOT, specification))
        for specification in specifications
    }
    assert len(hashes) == 4
    assert set(NEW_ARCHITECTURE_IDS) == {
        spec.architecture_id for spec in specifications if not spec.reused_probe
    }


def test_candidate_configuration_changes_only_frozen_grid_axes_and_identity() -> None:
    base = load_yaml(ROOT / "configs/models/phase4_probe_nsf.yaml")
    specification = next(
        spec
        for spec in load_architecture_specs(ROOT)
        if spec.architecture_id == "nsf-t06-w128"
    )
    candidate = candidate_model_configuration(ROOT, specification)
    assert candidate["architecture"]["flow"]["transforms"] == 6
    assert candidate["architecture"]["flow"]["conditioner_width"] == 128
    assert candidate["implementation_id"] == "phase5_nsf-t06-w128_v1"
    restored = dict(candidate)
    restored["implementation_id"] = base["implementation_id"]
    restored["architecture"] = dict(candidate["architecture"])
    restored["architecture"]["flow"] = dict(candidate["architecture"]["flow"])
    restored["architecture"]["flow"]["transforms"] = 10
    restored["architecture"]["flow"]["conditioner_width"] = 256
    assert restored == base


def _results(*, tie: bool = False) -> list[dict[str, object]]:
    rows = []
    means = {
        "nsf-t06-w128": 0.20,
        "nsf-t06-w256": 0.19,
        "nsf-t10-w128": 0.18,
        "nsf-t10-w256": 0.17,
    }
    parameters = {
        "nsf-t06-w128": 100,
        "nsf-t06-w256": 200,
        "nsf-t10-w128": 150,
        "nsf-t10-w256": 300,
    }
    for architecture in ALL_ARCHITECTURE_IDS:
        for seed in (0, 1, 2):
            rows.append(
                {
                    "architecture_id": architecture,
                    "seed": seed,
                    "trainable_parameter_count": parameters[architecture],
                    "development_mean_nlp_nat_per_target_dimension": (
                        0.2 if tie else means[architecture] + seed * 0.001
                    ),
                }
            )
    return rows


def test_selection_uses_three_seed_mean_and_never_selects_a_seed() -> None:
    result = select_architecture(_results())
    assert result["selected_architecture_id"] == PROBE_ARCHITECTURE_ID
    assert result["best_seed_selected"] is False
    assert result["new_fit_count"] == 9
    assert result["reused_probe_fit_count"] == 3
    assert result["calibration_accessed"] is False
    assert result["final_evaluation_accessed"] is False


def test_exact_tie_uses_lower_parameter_count() -> None:
    result = select_architecture(_results(tie=True))
    assert result["selected_architecture_id"] == "nsf-t06-w128"


def test_selection_rejects_missing_seed_or_best_seed_shortcut() -> None:
    rows = _results()
    rows.pop()
    with pytest.raises(TrainingGateError, match="exactly twelve"):
        select_architecture(rows)


def test_implementation_gate_cannot_execute_architecture_selection(tmp_path: Path) -> None:
    authorization_path = (
        ROOT / "configs/execution/phase5_architecture_selection_stack_authorization.yaml"
    )
    authorization = load_yaml(authorization_path)
    assert authorization["authorization_status"] == "authorized_implementation_only"
    assert authorization["authorization"]["scientific_data_access_authorized"] is False
    assert authorization["authorization"]["architecture_fit_execution_authorized"] is False
    assert (
        authorization["authorization"]["architecture_selection_execution_authorized"]
        is False
    )
    with pytest.raises(TrainingGateError, match="execution authorization is absent"):
        validate_architecture_execution_gate(
            ROOT,
            authorization_path=authorization_path,
            stage_a_publication_root=tmp_path / "stage-a",
            stage_b_publication_root=tmp_path / "stage-b",
            combined_publication_root=tmp_path / "combined",
            terminal_decision_path=tmp_path / "decision.json",
            probe_output_root=tmp_path / "probe",
        )
