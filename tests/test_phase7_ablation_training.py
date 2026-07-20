from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gwlens_mm.config import load_yaml
from gwlens_mm.schema import SplitName
from gwlens_mm.training.ablations import (
    ABLATION_VIEWS,
    AblatedDevelopmentDataset,
    AblatedStandardizedDataset,
    _validate_terminal_size_lock,
    ablation_model_configuration,
    apply_ablation_view,
    summarize_ablation_results,
    validate_ablation_stack_contract,
    validate_ablation_training_execution_gate,
)
from gwlens_mm.training.architecture import selected_model_configuration
from gwlens_mm.training.contracts import TrainingGateError, model_configuration_hash
from gwlens_mm.training.data import DevelopmentCase
from gwlens_mm.training.features import InputStandardizer, PreparedExample

ROOT = Path(__file__).resolve().parents[1]


def _example() -> PreparedExample:
    return PreparedExample(
        gw_strain=np.arange(48, dtype=np.float32).reshape(2, 3, 8),
        detector_mask=np.ones((2, 3), dtype=np.float32),
        astrometry_items=np.arange(45, dtype=np.float32).reshape(5, 9) + 1.0,
        astrometry_mask=np.ones(5, dtype=np.float32),
        scalar_features=np.arange(1, 23, dtype=np.float32),
        scalar_mask=np.ones(22, dtype=np.float32),
        modality_mask=np.ones(8, dtype=np.float32),
        lens_family_condition=np.asarray((1.0, 0.0), dtype=np.float32),
        target=np.asarray((2.0, 3.0), dtype=np.float32),
        physical_system_id="physical-ablation-a",
        lens_family="sie_external_shear",
        em_cell_signature="all",
        em_cell="all_modalities",
    )


def _standardizer() -> InputStandardizer:
    return InputStandardizer(
        scalar_mean=tuple(float(value) for value in range(1, 23)),
        scalar_standard_deviation=(2.0,) * 22,
        astrometry_mean=(1.0, 2.0, 3.0, 4.0, 5.0),
        astrometry_standard_deviation=(2.0,) * 5,
    )


class _FakeDataset:
    expected_split = SplitName.VALIDATION

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> PreparedExample:
        assert index == 0
        return _example()

    def development_case(self, index: int) -> DevelopmentCase:
        assert index == 0
        return DevelopmentCase(example=_example(), tail_view="high_magnification")


def test_implementation_gate_preserves_rc6_and_all_execution_flags_closed() -> None:
    contract = validate_ablation_stack_contract(ROOT)
    assert contract["addendum"]["preregistration_version"] == "1.1.0-rc.6"
    assert tuple(
        contract["authorization"]["implementation_contract"]["ablation_views"]
    ) == ABLATION_VIEWS
    flags = contract["authorization"]["authorization"]
    assert flags["ablation_view_implementation_authorized"] is True
    assert flags["fail_closed_runner_implementation_authorized"] is True
    assert flags["synthetic_fixture_tests_authorized"] is True
    assert all(
        value is False
        for key, value in flags.items()
        if key
        not in {
            "ablation_view_implementation_authorized",
            "fail_closed_runner_implementation_authorized",
            "synthetic_fixture_tests_authorized",
        }
    )


def test_ablation_model_changes_only_identity_and_input_view_metadata() -> None:
    architecture = "nsf-t06-w128"
    primary = selected_model_configuration(ROOT, architecture)
    hashes = set()
    for view in ABLATION_VIEWS:
        ablation = ablation_model_configuration(
            ROOT, architecture_id=architecture, view=view
        )
        assert ablation["architecture"] == primary["architecture"]
        assert ablation["optimization"] == primary["optimization"]
        assert ablation["development_evaluation"] == primary["development_evaluation"]
        assert ablation["ablation"]["target_changed"] is False
        assert ablation["ablation"]["optimizer_or_budget_changed"] is False
        assert ablation["ablation"]["base_model_configuration_hash"] == (
            model_configuration_hash(primary)
        )
        hashes.add(model_configuration_hash(ablation))
    assert len(hashes) == 2


def test_views_apply_after_primary_standardization_and_preserve_labels() -> None:
    source = _example()
    standardizer = _standardizer()
    standardized = standardizer.transform(source)
    for view in ABLATION_VIEWS:
        dataset = AblatedStandardizedDataset(_FakeDataset(), standardizer, view)  # type: ignore[arg-type]
        observed = dataset[0]
        expected = apply_ablation_view(standardized, view)
        for field in (
            "gw_strain",
            "detector_mask",
            "astrometry_items",
            "astrometry_mask",
            "scalar_features",
            "scalar_mask",
            "modality_mask",
            "lens_family_condition",
            "target",
        ):
            assert np.array_equal(getattr(observed, field), getattr(expected, field))
        assert observed.physical_system_id == source.physical_system_id
        assert observed.em_cell == source.em_cell
    with pytest.raises(TrainingGateError, match="unknown ablation"):
        apply_ablation_view(source, "combined")


def test_development_wrapper_preserves_tail_metadata_outside_model_inputs() -> None:
    wrapped = AblatedDevelopmentDataset(
        _FakeDataset(), _standardizer(), "em_only"  # type: ignore[arg-type]
    )
    case = wrapped[0]
    assert case.tail_view == "high_magnification"
    assert np.all(case.example.gw_strain == 0.0)
    assert np.all(case.example.detector_mask == 0.0)
    assert np.array_equal(case.example.target, _example().target)


def test_six_result_summary_requires_both_views_and_all_three_seeds() -> None:
    results = [
        {
            "status": "completed_ablation_fit_and_development_validation",
            "ablation_view": view,
            "selected_architecture_id": "nsf-t06-w128",
            "seed": seed,
            "identity": {
                "model_configuration_hash": (
                    "a" * 64 if view == "gw_only" else "b" * 64
                ),
                "training_code_commit": "1" * 40,
                "training_environment_sha256": "2" * 64,
                "train_manifest_sha256": "3" * 64,
                "validation_manifest_sha256": "4" * 64,
                "final_evaluation_commitment_sha256": "5" * 64,
                "membership_sha256": "6" * 64,
                "input_standardizer_sha256": "7" * 64,
                "target_standardizer_sha256": "8" * 64,
                "training_rung_count": 65536,
                "seed": seed,
            },
            "development": {
                "mean_nlp_nat_per_target_dimension": 0.2 + 0.01 * seed
            },
            "optimizer_or_budget_changed": False,
            "architecture_or_size_selection_authorized": False,
            "calibration_or_sbc_accessed": False,
            "final_evaluation_accessed": False,
        }
        for view in ABLATION_VIEWS
        for seed in (0, 1, 2)
    ]
    summary = summarize_ablation_results(results)
    assert summary["status"] == "completed_six_preregistered_ablation_fits"
    assert summary["fit_count"] == 6
    assert summary["selected_architecture_id"] == "nsf-t06-w128"
    assert summary["best_seed_selected"] is False
    assert summary["final_evaluation_accessed"] is False
    means = [
        row["mean_validation_nlp_nat_per_target_dimension"]
        for row in summary["views"]
    ]
    assert means == pytest.approx([0.21, 0.21])
    with pytest.raises(TrainingGateError, match="six results"):
        summarize_ablation_results(results[:-1])


def test_implementation_authorization_cannot_execute_or_index_scientific_data(
    tmp_path: Path,
) -> None:
    authorization = ROOT / (
        "configs/execution/phase7_ablation_training_stack_authorization.yaml"
    )
    with pytest.raises(TrainingGateError, match="authorization is absent"):
        validate_ablation_training_execution_gate(
            ROOT,
            authorization_path=authorization,
            stage_a_publication_root=tmp_path / "stage-a",
            stage_b_publication_root=tmp_path / "stage-b",
            combined_publication_root=tmp_path / "combined",
            correction_publication_root=tmp_path / "correction",
            terminal_size_decision_path=tmp_path / "terminal.json",
            selected_architecture_decision_path=tmp_path / "selection.json",
            primary_rung_preparation_path=tmp_path / "preparation.json",
        )
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    "label",
    [
        "lock_train_131k_saturated",
        "lock_train_131k_resource_capped_data_limited",
    ],
)
def test_ablation_gate_accepts_both_exact_terminal_131k_labels(
    tmp_path: Path, label: str
) -> None:
    decision = {
        "status": "terminal_learning_curve_decision_complete",
        "comparison": "corrected_train_65k_to_train_131k_terminal",
        "decision": label,
        "selected_training_count": 131072,
        "architecture_selection_review_allowed": True,
        "extension_above_131072_authorized": False,
        "all_three_probe_seeds_retained": True,
        "best_seed_selected": False,
        "calibration_accessed": False,
        "final_evaluation_accessed": False,
    }
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(decision) + "\n")
    authorization = {
        "locked_training_rung": 131072,
        "terminal_size_decision_path": str(path),
        "terminal_size_decision_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert _validate_terminal_size_lock(authorization, path)["decision"] == label


def test_runner_defaults_to_a_nonexecuting_plan() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/phase7/run_ablation_fit.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "implementation_ready_ablation_fits_blocked"
    assert result["maximum_future_fits"] == 6
    assert result["optimizer_started"] is False
    assert result["final_evaluation_accessed"] is False


def test_current_authorization_does_not_open_any_scientific_boundary() -> None:
    authorization = load_yaml(
        ROOT / "configs/execution/phase7_ablation_training_stack_authorization.yaml"
    )
    assert authorization["stop_after_implementation"] is True
    assert authorization["implementation_contract"]["maximum_future_fits"] == 6
    assert authorization["authorization"]["scientific_data_access_authorized"] is False
    assert authorization["authorization"]["ablation_fit_execution_authorized"] is False
    assert authorization["authorization"]["final_evaluation_unsealing_authorized"] is False
    assert authorization["authorization"]["gwosc_gwtc_access_authorized"] is False
