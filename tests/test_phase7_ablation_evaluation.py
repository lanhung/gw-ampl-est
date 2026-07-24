from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gwlens_mm.schema import SplitName
from gwlens_mm.training.ablation_evaluation import (
    ABLATION_EVALUATION_CONFIG_HASH,
    AblatedCalibrationDataset,
    AblatedIIDDataset,
    AblationEvaluationGateError,
    aggregate_ablation_iid_comparisons,
    dry_run_ablation_evaluation_plan,
    fit_ablation_calibration_map,
    load_ablation_evaluation_contract,
    paired_ablation_iid_comparison,
    summarize_ablation_iid_scores,
    validate_matching_ablation_map,
)
from gwlens_mm.training.ablations import apply_ablation_view
from gwlens_mm.training.data import CalibrationSBCCase
from gwlens_mm.training.features import InputStandardizer, PreparedExample
from gwlens_mm.training.final_inference import FinalEvaluationCase

ROOT = Path(__file__).resolve().parents[1]
TEST_COUNT = 64
CELLS = tuple(f"cell-{index}" for index in range(8))


def _example(identifier: str = "system-0") -> PreparedExample:
    return PreparedExample(
        gw_strain=np.arange(48, dtype=np.float32).reshape(2, 3, 8),
        detector_mask=np.ones((2, 3), dtype=np.float32),
        astrometry_items=np.arange(45, dtype=np.float32).reshape(5, 9),
        astrometry_mask=np.ones(5, dtype=np.float32),
        scalar_features=np.arange(22, dtype=np.float32),
        scalar_mask=np.ones(22, dtype=np.float32),
        modality_mask=np.ones(8, dtype=np.float32),
        lens_family_condition=np.asarray((1.0, 0.0), dtype=np.float32),
        target=np.asarray((2.0, 3.0), dtype=np.float32),
        physical_system_id=identifier,
        lens_family="sie_external_shear",
        em_cell_signature="all",
        em_cell=CELLS[0],
    )


def _standardizer() -> InputStandardizer:
    return InputStandardizer(
        scalar_mean=(0.0,) * 22,
        scalar_standard_deviation=(1.0,) * 22,
        astrometry_mean=(0.0,) * 5,
        astrometry_standard_deviation=(1.0,) * 5,
    )


class _CalibrationDataset:
    expected_split = SplitName.CALIBRATION_FIT

    def __len__(self) -> int:
        return 1

    def calibration_sbc_case(self, index: int) -> CalibrationSBCCase:
        assert index == 0
        return CalibrationSBCCase(_example(), CELLS[0])


class _StandardizedFinalDataset:
    def __init__(self, split: SplitName = SplitName.IID_TEST) -> None:
        self.dataset = SimpleNamespace(
            publication=SimpleNamespace(
                specification=SimpleNamespace(split=split)
            )
        )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> FinalEvaluationCase:
        assert index == 0
        return FinalEvaluationCase(_example(), SplitName.IID_TEST.value, "iid")


def _score_payload(offset: float = 0.0) -> dict[str, np.ndarray]:
    identifiers = np.asarray(
        [f"system-{index:03d}" for index in range(TEST_COUNT)], dtype=np.str_
    )
    families = np.asarray(
        [
            "sie_external_shear" if index % 2 == 0 else "epl_external_shear"
            for index in range(TEST_COUNT)
        ],
        dtype=np.str_,
    )
    cells = np.asarray(
        [CELLS[index % len(CELLS)] for index in range(TEST_COUNT)], dtype=np.str_
    )
    truth = np.column_stack(
        (
            np.linspace(0.0, 1.0, TEST_COUNT),
            np.linspace(1.0, 2.0, TEST_COUNT),
        )
    )
    score: dict[str, np.ndarray] = {
        "physical_system_ids": identifiers,
        "lens_families": families,
        "em_cells": cells,
        "splits": np.asarray([SplitName.IID_TEST.value] * TEST_COUNT, dtype=np.str_),
        "diagnostic_context_ids": np.asarray(["iid"] * TEST_COUNT, dtype=np.str_),
        "truth": truth,
        "truth_log_density": -np.ones(TEST_COUNT),
        "nlp_nat_per_target_dimension": (
            np.linspace(0.1, 0.2, TEST_COUNT) + offset
        ),
        "crps": np.full((TEST_COUNT, 2), 0.2 + offset),
    }
    for level in (50, 80, 90, 95):
        marginal = np.ones((TEST_COUNT, 2), dtype=bool)
        joint = np.ones(TEST_COUNT, dtype=bool)
        score[f"marginal_covered_{level}"] = marginal
        score[f"joint_covered_{level}"] = joint
        score[f"marginal_interval_width_{level}"] = np.full(
            (TEST_COUNT, 2), 1.0 + offset
        )
    return score


def test_rc8_contract_and_implementation_gate_keep_science_closed() -> None:
    contract = load_ablation_evaluation_contract(ROOT)
    assert contract["config"]["preregistration_version"] == "1.1.0-rc.8"
    assert (
        contract["authorization"]["frozen_addendum"]["canonical_hash"]
        == ABLATION_EVALUATION_CONFIG_HASH
    )
    assert all(value is False for value in contract["config"]["execution"].values())
    plan = dry_run_ablation_evaluation_plan(ROOT)
    assert plan["calibration_map_count"] == 6
    assert plan["iid_score_artifact_count"] == 6
    assert plan["paired_comparison_count"] == 6
    assert plan["scientific_execution_authorized"] is False
    assert plan["ablation_sbc_authorized"] is False


def test_calibration_and_iid_adapters_apply_view_after_standardization() -> None:
    calibration = AblatedCalibrationDataset(
        _CalibrationDataset(), _standardizer(), "em_only"  # type: ignore[arg-type]
    )[0]
    assert np.all(calibration.example.gw_strain == 0.0)
    assert np.all(calibration.example.detector_mask == 0.0)
    assert calibration.example.physical_system_id == "system-0"
    assert calibration.em_cell == CELLS[0]

    iid = AblatedIIDDataset(
        _StandardizedFinalDataset(), "gw_only"  # type: ignore[arg-type]
    )[0]
    expected_iid = apply_ablation_view(_example(), "gw_only")
    assert np.array_equal(
        iid.example.scalar_features, expected_iid.scalar_features
    )
    assert np.array_equal(iid.example.scalar_mask, expected_iid.scalar_mask)
    assert iid.example.physical_system_id == "system-0"
    with pytest.raises(AblationEvaluationGateError, match="restricted to IID"):
        AblatedIIDDataset(
            _StandardizedFinalDataset(SplitName.PARAMETER_REGION_OOD),  # type: ignore[arg-type]
            "gw_only",
        )


def test_each_view_seed_fits_its_own_map_on_primary_calibration_cases() -> None:
    identifiers = np.asarray(
        [f"calibration-{index:03d}" for index in range(TEST_COUNT)], dtype=np.str_
    )
    cells = np.asarray(
        [CELLS[index % len(CELLS)] for index in range(TEST_COUNT)], dtype=np.str_
    )
    score = {
        "marginal_scores": np.column_stack(
            (
                np.linspace(0.01, 0.99, TEST_COUNT),
                np.linspace(0.99, 0.01, TEST_COUNT),
            )
        ),
        "joint_scores": np.linspace(0.01, 0.99, TEST_COUNT),
        "physical_system_ids": identifiers,
        "em_cells": cells,
    }
    fitted = fit_ablation_calibration_map(
        score,
        view="gw_only",
        model_seed=1,
        checkpoint_sha256="a" * 64,
        primary_calibration_case_ids=tuple(identifiers),
        expected_count=TEST_COUNT,
        expected_per_cell=TEST_COUNT // 8,
    )
    validate_matching_ablation_map(
        fitted,
        view="gw_only",
        model_seed=1,
        checkpoint_sha256="a" * 64,
    )
    with pytest.raises(AblationEvaluationGateError, match="does not match"):
        validate_matching_ablation_map(
            fitted,
            view="em_only",
            model_seed=1,
            checkpoint_sha256="a" * 64,
        )
    with pytest.raises(AblationEvaluationGateError, match="does not match"):
        validate_matching_ablation_map(
            {"status": "fitted_split_conformal_region_level_maps"},
            view="gw_only",
            model_seed=1,
            checkpoint_sha256="a" * 64,
        )
    with pytest.raises(AblationEvaluationGateError, match="not byte-order identical"):
        fit_ablation_calibration_map(
            score,
            view="gw_only",
            model_seed=1,
            checkpoint_sha256="a" * 64,
            primary_calibration_case_ids=tuple(reversed(identifiers)),
            expected_count=TEST_COUNT,
            expected_per_cell=TEST_COUNT // 8,
        )


def test_iid_summary_reports_all_families_cells_counts_and_no_selection() -> None:
    summary = summarize_ablation_iid_scores(
        _score_payload(), view="em_only", model_seed=2, expected_count=TEST_COUNT
    )
    assert summary["overall"]["case_count"] == TEST_COUNT
    assert set(summary["lens_families"]) == {
        "sie_external_shear",
        "epl_external_shear",
    }
    assert set(summary["em_cells"]) == set(CELLS)
    assert summary["overall"]["coverage"]["0.90"]["joint"]["covered_count"] == [
        TEST_COUNT
    ]
    assert summary["best_seed_selected"] is False
    assert summary["result_can_trigger_retraining_or_tuning"] is False
    assert summary["sbc_executed"] is False
    assert summary["ood_or_mismatch_executed"] is False


def test_paired_iid_bootstrap_is_deterministic_and_rejects_case_drift() -> None:
    primary = _score_payload()
    ablation = _score_payload(offset=0.05)
    first = paired_ablation_iid_comparison(
        primary,
        ablation,
        view="gw_only",
        model_seed=0,
        expected_count=TEST_COUNT,
    )
    second = paired_ablation_iid_comparison(
        primary,
        ablation,
        view="gw_only",
        model_seed=0,
        expected_count=TEST_COUNT,
    )
    assert first == second
    assert first["paired_bootstrap"]["nlp_nat_per_target_dimension"][
        "point_estimate"
    ] == pytest.approx([0.05])
    assert first["paired_bootstrap"]["crps"]["point_estimate"] == pytest.approx(
        [0.05, 0.05]
    )
    assert first["superiority_gate"] is None
    assert first["result_can_trigger_retraining_or_tuning"] is False

    changed = dict(ablation)
    identifiers = np.asarray(ablation["physical_system_ids"]).copy()
    identifiers[[0, 1]] = identifiers[[1, 0]]
    changed["physical_system_ids"] = identifiers
    with pytest.raises(AblationEvaluationGateError, match="ordering differs"):
        paired_ablation_iid_comparison(
            primary,
            changed,
            view="gw_only",
            model_seed=0,
            expected_count=TEST_COUNT,
        )


def test_all_seed_aggregate_requires_exact_grid_and_never_selects_seed() -> None:
    primary = _score_payload()
    comparisons = [
        paired_ablation_iid_comparison(
            primary,
            _score_payload(offset=0.01 * (seed + 1)),
            view=view,
            model_seed=seed,
            expected_count=TEST_COUNT,
        )
        for view in ("gw_only", "em_only")
        for seed in (0, 1, 2)
    ]
    aggregate = aggregate_ablation_iid_comparisons(comparisons)
    assert aggregate["comparison_count"] == 6
    assert set(aggregate["views"]) == {"gw_only", "em_only"}
    assert aggregate["best_seed_selected"] is False
    assert aggregate["seed_pooling_used_for_model_selection"] is False
    for view in ("gw_only", "em_only"):
        nlp = aggregate["views"][view]["aggregate_point_estimates"][
            "nlp_nat_per_target_dimension"
        ]
        assert nlp["mean_across_seeds"] == pytest.approx([0.02])
        assert nlp["sample_standard_deviation_across_seeds"] == pytest.approx(
            [0.01]
        )
    with pytest.raises(AblationEvaluationGateError, match="both views"):
        aggregate_ablation_iid_comparisons(comparisons[:-1])
