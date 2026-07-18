from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from gwlens_mm.config import load_yaml
from gwlens_mm.provenance import configuration_hash
from gwlens_mm.training.calibration import (
    SBC_STATISTICS,
    _chi_square_survival,
    calibrated_region_coverage,
    calibration_score_artifact,
    conformal_order_statistic,
    deterministic_sbc_subset,
    empirical_pit_scores,
    evaluate_sbc_histograms,
    fit_region_calibration,
    holm_step_down,
    joint_hpd_scores,
    randomized_rank_from_counts,
    sbc_ranks,
    sbc_score_artifact,
    wilson_interval,
)

ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION_HASH = "033b996930c93e7e4a9881fc3de49bb85cf4be96fcbd890bf2543b46368c9d8e"
EM_CELLS = (
    "full_precise_spectroscopic",
    "full_photometric_redshifts",
    "no_velocity_dispersion",
    "no_source_redshift",
    "no_lens_redshift",
    "astrometry_redshifts_only",
    "astrometry_kinematics_no_einstein_scale",
    "sparse_astrometry_timing_only",
)


def test_phase6_design_is_scientifically_frozen_but_execution_closed() -> None:
    config = load_yaml(ROOT / "configs/statistics/calibration_sbc_preregistration.yaml")
    assert configuration_hash(config) == PREREGISTRATION_HASH
    assert config["preregistration_version"] == "1.1.0-rc.5"
    assert config["parent_direct_target_contract"]["estimand_changed"] is False
    assert config["calibration_fit_split"]["accepted_physical_system_count"] == 4096
    assert config["independent_sbc"]["deterministic_replicate_count"] == 1024
    assert config["credible_region_calibration"]["changes_posterior_samples_or_density"] is False
    assert config["credible_region_calibration"]["maps"]["tail_specific_map_fitting"] is False
    assert config["execution"] == {
        "calibration_sbc_materialization_enabled": False,
        "calibration_fit_enabled": False,
        "sbc_enabled": False,
        "selected_model_access_enabled": False,
        "final_evaluation_access_enabled": False,
        "gwosc_gwtc_access_enabled": False,
    }
    gate = load_yaml(ROOT / "configs/execution/phase6_calibration_sbc_stack_authorization.yaml")
    assert gate["authorization_status"].endswith("implementation_only")
    assert gate["authorization"]["scientific_data_access_authorized"] is False
    assert gate["authorization"]["calibration_fit_authorized"] is False
    assert gate["authorization"]["sbc_execution_authorized"] is False


def test_empirical_marginal_and_joint_scores_have_frozen_semantics() -> None:
    draws = np.asarray(
        [
            [[-2.0, 0.0], [-1.0, 1.0], [0.0, 2.0], [1.0, 3.0]],
            [[0.0, 0.0], [0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
        ]
    )
    truth = np.asarray([[0.0, 1.5], [0.0, 4.0]])
    scores = empirical_pit_scores(draws, truth)
    assert scores[0].tolist() == pytest.approx([0.25, 0.0])
    assert scores[1].tolist() == pytest.approx([0.5, 1.0])
    density = joint_hpd_scores(
        np.asarray([[3.0, 2.0, 1.0], [0.0, -1.0, -2.0]]),
        np.asarray([2.0, 1.0]),
    )
    assert density.tolist() == pytest.approx([2 / 3, 0.0])


def test_conformal_threshold_uses_ceil_n_plus_one_order_statistic() -> None:
    result = conformal_order_statistic(np.arange(10) / 10.0, 0.8)
    assert result["one_based_order_statistic"] == 9
    assert result["raw_region_mass_threshold"] == pytest.approx(0.8)
    capped = conformal_order_statistic(np.arange(10) / 10.0, 0.95)
    assert capped["one_based_order_statistic"] == 10
    lower, upper = wilson_interval(90, 100)
    assert lower < 0.9 < upper


def test_region_calibration_requires_exact_balanced_em_cells_and_applies_maps() -> None:
    cells = tuple(cell for cell in EM_CELLS for _ in range(512))
    base = np.linspace(0.0, 1.0, 512, endpoint=False)
    marginal = np.concatenate(
        [np.column_stack((base, base[::-1])) for _ in EM_CELLS], axis=0
    )
    joint = np.tile(base, len(EM_CELLS))
    fitted = fit_region_calibration(marginal, joint, cells)
    assert fitted["calibration_case_count"] == 4096
    assert set(fitted["em_cells"]) == set(EM_CELLS)
    assert fitted["posterior_samples_or_density_changed"] is False
    independent = calibrated_region_coverage(fitted, marginal, joint, cells)
    assert independent["case_count"] == 4096
    for level in ("0.50", "0.80", "0.90", "0.95"):
        assert independent["coverage"][level]["joint_coverage"] >= float(level)
        assert len(independent["coverage"][level]["joint_wilson_95"]) == 2
    assert set(independent["em_cell_coverage_using_cell_maps"]) == set(EM_CELLS)
    assert set(independent["coverage_using_global_map"]) == {
        "0.50",
        "0.80",
        "0.90",
        "0.95",
    }
    with pytest.raises(ValueError, match="not exactly balanced"):
        fit_region_calibration(
            marginal,
            joint,
            cells[:512] + (EM_CELLS[0],) + cells[513:],
        )


def test_sbc_subset_is_deterministic_and_order_independent() -> None:
    identifiers = tuple(f"system-{index:04d}" for index in range(2048))
    forward = deterministic_sbc_subset(identifiers, root_seed=2026071601)
    reverse = deterministic_sbc_subset(tuple(reversed(identifiers)), root_seed=2026071601)
    assert len(forward) == 1024
    assert forward == reverse


def test_sbc_ranks_cover_marginal_derived_and_joint_statistics() -> None:
    draws = np.asarray(
        [
            [[0.0, 2.0], [1.0, 1.0], [2.0, 0.0], [3.0, -1.0]],
            [[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
        ]
    )
    truth = np.asarray([[1.5, 0.5], [0.5, 1.5]])
    draw_density = np.asarray([[0.0, 1.0, 2.0, 3.0], [-3.0, -2.0, -1.0, 0.0]])
    truth_density = np.asarray([1.5, -1.5])
    ranks = sbc_ranks(
        draws,
        truth,
        ("system-a", "system-b"),
        posterior_draw_log_density=draw_density,
        truth_log_density=truth_density,
    )
    assert set(ranks) == set(SBC_STATISTICS)
    assert all(value.shape == (2,) for value in ranks.values())
    assert ranks["log_abs_mu_primary"].tolist() == [2, 2]
    assert ranks["joint_log_density_rank"].tolist() == [2, 2]
    assert randomized_rank_from_counts(
        2,
        0,
        physical_system_id="system-a",
        statistic="log_abs_mu_primary",
    ) == ranks["log_abs_mu_primary"][0]
    tied = randomized_rank_from_counts(
        1,
        3,
        physical_system_id="system-tie",
        statistic="log_abs_mu_primary",
    )
    assert 1 <= tied <= 4


def test_discrete_uniform_histogram_and_holm_are_machine_readable() -> None:
    assert _chi_square_survival(30.0, 19) == pytest.approx(
        0.05179845889302389, rel=1e-12
    )
    assert _chi_square_survival(10.0, 4) == pytest.approx(
        0.04042768199451279, rel=1e-12
    )
    possible_ranks = np.arange(1024, dtype=np.int64)
    ranks = {statistic: possible_ranks.copy() for statistic in SBC_STATISTICS}
    result = evaluate_sbc_histograms(ranks)
    assert result["replicate_count"] == 1024
    assert result["histogram_bins"] == 20
    assert result["any_holm_rejection"] is False
    assert all(
        sum(value["observed_bin_counts"]) == 1024
        for value in result["statistics"].values()
    )
    holm = holm_step_down(
        {"a": 0.001, "b": 0.02, "c": 0.50}, familywise_alpha=0.05
    )
    assert holm["a"]["rejected"] is True
    assert holm["b"]["rejected"] is True
    assert holm["c"]["rejected"] is False
    with pytest.raises(ValueError, match="exactly 1,024"):
        evaluate_sbc_histograms(
            {statistic: possible_ranks[:-1] for statistic in SBC_STATISTICS}
        )


def test_statistics_runner_keeps_calibration_fit_and_sbc_independent(
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "calibration.npz"
    sbc_path = tmp_path / "sbc.npz"
    cells = np.asarray(tuple(cell for cell in EM_CELLS for _ in range(512)))
    base = np.linspace(0.0, 1.0, 4096, endpoint=False)
    np.savez(
        calibration_path,
        marginal_scores=np.column_stack((base, base[::-1])),
        joint_scores=base,
        em_cells=cells,
        physical_system_ids=np.asarray(
            [f"calibration-{index:04d}" for index in range(4096)]
        ),
        split=np.asarray("calibration_fit"),
        model_seed=np.asarray(0, dtype=np.int64),
        architecture_id=np.asarray("nsf-t10-w256"),
        checkpoint_sha256=np.asarray("a" * 64),
        publication_manifest_sha256=np.asarray("b" * 64),
        inference_commit=np.asarray("c" * 40),
    )
    sbc_cells = np.asarray(tuple(EM_CELLS[index % 8] for index in range(1024)))
    rank_arrays = {
        f"rank_{statistic}": np.arange(1024, dtype=np.int64)
        for statistic in SBC_STATISTICS
    }
    np.savez(
        sbc_path,
        marginal_scores=np.column_stack((base[:1024], base[1024:2048])),
        joint_scores=base[:1024],
        em_cells=sbc_cells,
        physical_system_ids=np.asarray(
            [f"sbc-{index:04d}" for index in range(1024)]
        ),
        split=np.asarray("sbc_diagnostic"),
        model_seed=np.asarray(0, dtype=np.int64),
        architecture_id=np.asarray("nsf-t10-w256"),
        checkpoint_sha256=np.asarray("a" * 64),
        publication_manifest_sha256=np.asarray("b" * 64),
        inference_commit=np.asarray("c" * 40),
        **rank_arrays,
    )
    output = tmp_path / "output"
    authorization = {
        "authorization_status": "authorized_calibration_sbc_statistics_only",
        "authorization": {
            "calibration_fit_authorized": True,
            "sbc_execution_authorized": True,
            "final_evaluation_authorized": False,
            "model_retraining_or_tuning_authorized": False,
            "gwosc_gwtc_access_authorized": False,
        },
        "score_artifacts": {
            "calibration_scores_path": str(calibration_path),
            "calibration_scores_sha256": hashlib.sha256(
                calibration_path.read_bytes()
            ).hexdigest(),
            "sbc_ranks_and_scores_path": str(sbc_path),
            "sbc_ranks_and_scores_sha256": hashlib.sha256(
                sbc_path.read_bytes()
            ).hexdigest(),
        },
        "score_identity": {
            "model_seed": "0",
            "architecture_id": "nsf-t10-w256",
            "checkpoint_sha256": "a" * 64,
            "publication_manifest_sha256": "b" * 64,
            "inference_commit": "c" * 40,
        },
        "statistics_output_root": str(output),
    }
    authorization_path = tmp_path / "authorization.yaml"
    authorization_path.write_text(yaml.safe_dump(authorization, sort_keys=True))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/phase6/run_calibration_sbc_statistics.py"),
            "--authorization",
            str(authorization_path),
            "--calibration-scores",
            str(calibration_path),
            "--sbc-ranks-and-scores",
            str(sbc_path),
            "--output-root",
            str(output),
            "--execute",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    summary = json.loads((output / "run_summary.json").read_text())
    assert summary["calibration_map_fitted_from_calibration_fit_only"] is True
    assert summary["sbc_used_to_fit_calibration_map"] is False
    assert summary["final_evaluation_accessed"] is False
    assert (output / "calibration_region_maps.json").is_file()
    assert (output / "sbc_rank_summary.json").is_file()


def test_exact_score_artifacts_keep_ids_cells_and_splits_independent() -> None:
    calibration_draws = np.zeros((16, 32, 2), dtype=np.float64)
    calibration_truth = np.zeros((16, 2), dtype=np.float64)
    calibration_density = np.zeros((16, 32), dtype=np.float64)
    cells = tuple(cell for cell in EM_CELLS for _ in range(2))
    calibration = calibration_score_artifact(
        calibration_draws,
        calibration_truth,
        calibration_density,
        np.zeros(16),
        tuple(f"cal-{index}" for index in range(16)),
        cells,
        expected_count=16,
        expected_draw_count=32,
    )
    assert calibration["marginal_scores"].shape == (16, 2)
    assert calibration["physical_system_ids"].dtype.kind == "U"
    sbc = sbc_score_artifact(
        calibration_draws,
        calibration_truth,
        calibration_density,
        np.zeros(16),
        tuple(f"sbc-{index}" for index in range(16)),
        cells,
        expected_count=16,
        expected_draw_count=32,
    )
    assert set(f"rank_{name}" for name in SBC_STATISTICS) < set(sbc)
    assert not set(calibration["physical_system_ids"]) & set(
        sbc["physical_system_ids"]
    )
