#!/usr/bin/env python3
"""Fit region maps and evaluate independent SBC after a future exact gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from gwlens_mm.config import load_yaml
from gwlens_mm.training.calibration import (
    SBC_STATISTICS,
    calibrated_region_coverage,
    evaluate_sbc_histograms,
    fit_region_calibration,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def _load_npz(path: Path) -> Mapping[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--calibration-scores", type=Path)
    parser.add_argument("--sbc-ranks-and-scores", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.execute:
        print(
            json.dumps(
                {
                    "status": "calibration_sbc_statistics_implementation_ready",
                    "calibration_fitted": False,
                    "sbc_executed": False,
                    "final_evaluation_accessed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    required = (
        arguments.authorization,
        arguments.calibration_scores,
        arguments.sbc_ranks_and_scores,
        arguments.output_root,
    )
    if any(value is None for value in required):
        raise ValueError("calibration/SBC execution requires every reviewed artifact")
    authorization = load_yaml(arguments.authorization)
    if authorization.get("authorization_status") != "authorized_calibration_sbc_statistics_only":
        raise PermissionError("calibration/SBC statistics are not authorized")
    flags = authorization.get("authorization", {})
    if not (
        flags.get("calibration_fit_authorized") is True
        and flags.get("sbc_execution_authorized") is True
        and flags.get("final_evaluation_authorized") is False
        and flags.get("model_retraining_or_tuning_authorized") is False
        and flags.get("gwosc_gwtc_access_authorized") is False
    ):
        raise PermissionError("calibration/SBC execution flags are inconsistent")
    artifacts = authorization.get("score_artifacts", {})
    if (
        arguments.calibration_scores.resolve()
        != Path(str(artifacts.get("calibration_scores_path", ""))).resolve()
        or _sha256(arguments.calibration_scores)
        != artifacts.get("calibration_scores_sha256")
        or arguments.sbc_ranks_and_scores.resolve()
        != Path(str(artifacts.get("sbc_ranks_and_scores_path", ""))).resolve()
        or _sha256(arguments.sbc_ranks_and_scores)
        != artifacts.get("sbc_ranks_and_scores_sha256")
    ):
        raise ValueError("calibration/SBC score artifact identity mismatch")
    expected_output = Path(str(authorization.get("statistics_output_root", ""))).resolve()
    if arguments.output_root.resolve() != expected_output or arguments.output_root.exists():
        raise ValueError("calibration/SBC output identity is unauthorized or already exists")
    calibration = _load_npz(arguments.calibration_scores)
    sbc = _load_npz(arguments.sbc_ranks_and_scores)
    if (
        sbc.get("marginal_scores", np.empty((0, 0))).shape != (1024, 2)
        or sbc.get("joint_scores", np.empty(0)).shape != (1024,)
        or sbc.get("em_cells", np.empty(0)).shape != (1024,)
    ):
        raise ValueError("SBC score artifact must contain exactly 1,024 replicates")
    calibration_map = fit_region_calibration(
        calibration["marginal_scores"],
        calibration["joint_scores"],
        tuple(str(value) for value in calibration["em_cells"]),
    )
    ranks = {statistic: sbc[f"rank_{statistic}"] for statistic in SBC_STATISTICS}
    sbc_summary = evaluate_sbc_histograms(ranks, expected_replicate_count=1024)
    independent_coverage = calibrated_region_coverage(
        calibration_map,
        sbc["marginal_scores"],
        sbc["joint_scores"],
        tuple(str(value) for value in sbc["em_cells"]),
    )
    arguments.output_root.mkdir(parents=True, exist_ok=False)
    _atomic_json(arguments.output_root / "calibration_region_maps.json", calibration_map)
    _atomic_json(arguments.output_root / "sbc_rank_summary.json", sbc_summary)
    _atomic_json(
        arguments.output_root / "independent_calibrated_coverage.json",
        independent_coverage,
    )
    result = {
        "status": "completed_calibration_fit_and_independent_sbc",
        "calibration_score_sha256": artifacts["calibration_scores_sha256"],
        "sbc_score_sha256": artifacts["sbc_ranks_and_scores_sha256"],
        "calibration_map_fitted_from_calibration_fit_only": True,
        "sbc_used_to_fit_calibration_map": False,
        "model_retrained_or_tuned": False,
        "final_evaluation_accessed": False,
    }
    _atomic_json(arguments.output_root / "run_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
