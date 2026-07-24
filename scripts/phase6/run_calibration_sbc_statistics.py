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


def _scalar_string(value: np.ndarray, *, name: str) -> str:
    if value.shape != ():
        raise ValueError(f"{name} score metadata is not scalar")
    result = str(value.item())
    if not result:
        raise ValueError(f"{name} score metadata is empty")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--calibration-scores", type=Path)
    parser.add_argument("--sbc-ranks-and-scores", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int)
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
        arguments.seed,
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
        and flags.get("checkpoint_access_authorized") is False
        and flags.get("final_evaluation_authorized") is False
        and flags.get("model_retraining_or_tuning_authorized") is False
        and flags.get("gwosc_gwtc_access_authorized") is False
    ):
        raise PermissionError("calibration/SBC execution flags are inconsistent")
    if arguments.seed not in (0, 1, 2):
        raise ValueError("calibration/SBC statistics require model seed 0, 1 or 2")
    seed_key = str(arguments.seed)
    selected = authorization.get("selected_architecture", {})
    if (
        authorization.get("authorized_model_seeds") != [0, 1, 2]
        or int(selected.get("locked_training_rung", -1)) != 131072
        or not str(selected.get("architecture_id", ""))
    ):
        raise PermissionError("calibration/SBC statistics seed or architecture lock changed")
    artifacts_by_seed = authorization.get("score_artifacts_by_seed", {})
    artifacts = (
        artifacts_by_seed.get(seed_key, {})
        if isinstance(artifacts_by_seed, dict)
        else {}
    )
    calibration_artifact = artifacts.get("calibration_fit", {})
    sbc_artifact = artifacts.get("sbc_diagnostic", {})
    if (
        arguments.calibration_scores.resolve()
        != Path(str(calibration_artifact.get("path", ""))).resolve()
        or _sha256(arguments.calibration_scores)
        != calibration_artifact.get("sha256")
        or arguments.sbc_ranks_and_scores.resolve()
        != Path(str(sbc_artifact.get("path", ""))).resolve()
        or _sha256(arguments.sbc_ranks_and_scores)
        != sbc_artifact.get("sha256")
    ):
        raise ValueError("calibration/SBC score artifact identity mismatch")
    roots = authorization.get("statistics_output_roots", {})
    expected_output = Path(str(roots.get(seed_key, ""))).resolve()
    if arguments.output_root.resolve() != expected_output or arguments.output_root.exists():
        raise ValueError("calibration/SBC output identity is unauthorized or already exists")
    calibration = _load_npz(arguments.calibration_scores)
    sbc = _load_npz(arguments.sbc_ranks_and_scores)
    calibration_ids = tuple(
        str(value) for value in calibration.get("physical_system_ids", ())
    )
    sbc_ids = tuple(str(value) for value in sbc.get("physical_system_ids", ()))
    if (
        calibration.get("marginal_scores", np.empty((0, 0))).shape != (4096, 2)
        or calibration.get("joint_scores", np.empty(0)).shape != (4096,)
        or calibration.get("em_cells", np.empty(0)).shape != (4096,)
        or len(calibration_ids) != 4096
        or len(set(calibration_ids)) != 4096
        or sbc.get("marginal_scores", np.empty((0, 0))).shape != (1024, 2)
        or sbc.get("joint_scores", np.empty(0)).shape != (1024,)
        or sbc.get("em_cells", np.empty(0)).shape != (1024,)
        or len(sbc_ids) != 1024
        or len(set(sbc_ids)) != 1024
        or bool(set(calibration_ids) & set(sbc_ids))
    ):
        raise ValueError("calibration/SBC score artifacts violate count or disjointness")
    score_identity = {}
    for name in (
        "model_seed",
        "architecture_id",
        "checkpoint_sha256",
        "publication_manifest_sha256",
        "inference_commit",
    ):
        calibration_value = _scalar_string(calibration[name], name=name)
        if calibration_value != _scalar_string(sbc[name], name=name):
            raise ValueError(f"calibration/SBC score artifacts mix {name}")
        score_identity[name] = calibration_value
    if score_identity != {
        str(key): str(value)
        for key, value in authorization.get("score_identities_by_seed", {})
        .get(seed_key, {})
        .items()
    }:
        raise ValueError("calibration/SBC score identity differs from authorization")
    if (
        _scalar_string(calibration["split"], name="split") != "calibration_fit"
        or _scalar_string(sbc["split"], name="split") != "sbc_diagnostic"
    ):
        raise ValueError("calibration/SBC score split identities are invalid")
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
    calibration_map_sha256 = _sha256(
        arguments.output_root / "calibration_region_maps.json"
    )
    sbc_summary_sha256 = _sha256(arguments.output_root / "sbc_rank_summary.json")
    independent_coverage_sha256 = _sha256(
        arguments.output_root / "independent_calibrated_coverage.json"
    )
    result = {
        "status": "completed_calibration_fit_and_independent_sbc",
        "calibration_score_sha256": calibration_artifact["sha256"],
        "sbc_score_sha256": sbc_artifact["sha256"],
        "score_identity": score_identity,
        "calibration_map_fitted_from_calibration_fit_only": True,
        "sbc_used_to_fit_calibration_map": False,
        "calibration_map_sha256": calibration_map_sha256,
        "sbc_summary_sha256": sbc_summary_sha256,
        "independent_coverage_sha256": independent_coverage_sha256,
        "model_retrained_or_tuned": False,
        "final_evaluation_accessed": False,
    }
    _atomic_json(arguments.output_root / "run_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
