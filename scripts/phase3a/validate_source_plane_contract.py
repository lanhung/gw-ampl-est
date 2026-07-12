#!/usr/bin/env python3
"""Run the frozen RC.4 source-boundary solver-agreement gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from gwlens_mm.config import load_yaml
from gwlens_mm.physics.quantities import LensFamily
from gwlens_mm.production.source_plane import (
    boundary_points,
    compare_solver_unions,
    scaled_boundary_points,
)

ROOT = Path(__file__).resolve().parents[2]


def _cases() -> Tuple[Dict[str, Any], ...]:
    specifications = (
        ("sie-small-flat-shear", LensFamily.SIE_EXTERNAL_SHEAR, 0.3, 0.4, 0.0, 0.15, 0.0, 2.0),
        ("sie-large-round-shear", LensFamily.SIE_EXTERNAL_SHEAR, 3.0, 1.0, 0.7, 0.15, 1.1, 2.0),
        ("sie-small-round-zero", LensFamily.SIE_EXTERNAL_SHEAR, 0.3, 1.0, 1.2, 0.0, 0.0, 2.0),
        ("sie-large-flat-zero", LensFamily.SIE_EXTERNAL_SHEAR, 3.0, 0.4, 2.2, 0.0, 0.0, 2.0),
        ("epl-small-flat-shallow", LensFamily.EPL_EXTERNAL_SHEAR, 0.3, 0.4, 0.0, 0.15, 0.0, 1.6),
        ("epl-large-round-steep", LensFamily.EPL_EXTERNAL_SHEAR, 3.0, 1.0, 0.7, 0.15, 1.1, 2.5),
        ("epl-small-round-steep", LensFamily.EPL_EXTERNAL_SHEAR, 0.3, 1.0, 1.2, 0.0, 0.0, 2.5),
        ("epl-large-flat-shallow", LensFamily.EPL_EXTERNAL_SHEAR, 3.0, 0.4, 2.2, 0.0, 0.0, 1.6),
    )
    cases = []
    for identifier, family, theta_e, axis_ratio, angle, shear, shear_angle, slope in specifications:
        cases.append(
            {
                "case_id": identifier,
                "family": family.value,
                "z_lens": 0.5,
                "z_source": 1.5,
                "parameters": {
                    "einstein_radius_arcsec": theta_e,
                    "axis_ratio": axis_ratio,
                    "position_angle_rad": angle,
                    "shear_gamma1": shear * math.cos(2.0 * shear_angle),
                    "shear_gamma2": shear * math.sin(2.0 * shear_angle),
                    "density_slope": slope,
                },
            }
        )
    return tuple(cases)


def _evaluate(
    task: Tuple[Mapping[str, Any], Tuple[float, float], Mapping[str, Any]],
) -> Dict[str, Any]:
    case, source_position, contract = task
    started = time.perf_counter()
    agreement = compare_solver_unions(
        LensFamily(str(case["family"])),
        float(case["z_lens"]),
        float(case["z_source"]),
        source_position,
        case["parameters"],
        contract,
    )
    return {
        "case_id": case["case_id"],
        "source_position_arcsec": source_position,
        "runtime_seconds": time.perf_counter() - started,
        **asdict(agreement),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()
    config = load_yaml(ROOT / "configs/statistics/phase2_preregistration.yaml")
    contract = config["lens_solver"]["numerical_contract"]
    count = int(contract["support_audit"]["boundary_source_points_per_edge"])
    normalized_boundary = boundary_points(count)
    tasks = []
    for case in _cases():
        theta_e = float(case["parameters"]["einstein_radius_arcsec"])
        for source_position in scaled_boundary_points(normalized_boundary, theta_e):
            tasks.append((case, source_position, contract))
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(_evaluate, tasks, chunksize=1))
    failures = [row for row in rows if not row["passed"]]
    output = {
        "status": "passed" if not failures else "failed",
        "preregistration_version": config["preregistration_version"],
        "case_count": len(_cases()),
        "boundary_points_per_case": len(normalized_boundary),
        "comparison_count": len(rows),
        "failure_count": len(failures),
        "maximum_position_difference_arcsec": max(
            row["maximum_position_difference_arcsec"] for row in rows
        ),
        "maximum_relative_magnification_difference": max(
            row["maximum_relative_magnification_difference"] for row in rows
        ),
        "walltime_seconds": time.perf_counter() - started,
        "workers": args.workers,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
