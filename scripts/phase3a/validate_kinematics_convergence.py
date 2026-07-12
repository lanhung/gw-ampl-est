#!/usr/bin/env python3
"""Run the frozen balanced 4,000-versus-16,000 Galkin convergence gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Mapping

from gwlens_mm.config import load_yaml
from gwlens_mm.physics.quantities import LensFamily
from gwlens_mm.production.dynamics import galkin_velocity_dispersion

ROOT = Path(__file__).resolve().parents[2]


def _seed(identifier: str) -> int:
    return int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:4], "big")


def _evaluate(case: Mapping[str, Any]) -> Dict[str, Any]:
    family = LensFamily(str(case["lens_family"]))
    started = time.perf_counter()
    low = galkin_velocity_dispersion(
        lens_family=family,
        z_lens=float(case["z_lens"]),
        z_source=float(case["z_source"]),
        einstein_radius_arcsec=float(case["einstein_radius_arcsec"]),
        density_slope=float(case["density_slope"]),
        effective_radius_arcsec=float(case["effective_radius_arcsec"]),
        anisotropy_radius_over_effective_radius=float(case["anisotropy_ratio"]),
        monte_carlo_samples=4000,
        seed=int(case["seed"]),
    )
    low_seconds = time.perf_counter() - started
    started = time.perf_counter()
    reference = galkin_velocity_dispersion(
        lens_family=family,
        z_lens=float(case["z_lens"]),
        z_source=float(case["z_source"]),
        einstein_radius_arcsec=float(case["einstein_radius_arcsec"]),
        density_slope=float(case["density_slope"]),
        effective_radius_arcsec=float(case["effective_radius_arcsec"]),
        anisotropy_radius_over_effective_radius=float(case["anisotropy_ratio"]),
        monte_carlo_samples=16000,
        seed=int(case["seed"]),
    )
    reference_seconds = time.perf_counter() - started
    relative = abs(low.velocity_dispersion_km_s - reference.velocity_dispersion_km_s) / (
        reference.velocity_dispersion_km_s
    )
    return {
        **dict(case),
        "effective_density_slope": low.density_slope,
        "dispersion_4000_km_s": low.velocity_dispersion_km_s,
        "dispersion_16000_km_s": reference.velocity_dispersion_km_s,
        "relative_difference": relative,
        "runtime_4000_seconds": low_seconds,
        "runtime_16000_seconds": reference_seconds,
        "passed": relative <= 0.02,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    preregistration = load_yaml(ROOT / "configs/statistics/phase2_preregistration.yaml")
    cells = tuple(preregistration["em_observation_model"]["availability_cells"])
    cases = []
    for cell_index, cell in enumerate(cells):
        for family_index, family in enumerate(
            (LensFamily.SIE_EXTERNAL_SHEAR, LensFamily.EPL_EXTERNAL_SHEAR)
        ):
            index = 2 * cell_index + family_index
            identifier = f"kinematics-{cell}-{family.value}"
            cases.append(
                {
                    "case_id": identifier,
                    "em_cell": cell,
                    "lens_family": family.value,
                    "z_lens": 0.25 if index % 2 == 0 else 0.8,
                    "z_source": 1.0 if index % 2 == 0 else 2.5,
                    "einstein_radius_arcsec": 0.5 if index % 4 < 2 else 2.5,
                    "density_slope": 1.7 if index % 2 == 0 else 2.4,
                    "effective_radius_arcsec": 0.35 if index % 4 < 2 else 1.8,
                    "anisotropy_ratio": 0.6 if index % 8 < 4 else 4.0,
                    "seed": _seed(identifier),
                }
            )
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(_evaluate, cases))
    elapsed = time.perf_counter() - started
    rows.sort(key=lambda row: str(row["case_id"]))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    maximum = max(float(row["relative_difference"]) for row in rows)
    passed = all(bool(row["passed"]) for row in rows)
    summary = {
        "status": "passed" if passed else "failed",
        "case_count": len(rows),
        "workers": args.workers,
        "elapsed_seconds": elapsed,
        "maximum_relative_difference": maximum,
        "required_maximum_relative_difference": 0.02,
        "both_lens_families": sorted({str(row["lens_family"]) for row in rows}),
        "em_cells": sorted({str(row["em_cell"]) for row in rows}),
        "process_local_deterministic_seed": True,
    }
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
