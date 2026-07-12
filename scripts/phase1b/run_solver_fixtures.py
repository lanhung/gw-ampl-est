#!/usr/bin/env python3
"""Run deterministic SIS/SIE/EPL numerical contracts before smoke generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gwlens_mm.physics.lenstronomy_adapter import LenstronomyAdapter
from gwlens_mm.physics.quantities import LensFamily
from gwlens_mm.physics.solver import SISSolver, validate_solver_contract


def lens_parameters(family: LensFamily) -> dict[str, float]:
    parameters = {
        "einstein_radius_arcsec": 1.1,
        "axis_ratio": 0.72,
        "position_angle_rad": 0.2,
        "shear_gamma1": 0.04,
        "shear_gamma2": -0.02,
    }
    if family is LensFamily.EPL_EXTERNAL_SHEAR:
        parameters["axis_ratio"] = 0.75
        parameters["density_slope"] = 2.1
    return parameters


def summarize(name: str, solution: Any, selected_ids: tuple[str, str]) -> dict[str, Any]:
    if not solution.valid:
        raise RuntimeError(f"fixture {name} is invalid: {solution.validity_reason}")
    image_ids = {image.image_id for image in solution.physical_images}
    if not set(selected_ids) <= image_ids:
        raise RuntimeError(f"fixture {name} selected pair is not physical")
    return {
        "fixture": name,
        "lens_family": solution.lens_family.value,
        "solver": solution.solver_name,
        "solver_version": solution.solver_version,
        "image_count": len(solution.physical_images),
        "selected_image_ids": list(selected_ids),
        "selected_pair_is_first_two": list(selected_ids) == ["image_0", "image_1"],
        "images": [
            {
                "image_id": image.image_id,
                "position_arcsec": list(image.position_arcsec),
                "mu_signed": image.mu_signed,
                "arrival_time_seconds": image.arrival_time_dimensionless,
                "parity": image.parity.value,
                "morse_class": image.morse_class.value,
            }
            for image in solution.physical_images
        ],
    }


def run() -> dict[str, Any]:
    sis = SISSolver()
    sis_solution = sis.solve((0.5, 0.0), {"einstein_radius_arcsec": 1.0})
    validate_solver_contract(sis, [((0.5, 0.0), {"einstein_radius_arcsec": 1.0})])

    cases = [summarize("sis_double", sis_solution, ("sis_plus", "sis_minus"))]
    for name, family, source, expected, selected in (
        ("sie_double", LensFamily.SIE_EXTERNAL_SHEAR, (0.5, 0.1), 2, ("image_0", "image_1")),
        (
            "sie_quad_nonconsecutive",
            LensFamily.SIE_EXTERNAL_SHEAR,
            (0.03, 0.02),
            4,
            ("image_0", "image_2"),
        ),
        ("epl_quad", LensFamily.EPL_EXTERNAL_SHEAR, (0.03, 0.02), 4, ("image_0", "image_1")),
    ):
        adapter = LenstronomyAdapter(family, 0.5, 1.5)
        parameters = lens_parameters(family)
        solution = adapter.solve(source, parameters)
        validate_solver_contract(adapter, [(source, parameters)])
        if len(solution.physical_images) != expected:
            actual = len(solution.physical_images)
            raise RuntimeError(f"{name} expected {expected} images, got {actual}")
        cases.append(summarize(name, solution, selected))
    return {"status": "passed", "fixtures": cases}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(text, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text, encoding="utf-8")
        print(arguments.output)


if __name__ == "__main__":
    main()
