#!/usr/bin/env python3
"""Save deterministic evidence for every frozen mass-sheet transformation rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gwlens_mm.physics.quantities import ImageParity, LensFamily, MorseClass
from gwlens_mm.physics.solver import (
    LensSystemSolution,
    PhysicalImage,
    apply_mass_sheet_transform,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = (0.3, -0.2)
    solution = LensSystemSolution(
        LensFamily.EPL_EXTERNAL_SHEAR,
        (
            PhysicalImage(
                "minimum",
                (1.2, 0.1),
                4.0,
                ImageParity.POSITIVE,
                MorseClass.MINIMUM,
                fermat_potential_dimensionless=-1.5,
                arrival_time_seconds=20.0,
            ),
            PhysicalImage(
                "saddle",
                (-0.8, 0.3),
                -2.0,
                ImageParity.NEGATIVE,
                MorseClass.SADDLE,
                fermat_potential_dimensionless=2.5,
                arrival_time_seconds=120.0,
            ),
        ),
        "phase3a-mst-fixture",
        "1",
    )
    kappa_ext = 0.12
    lambda_mst = 1.0 - kappa_ext
    transformed, transformed_source = apply_mass_sheet_transform(
        solution, source, kappa_ext
    )
    before = solution.physical_images
    after = transformed.physical_images
    source_scaling_error = float(
        np.max(np.abs(np.asarray(transformed_source) - lambda_mst * np.asarray(source)))
    )
    position_error = float(
        np.max(
            np.abs(
                np.asarray([image.position_arcsec for image in after])
                - np.asarray([image.position_arcsec for image in before])
            )
        )
    )
    signed_magnification_error = float(
        np.max(
            np.abs(
                np.asarray([image.mu_signed for image in after])
                - np.asarray([image.mu_signed for image in before]) / lambda_mst**2
            )
        )
    )
    absolute_magnification_error = float(
        np.max(
            np.abs(
                np.abs([image.mu_signed for image in after])
                - np.abs([image.mu_signed for image in before]) / lambda_mst**2
            )
        )
    )
    before_fermat = float(before[1].fermat_potential_dimensionless) - float(
        before[0].fermat_potential_dimensionless
    )
    after_fermat = float(after[1].fermat_potential_dimensionless) - float(
        after[0].fermat_potential_dimensionless
    )
    fermat_error = abs(after_fermat - lambda_mst * before_fermat)
    before_delay = float(before[1].arrival_time_seconds) - float(
        before[0].arrival_time_seconds
    )
    after_delay = float(after[1].arrival_time_seconds) - float(
        after[0].arrival_time_seconds
    )
    delay_error = abs(after_delay - lambda_mst * before_delay)
    errors = {
        "source_scaling_absolute_error": source_scaling_error,
        "image_position_invariance_absolute_error_arcsec": position_error,
        "signed_magnification_scaling_absolute_error": signed_magnification_error,
        "absolute_magnification_scaling_absolute_error": absolute_magnification_error,
        "fermat_difference_scaling_absolute_error": fermat_error,
        "physical_delay_scaling_absolute_error_seconds": delay_error,
    }
    tolerance = 1.0e-12
    result = {
        "status": "passed" if max(errors.values()) <= tolerance else "failed",
        "kappa_ext": kappa_ext,
        "lambda_mst": lambda_mst,
        "absolute_tolerance": tolerance,
        **errors,
        "external_convergence_connected_to_magnification_and_delay": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
