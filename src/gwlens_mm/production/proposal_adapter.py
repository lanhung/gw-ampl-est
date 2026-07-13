"""Adapters from exact latent proposals to the qualified production generator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..proposals.target_anchored import (
    TargetAnchoredSpecification,
    V3Component,
    log_central_density,
    log_rc5_density,
    log_target_density,
    log_v3_density,
    sample_evaluation_target,
    sample_rc5,
    sample_v3,
)
from .population import PopulationDraw


@dataclass(frozen=True)
class ProductionProposalDraw:
    population: PopulationDraw
    component: str
    component_log_densities: Mapping[str, float]


def _population(draw: Any, log_q: float, log_p: float) -> PopulationDraw:
    shear_gamma1 = draw.shear_amplitude * math.cos(2.0 * draw.shear_angle_rad)
    shear_gamma2 = draw.shear_amplitude * math.sin(2.0 * draw.shear_angle_rad)
    return PopulationDraw(
        lens_family=draw.lens_family,
        z_lens=draw.z_lens,
        z_source=draw.z_source,
        lens_parameters={
            "z_lens": draw.z_lens,
            "z_source": draw.z_source,
            "einstein_radius_arcsec": draw.einstein_radius_arcsec,
            "axis_ratio": draw.axis_ratio,
            "position_angle_rad": draw.position_angle_rad,
            "shear_amplitude": draw.shear_amplitude,
            "shear_angle_rad": draw.shear_angle_rad,
            "shear_gamma1": shear_gamma1,
            "shear_gamma2": shear_gamma2,
            "density_slope": draw.density_slope,
            "external_convergence": draw.external_convergence,
        },
        source_position_arcsec=draw.source_position_arcsec,
        external_convergence=draw.external_convergence,
        source_parameters={
            "mass_1_source": draw.mass_1_source,
            "mass_2_source": draw.mass_2_source,
            "mass_ratio": draw.mass_ratio,
            "a_1": draw.a_1,
            "a_2": draw.a_2,
            "tilt_1": draw.tilt_1,
            "tilt_2": draw.tilt_2,
            "phi_12": draw.phi_12,
            "phi_jl": draw.phi_jl,
            "theta_jn": draw.theta_jn,
            "phase": draw.phase,
            "ra": draw.ra,
            "dec": draw.dec,
            "psi": draw.psi,
        },
        proposal_log_probability=log_q,
        evaluation_log_probability=log_p,
    )


def sample_production_proposal(
    rng: np.random.Generator,
    *,
    mode: str,
    proposal_config: Mapping[str, Any],
) -> ProductionProposalDraw:
    """Sample RC.5, v3, or the direct target with exact density provenance."""

    specification = TargetAnchoredSpecification.from_mapping(proposal_config)
    if mode == "evaluation_target_direct":
        draw = sample_evaluation_target(rng, specification)
        log_target = log_target_density(draw, specification)
        if not math.isfinite(log_target):
            raise ValueError("direct evaluation-target proposal returned nonfinite density")
        return ProductionProposalDraw(
            _population(draw, log_target, log_target),
            "evaluation_target",
            {"evaluation_target": log_target},
        )
    if mode == "rc5_control":
        draw = sample_rc5(rng, specification)
        log_rc5 = log_rc5_density(draw, specification)
        log_target = log_target_density(draw, specification)
        return ProductionProposalDraw(
            _population(draw, log_rc5, log_target),
            "rc5_broad",
            {"rc5_broad": log_rc5},
        )
    if mode != "proposal_v3_candidate":
        raise ValueError(f"unsupported production proposal mode: {mode}")
    component, draw = sample_v3(rng, specification)
    component_logs = {
        "rc5_broad": log_rc5_density(draw, specification),
        "evaluation_target": log_target_density(draw, specification),
        "central": log_central_density(draw, specification),
    }
    log_q = log_v3_density(draw, specification)
    log_p = component_logs["evaluation_target"]
    if not all(math.isfinite(value) for value in (*component_logs.values(), log_q, log_p)):
        raise ValueError("production proposal returned nonfinite density provenance")
    return ProductionProposalDraw(
        _population(draw, log_q, log_p),
        component.value if isinstance(component, V3Component) else str(component),
        component_logs,
    )
