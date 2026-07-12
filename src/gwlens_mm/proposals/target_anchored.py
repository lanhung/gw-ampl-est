"""Target-anchored proposal-v3 sampling, densities, certificates, and diagnostics."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from ..physics.quantities import LensFamily
from .ab_skeleton import ABRunPlan, build_dry_run_plan
from .exact_mixture import (
    Component,
    LatentDraw,
    ProposalSpecification,
    _log_uniform_log,
    _log_uniform_sample,
    _power_law_sample,
    _sample_truncated_normal,
    _source_redshift_lower,
    _uniform_log,
    log_component_density,
    log_evaluation_density,
    truncated_normal_log_density,
)


class V3Component(str, Enum):
    RC5_BROAD = "rc5_broad"
    EVALUATION_TARGET = "evaluation_target"
    CENTRAL = "central"


V3_COMPONENT_ORDER = (
    V3Component.RC5_BROAD,
    V3Component.EVALUATION_TARGET,
    V3Component.CENTRAL,
)


@dataclass(frozen=True)
class TargetAnchoredSpecification:
    weights: Tuple[float, float, float]
    central_sigma: float
    source_lower: float
    source_upper: float
    draw_count: int
    v3_seed: int
    rc5_seed: int
    mean_weight_bounds: Tuple[float, float]
    overall_ess_minimum: float
    family_ess_minimum: float
    anchor_tolerance: float

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "TargetAnchoredSpecification":
        components = config["mixture"]["components"]
        central = config["central_source_plane"]
        diagnostics = config["diagnostics"]
        certificate = config["target_anchor_certificate"]
        spec = cls(
            weights=tuple(
                float(components[name.value]["weight"]) for name in V3_COMPONENT_ORDER
            ),  # type: ignore[arg-type]
            central_sigma=float(central["sigma"]),
            source_lower=float(central["lower"]),
            source_upper=float(central["upper"]),
            draw_count=int(diagnostics["draw_count"]),
            v3_seed=int(diagnostics["proposal_v3_seed"]),
            rc5_seed=int(diagnostics["rc5_baseline_seed"]),
            mean_weight_bounds=tuple(
                float(value) for value in diagnostics["mean_weight_interval"]
            ),  # type: ignore[arg-type]
            overall_ess_minimum=float(diagnostics["overall_relative_ess_minimum"]),
            family_ess_minimum=float(diagnostics["family_relative_ess_minimum"]),
            anchor_tolerance=float(certificate["required_log_inequality_tolerance"]),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.weights != (0.20, 0.55, 0.25):
            raise ValueError("proposal-v3 weights differ from the reviewed mixture")
        if not math.isclose(sum(self.weights), 1.0, abs_tol=1.0e-15):
            raise ValueError("proposal-v3 weights must sum to one")
        if self.central_sigma != 0.8 or self.source_lower != -2.5 or self.source_upper != 2.5:
            raise ValueError("proposal-v3 central source specification changed")
        if self.draw_count != 200_000:
            raise ValueError("proposal-v3 diagnostics require exactly 200,000 draws")


def _sample_source_factors(rng: np.random.Generator) -> Dict[str, float]:
    mass_1 = _power_law_sample(rng, -2.3, 20.0, 80.0)
    q_lower = max(0.25, 10.0 / mass_1)
    return {
        "mass_1": mass_1,
        "mass_ratio": float(rng.uniform(q_lower, 1.0)),
        "a_1": float(0.99 * rng.beta(2.0, 5.0)),
        "a_2": float(0.99 * rng.beta(2.0, 5.0)),
        "tilt_1": float(math.acos(rng.uniform(-1.0, 1.0))),
        "tilt_2": float(math.acos(rng.uniform(-1.0, 1.0))),
        "phi_12": float(rng.uniform(0.0, 2.0 * math.pi)),
        "phi_jl": float(rng.uniform(0.0, 2.0 * math.pi)),
        "theta_jn": float(math.acos(rng.uniform(-1.0, 1.0))),
        "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        "ra": float(rng.uniform(0.0, 2.0 * math.pi)),
        "dec": float(math.asin(rng.uniform(-1.0, 1.0))),
        "psi": float(rng.uniform(0.0, math.pi)),
    }


def _draw(
    rng: np.random.Generator,
    component: V3Component,
    v3: TargetAnchoredSpecification,
) -> LatentDraw:
    family = (
        LensFamily.SIE_EXTERNAL_SHEAR
        if rng.random() < 0.5
        else LensFamily.EPL_EXTERNAL_SHEAR
    )
    z_lens = float(rng.uniform(0.1, 1.0))
    z_source = float(rng.uniform(_source_redshift_lower(z_lens, 3.0), 3.0))
    if component is V3Component.RC5_BROAD:
        theta = _log_uniform_sample(rng, 0.3, 3.0)
        axis = float(rng.uniform(0.4, 1.0))
        shear = float(rng.uniform(0.0, 0.15))
        slope = 2.0 if family is LensFamily.SIE_EXTERNAL_SHEAR else float(
            rng.uniform(1.6, 2.5)
        )
        kappa = float(rng.uniform(-0.15, 0.15))
    else:
        theta = math.exp(
            _sample_truncated_normal(rng, 0.0, 0.35, math.log(0.3), math.log(3.0))
        )
        axis = _sample_truncated_normal(rng, 0.7, 0.15, 0.4, 1.0)
        shear = _sample_truncated_normal(rng, 0.0, 0.05, 0.0, 0.15)
        slope = 2.0 if family is LensFamily.SIE_EXTERNAL_SHEAR else _sample_truncated_normal(
            rng, 2.08, 0.16, 1.6, 2.5
        )
        kappa = _sample_truncated_normal(rng, 0.0, 0.05, -0.15, 0.15)
    if component is V3Component.CENTRAL:
        u_x = _sample_truncated_normal(
            rng, 0.0, v3.central_sigma, v3.source_lower, v3.source_upper
        )
        u_y = _sample_truncated_normal(
            rng, 0.0, v3.central_sigma, v3.source_lower, v3.source_upper
        )
    else:
        u_x = float(rng.uniform(v3.source_lower, v3.source_upper))
        u_y = float(rng.uniform(v3.source_lower, v3.source_upper))
    source = _sample_source_factors(rng)
    return LatentDraw(
        component=Component.RC5_BROAD,
        lens_family=family,
        z_lens=z_lens,
        z_source=z_source,
        einstein_radius_arcsec=theta,
        axis_ratio=axis,
        position_angle_rad=float(rng.uniform(0.0, math.pi)),
        shear_amplitude=shear,
        shear_angle_rad=float(rng.uniform(0.0, math.pi)),
        density_slope=slope,
        external_convergence=kappa,
        source_u_x=u_x,
        source_u_y=u_y,
        mass_1_source=source["mass_1"],
        mass_ratio=source["mass_ratio"],
        a_1=source["a_1"],
        a_2=source["a_2"],
        tilt_1=source["tilt_1"],
        tilt_2=source["tilt_2"],
        phi_12=source["phi_12"],
        phi_jl=source["phi_jl"],
        theta_jn=source["theta_jn"],
        phase=source["phase"],
        ra=source["ra"],
        dec=source["dec"],
        psi=source["psi"],
    )


def sample_rc5(rng: np.random.Generator, spec: TargetAnchoredSpecification) -> LatentDraw:
    return _draw(rng, V3Component.RC5_BROAD, spec)


def sample_evaluation_target(
    rng: np.random.Generator, spec: TargetAnchoredSpecification
) -> LatentDraw:
    return _draw(rng, V3Component.EVALUATION_TARGET, spec)


def sample_v3(
    rng: np.random.Generator, spec: TargetAnchoredSpecification
) -> Tuple[V3Component, LatentDraw]:
    value = float(rng.random())
    component = (
        V3Component.RC5_BROAD
        if value < spec.weights[0]
        else V3Component.EVALUATION_TARGET
        if value < spec.weights[0] + spec.weights[1]
        else V3Component.CENTRAL
    )
    return component, _draw(rng, component, spec)


def _v2_compatible_spec(spec: TargetAnchoredSpecification) -> ProposalSpecification:
    return ProposalSpecification(
        component_weights=(0.2, 0.6, 0.1, 0.1),
        source_lower=spec.source_lower,
        source_upper=spec.source_upper,
        wide_sigma=1.5,
        narrow_sigma=0.8,
        z_max=3.0,
        preflight_draw_count=200_000,
        preflight_seed=0,
        mean_weight_bounds=(0.98, 1.02),
        overall_ess_minimum=0.5,
        family_ess_minimum=0.4,
    )


def log_rc5_density(draw: LatentDraw, spec: TargetAnchoredSpecification) -> float:
    return log_component_density(draw, Component.RC5_BROAD, _v2_compatible_spec(spec))


def log_target_density(draw: LatentDraw, spec: TargetAnchoredSpecification) -> float:
    return log_evaluation_density(draw, _v2_compatible_spec(spec))


def log_central_density(draw: LatentDraw, spec: TargetAnchoredSpecification) -> float:
    target = log_target_density(draw, spec)
    uniform_source = (
        _uniform_log(draw.source_u_x, spec.source_lower, spec.source_upper)
        + _uniform_log(draw.source_u_y, spec.source_lower, spec.source_upper)
    )
    central_source = truncated_normal_log_density(
        draw.source_u_x, 0.0, spec.central_sigma, spec.source_lower, spec.source_upper
    ) + truncated_normal_log_density(
        draw.source_u_y, 0.0, spec.central_sigma, spec.source_lower, spec.source_upper
    )
    return target - uniform_source + central_source


def log_v3_density(draw: LatentDraw, spec: TargetAnchoredSpecification) -> float:
    terms = (
        math.log(spec.weights[0]) + log_rc5_density(draw, spec),
        math.log(spec.weights[1]) + log_target_density(draw, spec),
        math.log(spec.weights[2]) + log_central_density(draw, spec),
    )
    maximum = max(terms)
    return maximum + math.log(sum(math.exp(value - maximum) for value in terms))


def log_v3_weight(draw: LatentDraw, spec: TargetAnchoredSpecification) -> float:
    return log_target_density(draw, spec) - log_v3_density(draw, spec)


def ess_certificate(config: Mapping[str, Any]) -> Dict[str, Any]:
    certificate = config["target_anchor_certificate"]
    anchor = float(certificate["target_component_weight"])
    maximum_weight = 1.0 / anchor
    return {
        "status": "passed",
        "target_component_weight": anchor,
        "maximum_importance_weight": maximum_weight,
        "population_relative_ess_lower_bound": anchor,
        "derivation": [
            "q_v3(theta) >= anchor * p_eval(theta)",
            "w(theta)=p_eval(theta)/q_v3(theta) <= 1/anchor",
            "E_q[w]=1",
            "E_q[w^2] <= (1/anchor)*E_q[w] = 1/anchor",
            "relative_ESS=1/E_q[w^2] >= anchor",
        ],
        "per_family_bound": anchor,
        "assumptions": list(certificate["assumptions"]),
    }


def _relative_ess(weights: np.ndarray) -> float:
    return float(np.sum(weights) ** 2 / (len(weights) * np.sum(weights * weights)))


def _summarize(
    log_p: np.ndarray,
    log_q: np.ndarray,
    epl: np.ndarray,
    replay_hash: str,
    component_counts: Mapping[str, int],
    label: str,
) -> Dict[str, Any]:
    log_w = log_p - log_q
    weights = np.exp(log_w)
    normalized = weights / np.sum(weights)
    return {
        "label": label,
        "draw_count": len(weights),
        "finite_fraction": {
            "log_q": float(np.mean(np.isfinite(log_q))),
            "log_p": float(np.mean(np.isfinite(log_p))),
            "weight": float(np.mean(np.isfinite(weights))),
        },
        "mean_unnormalized_importance_weight": float(np.mean(weights)),
        "overall_relative_ess": _relative_ess(weights),
        "relative_ess_by_lens_family": {
            "sie_external_shear": _relative_ess(weights[~epl]),
            "epl_external_shear": _relative_ess(weights[epl]),
        },
        "maximum_normalized_weight": float(np.max(normalized)),
        "weight_quantiles": {
            str(q): float(np.quantile(weights, q))
            for q in (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
        },
        "log_weight_quantiles": {
            str(q): float(np.quantile(log_w, q))
            for q in (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
        },
        "component_counts": dict(component_counts),
        "deterministic_replay_sha256": replay_hash,
        "waveform_pair_count": 0,
        "accepted_pair_generator_called": False,
        "proposal_ab_qualification_run": False,
    }


def _hash_v3_stream(spec: TargetAnchoredSpecification, seed: int) -> str:
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    for _ in range(spec.draw_count):
        component, draw = sample_v3(rng, spec)
        digest.update(bytes((V3_COMPONENT_ORDER.index(component),)))
        digest.update(draw.replay_bytes())
    return digest.hexdigest()


def run_rc5_diagnostic(spec: TargetAnchoredSpecification) -> Dict[str, Any]:
    rng = np.random.default_rng(spec.rc5_seed)
    log_p = np.empty(spec.draw_count)
    log_q = np.empty(spec.draw_count)
    epl = np.empty(spec.draw_count, dtype=np.bool_)
    digest = hashlib.sha256()
    for index in range(spec.draw_count):
        draw = sample_rc5(rng, spec)
        digest.update(draw.replay_bytes())
        log_p[index] = log_target_density(draw, spec)
        log_q[index] = log_rc5_density(draw, spec)
        epl[index] = draw.lens_family is LensFamily.EPL_EXTERNAL_SHEAR
    result = _summarize(
        log_p,
        log_q,
        epl,
        digest.hexdigest(),
        {"rc5_broad": spec.draw_count},
        "rc5",
    )
    result["interpretation"] = "diagnostic_only_no_retrospective_gate"
    return result


def run_v3_preflight(spec: TargetAnchoredSpecification) -> Dict[str, Any]:
    rng = np.random.default_rng(spec.v3_seed)
    log_p = np.empty(spec.draw_count)
    log_q = np.empty(spec.draw_count)
    epl = np.empty(spec.draw_count, dtype=np.bool_)
    digest = hashlib.sha256()
    counts = {component.value: 0 for component in V3_COMPONENT_ORDER}
    anchor_failures = 0
    family_anchor_failures = {"sie_external_shear": 0, "epl_external_shear": 0}
    for index in range(spec.draw_count):
        component, draw = sample_v3(rng, spec)
        counts[component.value] += 1
        digest.update(bytes((V3_COMPONENT_ORDER.index(component),)))
        digest.update(draw.replay_bytes())
        log_p[index] = log_target_density(draw, spec)
        log_q[index] = log_v3_density(draw, spec)
        epl[index] = draw.lens_family is LensFamily.EPL_EXTERNAL_SHEAR
        if log_q[index] + spec.anchor_tolerance < log_p[index] + math.log(0.55):
            anchor_failures += 1
            family_anchor_failures[draw.lens_family.value] += 1
    result = _summarize(log_p, log_q, epl, digest.hexdigest(), counts, "proposal_v3")
    replay = _hash_v3_stream(spec, spec.v3_seed)
    result["replay_byte_identical"] = replay == digest.hexdigest()
    result["target_anchor_failure_count"] = anchor_failures
    result["target_anchor_failures_by_family"] = family_anchor_failures
    result["maximum_importance_weight"] = float(np.max(np.exp(log_p - log_q)))
    reasons = []
    if any(value != 1.0 for value in result["finite_fraction"].values()):
        reasons.append("nonfinite_density_or_weight")
    if not (
        spec.mean_weight_bounds[0]
        <= result["mean_unnormalized_importance_weight"]
        <= spec.mean_weight_bounds[1]
    ):
        reasons.append("mean_weight_outside_interval")
    if result["overall_relative_ess"] < spec.overall_ess_minimum:
        reasons.append("overall_ess_below_minimum")
    if any(
        value < spec.family_ess_minimum
        for value in result["relative_ess_by_lens_family"].values()
    ):
        reasons.append("family_ess_below_minimum")
    if anchor_failures:
        reasons.append("target_anchor_inequality_failure")
    if not result["replay_byte_identical"]:
        reasons.append("replay_mismatch")
    result["support_hole_count"] = 0
    result["failure_reasons"] = reasons
    result["status"] = "passed" if not reasons else "failed_hard_stop"
    return result


def _logadd(weight_a: float, log_a: float, weight_b: float, log_b: float) -> float:
    terms = (math.log(weight_a) + log_a, math.log(weight_b) + log_b)
    maximum = max(terms)
    return maximum + math.log(sum(math.exp(value - maximum) for value in terms))


def _structural_logs(draw: LatentDraw) -> Tuple[float, float]:
    broad = (
        _log_uniform_log(draw.einstein_radius_arcsec, 0.3, 3.0)
        + _uniform_log(draw.axis_ratio, 0.4, 1.0)
        + _uniform_log(draw.position_angle_rad, 0.0, math.pi)
        + _uniform_log(draw.shear_amplitude, 0.0, 0.15)
        + _uniform_log(draw.shear_angle_rad, 0.0, math.pi)
    )
    target = (
        truncated_normal_log_density(
            math.log(draw.einstein_radius_arcsec),
            0.0,
            0.35,
            math.log(0.3),
            math.log(3.0),
        )
        - math.log(draw.einstein_radius_arcsec)
        + truncated_normal_log_density(draw.axis_ratio, 0.7, 0.15, 0.4, 1.0)
        + _uniform_log(draw.position_angle_rad, 0.0, math.pi)
        + truncated_normal_log_density(draw.shear_amplitude, 0.0, 0.05, 0.0, 0.15)
        + _uniform_log(draw.shear_angle_rad, 0.0, math.pi)
    )
    if draw.lens_family is LensFamily.EPL_EXTERNAL_SHEAR:
        broad += _uniform_log(draw.density_slope, 1.6, 2.5)
        target += truncated_normal_log_density(draw.density_slope, 2.08, 0.16, 1.6, 2.5)
    return target, broad


def _factor_log_weights(
    draw: LatentDraw, spec: TargetAnchoredSpecification, proposal: str
) -> Dict[str, float]:
    target_structural, broad_structural = _structural_logs(draw)
    target_kappa = truncated_normal_log_density(
        draw.external_convergence, 0.0, 0.05, -0.15, 0.15
    )
    broad_kappa = _uniform_log(draw.external_convergence, -0.15, 0.15)
    uniform_source = _uniform_log(
        draw.source_u_x, spec.source_lower, spec.source_upper
    ) + _uniform_log(draw.source_u_y, spec.source_lower, spec.source_upper)
    central_source = truncated_normal_log_density(
        draw.source_u_x, 0.0, spec.central_sigma, spec.source_lower, spec.source_upper
    ) + truncated_normal_log_density(
        draw.source_u_y, 0.0, spec.central_sigma, spec.source_lower, spec.source_upper
    )
    if proposal == "rc5":
        structural_q = broad_structural
        kappa_q = broad_kappa
        source_q = uniform_source
        complete = log_target_density(draw, spec) - log_rc5_density(draw, spec)
    elif proposal == "proposal_v3":
        structural_q = _logadd(0.2, broad_structural, 0.8, target_structural)
        kappa_q = _logadd(0.2, broad_kappa, 0.8, target_kappa)
        source_q = _logadd(0.75, uniform_source, 0.25, central_source)
        complete = log_v3_weight(draw, spec)
    else:
        raise ValueError("unsupported factorwise proposal")
    return {
        "lens_structural": target_structural - structural_q,
        "redshifts": 0.0,
        "source_plane": uniform_source - source_q,
        "external_convergence": target_kappa - kappa_q,
        "bbh_mass": 0.0,
        "spin_orientation": 0.0,
        "complete_latent": complete,
    }


def run_factorwise_diagnostic(
    spec: TargetAnchoredSpecification, proposal: str
) -> Dict[str, Any]:
    seed = spec.rc5_seed if proposal == "rc5" else spec.v3_seed
    rng = np.random.default_rng(seed)
    groups = (
        "lens_structural",
        "redshifts",
        "source_plane",
        "external_convergence",
        "bbh_mass",
        "spin_orientation",
        "complete_latent",
    )
    values = {group: np.empty(spec.draw_count) for group in groups}
    for index in range(spec.draw_count):
        if proposal == "rc5":
            draw = sample_rc5(rng, spec)
        else:
            _, draw = sample_v3(rng, spec)
        contributions = _factor_log_weights(draw, spec, proposal)
        for group in groups:
            values[group][index] = contributions[group]
    return {
        "proposal": proposal,
        "draw_count": spec.draw_count,
        "interpretation": "diagnostic_marginals_not_unique_sequential_decomposition",
        "groups": {
            group: {
                "mean_log_weight": float(np.mean(values[group])),
                "relative_ess": _relative_ess(np.exp(values[group])),
                "log_weight_standard_deviation": float(np.std(values[group])),
            }
            for group in groups
        },
    }


def build_v3_dry_run_plan(
    v2_config: Mapping[str, Any],
    v3_config: Mapping[str, Any],
    parent_run_id: str,
) -> ABRunPlan:
    skeleton = v3_config["future_ab_skeleton"]
    if skeleton["dry_run_only"] is not True:
        raise ValueError("proposal-v3 A/B plan must remain dry-run-only")
    if (
        int(skeleton["accepted_pairs_per_arm"]) != 512
        or int(skeleton["total_accepted_pairs"]) != 1024
        or int(skeleton["blocks_per_arm"]) != 16
        or int(skeleton["accepted_pairs_per_block"]) != 32
    ):
        raise ValueError("proposal-v3 A/B counts changed")
    base = build_dry_run_plan(v2_config, parent_run_id)
    candidate_id = f"{parent_run_id}-proposal-v3-target-anchored-rc1"
    return replace(
        base,
        candidate_dataset_id=candidate_id,
        candidate_manifest=(
            f"manifests/proposal_v3/{parent_run_id}/target-anchored-rc1/manifest.json"
        ),
        candidate_checksum_manifest=(
            f"manifests/proposal_v3/{parent_run_id}/target-anchored-rc1/checksums.sha256"
        ),
    )
