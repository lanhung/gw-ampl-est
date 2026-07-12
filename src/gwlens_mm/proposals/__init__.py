"""Exact, deterministic proposal distributions and dry-run A/B contracts."""

from .ab_skeleton import ABRunPlan, build_dry_run_plan
from .exact_mixture import (
    Component,
    LatentDraw,
    ProposalSpecification,
    evaluate_latent_preflight,
    log_component_density,
    log_evaluation_density,
    log_importance_weight,
    log_mixture_density,
    sample_latent,
)
from .target_anchored import (
    TargetAnchoredSpecification,
    build_v3_dry_run_plan,
    ess_certificate,
    log_target_density,
    log_v3_density,
    log_v3_weight,
    sample_evaluation_target,
    sample_v3,
)

__all__ = [
    "ABRunPlan",
    "Component",
    "LatentDraw",
    "ProposalSpecification",
    "TargetAnchoredSpecification",
    "build_dry_run_plan",
    "build_v3_dry_run_plan",
    "ess_certificate",
    "evaluate_latent_preflight",
    "log_component_density",
    "log_evaluation_density",
    "log_importance_weight",
    "log_mixture_density",
    "log_target_density",
    "log_v3_density",
    "log_v3_weight",
    "sample_evaluation_target",
    "sample_latent",
    "sample_v3",
]
