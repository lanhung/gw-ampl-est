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

__all__ = [
    "ABRunPlan",
    "Component",
    "LatentDraw",
    "ProposalSpecification",
    "build_dry_run_plan",
    "evaluate_latent_preflight",
    "log_component_density",
    "log_evaluation_density",
    "log_importance_weight",
    "log_mixture_density",
    "sample_latent",
]
