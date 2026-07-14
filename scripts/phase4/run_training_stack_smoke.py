#!/usr/bin/env python3
"""Run a bounded in-memory model smoke without reading any dataset."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from gwlens_mm.training.contracts import load_training_stack_contract, model_configuration_hash
from gwlens_mm.training.engine import (
    set_deterministic_training_seed,
    validate_engineering_smoke_limits,
)
from gwlens_mm.training.model import build_probe_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--sample-count", type=int, default=16384)
    parser.add_argument("--examples", type=int, default=1)
    parser.add_argument("--optimizer-steps", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser


def _tensor_digest(values: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def run_smoke(
    root: Path, *, sample_count: int, examples: int, optimizer_steps: int
) -> Mapping[str, Any]:
    authorization, config = load_training_stack_contract(root)
    validate_engineering_smoke_limits(
        examples=examples,
        optimizer_steps=optimizer_steps,
        authorization=authorization,
    )
    if sample_count <= 0 or sample_count > int(config["inputs"]["gw"]["sample_count"]):
        raise ValueError("smoke sample count is outside the model contract")
    torch = importlib.import_module("torch")
    set_deterministic_training_seed(0)
    model = build_probe_model(config, seed=0)
    model.train()
    mask = torch.ones(examples, 2, 3)
    if examples:
        mask[0, 0, 2] = 0.0
    batch = {
        "gw_strain": torch.randn(examples, 2, 3, sample_count) * mask[..., None],
        "detector_mask": mask,
        "astrometry_items": torch.randn(examples, 5, 9),
        "astrometry_mask": torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float32).repeat(
            examples, 1
        ),
        "scalar_features": torch.randn(examples, 22),
        "scalar_mask": torch.ones(examples, 22),
        "modality_mask": torch.ones(examples, 7),
        "lens_family_condition": torch.tensor([[1.0, 0.0]]).repeat(examples, 1),
    }
    target = torch.randn(examples, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    for _ in range(optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        log_probability = model.log_prob(target, batch)
        loss = -log_probability.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("engineering smoke loss is nonfinite")
        loss.backward()
        if any(
            not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        ):
            raise FloatingPointError("engineering smoke gradient is nonfinite")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        final_log_probability = model.log_prob(target, batch)
        posterior_samples = model.sample(3, batch)
        posterior_sample_log_probability = model.sample_log_prob(
            posterior_samples, batch
        )
    if not torch.isfinite(final_log_probability).all() or not torch.isfinite(
        posterior_samples
    ).all() or not torch.isfinite(posterior_sample_log_probability).all():
        raise FloatingPointError("engineering smoke outputs are nonfinite")
    result = {
        "status": "passed_engineering_only",
        "model_configuration_hash": model_configuration_hash(config),
        "torch_version": str(torch.__version__),
        "examples": examples,
        "sample_count": sample_count,
        "optimizer_steps": optimizer_steps,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "losses": losses,
        "log_probability_shape": list(final_log_probability.shape),
        "posterior_sample_shape": list(posterior_samples.shape),
        "posterior_sample_log_probability_shape": list(
            posterior_sample_log_probability.shape
        ),
        "replay_sha256": _tensor_digest(
            (
                final_log_probability,
                posterior_samples,
                posterior_sample_log_probability,
            )
        ),
        "scientific_data_read": False,
        "stage_a_data_read": False,
        "checkpoint_written": False,
        "scientific_metric_reported": False,
        "model_selection_performed": False,
    }
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_smoke(
        arguments.root.resolve(),
        sample_count=arguments.sample_count,
        examples=arguments.examples,
        optimizer_steps=arguments.optimizer_steps,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
