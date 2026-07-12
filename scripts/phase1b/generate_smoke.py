#!/usr/bin/env python3
"""Run or resume the authorized Phase 1B engineering-smoke generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gwlens_mm.config import load_yaml, validate_smoke_configuration
from gwlens_mm.smoke.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--authorizing-commit", required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--result", type=Path)
    arguments = parser.parse_args()
    config = load_yaml(arguments.config)
    validate_smoke_configuration(config, expected_execution_authorized=True)
    result = run_pipeline(
        config,
        arguments.generator_commit,
        arguments.authorizing_commit,
        arguments.policy_root,
        stop_after=arguments.stop_after,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.result:
        arguments.result.parent.mkdir(parents=True, exist_ok=True)
        arguments.result.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
