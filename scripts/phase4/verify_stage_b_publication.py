#!/usr/bin/env python3
"""Independently verify the atomic Stage B and combined 65k publications."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

from gwlens_mm.training.rung65 import validate_stage_b_completion_evidence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--stage-a-publication", required=True, type=Path)
    parser.add_argument("--stage-b-publication", required=True, type=Path)
    parser.add_argument("--combined-publication", required=True, type=Path)
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--orchestration-commit", required=True)
    parser.add_argument("--preregistration-hash", required=True)
    parser.add_argument("--minimum-free-bytes", type=int, default=100_000_000_000)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    evidence = validate_stage_b_completion_evidence(
        result_path=arguments.result,
        stage_a_publication_root=arguments.stage_a_publication,
        stage_b_publication_root=arguments.stage_b_publication,
        combined_publication_root=arguments.combined_publication,
        expected_generator_commit=arguments.generator_commit,
        expected_orchestration_commit=arguments.orchestration_commit,
        expected_preregistration_hash=arguments.preregistration_hash,
        minimum_remaining_free_bytes=arguments.minimum_free_bytes,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    partial = arguments.output.with_name(arguments.output.name + ".partial")
    partial.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(partial, arguments.output)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
