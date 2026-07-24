#!/usr/bin/env python3
"""Build a non-authorizing sealed final-evaluation release packet."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

from gwlens_mm.production.final_evaluation_authorization import (
    build_final_materialization_release_packet,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--terminal-decision", type=Path, required=True)
    parser.add_argument("--architecture-decision", type=Path, required=True)
    parser.add_argument("--reference-catalog", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--exact-wheel-result", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    packet = build_final_materialization_release_packet(
        arguments.root.resolve(),
        implementation_commit=arguments.implementation_commit,
        terminal_decision_path=arguments.terminal_decision,
        architecture_decision_path=arguments.architecture_decision,
        reference_catalog_path=arguments.reference_catalog,
        wheel_path=arguments.wheel,
        exact_wheel_test_result_path=arguments.exact_wheel_result,
        environment_lock_path=arguments.environment_lock,
        output_path=arguments.output,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    partial = arguments.output.with_name(arguments.output.name + ".partial")
    partial.write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
