#!/usr/bin/env python3
"""Create a non-authorizing terminal architecture release-review packet."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from gwlens_mm.training.terminal_architecture_authorization import (
    build_terminal_architecture_release_packet,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--terminal-probe-authorization", required=True, type=Path)
    parser.add_argument("--terminal-decision", required=True, type=Path)
    parser.add_argument("--probe-output-root", required=True, type=Path)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--exact-wheel-test-result", required=True, type=Path)
    parser.add_argument("--environment-lock", required=True, type=Path)
    parser.add_argument("--architecture-output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.output.exists():
        raise FileExistsError("terminal architecture release packet already exists")
    packet = build_terminal_architecture_release_packet(
        arguments.root.resolve(),
        terminal_probe_authorization_path=arguments.terminal_probe_authorization,
        terminal_decision_path=arguments.terminal_decision,
        probe_output_root=arguments.probe_output_root,
        training_commit=arguments.training_commit,
        wheel_path=arguments.wheel,
        exact_wheel_test_result_path=arguments.exact_wheel_test_result,
        environment_lock_path=arguments.environment_lock,
        architecture_output_root=arguments.architecture_output_root,
        output_path=arguments.output,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    partial = arguments.output.with_name(arguments.output.name + ".partial")
    partial.write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
