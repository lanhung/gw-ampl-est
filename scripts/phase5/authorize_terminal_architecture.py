#!/usr/bin/env python3
"""Create exact terminal architecture authorization after delegated review."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import yaml

from gwlens_mm.training.terminal_architecture_authorization import (
    build_terminal_architecture_authorization,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--release-packet", required=True, type=Path)
    parser.add_argument("--delegated-review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.output.exists():
        raise FileExistsError("terminal architecture authorization already exists")
    authorization = build_terminal_architecture_authorization(
        arguments.root.resolve(),
        release_packet_path=arguments.release_packet,
        delegated_review_path=arguments.delegated_review,
        output_path=arguments.output,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    partial = arguments.output.with_name(arguments.output.name + ".partial")
    partial.write_text(
        yaml.safe_dump(authorization, sort_keys=False), encoding="utf-8"
    )
    partial.replace(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
