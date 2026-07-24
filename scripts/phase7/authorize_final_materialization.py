#!/usr/bin/env python3
"""Create sealed materialization authorization after exact delegated review."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

import yaml

from gwlens_mm.production.final_evaluation_authorization import (
    build_final_materialization_authorization,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--release-packet", type=Path, required=True)
    parser.add_argument("--delegated-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.output.exists():
        raise FileExistsError("final materialization authorization already exists")
    authorization = build_final_materialization_authorization(
        arguments.root.resolve(),
        release_packet_path=arguments.release_packet,
        delegated_review_path=arguments.delegated_review,
        output_path=arguments.output,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    partial = arguments.output.with_name(arguments.output.name + ".partial")
    partial.write_text(
        yaml.safe_dump(authorization, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(partial, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
