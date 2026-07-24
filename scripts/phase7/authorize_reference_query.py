#!/usr/bin/env python3
"""Promote one exact RC.7 reference query after delegated review."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from gwlens_mm.training.reference_authorization import (
    build_reference_query_authorization,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--release-packet", type=Path, required=True)
    parser.add_argument("--delegated-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    output = (root / arguments.output).resolve()
    authorization = build_reference_query_authorization(
        root,
        release_packet_path=(root / arguments.release_packet).resolve(),
        delegated_review_path=(root / arguments.delegated_review).resolve(),
        output_path=output,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(yaml.safe_dump(authorization, sort_keys=False))
    os.replace(temporary, output)
    print(json.dumps(authorization, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
