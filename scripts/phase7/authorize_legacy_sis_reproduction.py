#!/usr/bin/env python3
"""Create one exact legacy SIS read-only authorization after review."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from gwlens_mm.training.legacy_sis_authorization import (
    build_legacy_sis_authorization,
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
    authorization = build_legacy_sis_authorization(
        root,
        release_packet_path=(root / arguments.release_packet).resolve(),
        delegated_review_path=(root / arguments.delegated_review).resolve(),
        output_path=output,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(
        yaml.safe_dump(authorization, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(partial, output)
    print(json.dumps(authorization, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
