#!/usr/bin/env python3
"""Emit the execution-disabled RC.7 reference-baseline plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from gwlens_mm.training.reference_baseline import write_dry_run_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_dry_run_plan(args.root.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
