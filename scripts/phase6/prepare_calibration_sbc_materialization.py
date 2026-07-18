#!/usr/bin/env python3
"""Render the frozen calibration/SBC data-pool plan without generating data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gwlens_mm.production.calibration_sbc import dry_run_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if arguments.execute:
        raise PermissionError("calibration/SBC materialization remains unauthorized")
    print(json.dumps(dry_run_plan(arguments.root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
