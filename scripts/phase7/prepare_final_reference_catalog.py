#!/usr/bin/env python3
"""Build the structured final-evaluation reference catalog without opening strain."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

from gwlens_mm.production.final_evaluation_authorization import (
    build_final_reference_catalog,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--terminal-probe-authorization",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--calibration-sbc-result",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    catalog = build_final_reference_catalog(
        arguments.root.resolve(),
        terminal_probe_authorization_path=arguments.terminal_probe_authorization,
        calibration_sbc_result_path=arguments.calibration_sbc_result,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    partial = arguments.output.with_name(arguments.output.name + ".partial")
    partial.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
