#!/usr/bin/env python3
"""Bind six completed score artifacts for delegated statistics review."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from gwlens_mm.training.calibration_execution_authorization import (
    build_statistics_release_packet,
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--score-authorization", type=Path, required=True)
    parser.add_argument("--statistics-output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    packet = build_statistics_release_packet(
        arguments.root,
        score_authorization_path=arguments.score_authorization,
        statistics_output_root=arguments.statistics_output_root,
        output_path=arguments.output,
    )
    _atomic_json(arguments.output, packet)
    print(json.dumps(packet, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
