#!/usr/bin/env python3
"""Create an exact terminal-probe authorization after delegated review."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import yaml

from gwlens_mm.training.terminal_authorization import (
    build_terminal_probe_authorization,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--release-packet", required=True, type=Path)
    parser.add_argument("--delegated-review", required=True, type=Path)
    parser.add_argument("--retained-65k-output-root", required=True, type=Path)
    parser.add_argument("--training-output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("terminal probe authorization already exists")
    authorization = build_terminal_probe_authorization(
        args.root.resolve(),
        release_packet_path=args.release_packet,
        delegated_review_path=args.delegated_review,
        authorization_output_path=args.output,
        retained_65k_output_root=args.retained_65k_output_root,
        training_output_root=args.training_output_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    partial.write_text(
        yaml.safe_dump(authorization, sort_keys=False), encoding="utf-8"
    )
    partial.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
