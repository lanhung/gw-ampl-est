#!/usr/bin/env python3
"""Create a non-authorizing review packet for the terminal 131k probe."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

from gwlens_mm.training.terminal_release import (
    prepare_terminal_probe_review_packet,
    validate_terminal_release_checkout_paths,
)


def _git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _gpu_names() -> tuple[str, ...]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())


def _verify_release_checkout(root: Path, training_commit: str) -> str:
    head = _git_output(root, "rev-parse", "HEAD")
    if head != training_commit:
        subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", training_commit, head],
            check=True,
        )
        changed = _git_output(
            root, "diff", "--name-only", f"{training_commit}..{head}"
        ).splitlines()
        validate_terminal_release_checkout_paths(changed)
    if _git_output(root, "status", "--porcelain"):
        raise RuntimeError("terminal probe review requires a clean checkout")
    return head


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--closeout-result", required=True, type=Path)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--environment-lock", required=True, type=Path)
    parser.add_argument("--wheel-test-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    review_checkout_commit = _verify_release_checkout(
        args.root.resolve(), args.training_commit
    )
    packet = prepare_terminal_probe_review_packet(
        args.root,
        closeout_result_path=args.closeout_result,
        training_commit=args.training_commit,
        review_checkout_commit=review_checkout_commit,
        wheel_path=args.wheel,
        environment_lock_path=args.environment_lock,
        wheel_test_result_path=args.wheel_test_result,
        gpu_names=_gpu_names(),
    )
    _atomic_json(args.output, packet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
