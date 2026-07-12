#!/usr/bin/env python3
"""Render the small Phase 1B composition and numerical-contract diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    data = json.loads(arguments.validation.read_text(encoding="utf-8"))

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
    families = ["SIS", "SIE+shear", "EPL+shear"]
    counts = [
        data["family_counts"]["sis"],
        data["family_counts"]["sie_external_shear"],
        data["family_counts"]["epl_external_shear"],
    ]
    axes[0].bar(families, counts, color=("#4477AA", "#66CCEE", "#228833"))
    axes[0].set_ylabel("Accepted pairs")
    axes[0].set_title("Engineering-smoke composition")
    axes[0].set_ylim(0, 27)
    for index, count in enumerate(counts):
        axes[0].text(index, count + 0.5, str(count), ha="center")

    labels = ["Matched response", "Morse phase", "Preprocessing"]
    errors = [
        data["matched_response_max_relative_error"],
        data["morse_phase_max_relative_error"],
        data["preprocessing_max_relative_error"],
    ]
    axes[1].barh(labels, errors, color="#CC6677")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Maximum relative error")
    axes[1].set_title("Numerical contracts")
    axes[1].set_xlim(1e-18, 1e-14)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
