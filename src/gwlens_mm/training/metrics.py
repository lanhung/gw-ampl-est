"""Development-only metrics used by the preregistered learning curve."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


def negative_log_probability_per_dimension(log_probability: np.ndarray, dimensions: int) -> float:
    values = np.asarray(log_probability, dtype=np.float64)
    if dimensions <= 0 or values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("NLP inputs must be a finite vector and positive dimension")
    return float(-np.mean(values) / dimensions)


def empirical_crps(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute empirical CRPS in O(S log S) per case and dimension."""

    draws = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if draws.ndim != 3 or truth.shape != (draws.shape[0], draws.shape[2]):
        raise ValueError("samples must be (cases, draws, dimensions) with matching targets")
    if not np.all(np.isfinite(draws)) or not np.all(np.isfinite(truth)):
        raise ValueError("CRPS inputs contain NaN or Inf")
    ordered = np.sort(draws, axis=1)
    draw_count = draws.shape[1]
    coefficients = 2.0 * np.arange(1, draw_count + 1) - draw_count - 1.0
    first = np.mean(np.abs(draws - truth[:, None, :]), axis=1)
    half_pairwise = np.sum(
        ordered * coefficients[None, :, None], axis=1
    ) / (draw_count**2)
    return first - half_pairwise


def central_coverage(
    samples: np.ndarray,
    target: np.ndarray,
    levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
) -> Mapping[str, np.ndarray]:
    draws = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if draws.ndim != 3 or truth.shape != (draws.shape[0], draws.shape[2]):
        raise ValueError("coverage samples and targets have incompatible shapes")
    result: Dict[str, np.ndarray] = {}
    for level in levels:
        if not 0 < level < 1:
            raise ValueError("coverage levels must lie strictly between zero and one")
        alpha = (1.0 - level) / 2.0
        lower = np.quantile(draws, alpha, axis=1)
        upper = np.quantile(draws, 1.0 - alpha, axis=1)
        contained = (truth >= lower) & (truth <= upper)
        result[f"marginal_{level:.2f}"] = np.mean(contained, axis=0)
        result[f"joint_{level:.2f}"] = np.asarray(np.mean(np.all(contained, axis=1)))
        result[f"width_{level:.2f}"] = np.mean(upper - lower, axis=0)
    return result


def grouped_coverage(
    samples: np.ndarray,
    target: np.ndarray,
    groups: Sequence[str],
    *,
    level: float = 0.9,
) -> Mapping[str, Tuple[int, np.ndarray]]:
    if len(groups) != len(target):
        raise ValueError("group labels must match case count")
    result: Dict[str, Tuple[int, np.ndarray]] = {}
    group_array = np.asarray(groups, dtype=object)
    for group in sorted(set(groups)):
        selected = group_array == group
        coverage = central_coverage(samples[selected], target[selected], (level,))
        result[group] = (int(np.count_nonzero(selected)), coverage[f"marginal_{level:.2f}"])
    return result
