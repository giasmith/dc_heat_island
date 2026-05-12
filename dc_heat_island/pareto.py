"""Pareto dominance helpers for energy-water-land suitability costs."""

from __future__ import annotations

import numpy as np


def pareto_mask_minimize(objectives: np.ndarray) -> np.ndarray:
    """Return True for non-dominated rows in a minimization problem.

    A row is dominated when another valid row is lower or equal for every
    objective and strictly lower for at least one objective.
    """

    values = np.asarray(objectives, dtype="float64")
    if values.ndim != 2:
        raise ValueError("objectives must be a 2D array")

    valid = np.all(np.isfinite(values), axis=1)
    mask = np.zeros(values.shape[0], dtype=bool)
    valid_indices = np.where(valid)[0]
    valid_values = values[valid_indices]
    efficient = np.ones(valid_values.shape[0], dtype=bool)

    for i, point in enumerate(valid_values):
        if not efficient[i]:
            continue
        dominated = np.any(np.all(valid_values <= point, axis=1) & np.any(valid_values < point, axis=1))
        if dominated:
            efficient[i] = False

    mask[valid_indices] = efficient
    return mask


def cost_matrix_from_channels(
    water_stress: np.ndarray,
    grid_distance: np.ndarray,
    land_cost: np.ndarray,
) -> np.ndarray:
    """Flatten W(p), D(p), and L(p) into an ``N x 3`` cost matrix."""

    if water_stress.shape != grid_distance.shape or water_stress.shape != land_cost.shape:
        raise ValueError("all cost surfaces must have the same shape")
    return np.column_stack([water_stress.ravel(), grid_distance.ravel(), land_cost.ravel()])

