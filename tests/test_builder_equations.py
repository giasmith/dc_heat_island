from __future__ import annotations

import unittest

import numpy as np

from dc_heat_island.builder import (
    compute_grid_distance,
    compute_land_cost,
    compute_slope,
    map_land_cover_penalty,
)
from dc_heat_island.config import LandCostConfig
from dc_heat_island.pareto import cost_matrix_from_channels, pareto_mask_minimize


class BuilderEquationTests(unittest.TestCase):
    def test_grid_distance_uses_metric_sampling(self) -> None:
        grid = np.zeros((5, 5), dtype="uint8")
        grid[2, 2] = 1
        distance = compute_grid_distance(grid, x_resolution=10, y_resolution=20)
        self.assertEqual(distance[2, 2], 0)
        self.assertAlmostEqual(float(distance[2, 3]), 10)
        self.assertAlmostEqual(float(distance[3, 2]), 20)

    def test_land_cover_penalty_mapping(self) -> None:
        land_cover = np.array([[4, 7, 2, 8, 1, 5, 9]], dtype="float32")
        penalties = map_land_cover_penalty(land_cover, LandCostConfig())
        self.assertEqual(penalties[0, 0], 0)
        self.assertEqual(penalties[0, 1], 0)
        self.assertEqual(penalties[0, 2], 1)
        self.assertEqual(penalties[0, 3], 1)
        self.assertEqual(penalties[0, 4], 1000000)
        self.assertEqual(penalties[0, 5], 1000000)
        self.assertEqual(penalties[0, 6], 5)

    def test_land_cost_equation(self) -> None:
        slope = np.array([[0.5, 1.0]], dtype="float32")
        penalty = np.array([[2.0, 3.0]], dtype="float32")
        np.testing.assert_allclose(compute_land_cost(slope, penalty, alpha=2.0), [[3.0, 5.0]])

    def test_slope_gradient_magnitude(self) -> None:
        elevation = np.array([[0, 10, 20], [0, 10, 20], [0, 10, 20]], dtype="float32")
        slope = compute_slope(elevation, x_resolution=10, y_resolution=10)
        np.testing.assert_allclose(slope, np.ones_like(elevation), atol=1e-6)

    def test_pareto_minimize_cost_vector(self) -> None:
        water = np.array([[1, 2, 1]], dtype="float32")
        grid = np.array([[1, 2, 3]], dtype="float32")
        land = np.array([[1, 2, 0]], dtype="float32")
        costs = cost_matrix_from_channels(water, grid, land)
        self.assertEqual(pareto_mask_minimize(costs).tolist(), [True, False, True])


if __name__ == "__main__":
    unittest.main()

