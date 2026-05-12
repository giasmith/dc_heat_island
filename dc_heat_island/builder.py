"""Build carrying-capacity state tensors from geospatial data.

The implementation follows the Section 3 methodology:

* every layer is aligned to the master land-cover grid;
* W(p) is sampled from WRI baseline water stress and clipped to [0, 5];
* G(c, r) is a binary rasterized grid-infrastructure matrix;
* D(p) is the exact Euclidean distance to the nearest grid pixel;
* L(p) = alpha * |grad E(X, Y)| + Omega(v(X, Y));
* channels are stacked into a deterministic 3D state tensor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import rioxarray
import torch
import xarray as xr
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

from dc_heat_island.config import LandCostConfig, ProjectConfig, init_directories
from dc_heat_island.pareto import cost_matrix_from_channels, pareto_mask_minimize


class TensorBuilder:
    """Build tiled PyTorch tensors from configured geospatial inputs."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.tile_size = config.spatial.TILE_SIZE
        init_directories(config)

        self.master = self._open_raster(config.paths.LAND_COVER, name="land_cover", resampling=Resampling.nearest)
        self.master_crs = self.master.rio.crs
        self.master_transform = self.master.rio.transform()
        self.master_height = int(self.master.rio.height)
        self.master_width = int(self.master.rio.width)
        self.x_resolution = abs(float(self.master_transform.a))
        self.y_resolution = abs(float(self.master_transform.e))

    def build_state_dataset(self) -> xr.Dataset:
        """Construct the co-registered state tensor dataset."""

        land_cover = self.master.astype("float32")
        elevation = self._open_raster(self.config.paths.DEM, name="elevation", resampling=Resampling.bilinear)
        water_stress = self._load_water_stress()
        grid_binary = self._rasterize_vector(
            self.config.paths.GRID_INFRASTRUCTURE,
            layer=self.config.vectors.GRID_LAYER,
            burn_value=1.0,
            dtype="uint8",
        )
        grid_distance = compute_grid_distance(grid_binary, self.x_resolution, self.y_resolution)
        slope = compute_slope(elevation.values, self.x_resolution, self.y_resolution)
        land_cover_penalty = map_land_cover_penalty(land_cover.values, self.config.land_cost)
        land_cost = compute_land_cost(slope, land_cover_penalty, self.config.land_cost.ALPHA)

        dataset = xr.Dataset(
            data_vars={
                "water_stress": self._array_to_dataarray(np.clip(water_stress, 0, 5), "water_stress"),
                "grid_distance": self._array_to_dataarray(grid_distance, "grid_distance"),
                "land_cost": self._array_to_dataarray(land_cost, "land_cost"),
                "slope": self._array_to_dataarray(slope, "slope"),
                "land_cover_penalty": self._array_to_dataarray(land_cover_penalty, "land_cover_penalty"),
                "elevation": elevation.astype("float32"),
                "land_cover": land_cover,
                "grid_binary": self._array_to_dataarray(grid_binary.astype("float32"), "grid_binary"),
            },
            attrs={
                "crs": str(self.master_crs),
                "transform": tuple(self.master_transform),
                "channel_order": list(self.config.TENSOR_CHANNELS),
                "x_resolution_m": self.x_resolution,
                "y_resolution_m": self.y_resolution,
            },
        )
        return dataset

    def generate_tiles(self, dataset: xr.Dataset | None = None) -> dict[str, object]:
        """Write fixed-size tiled tensors to ``config.paths.OUT_DIR``."""

        state = dataset or self.build_state_dataset()
        channel_order = tuple(self.config.TENSOR_CHANNELS)
        missing = [channel for channel in channel_order if channel not in state]
        if missing:
            raise KeyError(f"Dataset is missing configured channels: {missing}")

        output_dir = self.config.paths.OUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        tile_records = []
        y_size = int(state.sizes["y"])
        x_size = int(state.sizes["x"])

        for y0 in range(0, y_size, self.tile_size):
            for x0 in range(0, x_size, self.tile_size):
                y1 = min(y0 + self.tile_size, y_size)
                x1 = min(x0 + self.tile_size, x_size)
                if self.config.spatial.DROP_INCOMPLETE_TILES and (y1 - y0 != self.tile_size or x1 - x0 != self.tile_size):
                    continue
                tile = state.isel(y=slice(y0, y1), x=slice(x0, x1))
                tensor = self._tile_to_tensor(tile, channel_order)
                path = output_dir / f"tile_y{y0:06d}_x{x0:06d}.pt"
                torch.save(
                    {
                        "tensor": tensor,
                        "channels": channel_order,
                        "row_start": y0,
                        "col_start": x0,
                        "crs": str(self.master_crs),
                        "transform": tuple(self.master_transform),
                    },
                    path,
                )
                tile_records.append({"path": str(path), "row_start": y0, "col_start": x0})

        summary = {
            "tile_count": len(tile_records),
            "tile_size": self.tile_size,
            "channels": list(channel_order),
            "output_dir": str(output_dir),
            "tiles": tile_records,
        }
        summary_path = self.config.paths.DIAGNOSTIC_DIR / "tensor_build_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def pareto_mask(self, dataset: xr.Dataset | None = None) -> np.ndarray:
        """Return a 2D Pareto frontier mask for C(p) = [W(p), D(p), L(p)]."""

        state = dataset or self.build_state_dataset()
        costs = cost_matrix_from_channels(
            state["water_stress"].values,
            state["grid_distance"].values,
            state["land_cost"].values,
        )
        return pareto_mask_minimize(costs).reshape(state.sizes["y"], state.sizes["x"])

    def _open_raster(self, path: Path, name: str, resampling: Resampling) -> xr.DataArray:
        raster = rioxarray.open_rasterio(path, masked=True)
        if "band" in raster.dims:
            raster = raster.isel(band=0, drop=True)
        raster = raster.rename(name).astype("float32")
        if name != "land_cover":
            raster = raster.rio.reproject_match(self.master, resampling=resampling)
        return raster

    def _load_water_stress(self) -> np.ndarray:
        path = self.config.paths.WATER_STRESS
        try:
            return self._open_raster(path, name="water_stress", resampling=Resampling.bilinear).values.astype("float32")
        except (RasterioIOError, ValueError):
            return self._rasterize_vector(
                path,
                layer=self.config.vectors.WATER_STRESS_LAYER,
                value_column=self.config.vectors.WATER_STRESS_VALUE_COLUMN,
                fill=np.nan,
                dtype="float32",
            )

    def _rasterize_vector(
        self,
        path: Path,
        layer: str | None = None,
        value_column: str | None = None,
        burn_value: float = 1.0,
        fill: float = 0.0,
        dtype: str = "float32",
    ) -> np.ndarray:
        read_kwargs = {"layer": layer} if layer else {}
        frame = gpd.read_file(path, **read_kwargs)
        if frame.empty:
            return np.full((self.master_height, self.master_width), fill, dtype=dtype)
        frame = frame.to_crs(self.master_crs)
        if value_column:
            if value_column not in frame.columns:
                raise KeyError(f"{path} does not contain vector value column {value_column!r}")
            shapes: Iterable[tuple[object, float]] = zip(frame.geometry, frame[value_column].astype("float32"))
        else:
            shapes = ((geometry, burn_value) for geometry in frame.geometry)
        return rasterize(
            shapes=shapes,
            out_shape=(self.master_height, self.master_width),
            transform=self.master_transform,
            fill=fill,
            all_touched=True,
            dtype=dtype,
        )

    def _array_to_dataarray(self, values: np.ndarray, name: str) -> xr.DataArray:
        return xr.DataArray(
            values.astype("float32"),
            dims=("y", "x"),
            coords={"y": self.master.y, "x": self.master.x},
            name=name,
        ).rio.write_crs(self.master_crs)

    def _tile_to_tensor(self, tile: xr.Dataset, channel_order: tuple[str, ...]) -> torch.Tensor:
        arrays = []
        for channel in channel_order:
            values = tile[channel].values.astype("float32")
            values = np.nan_to_num(
                values,
                nan=self.config.spatial.NODATA_VALUE,
                posinf=self.config.land_cost.HIGH_PENALTY,
                neginf=self.config.spatial.NODATA_VALUE,
            )
            arrays.append(values)
        return torch.from_numpy(np.stack(arrays).astype("float32"))


def compute_grid_distance(grid_binary: np.ndarray, x_resolution: float, y_resolution: float) -> np.ndarray:
    """Exact Euclidean distance D(p) to the nearest grid infrastructure pixel."""

    grid = np.asarray(grid_binary)
    if not np.any(grid > 0):
        raise ValueError("grid infrastructure raster contains no positive pixels")
    return distance_transform_edt(grid <= 0, sampling=(y_resolution, x_resolution)).astype("float32")


def compute_slope(elevation: np.ndarray, x_resolution: float, y_resolution: float) -> np.ndarray:
    """Topographic slope as the spatial gradient magnitude |grad E(X, Y)|."""

    values = np.asarray(elevation, dtype="float32")
    filled = np.where(np.isfinite(values), values, np.nanmedian(values[np.isfinite(values)]))
    grad_y, grad_x = np.gradient(filled, y_resolution, x_resolution)
    return np.sqrt(grad_x**2 + grad_y**2).astype("float32")


def map_land_cover_penalty(land_cover: np.ndarray, config: LandCostConfig) -> np.ndarray:
    """Map CLCD classes to Omega(v(X, Y))."""

    values = np.asarray(land_cover)
    penalty = np.full(values.shape, config.DEFAULT_PENALTY, dtype="float32")
    penalty[np.isin(values, config.ZERO_COST_CLASSES)] = 0.0
    penalty[np.isin(values, config.INTERMEDIATE_COST_CLASSES)] = config.INTERMEDIATE_PENALTY
    penalty[np.isin(values, config.HIGH_COST_CLASSES)] = config.HIGH_PENALTY
    penalty[~np.isfinite(values)] = np.nan
    return penalty


def compute_land_cost(slope: np.ndarray, land_cover_penalty: np.ndarray, alpha: float) -> np.ndarray:
    """Composite land cost L(p) = alpha * slope + Omega(v(X, Y))."""

    if slope.shape != land_cover_penalty.shape:
        raise ValueError("slope and land-cover penalty arrays must share shape")
    return (alpha * slope.astype("float32") + land_cover_penalty.astype("float32")).astype("float32")

