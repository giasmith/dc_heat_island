"""Configuration for building energy-water-land carrying-capacity tensors."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: Path | str) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


@dataclass
class PathsConfig:
    """Paths for every geospatial layer used by the methodology."""

    BASE_DATA: Path = field(default_factory=lambda: _path_from_env("DHI_DATA_ROOT", REPO_ROOT / "data"))
    CLOUD_DATA: Path = field(default_factory=lambda: _path_from_env("DHI_CLOUD_DATA_ROOT", REPO_ROOT / "data" / "cloud"))
    LAND_COVER: Path = field(
        default_factory=lambda: _path_from_env("DHI_LAND_COVER", REPO_ROOT / "data" / "cloud" / "CLCD_v01_2025_albert.tif")
    )
    DEM: Path = field(
        default_factory=lambda: _path_from_env("DHI_DEM", REPO_ROOT / "data" / "cloud" / "CHINA_DEM_SRTMGL1.tif")
    )
    WATER_STRESS: Path = field(
        default_factory=lambda: _path_from_env("DHI_WATER_STRESS", REPO_ROOT / "data" / "cloud" / "Aq40_Y2023D07M05.gdb")
    )
    GRID_INFRASTRUCTURE: Path = field(
        default_factory=lambda: _path_from_env("DHI_GRID_INFRASTRUCTURE", REPO_ROOT / "data" / "clean" / "resources_filtered.gpkg")
    )
    ENERGY_PLANTS: Path | None = field(
        default_factory=lambda: _optional_path_from_env("DHI_ENERGY_PLANTS", REPO_ROOT / "data" / "clean" / "ENERGY_CHINA_CLEAN" / "china_energy_all_merged.csv")
    )
    OUT_DIR: Path = field(default_factory=lambda: _path_from_env("DHI_OUTPUT_DIR", REPO_ROOT / "outputs" / "tensors"))
    DIAGNOSTIC_DIR: Path = field(default_factory=lambda: _path_from_env("DHI_DIAGNOSTIC_DIR", REPO_ROOT / "outputs" / "diagnostics"))


def _optional_path_from_env(name: str, default: Path | str) -> Path | None:
    value = os.getenv(name, str(default))
    if value.strip().lower() in {"", "none", "null"}:
        return None
    return Path(value).expanduser()


@dataclass
class VectorLayerConfig:
    """Optional vector-layer names and value columns."""

    WATER_STRESS_LAYER: str | None = os.getenv("DHI_WATER_STRESS_LAYER")
    WATER_STRESS_VALUE_COLUMN: str = os.getenv("DHI_WATER_STRESS_VALUE_COLUMN", "bws_score")
    GRID_LAYER: str | None = os.getenv("DHI_GRID_LAYER")


@dataclass
class SpatialConfig:
    """Grid and tile settings for tensor generation."""

    TILE_SIZE: int = int(os.getenv("DHI_TILE_SIZE", "512"))
    TARGET_RES_METERS: float = float(os.getenv("DHI_TARGET_RES_METERS", "300"))
    DROP_INCOMPLETE_TILES: bool = os.getenv("DHI_DROP_INCOMPLETE_TILES", "1") not in {"0", "false", "False"}
    NODATA_VALUE: float = float(os.getenv("DHI_NODATA_VALUE", "-9999"))


@dataclass
class LandCostConfig:
    """Penalty mapping for the land-cost equation L(p) = alpha * slope + Omega(v)."""

    ALPHA: float = float(os.getenv("DHI_LAND_COST_ALPHA", "1.0"))
    ZERO_COST_CLASSES: tuple[int, ...] = (4, 7)  # grassland, barren
    INTERMEDIATE_COST_CLASSES: tuple[int, ...] = (2, 8)  # forest, impervious
    HIGH_COST_CLASSES: tuple[int, ...] = (1, 5)  # cropland, water
    INTERMEDIATE_PENALTY: float = float(os.getenv("DHI_INTERMEDIATE_LAND_PENALTY", "1.0"))
    HIGH_PENALTY: float = float(os.getenv("DHI_HIGH_LAND_PENALTY", "1000000"))
    DEFAULT_PENALTY: float = float(os.getenv("DHI_DEFAULT_LAND_PENALTY", "5.0"))


@dataclass
class ProjectConfig:
    """Top-level configuration passed into ``TensorBuilder``."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    vectors: VectorLayerConfig = field(default_factory=VectorLayerConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    land_cost: LandCostConfig = field(default_factory=LandCostConfig)
    TENSOR_CHANNELS: tuple[str, ...] = (
        "water_stress",
        "grid_distance",
        "land_cost",
        "slope",
        "land_cover_penalty",
        "elevation",
        "land_cover",
        "grid_binary",
    )


def _coerce_value(target: Any, key: str, current: Any, value: Any) -> Any:
    if isinstance(target, PathsConfig) and value is not None:
        return Path(value).expanduser() if value is not None else None
    if isinstance(current, tuple):
        return tuple(value)
    return value


def _apply_overrides(target: Any, overrides: dict[str, Any]) -> None:
    if not is_dataclass(target):
        return
    known = {item.name for item in fields(target)}
    for key, value in overrides.items():
        if key not in known:
            raise KeyError(f"Unknown config key: {target.__class__.__name__}.{key}")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_overrides(current, value)
        else:
            setattr(target, key, _coerce_value(target, key, current, value))


def load_project_config(path: Path | str | None = None) -> ProjectConfig:
    """Load default config, optionally overridden by a YAML file."""

    project_config = ProjectConfig()
    if path is None:
        return project_config

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load a YAML config file.") from exc

    config_path = Path(path).expanduser()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    _apply_overrides(project_config, data)
    return project_config


def init_directories(project_config: ProjectConfig | None = None) -> None:
    """Create output directories used by the builder."""

    active = project_config or config
    active.paths.OUT_DIR.mkdir(parents=True, exist_ok=True)
    active.paths.DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)


config = ProjectConfig()
