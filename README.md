# Data-Center Heat-Island Carrying Capacity

This repository builds geospatial state tensors for the energy-water-land carrying-capacity methodology in the May 2026 deliverable. It converts heterogeneous geospatial layers into co-registered matrices, computes the methodology cost surfaces, and writes fixed-size PyTorch tensor tiles.

## Methodology Implemented

The builder follows Section 3, "Methodology":

1. **Matrix discretization**
   - Land cover is used as the master grid.
   - Every raster is reprojected to the master grid.
   - Vector infrastructure is rasterized onto the same transform.

2. **Water stress**
   - The WRI Aqueduct baseline annual water stress layer is sampled as `W(p)`.
   - Values are clipped to `0 <= W(p) <= 5`.

3. **Topological distance**
   - Grid infrastructure is rasterized as binary matrix `G(c, r)`.
   - Exact Euclidean distance transform produces `D(p)`, the metric distance to the nearest infrastructure pixel.

4. **Land cost**
   - DEM slope is computed as `|grad E(X, Y)|`.
   - Land-cover penalties map CLCD classes into `Omega(v(X, Y))`.
   - Composite cost is `L(p) = alpha * slope + Omega(v(X, Y))`.

5. **Pareto frontier**
   - The minimization cost vector is `C(p) = [W(p), D(p), L(p)]`.
   - Pareto frontier pixels are non-dominated across those three costs.

## Repository Layout

- `dc_heat_island/config.py` - portable config dataclasses and YAML override loader.
- `dc_heat_island/builder.py` - geospatial alignment, rasterization, cost surfaces, tensor writing.
- `dc_heat_island/pareto.py` - Pareto minimization helpers.
- `scripts/build_tensors.py` - command-line entry point.
- `config.example.yaml` - example path and penalty configuration.
- `tests/` - equation-level unit tests.

## Setup

```bash
python -m pip install -r requirements.txt
```

Copy `config.example.yaml` and edit paths for your machine:

```bash
cp config.example.yaml config.local.yaml
```

The most important inputs are:

- `LAND_COVER`: CLCD raster used as the master grid.
- `DEM`: elevation raster for slope.
- `WATER_STRESS`: WRI Aqueduct water stress raster or vector source.
- `GRID_INFRASTRUCTURE`: transmission line/substation vector layer.
- `OUT_DIR`: tensor output directory.
- `DIAGNOSTIC_DIR`: JSON diagnostics output directory.

You can also override paths with environment variables such as `DHI_LAND_COVER`, `DHI_DEM`, `DHI_WATER_STRESS`, `DHI_GRID_INFRASTRUCTURE`, and `DHI_OUTPUT_DIR`.

## Build Tensors

```bash
python scripts/build_tensors.py --config config.local.yaml
```

For a lighter validation run that builds cost surfaces and Pareto diagnostics without writing tensor tiles:

```bash
python scripts/build_tensors.py --config config.local.yaml --diagnostics-only
```

Tensor files are saved as `.pt` dictionaries with:

- `tensor`: shape `(channels, tile_height, tile_width)`
- `channels`: deterministic channel names
- `row_start` and `col_start`: source-grid tile origin
- `crs` and `transform`: geospatial metadata

## Tests

```bash
python -m unittest discover tests
```
