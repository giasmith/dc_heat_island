"""Build data-center heat-island carrying-capacity tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dc_heat_island.builder import TensorBuilder
from dc_heat_island.config import load_project_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config override file.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override tensor output directory.")
    parser.add_argument("--diagnostics-only", action="store_true", help="Build state surfaces and Pareto mask without writing tiles.")
    args = parser.parse_args()

    config = load_project_config(args.config)
    if args.output_dir:
        config.paths.OUT_DIR = args.output_dir

    builder = TensorBuilder(config)
    dataset = builder.build_state_dataset()
    pareto_mask = builder.pareto_mask(dataset)
    diagnostics = {
        "shape": {"y": int(dataset.sizes["y"]), "x": int(dataset.sizes["x"])},
        "channels": list(config.TENSOR_CHANNELS),
        "pareto_frontier_pixels": int(pareto_mask.sum()),
    }

    config.paths.DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    diagnostic_path = config.paths.DIAGNOSTIC_DIR / "state_dataset_diagnostics.json"
    diagnostic_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    if args.diagnostics_only:
        print(json.dumps(diagnostics, indent=2))
        return

    summary = builder.generate_tiles(dataset)
    print(json.dumps({**diagnostics, **summary}, indent=2))


if __name__ == "__main__":
    main()

