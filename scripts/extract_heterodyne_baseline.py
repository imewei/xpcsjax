"""Extract heterodyne C044 baseline from existing CLI outputs.

The source heterodyne CLI was already run against
``/home/wei/Documents/Projects/data/C044/heterodyne_config.yaml`` and produced
``nlsq_parameters.json`` + ``nlsq_metadata.json`` under
``heterodyne_results/output/``. This script repackages those outputs into the
xpcsjax baseline JSON format (mirroring
``tests/characterization/fixtures/baselines/*.json``).

Usage::

    uv run python scripts/extract_heterodyne_baseline.py

The result is written to
``tests/heterodyne/fixtures/baselines/two_component_c044.json``.

This avoids re-running the multi-minute CLI fit (160 s, 3 angles) when the
result is already on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

XPCSJAX_ROOT = Path("/home/wei/Documents/GitHub/xpcsjax")
BASELINES_DIR = XPCSJAX_ROOT / "tests" / "heterodyne" / "fixtures" / "baselines"

CONFIG_PATH = Path("/home/wei/Documents/Projects/data/C044/heterodyne_config.yaml")
HET_OUTPUT_DIR = (
    Path("/home/wei/Documents/Projects/data/C044") / "heterodyne_results" / "output"
)

LABEL = "two_component_c044"


def main() -> int:
    params_path = HET_OUTPUT_DIR / "nlsq_parameters.json"
    meta_path = HET_OUTPUT_DIR / "nlsq_metadata.json"
    if not params_path.exists() or not meta_path.exists():
        raise SystemExit(
            f"Source heterodyne CLI outputs missing: {params_path}, {meta_path}"
        )

    params = json.loads(params_path.read_text())
    meta = json.loads(meta_path.read_text())

    # final_cost is the trust-region "cost" (0.5 * Σ r²); convert to χ² = 2 * cost.
    final_cost = float(meta.get("final_cost", 0.0))
    chi_squared = 2.0 * final_cost

    success = bool(meta.get("success", False))
    convergence_status = "converged" if success else "failed"

    baseline = {
        "config_path": str(CONFIG_PATH),
        "label": LABEL,
        "analysis_mode": "two_component",
        "parameter_names": params["parameter_names"],
        "parameters": params["parameters"],
        "uncertainties": params["uncertainties"],
        "chi_squared": chi_squared,
        "reduced_chi_squared": float(meta.get("reduced_chi_squared", float("nan"))),
        "convergence_status": convergence_status,
        "n_iterations": int(meta.get("n_iterations", 0)),
        "n_function_evals": int(meta.get("n_function_evals", 0)),
        "wall_time_seconds": float(meta.get("wall_time_seconds", float("nan"))),
        "convergence_reason": meta.get("convergence_reason", ""),
        "metadata": meta.get("metadata", {}),
        "source": {
            "generator": "heterodyne CLI (heterodyne --config <yaml> --method nlsq)",
            "params_path": str(params_path),
            "metadata_path": str(meta_path),
        },
    }

    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BASELINES_DIR / f"{LABEL}.json"
    out_path.write_text(json.dumps(baseline, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
