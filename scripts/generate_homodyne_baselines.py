"""Generate homodyne fit baselines for the Phase 5 characterization gate.

Run this script INSIDE THE SOURCE HOMODYNE VENV:

    cd /home/wei/Documents/GitHub/homodyne
    uv run python /home/wei/Documents/GitHub/xpcsjax/scripts/generate_homodyne_baselines.py

It runs the canonical xpcsjax fixture configs through the SOURCE homodyne
package and pins the fit results as frozen JSON baselines under
tests/characterization/fixtures/baselines/. xpcsjax must then reproduce
these bit-for-bit (rtol=1e-10) — that's the Phase 5 gate.

The fixture YAMLs are at their original absolute paths on this machine
(see CONFIGS dict below). When Phase 5 hardens, we'll bundle small
fixtures under tests/characterization/fixtures/configs/ proper.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

XPCSJAX_ROOT = Path("/home/wei/Documents/GitHub/xpcsjax")
BASELINES_DIR = XPCSJAX_ROOT / "tests" / "characterization" / "fixtures" / "baselines"

CONFIGS: dict[str, Path] = {
    "static_simon": Path(
        "/home/wei/Documents/Projects/data/Simon/homodyne_static_config.yaml"
    ),
    "laminar_c020": Path(
        "/home/wei/Documents/Projects/data/C020/homodyne_laminar_flow_config.yaml"
    ),
}


def _json_safe(obj: Any) -> Any:
    """Recursive json.dumps coercion: numpy → list, paths → str, etc."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def run_one(label: str, config_path: Path) -> dict[str, Any]:
    """Run a single homodyne fit and return a JSON-serializable summary."""
    print(f"\n=== {label} : {config_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"config missing: {config_path}")

    # Import inside the function so any error gets attributed to a specific config
    from homodyne.config import ConfigManager
    from homodyne.data import load_xpcs_data
    from homodyne.optimization import fit_nlsq_jax

    cfg = ConfigManager(str(config_path))
    data = load_xpcs_data(str(config_path))
    result = fit_nlsq_jax(data, cfg)

    # OptimizationResult shape may vary — defensively pick attributes that exist
    out: dict[str, Any] = {
        "config_path": str(config_path),
        "label": label,
    }
    for attr in (
        "parameters",
        "parameter_errors",
        "chi_squared",
        "r_squared",
        "dof",
        "convergence_status",
        "n_iterations",
        "model_metadata",
    ):
        if hasattr(result, attr):
            out[attr] = _json_safe(getattr(result, attr))

    # Some result objects expose params as positional array instead of named dict
    if "parameters" not in out and hasattr(result, "params"):
        out["parameters_array"] = _json_safe(result.params)

    return out


def main() -> int:
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, str] = {}

    for label, cfg_path in CONFIGS.items():
        try:
            baseline = run_one(label, cfg_path)
        except Exception as e:  # noqa: BLE001
            print(f"   FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            summary[label] = f"FAILED ({type(e).__name__})"
            continue

        out_path = BASELINES_DIR / f"{label}.json"
        with out_path.open("w") as f:
            json.dump(baseline, f, indent=2, default=_json_safe)
        print(f"   → wrote {out_path}")
        summary[label] = "OK"

    print("\n=== summary ===")
    for label, status in summary.items():
        print(f"  {label}: {status}")
    return 0 if all(v == "OK" for v in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
