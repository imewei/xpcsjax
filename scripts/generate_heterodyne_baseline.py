"""Generate heterodyne fit baseline for Phase 6/7 validation.

Run INSIDE THE SOURCE HETERODYNE VENV:

    cd /home/wei/Documents/GitHub/heterodyne
    uv run python /home/wei/Documents/GitHub/xpcsjax/scripts/generate_heterodyne_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

XPCSJAX_ROOT = Path("/home/wei/Documents/GitHub/xpcsjax")
BASELINES_DIR = XPCSJAX_ROOT / "tests" / "heterodyne" / "fixtures" / "baselines"

CONFIGS: dict[str, Path] = {
    "two_component_c044": Path(
        "/home/wei/Documents/Projects/data/C044/heterodyne_config.yaml"
    ),
}


def _json_safe(obj: Any) -> Any:
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
    print(f"\n=== {label} : {config_path}", flush=True)
    if not config_path.exists():
        raise FileNotFoundError(f"config missing: {config_path}")

    from heterodyne.config import ConfigManager
    from heterodyne.data import load_xpcs_data
    from heterodyne.optimization import fit_nlsq_jax

    cfg = ConfigManager(str(config_path))
    data = load_xpcs_data(str(config_path))
    result = fit_nlsq_jax(data, cfg)

    out: dict[str, Any] = {"config_path": str(config_path), "label": label}
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
            print(f"   FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            summary[label] = f"FAILED ({type(e).__name__})"
            continue

        out_path = BASELINES_DIR / f"{label}.json"
        with out_path.open("w") as f:
            json.dump(baseline, f, indent=2, default=_json_safe)
        print(f"   → wrote {out_path}", flush=True)
        summary[label] = "OK"

    print("\n=== summary ===", flush=True)
    for label, status in summary.items():
        print(f"  {label}: {status}", flush=True)
    return 0 if all(v == "OK" for v in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
