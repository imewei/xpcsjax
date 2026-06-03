"""Generate heterodyne (two_component) residual-parity baselines.

MAINTAINER-LOCAL: Run this INSIDE xpcsjax's venv with the upstream
``heterodyne`` package installed (it is NOT a declared dependency —
install editable first):

    uv pip install -e /path/to/heterodyne --no-deps
    uv run python scripts/generate_heterodyne_baselines.py

It pins upstream ``compute_multi_angle_residuals`` output on a small, fully
deterministic synthetic two_component input as a frozen JSON fixture under
tests/characterization/fixtures/baselines/. xpcsjax must reproduce it
element-for-element — that's the heterodyne parity gate (the homodyne gate
only covered static/laminar modes; heterodyne had no oracle, which is how the
t=0-boundary-mask divergence in compute_multi_angle_residuals went unnoticed).

Only the maintainer runs this, and only when upstream's residual convention
changes. The characterization test itself imports stdlib only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

XPCSJAX_ROOT = Path(__file__).resolve().parent.parent
BASELINES_DIR = XPCSJAX_ROOT / "tests" / "characterization" / "fixtures" / "baselines"

# Fully deterministic synthetic spec — small enough to commit, large enough
# that the t=0 boundary row/col and the diagonal are all exercised.
SPEC: dict[str, Any] = {
    "N": 8,
    "n_phi": 2,
    "q": 0.0054,
    "dt": 0.5,
    "t_max": 0.07,
    "seed": 1234,
    # canonical 14-param order: D0_ref, alpha_ref, D_offset_ref, D0_sample,
    # alpha_sample, D_offset_sample, v0, beta, v_offset, f0, f1, f2, f3, phi0
    "params": [5000.0, 0.5, 100.0, 5000.0, 0.5, 100.0, 50.0, 0.5, 5.0,
               0.5, 0.01, 1.0, 0.5, 2.0],
    "phi_angles": [10.0, 90.0],
    "contrasts": [1.0, 1.0],
    "offsets": [1.0, 1.0],
}


def build_inputs(spec: dict[str, Any]) -> dict[str, np.ndarray]:
    """Reconstruct the deterministic inputs from the spec (shared with the test)."""
    n, n_phi = spec["N"], spec["n_phi"]
    rng = np.random.default_rng(spec["seed"])
    return {
        "t": np.linspace(0.0, spec["t_max"], n, dtype=np.float64),
        "phi": np.asarray(spec["phi_angles"], dtype=np.float64),
        "params": np.asarray(spec["params"], dtype=np.float64),
        "c2": rng.uniform(1.0, 1.5, size=(n_phi, n, n)).astype(np.float64),
        "weights": np.ones((n_phi, n, n), dtype=np.float64),
        "contrasts": np.asarray(spec["contrasts"], dtype=np.float64),
        "offsets": np.asarray(spec["offsets"], dtype=np.float64),
    }


def main() -> int:
    import jax.numpy as jnp
    from heterodyne.core.jax_backend import compute_multi_angle_residuals

    inp = build_inputs(SPEC)
    resid = np.asarray(
        compute_multi_angle_residuals(
            jnp.asarray(inp["params"]),
            jnp.asarray(inp["t"]),
            SPEC["q"],
            SPEC["dt"],
            jnp.asarray(inp["phi"]),
            jnp.asarray(inp["c2"]),
            jnp.asarray(inp["weights"]),
            jnp.asarray(inp["contrasts"]),
            jnp.asarray(inp["offsets"]),
        )
    )

    fixture = {
        "label": "heterodyne_residuals_smoke",
        "source": "heterodyne.core.jax_backend.compute_multi_angle_residuals",
        "spec": SPEC,
        "upstream_residuals": resid.tolist(),
        "upstream_shape": list(resid.shape),
    }

    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BASELINES_DIR / "heterodyne_residuals.json"
    with out_path.open("w") as f:
        json.dump(fixture, f, indent=2)
    print(f"wrote {out_path}  (residual size={resid.size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
