"""Real-data smoke check for heterodyne (two_component) NLSQ via xpcsjax.

Task 30 (xpcsjax NLSQ merge plan). Loads the cached C044 c2 fixture, picks the
first phi angle that the source heterodyne CLI actually fit (the angle nearest
-5.79° in ``phi_list.txt``), routes through the new heterodyne path in
:func:`xpcsjax.optimization.nlsq.adapter.get_or_create_model`, runs an NLSQ
``curve_fit`` end-to-end, and compares against the source heterodyne baseline
extracted to ``tests/heterodyne/fixtures/baselines/two_component_c044.json``.

Gating
------
This is a slow real-data fit (multi-second per angle on CPU). The test is
gated by the same env var as the Phase-5 homodyne characterization suite:

    XPCSJAX_RUN_CHARACTERIZATION=1 uv run pytest \
        tests/heterodyne/test_two_component_real_data.py -v

Tolerance notes
---------------
The xpcsjax baseline is the source heterodyne CLI's multi-angle joint fit
(3 angles), while this test runs a single-angle fit. The baseline parameter
values are therefore only weakly comparable. The smoke test asserts:

1. The fit converges (≤ max_nfev, finite chi²).
2. All 14 fitted physics params stay within their configured bounds.
3. Reduced χ² is finite and not catastrophically large
   (< 100 — the source baseline reduced χ² is ~0.86).

It does not assert per-parameter agreement with the baseline (which would
require running the same multi-angle joint fit).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

C044_DATA_DIR = Path("/home/wei/Documents/Projects/data/C044")
C044_CONFIG = C044_DATA_DIR / "heterodyne_config.yaml"
C044_C2_CACHE = C044_DATA_DIR / "cached_c2_q0.0054_frames_1000_2000.npz"
C044_PHI_LIST = C044_DATA_DIR / "phi_list.txt"
BASELINE_JSON = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "baselines"
    / "two_component_c044.json"
)


_SLOW_GATE = pytest.mark.skipif(
    os.environ.get("XPCSJAX_RUN_CHARACTERIZATION") != "1",
    reason="Slow real-data fit; set XPCSJAX_RUN_CHARACTERIZATION=1 to enable.",
)


def _require_fixture(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Missing C044 fixture: {path}")


@pytest.fixture(scope="module")
def c044_payload() -> dict[str, object]:
    """Load the C044 c2 cache and slice the first source-fitted phi angle."""
    _require_fixture(C044_C2_CACHE)
    _require_fixture(C044_CONFIG)
    _require_fixture(C044_PHI_LIST)

    cache = np.load(C044_C2_CACHE)
    c2_all = np.asarray(cache["c2"])  # (n_phi, 1001, 1001)
    phi_all = np.asarray(cache["phi"])  # degrees
    t_frames = np.asarray(cache["t"])  # frame indices (999..1999)
    q_vals = np.asarray(cache["q_values"])

    # The source heterodyne CLI filters phi via the config's phi_filtering
    # block (range ±10° around 0° and 85-95° around 90°) and fits 3 angles.
    # Pick the first one (-5.79°) so the smoke fit covers an angle the
    # baseline also fit.
    target_phi = -5.793084144592285  # from nlsq_metadata.json
    phi_idx = int(np.argmin(np.abs(phi_all - target_phi)))

    # Match the source heterodyne pipeline: drop the leading time point.
    c2 = c2_all[phi_idx, 1:, 1:]  # (1000, 1000)
    n_t = c2.shape[0]
    t = np.arange(n_t, dtype=np.float64)  # frame indices, dt scales to seconds

    # q: heterodyne stores per-angle q; use the value at the selected angle,
    # falling back to the config q if zero/missing.
    q = float(q_vals[phi_idx]) if q_vals[phi_idx] > 0 else 0.0054

    # dt from the config
    import yaml

    with C044_CONFIG.open() as f:
        cfg = yaml.safe_load(f)
    dt = float(cfg["analyzer_parameters"]["dt"])

    return {
        "c2": c2,
        "t": t,
        "dt": dt,
        "q": q,
        "phi_angle": float(phi_all[phi_idx]),
        "phi_idx_in_cache": phi_idx,
        "config": cfg,
    }


@pytest.fixture(scope="module")
def baseline() -> dict[str, object]:
    if not BASELINE_JSON.exists():
        pytest.skip(f"Baseline JSON missing: {BASELINE_JSON}")
    return json.loads(BASELINE_JSON.read_text())


def _initial_params_and_bounds(cfg: dict, baseline_params: list[float] | None):
    """Build [contrast, offset, *physics_14] initial guess + bounds from config.

    Initial physics params come from the config ``parameters`` section (the
    C044 config ships warm-start values from the source CLI run). Bounds come
    from ``parameter_space.bounds``.
    """
    params_cfg = cfg["parameters"]
    bounds_list = cfg["parameter_space"]["bounds"]
    bounds_by_name = {b["name"]: b for b in bounds_list}

    # Physics order matches xpcsjax registry ordering
    physics_order = [
        ("reference", "D0_ref"),
        ("reference", "alpha_ref"),
        ("reference", "D_offset_ref"),
        ("sample", "D0_sample"),
        ("sample", "alpha_sample"),
        ("sample", "D_offset_sample"),
        ("velocity", "v0"),
        ("velocity", "beta"),
        ("velocity", "v_offset"),
        ("fraction", "f0"),
        ("fraction", "f1"),
        ("fraction", "f2"),
        ("fraction", "f3"),
        ("angle", "phi0"),
    ]

    physics_init = []
    physics_lo = []
    physics_hi = []
    for group, name in physics_order:
        block = params_cfg[group][name]
        physics_init.append(float(block["value"]))
        # bounds prefer parameter_space (used by source CLI); fall back to per-param
        b = bounds_by_name.get(name, block)
        physics_lo.append(float(b["min"]))
        physics_hi.append(float(b["max"]))

    # Scaling: leave free between configured min/max
    contrast_b = bounds_by_name["contrast"]
    offset_b = bounds_by_name["offset"]

    p0 = np.array(
        [
            float(params_cfg["scaling"]["contrast"]["value"]),
            float(params_cfg["scaling"]["offset"]["value"]),
            *physics_init,
        ],
        dtype=np.float64,
    )
    lower = np.array(
        [float(contrast_b["min"]), float(offset_b["min"]), *physics_lo],
        dtype=np.float64,
    )
    upper = np.array(
        [float(contrast_b["max"]), float(offset_b["max"]), *physics_hi],
        dtype=np.float64,
    )

    # Clamp init into bounds to avoid bound-violation errors from curve_fit.
    p0 = np.clip(p0, lower + 1e-12, upper - 1e-12)

    return p0, (lower, upper)


@_SLOW_GATE
def test_two_component_smoke_via_adapter(c044_payload, baseline):
    """End-to-end: HeterodyneModel through xpcsjax NLSQ adapter, single angle.

    Drives :func:`xpcsjax.optimization.nlsq.adapter._get_or_create_heterodyne_model`
    + ``nlsq.curve_fit`` with the C044 real data at phi ≈ -5.79°.
    """
    pytest.importorskip("nlsq")
    from nlsq import curve_fit

    from xpcsjax.optimization.nlsq.adapter import get_or_create_model

    c2 = np.asarray(c044_payload["c2"], dtype=np.float64)
    t = np.asarray(c044_payload["t"], dtype=np.float64)
    dt = float(c044_payload["dt"])
    q = float(c044_payload["q"])
    phi_angle = float(c044_payload["phi_angle"])

    # 1. Route through the adapter — this exercises the new two_component branch.
    model, model_func, cache_hit = get_or_create_model(
        analysis_mode="two_component",
        phi_angles=np.array([phi_angle], dtype=np.float64),
        q=q,
        per_angle_scaling=True,
        config=None,
        enable_jit=True,
        t=t,
        dt=dt,
    )
    assert model.__class__.__name__ == "HeterodyneModel"
    assert cache_hit is False  # heterodyne route is uncached
    assert callable(model_func)

    # 2. Build initial guess + bounds from the C044 config.
    p0, (lower, upper) = _initial_params_and_bounds(
        c044_payload["config"], baseline_params=None
    )
    assert p0.shape == (16,)  # contrast + offset + 14 physics
    assert lower.shape == (16,) and upper.shape == (16,)
    assert np.all(lower <= p0) and np.all(p0 <= upper)

    # 3. Run NLSQ curve_fit. ydata is the FLATTENED c2; xdata is a dummy index
    # array (the closure evaluates on the stored ``t``).
    ydata = c2.ravel()
    xdata = np.arange(ydata.size, dtype=np.float64)

    popt, pcov = curve_fit(
        f=model_func,
        xdata=xdata,
        ydata=ydata,
        p0=p0,
        bounds=(lower, upper),
        method="trf",
        loss="soft_l1",
        ftol=1e-8,
        gtol=1e-8,
        xtol=1e-8,
        max_nfev=200,  # smoke check — cap iterations
    )

    # 4. Smoke assertions
    assert popt.shape == (16,)
    assert np.all(np.isfinite(popt))
    assert np.all(popt >= lower - 1e-9), f"popt below lower: {popt - lower}"
    assert np.all(popt <= upper + 1e-9), f"popt above upper: {upper - popt}"

    # Residuals and chi²
    y_pred = model_func(xdata, *popt)
    residuals = y_pred - ydata
    chi2 = float(np.sum(residuals**2))
    n_data = ydata.size
    n_params = popt.size
    dof = max(1, n_data - n_params)
    red_chi2 = chi2 / dof
    assert np.isfinite(chi2)
    assert np.isfinite(red_chi2)
    # The source baseline's per-angle reduced χ² for this angle is ~0.74.
    # The single-angle smoke fit may not reach the same optimum but should
    # not be catastrophically off.
    assert red_chi2 < 100.0, (
        f"Reduced chi² = {red_chi2:.3g} is suspiciously large; "
        f"smoke threshold = 100"
    )

    # Print a small report for the human reader.
    print(
        f"\n[two_component smoke] phi={phi_angle:.4f}°, q={q:.5g}, "
        f"n_data={n_data}, n_params={n_params}, "
        f"chi²={chi2:.4g}, reduced_chi²={red_chi2:.4g}"
    )

    # 5. Loose comparison against the source heterodyne baseline (which is a
    # joint 3-angle fit, so we only check that the single-angle fit is in the
    # same ballpark, not bit-identical).
    baseline_params = np.asarray(baseline["parameters"], dtype=np.float64)
    fit_physics = popt[2:]  # drop (contrast, offset)
    assert baseline_params.shape == fit_physics.shape

    # Per-parameter relative agreement: report only, no assertion. The smoke
    # threshold below is a single aggregate sanity check.
    rel_agree = np.abs(fit_physics - baseline_params) / (
        np.abs(baseline_params) + 1.0
    )
    print(f"[two_component smoke] mean |Δp|/(|p|+1) vs CLI baseline: "
          f"{float(np.mean(rel_agree)):.3g}")
    # Sanity: parameter vector should not be NaN/inf
    assert np.all(np.isfinite(fit_physics))


def test_two_component_adapter_rejects_multi_angle():
    """Heterodyne adapter is single-angle only — assert it rejects multi-angle."""
    from xpcsjax.optimization.nlsq.adapter import _get_or_create_heterodyne_model

    with pytest.raises(ValueError, match="single-angle"):
        _get_or_create_heterodyne_model(
            phi_angles=np.array([0.0, 90.0]),
            q=0.005,
            t=np.arange(10.0),
            dt=0.1,
        )


def test_two_component_adapter_requires_t_and_dt():
    """Heterodyne routing requires t/dt — assert get_or_create_model raises."""
    from xpcsjax.optimization.nlsq.adapter import get_or_create_model

    with pytest.raises(ValueError, match="requires"):
        get_or_create_model(
            analysis_mode="two_component",
            phi_angles=np.array([0.0]),
            q=0.005,
            t=None,
            dt=None,
        )
