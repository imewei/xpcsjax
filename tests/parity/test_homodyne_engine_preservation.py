"""Phase-1.0 homodyne engine preservation suite (executor-runnable golden net).

WHY THIS EXISTS
---------------
Phase 1 will thread a pluggable ``PointEvaluator`` through the homodyne
stratification engine class
``xpcsjax.optimization.nlsq.strategies.residual_jit.StratifiedResidualFunctionJIT``,
replacing its hard-coded ``compute_g2_scaled(...)`` calls with
``evaluator.eval_points(...)``. That refactor MUST be behavior-preserving for the
homodyne (``laminar_flow``) path. The maintainer-local
``XPCSJAX_RUN_CHARACTERIZATION=1`` live oracle CANNOT run on a fresh checkout
(it needs the upstream ``homodyne`` package + ``/home/wei`` datasets), so this
file is the executor-runnable safety net: golden snapshots of the CURRENT
homodyne engine behavior that the Phase-1 refactor must keep bit-identical.

Everything here is synthetic, in-repo, fixed-seed, and small. No upstream
package, no external datasets. Runs on a fresh checkout with only ``make dev``.

TWO LAYERS
----------
1. UNIT golden (the critical one): constructs ``StratifiedResidualFunctionJIT``
   EXACTLY as the production ``laminar_flow`` stratified-LS path does
   (``strategies/stratified_ls.py``: build a flat angle-stratified
   ``stratified_data`` -> ``create_stratified_chunks(...)`` ->
   ``StratifiedResidualFunctionJIT(stratified_data=chunked, per_angle_scaling=...,
   physical_param_names=...)``), evaluates the residual at a FIXED parameter
   vector, and snapshots the residual VECTOR. This directly guards the class
   Phase 1 modifies, without needing a 1M-point fit.

2. END-TO-END golden: a small synthetic ``laminar_flow`` fit driven through the
   public ``fit_nlsq`` path with per-angle scaling on. Snapshots parameters,
   objective, uncertainties (shape + finite pattern), the ``nlsq_diagnostics``
   key set, anti-degeneracy activation flags, the selected strategy/tier, and
   the covariance shape + NaN/finite pattern.

GOLDEN MECHANISM
----------------
Goldens live in ``tests/parity/_golden/`` and are committed. On a normal run the
goldens MUST already exist and the test ASSERTS against them (so a behavior
change fails). Set ``XPCSJAX_REGEN_GOLDEN=1`` to (re)write them, or they are
written automatically on the first run if absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tests.parity._golden_util import load_or_init_golden
from xpcsjax.optimization.nlsq.strategies.residual_jit import (
    StratifiedResidualFunctionJIT,
)
from xpcsjax.optimization.nlsq.strategies.stratified_ls import (
    create_stratified_chunks,
)

_GOLDEN_DIR = Path(__file__).parent / "_golden"
_REGEN = os.environ.get("XPCSJAX_REGEN_GOLDEN") == "1"

# Laminar-flow physical parameter names (homodyne 7-param model), matching the
# production ``parameter_names`` used by the live ``laminar_flow`` fit fixtures.
_PHYS_NAMES = [
    "D0",
    "alpha",
    "D_offset",
    "gamma_dot_t0",
    "beta",
    "gamma_dot_t_offset",
    "phi0",
]


# ---------------------------------------------------------------------------
# Fixed synthetic dataset + fixed evaluation params (deterministic).
# ---------------------------------------------------------------------------


def _build_stratified_residual_fn() -> tuple[StratifiedResidualFunctionJIT, np.ndarray]:
    """Construct ``StratifiedResidualFunctionJIT`` the way the production
    ``laminar_flow`` stratified-LS path does, on SMALL synthetic angle-stratified
    homodyne data with per-angle scaling on.

    Production path mirrored (``strategies/stratified_ls.py``):
      * a ``stratified_data`` object exposing the flat angle-stratified arrays
        (``phi_flat``/``t1_flat``/``t2_flat``/``g2_flat``), the shared metadata
        (``sigma`` 3D grid, ``q``, ``L``, ``dt``), and ``chunk_sizes`` (the
        original angle-complete chunk boundaries);
      * ``create_stratified_chunks(stratified_data, target_chunk_size)`` builds
        the ``.chunks`` / ``.sigma`` ``StratifiedChunkedData`` container — the
        exact production helper, not a re-implementation;
      * ``StratifiedResidualFunctionJIT(stratified_data=chunked,
        per_angle_scaling=True, physical_param_names=_PHYS_NAMES)`` — the exact
        production constructor call at ``stratified_ls.py:405``.

    Returns the residual function plus the FIXED parameter vector to evaluate at.
    Per-angle layout: ``[contrast_0..n-1, offset_0..n-1, *physical]``.
    """
    phi_unique = np.array([0.0, 45.0, 90.0], dtype=np.float64)  # degrees
    n_phi = len(phi_unique)
    t = np.linspace(0.0, 6.0, 7, dtype=np.float64)  # n_t = 7
    n_t = len(t)
    q = 0.0237
    L = 2_000_000.0
    dt = 0.1

    # Full (t1, t2) grid over the unique time axis. Diagonal (t1 == t2) points
    # are masked inside the engine, so we keep the full grid for fidelity.
    t1_mesh, t2_mesh = np.meshgrid(t, t, indexing="ij")
    t1_grid = t1_mesh.ravel()
    t2_grid = t2_mesh.ravel()
    n_per_angle = t1_grid.size

    # Deterministic synthetic g2-like observations (no RNG: pure function of the
    # grid so the golden is reproducible from source alone).
    g2_seed = 1.0 + 0.3 * np.exp(-0.05 * np.abs(t1_grid - t2_grid))

    # Angle-complete stratified chunking: split the per-angle index range into K
    # blocks; for each block, concatenate that block across ALL angles -> one
    # chunk. Each chunk therefore contains every phi angle, exactly as the
    # production stratifier guarantees (and as ``validate_chunk_structure``
    # enforces).
    n_chunks = 2
    blocks = np.array_split(np.arange(n_per_angle), n_chunks)

    phi_parts, t1_parts, t2_parts, g2_parts, chunk_sizes = [], [], [], [], []
    for blk in blocks:
        size = 0
        for angle_idx, p in enumerate(phi_unique):
            phi_parts.append(np.full(blk.size, p, dtype=np.float64))
            t1_parts.append(t1_grid[blk])
            t2_parts.append(t2_grid[blk])
            # Per-angle offset keeps the residual non-degenerate across angles.
            g2_parts.append(g2_seed[blk] + 1e-4 * (angle_idx + 1))
            size += blk.size
        chunk_sizes.append(size)

    class _StratData:
        """Minimal stratified-data view exposing exactly the attributes that
        ``create_stratified_chunks`` reads in production."""

    sd = _StratData()
    sd.phi_flat = np.concatenate(phi_parts)
    sd.t1_flat = np.concatenate(t1_parts)
    sd.t2_flat = np.concatenate(t2_parts)
    sd.g2_flat = np.concatenate(g2_parts)
    sd.sigma = np.full((n_phi, n_t, n_t), 1.0, dtype=np.float64)
    sd.q = q
    sd.L = L
    sd.dt = dt
    sd.chunk_sizes = chunk_sizes

    chunked = create_stratified_chunks(sd, target_chunk_size=100_000)

    residual_fn = StratifiedResidualFunctionJIT(
        stratified_data=chunked,
        per_angle_scaling=True,
        physical_param_names=_PHYS_NAMES,
        fixed_contrast_per_angle=None,
        fixed_offset_per_angle=None,
    )
    # Guards the exact production invariant (all chunks angle-complete) before we
    # snapshot — a regression that breaks chunking would fail loudly here.
    residual_fn.validate_chunk_structure()

    # Per-angle params: [contrast_0..n-1, offset_0..n-1, *physical].
    params = np.concatenate(
        [
            np.full(n_phi, 0.3, dtype=np.float64),
            np.full(n_phi, 1.0, dtype=np.float64),
            np.array([1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0], dtype=np.float64),
        ]
    )
    return residual_fn, params


def _build_laminar_fit():
    """Small synthetic ``laminar_flow`` fit through the public ``fit_nlsq`` path
    with per-angle scaling on. CMA-ES / multi-start / anti-degeneracy disabled so
    the in-memory STANDARD curve_fit path runs deterministically (mirrors
    ``tests/optimization/test_l4_callback_observational.py::_build_laminar_fit``).
    """
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.homodyne_model import HomodyneModel
    from xpcsjax.optimization.nlsq import fit_nlsq

    n_t = 8
    phi = np.array([0.0, 90.0], dtype=np.float64)
    t = np.linspace(0.0, float(n_t - 1), n_t, dtype=np.float64)
    true_params = np.array([1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0], dtype=np.float64)

    config_dict = {
        "analysis_mode": "laminar_flow",
        "analyzer_parameters": {
            "dt": 0.1,
            "start_frame": 1,
            "end_frame": n_t,
            "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": n_t},
            "scattering": {"wavevector_q": 0.0237},
            "geometry": {"stator_rotor_gap": 2000000},
        },
        "initial_parameters": {
            "parameter_names": list(_PHYS_NAMES),
            "values": true_params.tolist(),
        },
        "optimization": {
            "method": "nlsq",
            "nlsq": {
                "analysis_mode": "laminar_flow",
                "max_iterations": 50,
                "loss": "linear",
                "cmaes": {"enable": False, "auto_select": False},
                "multi_start": {"enable": False},
                "anti_degeneracy": {"enable": False},
            },
            "stratification": {"enabled": False},
        },
    }

    cfg = ConfigManager(config_override=config_dict)
    model = HomodyneModel(cfg.config)
    c2 = np.asarray(
        model.compute_c2(true_params, phi, contrast=0.3, offset=1.0),
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed=20260529)
    c2 = c2 + rng.normal(0.0, 5e-4, size=c2.shape)

    data = {
        "phi_angles_list": phi,
        "c2_exp": c2,
        "t1": t,
        "t2": t,
        "wavevector_q_list": np.array([0.0237], dtype=np.float64),
    }
    return fit_nlsq(data, cfg)


# ---------------------------------------------------------------------------
# LAYER 1 — UNIT golden: StratifiedResidualFunctionJIT residual vector.
# ---------------------------------------------------------------------------


def test_stratified_residual_jit_golden():
    """Snapshot the residual VECTOR emitted by the production
    ``StratifiedResidualFunctionJIT`` at a fixed parameter vector. This is the
    class Phase 1 swaps ``compute_g2_scaled`` -> ``evaluator.eval_points`` inside;
    a bit-level change in the residual would fail this gate at ``rtol=1e-10``.
    """
    residual_fn, params = _build_stratified_residual_fn()
    residual = np.asarray(residual_fn(params), dtype=np.float64)

    golden_path = _GOLDEN_DIR / "stratified_residual_jit.npz"
    golden = load_or_init_golden(
        golden_path,
        regen=_REGEN,
        payload=lambda: {"residual": residual, "n_phi": np.int64(residual_fn.n_phi)},
    )

    assert residual.shape == golden["residual"].shape, (
        f"residual shape changed: {residual.shape} != {tuple(golden['residual'].shape)}"
    )
    assert int(golden["n_phi"]) == residual_fn.n_phi
    np.testing.assert_allclose(
        residual,
        golden["residual"],
        rtol=1e-10,
        atol=0.0,
        err_msg=(
            "StratifiedResidualFunctionJIT residual drifted from golden — the "
            "Phase-1 PointEvaluator refactor changed homodyne engine behavior."
        ),
    )


# ---------------------------------------------------------------------------
# LAYER 2 — END-TO-END golden: small laminar_flow fit through fit_nlsq.
# ---------------------------------------------------------------------------


def test_laminar_flow_end_to_end_golden():
    """Snapshot the public-path ``laminar_flow`` fit result: parameters,
    objective, uncertainties (shape + finite pattern), ``nlsq_diagnostics`` key
    set, anti-degeneracy activation flags, strategy/status, and covariance shape
    + finite pattern. Params/objective compared at ``rtol=1e-10``; keys / flags /
    shapes / patterns compared EXACTLY.
    """
    result = _build_laminar_fit()

    params = np.asarray(result.parameters, dtype=np.float64)
    chi_squared = float(result.chi_squared)
    uncertainties = np.asarray(result.uncertainties, dtype=np.float64)
    covariance = np.asarray(result.covariance, dtype=np.float64)
    diag = dict(result.nlsq_diagnostics or {})

    # Anti-degeneracy activation flags (present at all dataset sizes for laminar).
    flags = {
        "hierarchical_active": bool(diag.get("hierarchical_active")),
        "regularization_active": bool(diag.get("regularization_active")),
        "shear_weighting": str(diag.get("shear_weighting")),
    }
    has_gradient_monitor = "gradient_monitor" in diag

    # Selected strategy / tier + convergence status (exact-match contract).
    strategy = str(getattr(result, "strategy", None) or diag.get("strategy") or "")
    convergence_status = str(getattr(result, "convergence_status", ""))
    quality_flag = str(getattr(result, "quality_flag", ""))

    diag_keys = sorted(diag.keys())
    uncert_finite = np.isfinite(uncertainties)
    cov_finite = np.isfinite(covariance)

    golden_path = _GOLDEN_DIR / "laminar_flow_end_to_end.npz"
    golden = load_or_init_golden(
        golden_path,
        regen=_REGEN,
        payload=lambda: {
            "parameters": params,
            "chi_squared": np.float64(chi_squared),
            "uncertainties_shape": np.asarray(uncertainties.shape, dtype=np.int64),
            "uncertainties_finite": uncert_finite,
            "covariance_shape": np.asarray(covariance.shape, dtype=np.int64),
            "covariance_finite": cov_finite,
            "diag_keys": np.asarray(diag_keys, dtype=object),
            "flag_hierarchical_active": np.bool_(flags["hierarchical_active"]),
            "flag_regularization_active": np.bool_(flags["regularization_active"]),
            "flag_shear_weighting": np.asarray(flags["shear_weighting"], dtype=object),
            "has_gradient_monitor": np.bool_(has_gradient_monitor),
            "strategy": np.asarray(strategy, dtype=object),
            "convergence_status": np.asarray(convergence_status, dtype=object),
            "quality_flag": np.asarray(quality_flag, dtype=object),
        },
    )

    # --- numerical: params + objective at rtol=1e-10 ---
    assert params.shape == tuple(golden["parameters"].shape), (
        f"parameter vector length changed: {params.shape} != {tuple(golden['parameters'].shape)}"
    )
    np.testing.assert_allclose(
        params,
        golden["parameters"],
        rtol=1e-10,
        atol=0.0,
        err_msg="laminar_flow fitted parameters drifted from golden.",
    )
    np.testing.assert_allclose(
        chi_squared,
        float(golden["chi_squared"]),
        rtol=1e-10,
        atol=0.0,
        err_msg="laminar_flow chi_squared drifted from golden.",
    )

    # --- exact: uncertainty shape + finite pattern ---
    assert tuple(uncertainties.shape) == tuple(int(x) for x in golden["uncertainties_shape"])
    assert np.array_equal(uncert_finite, golden["uncertainties_finite"]), (
        "uncertainties finite/NaN pattern changed."
    )

    # --- exact: covariance shape + finite pattern ---
    assert tuple(covariance.shape) == tuple(int(x) for x in golden["covariance_shape"])
    assert np.array_equal(cov_finite, golden["covariance_finite"]), (
        "covariance finite/NaN pattern changed."
    )

    # --- exact: diagnostics key set ---
    golden_keys = [str(k) for k in golden["diag_keys"].tolist()]
    assert diag_keys == golden_keys, (
        f"nlsq_diagnostics key set changed: {diag_keys} != {golden_keys}"
    )

    # --- exact: anti-degeneracy activation flags ---
    assert flags["hierarchical_active"] == bool(golden["flag_hierarchical_active"])
    assert flags["regularization_active"] == bool(golden["flag_regularization_active"])
    assert flags["shear_weighting"] == str(golden["flag_shear_weighting"])
    assert has_gradient_monitor == bool(golden["has_gradient_monitor"])

    # --- exact: strategy / tier / status ---
    assert strategy == str(golden["strategy"])
    assert convergence_status == str(golden["convergence_status"])
    assert quality_flag == str(golden["quality_flag"])


@pytest.mark.parametrize("golden_name", ["stratified_residual_jit", "laminar_flow_end_to_end"])
def test_golden_files_are_committed(golden_name):
    """On a normal (non-regen) run the goldens MUST already exist on disk — this
    catches an accidentally-uncommitted golden that would let the assertions
    above silently regenerate instead of guarding.
    """
    if _REGEN:
        pytest.skip("regeneration mode: goldens are being (re)written this run")
    path = _GOLDEN_DIR / f"{golden_name}.npz"
    assert path.exists(), (
        f"golden {path} is missing — commit tests/parity/_golden/ or run with "
        f"XPCSJAX_REGEN_GOLDEN=1 to (re)generate."
    )
