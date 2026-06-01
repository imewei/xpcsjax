"""Laminar-flow-parity logging banners for the heterodyne ``two_component`` NLSQ paths.

The homodyne / ``laminar_flow`` path narrates strategy selection, anti-degeneracy
layer setup, stratification diagnostics, a gradient sanity check, and an
optimization-results block (emitted across
``xpcsjax.optimization.nlsq.wrapper`` / ``core`` / ``anti_degeneracy_controller``).
The heterodyne paths historically emitted only the open / close banner
(``heterodyne_core.log_heterodyne_start`` / ``log_heterodyne_completion``),
leaving the multi-minute solve **silent** between them — most visibly the
stratified-LS path (``heterodyne_stratified_ls``), which logged nothing at all.

These helpers reproduce the laminar log *surface* for every ``two_component``
path while reading each path's **real** values, so the narration stays honest:
the stratified-LS path runs a plain joint least-squares solve (NOT the full
``AntiDegeneracyController``), so its defense summary reports
``hierarchical_active=False`` / ``regularization_active=False`` exactly as its
``nlsq_diagnostics`` already declare. Fabricating "Layer 2 Enabled: True" here
would contradict the diagnostics contract pinned by the heterodyne tests.

Banner widths mirror laminar exactly: the controller-style anti-degeneracy
banners are 60 wide (``anti_degeneracy_controller``); the wrapper-style path /
results banners are 80 wide (``wrapper``).

Pure logging: every function reads its arguments and emits log records only.
None mutate solver state or influence numerics — log text is invisible to the
``rtol=1e-10`` parity gates.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)

_W60 = "=" * 60
_W80 = "=" * 80


# ---------------------------------------------------------------------------
# Strategy selection + path-activation banners (wrapper-parity, 80 wide)
# ---------------------------------------------------------------------------
def log_strategy_selection(strategy: str, detail: str) -> None:
    """Mirror laminar ``wrapper`` "Strategy selection: <name> (<detail>)".

    ``detail`` is the human-readable routing reason (e.g. the point count vs the
    1M stratified-LS gate, or the memory tier). Emitted once per analysis from
    the dispatcher so the two_component log records WHICH solver path ran.
    """
    logger.info("Strategy selection: %s (%s)", strategy, detail)


def log_stratified_path_activated(n_points: int) -> None:
    """Mirror laminar "STRATIFIED LEAST-SQUARES PATH ACTIVATED" banner (80 wide)."""
    logger.info(_W80)
    logger.info("STRATIFIED LEAST-SQUARES PATH ACTIVATED (heterodyne two_component)")
    logger.info("Solving the >=1M joint fit with NLSQ's least_squares()")
    logger.info("  Real data points: %s", f"{int(n_points):,}")
    logger.info(_W80)


def log_physical_parameters(analysis_mode: str, names: Sequence[str]) -> None:
    """Mirror laminar "Physical parameters for <mode>: [...]"."""
    logger.info("Physical parameters for %s: %s", analysis_mode, list(names))


# ---------------------------------------------------------------------------
# Anti-degeneracy banners (controller-parity, 60 wide)
# ---------------------------------------------------------------------------
def log_effective_mode(
    mode: str,
    *,
    n_phi: int,
    n_physics: int,
    n_scaling: int,
    threshold: int | None = None,
) -> None:
    """Mirror the controller's mode-selection banner for the resolved mode.

    Reports heterodyne's real parameter budget (``n_physics`` is 14 for
    two_component, not laminar's 7) so the "X physical + Y scaling = Z total"
    line is accurate per mode.
    """
    logger.info(_W60)
    logger.info("ANTI-DEGENERACY: Effective per-angle mode '%s'", mode)
    if threshold is not None:
        rel = ">=" if n_phi >= threshold else "<"
        logger.info(
            "  Reason: n_phi (%d) %s constant_scaling_threshold (%d)",
            n_phi,
            rel,
            threshold,
        )
    if mode == "averaged":
        descr = f"{n_scaling} averaged scaling"
    elif mode == "fourier":
        descr = f"{n_scaling} Fourier coeffs"
    elif mode == "individual":
        descr = f"{n_scaling} per-angle scaling"
    else:
        descr = f"{n_scaling} scaling"
    logger.info(
        "  Parameters: %d physical + %s = %d total",
        n_physics,
        descr,
        n_physics + n_scaling,
    )
    logger.info(_W60)


def log_anti_degeneracy_defense(diagnostics: dict[str, Any] | None) -> None:
    """Emit the laminar-style "ANTI-DEGENERACY DEFENSE" summary from real values.

    Reads the assembled ``nlsq_diagnostics`` block
    (``assemble_anti_degeneracy_diagnostics``) so the reported layer activity is
    HONEST per path: the stratified-LS / sequential / out-of-core paths report
    ``hierarchical_active=False`` / ``regularization_active=False`` because those
    layers do not run there, while the in-memory / streaming paths report the
    real active layers they ran. L5 (shear weighting) is reported via its
    sentinel — heterodyne has no shear term, so it is structurally inactive.
    """
    diag = diagnostics or {}
    logger.info(_W60)
    logger.info("ANTI-DEGENERACY DEFENSE (heterodyne two_component)")
    logger.info("  per_angle_mode: %s", diag.get("per_angle_mode", "?"))
    logger.info("  L1 reparameterization: %s", _layer_state(diag, "per_angle_mode"))
    logger.info("  L2 hierarchical_active: %s", bool(diag.get("hierarchical_active", False)))
    logger.info(
        "  L3 regularization_active: %s", bool(diag.get("regularization_active", False))
    )
    gm = diag.get("gradient_monitor")
    if gm is not None:
        mech = gm.get("mechanism", "?") if isinstance(gm, dict) else "?"
        logger.info("  L4 gradient_monitor: %s", mech)
    else:
        logger.info("  L4 gradient_monitor: not_run")
    logger.info("  L5 shear_weighting: %s", diag.get("shear_weighting", "?"))
    logger.info(_W60)


def _layer_state(diag: dict[str, Any], mode_key: str) -> str:
    """L1 (Fourier/constant reparam) is active for every optimized mode and
    skipped only for the frozen ``fixed_constant`` mode (laminar parity)."""
    mode = str(diag.get(mode_key, ""))
    return "inactive (fixed_constant)" if mode == "fixed_constant" else "active"


# ---------------------------------------------------------------------------
# Quantile scaling (controller-parity)
# ---------------------------------------------------------------------------
def log_quantile_scaling(contrast_pa: np.ndarray, offset_pa: np.ndarray) -> None:
    """Mirror laminar "Quantile-based per-angle estimation complete" with the
    range AND the mean/std summary (laminar logs both; the old heterodyne path
    logged only the range, and duplicated it three times)."""
    c = np.asarray(contrast_pa, dtype=np.float64)
    o = np.asarray(offset_pa, dtype=np.float64)
    logger.info("Quantile-based per-angle estimation complete:")
    logger.info("  Contrast range: [%.4f, %.4f]", float(np.nanmin(c)), float(np.nanmax(c)))
    logger.info("  Offset range: [%.4f, %.4f]", float(np.nanmin(o)), float(np.nanmax(o)))
    logger.info("  n_phi: %d", c.size)
    logger.info("  Contrast: mean=%.4f, std=%.4f", float(np.nanmean(c)), float(np.nanstd(c)))
    logger.info("  Offset: mean=%.4f, std=%.4f", float(np.nanmean(o)), float(np.nanstd(o)))


# ---------------------------------------------------------------------------
# Stratification diagnostics (wrapper-parity)
# ---------------------------------------------------------------------------
def log_stratification_diagnostics(
    diag: Any, *, n_chunks: int, n_points: int, n_phi: int
) -> None:
    """Mirror laminar "Stratified Residual Function Diagnostics" block.

    ``diag`` may be a ``StratificationDiagnostics`` dataclass (from
    ``strategies.chunking``) or a plain dict; both are read tolerantly.
    """

    def _get(key: str) -> Any:
        if isinstance(diag, dict):
            return diag.get(key)
        return getattr(diag, key, None)

    logger.info("Stratified Residual Function Diagnostics:")
    logger.info("  Chunks: %d", n_chunks)
    logger.info("  Real points: %s", f"{int(n_points):,}")
    logger.info("  Angles (phi): %d", n_phi)
    thr = _get("throughput_points_per_sec")
    if isinstance(thr, (int, float)):
        logger.info("  Throughput: %.2fM pts/s", float(thr) / 1e6)
    et = _get("execution_time_ms")
    if isinstance(et, (int, float)):
        logger.info("  Stratification time: %.3f ms", float(et))


# ---------------------------------------------------------------------------
# Fit start + results blocks (wrapper-parity)
# ---------------------------------------------------------------------------
def log_fit_start(n_params: int, n_points: int, *, n_chunks: int | None = None) -> None:
    """Mirror laminar "Starting NLSQ least_squares() optimization..." block."""
    logger.info("Starting NLSQ least_squares() optimization...")
    logger.info("  Initial parameters: %d parameters", n_params)
    logger.info("  Bounds: provided")
    if n_chunks is not None:
        logger.info("  Residual chunks: %d", n_chunks)
    logger.info("  Real data points: %s", f"{int(n_points):,}")


def log_optimization_results(
    *,
    success: bool,
    message: str | None,
    n_iterations: int,
    initial_cost: float | None,
    final_cost: float,
    wall_time: float,
    function_evals: int | None = None,
) -> None:
    """Mirror laminar "OPTIMIZATION RESULTS" block (80 wide).

    ``initial_cost``/``function_evals`` are optional because the heterodyne
    adapter does not always surface them; missing values are reported as ``n/a``
    rather than fabricated.
    """
    logger.info(_W80)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("  Status: %s", "SUCCESS" if success else "FAILED")
    logger.info("  Message: %s", message if message else "n/a")
    logger.info(
        "  Function evaluations: %s",
        function_evals if function_evals is not None else "n/a",
    )
    logger.info("  Iterations: %d", n_iterations)
    if initial_cost is not None:
        logger.info("  Initial cost: %.6e", initial_cost)
        logger.info("  Final cost: %.6e", final_cost)
        if initial_cost > 0.0:
            reduction = (initial_cost - final_cost) / initial_cost
            logger.info("  Cost reduction: %+.2f%%", reduction * 100.0)
    else:
        logger.info("  Final cost: %.6e", final_cost)
    logger.info("  Total time: %.2fs", wall_time)
    logger.info(_W80)


def log_stratified_complete(chi2: float, reduced_chi2: float) -> None:
    """Mirror laminar "STRATIFIED LEAST-SQUARES COMPLETE" sub-banner (80 wide)."""
    logger.info(_W80)
    logger.info("STRATIFIED LEAST-SQUARES COMPLETE")
    logger.info("Final chi2: %.4e, Reduced chi2: %.4f", chi2, reduced_chi2)
    logger.info(_W80)
