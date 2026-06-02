"""Core NLSQ fitting for heterodyne analysis.

Unified entry point for NLSQ optimization with:
- Global optimization selection (CMA-ES → multi-start → local)
- Adapter/wrapper fallback with automatic recovery
- Memory-aware strategy selection
- Per-angle and multi-angle fitting
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from xpcsjax.core.heterodyne_jax_backend import (
    compute_c2_heterodyne,
    compute_multi_angle_residuals,
    compute_residuals,
)
from xpcsjax.optimization.nlsq.anti_degeneracy_diagnostics import (
    assemble_anti_degeneracy_diagnostics,
)
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_results import NLSQResult
from xpcsjax.optimization.nlsq.results import OptimizationResult
from xpcsjax.optimization.nlsq.validation import classify_quality_flag
from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    # The runtime object the fitter receives is the stateful dataclass in
    # ``heterodyne_model_stateful`` (which exposes ``.t``, ``.q``, ``.dt``,
    # ``.scaling``, ``.param_manager``, ``.set_params``). The bare wrapper in
    # ``heterodyne_model`` is a PhysicsModelBase adapter without those fields,
    # so typing against it produced ~10 spurious "no attribute" mypy errors.
    from xpcsjax.core.heterodyne_model_stateful import (
        HeterodyneModel as HeterodyneModel,
    )
    from xpcsjax.optimization.nlsq.results import ConvergenceStatus, QualityFlag

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — gated for graceful degradation
# ---------------------------------------------------------------------------

# NOTE: every optional import below binds the imported names to ``None`` in
# the ImportError branch. Without this, Pyright cannot reason through the
# ``if HAS_X: X(...)`` runtime gates and emits ~10 "X is possibly unbound"
# warnings per call site. With explicit ``None`` bindings the type becomes
# ``T | None`` and narrows correctly. Call sites still gate on the ``HAS_X``
# flag; the explicit ``is not None`` check at hot-path sites is belt-and-
# suspenders for readers, not a runtime necessity.
try:
    # The heterodyne-shaped NLSQAdapter / NLSQWrapper expect the upstream
    # contract (parameter_names + residual_fn). xpcsjax's own NLSQAdapter
    # (in adapter.py) is shaped differently. Use the ported heterodyne
    # adapter module so the orchestrator gets the contract it expects.
    from xpcsjax.optimization.nlsq.heterodyne_adapter import (
        NLSQAdapter,
        NLSQWrapper,
    )

    HAS_ADAPTERS = True
    HAS_WRAPPER = True
except ImportError:
    NLSQAdapter = None  # type: ignore[assignment,misc]
    NLSQWrapper = None  # type: ignore[assignment,misc]
    HAS_ADAPTERS = False
    HAS_WRAPPER = False

# Multi-start orchestration is intentionally NOT imported here: the v0.1
# ``_fit_multistart`` function raises NotImplementedError unconditionally (see
# its docstring for why — the upstream homodyne port called a class-style
# ``MultiStartOptimizer.fit(...)`` API that ``xpcsjax.optimization.nlsq.multistart``
# does not expose). Keep ``HAS_MULTISTART`` as a const ``False`` so the
# existing ``if HAS_MULTISTART: _fit_multistart(...)`` dispatch falls through
# to the warning + local-fit path instead of hitting NotImplementedError
# during normal smoke runs.
HAS_MULTISTART = False


def _post_solve_covariance_l4(joint_result: NLSQResult, config: NLSQConfig) -> dict[str, Any]:
    """Compute the legacy post-solve covariance-condition L4 diagnostic block.

    This is the fallback path used when the per-iteration gradient-collapse
    callback recorded zero observations (no per-iteration signal available).
    It derives ``max_gradient_ratio`` from the singular-value condition number
    of the fitted covariance:

    - finite condition number when the covariance is well-formed,
    - ``+inf`` when the covariance is singular,
    - ``nan`` when no covariance is available.

    Triggering rule: ``collapse_detected = (cov_cond >= threshold)`` with
    ``cov_cond == +inf`` always treated as collapse.

    Returns the legacy block keys (``collapse_detected``, ``max_gradient_ratio``,
    ``trigger_count``, ``scope``, ``ratio_threshold_configured``,
    ``consecutive_triggers_configured``, ``threshold_used``,
    ``computation_method``). The caller is responsible for tagging
    ``mechanism="post_solve_fallback"``.
    """
    if joint_result.covariance is not None:
        try:
            _cov_for_cond = np.asarray(joint_result.covariance, dtype=np.float64)
            _sv = np.linalg.svd(_cov_for_cond, compute_uv=False)
            _sv = np.where(_sv > 0, _sv, np.finfo(np.float64).tiny)
            _cov_condition = float(_sv[0] / _sv[-1])
            max_gradient_ratio = _cov_condition if np.isfinite(_cov_condition) else float("inf")
        except (np.linalg.LinAlgError, ValueError):
            max_gradient_ratio = float("inf")
    else:
        max_gradient_ratio = float("nan")

    _threshold = float(config.gradient_ratio_threshold)
    _collapse = (
        np.isfinite(max_gradient_ratio) and max_gradient_ratio >= _threshold
    ) or max_gradient_ratio == float("inf")
    return {
        "collapse_detected": bool(_collapse),
        "max_gradient_ratio": float(max_gradient_ratio),
        "trigger_count": int(_collapse),
        "scope": "post_solve_covariance_conditioning",
        "ratio_threshold_configured": float(config.gradient_ratio_threshold),
        "consecutive_triggers_configured": int(config.gradient_consecutive_triggers),
        "threshold_used": _threshold,
        "computation_method": "covariance_singular_value_ratio",
    }


def _build_l4_callback(
    model: HeterodyneModel,
    x0: np.ndarray,
    joint_residual_fn: Any,
    config: NLSQConfig,
) -> tuple[Any, Any]:
    """Build the L4 per-iteration gradient-collapse monitor and curve_fit callback.

    Returns ``(None, None)`` when gradient monitoring is disabled (so the caller
    builds no monitor and passes no callback, leaving the fit unchanged). When
    enabled, returns ``(monitor, callback)`` where the monitor watches the joint
    parameter layout ``[physics (n_physics) | scaling tail]`` and the callback is
    strictly observational — Phase-0 proved NLSQ's curve_fit callback fires
    per-iteration and never perturbs the solve.
    """
    if not config.enable_gradient_monitoring:
        return None, None

    import jax

    from xpcsjax.optimization.nlsq.gradient_monitor import (
        GradientCollapseMonitor,
        GradientMonitorConfig,
        build_gradient_collapse_callback,
    )

    n_physics = int(model.param_manager.n_varying)
    total = len(x0)
    gm_cfg = GradientMonitorConfig(
        ratio_threshold=float(config.gradient_ratio_threshold),
        consecutive_triggers=int(config.gradient_consecutive_triggers),
        check_interval=1,
    )
    monitor = GradientCollapseMonitor(
        gm_cfg,
        physical_indices=list(range(n_physics)),
        per_angle_indices=list(range(n_physics, total)),
    )

    def _loss(p: Any) -> Any:
        return 0.5 * jnp.sum(joint_residual_fn(jnp.asarray(p)) ** 2)

    grad_fn = jax.jit(jax.grad(_loss))
    callback = build_gradient_collapse_callback(monitor, grad_fn)
    return monitor, callback


def _assemble_l4_extras(
    monitor: Any,
    joint_result: NLSQResult,
    config: NLSQConfig,
    *,
    mode_label: str,
    result_is_monitored: bool = True,
) -> dict[str, Any]:
    """Assemble the L4 ``gradient_monitor`` diagnostics block from a monitor.

    Returns ``{}`` when ``monitor`` is ``None`` (monitoring disabled). Otherwise
    builds the per-iteration diagnostics, falling back to the post-solve
    covariance-condition block (tagged ``mechanism="post_solve_fallback"``) when
    the callback recorded zero observations. Wraps the result as
    ``{"gradient_monitor": block}``.

    ``result_is_monitored`` guards against a stale monitor: the callback is only
    passed to the ``NLSQAdapter``, so when the adapter fires the callback (≥ 1
    observation) but then fails and the unmonitored ``NLSQWrapper`` fallback
    produces the returned ``joint_result``, the monitor's per-iteration ratios
    describe a DISCARDED run's parameters. Pass ``result_is_monitored=False`` in
    that case to force the post-solve covariance-condition block (computed from
    the actual returned ``joint_result``, tagged ``mechanism="post_solve_fallback"``)
    instead of trusting the stale monitor. The default ``True`` keeps the happy
    path (adapter succeeded → returned result IS the monitored run) unchanged.
    """
    if monitor is None:
        return {}

    from xpcsjax.optimization.nlsq.gradient_monitor import gradient_monitor_diagnostics

    gm_block = gradient_monitor_diagnostics(monitor) if result_is_monitored else None
    if gm_block is None or gm_block["mechanism"] == "post_solve_fallback":
        gm_block = _post_solve_covariance_l4(joint_result, config)
        gm_block["mechanism"] = "post_solve_fallback"
    logger.info(
        "L4 gradient collapse monitor enabled (%s): "
        "mechanism=%s, n_observations=%s, max_gradient_ratio=%.3g, "
        "collapse_detected=%s.",
        mode_label,
        gm_block["mechanism"],
        gm_block.get("n_observations"),
        gm_block["max_gradient_ratio"],
        gm_block["collapse_detected"],
    )
    return {"gradient_monitor": gm_block}


try:
    from xpcsjax.optimization.nlsq.cmaes_wrapper import (
        CMAES_AVAILABLE,
        fit_with_cmaes,
    )

    HAS_CMAES = CMAES_AVAILABLE
except ImportError:
    fit_with_cmaes = None  # type: ignore[assignment,misc]
    HAS_CMAES = False

# Joint multistart escape (Task 3). ``run_multistart_nlsq`` is imported at MODULE
# scope (not lazily inside the escape) so the joint-multistart fallback test can
# monkeypatch ``heterodyne_core.run_multistart_nlsq``. ``multistart`` does NOT
# import ``heterodyne_core`` so there is no import cycle. ``HAS_JOINT_MULTISTART``
# reflects only whether the orchestrator is importable — the JOINT path runs
# ``run_multistart_nlsq`` sequentially (``n_workers=1``, the JAX-pickle
# constraint) regardless of the module-level ``HAS_MULTISTART=False`` flag, which
# gates only the legacy per-angle ``_fit_multistart`` stub.
try:
    from xpcsjax.optimization.nlsq.multistart import (
        MultiStartConfig,
        SingleStartResult,
        run_multistart_nlsq,
    )

    HAS_JOINT_MULTISTART = True
except ImportError:
    MultiStartConfig = None  # type: ignore[assignment,misc]
    SingleStartResult = None  # type: ignore[assignment,misc]
    run_multistart_nlsq = None  # type: ignore[assignment,misc]
    HAS_JOINT_MULTISTART = False

# Seed for the joint multistart LHS start generation. PINNED so the global
# search is bit-reproducible run to run (mirrors ``_JOINT_CMAES_SEED``).
_JOINT_MULTISTART_SEED = 42

try:
    # Heterodyne uses its own memory module (``STANDARD/LARGE/STREAMING`` enum
    # vocabulary). The homodyne ``memory.py`` uses
    # ``STANDARD/OUT_OF_CORE/HYBRID_STREAMING`` — importing from there left
    # ``NLSQStrategy.LARGE`` undefined at runtime in the heterodyne hot path.
    from xpcsjax.optimization.nlsq.heterodyne_memory import (
        NLSQStrategy,
        select_nlsq_strategy,
    )

    HAS_MEMORY = True
except ImportError:
    NLSQStrategy = None  # type: ignore[assignment,misc]
    select_nlsq_strategy = None  # type: ignore[assignment,misc]
    HAS_MEMORY = False

# Export availability flag for tests
NLSQ_AVAILABLE = HAS_ADAPTERS


# ---------------------------------------------------------------------------
# Shared diagnostics helper (used by every joint multi-phi path that returns
# an OptimizationResult — currently the Fourier path here and the constant
# path in heterodyne_constant_mode.py via re-import)
# ---------------------------------------------------------------------------


def _build_heterodyne_diagnostics(
    per_angle_mode: str,
    chi2_per_angle: np.ndarray,
    scaling_source: str,
    fourier_basis_dim: int | None,
    **extras: Any,
) -> dict[str, Any]:
    """Build the standard heterodyne ``nlsq_diagnostics`` dict.

    Centralises the five canonical keys every heterodyne-side
    :class:`OptimizationResult` carries so the Fourier-mode joint path here
    and the constant-mode joint path in :mod:`heterodyne_constant_mode` stay
    in lockstep. Extra mode-specific keys (e.g. ``contrast_per_angle_fixed``
    in constant mode, ``fourier_coeffs`` in Fourier mode) are passed through
    ``**extras``.

    The anti-degeneracy activation block (``hierarchical_active`` /
    ``regularization_active`` / ``shear_weighting``, plus the L4
    ``gradient_monitor`` when present) is assembled by the shared
    :func:`assemble_anti_degeneracy_diagnostics`, so heterodyne and homodyne
    surface the SAME activation-key set. Both ``*_active`` flags are emitted on
    EVERY path (``False`` when the layer did not run); only the per-layer DETAIL
    keys (``hierarchical_stages``, ``regularization_mode``, ...) remain
    conditional and flow through ``**extras`` verbatim. The
    ``"not_applicable_heterodyne"`` shear-weighting marker makes the homodyne
    L5 layer's N/A status explicit for heterodyne.
    """
    # The activation flags arrive in ``extras`` only when the layer ran; pop
    # them out so the always-emit assembler owns them (default ``False``).
    extras = dict(extras)
    hierarchical_active = bool(extras.pop("hierarchical_active", False))
    regularization_active = bool(extras.pop("regularization_active", False))
    gradient_monitor = extras.pop("gradient_monitor", None)

    base: dict[str, Any] = {
        "per_angle_mode": per_angle_mode,
        "chi2_per_angle": chi2_per_angle,
        "scaling_source": scaling_source,
        "fourier_basis_dim": fourier_basis_dim,
    }
    base.update(
        assemble_anti_degeneracy_diagnostics(
            hierarchical_active=hierarchical_active,
            regularization_active=regularization_active,
            shear_weighting="not_applicable_heterodyne",
            gradient_monitor=gradient_monitor,
            **extras,
        )
    )
    return base


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_nlsq_jax(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float = 0.0,
    config: NLSQConfig | None = None,
    weights: np.ndarray | jnp.ndarray | None = None,
    use_nlsq_library: bool = True,
    *,
    _skip_global_selection: bool = False,
    angle_idx: int = 0,
) -> NLSQResult:
    """Fit heterodyne model to correlation data using NLSQ.

    This is the unified entry point for all NLSQ optimization.  When called
    it first checks for global optimization methods:

    1. If ``cmaes.enable: true`` → delegates to CMA-ES
    2. If ``multi_start.enable: true`` → delegates to multi-start
    3. Otherwise → runs local trust-region optimization

    The adapter is tried first; on failure the wrapper provides automatic
    retry with progressive recovery (HybridRecoveryConfig).

    Args:
        model: HeterodyneModel instance with parameters configured.
        c2_data: Experimental correlation data, shape (N, N).
        phi_angle: Detector phi angle (degrees).
        config: NLSQ configuration (default if None).
        weights: Optional weights (1/sigma²) for weighted least squares.
        use_nlsq_library: Whether to prefer nlsq library over scipy.
        _skip_global_selection: Internal flag — skip CMA-ES / multi-start check.
        angle_idx: Per-angle scaling index for the fixed contrast/offset values.

    Returns:
        NLSQResult with fitted parameters and diagnostics.
    """
    if config is None:
        config = NLSQConfig()

    logger.info("=" * 60)
    logger.info("NLSQ OPTIMIZATION")
    logger.info("=" * 60)
    logger.info("phi=%s°, method=%s", phi_angle, config.method)

    # ------------------------------------------------------------------
    # Global optimization selection (CMA-ES → multi-start → local)
    # ------------------------------------------------------------------
    if not _skip_global_selection:
        global_result = _try_global_optimization(
            model,
            c2_data,
            phi_angle,
            config,
            weights,
            use_nlsq_library,
            angle_idx,
        )
        if global_result is not None:
            return global_result

    # ------------------------------------------------------------------
    # Local optimization
    # ------------------------------------------------------------------
    return _fit_local(model, c2_data, phi_angle, config, weights, use_nlsq_library, angle_idx)


def _aggregate_individual_results(
    per_angle_results: list[NLSQResult],
    model: HeterodyneModel,
    phi_angles: np.ndarray,
    c2_data: np.ndarray,
    wall_time: float,
    config: NLSQConfig | None = None,
    weights: np.ndarray | None = None,
) -> OptimizationResult:
    """Aggregate sequential per-angle ``NLSQResult``s into one ``OptimizationResult``.

    Each per-angle :class:`NLSQResult` carries only the ``n_physics``
    varying-physics parameters (see :func:`_fit_local`: per-angle
    contrast/offset are held fixed at the values
    ``model.scaling.get_for_angle(i)`` during the local fit). The
    aggregator packs the joint parameter vector as

    ``[physics_mean | contrast_0..contrast_{n_phi-1} | offset_0..offset_{n_phi-1}]``

    matching the ``n_physics + 2 * n_phi`` parameter-dim contract from
    the homodyne anti-degeneracy taxonomy
    (``tests/parity/test_mode_taxonomy.py``).

    The covariance matrix is **block-diagonal by construction**:

    - The leading ``n_physics × n_physics`` block holds the mean of the
      per-angle physics covariance sub-blocks (each per-angle fit ran
      independently, so the mean is the natural pooled estimate).
    - The trailing ``2 * n_phi`` scaling rows/columns carry zero variance
      because contrast/offset were held fixed during each per-angle fit.
    - All physics-vs-scaling and angle-vs-angle off-diagonals are exactly
      zero (no joint fit, no cross-correlation information available).

    Downstream consumers should read
    ``nlsq_diagnostics["covariance_structure"] == "block_diagonal_sequential"``
    to detect this case and avoid mistaking constructed zeros for
    fitted-zero correlations.

    Convergence status maps as follows:

    - All per-angle fits successful → ``"converged"`` / ``quality_flag="good"``
    - Mixed success/failure → ``"partial"`` / ``quality_flag="marginal"``
    """
    n_phi = len(per_angle_results)
    if n_phi == 0:
        raise ValueError("_aggregate_individual_results: at least one per-angle result required")

    n_physics = int(model.param_manager.n_varying)
    varying_names = list(model.param_manager.varying_names)
    total_dim = n_physics + 2 * n_phi

    # ------------------------------------------------------------------
    # Parameters: mean physics across angles + per-angle scaling tail
    # ------------------------------------------------------------------
    physics_per_angle = np.stack(
        [np.asarray(r.parameters, dtype=np.float64)[:n_physics] for r in per_angle_results]
    )
    physics_mean = physics_per_angle.mean(axis=0)
    contrast_per_angle = np.asarray(
        [float(model.scaling.contrast[i]) for i in range(n_phi)], dtype=np.float64
    )
    offset_per_angle = np.asarray(
        [float(model.scaling.offset[i]) for i in range(n_phi)], dtype=np.float64
    )
    aggregated_params = np.concatenate([physics_mean, contrast_per_angle, offset_per_angle])

    # ------------------------------------------------------------------
    # Block-diagonal covariance: mean of per-angle physics blocks; zeros
    # for the scaling tail (fixed during the per-angle fit).
    # ------------------------------------------------------------------
    covariance = np.zeros((total_dim, total_dim), dtype=np.float64)
    physics_cov_blocks: list[np.ndarray] = []
    for r in per_angle_results:
        if r.covariance is None:
            continue
        cov_arr = np.asarray(r.covariance, dtype=np.float64)
        if cov_arr.shape == (n_physics, n_physics):
            physics_cov_blocks.append(cov_arr)
        elif cov_arr.shape[0] >= n_physics:
            physics_cov_blocks.append(cov_arr[:n_physics, :n_physics])
    if physics_cov_blocks:
        covariance[:n_physics, :n_physics] = np.mean(physics_cov_blocks, axis=0)
    uncertainties = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    # ------------------------------------------------------------------
    # SSR + iteration aggregation
    # ------------------------------------------------------------------
    chi2_values: list[float] = []
    for i, r in enumerate(per_angle_results):
        if r.fitted_correlation is not None:
            residual = np.asarray(r.fitted_correlation, dtype=np.float64) - np.asarray(
                c2_data[i], dtype=np.float64
            )
            if weights is not None:
                w_i = weights[i] if weights.ndim == 3 else weights
                residual = residual * np.sqrt(np.asarray(w_i, dtype=np.float64))
            n_matrix = residual.shape[0]
            off_diag = ~np.eye(n_matrix, dtype=bool)
            # OptimizationResult.chi_squared is defined as data residual SSR.
            chi2_values.append(float(np.sum(residual[off_diag] ** 2)))
        elif r.final_cost is not None:
            # NLSQResult.final_cost follows least-squares convention:
            # final_cost = 0.5 * SSR. Convert back to SSR for result-level chi2.
            chi2_values.append(2.0 * float(r.final_cost))
        else:
            chi2_values.append(0.0)
    chi2_per_angle = np.asarray(chi2_values, dtype=np.float64)
    ssr = float(chi2_per_angle.sum())
    n_function_evals = int(sum(int(r.n_function_evals or 0) for r in per_angle_results))
    n_iterations_total = int(sum(int(r.n_iterations or 0) for r in per_angle_results))

    c2_arr = np.asarray(c2_data)
    if c2_arr.ndim == 3:
        n_data_total = int(c2_arr.shape[0] * c2_arr.shape[1] * c2_arr.shape[2])
    else:
        n_data_total = int(c2_arr.size)
    dof = max(n_data_total - total_dim, 1)
    reduced_chi2 = ssr / dof

    # ------------------------------------------------------------------
    # Convergence + quality
    # ------------------------------------------------------------------
    n_success = int(sum(bool(r.success) for r in per_angle_results))
    all_converged = n_success == n_phi
    convergence_status: ConvergenceStatus = "converged" if all_converged else "partial"
    quality_flag = classify_quality_flag(reduced_chi2=reduced_chi2)
    if not all_converged and quality_flag == "good":
        # Mixed-success aggregate should not advertise good quality even
        # when reduced_chi2 happens to land in the green band.
        quality_flag = "marginal"

    # Per-angle metadata (optimizer markers, CMA-ES winner labels, etc.)
    # is preserved so downstream consumers can audit which solver actually
    # ran per angle without keeping the raw NLSQResult list around.
    per_angle_metadata = [dict(r.metadata) for r in per_angle_results]
    per_angle_messages = [str(r.message) for r in per_angle_results]
    per_angle_success = np.asarray([bool(r.success) for r in per_angle_results], dtype=bool)

    # L2 hierarchical: no-op for individual mode. Each per-angle fit
    # already runs with scaling held fixed at the model's pre-computed
    # value (the per-angle equivalent of stage 1); a second joint refine
    # across angles is precisely what individual mode declines to do, so
    # there is no stage 2. We surface the flag in diagnostics so callers
    # can confirm the request was observed.
    hierarchical_extras: dict[str, Any] = {}
    if config is not None and config.enable_hierarchical:
        hierarchical_extras = {
            "hierarchical_stages": 1,
            "hierarchical_active": False,
            "hierarchical_scope": "individual_mode_no_stage2",
        }

    diagnostics = _build_heterodyne_diagnostics(
        per_angle_mode="individual",
        chi2_per_angle=chi2_per_angle,
        scaling_source="fixed_per_angle",
        fourier_basis_dim=None,
        covariance_structure="block_diagonal_sequential",
        parameter_names=varying_names,
        phi_angles=np.asarray(phi_angles, dtype=np.float64),
        contrast_per_angle=contrast_per_angle,
        offset_per_angle=offset_per_angle,
        physics_per_angle=physics_per_angle,
        n_phi_total=n_phi,
        n_phi_success=n_success,
        physics_aggregation="mean",
        n_function_evals=n_function_evals,
        n_iterations=n_iterations_total,
        wall_time_seconds=float(wall_time),
        per_angle_metadata=per_angle_metadata,
        per_angle_messages=per_angle_messages,
        per_angle_success=per_angle_success,
        **hierarchical_extras,
    )

    return OptimizationResult(
        parameters=aggregated_params,
        uncertainties=uncertainties,
        covariance=covariance,
        chi_squared=ssr,
        reduced_chi_squared=reduced_chi2,
        convergence_status=convergence_status,
        iterations=n_iterations_total,
        execution_time=float(wall_time),
        device_info={"backend": "cpu", "adapter": "sequential_per_angle"},
        recovery_actions=[],
        quality_flag=quality_flag,
        streaming_diagnostics=None,
        stratification_diagnostics=None,
        nlsq_diagnostics=diagnostics,
    )


def fit_nlsq_multi_phi(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: list[float] | np.ndarray,
    config: NLSQConfig | None = None,
    weights: np.ndarray | None = None,
) -> OptimizationResult:
    """Fit heterodyne model to multi-phi correlation data.

    Dispatches to a joint-fit path when ``config`` is supplied and
    ``len(phi_angles) > 1``; otherwise falls through to the sequential
    per-angle chain. **Every dispatch branch returns a single**
    :class:`OptimizationResult` with per-angle data living in
    ``result.nlsq_diagnostics`` (see
    :mod:`xpcsjax.optimization.nlsq.heterodyne_views` for the post-hoc
    reconstruction helpers ``reconstruct_per_angle_scaling`` and
    ``per_angle_chi2``).

    Dispatch table (driven by ``config.per_angle_mode`` after
    ``auto``-resolution by :func:`_resolve_effective_mode`):

    - ``"constant"`` → :func:`_fit_joint_constant_multi_phi`
      → :class:`OptimizationResult`
    - ``"averaged"`` → :func:`_fit_joint_averaged_multi_phi`
      → :class:`OptimizationResult`
    - ``"fourier"`` → :func:`_fit_joint_multi_phi`
      → :class:`OptimizationResult`
    - ``"individual"`` (explicit, multi-angle) → :func:`_fit_joint_multi_phi`
      via a ``FourierReparameterizer`` in ``"independent"`` mode (JOINT fit of
      ``[physics | 2*n_phi per-angle scaling]``, matching ``laminar_flow`` and
      upstream heterodyne) → :class:`OptimizationResult`
    - ``enable_cmaes=True`` → :func:`_fit_joint_cmaes_multi_phi`
      → :class:`OptimizationResult`
    - ``multistart=True`` → :func:`_fit_joint_multistart`
      → :class:`OptimizationResult`
    - ``config is None`` / single-angle (``len(phi_angles) <= 1``)
      → sequential per-angle warm-start chain, aggregated into one
      :class:`OptimizationResult` via :func:`_aggregate_individual_results`.

    The sequential-aggregate fallback result uses a **block-diagonal**
    covariance matrix: off-diagonal blocks between physics and the
    per-angle scaling tail (and between distinct angles) are zero **by
    construction**, not by fit. The diagnostic key
    ``covariance_structure="block_diagonal_sequential"`` flags this so
    downstream consumers do not mistake the zeros for fit-derived
    correlation estimates. The JOINT individual path (explicit, multi-angle)
    does NOT carry this key — it returns a real fitted covariance.

    Parameters
    ----------
    model : HeterodyneModel
        HeterodyneModel instance with parameters configured.
    c2_data : np.ndarray
        Correlation data, shape ``(n_phi, N, N)`` or ``(N, N)``.
    phi_angles : list[float] | np.ndarray
        Array of phi angles (degrees).
    config : NLSQConfig | None
        NLSQ configuration. When ``None`` the sequential per-angle
        fallback runs.
    weights : np.ndarray | None
        Optional weights, shape ``(n_phi, N, N)`` or ``(N, N)``.

    Returns
    -------
    OptimizationResult
        Joint-fit result (constant / averaged / fourier / CMA-ES paths)
        or sequential-aggregate result (individual / no-config /
        single-angle fallback). All branches share the unified shape;
        callers may dispatch on ``result.nlsq_diagnostics["per_angle_mode"]``
        for mode-specific post-processing.

    Notes
    -----
    The global escapes (CMA-ES, multistart) are seed-pinned and therefore
    reproducible **per fresh model**: their warm-start ``x0`` reads the stateful
    ``model.scaling`` (mutated by every prior fit), so "same seed → same result"
    holds for the same inputs on a freshly constructed :class:`HeterodyneModel`,
    not across repeated fits that reuse (and mutate) one model instance.
    """
    phi_angles = np.asarray(phi_angles)

    if c2_data.ndim == 2:
        c2_data = c2_data[np.newaxis, ...]

    if len(c2_data) != len(phi_angles):
        raise ValueError(
            f"Number of c2 matrices ({len(c2_data)}) doesn't match "
            f"number of phi angles ({len(phi_angles)})"
        )

    # ------------------------------------------------------------------
    # Determine whether to use homodyne-style joint multi-angle fitting.
    # ------------------------------------------------------------------
    use_joint = False
    # Pre-initialize so the ``if use_joint:`` branch below sees a bound name
    # even when the optional fourier_reparam import fails. ``use_joint`` is
    # only flipped True inside the try block where ``fourier`` is reassigned,
    # so this initial None is never actually consumed at runtime.
    fourier: Any = None
    if config is not None and len(phi_angles) > 1:
        # Resolve ``auto`` / explicit modes to a canonical dispatch token FIRST.
        # The resolver returns one of: "constant", "averaged", "fourier",
        # "individual". Resolving before the global-escape gate is what keeps the
        # scaling layout consistent: enabling CMA-ES / multistart must NOT change
        # which layout is used. The escapes honour ``effective_mode`` — the
        # ``auto → averaged`` default (and explicit ``constant``) run their own
        # ``[physics | scaling]`` global search instead of collapsing to Fourier.
        # Keeping the table explicit makes the threshold semantics testable in
        # isolation — see tests/optimization/test_heterodyne_modes.py.
        effective_mode = _resolve_effective_mode(config, len(phi_angles))

        # Global-escape gate. CMA-ES takes priority over multistart (matching the
        # per-angle ``_try_global_optimization`` ordering). ``escape_kind`` is
        # None when no global method is configured/available → plain dispatch.
        escape_kind: str | None = None
        if getattr(config, "enable_cmaes", False) and HAS_CMAES:
            escape_kind = "cmaes"
        elif getattr(config, "multistart", False) and HAS_JOINT_MULTISTART:
            escape_kind = "multistart"

        logger.info(
            "Per-angle dispatch: requested=%s, n_phi=%d, constant_threshold=%d, "
            "fourier_threshold=%d, effective=%s, escape=%s",
            config.per_angle_mode,
            len(phi_angles),
            config.constant_scaling_threshold,
            config.fourier_auto_threshold,
            effective_mode,
            escape_kind,
        )

        # The Fourier / individual escapes keep the existing Fourier-reparam
        # joint-problem builder (``_build_joint_problem``), which already
        # represents both layouts (``"fourier"`` / ``"independent"``) correctly.
        if escape_kind is not None and effective_mode in ("individual", "fourier"):
            if escape_kind == "cmaes":
                logger.info("CMA-ES enabled, delegating to joint multi-angle CMA-ES")
                return _fit_joint_cmaes_multi_phi(
                    model=model,
                    c2_data=c2_data,
                    phi_angles=phi_angles,
                    config=config,
                    weights=weights,
                )
            logger.info("Multistart enabled, delegating to joint multi-angle multistart")
            return _fit_joint_multistart(
                model=model,
                c2_data=c2_data,
                phi_angles=phi_angles,
                config=config,
                weights=weights,
                use_nlsq_library=True,
            )

        if effective_mode == "constant":
            # Lazy import: keeps the heterodyne_constant_mode module out of
            # heterodyne_core's namespace so ``hasattr(heterodyne_core,
            # '_fit_joint_constant_multi_phi')`` stays False (the Sub-PR A3
            # contract — the function lives in its own module, not here).
            from xpcsjax.optimization.nlsq.heterodyne_constant_mode import (
                _fit_joint_constant_multi_phi,
            )

            return _fit_joint_constant_multi_phi(
                model=model,
                c2_data=c2_data,
                phi_angles=phi_angles,
                config=config,
                weights=weights,
                global_escape_kind=escape_kind,
            )

        if effective_mode == "averaged":
            return _fit_joint_averaged_multi_phi(
                model=model,
                c2_data=c2_data,
                phi_angles=phi_angles,
                config=config,
                weights=weights,
                global_escape_kind=escape_kind,
            )

        if effective_mode == "fourier":
            try:
                from xpcsjax.optimization.nlsq.fourier_reparam import (
                    FourierReparamConfig,
                    FourierReparameterizer,
                )

                # NOTE: ``FourierReparamConfig.mode`` is typed as
                # ``Literal["independent", "fourier", "auto"]`` — a narrower
                # vocabulary than heterodyne's ``per_angle_mode``
                # (``"individual" | "fourier" | "auto" | "constant" |
                # "independent"``). We reach this branch only when the
                # resolver returned ``"fourier"``, so passing the literal
                # ``"fourier"`` is correct and silences the Pyright
                # incompatibility flagged since A1. The
                # ``FourierReparameterizer`` re-runs the auto/feasibility
                # check via ``_determine_mode`` and falls back to
                # ``independent`` internally if ``n_phi`` is too small for
                # the requested order — so we do not lose the auto-fallback
                # behaviour by pinning the string here.
                fourier_config = FourierReparamConfig(
                    mode="fourier",
                    fourier_order=config.fourier_order,
                    auto_threshold=config.fourier_auto_threshold,
                )
                phi_rad = np.deg2rad(phi_angles.astype(np.float64))
                fourier = FourierReparameterizer(phi_rad, fourier_config)
                use_joint = True
            except ImportError:
                logger.warning("fourier_reparam not available, falling back to sequential fits")

        elif effective_mode == "individual":
            # Explicit multi-angle ``individual`` is a JOINT fit (parity with
            # xpcsjax ``laminar_flow`` and upstream heterodyne). The per-angle
            # (contrast, offset) are packed as the ``2*n_phi`` scaling tail of
            # the joint vector ``[physics | contrast_0..N | offset_0..N]`` and
            # optimized jointly with physics via ``_fit_joint_multi_phi``,
            # exactly like the fourier branch — only the reparameterizer mode
            # differs (``"independent"`` = identity passthrough, no Fourier
            # basis). This replaces the old sequential-per-angle aggregate
            # (``mean(physics)`` reported as ``parameters``), which was an
            # inconsistent estimator whose parameters did not reproduce the
            # reported chi-squared. The sequential aggregate
            # (``_aggregate_individual_results``) survives ONLY as the
            # genuine fallback for ``config is None`` / single-angle
            # (``len(phi_angles) <= 1``) — both handled by this block's guard
            # (``config is not None and len(phi_angles) > 1``) being false.
            try:
                from xpcsjax.optimization.nlsq.fourier_reparam import (
                    FourierReparamConfig,
                    FourierReparameterizer,
                )

                # ``"independent"`` makes ``FourierReparameterizer`` an
                # identity passthrough: ``fourier_to_per_angle_jax`` returns
                # ``coeffs[:n_phi], coeffs[n_phi:]`` and ``get_bounds`` /
                # ``n_coeffs`` describe the ``2*n_phi`` per-angle layout — so
                # ``_fit_joint_multi_phi`` solves the individual problem with
                # no Fourier-only assumptions.
                fourier_config = FourierReparamConfig(
                    mode="independent",
                    fourier_order=config.fourier_order,
                    auto_threshold=config.fourier_auto_threshold,
                )
                phi_rad = np.deg2rad(phi_angles.astype(np.float64))
                fourier = FourierReparameterizer(phi_rad, fourier_config)
                use_joint = True
            except ImportError:
                logger.warning(
                    "fourier_reparam not available, falling back to sequential "
                    "individual fits"
                )

        # ``config is None`` / single-angle (len(phi_angles) <= 1) never reach
        # this block — they fall through to the sequential per-angle aggregate
        # below, the genuine individual-mode fallback.

    if use_joint:
        # Invariant: ``use_joint`` is only set to True inside the
        # ``if config is not None and len(phi_angles) > 1`` block above,
        # so config is guaranteed non-None here. mypy can't see the implicit
        # invariant — assert it for the type checker and as a belt-and-
        # suspenders runtime check.
        assert config is not None, "use_joint=True only when config is non-None"
        return _fit_joint_multi_phi(
            model,
            c2_data,
            phi_angles,
            config,
            weights,
            fourier,
        )

    # ------------------------------------------------------------------
    # Sequential per-angle fitting (warm-start chain)
    # ------------------------------------------------------------------
    t_seq_start = time.perf_counter()
    per_angle_results: list[NLSQResult] = []
    for i, phi in enumerate(phi_angles):
        if i > 0:
            logger.info(
                "Fitting phi angle %d/%d: %s° (warm-start from angle %s°)",
                i + 1,
                len(phi_angles),
                phi,
                phi_angles[i - 1],
            )
        else:
            logger.info("Fitting phi angle %d/%d: %s°", i + 1, len(phi_angles), phi)

        c2_i = c2_data[i]
        weights_i = weights[i] if weights is not None and weights.ndim == 3 else weights

        result = fit_nlsq_jax(
            model=model,
            c2_data=c2_i,
            phi_angle=float(phi),
            config=config,
            weights=weights_i,
            angle_idx=i,
        )
        result.metadata["phi_angle"] = float(phi)
        per_angle_results.append(result)

    return _aggregate_individual_results(
        per_angle_results=per_angle_results,
        model=model,
        phi_angles=phi_angles,
        c2_data=c2_data,
        wall_time=time.perf_counter() - t_seq_start,
        config=config,
        weights=weights,
    )


def _compute_per_angle_chi2(
    residuals: np.ndarray,
    c2_matrix: np.ndarray,
    n_params: int,
) -> tuple[float, float]:
    """Compute per-angle cost and noise-normalised reduced chi-squared.

    Joint fits produce one aggregated cost and chi2 for all angles. This
    helper reconstructs the per-angle statistics so each NLSQResult carries
    its own diagnostics rather than a copy of the joint value.

    Args:
        residuals: Flat off-diagonal residual vector from compute_residuals,
            length n*(n-1).
        c2_matrix: Per-angle experimental C2 matrix, shape (n, n).
        n_params: Number of varying physics parameters.

    Returns:
        ``(per_angle_cost, reduced_chi_squared)`` where ``per_angle_cost``
        is ``0.5*SSR`` and ``reduced_chi_squared`` is noise-normalised
        (target ≈ 1.0 for a good fit; MSE fallback when noise is degenerate).
    """
    ssr = float(np.sum(residuals**2))
    per_angle_cost = 0.5 * ssr

    n_matrix = c2_matrix.shape[0]
    n_valid = c2_matrix.size - n_matrix  # off-diagonal count (matches residuals length)
    n_dof = max(n_valid - n_params, 1)

    # Far-lag photon-noise estimate — same formula as _fit_local
    c2_np = np.asarray(c2_matrix)
    row_idx = np.arange(n_matrix)
    lag_mat = np.abs(row_idx[:, None] - row_idx[None, :])
    far_vals = c2_np[lag_mat >= n_matrix // 2]
    sigma2_noise = float(np.var(far_vals)) if far_vals.size > 1 else 0.0

    if sigma2_noise > 1e-12:
        reduced_chi2 = ssr / (sigma2_noise * n_dof)
    else:
        reduced_chi2 = ssr / n_dof  # MSE fallback

    return per_angle_cost, reduced_chi2


def _fit_joint_averaged_multi_phi(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: np.ndarray,
    config: NLSQConfig,
    weights: np.ndarray | None,
    *,
    global_escape_kind: str | None = None,
) -> OptimizationResult:
    """Joint multi-angle fit with averaged contrast/offset scaling.

    When ``global_escape_kind`` is ``"cmaes"`` or ``"multistart"`` the plain
    NLSQ solve below is used as the warm-start, a seed-pinned global search is
    run over the SAME ``[physics | avg_contrast, avg_offset]`` data residual,
    and the better (lower data-only SSR) vector is kept. This honours the
    ``auto → averaged`` default under CMA-ES / multistart instead of collapsing
    to the Fourier layout — matching the plain dispatch and laminar_flow's
    CMA-ES. An escape result carries ``nlsq_diagnostics["global_escape"]`` and,
    by the escape contract, NaN covariance / uncertainties and
    ``n_iterations=0`` (no covariance solve on the kept vector).

    Implements homodyne's `auto`-averaged anti-degeneracy path:
    per-angle quantile estimates are computed first, averaged to one contrast
    and one offset, and those two scaling parameters are optimized jointly
    with the physical model parameters.

    NOTE: despite the legacy filename overlap, this is NOT homodyne's `constant`
    mode. True `constant` mode (quantile estimates pre-fit and frozen) is
    implemented by `fit_joint_constant_multi_phi` (Sub-PR B), defined in
    `heterodyne_constant_mode.py`.

    Returns
    -------
    OptimizationResult
        One result for the entire joint solve. ``parameters`` has the
        ``physics_varying + [avg_contrast, avg_offset]`` layout (2 scaling
        params). Per-angle diagnostics — ``chi2_per_angle``,
        ``per_angle_mode='averaged'``, ``scaling_source='averaged_then_fitted'``,
        ``fourier_basis_dim=None``, ``shear_weighting='not_applicable_heterodyne'``
        — live in ``nlsq_diagnostics``, alongside the ``averaged_contrast`` /
        ``averaged_offset`` scalar extras. Mirrors the contract of
        :func:`_fit_joint_multi_phi` (Sub-PR C2) and
        :func:`xpcsjax.optimization.nlsq.heterodyne_constant_mode._fit_joint_constant_multi_phi`
        (Sub-PR B2).
    """
    from xpcsjax.config.parameter_registry import SCALING_PARAMS
    from xpcsjax.core.heterodyne_scaling_utils import compute_averaged_scaling

    t_start = time.perf_counter()

    param_manager = model.param_manager
    varying_names = list(param_manager.varying_names)
    n_physics_varying = param_manager.n_varying
    n_phi = len(phi_angles)

    physics_initial = np.asarray(param_manager.get_initial_values(), dtype=np.float64)
    physics_lower, physics_upper = param_manager.get_bounds()
    physics_initial = np.clip(physics_initial, physics_lower, physics_upper)

    # ------------------------------------------------------------------
    # L2 hierarchical two-stage: Stage 1 — physics-only solve with
    # quantile-fixed scaling (delegates to the constant-mode solver).
    # When `config.enable_hierarchical` is True we run the constant-mode
    # solver first to converge the physics block with scaling frozen,
    # then warm-start the joint solve below by overriding `physics_initial`
    # with the converged physics vector. See L2 docs in `_fit_joint_multi_phi`.
    # ------------------------------------------------------------------
    hierarchical_stage1_chi2: float | None = None
    if config.enable_hierarchical:
        logger.info(
            "L2 hierarchical (averaged mode) — Stage 1: physics-only solve "
            "with quantile-fixed scaling"
        )
        # Lazy import keeps the module out of heterodyne_core's namespace
        # except when explicitly used (consistent with the dispatch table).
        from xpcsjax.optimization.nlsq.heterodyne_constant_mode import (
            _fit_joint_constant_multi_phi,
        )

        stage1_result = _fit_joint_constant_multi_phi(
            model=model,
            c2_data=c2_data,
            phi_angles=phi_angles,
            config=config,
            weights=weights,
        )
        stage1_physics = np.asarray(stage1_result.parameters, dtype=np.float64)
        hierarchical_stage1_chi2 = float(stage1_result.chi_squared)
        # Override the initial physics vector for stage 2 (joint refine).
        # Clip to bounds defensively — stage 1 should already respect them,
        # but a constant-mode bound contraction is possible if config differs.
        physics_initial = np.clip(stage1_physics, physics_lower, physics_upper)
        logger.info(
            "L2 hierarchical (averaged mode) — Stage 1 done: chi2=%.6f, "
            "warm-starting stage 2 joint refine",
            hierarchical_stage1_chi2,
        )

    t = model.t
    q = model.q
    dt = model.dt

    t1_mesh, t2_mesh = np.meshgrid(np.asarray(t), np.asarray(t), indexing="ij")
    n_time_points = t1_mesh.size
    c2_flat = []
    t1_flat = []
    t2_flat = []
    phi_indices = []
    for i in range(n_phi):
        c2_flat.append(np.asarray(c2_data[i], dtype=np.float64).reshape(-1))
        t1_flat.append(t1_mesh.reshape(-1))
        t2_flat.append(t2_mesh.reshape(-1))
        phi_indices.append(np.full(n_time_points, i, dtype=np.int32))

    contrast_bounds = (
        SCALING_PARAMS["contrast"].min_bound,
        SCALING_PARAMS["contrast"].max_bound,
    )
    offset_bounds = (
        SCALING_PARAMS["offset"].min_bound,
        SCALING_PARAMS["offset"].max_bound,
    )

    logger.info("=" * 60)
    logger.info("AUTO AVERAGED SCALING: Computing per-angle scaling from quantiles")
    logger.info("=" * 60)
    avg_contrast, avg_offset, contrast_per_angle, offset_per_angle = compute_averaged_scaling(
        c2_data=np.concatenate(c2_flat),
        t1=np.concatenate(t1_flat),
        t2=np.concatenate(t2_flat),
        phi_indices=np.concatenate(phi_indices),
        n_phi=n_phi,
        contrast_bounds=contrast_bounds,
        offset_bounds=offset_bounds,
        log=logger,
    )

    x0 = np.concatenate([physics_initial, [avg_contrast, avg_offset]])
    lb = np.concatenate([physics_lower, [contrast_bounds[0], offset_bounds[0]]])
    ub = np.concatenate([physics_upper, [contrast_bounds[1], offset_bounds[1]]])
    joint_param_names = [*varying_names, "contrast", "offset"]

    logger.info(
        "Joint auto averaged fit: %d physical + 2 averaged scaling = %d total params, %d angles",
        n_physics_varying,
        len(x0),
        n_phi,
    )

    c2_data_batch = jnp.asarray(c2_data, dtype=jnp.float64)
    weights_batch = (
        jnp.asarray(weights, dtype=jnp.float64)
        if weights is not None
        else jnp.ones_like(c2_data_batch)
    )
    if weights_batch.ndim == 2:
        weights_batch = jnp.broadcast_to(weights_batch, c2_data_batch.shape)
    phi_angles_jax = jnp.asarray(phi_angles, dtype=jnp.float64)
    fixed_values_jax = jnp.asarray(param_manager.get_full_values(), dtype=jnp.float64)
    varying_indices_jax = jnp.array(param_manager.varying_indices, dtype=jnp.int32)

    # NOTE: must return a JAX array. NLSQ's masked_residual_func JIT-traces this
    # closure; np.asarray() on a traced result raises TracerArrayConversionError.
    def base_residual_fn(x: np.ndarray) -> Any:  # type: ignore[return-value]
        physics_varying = x[:n_physics_varying]
        contrast = x[n_physics_varying]
        offset = x[n_physics_varying + 1]

        full_jax = fixed_values_jax.at[varying_indices_jax].set(
            jnp.asarray(physics_varying, dtype=jnp.float64)
        )
        contrasts_jax = jnp.full((n_phi,), contrast, dtype=jnp.float64)
        offsets_jax = jnp.full((n_phi,), offset, dtype=jnp.float64)
        return compute_multi_angle_residuals(
            full_jax,
            t,
            q,
            dt,
            phi_angles_jax,
            c2_data_batch,
            weights_batch,
            contrasts_jax,
            offsets_jax,
        )

    # ------------------------------------------------------------------
    # L3 anti-degeneracy: wrap base residual with adaptive regularization.
    # Averaged mode collapses per-angle scaling to a SINGLE (contrast,
    # offset) pair, so per-angle CV is undefined (group size 1, std = 0).
    # The AdaptiveRegularizer's relative/CV branch is therefore a no-op
    # here; we still record the wiring as active and append two zero
    # penalty rows (preserving the contract that
    # ``regularization_penalty_count`` reflects the n_groups penalty rows
    # in the augmented residual) so behavioural-mode parity with the
    # fourier-mode path is preserved.
    # ------------------------------------------------------------------
    regularization_active = config.regularization_mode != "none"
    n_penalty_rows = 0
    if regularization_active:
        from xpcsjax.optimization.nlsq.adaptive_regularization import (
            AdaptiveRegularizationConfig,
            AdaptiveRegularizer,
        )

        reg_mode_jax: Any = "relative" if config.regularization_mode == "adaptive" else "absolute"
        reg_config = AdaptiveRegularizationConfig(
            enable=True,
            mode=reg_mode_jax,
            lambda_base=float(config.group_variance_lambda),
            target_cv=float(config.regularization_target_cv),
            auto_tune_lambda=False,
        )
        regularizer = AdaptiveRegularizer(reg_config, n_phi=n_phi, n_params=len(x0))
        n_penalty_rows = len(regularizer.group_indices)
        sqrt_lambda = float(np.sqrt(float(regularizer.lambda_value)))

        # ``sqrt_lambda`` is captured by reference so the diagnostic value
        # is still tied to the configured lambda; the penalty contribution
        # itself is degenerate-zero by construction (see comment above).
        _sqrt_lambda_capture = sqrt_lambda
        _n_penalty_rows_capture = n_penalty_rows

        def joint_residual_fn(x: np.ndarray) -> Any:  # type: ignore[return-value]
            r = base_residual_fn(x)
            # In averaged mode each "group" has a single scaling scalar, so
            # std = 0 → penalty contribution is exactly zero. We still emit
            # K rows of zeros so the augmented residual length is
            # ``n_data + K`` (the K-row contract). The optimizer therefore
            # sees the same objective ``||r_data||²``; this is the correct
            # degenerate-CV behaviour for the auto_averaged scaling layout.
            # ``_sqrt_lambda_capture`` is read to keep it in the closure
            # (Pyright unused-variable suppression).
            penalty_rows = jnp.zeros(_n_penalty_rows_capture, dtype=jnp.float64) * jnp.float64(
                _sqrt_lambda_capture
            )
            return jnp.concatenate([r, penalty_rows])
    else:
        joint_residual_fn = base_residual_fn  # type: ignore[assignment]

    # max_nfev is multiplied by n_phi here because the joint solve packs
    # all angles into a single residual vector; the per-angle budget
    # documented on NLSQConfig.max_nfev is preserved by scaling the
    # combined cap. See NLSQConfig.max_nfev docstring for the contract.
    joint_config = NLSQConfig(
        method=config.method if config.method != "lm" else "trf",
        ftol=config.ftol,
        xtol=config.xtol,
        gtol=config.gtol,
        max_nfev=(config.max_nfev * n_phi if config.max_nfev is not None else None),
        loss=config.loss,
        use_nlsq_library=config.use_nlsq_library,
        n_params=len(x0),
    )

    # L4: per-iteration gradient-collapse monitor (strictly observational).
    # See ``_build_l4_callback`` — returns (None, None) when disabled.
    _monitor, _l4_callback = _build_l4_callback(model, x0, joint_residual_fn, config)

    joint_result: NLSQResult | None = None
    # Tracks whether the RETURNED ``joint_result`` came from the monitored
    # adapter (the only backend the L4 callback is wired into). Stays False on
    # the unmonitored NLSQWrapper fallback path so _assemble_l4_extras does not
    # surface a stale per-iteration monitor against the wrapper's parameters.
    used_monitored_backend = False
    # Narrow via ``is not None`` instead of the HAS_X flag so Pyright sees
    # NLSQAdapter as bound. HAS_ADAPTERS is True iff NLSQAdapter was imported,
    # so the two predicates are equivalent at runtime.
    if NLSQAdapter is not None:
        try:
            joint_adapter = NLSQAdapter(parameter_names=joint_param_names)
            joint_result = joint_adapter.fit(
                residual_fn=joint_residual_fn,
                initial_params=x0,
                bounds=(lb, ub),
                config=joint_config,
                callback=_l4_callback,
            )
            if not joint_result.success:
                raise RuntimeError(f"Joint adapter returned success=False: {joint_result.message}")
            # Adapter succeeded → the returned result IS the monitored run.
            used_monitored_backend = True
        except (ValueError, RuntimeError, TypeError) as adapter_exc:
            logger.warning(
                "Joint auto averaged NLSQAdapter failed, falling back to NLSQWrapper: %s",
                adapter_exc,
            )
            joint_result = None

    if joint_result is None and NLSQWrapper is not None:
        joint_wrapper = NLSQWrapper(parameter_names=joint_param_names)
        joint_result = joint_wrapper.fit(
            residual_fn=joint_residual_fn,
            initial_params=x0,
            bounds=(lb, ub),
            config=joint_config,
        )

    if joint_result is None:
        raise ImportError("No NLSQ backend available for joint auto averaged multi-angle fit.")

    fitted_all = np.asarray(joint_result.parameters, dtype=np.float64)

    # Global escape (CMA-ES / multistart): warm-started at the plain solve,
    # keep-better over the SAME averaged data residual. ``global_escape_tag`` is
    # None on the plain path (no behaviour change) or when the search failed.
    fitted_all, global_escape_tag = _apply_global_escape(
        global_escape_kind,
        base_residual_fn,
        fitted_all,
        lb,
        ub,
        joint_config,
        joint_param_names,
        config,
        {"c2": c2_data, "phi": phi_angles},
    )
    is_escape = global_escape_tag is not None

    fitted_physics = fitted_all[:n_physics_varying]
    fitted_contrast = float(fitted_all[n_physics_varying])
    fitted_offset = float(fitted_all[n_physics_varying + 1])

    full_fitted = param_manager.expand_varying_to_full(fitted_physics)
    model.set_params(full_fitted)
    if hasattr(model, "scaling"):
        model.scaling.contrast[:] = fitted_contrast
        model.scaling.offset[:] = fitted_offset

    wall_time = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # Decompose per-angle chi^2 from the final residual.
    # ``compute_multi_angle_residuals`` returns an angle-major flat layout
    # (n_phi, n_per_angle) — n_per_angle = (n_time - 1) * (n_time - 2) because
    # the kernel excludes the diagonal AND the t=0 boundary row/col. Re-use the canonical helper from
    # heterodyne_constant_mode (same import the Fourier-mode joint path uses).
    # ------------------------------------------------------------------
    from xpcsjax.optimization.nlsq.heterodyne_constant_mode import (
        _decompose_chi2_per_angle,
    )

    # SSR conservation: decompose chi^2 on the *data-only* residual
    # (excluding any L3 penalty rows). See _fit_joint_multi_phi for the
    # same pattern.
    data_only_residual = np.asarray(base_residual_fn(fitted_all))
    n_time = c2_data.shape[1]
    n_per_angle = (n_time - 1) * (n_time - 2)  # off-diag, t=0 boundary excluded — matches kernel
    chi2_per_angle = _decompose_chi2_per_angle(
        final_residual=data_only_residual,
        n_phi=n_phi,
        n_per_angle=n_per_angle,
    )

    # ------------------------------------------------------------------
    # Build the single joint OptimizationResult.
    # SSR conservation: ``chi_squared`` is the *data-only* SSR, not
    # ``2 * nlsq_result.final_cost`` (which is the robust-loss cost when
    # ``config.loss != "linear"``). Using raw data residuals keeps
    # ``chi2_per_angle.sum() == chi_squared`` for every loss choice and
    # every regularization mode — the same invariant B2 / C2 locked in
    # for the other joint paths.
    # ------------------------------------------------------------------
    data_only_ssr = float(np.sum(data_only_residual**2))
    ssr = data_only_ssr
    # Full residual (including any penalty rows) — diagnostic only.
    final_residual = np.asarray(joint_residual_fn(fitted_all))
    total_ssr_with_penalty = float(np.sum(final_residual**2))
    n_total_params = int(joint_result.parameters.size)
    # Noise-normalised reduced chi^2 (targets ~1.0). The raw
    # ``joint_result.reduced_chi_squared`` is SSR/N² — i.e. MSE ≪ 1 on
    # normalised C2 data (C2 ~ 1, residuals ~ 5%) — which is not an
    # interpretable goodness-of-fit. Apply the same far-lag photon-noise
    # correction the single-angle / per-angle paths use so every heterodyne
    # path reports a comparable metric. ``chi_squared`` (= ssr) and
    # ``chi2_per_angle`` are left untouched, so the SSR-conservation invariant
    # (``chi2_per_angle.sum() == chi_squared``) still holds.
    from xpcsjax.optimization.nlsq.heterodyne_data_prep import (
        noise_normalized_reduced_chi2,
    )

    reduced_chi2 = noise_normalized_reduced_chi2(
        ssr=ssr,
        c2_data=c2_data,
        n_data_valid=int(data_only_residual.size),
        n_params=n_total_params,
    )

    # NaN-fill uncertainties / covariance when the NLSQ adapter could not
    # produce them (e.g. singular Jacobian after a non-converged solve) —
    # matches B2 / C2's contract so consumers see a uniform array shape.
    # Escape contract (mirrors ``_build_joint_result``): a kept global-escape
    # vector has no covariance solve, so uncertainties / covariance are NaN.
    if is_escape:
        uncertainties = np.full(n_total_params, np.nan, dtype=np.float64)
        covariance = np.full((n_total_params, n_total_params), np.nan, dtype=np.float64)
    else:
        uncertainties = (
            np.asarray(joint_result.uncertainties, dtype=np.float64)
            if joint_result.uncertainties is not None
            else np.full(n_total_params, np.nan, dtype=np.float64)
        )
        covariance = (
            np.asarray(joint_result.covariance, dtype=np.float64)
            if joint_result.covariance is not None
            else np.full((n_total_params, n_total_params), np.nan, dtype=np.float64)
        )

    convergence_status: ConvergenceStatus = "converged" if joint_result.success else "failed"
    quality_flag: QualityFlag = "good" if joint_result.success else "marginal"

    # ------------------------------------------------------------------
    # L2 anti-degeneracy: hierarchical two-stage solve.
    #
    # Stage 1 (physics-only with quantile-fixed scaling) ran above —
    # before the joint solve — when `config.enable_hierarchical` was True,
    # producing `hierarchical_stage1_chi2` and a warm-started
    # `physics_initial`. Stage 2 is the joint refine the surrounding code
    # already executed (scaling unfrozen, jointly fit with physics).
    #
    # SSR conservation invariant (`chi2_per_angle.sum() == chi_squared`)
    # still holds for stage 2 because the joint solve uses the canonical
    # multi-angle residual decomposition.
    # ------------------------------------------------------------------
    hierarchical_extras: dict[str, Any] = {}
    if config.enable_hierarchical and hierarchical_stage1_chi2 is not None:
        logger.info(
            "L2 hierarchical (averaged mode) — Stage 2 done: chi2=%.6f (stage1=%.6f)",
            ssr,
            hierarchical_stage1_chi2,
        )
        hierarchical_extras = {
            "hierarchical_stages": 2,
            "hierarchical_active": True,
            "hierarchical_scope": "full_two_stage",
            "hierarchical_stage1_chi2": hierarchical_stage1_chi2,
            "hierarchical_stage2_chi2": float(ssr),
        }

    # ------------------------------------------------------------------
    # L3 anti-degeneracy: adaptive CV regularization (full integration).
    # When ``config.regularization_mode != "none"`` the residual factory
    # above wrapped ``base_residual_fn`` with an L3 augmentation. In
    # averaged mode the per-group size is 1 (a single contrast and a
    # single offset scalar), so CV is degenerate-zero and the appended
    # penalty rows are themselves zero — the optimizer-visible objective
    # is therefore unchanged. The diagnostics still record the wiring as
    # active and the augmented residual still carries the K penalty rows
    # (the K-row contract) so behavioural-mode parity with fourier mode
    # is preserved.
    # ------------------------------------------------------------------
    regularization_extras: dict[str, Any] = {}
    if regularization_active:
        logger.info(
            "L3 adaptive regularization enabled (averaged mode): "
            "mode=%s, lambda=%.6g, target_cv=%.3f, penalty_rows=%d "
            "(degenerate-zero in averaged mode: group size 1).",
            config.regularization_mode,
            config.group_variance_lambda,
            config.regularization_target_cv,
            n_penalty_rows,
        )
        regularization_extras = {
            "regularization_active": True,
            "regularization_mode": config.regularization_mode,
            "regularization_lambda_applied": float(config.group_variance_lambda),
            "regularization_penalty_count": int(n_penalty_rows),
            "regularization_data_residual_ssr": data_only_ssr,
            "regularization_total_ssr_with_penalty": total_ssr_with_penalty,
            "regularization_scope": "full_residual_augmentation",
        }

    # ------------------------------------------------------------------
    # L4 anti-degeneracy: gradient collapse monitor (full integration).
    #
    # The monitor records the per-iteration physical/per-angle gradient ratio
    # via NLSQ's curve_fit callback (the strictly-observational mechanism built
    # in ``_build_l4_callback``). When the callback recorded zero observations,
    # ``_assemble_l4_extras`` falls back to the post-solve covariance-condition
    # block (the singular-value spectrum of ``cov ≈ (J^T J)^-1``, tagged
    # ``mechanism="post_solve_fallback"``).
    # ------------------------------------------------------------------
    gradient_monitor_extras = _assemble_l4_extras(
        _monitor,
        joint_result,
        config,
        mode_label="averaged mode",
        result_is_monitored=used_monitored_backend,
    )

    diagnostics = _build_heterodyne_diagnostics(
        per_angle_mode="averaged",
        chi2_per_angle=chi2_per_angle,
        scaling_source="averaged_then_fitted",
        fourier_basis_dim=None,
        averaged_contrast=fitted_contrast,
        averaged_offset=fitted_offset,
        parameter_names=joint_param_names,
        contrast_per_angle_quantile=np.asarray(contrast_per_angle, dtype=np.float64),
        offset_per_angle_quantile=np.asarray(offset_per_angle, dtype=np.float64),
        contrast_initial_average=float(avg_contrast),
        offset_initial_average=float(avg_offset),
        phi_angles=np.asarray(phi_angles, dtype=np.float64),
        n_angles_joint=n_phi,
        convergence_reason=("global_escape" if is_escape else joint_result.convergence_reason),
        n_function_evals=(0 if is_escape else int(joint_result.n_function_evals or 0)),
        n_iterations=(0 if is_escape else int(joint_result.n_iterations or 0)),
        wall_time_seconds=wall_time,
        message=("global escape" if is_escape else str(joint_result.message)),
        **hierarchical_extras,
        **regularization_extras,
        **gradient_monitor_extras,
    )
    # Tag a global-escape assembly so callers can distinguish it from a plain
    # joint fit (the plain fit leaves this key absent).
    if global_escape_tag is not None:
        diagnostics["global_escape"] = global_escape_tag

    logger.info(
        "Joint auto averaged fit complete: success=%s, cost=%.6f, "
        "n_evals=%d, wall_time=%.2fs, %d angles%s",
        joint_result.success,
        joint_result.final_cost or 0.0,
        joint_result.n_function_evals or 0,
        wall_time,
        n_phi,
        f" [escape={global_escape_tag}]" if is_escape else "",
    )

    return OptimizationResult(
        parameters=np.asarray(fitted_all, dtype=np.float64),
        uncertainties=uncertainties,
        covariance=covariance,
        chi_squared=ssr,
        reduced_chi_squared=reduced_chi2,
        convergence_status=convergence_status,
        iterations=(0 if is_escape else int(joint_result.n_iterations or 0)),
        execution_time=wall_time,
        device_info={"backend": "cpu", "adapter": "nlsq.CurveFit"},
        recovery_actions=[],
        quality_flag=quality_flag,
        streaming_diagnostics=None,
        stratification_diagnostics=None,
        nlsq_diagnostics=diagnostics,
    )


# Phase-6 minimal stub: delegates to the standard joint Fourier fit so the
# return shape is ``OptimizationResult`` (matches the constant/averaged/Fourier
# paths).  A real CMA-ES escape with NLSQ warm-start and Fourier-reparam
# Deterministic seed pinned on the joint CMA-ES escape. ``CMAESWrapperConfig``
# (and NLSQ's ``CMAESConfig``) default ``seed=None`` → non-reproducible; the
# escape MUST pin it so the global search is bit-reproducible run to run.
_JOINT_CMAES_SEED = 42

# Per-angle CMA-ES escape seed. Offset by ``angle_idx`` at the call site so each
# angle's stochastic search is individually reproducible yet decorrelated from
# the others (mirrors ``_JOINT_CMAES_SEED``'s pinning; a single shared seed would
# make every angle explore the identical random trajectory). Without this the
# per-angle ``_fit_cmaes`` path left ``CMAESWrapperConfig.seed=None`` →
# non-reproducible, unlike the seed-pinned joint escapes.
_PER_ANGLE_CMAES_SEED = 42


def _build_joint_fourier(
    config: NLSQConfig, phi_angles: np.ndarray
) -> Any:
    """Build the mode-appropriate :class:`FourierReparameterizer` for the joint fit.

    Mirrors the per-mode dispatch in :func:`fit_nlsq_multi_phi`: ``individual``
    resolves to an identity-passthrough (``"independent"``) reparameterizer
    (free ``2*n_phi`` per-angle scaling); any other resolved mode keeps the
    Fourier basis (the reparameterizer re-checks feasibility and degrades to
    independent internally if ``n_phi`` is too small).
    """
    from xpcsjax.optimization.nlsq.fourier_reparam import (
        FourierReparamConfig,
        FourierReparameterizer,
    )

    effective_mode = _resolve_effective_mode(config, len(np.asarray(phi_angles)))
    reparam_mode: Any = "independent" if effective_mode == "individual" else "fourier"
    fourier_config = FourierReparamConfig(
        mode=reparam_mode,
        fourier_order=config.fourier_order,
        auto_threshold=config.fourier_auto_threshold,
    )
    phi_rad = np.deg2rad(np.asarray(phi_angles).astype(np.float64))
    return FourierReparameterizer(phi_rad, fourier_config)


def _fit_joint_cmaes_multi_phi(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: np.ndarray,
    config: NLSQConfig,
    weights: np.ndarray | None,
) -> OptimizationResult:
    """Joint multi-angle CMA-ES escape — additive global search over the joint vector.

    Lifts heterodyne's proven PER-ANGLE pattern (:func:`_fit_cmaes`) to the
    joint multi-angle objective:

    1. **Warm-start** — run the plain joint fit (:func:`_fit_joint_multi_phi`)
       over the SAME :class:`JointProblem` to get a local optimum ``x_warm``.
    2. **Global search** — seed-pinned :func:`fit_with_cmaes` over the joint
       residual ``prob.joint_residual_fn`` (``model_func`` returns the residual,
       ``ydata`` is zeros, so CMA-ES minimises ``||residual||²`` directly).
    3. **Keep-better** — recompute the escape's data-only SSR at the CMA-ES
       optimum and keep CMA-ES only if it succeeded AND did not increase the
       SSR vs the warm-start; otherwise keep the warm-start vector. Either way
       the result carries a ``global_escape`` diagnostics tag.

    The plain joint fit is NOT modified — this path is reached only when
    ``config.enable_cmaes`` is True. On any failure the escape falls back to the
    plain joint fit (best-effort).
    """
    try:
        fourier = _build_joint_fourier(config, phi_angles)
        prob = _build_joint_problem(
            model, c2_data, phi_angles, config, weights, fourier=fourier
        )

        # Phase 1: warm-start via the plain joint fit over the SAME problem.
        warm = _fit_joint_multi_phi(
            model=model,
            c2_data=c2_data,
            phi_angles=np.asarray(phi_angles),
            config=config,
            weights=weights,
            fourier=prob.fourier,
        )
        x_warm = np.asarray(warm.parameters, dtype=np.float64)
        ssr_warm = float(warm.chi_squared)

        # Data-only SSR (excludes any L3 penalty rows) so the keep-better
        # comparison is apples-to-apples with ``warm.chi_squared``.
        base_residual_fn = prob.meta["base_residual_fn"]

        def _data_ssr(x: np.ndarray) -> float:
            return float(np.sum(np.asarray(base_residual_fn(x), dtype=np.float64) ** 2))

        # Phase 2: CMA-ES global search over the joint residual. ``model_func``
        # returns the residual vector; ydata=zeros ⇒ CMA-ES minimises ||r||².
        #
        # Tracer-safety (mirrors per-angle ``_fit_cmaes``): cmaes_wrapper wraps
        # this closure in ``normalized_model_func`` and passes JAX *tracers* for
        # ``params`` during JIT tracing of parameter normalization. Stack with
        # ``jnp.stack`` (NOT ``np.asarray``) so the joint residual JIT-traces
        # cleanly — ``np.asarray`` on a tracer raises TracerArrayConversionError.
        joint_residual_fn = prob.joint_residual_fn

        def model_func(_x: np.ndarray, *params: Any) -> Any:
            x_vec = jnp.stack(params).astype(jnp.float64)
            return joint_residual_fn(x_vec)  # type: ignore[arg-type]

        rdim = int(np.asarray(joint_residual_fn(x_warm)).size)

        from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESWrapperConfig

        # Build the wrapper config by hand (NOT ``from_nlsq_config``): that
        # helper expects the *homodyne* NLSQConfig (different field names —
        # heterodyne uses ``cmaes_max_iterations`` / ``cmaes_tolx`` /
        # ``cmaes_tolfun``). Same rationale as the per-angle ``_fit_cmaes``.
        # The seed is PINNED so the global search is bit-reproducible.
        cfg_cmaes = CMAESWrapperConfig(
            seed=_JOINT_CMAES_SEED,
            refine_with_nlsq=True,
            max_generations=getattr(config, "cmaes_max_iterations", None),
            popsize=getattr(config, "cmaes_population_size", None),
            tol_x=float(getattr(config, "cmaes_tolx", 1e-8)),
            tol_fun=float(getattr(config, "cmaes_tolfun", 1e-8)),
            restart_strategy=str(getattr(config, "cmaes_restart_strategy", "bipop")),
            max_restarts=int(getattr(config, "cmaes_max_restarts", 9)),
        )

        assert fit_with_cmaes is not None, "HAS_CMAES guards entry to the escape"
        cres = fit_with_cmaes(
            model_func=model_func,
            xdata=np.arange(rdim, dtype=np.float64),
            ydata=np.zeros(rdim, dtype=np.float64),
            p0=x_warm,
            bounds=(prob.lb, prob.ub),
            sigma=None,
            config=cfg_cmaes,
        )

        # Phase 3: keep-better. ``fit_with_cmaes`` reports the FULL residual SSR
        # as ``chi_squared`` (sum of squared residuals over the vector we fed
        # it); recompute the data-only SSR at the CMA-ES optimum for a clean
        # comparison with the warm-start's data-only ``chi_squared``.
        if cres.success and cres.parameters is not None:
            x_cmaes = np.asarray(cres.parameters, dtype=np.float64)
            cmaes_ssr = _data_ssr(x_cmaes)
        else:
            x_cmaes = x_warm
            cmaes_ssr = float("inf")

        if cres.success and cmaes_ssr <= ssr_warm * (1.0 + 1e-12):
            x_final, escape = x_cmaes, "cmaes"
        else:
            x_final, escape = x_warm, "cmaes_warmstart_kept"

        logger.info(
            "Joint CMA-ES escape: warm SSR=%.6e, cmaes SSR=%.6e → kept %s",
            ssr_warm,
            cmaes_ssr,
            escape,
        )

        return _build_joint_result(
            model,
            prob,
            c2_data,
            np.asarray(x_final, dtype=np.float64),
            phi_angles,
            config,
            weights,
            global_escape=escape,
        )
    except Exception as exc:  # noqa: BLE001 - best-effort escape, fall back to plain fit
        logger.warning(
            "Joint CMA-ES escape failed (%s: %s); falling back to plain joint fit",
            type(exc).__name__,
            exc,
        )
        fourier_fb = _build_joint_fourier(config, phi_angles)
        return _fit_joint_multi_phi(
            model=model,
            c2_data=c2_data,
            phi_angles=np.asarray(phi_angles),
            config=config,
            weights=weights,
            fourier=fourier_fb,
        )


def _fit_joint_multistart(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: np.ndarray,
    config: NLSQConfig,
    weights: np.ndarray | None,
    use_nlsq_library: bool,  # noqa: ARG001 - dispatch-signature parity (unused here)
) -> OptimizationResult:
    """Joint multi-angle MULTISTART escape — LHS global search over the joint vector.

    Lifts heterodyne's joint objective into ``run_multistart_nlsq``:

    1. **Problem** — build the shared :class:`JointProblem` once; ``bounds2`` is
       the ``(n_params, 2)`` box ``run_multistart_nlsq`` expects.
    2. **Starts** — a seed-pinned (:data:`_JOINT_MULTISTART_SEED`) Latin-Hypercube
       sweep. Each start re-runs the plain joint fit seeded at ``x_start`` via the
       Task-3 ``x0_override`` kwarg; the winner is selected by data-only SSR
       (``cost_func``), matching the keep-better UNIT used by the CMA-ES escape.
    3. **Keep-better** — compare the multistart winner against the default joint
       fit (no override) on data-only SSR and keep whichever is lower, so the
       escape is never worse than the plain fit.

    The result carries ``global_escape="multistart"`` (or
    ``"multistart_default_kept"`` when the default fit wins). On any failure the
    escape falls back to the plain joint fit (best-effort), exactly like the
    CMA-ES escape. Runs SEQUENTIALLY (``n_workers=1``): the single-fit worker
    closes over a JAX ``HeterodyneModel`` that is not process-picklable.
    """
    try:
        fourier = _build_joint_fourier(config, phi_angles)
        prob = _build_joint_problem(
            model, c2_data, phi_angles, config, weights, fourier=fourier
        )
        bounds2 = np.stack([prob.lb, prob.ub], axis=1)  # (n_params, 2)

        # Data-only SSR (excludes any L3 penalty rows) — the keep-better unit,
        # identical to the CMA-ES escape's comparison.
        base_residual_fn = prob.meta["base_residual_fn"]

        def _data_ssr(x: np.ndarray) -> float:
            return float(np.sum(np.asarray(base_residual_fn(x), dtype=np.float64) ** 2))

        # Seed-pinned LHS multistart config. ``n_starts`` is heterodyne's flat
        # ``multistart_n``. ``n_workers=1`` (JAX-pickle constraint). Screening is
        # left off: the cost_func IS the data-only SSR, so every start is a full
        # joint solve anyway (no cheap pre-screen surrogate).
        assert MultiStartConfig is not None, "HAS_JOINT_MULTISTART guards entry"
        ms_cfg = MultiStartConfig(
            enable=True,
            n_starts=int(getattr(config, "multistart_n", 10)),
            seed=_JOINT_MULTISTART_SEED,
            sampling_strategy="latin_hypercube",
            n_workers=1,
            use_screening=False,
        )

        def single_fit_func(_data: dict[str, Any], x_start: np.ndarray) -> Any:
            res = _fit_joint_multi_phi(
                model=model,
                c2_data=c2_data,
                phi_angles=np.asarray(phi_angles),
                config=config,
                weights=weights,
                fourier=prob.fourier,
                x0_override=np.asarray(x_start, dtype=np.float64),
            )
            x_fit = np.asarray(res.parameters, dtype=np.float64)
            assert SingleStartResult is not None, "HAS_JOINT_MULTISTART guards entry"
            return SingleStartResult(
                start_idx=0,
                initial_params=np.asarray(x_start, dtype=np.float64),
                final_params=x_fit,
                chi_squared=_data_ssr(x_fit),
                success=bool(getattr(res, "success", True)),
                message=str(getattr(res, "message", "")),
            )

        def cost_func(x: np.ndarray) -> float:
            return 0.5 * _data_ssr(np.asarray(x, dtype=np.float64))

        assert run_multistart_nlsq is not None, "HAS_JOINT_MULTISTART guards entry"
        ms = run_multistart_nlsq(
            data={"c2": c2_data, "phi": phi_angles},
            bounds=bounds2,
            config=ms_cfg,
            single_fit_func=single_fit_func,
            cost_func=cost_func,
        )
        x_ms = np.asarray(ms.best.final_params, dtype=np.float64)
        ssr_ms = _data_ssr(x_ms)

        # Keep-better vs the default joint fit (no override).
        default = _fit_joint_multi_phi(
            model=model,
            c2_data=c2_data,
            phi_angles=np.asarray(phi_angles),
            config=config,
            weights=weights,
            fourier=prob.fourier,
        )
        x_default = np.asarray(default.parameters, dtype=np.float64)
        ssr_default = _data_ssr(x_default)

        if ssr_ms <= ssr_default * (1.0 + 1e-12):
            x_final, escape = x_ms, "multistart"
        else:
            x_final, escape = x_default, "multistart_default_kept"

        logger.info(
            "Joint multistart escape: best-start SSR=%.6e, default SSR=%.6e → kept %s",
            ssr_ms,
            ssr_default,
            escape,
        )

        return _build_joint_result(
            model,
            prob,
            c2_data,
            np.asarray(x_final, dtype=np.float64),
            phi_angles,
            config,
            weights,
            global_escape=escape,
        )
    except Exception as exc:  # noqa: BLE001 - best-effort escape, fall back to plain fit
        logger.warning(
            "Joint multistart escape failed (%s: %s); falling back to plain joint fit",
            type(exc).__name__,
            exc,
        )
        fourier_fb = _build_joint_fourier(config, phi_angles)
        return _fit_joint_multi_phi(
            model=model,
            c2_data=c2_data,
            phi_angles=np.asarray(phi_angles),
            config=config,
            weights=weights,
            fourier=fourier_fb,
        )


# ---------------------------------------------------------------------------
# Shared joint global-escape machinery for the AVERAGED / CONSTANT layouts.
#
# The Fourier/individual escapes (``_fit_joint_cmaes_multi_phi`` /
# ``_fit_joint_multistart``) optimize the Fourier-reparam joint vector via
# ``_build_joint_problem``. The averaged (2 scaling params) and constant
# (frozen scaling) layouts have their OWN ``base_residual_fn`` + ``[physics |
# scaling]`` vector built inline by ``_fit_joint_averaged_multi_phi`` /
# ``_fit_joint_constant_multi_phi``. To honour the ``auto → averaged`` default
# (and explicit ``constant``) under CMA-ES / multistart — matching the plain
# path AND laminar_flow's CMA-ES, which honours ``use_averaged_scaling`` — those
# two solvers accept a ``global_escape_kind`` and run the search over their own
# data residual via the helpers below. Keep-better (escape kept only if it does
# not increase the data-only SSR) and the NaN-covariance / n_iterations=0 escape
# contract are applied by the solver, exactly like the Fourier escape.
# ---------------------------------------------------------------------------


def _solve_residual_nlsq(
    residual_fn: Any,
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    solver_config: NLSQConfig,
    param_names: list[str],
) -> np.ndarray:
    """Local trust-region solve of ``residual_fn`` from ``x0`` (adapter→wrapper).

    Mirrors the adapter-primary / wrapper-fallback dispatch the averaged and
    constant solvers use for their warm-start solve, but without the L4 monitor
    callback (the per-start refines inside a multistart escape are not
    monitored). Returns the fitted parameter vector.
    """
    res = None
    if NLSQAdapter is not None:
        try:
            res = NLSQAdapter(parameter_names=param_names).fit(
                residual_fn=residual_fn,
                initial_params=np.asarray(x0, dtype=np.float64),
                bounds=(np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64)),
                config=solver_config,
            )
            if not res.success:
                raise RuntimeError(res.message)
        except (ValueError, RuntimeError, TypeError):
            res = None
    if res is None and NLSQWrapper is not None:
        res = NLSQWrapper(parameter_names=param_names).fit(
            residual_fn=residual_fn,
            initial_params=np.asarray(x0, dtype=np.float64),
            bounds=(np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64)),
            config=solver_config,
        )
    if res is None:  # pragma: no cover — guarded by callers
        raise ImportError("No NLSQ backend available for residual solve.")
    return np.asarray(res.parameters, dtype=np.float64)


def _cmaes_joint_candidate(
    base_residual_fn: Any,
    x_warm: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    config: NLSQConfig,
) -> np.ndarray | None:
    """Seed-pinned CMA-ES global search over ``base_residual_fn`` from ``x_warm``.

    Returns the CMA-ES optimum (``None`` when the search did not succeed so the
    caller keeps the warm-start). Mirrors ``_fit_joint_cmaes_multi_phi`` Phase 2
    but over the averaged/constant data residual (not the Fourier-augmented
    one); ``ydata=zeros`` ⇒ CMA-ES minimises ``||residual||²`` directly.
    """
    from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESWrapperConfig

    x_warm = np.asarray(x_warm, dtype=np.float64)

    # Tracer-safety: cmaes_wrapper passes JAX tracers during JIT tracing of
    # parameter normalization — stack with jnp.stack, not np.asarray.
    def model_func(_x: np.ndarray, *params: Any) -> Any:
        x_vec = jnp.stack(params).astype(jnp.float64)
        return base_residual_fn(x_vec)

    rdim = int(np.asarray(base_residual_fn(x_warm)).size)
    cfg_cmaes = CMAESWrapperConfig(
        seed=_JOINT_CMAES_SEED,
        refine_with_nlsq=True,
        max_generations=getattr(config, "cmaes_max_iterations", None),
        popsize=getattr(config, "cmaes_population_size", None),
        tol_x=float(getattr(config, "cmaes_tolx", 1e-8)),
        tol_fun=float(getattr(config, "cmaes_tolfun", 1e-8)),
        restart_strategy=str(getattr(config, "cmaes_restart_strategy", "bipop")),
        max_restarts=int(getattr(config, "cmaes_max_restarts", 9)),
        # ``sigma`` config field = initial CMA-ES step (fraction of search
        # range), NOT the ``sigma=`` arg to fit_with_cmaes (per-point weight).
        # Honour ``cmaes_sigma0`` here as the per-angle path does; omitting it
        # silently pinned the joint escape to the 0.5 default.
        sigma=float(getattr(config, "cmaes_sigma0", 0.5)),
    )
    assert fit_with_cmaes is not None, "HAS_CMAES guards entry to the escape"
    cres = fit_with_cmaes(
        model_func=model_func,
        xdata=np.arange(rdim, dtype=np.float64),
        ydata=np.zeros(rdim, dtype=np.float64),
        p0=x_warm,
        bounds=(np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64)),
        sigma=None,
        config=cfg_cmaes,
    )
    if cres.success and cres.parameters is not None:
        return np.asarray(cres.parameters, dtype=np.float64)
    return None


def _multistart_joint_candidate(
    base_residual_fn: Any,
    x_warm: np.ndarray,  # noqa: ARG001 - LHS samples its own starts; signature parity
    lb: np.ndarray,
    ub: np.ndarray,
    solver_config: NLSQConfig,
    param_names: list[str],
    config: NLSQConfig,
    data: dict[str, Any],
) -> np.ndarray | None:
    """Seed-pinned LHS multistart over ``base_residual_fn``; returns the best start.

    Mirrors ``_fit_joint_multistart`` but each start is a local trust-region
    refine of the averaged/constant data residual (``_solve_residual_nlsq``)
    rather than a Fourier-reparam joint solve. The keep-better vs the warm-start
    is applied by the caller (``_apply_global_escape``).
    """
    if not HAS_JOINT_MULTISTART:
        return None
    bounds2 = np.stack([np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64)], axis=1)

    def _ssr(x: np.ndarray) -> float:
        return float(np.sum(np.asarray(base_residual_fn(x), dtype=np.float64) ** 2))

    assert MultiStartConfig is not None and SingleStartResult is not None
    ms_cfg = MultiStartConfig(
        enable=True,
        n_starts=int(getattr(config, "multistart_n", 10)),
        seed=_JOINT_MULTISTART_SEED,
        sampling_strategy="latin_hypercube",
        n_workers=1,
        use_screening=False,
    )

    def single_fit_func(_data: dict[str, Any], x_start: np.ndarray) -> Any:
        x_fit = _solve_residual_nlsq(
            base_residual_fn, np.asarray(x_start, dtype=np.float64), lb, ub, solver_config, param_names
        )
        return SingleStartResult(
            start_idx=0,
            initial_params=np.asarray(x_start, dtype=np.float64),
            final_params=x_fit,
            chi_squared=_ssr(x_fit),
            success=True,
            message="",
        )

    def cost_func(x: np.ndarray) -> float:
        return 0.5 * _ssr(np.asarray(x, dtype=np.float64))

    assert run_multistart_nlsq is not None
    ms = run_multistart_nlsq(
        data=data,
        bounds=bounds2,
        config=ms_cfg,
        single_fit_func=single_fit_func,
        cost_func=cost_func,
    )
    return np.asarray(ms.best.final_params, dtype=np.float64)


def _escape_keeps_candidate(ssr_warm: float, ssr_cand: float) -> bool:
    """Keep-better decision for the joint global escape, NaN-safe.

    A non-finite warm-start SSR must NEVER win over a finite candidate: the
    naive ``ssr_cand <= ssr_warm * (1 + eps)`` evaluates to ``False`` when
    ``ssr_warm`` is NaN/Inf, which would discard a real (finite) escape result
    in favour of a NaN warm-start fit and return it tagged as a success — a
    data-integrity defect. Rules:

    * a non-finite candidate is never an improvement;
    * a finite candidate always beats a non-finite warm start;
    * otherwise the original within-tolerance keep-better comparison applies.
    """
    if not np.isfinite(ssr_cand):
        return False
    if not np.isfinite(ssr_warm):
        return True
    return bool(ssr_cand <= ssr_warm * (1.0 + 1e-12))


def _apply_global_escape(
    escape_kind: str | None,
    base_residual_fn: Any,
    x_warm: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    solver_config: NLSQConfig,
    param_names: list[str],
    config: NLSQConfig,
    multistart_data: dict[str, Any],
) -> tuple[np.ndarray, str | None]:
    """Run a global escape over the data residual and keep-better vs ``x_warm``.

    Returns ``(x_final, global_escape_tag)``. The tag is ``None`` when no escape
    was requested or the search failed (best-effort → keep warm-start, no tag);
    ``"<kind>"`` when the escape improved the data-only SSR; or
    ``"<kind>_warmstart_kept"`` when the search ran but did not beat the warm
    start. Shared by the averaged and constant solvers so keep-better semantics
    live in ONE place. Never raises — search failures fall back to ``x_warm``.
    """
    if escape_kind is None:
        return np.asarray(x_warm, dtype=np.float64), None
    x_warm = np.asarray(x_warm, dtype=np.float64)

    def _ssr(x: np.ndarray) -> float:
        return float(np.sum(np.asarray(base_residual_fn(x), dtype=np.float64) ** 2))

    ssr_warm = _ssr(x_warm)
    try:
        if escape_kind == "cmaes":
            cand = _cmaes_joint_candidate(base_residual_fn, x_warm, lb, ub, config)
        elif escape_kind == "multistart":
            cand = _multistart_joint_candidate(
                base_residual_fn, x_warm, lb, ub, solver_config, param_names, config, multistart_data
            )
        else:  # pragma: no cover — unknown kind treated as no escape
            return x_warm, None
    except Exception as exc:  # noqa: BLE001 - best-effort escape; keep warm-start
        logger.warning(
            "Joint %s escape failed (%s: %s); keeping warm-start fit",
            escape_kind,
            type(exc).__name__,
            exc,
        )
        return x_warm, None

    if cand is None:
        return x_warm, f"{escape_kind}_warmstart_kept"
    cand = np.asarray(cand, dtype=np.float64)
    cand_ssr = _ssr(cand)
    logger.info(
        "Joint %s escape (%s layout): warm SSR=%.6e, escape SSR=%.6e",
        escape_kind,
        "averaged/constant",
        ssr_warm,
        cand_ssr,
    )
    if _escape_keeps_candidate(ssr_warm, cand_ssr):
        return cand, escape_kind
    return x_warm, f"{escape_kind}_warmstart_kept"


def _resolve_effective_mode(config: NLSQConfig, n_phi: int) -> str:
    """Map ``config.per_angle_mode`` + ``n_phi`` to a canonical dispatch token.

    Returns one of:

    * ``"constant"`` — frozen per-angle (β, ō) from diagonal-quantile estimator;
      optimizer dimension is ``n_physics_varying`` only.
    * ``"averaged"`` — one (β̄, ō̄) pair optimized jointly with physics. This
      is the homodyne ``auto``-averaged anti-degeneracy path.
    * ``"fourier"`` — Fourier-basis reparameterization of per-angle scaling
      (smooth angular variation).
    * ``"individual"`` — ``n_phi`` independent per-angle ``(contrast, offset)``
      optimized JOINTLY with physics via :func:`_fit_joint_multi_phi`
      (:class:`FourierReparameterizer` ``"independent"`` mode), matching
      ``laminar_flow`` and upstream heterodyne. (The sequential per-angle
      aggregate survives only as the ``config is None`` / single-angle
      fallback inside :func:`fit_nlsq_multi_phi`.)

    ``auto`` resolution is unified with the homodyne
    :class:`AntiDegeneracyController` — ``auto`` only ever selects
    ``individual`` or ``averaged``::

        n_phi <  constant_scaling_threshold (3) -> "individual"
        n_phi >= constant_scaling_threshold (3) -> "averaged"

    ``constant`` and ``fourier`` are NEVER auto-selected; the user must request
    them explicitly via ``anti_degeneracy.per_angle_mode`` in the config (or a
    CLI option). ``fourier_auto_threshold`` therefore has no effect under
    ``auto`` and is consulted only when ``fourier`` is requested explicitly.

    Explicit modes (``"constant"``, ``"fourier"``, ``"individual"``) pass
    through unchanged. The legacy alias ``"independent"`` is already rewritten
    to ``"individual"`` by :meth:`NLSQConfig.__post_init__`.
    """
    requested = config.per_angle_mode
    if requested == "constant":
        return "constant"
    if requested == "fourier":
        return "fourier"
    if requested == "individual":
        return "individual"
    # requested == "auto" — unified with the homodyne AntiDegeneracyController:
    # few angles -> per-angle individual scaling; otherwise the averaged
    # single-pair scaling. constant/fourier are never auto-selected.
    constant_threshold = max(int(config.constant_scaling_threshold), 1)
    if n_phi < constant_threshold:
        return "individual"
    return "averaged"


@dataclass
class JointProblem:
    """Constructed heterodyne joint LSQ problem (residual + x0 + bounds + reparam).

    Shared by :func:`_fit_joint_multi_phi` (the plain joint fit) and the
    upcoming CMA-ES / multistart global escapes so all three optimize the SAME
    objective. ``joint_residual_fn`` is the L3-augmented residual NLSQ minimizes;
    ``meta`` carries the bookkeeping the caller needs to assemble diagnostics
    (``base_residual_fn`` for the data-only SSR, the scaling-tail size, the
    regularization-active flag + penalty-row count, and any L2 stage-1 chi^2).
    """

    joint_residual_fn: Callable[[np.ndarray], Any]
    x0: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    fourier: Any
    meta: dict[str, Any]


def _build_joint_problem(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: np.ndarray,
    config: NLSQConfig,
    weights: np.ndarray | None,
    *,
    fourier: Any = None,
) -> JointProblem:
    """Build the joint heterodyne LSQ problem (residual + x0 + bounds + reparam).

    Lifts the inline construction of :func:`_fit_joint_multi_phi` VERBATIM so the
    plain joint fit and the global escapes optimize the SAME objective. When
    ``fourier`` is provided (already built by the dispatch in
    :func:`fit_nlsq_multi_phi`) it is reused; otherwise a mode-appropriate
    ``FourierReparameterizer`` is built here in ``"independent"`` mode (the
    individual per-angle layout ``[physics | contrast_0..N | offset_0..N]``).

    Includes the L2 hierarchical Stage 1 physics-only solve (run when
    ``config.enable_hierarchical`` is True) — its converged physics vector
    warm-starts ``x0`` and its chi^2 is surfaced via ``meta``.

    Returns
    -------
    JointProblem
        ``joint_residual_fn`` is the L3-augmented residual (identity to
        ``base_residual_fn`` when ``config.regularization_mode == "none"``).
        ``meta`` carries ``base_residual_fn``, ``n_physics_varying``,
        ``scaling_tail_size`` (``2*(2K+1)`` fourier / ``2*n_phi`` individual),
        ``regularization_active``, ``n_penalty_rows``, and
        ``hierarchical_stage1_chi2`` (``None`` when L2 is disabled).
    """
    param_manager = model.param_manager
    varying_names = param_manager.varying_names
    n_physics_varying = param_manager.n_varying
    n_phi = len(phi_angles)

    # Build the reparameterizer here if the dispatch did not supply one. The
    # ``"independent"`` mode is the individual per-angle layout (free contrast /
    # offset per angle); an identity passthrough w.r.t. the per-angle scaling.
    if fourier is None:
        from xpcsjax.optimization.nlsq.fourier_reparam import (
            FourierReparamConfig,
            FourierReparameterizer,
        )

        # Mirrors the ``individual`` dispatch in ``fit_nlsq_multi_phi``:
        # ``"independent"`` mode makes the reparameterizer an identity
        # passthrough describing the ``2*n_phi`` per-angle scaling layout.
        fourier_config = FourierReparamConfig(
            mode="independent",
            fourier_order=config.fourier_order,
            auto_threshold=config.fourier_auto_threshold,
        )
        phi_rad = np.deg2rad(np.asarray(phi_angles, dtype=np.float64))
        fourier = FourierReparameterizer(phi_rad, fourier_config)

    # Physics parameter initial values and bounds
    physics_initial = param_manager.get_initial_values()
    physics_lower, physics_upper = param_manager.get_bounds()
    physics_initial = np.clip(physics_initial, physics_lower, physics_upper)

    # ------------------------------------------------------------------
    # L2 hierarchical two-stage: Stage 1 — physics-only solve with
    # quantile-fixed scaling (delegates to the constant-mode solver).
    # When `config.enable_hierarchical` is True we run the constant-mode
    # solver first to converge the physics block with scaling frozen,
    # then warm-start the joint Fourier solve below by overriding
    # `physics_initial` with the converged physics vector. The Fourier
    # coefficients keep their deterministic initial values from
    # `fourier.get_initial_coefficients`.
    # ------------------------------------------------------------------
    hierarchical_stage1_chi2: float | None = None
    if config.enable_hierarchical:
        logger.info(
            "L2 hierarchical (fourier mode) — Stage 1: physics-only solve "
            "with quantile-fixed scaling"
        )
        from xpcsjax.optimization.nlsq.heterodyne_constant_mode import (
            _fit_joint_constant_multi_phi,
        )

        stage1_result = _fit_joint_constant_multi_phi(
            model=model,
            c2_data=c2_data,
            phi_angles=phi_angles,
            config=config,
            weights=weights,
        )
        stage1_physics = np.asarray(stage1_result.parameters, dtype=np.float64)
        hierarchical_stage1_chi2 = float(stage1_result.chi_squared)
        physics_initial = np.clip(stage1_physics, physics_lower, physics_upper)
        logger.info(
            "L2 hierarchical (fourier mode) — Stage 1 done: chi2=%.6f, "
            "warm-starting stage 2 joint refine",
            hierarchical_stage1_chi2,
        )

    # Fourier coefficient initial values and bounds
    scaling = model.scaling
    contrast_init = float(scaling.contrast[0]) if len(scaling.contrast) > 0 else 0.5
    offset_init = float(scaling.offset[0]) if len(scaling.offset) > 0 else 1.0
    fourier_initial = fourier.get_initial_coefficients(contrast_init, offset_init)
    fourier_lower, fourier_upper = fourier.get_bounds()

    # Combined parameter vector
    x0 = np.concatenate([physics_initial, fourier_initial])
    lb = np.concatenate([physics_lower, fourier_lower])
    ub = np.concatenate([physics_upper, fourier_upper])

    logger.info(
        "Joint multi-angle fit: %d physics + %d Fourier = %d total params, %d angles",
        n_physics_varying,
        fourier.n_coeffs,
        len(x0),
        n_phi,
    )

    # Pre-convert data to JAX arrays (outside closure — constants)
    t, q, dt = model.t, model.q, model.dt
    c2_data_list = [jnp.asarray(c2_data[i], dtype=jnp.float64) for i in range(n_phi)]
    weights_list: list[jnp.ndarray | None] = []
    for i in range(n_phi):
        if weights is not None and weights.ndim == 3:
            weights_list.append(jnp.asarray(weights[i], dtype=jnp.float64))
        elif weights is not None:
            weights_list.append(jnp.asarray(weights, dtype=jnp.float64))
        else:
            weights_list.append(None)

    # Pre-stack batched arrays for compute_multi_angle_residuals.
    # weights_list entries may be None (unweighted) — materialise ones_like
    # so the stacked weights_batch is always a concrete (n_phi, N, N) array.
    c2_data_batch = jnp.stack(c2_data_list, axis=0)  # (n_phi, N, N)
    weights_batch = jnp.stack(
        [
            (w if w is not None else jnp.ones_like(c2_data_list[i]))
            for i, w in enumerate(weights_list)
        ],
        axis=0,
    )  # (n_phi, N, N)
    phi_angles_jax = jnp.asarray(phi_angles, dtype=jnp.float64)  # (n_phi,)

    fixed_values_jax = jnp.asarray(param_manager.get_full_values(), dtype=jnp.float64)
    varying_indices_jax = jnp.array(param_manager.varying_indices, dtype=jnp.int32)

    # NOTE: must return a JAX array. NLSQ's masked_residual_func JIT-traces
    # this closure; calling ``np.asarray`` on a traced result raises
    # TracerArrayConversionError. Same fix as
    # ``_fit_joint_averaged_multi_phi`` / ``_fit_joint_constant_multi_phi``
    # — the kernel returns ``jnp.ndarray`` and NLSQ casts at its boundary.
    def base_residual_fn(x: np.ndarray) -> Any:  # type: ignore[return-value]
        """Compute concatenated residuals across all angles via vmap.

        Routes through ``compute_multi_angle_residuals`` (jit + vmap) to
        replace the previous n_phi serial kernel dispatches with a single
        batched XLA call.  Fourier reparameterization is preserved: the
        combined parameter vector is split into physics and Fourier parts,
        and ``fourier.fourier_to_per_angle`` converts coefficients to
        per-angle contrast/offset arrays before the batched residual call.
        """
        # Split combined vector
        physics_varying = x[:n_physics_varying]
        fourier_coeffs = x[n_physics_varying:]

        # Reconstruct full physics parameter array (immutable JAX scatter)
        varying_jax = jnp.asarray(physics_varying, dtype=jnp.float64)
        full_jax = fixed_values_jax.at[varying_indices_jax].set(varying_jax)

        # Convert Fourier coefficients → per-angle contrast/offset. Must use
        # the JIT-safe variant: the numpy fourier_to_per_angle calls np.asarray
        # on the traced coefficient slice, raising TracerArrayConversionError
        # inside NLSQ's JIT-compiled residual (silently degraded the fourier fit).
        contrasts_jax, offsets_jax = fourier.fourier_to_per_angle_jax(fourier_coeffs)

        # Single batched vmap call — eliminates n_phi serial dispatches
        return compute_multi_angle_residuals(
            full_jax,
            t,
            q,
            dt,
            phi_angles_jax,
            c2_data_batch,
            weights_batch,
            contrasts_jax,
            offsets_jax,
        )

    # ------------------------------------------------------------------
    # L3 anti-degeneracy: wrap base residual with adaptive CV-regularization.
    # When ``config.regularization_mode != "none"`` we build an
    # AdaptiveRegularizer keyed to the per-angle scaling groups (contrast +
    # offset, derived from the Fourier coefficients) and append penalty rows
    # to the residual vector. NLSQ's trust-region solver minimises ``||r||²``,
    # so K appended rows with values ``sqrt(lambda) * CV_g`` yield an extra
    # ``lambda * sum_g(CV_g^2)`` penalty term — the JIT-traceable variant of
    # the CV-based regularizer documented in
    # ``adaptive_regularization.AdaptiveRegularizer``. Penalty rows operate
    # on the *per-angle scaling arrays* derived from the Fourier coefficients
    # (the natural target since Fourier reparameterization may smooth the
    # raw coefficient variance away from the per-angle CV that actually
    # matters).
    #
    # Wrapping happens here (inside the residual factory) rather than after
    # the solve so NLSQ's CurveFit sees the augmented residual end-to-end.
    # ``base_residual_fn`` is preserved for the data-only SSR diagnostic.
    # ------------------------------------------------------------------
    regularization_active = config.regularization_mode != "none"
    n_penalty_rows = 0
    if regularization_active:
        from xpcsjax.optimization.nlsq.adaptive_regularization import (
            AdaptiveRegularizationConfig,
            AdaptiveRegularizer,
        )

        reg_mode_jax: Any = "relative" if config.regularization_mode == "adaptive" else "absolute"
        reg_config = AdaptiveRegularizationConfig(
            enable=True,
            mode=reg_mode_jax,
            lambda_base=float(config.group_variance_lambda),
            target_cv=float(config.regularization_target_cv),
            # Disable auto-tune so ``lambda_value`` is the user-specified
            # ``group_variance_lambda``; the auto-tune formula assumes a
            # different (scalar-loss) integration mode.
            auto_tune_lambda=False,
        )
        regularizer = AdaptiveRegularizer(reg_config, n_phi=n_phi, n_params=len(x0))
        n_penalty_rows = len(regularizer.group_indices)

        # JAX-traceable penalty rows. AdaptiveRegularizer's group_indices
        # default to (0, n_phi) and (n_phi, 2*n_phi) — but our combined
        # parameter vector is [physics_varying | fourier_coeffs], not
        # [contrast(n_phi) | offset(n_phi) | physics]. We therefore compute
        # CV directly from the per-angle scaling arrays derived from the
        # Fourier coefficients (contrasts_jax, offsets_jax), bypassing the
        # raw group_indices which assume a different layout.
        sqrt_lambda = float(np.sqrt(float(regularizer.lambda_value)))

        def joint_residual_fn(x: np.ndarray) -> Any:  # type: ignore[return-value]
            r = base_residual_fn(x)
            # Per-angle contrast/offset via the shared JIT-safe conversion (the
            # numpy fourier_to_per_angle calls np.asarray on the traced slice and
            # crashes inside the JIT-compiled residual).
            contrasts, offsets = fourier.fourier_to_per_angle_jax(x[n_physics_varying:])
            # CV = std / |mean| (safe divide)
            c_mean = jnp.mean(contrasts)
            c_cv = jnp.where(
                jnp.abs(c_mean) > 1e-10,
                jnp.std(contrasts) / jnp.abs(c_mean),
                jnp.std(contrasts),
            )
            o_mean = jnp.mean(offsets)
            o_cv = jnp.where(
                jnp.abs(o_mean) > 1e-10,
                jnp.std(offsets) / jnp.abs(o_mean),
                jnp.std(offsets),
            )
            penalty_rows = jnp.array([sqrt_lambda * c_cv, sqrt_lambda * o_cv], dtype=jnp.float64)
            return jnp.concatenate([r, penalty_rows])
    else:
        joint_residual_fn = base_residual_fn  # type: ignore[assignment]

    meta: dict[str, Any] = {
        "base_residual_fn": base_residual_fn,
        "n_physics_varying": int(n_physics_varying),
        "scaling_tail_size": int(len(fourier_initial)),
        "regularization_active": regularization_active,
        "n_penalty_rows": int(n_penalty_rows),
        "hierarchical_stage1_chi2": hierarchical_stage1_chi2,
        "varying_names": list(varying_names),
    }

    return JointProblem(
        joint_residual_fn=joint_residual_fn,
        x0=x0,
        lb=lb,
        ub=ub,
        fourier=fourier,
        meta=meta,
    )


def _fit_joint_multi_phi(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: np.ndarray,
    config: NLSQConfig,
    weights: np.ndarray | None,
    fourier: Any,
    x0_override: np.ndarray | None = None,
) -> OptimizationResult:
    """Joint multi-angle fit with reparameterized per-angle scaling.

    Shared joint solver for BOTH the ``fourier`` and ``individual`` per-angle
    modes — the only difference is the :class:`FourierReparameterizer` mode the
    caller passes in:

    * ``fourier`` (``fourier.use_fourier=True``) — per-angle scaling is a
      truncated Fourier basis; optimizer vector is
      ``[physics_varying | fourier_contrast_coeffs | fourier_offset_coeffs]``
      (``physics + 2*(2K+1)``).
    * ``individual`` (``fourier.use_fourier=False``, mode ``"independent"``) —
      per-angle scaling is free; optimizer vector is
      ``[physics_varying | contrast_0..N | offset_0..N]`` (``physics + 2*n_phi``),
      matching xpcsjax ``laminar_flow`` and upstream heterodyne. The
      reparameterizer is an identity passthrough in this mode.

    The residual function evaluates all angles, using the reparameterizer to
    convert coefficients → per-angle contrast/offset at each evaluation (an
    identity map in individual mode).

    This is the heterodyne equivalent of homodyne's AntiDegeneracyController
    joint-fit path.

    Returns
    -------
    OptimizationResult
        One result for the entire joint solve.  ``parameters`` has the
        ``physics_varying + 2*(2K+1)`` (fourier) or ``physics_varying + 2*n_phi``
        (individual) layout. Per-angle diagnostics — ``chi2_per_angle``,
        ``fourier_basis_dim`` (``None`` in individual mode), ``per_angle_mode``
        (``'fourier'`` or ``'individual'``), ``scaling_source='fitted'``,
        ``shear_weighting='not_applicable_heterodyne'`` — live in
        ``nlsq_diagnostics``.  Mirrors the contract of
        :func:`xpcsjax.optimization.nlsq.heterodyne_constant_mode._fit_joint_constant_multi_phi`
        (Sub-PR B2).
    """
    t_start = time.perf_counter()

    n_phi = len(phi_angles)

    # Construct the joint LSQ problem (residual + x0 + bounds + reparam) via the
    # shared helper so the plain fit and the global escapes optimize the SAME
    # objective. ``fourier`` was already built by the dispatch — thread it
    # through to avoid double-building the reparameterizer. The result-tail
    # bookkeeping (scaling, base residual, regularization/hierarchical meta) is
    # consumed by ``_build_joint_result``, not here.
    prob = _build_joint_problem(model, c2_data, phi_angles, config, weights, fourier=fourier)
    joint_residual_fn = prob.joint_residual_fn
    # ``x0_override`` (Task 3) lets the joint multistart escape seed the solver at
    # an arbitrary LHS start. Default ``None`` ⇒ today's ``prob.x0`` (the
    # warm-start built from ``model.scaling`` + the deterministic Fourier coeffs),
    # so the plain fit and every existing caller are byte-identical. The override
    # is clipped to the problem bounds so an out-of-range LHS draw cannot escape
    # the feasible box.
    if x0_override is not None:
        x0 = np.clip(np.asarray(x0_override, dtype=np.float64), prob.lb, prob.ub)
    else:
        x0 = prob.x0
    lb = prob.lb
    ub = prob.ub
    fourier = prob.fourier
    varying_names = prob.meta["varying_names"]

    # Run optimization via NLSQAdapter (primary) with NLSQWrapper fallback.
    # max_nfev is multiplied by n_phi here because the Fourier joint solve
    # packs all angles into a single residual vector; the per-angle budget
    # documented on NLSQConfig.max_nfev is preserved by scaling the
    # combined cap. See NLSQConfig.max_nfev docstring for the contract.
    joint_config = NLSQConfig(
        method=config.method if config.method != "lm" else "trf",
        ftol=config.ftol,
        xtol=config.xtol,
        gtol=config.gtol,
        max_nfev=(config.max_nfev * n_phi if config.max_nfev is not None else None),
    )

    joint_result: NLSQResult | None = None
    # Scaling-tail parameter names depend on the reparameterizer mode: Fourier
    # coefficients (``fourier_i``) vs. per-angle individual scaling
    # (``contrast_i`` / ``offset_i``). ``get_coefficient_labels`` returns the
    # right vocabulary for both modes.
    joint_param_names = list(varying_names) + list(fourier.get_coefficient_labels())

    # L4: per-iteration gradient-collapse monitor (strictly observational).
    # Joint layout is [physics (n_physics) | fourier coeffs] — the fourier
    # coefficients are the per-angle (scaling) tail. See ``_build_l4_callback``.
    _monitor, _l4_callback = _build_l4_callback(model, x0, joint_residual_fn, config)

    # Tracks whether the RETURNED ``joint_result`` came from the monitored
    # adapter (the only backend the L4 callback is wired into). Stays False on
    # the unmonitored NLSQWrapper fallback path so _assemble_l4_extras does not
    # surface a stale per-iteration monitor against the wrapper's parameters.
    used_monitored_backend = False
    if NLSQAdapter is not None:  # ``HAS_ADAPTERS`` equivalent; narrows for Pyright
        try:
            joint_adapter = NLSQAdapter(parameter_names=joint_param_names)
            joint_result = joint_adapter.fit(
                residual_fn=joint_residual_fn,
                initial_params=x0,
                bounds=(lb, ub),
                config=joint_config,
                callback=_l4_callback,
            )
            if not joint_result.success:
                raise RuntimeError(f"Joint adapter returned success=False: {joint_result.message}")
            # Adapter succeeded → the returned result IS the monitored run.
            used_monitored_backend = True
        except (ValueError, RuntimeError, TypeError) as adapter_exc:
            logger.warning("Joint NLSQAdapter failed, falling back to NLSQWrapper: %s", adapter_exc)
            joint_result = None

    if joint_result is None and NLSQWrapper is not None:
        joint_wrapper = NLSQWrapper(parameter_names=joint_param_names)
        joint_result = joint_wrapper.fit(
            residual_fn=joint_residual_fn,
            initial_params=x0,
            bounds=(lb, ub),
            config=joint_config,
        )

    if joint_result is None:
        raise ImportError(
            "No NLSQ backend available for joint multi-angle fit. "
            "Ensure heterodyne.optimization.nlsq.adapter is importable."
        )

    wall_time = time.perf_counter() - t_start

    return _build_joint_result(
        model,
        prob,
        c2_data,
        np.asarray(joint_result.parameters, dtype=np.float64),
        phi_angles,
        config,
        weights,
        joint_result=joint_result,
        joint_param_names=joint_param_names,
        wall_time=wall_time,
        monitor=_monitor,
        used_monitored_backend=used_monitored_backend,
    )


def _build_joint_result(
    model: HeterodyneModel,
    prob: JointProblem,
    c2_data: np.ndarray,
    x_final: np.ndarray,
    phi_angles: np.ndarray,
    config: NLSQConfig,
    weights: np.ndarray | None,
    *,
    joint_result: NLSQResult | None = None,
    joint_param_names: list[str] | None = None,
    wall_time: float = 0.0,
    monitor: Any = None,
    used_monitored_backend: bool = False,
    global_escape: str | None = None,
) -> OptimizationResult:
    """Assemble the joint :class:`OptimizationResult` from a final parameter vector.

    Behavior-preserving extraction of :func:`_fit_joint_multi_phi`'s result tail
    so the plain joint fit and the global escapes (CMA-ES, multistart) emit an
    IDENTICAL-contract result — per-angle χ² (SSR conservation), symmetric
    diagnostics, L2/L3/L4 extras — just evaluated at a possibly different
    ``x_final``.

    When ``joint_result`` is ``None`` (a global escape that did not run NLSQ's
    adapter to produce one), uncertainties/covariance are NaN-filled and
    convergence is reported as ``"converged"`` (the escape only returns a vector
    it has already accepted). ``global_escape``, when set (e.g. ``"cmaes"``),
    is surfaced in ``nlsq_diagnostics`` so callers can tell a global-escape
    result from a plain joint fit.
    """
    param_manager = model.param_manager
    scaling = model.scaling
    n_phi = len(phi_angles)

    base_residual_fn = prob.meta["base_residual_fn"]
    joint_residual_fn = prob.joint_residual_fn
    fourier = prob.fourier
    n_physics_varying = prob.meta["n_physics_varying"]
    varying_names = prob.meta["varying_names"]
    regularization_active = prob.meta["regularization_active"]
    n_penalty_rows = prob.meta["n_penalty_rows"]
    hierarchical_stage1_chi2 = prob.meta["hierarchical_stage1_chi2"]

    if joint_param_names is None:
        joint_param_names = list(varying_names) + list(fourier.get_coefficient_labels())

    # Extract results
    fitted_params_full = np.asarray(x_final, dtype=np.float64)
    fitted_physics = fitted_params_full[:n_physics_varying]
    fitted_fourier = fitted_params_full[n_physics_varying:]
    fitted_contrast, fitted_offset = fourier.fourier_to_per_angle(fitted_fourier)

    # Update model with fitted physics parameters
    full_fitted = param_manager.expand_varying_to_full(fitted_physics)
    model.set_params(full_fitted)

    # Update model scaling
    if len(scaling.contrast) == n_phi:
        scaling.contrast[:] = fitted_contrast
        scaling.offset[:] = fitted_offset

    # ------------------------------------------------------------------
    # Decompose per-angle chi^2 from the final residual.
    # ``compute_multi_angle_residuals`` returns an angle-major flat layout
    # (n_phi, n_per_angle) — n_per_angle = (n_time - 1) * (n_time - 2) because
    # the kernel excludes BOTH the t=0 boundary row/col and the diagonal. Re-
    # import the helper from the constant-mode module to keep one canonical
    # implementation.
    # TODO(C3): consolidate _decompose_chi2_per_angle when the averaged path
    # also returns OptimizationResult, so all three joint paths share the
    # same helper without crossing module boundaries.
    # ------------------------------------------------------------------
    from xpcsjax.optimization.nlsq.heterodyne_constant_mode import (
        _decompose_chi2_per_angle,
    )

    # SSR conservation: decompose chi^2 on the *data-only* residual (excluding
    # any L3 penalty rows). The base residual is what
    # ``compute_multi_angle_residuals`` returns; the L3-augmented residual may
    # carry extra rows that must NOT contribute to per-angle chi^2.
    data_only_residual = np.asarray(base_residual_fn(fitted_params_full))
    n_time = c2_data.shape[1]
    n_per_angle = (n_time - 1) * (n_time - 2)  # off-diag, t=0 boundary excluded — matches kernel
    chi2_per_angle = _decompose_chi2_per_angle(
        final_residual=data_only_residual,
        n_phi=n_phi,
        n_per_angle=n_per_angle,
    )

    # ------------------------------------------------------------------
    # Build the single joint OptimizationResult.
    # ------------------------------------------------------------------
    # SSR conservation: ``chi_squared`` is the raw residual SSR, not
    # ``2 * nlsq_result.final_cost`` (which is the robust-loss cost when
    # ``config.loss != "linear"``). Using raw residuals keeps
    # ``chi2_per_angle.sum() == chi_squared`` for every loss choice —
    # the same invariant B2 locked in for constant mode.
    # When L3 regularization is active, ``chi_squared`` reports the
    # *data-only* SSR — the penalty contribution is excluded so the
    # SSR conservation invariant (``chi2_per_angle.sum() == chi_squared``)
    # is preserved regardless of regularization mode.
    data_only_ssr = float(np.sum(data_only_residual**2))
    ssr = data_only_ssr
    # Full residual (including any penalty rows) — used for DoF and total
    # cost diagnostics only.
    final_residual = np.asarray(joint_residual_fn(fitted_params_full))
    total_ssr_with_penalty = float(np.sum(final_residual**2))
    n_total_params = int(fitted_params_full.size)
    # Noise-normalised reduced chi^2 (targets ~1.0); see the averaged path for
    # the rationale. Only ``reduced_chi_squared`` changes — ``chi_squared``
    # (= ssr) and ``chi2_per_angle`` are untouched, preserving SSR conservation.
    from xpcsjax.optimization.nlsq.heterodyne_data_prep import (
        noise_normalized_reduced_chi2,
    )

    reduced_chi2 = noise_normalized_reduced_chi2(
        ssr=ssr,
        c2_data=c2_data,
        n_data_valid=int(data_only_residual.size),
        n_params=n_total_params,
    )

    # NaN-fill uncertainties/covariance when the NLSQ adapter could not
    # produce them (e.g. singular Jacobian after a non-converged solve) —
    # matches B2's contract so consumers see a uniform array shape.
    uncertainties = (
        np.asarray(joint_result.uncertainties, dtype=np.float64)
        if joint_result is not None and joint_result.uncertainties is not None
        else np.full(n_total_params, np.nan, dtype=np.float64)
    )
    covariance = (
        np.asarray(joint_result.covariance, dtype=np.float64)
        if joint_result is not None and joint_result.covariance is not None
        else np.full((n_total_params, n_total_params), np.nan, dtype=np.float64)
    )

    # When no NLSQ result backs this assembly (a global escape that returns a
    # pre-accepted vector) report success: the escape only emits a vector it
    # has already compared and kept.
    solve_success = joint_result.success if joint_result is not None else True
    convergence_status: ConvergenceStatus = "converged" if solve_success else "failed"
    quality_flag: QualityFlag = "good" if solve_success else "marginal"

    # ------------------------------------------------------------------
    # L2 anti-degeneracy: hierarchical two-stage solve.
    #
    # Stage 1 (physics-only with quantile-fixed scaling) ran above — before
    # the joint Fourier solve — when `config.enable_hierarchical` was True,
    # producing `hierarchical_stage1_chi2` and a warm-started
    # `physics_initial`. Stage 2 is the joint refine the surrounding code
    # already executed (Fourier coefficients unfrozen, jointly fit with
    # physics over `[physics | fourier_coeffs]`).
    #
    # The SSR conservation invariant (`chi2_per_angle.sum() == chi_squared`)
    # still holds for stage 2 because the joint solve uses the canonical
    # multi-angle residual decomposition.
    # ------------------------------------------------------------------
    hierarchical_extras: dict[str, Any] = {}
    if config.enable_hierarchical and hierarchical_stage1_chi2 is not None:
        logger.info(
            "L2 hierarchical (fourier mode) — Stage 2 done: chi2=%.6f (stage1=%.6f)",
            ssr,
            hierarchical_stage1_chi2,
        )
        hierarchical_extras = {
            "hierarchical_stages": 2,
            "hierarchical_active": True,
            "hierarchical_scope": "full_two_stage",
            "hierarchical_stage1_chi2": hierarchical_stage1_chi2,
            "hierarchical_stage2_chi2": float(ssr),
        }

    # ------------------------------------------------------------------
    # L3 anti-degeneracy: adaptive CV regularization (full integration).
    # When ``config.regularization_mode != "none"`` the residual factory
    # above wrapped ``base_residual_fn`` with an L3 augmentation: K penalty
    # rows (one per scaling group — contrast + offset) with values
    # ``sqrt(lambda) * CV_g`` were appended to the residual vector. NLSQ's
    # trust-region solver minimises ``||r||²``, so the augmented residual
    # adds ``lambda * sum_g(CV_g^2)`` to the data-fit objective.
    #
    # ``regularization_data_residual_ssr`` records the data-only SSR (used
    # as ``chi_squared`` in the OptimizationResult — preserves the SSR
    # conservation invariant ``chi2_per_angle.sum() == chi_squared``).
    # ``regularization_total_ssr_with_penalty`` reports the full augmented
    # SSR for diagnostic comparison.
    # ------------------------------------------------------------------
    regularization_extras: dict[str, Any] = {}
    if regularization_active:
        logger.info(
            "L3 adaptive regularization enabled (fourier mode): "
            "mode=%s, lambda=%.6g, target_cv=%.3f, penalty_rows=%d.",
            config.regularization_mode,
            config.group_variance_lambda,
            config.regularization_target_cv,
            n_penalty_rows,
        )
        regularization_extras = {
            "regularization_active": True,
            "regularization_mode": config.regularization_mode,
            "regularization_lambda_applied": float(config.group_variance_lambda),
            "regularization_penalty_count": int(n_penalty_rows),
            "regularization_data_residual_ssr": data_only_ssr,
            "regularization_total_ssr_with_penalty": total_ssr_with_penalty,
            "regularization_scope": "full_residual_augmentation",
        }

    # ------------------------------------------------------------------
    # L4 anti-degeneracy: gradient collapse monitor (full integration).
    #
    # The fourier joint solve fits ``[physics | fourier_coeffs]`` jointly;
    # gradient collapse here typically indicates an under-constrained Fourier
    # basis or a near-degenerate physics-vs-scaling subspace. The monitor
    # records the per-iteration gradient ratio via NLSQ's curve_fit callback
    # (built in ``_build_l4_callback``); when it recorded zero observations,
    # ``_assemble_l4_extras`` falls back to the post-solve covariance-condition
    # block (tagged ``mechanism="post_solve_fallback"``).
    # ------------------------------------------------------------------
    # The joint solver is shared between ``fourier`` and ``individual`` per-angle
    # modes (the ONLY difference is the FourierReparameterizer mode — Fourier
    # basis vs. identity passthrough). Derive the reported mode from the
    # reparameterizer so the diagnostics reflect the actual layout: individual
    # mode reports ``per_angle_mode="individual"`` with no Fourier basis dim.
    is_individual = not fourier.use_fourier
    per_angle_mode_label = "individual" if is_individual else "fourier"

    # L4 extras require both a monitor and the NLSQ result it described. A
    # global escape supplies neither (monitor=None, joint_result=None), so the
    # block is omitted — ``_assemble_l4_extras`` would itself short-circuit to
    # ``{}`` on ``monitor is None``, but guarding here keeps the typed contract
    # (it expects a non-None ``NLSQResult``) honest.
    if monitor is not None and joint_result is not None:
        gradient_monitor_extras = _assemble_l4_extras(
            monitor,
            joint_result,
            config,
            mode_label=f"{per_angle_mode_label} mode (joint)",
            result_is_monitored=used_monitored_backend,
        )
    else:
        gradient_monitor_extras = {}

    # Solve-shape diagnostics. A global escape supplies no NLSQResult; report
    # neutral defaults (the escape's own convergence is summarised by the
    # ``global_escape`` tag below).
    convergence_reason = (
        joint_result.convergence_reason if joint_result is not None else "global_escape"
    )
    n_function_evals = int(joint_result.n_function_evals or 0) if joint_result is not None else 0
    n_iterations = int(joint_result.n_iterations or 0) if joint_result is not None else 0
    solve_message = str(joint_result.message) if joint_result is not None else "global escape"

    # Fourier-specific diagnostic extras are only meaningful in Fourier mode.
    # In individual mode the "coefficients" ARE the per-angle scaling values,
    # so the basis-reduction metadata is omitted (kept ``None``/absent).
    fourier_extras: dict[str, Any] = {}
    if not is_individual:
        fourier_extras = {
            "fourier_mode": fourier.config.mode,
            "fourier_order": fourier.order,
            "fourier_coeffs": fitted_fourier.tolist(),
            "fourier_n_coeffs": fourier.n_coeffs,
            "fourier_reduction": fourier.get_diagnostics()["reduction_ratio"],
        }

    diagnostics = _build_heterodyne_diagnostics(
        per_angle_mode=per_angle_mode_label,
        chi2_per_angle=chi2_per_angle,
        scaling_source="fitted",
        fourier_basis_dim=None if is_individual else fourier.n_coeffs_per_param,
        parameter_names=joint_param_names,
        contrast_per_angle_fitted=np.asarray(fitted_contrast, dtype=np.float64),
        offset_per_angle_fitted=np.asarray(fitted_offset, dtype=np.float64),
        phi_angles=np.asarray(phi_angles, dtype=np.float64),
        n_angles_joint=n_phi,
        **fourier_extras,
        convergence_reason=convergence_reason,
        n_function_evals=n_function_evals,
        n_iterations=n_iterations,
        wall_time_seconds=wall_time,
        message=solve_message,
        **hierarchical_extras,
        **regularization_extras,
        **gradient_monitor_extras,
    )

    # Tag a global-escape assembly so callers can distinguish it from a plain
    # joint fit (the plain fit leaves this key absent).
    if global_escape is not None:
        diagnostics["global_escape"] = global_escape

    logger.info(
        "Joint multi-angle fit complete: success=%s, cost=%.6f, "
        "n_evals=%d, wall_time=%.2fs, %d angles%s",
        solve_success,
        (joint_result.final_cost or 0.0) if joint_result is not None else ssr,
        n_function_evals,
        wall_time,
        n_phi,
        f" [escape={global_escape}]" if global_escape is not None else "",
    )

    return OptimizationResult(
        parameters=np.asarray(fitted_params_full, dtype=np.float64),
        uncertainties=uncertainties,
        covariance=covariance,
        chi_squared=ssr,
        reduced_chi_squared=reduced_chi2,
        convergence_status=convergence_status,
        iterations=n_iterations,
        execution_time=wall_time,
        device_info={"backend": "cpu", "adapter": "nlsq.CurveFit"},
        recovery_actions=[],
        quality_flag=quality_flag,
        streaming_diagnostics=None,
        stratification_diagnostics=None,
        nlsq_diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_global_optimization(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float,
    config: NLSQConfig,
    weights: np.ndarray | jnp.ndarray | None,
    use_nlsq_library: bool,
    angle_idx: int = 0,
) -> NLSQResult | None:
    """Attempt CMA-ES or multi-start if configured.

    Returns the result if a global method was selected, or ``None`` to
    fall through to local optimization.

    Notes
    -----
    The annotation stays ``NLSQResult | None`` because this is the
    per-angle global-search entry called from :func:`fit_nlsq_jax`
    (which also returns ``NLSQResult``). The C-series return-shape
    alignment converted the multi-phi joint paths only; the per-angle
    chain is still NLSQResult-shaped.

    ``_fit_multistart`` was converted to return :class:`OptimizationResult`
    in C4 (forward-looking, since the eventual multistart wiring will
    aggregate multi-phi results), but the runtime branch is unreachable:
    ``HAS_MULTISTART`` is hard-coded ``False`` at module import. The
    ``# type: ignore[return-value]`` below documents that dead-code
    typing gap; it will go away once the per-angle path itself is
    migrated to :class:`OptimizationResult` (tracked alongside the
    ``individual``-mode aggregation as a Phase-6 follow-up).
    """
    # CMA-ES has highest priority
    if getattr(config, "enable_cmaes", False):
        if HAS_CMAES:
            logger.info("CMA-ES enabled, delegating to fit_with_cmaes")
            return _fit_cmaes(model, c2_data, phi_angle, config, weights, angle_idx)
        logger.warning(
            "CMA-ES enabled in config but not available (cma not installed). "
            "Install with: uv add cma. Falling back."
        )

    # Multi-start is second priority. HAS_MULTISTART is hard-coded False
    # at module import (see top-of-file note), so this branch is
    # unreachable at runtime; the type: ignore documents the
    # OptimizationResult-vs-NLSQResult gap for dead code.
    if getattr(config, "multistart", False):
        if HAS_MULTISTART:
            logger.info("Multi-start enabled, delegating to multi-start optimizer")
            return _fit_multistart(  # type: ignore[return-value]
                model,
                c2_data,
                phi_angle,
                config,
                weights,
                use_nlsq_library,
            )
        logger.warning(
            "Multi-start enabled in config but multistart module not available. "
            "Falling back to local optimization."
        )

    return None


def _fit_cmaes(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float,
    config: NLSQConfig,
    weights: np.ndarray | jnp.ndarray | None,
    angle_idx: int = 0,
) -> NLSQResult:
    """Run CMA-ES global optimization with NLSQ warm-start and two-phase comparison.

    Phase structure (mirrors the homodyne CMA-ES path):

    - **Phase 1**: Local NLSQ refinement to get a warm-start point.
    - **Phase 2**: CMA-ES global search using the NLSQ result as initial guess.
      Calls :func:`xpcsjax.optimization.nlsq.cmaes_wrapper.fit_with_cmaes`
      with its real positional signature
      ``(model_func, xdata, ydata, p0, bounds, sigma, config)``. The previous
      port called it with a homemade keyword API
      (``objective_fn=, residual_fn=, n_data=, anti_degeneracy=``) that no
      longer exists; mypy flagged it and the smoke tests never reached the
      branch. Fixed here so v0.1 actually delivers on the "CMA-ES global
      search for multi-scale problems" claim for heterodyne.
    - **Phase 3**: Compare NLSQ vs CMA-ES by least-squares cost, keep the
      better result. ``CMAESResult`` exposes ``chi_squared`` (sum of squared
      residuals); we halve it to compare against NLSQ's
      ``final_cost = 0.5 * SSR`` convention.
    """
    from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESWrapperConfig

    param_manager = model.param_manager

    initial_varying = param_manager.get_initial_values()
    lower_bounds, upper_bounds = param_manager.get_bounds()
    initial_varying = np.clip(initial_varying, lower_bounds, upper_bounds)

    c2_jax = jnp.asarray(c2_data, dtype=jnp.float64)
    weights_jax = jnp.asarray(weights, dtype=jnp.float64) if weights is not None else None
    t, q, dt = model.t, model.q, model.dt
    contrast_val, offset_val = model.scaling.get_for_angle(angle_idx)

    # ------------------------------------------------------------------
    # Phase 1: NLSQ warm-start
    # ------------------------------------------------------------------
    nlsq_result: NLSQResult | None = None
    cmaes_x0 = initial_varying

    try:
        logger.info("CMA-ES Phase 1: NLSQ warm-start refinement")
        nlsq_result = _fit_local(
            model,
            c2_data,
            phi_angle,
            config,
            weights,
            use_nlsq_library=config.use_nlsq_library,
            angle_idx=angle_idx,
        )
        if nlsq_result.success:
            cmaes_x0 = nlsq_result.parameters.copy()
            logger.info(
                "NLSQ warm-start succeeded: cost=%.6e, chi2_red=%.4f",
                nlsq_result.final_cost or float("inf"),
                nlsq_result.reduced_chi_squared or float("inf"),
            )
        else:
            logger.warning(
                "NLSQ warm-start failed (%s), using raw initial params for CMA-ES",
                nlsq_result.message,
            )
    except (ValueError, RuntimeError, ImportError) as e:
        logger.warning(
            "NLSQ warm-start raised %s: %s — proceeding with raw p0",
            type(e).__name__,
            e,
        )

    # ------------------------------------------------------------------
    # Phase 2 auto-skip (parity with homodyne core.py:2296-2362)
    # ------------------------------------------------------------------
    # When the NLSQ warm-start already lands a good fit (reduced χ² below
    # threshold), skip the expensive CMA-ES global search — a warm-started
    # CMA-ES is a local refinement that rarely improves on a good NLSQ solution.
    # Honors ``cmaes_warmstart_auto_skip`` / ``cmaes_warmstart_skip_threshold``,
    # which were previously dropped on the heterodyne per-angle path (only
    # laminar_flow's core.py honored them), so a "skip when the warm-start is
    # good" run silently still paid for the full global search.
    warmstart_auto_skip = bool(getattr(config, "cmaes_warmstart_auto_skip", True))
    warmstart_skip_threshold = float(
        getattr(config, "cmaes_warmstart_skip_threshold", 5.0)
    )
    if (
        warmstart_auto_skip
        and nlsq_result is not None
        and nlsq_result.success
        and nlsq_result.reduced_chi_squared is not None
        and np.isfinite(nlsq_result.reduced_chi_squared)
        and nlsq_result.reduced_chi_squared < warmstart_skip_threshold
    ):
        logger.info(
            "CMA-ES auto-skip: NLSQ warm-start reduced χ²=%.4f < threshold=%.1f; "
            "skipping CMA-ES global search.",
            nlsq_result.reduced_chi_squared,
            warmstart_skip_threshold,
        )
        # ``_fit_local`` already left the model at the warm-start params and set
        # fitted_correlation / reduced_chi_squared, so the result is complete —
        # we only re-tag the optimizer metadata to reflect the CMA-ES context.
        nlsq_result.metadata["optimizer"] = "cmaes"
        nlsq_result.metadata["cmaes_winner"] = "nlsq_warmstart_auto_skip"
        nlsq_result.metadata["cmaes_skipped"] = True
        nlsq_result.metadata["warmstart_skip_threshold"] = warmstart_skip_threshold
        # Diagnostics-contract symmetry with the joint escapes (which tag
        # nlsq_diagnostics["global_escape"]). For the per-angle path the tag
        # rides in per-angle metadata — the only field aggregated into
        # ``nlsq_diagnostics["per_angle_metadata"]``.
        nlsq_result.metadata["global_escape"] = "cmaes_warmstart_auto_skip"
        nlsq_result.metadata["quality_flag"] = classify_quality_flag(
            nlsq_result.reduced_chi_squared
        )
        _log_result(nlsq_result)
        return nlsq_result

    # Ensure model parameters are reset for CMA-ES (NLSQ may have modified them)
    model.set_params(param_manager.expand_varying_to_full(initial_varying))

    # ------------------------------------------------------------------
    # Phase 2: CMA-ES global optimization
    # ------------------------------------------------------------------
    # Build the ``model_func(xdata, *params) -> ydata_flat`` closure that
    # fit_with_cmaes requires. xdata is a dummy index array — the heterodyne
    # kernel pulls t/q/dt/phi/contrast/offset from closure, not from xdata.
    #
    # IMPORTANT (tracer-safety): CMA-ES wraps this closure in
    # ``normalized_model_func`` (cmaes_wrapper.py:967) which passes JAX
    # *tracers* for ``varying_params`` when JIT-tracing the parameter
    # normalization. Mixing numpy assignment (``full[idx] = tracer``) with
    # tracer values raises ``ValueError: setting an array element with a
    # sequence``. Use pure-JAX scatter (``.at[].set()``) instead so the
    # closure JIT-traces cleanly.
    full_template_jax = jnp.asarray(param_manager.get_full_values(), dtype=jnp.float64)
    varying_indices_jax = jnp.asarray(list(param_manager.varying_indices), dtype=jnp.int32)

    def model_func(_: np.ndarray, *varying_params: Any) -> Any:
        varying_jax = jnp.stack(varying_params).astype(jnp.float64)
        full_jax = full_template_jax.at[varying_indices_jax].set(varying_jax)
        c2_pred = compute_c2_heterodyne(full_jax, t, q, dt, phi_angle, contrast_val, offset_val)
        return c2_pred.flatten()

    ydata = np.asarray(c2_jax).flatten().astype(np.float64)
    xdata = np.arange(ydata.size, dtype=np.float64)
    if weights_jax is not None:
        weights_np = np.asarray(weights_jax).flatten().astype(np.float64)
        # weights = 1/σ² ⇒ σ = 1/√weights. Guard zeros (unweighted samples)
        # by passing σ = 1 there so they fall back to uniform weighting.
        safe_w = np.where(weights_np > 0, weights_np, 1.0)
        sigma = 1.0 / np.sqrt(safe_w)
    else:
        sigma = None

    # Build the wrapper config directly. Don't use
    # ``CMAESWrapperConfig.from_nlsq_config(config)`` here: that helper expects
    # the *homodyne* :class:`NLSQConfig` (different module, different field
    # names — heterodyne uses ``cmaes_tolx`` / ``cmaes_tolfun`` /
    # ``cmaes_max_iterations`` where homodyne has ``cmaes_tol_x`` /
    # ``cmaes_tol_fun`` / ``cmaes_max_generations``). Pyright correctly flags
    # the cross-class pass; mapping the heterodyne fields by hand is the right
    # answer until the two NLSQConfigs converge in Phase 6.
    cmaes_wrapper_config = CMAESWrapperConfig(
        # Reproducibility: pin the RNG seed, offset per angle so the N searches
        # are decorrelated. Without this the per-angle path left seed=None →
        # non-reproducible, unlike the seed-pinned joint escapes.
        seed=_PER_ANGLE_CMAES_SEED + angle_idx,
        # Honor the configured CMA-ES initial step size. NOTE: this is the
        # ``sigma`` *config field* (initial step, fraction of search range), NOT
        # the ``sigma=`` argument to fit_with_cmaes below (per-point measurement
        # uncertainty). cmaes_sigma0 was previously dropped → wrapper used 0.5.
        sigma=float(getattr(config, "cmaes_sigma0", 0.5)),
        max_generations=getattr(config, "cmaes_max_iterations", None),
        popsize=getattr(config, "cmaes_population_size", None),
        tol_x=float(getattr(config, "cmaes_tolx", 1e-8)),
        tol_fun=float(getattr(config, "cmaes_tolfun", 1e-8)),
        restart_strategy=str(getattr(config, "cmaes_restart_strategy", "bipop")),
        max_restarts=int(getattr(config, "cmaes_max_restarts", 9)),
    )
    logger.info("CMA-ES Phase 2: global search (warm-started)")
    # Invariant: this function is only entered from ``_try_global_optimization``
    # when ``HAS_CMAES`` is True, which is True iff ``fit_with_cmaes`` was
    # imported. Narrow for Pyright.
    assert fit_with_cmaes is not None, "HAS_CMAES guards entry to _fit_cmaes"
    cmaes_result = fit_with_cmaes(
        model_func=model_func,
        xdata=xdata,
        ydata=ydata,
        p0=np.asarray(cmaes_x0, dtype=np.float64),
        bounds=(lower_bounds, upper_bounds),
        sigma=sigma,
        config=cmaes_wrapper_config,
    )

    # ------------------------------------------------------------------
    # Phase 3: Compare NLSQ vs CMA-ES, keep the better result
    # ------------------------------------------------------------------
    nlsq_cost = (
        float(nlsq_result.final_cost)
        if (nlsq_result and nlsq_result.success and nlsq_result.final_cost is not None)
        else float("inf")
    )
    # Recompute CMA-ES cost using off-diagonal residuals so the comparison
    # is on the same footing as nlsq_cost (= 0.5 * off-diagonal SSR).
    # cmaes_result.chi_squared uses the full NxN matrix fed to fit_with_cmaes
    # (including diagonal), which inflates the cost relative to NLSQ and
    # would always make CMA-ES appear worse, defeating Phase 3's purpose.
    if cmaes_result.success and cmaes_result.parameters is not None:
        try:
            _cmaes_full = param_manager.expand_varying_to_full(
                np.asarray(cmaes_result.parameters, dtype=np.float64)
            )
            _off_diag_res = compute_residuals(
                jnp.asarray(_cmaes_full, dtype=jnp.float64),
                t,
                q,
                dt,
                phi_angle,
                c2_jax,
                weights_jax,
                contrast_val,
                offset_val,
            )
            cmaes_cost = 0.5 * float(jnp.sum(_off_diag_res**2))
        except Exception as exc:
            logger.warning(
                "Phase 3: CMA-ES cost computation failed (%s); treating as inf so "
                "the NLSQ result wins by default. Inspect the off-diagonal residual "
                "block if this recurs.",
                exc,
                exc_info=True,
            )
            cmaes_cost = float("inf")
    else:
        cmaes_cost = float("inf")

    if nlsq_cost <= cmaes_cost and nlsq_result is not None and nlsq_result.success:
        result = nlsq_result
        winner = "nlsq"
        logger.info(
            "Phase 3: NLSQ wins (cost=%.6e vs CMA-ES=%.6e)",
            nlsq_cost,
            cmaes_cost,
        )
    else:
        result = _cmaes_to_nlsq_result(
            cmaes_result, cmaes_cost, parameter_names=param_manager.varying_names
        )
        winner = "cmaes"
        logger.info(
            "Phase 3: CMA-ES wins (cost=%.6e vs NLSQ=%.6e)",
            cmaes_cost,
            nlsq_cost,
        )

    # ------------------------------------------------------------------
    # Post-fit: update model, classify quality
    # ------------------------------------------------------------------
    if result.success:
        full_fitted = param_manager.expand_varying_to_full(result.parameters)
        fitted_c2 = compute_c2_heterodyne(
            jnp.asarray(full_fitted), t, q, dt, phi_angle, contrast_val, offset_val
        )
        result.fitted_correlation = np.asarray(fitted_c2)
        model.set_params(full_fitted)

    # Apply same chi2 correction as _fit_local (DOF + σ² normalization)
    if result.final_cost is not None:
        n_matrix = c2_jax.shape[0]
        n_valid = c2_jax.size - n_matrix
        n_dof_valid = max(n_valid - len(param_manager.varying_names), 1)
        c2_np = np.asarray(c2_jax)
        row_idx = np.arange(n_matrix)
        lag_mat = np.abs(row_idx[:, None] - row_idx[None, :])
        far_vals = c2_np[lag_mat >= n_matrix // 2]
        sigma2_noise = float(np.var(far_vals)) if far_vals.size > 1 else 0.0
        if sigma2_noise > 1e-12:
            ssr = 2.0 * result.final_cost
            result.reduced_chi_squared = ssr / (sigma2_noise * n_dof_valid)

    quality_flag = classify_quality_flag(result.reduced_chi_squared)
    result.metadata["optimizer"] = "cmaes"
    result.metadata["cmaes_winner"] = winner
    result.metadata["cmaes_cost"] = cmaes_cost
    result.metadata["nlsq_warmstart_cost"] = nlsq_cost
    result.metadata["quality_flag"] = quality_flag
    # Diagnostics-contract symmetry with the joint escapes (which tag
    # nlsq_diagnostics["global_escape"]). For the per-angle path the tag rides
    # in per-angle metadata: "cmaes" when CMA-ES won Phase 3, else
    # "cmaes_warmstart_kept" (CMA-ES ran but the NLSQ warm-start was kept) —
    # mirroring the joint escape's "<kind>" / "<kind>_warmstart_kept" values.
    result.metadata["global_escape"] = (
        "cmaes" if winner == "cmaes" else "cmaes_warmstart_kept"
    )

    _log_result(result)
    return result


def _cmaes_to_nlsq_result(
    cmaes_result: Any,
    final_cost: float,
    *,
    parameter_names: list[str],
) -> NLSQResult:
    """Pack a :class:`CMAESResult` into the :class:`NLSQResult` shape so
    downstream consumers (DOF correction, post-fit logging, multi-phi joining)
    see a uniform structure regardless of which optimizer won Phase 3.

    Naming convention: ``final_cost = 0.5 * SSR`` matches NLSQ's least-squares
    convention; CMA-ES reports ``chi_squared = SSR`` so the caller already
    halved it before passing it in.
    """
    diag = dict(cmaes_result.diagnostics) if cmaes_result.diagnostics else {}
    return NLSQResult(
        parameters=np.asarray(cmaes_result.parameters),
        parameter_names=list(parameter_names),
        success=bool(cmaes_result.success),
        message=str(cmaes_result.message),
        covariance=np.asarray(cmaes_result.covariance)
        if cmaes_result.covariance is not None
        else None,
        final_cost=final_cost,
        n_iterations=int(diag.get("generations", 0)),
        n_function_evals=int(diag.get("evaluations", 0)),
        convergence_reason=str(diag.get("convergence_reason", "")),
        metadata={"cmaes_diagnostics": diag},
    )


# Phase-6 minimal stub: delegates to the standard joint Fourier fit so the
# return shape is ``OptimizationResult``.  A real multistart implementation
# (LHS sampling over physics priors + perturbation + best-by-chi-squared
# selection) wired against ``run_multistart_nlsq`` lands in a later phase
# alongside a heterodyne-shaped ``single_fit_func`` adapter.
#
# Note this entry is per-angle (signature parallels ``_fit_local`` /
# ``_fit_cmaes`` — scalar ``phi_angle``, single ``(N, N)`` ``c2_data``).
# The dispatcher at ~line 1175 is also gated behind ``HAS_MULTISTART`` which
# is hard-coded ``False`` at module import, so this body is unreachable in
# v0.1; the conversion is purely about getting the return shape right so the
# top-level ``fit_nlsq_multi_phi`` annotation (Task C5) can be tightened.
def _fit_multistart(
    _model: HeterodyneModel,
    _c2_data: np.ndarray | jnp.ndarray,
    _phi_angle: float,
    _config: NLSQConfig,
    _weights: np.ndarray | jnp.ndarray | None,
    _use_nlsq_library: bool,
) -> OptimizationResult:
    """Heterodyne multistart escape (Phase-6 minimal stub).

    Currently delegates to the standard joint Fourier fit
    (:func:`_fit_joint_multi_phi`) with a single-phi batch (the per-angle
    ``c2_data`` is wrapped as a length-1 stack) so callers receive a
    uniform :class:`OptimizationResult`.  Full LHS multistart over physics
    priors with best-by-chi-squared selection lands when Phase 6's
    ``run_multistart_nlsq`` adapter work is completed.

    Parameter names retain the leading underscore (``_model``, etc.) because
    this body forwards them through ``_fit_joint_multi_phi``; the dispatcher
    at ``_try_global_optimization`` calls this positionally, so the order
    must stay aligned with ``_fit_cmaes`` / ``_fit_local``.
    """
    from xpcsjax.optimization.nlsq.fourier_reparam import (
        FourierReparamConfig,
        FourierReparameterizer,
    )

    c2_array = np.asarray(_c2_data)
    if c2_array.ndim == 2:
        c2_batch = c2_array[np.newaxis, ...]
    else:
        c2_batch = c2_array
    phi_angles_array = np.asarray([_phi_angle], dtype=np.float64)

    fourier_config = FourierReparamConfig(
        mode="fourier",
        fourier_order=_config.fourier_order,
        auto_threshold=_config.fourier_auto_threshold,
    )
    phi_rad = np.deg2rad(phi_angles_array)
    fourier = FourierReparameterizer(phi_rad, fourier_config)
    return _fit_joint_multi_phi(
        model=_model,
        c2_data=c2_batch,
        phi_angles=phi_angles_array,
        config=_config,
        weights=_weights if _weights is None else np.asarray(_weights),
        fourier=fourier,
    )


def _fit_local(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float,
    config: NLSQConfig,
    weights: np.ndarray | jnp.ndarray | None,
    use_nlsq_library: bool,
    angle_idx: int = 0,
) -> NLSQResult:
    """Run local (single-start) optimization with adapter/wrapper fallback.

    Tries adapter first; on failure falls back to wrapper with progressive
    recovery.
    """
    t_start = time.perf_counter()

    param_manager = model.param_manager
    varying_names = param_manager.varying_names
    n_varying = param_manager.n_varying

    logger.info("Fitting %d parameters: %s", n_varying, varying_names)

    # Memory-aware strategy selection. ``HAS_MEMORY`` is True iff both
    # ``select_nlsq_strategy`` and ``NLSQStrategy`` imported successfully —
    # narrow on the names themselves so Pyright sees them as bound.
    if select_nlsq_strategy is not None and NLSQStrategy is not None:
        n_data_est = np.asarray(c2_data).size
        decision = select_nlsq_strategy(n_data_est, n_varying)
        if decision.strategy in (NLSQStrategy.LARGE, NLSQStrategy.STREAMING):
            logger.debug(
                "Estimated peak memory (%.2f GB) exceeds threshold (%.2f GB). "
                "Fit may fail with OOM.",
                decision.peak_memory_gb,
                decision.threshold_gb,
            )

    # Get initial values and bounds
    initial_varying = param_manager.get_initial_values()
    lower_bounds, upper_bounds = param_manager.get_bounds()
    initial_varying = np.clip(initial_varying, lower_bounds, upper_bounds)

    # Convert data to JAX arrays
    c2_jax = jnp.asarray(c2_data, dtype=jnp.float64)
    weights_jax = jnp.asarray(weights, dtype=jnp.float64) if weights is not None else None

    if weights_jax is not None and weights_jax.shape != c2_jax.shape:
        raise ValueError(
            f"Weights shape {weights_jax.shape} does not match data shape {c2_jax.shape}"
        )

    # Capture constants
    fixed_values = jnp.asarray(param_manager.get_full_values(), dtype=jnp.float64)
    varying_indices = jnp.array(param_manager.varying_indices)
    n_data = c2_jax.size
    t, q, dt = model.t, model.q, model.dt

    # Per-angle scaling — fixed during local optimization (constant mode parity)
    contrast_val, offset_val = model.scaling.get_for_angle(angle_idx)

    # Build residual functions
    def jax_residual_fn(_x: jnp.ndarray, *varying_params: float) -> jnp.ndarray:
        """Pure JAX residual function for nlsq tracing."""
        varying_array = jnp.array(varying_params, dtype=jnp.float64)
        full_params = fixed_values.at[varying_indices].set(varying_array)
        return compute_residuals(
            full_params,
            t,
            q,
            dt,
            phi_angle,
            c2_jax,
            weights_jax,
            contrast_val,
            offset_val,
        )

    numpy_residual_fn = _make_numpy_residual_fn(
        model, c2_data, phi_angle, weights, contrast_val, offset_val
    )

    # ------------------------------------------------------------------
    # Adapter → wrapper fallback chain
    # ------------------------------------------------------------------
    adapter_error: Exception | None = None
    fallback_occurred = False
    result: NLSQResult | None = None

    if use_nlsq_library and NLSQAdapter is not None:  # HAS_ADAPTERS equivalent
        try:
            adapter = NLSQAdapter(parameter_names=varying_names)
            logger.debug("Attempting optimization with NLSQAdapter (JAX)")

            result = adapter.fit_jax(
                jax_residual_fn=jax_residual_fn,
                initial_params=initial_varying,
                bounds=(lower_bounds, upper_bounds),
                config=config,
                n_data=n_data,
            )

            if result.success:
                logger.info("NLSQAdapter optimization succeeded")
            else:
                raise RuntimeError(f"Adapter returned success=False: {result.message}")

        except (ValueError, RuntimeError, TypeError, ImportError, OSError) as e:
            adapter_error = e
            logger.warning("NLSQAdapter failed, falling back to wrapper: %s", e)
            fallback_occurred = True
            result = None

    # Wrapper fallback (or primary if use_nlsq_library=False)
    if result is None and NLSQWrapper is not None:  # HAS_WRAPPER equivalent
        try:
            wrapper = NLSQWrapper(parameter_names=varying_names)
            logger.debug("Attempting optimization with NLSQWrapper")

            result = wrapper.fit(
                residual_fn=numpy_residual_fn,
                initial_params=initial_varying,
                bounds=(lower_bounds, upper_bounds),
                config=config,
            )

            if fallback_occurred:
                logger.info("NLSQWrapper fallback succeeded")
            else:
                logger.info("NLSQWrapper optimization succeeded")

        except (ValueError, RuntimeError, TypeError, MemoryError) as wrapper_error:
            logger.error(
                "Both adapter and wrapper failed: adapter=%s, wrapper=%s",
                adapter_error,
                wrapper_error,
            )
            result = NLSQResult(
                parameters=initial_varying,
                parameter_names=varying_names,
                success=False,
                message=f"All optimizers failed. Adapter: {adapter_error}; "
                f"Wrapper: {wrapper_error}",
            )

    if result is None:
        raise ImportError(
            "No NLSQ optimization backend available. "
            "Ensure heterodyne.optimization.nlsq.adapter is importable."
        )

    # ------------------------------------------------------------------
    # Post-fit: compute fitted correlation, update model
    # ------------------------------------------------------------------
    if result.success:
        full_fitted = param_manager.expand_varying_to_full(result.parameters)
        fitted_c2 = compute_c2_heterodyne(
            jnp.asarray(full_fitted),
            t,
            q,
            dt,
            phi_angle,
            contrast_val,
            offset_val,
        )
        result.fitted_correlation = np.asarray(fitted_c2)
        model.set_params(full_fitted)

    # ------------------------------------------------------------------
    # Post-fit: correct reduced chi-squared
    #
    # The raw chi2 from adapter.fit_jax is SSR / (N² − n_params), where
    # SSR = Σ r² over the full N×N residual vector.  Two corrections:
    #
    #   1. DOF: the N diagonal residuals are forced to 0 by the
    #      non_diagonal mask in compute_residuals — they should be
    #      excluded from the degrees-of-freedom count.
    #      n_valid = N*(N−1) instead of N².
    #
    #   2. σ² normalization: without dividing by measurement noise,
    #      chi2 = MSE ≪ 1 for normalized C2 data (C2 ~ 1, residuals ~ 5%).
    #      We estimate σ²_noise from the far-lag plateau of the C2 matrix
    #      (|t2−t1| ≥ N/2), where correlations have fully decayed and
    #      the remaining variance is photon-counting noise.
    #
    # chi2_corrected = SSR / (σ²_noise × n_dof_valid)  →  ~1 for good fits
    # ------------------------------------------------------------------
    if result.final_cost is not None:
        n_matrix = c2_jax.shape[0]
        n_valid = c2_jax.size - n_matrix  # exclude N diagonal zeros
        n_dof_valid = max(n_valid - n_varying, 1)

        c2_np = np.asarray(c2_jax)
        row_idx = np.arange(n_matrix)
        lag_mat = np.abs(row_idx[:, None] - row_idx[None, :])
        far_mask = lag_mat >= n_matrix // 2  # diagonal (lag=0) not included
        far_vals = c2_np[far_mask]
        sigma2_noise = float(np.var(far_vals)) if far_vals.size > 1 else 0.0

        if sigma2_noise > 1e-12:
            ssr = 2.0 * result.final_cost
            chi2_corrected = ssr / (sigma2_noise * n_dof_valid)
            logger.debug(
                "chi2 correction: σ²_noise=%.4e  n_valid=%d  SSR=%.4e  "
                "raw_chi2=%.4g → chi2_corrected=%.4f",
                sigma2_noise,
                n_valid,
                ssr,
                result.reduced_chi_squared or float("nan"),
                chi2_corrected,
            )
            result.reduced_chi_squared = chi2_corrected
        else:
            logger.warning(
                "chi2 noise estimate near-zero (σ²=%.2e); reporting uncorrected MSE chi2",
                sigma2_noise,
            )

    result.metadata["fallback_occurred"] = fallback_occurred
    if adapter_error is not None:
        result.metadata["adapter_error"] = str(adapter_error)
    result.metadata["optimizer"] = "local"
    result.metadata["wall_time_total"] = time.perf_counter() - t_start

    _log_result(result)
    return result


def _make_numpy_residual_fn(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float,
    weights: np.ndarray | jnp.ndarray | None,
    contrast: float = 1.0,
    offset: float = 1.0,
) -> Any:
    """Create a numpy residual function closed over model/data.

    Returns a callable ``(varying_params: np.ndarray) -> np.ndarray``.

    Hot-path optimisation: ``fixed_values`` and ``varying_indices`` are
    pre-captured as JAX device arrays at construction time so each call
    only performs a single ``jnp.asarray`` (for the incoming numpy vector)
    and one ``jnp.ndarray.at[].set()`` scatter instead of a Python loop
    plus a full host copy.
    """
    param_manager = model.param_manager
    c2_jax = jnp.asarray(c2_data, dtype=jnp.float64)
    weights_jax = jnp.asarray(weights, dtype=jnp.float64) if weights is not None else None
    t, q, dt = model.t, model.q, model.dt

    # Pre-capture as JAX device arrays — allocated once, reused every call.
    # NOTE: fixed_values snapshot is taken at construction time. Do not mutate
    # param_manager between construction and optimizer completion.
    fixed_values = jnp.asarray(param_manager.get_full_values(), dtype=jnp.float64)
    varying_indices = jnp.array(param_manager.varying_indices, dtype=jnp.int32)

    def residual_fn(varying_params: np.ndarray) -> np.ndarray:
        varying_jax = jnp.asarray(varying_params, dtype=jnp.float64)
        full_params = fixed_values.at[varying_indices].set(varying_jax)
        # Return JAX array directly — np.asarray() on the result here would
        # trigger TracerArrayConversionError when NLSQWrapper's @jit traces
        # this function with traced parameter scalars.
        return compute_residuals(  # type: ignore[return-value]
            full_params,
            t,
            q,
            dt,
            phi_angle,
            c2_jax,
            weights_jax,
            contrast,
            offset,
        )

    return residual_fn


def _log_result(result: NLSQResult) -> None:
    """Log optimization results summary."""
    logger.info("=" * 60)
    logger.info("NLSQ OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    status = "SUCCESS" if result.success else "FAILED"
    logger.info("Status: %s", status)
    logger.info("Message: %s", result.message)

    if result.final_cost is not None:
        logger.info("Final cost: %.6e", result.final_cost)
    if result.reduced_chi_squared is not None:
        logger.info("Reduced χ²: %.4f", result.reduced_chi_squared)
    if result.wall_time_seconds is not None:
        logger.info("Wall time: %.2f s", result.wall_time_seconds)

    if result.success:
        for name, val in zip(result.parameter_names, result.parameters, strict=True):
            unc_val = result.get_uncertainty(name)
            if unc_val is not None:
                logger.info("  %s: %.6g ± %.3g", name, val, unc_val)
            else:
                logger.info("  %s: %.6g", name, val)

    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI-facing joint-fit completion logging (homodyne parity)
#
# The homodyne path logs an "NLSQ OPTIMIZATION COMPLETE" block with the fitted
# physical parameters from ``core._log_optimization_results`` (called once at
# the end of ``core.fit_nlsq_jax``). The heterodyne multi-phi joint paths
# (averaged / constant / fourier / individual / hybrid_streaming) returned an
# ``OptimizationResult`` without ever emitting that block, so the two_component
# log jumped straight from the adapter fit to the CLI results table. The helper
# below restores parity: it is invoked once per analysis from the CLI-facing
# dispatch ``optimization.nlsq._fit_nlsq_heterodyne`` (the heterodyne analog of
# ``fit_nlsq_jax``), not from the per-angle / per-trial sub-fits.
# ---------------------------------------------------------------------------

# Per-angle scaling lives in ``nlsq_diagnostics`` under a mode-specific suffix
# (individual: plain, averaged: ``_quantile``, constant: ``_fixed``, fourier:
# ``_fitted``). The completion logger reads whichever is present so the
# mean-scaling line stays mode-agnostic.
_CONTRAST_DIAG_KEYS = (
    "contrast_per_angle",
    "contrast_per_angle_quantile",
    "contrast_per_angle_fixed",
    "contrast_per_angle_fitted",
)
_OFFSET_DIAG_KEYS = (
    "offset_per_angle",
    "offset_per_angle_quantile",
    "offset_per_angle_fixed",
    "offset_per_angle_fitted",
)


def _mean_scaling_from_diagnostics(
    diagnostics: dict[str, Any] | None,
) -> tuple[float | None, float | None]:
    """Return ``(mean_contrast, mean_offset)`` from whichever per-angle scaling
    key the active mode populated, or ``(None, None)`` if none is present."""
    if not diagnostics:
        return None, None

    def _first(keys: tuple[str, ...], scalar_key: str) -> float | None:
        for k in keys:
            v = diagnostics.get(k)
            if v is not None:
                arr = np.asarray(v, dtype=np.float64)
                if arr.size:
                    return float(np.nanmean(arr))
        s = diagnostics.get(scalar_key)
        return float(s) if s is not None else None

    return (
        _first(_CONTRAST_DIAG_KEYS, "averaged_contrast"),
        _first(_OFFSET_DIAG_KEYS, "averaged_offset"),
    )


def log_heterodyne_start(analysis_mode: str, per_angle_mode: str, n_phi: int) -> None:
    """Log the opening ``NLSQ OPTIMIZATION`` banner for the heterodyne dispatch.

    Mirrors the opening block ``core.fit_nlsq_jax`` emits for the homodyne /
    laminar_flow path so the two_component log opens with the same banner. The
    ``per_angle_mode`` reported here is the *requested* mode (``auto`` is not
    yet resolved at this point); the subsequent ``Per-angle dispatch`` line
    inside :func:`fit_nlsq_multi_phi` records the resolved effective mode.
    """
    logger.info("=" * 60)
    logger.info("NLSQ OPTIMIZATION")
    logger.info("=" * 60)
    logger.info("Analysis mode: %s", analysis_mode)
    logger.info("Per-angle mode: %s", per_angle_mode)
    logger.info("Angles: %d", n_phi)


def log_heterodyne_completion(
    result: OptimizationResult,
    varying_names: list[str],
    n_physics: int,
    n_phi: int,
) -> None:
    """Log a homodyne-parity ``NLSQ OPTIMIZATION COMPLETE`` block.

    Mirrors the block ``core._log_optimization_results`` emits for the
    homodyne / laminar_flow path so the two_component log carries the same
    status / χ² / fitted-physical-parameter summary. Pure logging — reads from
    ``result`` only, never mutates state.

    Physics parameters lead the joint vector for every mode **except**
    ``hybrid_streaming`` (which packs scaling first); that one case is handled
    explicitly so the per-name table is never mislabeled.
    """
    diag = result.nlsq_diagnostics or {}
    mode = diag.get("per_angle_mode", "?")
    params = np.asarray(result.parameters, dtype=np.float64)
    unc = (
        np.asarray(result.uncertainties, dtype=np.float64)
        if result.uncertainties is not None
        else None
    )

    logger.info("=" * 60)
    logger.info("NLSQ OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info("Status: %s", "SUCCESS" if result.success else "FAILED")
    logger.info("Per-angle mode: %s", mode)
    logger.info("Iterations: %d", result.iterations)
    logger.info("Execution time: %.3fs", result.execution_time)
    logger.info("chi2 = %.6e", result.chi_squared)
    logger.info("Reduced chi2 = %.6f", result.reduced_chi_squared)
    logger.info("Quality: %s", result.quality_flag)

    if n_physics > 0 and params.size >= n_physics:
        if mode == "hybrid_streaming":
            phys_vals = params[-n_physics:]
            phys_unc = (
                unc[-n_physics:] if unc is not None and unc.size >= n_physics else None
            )
        else:
            phys_vals = params[:n_physics]
            phys_unc = (
                unc[:n_physics] if unc is not None and unc.size >= n_physics else None
            )

        logger.info("Fitted parameters (%d physical, %d angles):", n_physics, n_phi)
        logger.info("  Physical parameters:")
        for i, name in enumerate(varying_names[:n_physics]):
            unc_val = float(phys_unc[i]) if phys_unc is not None else 0.0
            logger.info("    %s: %.6g +/- %.6g", name, float(phys_vals[i]), unc_val)

    mean_contrast, mean_offset = _mean_scaling_from_diagnostics(diag)
    if mean_contrast is not None and mean_offset is not None:
        logger.info(
            "  Mean scaling: contrast=%.4f, offset=%.4f", mean_contrast, mean_offset
        )

    logger.info("=" * 60)

    # Laminar-parity anti-degeneracy DEFENSE summary. Read from the assembled
    # ``nlsq_diagnostics`` so the reported layer activity is HONEST per path
    # (stratified-LS / sequential report inactive L2/L3; in-memory / streaming
    # report the layers they actually ran). Runs once per analysis for EVERY
    # heterodyne path because this completion helper is the shared chokepoint.
    from xpcsjax.optimization.nlsq.heterodyne_logging import (
        log_anti_degeneracy_defense,
    )

    log_anti_degeneracy_defense(diag)
