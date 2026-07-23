"""Core NLSQ fitting for heterodyne analysis.

Unified entry point for NLSQ optimization with:
- Global optimization selection (CMA-ES → multi-start → local)
- Adapter/wrapper fallback with automatic recovery
- Memory-aware strategy selection
- Per-angle and multi-angle fitting
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_args

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
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig, ResolvedPerAngleMode
from xpcsjax.optimization.nlsq.heterodyne_results import NLSQResult
from xpcsjax.optimization.nlsq.results import OptimizationResult
from xpcsjax.optimization.nlsq.validation import classify_quality_flag
from xpcsjax.utils.logging import get_logger, log_exception

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


# Loggers that emit the sub-solver failure noise of a CMA-ES / multistart
# warm-start PROBE. On a degenerate Jacobian (e.g. C044 ``two_component``) the
# warm-start NLSQ solve is EXPECTED to not converge — the keep-better floor
# reverts it to x0 and the global search (Phase 2) refines from there, so the
# run is healthy. But the adapter/wrapper/nlsq sub-loggers each scream
# ``ERROR/WARNING`` for that expected, recovered probe, which reads as a hard
# failure. ``_quiet_warm_start_probe_logging`` drops exactly those records for
# the duration of a probe. The fixed names below are passed to ``getLogger``
# (which CREATES them if absent) so the filter is in place even on the first
# nlsq call; numerics are untouched (only a logging filter is attached).
_WARM_START_PROBE_NOISE_LOGGERS = (
    "xpcsjax.optimization.nlsq.heterodyne_adapter",
    "xpcsjax.optimization.nlsq.heterodyne_result_builder",
    "xpcsjax.optimization.nlsq.heterodyne_core",
    "nlsq",
    "nlsq.curve_fit",
    "nlsq.least_squares",
    "nlsq.optimizer",
    "nlsq.optimizer.trf",
)

# Substrings identifying the EXPECTED, recovered-from failure records of a
# warm-start probe. The filter drops ONLY records matching one of these, so a
# genuinely unexpected message (an OOM, a dependency error, anything off-script)
# from the same loggers still propagates — no blanket observability blackout.
_WARM_START_PROBE_NOISE_PATTERNS = (
    "Optimization failed",  # nlsq.curve_fit + heterodyne_adapter convergence error
    "Inner optimization loop",  # nlsq.optimizer.trf trust-region inner-limit
    "Convergence reason",  # nlsq.least_squares nan-gradient summary
    "NLSQWrapper:",  # NLSQWrapper tier/attempt/retry messages
    "falling back to NLSQWrapper",  # heterodyne_core adapter fallback
    "NLSQ failed",  # heterodyne_result_builder
    "degraded data-only SSR",  # heterodyne_core L2 keep-better floor revert
)


class _WarmStartProbeNoiseFilter(logging.Filter):
    """Drop the known expected sub-solver failure records of a warm-start probe.

    Returns ``False`` (drop) only for records whose message contains one of
    :data:`_WARM_START_PROBE_NOISE_PATTERNS`; every other record passes through,
    so unexpected failures during the probe stay visible.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - never let a formatting error hide a record
            return True
        return not any(pat in message for pat in _WARM_START_PROBE_NOISE_PATTERNS)


@contextmanager
def _quiet_warm_start_probe_logging() -> Iterator[None]:
    """Attach a message-scoped noise filter to the probe loggers, then detach it.

    Suppresses the EXPECTED, recovered-from sub-solver failure barrage of a
    CMA-ES / multistart warm-start probe WITHOUT a blanket level mute: only
    records matching :data:`_WARM_START_PROBE_NOISE_PATTERNS` are dropped, so an
    unexpected error from the same loggers during the probe is still emitted. A
    non-degenerate problem produces no such records, so this is a no-op there.
    Idempotent per logger (a logger appearing twice gets one filter instance) and
    fully removed on exit. Changes ONLY logging — no solver behavior or numerics.
    """
    names = set(_WARM_START_PROBE_NOISE_LOGGERS)
    names.update(n for n in logging.root.manager.loggerDict if n.startswith("nlsq"))
    noise_filter = _WarmStartProbeNoiseFilter()
    loggers = [logging.getLogger(name) for name in names]
    for lg in loggers:
        lg.addFilter(noise_filter)
    try:
        yield
    finally:
        for lg in loggers:
            lg.removeFilter(noise_filter)


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

# Multi-start orchestration is intentionally NOT imported here: the
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
    *,
    scaling_first: bool = False,
) -> tuple[Any, Any]:
    """Build the L4 per-iteration gradient-collapse monitor and curve_fit callback.

    Returns ``(None, None)`` when gradient monitoring is disabled (so the caller
    builds no monitor and passes no callback, leaving the fit unchanged). When
    enabled, returns ``(monitor, callback)`` and the callback is strictly
    observational — Phase-0 proved NLSQ's curve_fit callback fires per-iteration
    and never perturbs the solve.

    The monitor partitions the joint vector into physical vs per-angle (scaling)
    index sets, which depends on the caller's layout:

    - ``scaling_first=False`` (default): PHYSICS-FIRST ``[physics | scaling tail]``
      — used by ``_fit_joint_averaged_multi_phi`` (x0 = ``[physics, contrast,
      offset]``).
    - ``scaling_first=True``: canonical SCALING-FIRST ``[scaling_head | physics]``
      — used by ``_fit_joint_multi_phi`` (x0 = ``[scaling_head, physics]``).

    Passing the wrong layout silently mislabels which gradients are "physical" vs
    "per-angle" in the diagnostics (L4 is observation-only, so the SOLVE is
    unaffected, but the reported ratios would be meaningless).
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
    n_scaling = max(0, total - n_physics)
    if scaling_first:
        # SCALING-FIRST [scaling_head | physics]: physics is the tail.
        physical_indices = list(range(n_scaling, total))
        per_angle_indices = list(range(0, n_scaling))
    else:
        # PHYSICS-FIRST [physics | scaling tail]: physics is the head.
        physical_indices = list(range(n_physics))
        per_angle_indices = list(range(n_physics, total))
    gm_cfg = GradientMonitorConfig(
        ratio_threshold=float(config.gradient_ratio_threshold),
        consecutive_triggers=int(config.gradient_consecutive_triggers),
        check_interval=1,
    )
    monitor = GradientCollapseMonitor(
        gm_cfg,
        physical_indices=physical_indices,
        per_angle_indices=per_angle_indices,
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
# an OptimizationResult — the averaged/individual joint paths here and the
# constant path in heterodyne_constant_mode.py via re-import)
# ---------------------------------------------------------------------------


def _build_heterodyne_diagnostics(
    per_angle_mode: str,
    chi2_per_angle: np.ndarray,
    scaling_source: str,
    **extras: Any,
) -> dict[str, Any]:
    """Build the standard heterodyne ``nlsq_diagnostics`` dict.

    Centralises the canonical keys every heterodyne-side
    :class:`OptimizationResult` carries so the averaged/individual joint paths
    here and the constant-mode joint path in :mod:`heterodyne_constant_mode`
    stay in lockstep. Extra mode-specific keys (e.g.
    ``contrast_per_angle_fixed`` in constant mode) are passed through
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
    # A caller (e.g. the streaming result builder) may supply n_optimized
    # explicitly; that takes precedence over deriving it from the mode token.
    explicit_n_optimized = extras.pop("n_optimized", None)

    # Optimizer scaling-head length (constant -> 0, averaged -> 2,
    # individual -> 2*n_phi). Mirrors the laminar standard-path key
    # (``wrapper.py``) and the streaming/constant-mode paths so the
    # ``n_optimized`` diagnostic is symmetric across every heterodyne path.
    # Only the joint paths pass a RESOLVED scaling token here; the streaming
    # path passes a path label (e.g. ``"hybrid_streaming"``) for which the
    # scaling-head length is not derivable from the token alone — in that case
    # honour an explicit value if given, else omit the key (the streaming
    # anti_degeneracy block already carries its own n_optimized).
    from xpcsjax.optimization.nlsq.per_angle_mode import (
        PerAngleMode,
        n_optimized,
    )

    n_phi = int(np.asarray(chi2_per_angle).size)

    base: dict[str, Any] = {
        "per_angle_mode": per_angle_mode,
        "chi2_per_angle": chi2_per_angle,
        "scaling_source": scaling_source,
    }
    if explicit_n_optimized is not None:
        base["n_optimized"] = int(explicit_n_optimized)
    elif per_angle_mode in get_args(PerAngleMode):
        base["n_optimized"] = n_optimized(per_angle_mode, n_phi)  # type: ignore[arg-type]
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
    retry with progressive recovery (``HybridRecoveryConfig``).

    Parameters
    ----------
    model : HeterodyneModel
        Model instance with parameters configured.
    c2_data : np.ndarray or jnp.ndarray
        Experimental correlation data, shape ``(N, N)``.
    phi_angle : float, optional
        Detector phi angle in degrees.
    config : NLSQConfig, optional
        NLSQ configuration; a default :class:`NLSQConfig` is used when ``None``.
    weights : np.ndarray or jnp.ndarray, optional
        Weights (``1/sigma**2``) for weighted least squares.
    use_nlsq_library : bool, optional
        Prefer the ``nlsq`` library over the SciPy fallback when ``True``.
    _skip_global_selection : bool, optional
        Internal flag — skip the CMA-ES / multi-start global selection gate.
    angle_idx : int, optional
        Per-angle scaling index for the fixed contrast/offset values.

    Returns
    -------
    NLSQResult
        Fitted parameters and diagnostics.

    See Also
    --------
    fit_nlsq_multi_phi : Multi-angle public dispatcher for the joint fit.
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
            # Match the kernel / joint-path convention (n_per_angle =
            # (N-1)*(N-2)): exclude the diagonal AND the t=0 boundary row/col.
            # Masking only the diagonal leaks t=0 boundary residuals into the SSR,
            # inconsistent with every joint path's reduced-chi2.
            valid_mask = ~np.eye(n_matrix, dtype=bool)
            valid_mask[0, :] = False
            valid_mask[:, 0] = False
            # OptimizationResult.chi_squared is defined as data residual SSR.
            chi2_values.append(float(np.sum(residual[valid_mask] ** 2)))
        elif r.final_cost is not None:
            # On the sequential per-angle path the results are produced by
            # ``_fit_local``, whose own chi2 correction uses ``ssr = 2.0 *
            # result.final_cost`` (heterodyne_core.py ~4342) — i.e. for THIS path
            # final_cost follows the least-squares convention final_cost = 0.5*SSR
            # (the scipy/wrapper adapter cost), so 2*final_cost recovers SSR.
            # (The joint paths use a different builder where final_cost = full SSR
            # and recompute from raw residuals; do not conflate the two.)
            chi2_values.append(2.0 * float(r.final_cost))
        else:
            chi2_values.append(0.0)
    chi2_per_angle = np.asarray(chi2_values, dtype=np.float64)
    ssr = float(chi2_per_angle.sum())
    n_function_evals = int(sum(int(r.n_function_evals or 0) for r in per_angle_results))
    n_iterations_total = int(sum(int(r.n_iterations or 0) for r in per_angle_results))

    c2_arr = np.asarray(c2_data)
    if c2_arr.ndim == 3:
        # Per-angle valid points exclude the diagonal and the t=0 boundary row/col
        # (n_per_angle = (N-1)*(N-2)), matching the SSR masking above and every
        # joint path's convention.
        n_phi_c2, n_time_c2 = int(c2_arr.shape[0]), int(c2_arr.shape[1])
        n_data_total = n_phi_c2 * max(n_time_c2 - 1, 0) * max(n_time_c2 - 2, 0)
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
        n_physics=n_physics,
    )


def _should_hint_enable_escape(*, success: bool, enable_cmaes: bool, multistart: bool) -> bool:
    """Return whether a FAILED joint fit had NO global escape enabled.

    The actionable case: a degenerate 14-D ``two_component`` fit did not converge
    and neither the CMA-ES nor the multistart joint escape was enabled, so the
    global rescue that would likely reach a good basin (e.g. C044) never ran.
    """
    return (not success) and (not enable_cmaes) and (not multistart)


def log_enable_escape_hint(result: Any, config: Any) -> None:
    """Emit an actionable hint when a joint fit failed with no global escape on.

    Strictly diagnostic (no numeric effect). Silent when the fit converged or when
    a global escape was already enabled — see :func:`_should_hint_enable_escape`.
    The keep-better floor warm-starts the escape from the Stage-1 fit, so enabling
    it cannot worsen the result. Robust to missing attributes (defaults to a
    converged / escape-enabled reading, i.e. no hint) and never raises.
    """
    success = bool(getattr(result, "success", True))
    enable_cmaes = bool(getattr(config, "enable_cmaes", False))
    multistart = bool(getattr(config, "multistart", False))
    if not _should_hint_enable_escape(
        success=success, enable_cmaes=enable_cmaes, multistart=multistart
    ):
        return
    logger.warning(
        "Heterodyne joint fit did not converge and NO global escape is enabled. "
        "A degenerate 14-D two_component fit (e.g. C044) usually needs the CMA-ES "
        "global escape to reach a good basin. Enable it in the config under "
        "optimization.nlsq:\n"
        "    cmaes:\n"
        "      enable: true\n"
        "      n_seeds: 3   # >=3: single-seed CMA-ES is not run-to-run "
        "reproducible on the large objective; 3 seeds keep-best.\n"
        "The keep-better floor warm-starts the escape from the Stage-1 fit, so "
        "enabling it cannot worsen the result."
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
    - ``"individual"`` (explicit, multi-angle) → :func:`_fit_joint_multi_phi`
      (JOINT fit of the canonical scaling-first
      ``[2*n_phi per-angle scaling | physics]`` vector, matching ``laminar_flow``
      and upstream heterodyne) → :class:`OptimizationResult`
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
        Joint-fit result (constant / averaged / CMA-ES paths)
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

    The returned ``result.parameters`` is **physics-first** —
    ``[physics | contrast | offset]`` — which is the opposite of homodyne's
    scaling-first layout. The per-angle scaling tail is recovered with
    :func:`xpcsjax.optimization.nlsq.heterodyne_views.reconstruct_per_angle_scaling`.

    This package is NLSQ-only: there is no Bayesian / MCMC dispatch branch here.

    See Also
    --------
    fit_nlsq_jax : Single-angle NLSQ entry point.
    fit_two_component_via_engine : Shared-engine in-memory route used by
        ``_fit_nlsq_heterodyne`` for the in-scope per-angle modes.

    Examples
    --------
    >>> from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    >>> config = NLSQConfig(per_angle_mode="auto")
    >>> result = fit_nlsq_multi_phi(model, c2_data, phi_angles, config)
    >>> result.nlsq_diagnostics["per_angle_mode"]  # resolved dispatch token
    'averaged'
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
    if config is not None and len(phi_angles) > 1:
        # Resolve ``auto`` / explicit modes to a canonical dispatch token FIRST.
        # The resolver returns one of: "constant", "averaged", "individual"
        # (and rejects any other token). Resolving before the global-escape gate
        # is what keeps the scaling layout consistent: enabling CMA-ES /
        # multistart must NOT change which layout is used. The escapes honour
        # ``effective_mode`` — the ``auto → averaged`` default (and explicit
        # ``constant``) run their own ``[physics | scaling]`` global search.
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
            "effective=%s, escape=%s",
            config.per_angle_mode,
            len(phi_angles),
            config.constant_scaling_threshold,
            effective_mode,
            escape_kind,
        )

        # The individual escape uses the scaling-first ``_build_joint_problem``.
        if escape_kind is not None and effective_mode == "individual":
            if escape_kind == "cmaes":
                logger.info(
                    "CMA-ES enabled, delegating to joint multi-angle CMA-ES "
                    "(global search — this can run for several minutes with no "
                    "per-iteration log output; the next CMA-ES line appears at "
                    "phase boundaries / completion, not a hang)"
                )
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

        if effective_mode == "individual":
            # Explicit multi-angle ``individual`` is a JOINT fit (parity with
            # xpcsjax ``laminar_flow`` and upstream heterodyne). The per-angle
            # (contrast, offset) are packed as the ``2*n_phi`` scaling tail of
            # the canonical scaling-first joint vector
            # ``[contrast_0..N | offset_0..N | physics]`` and optimized jointly
            # with physics via ``_fit_joint_multi_phi`` (which builds the layout
            # from ``config`` itself). This replaces
            # the old sequential-per-angle aggregate (``mean(physics)`` reported
            # as ``parameters``), which was an inconsistent estimator whose
            # parameters did not reproduce the reported chi-squared. The
            # sequential aggregate (``_aggregate_individual_results``) survives
            # ONLY as the genuine fallback for ``config is None`` / single-angle
            # (``len(phi_angles) <= 1``) — both handled by this block's guard
            # (``config is not None and len(phi_angles) > 1``) being false.
            use_joint = True

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
    helper reconstructs the per-angle statistics so each ``NLSQResult`` carries
    its own diagnostics rather than a copy of the joint value.

    Parameters
    ----------
    residuals : np.ndarray
        Flat off-diagonal residual vector from ``compute_residuals``, length
        ``(n - 1) * (n - 2)`` (the kernel mask excludes both the t=0 boundary
        row/column and the ``t1 == t2`` diagonal).
    c2_matrix : np.ndarray
        Per-angle experimental C2 matrix, shape ``(n, n)``.
    n_params : int
        Number of varying physics parameters.

    Returns
    -------
    tuple of float
        ``(per_angle_cost, reduced_chi_squared)`` where ``per_angle_cost`` is
        ``0.5 * SSR`` and ``reduced_chi_squared`` is noise-normalised (target
        ``approx 1.0`` for a good fit; MSE fallback when noise is degenerate).
    """
    ssr = float(np.sum(residuals**2))
    per_angle_cost = 0.5 * ssr

    n_matrix = c2_matrix.shape[0]
    # Use the actual residual length: the kernel mask excludes the t=0 boundary
    # row/column AND the diagonal, so the valid count is (n-1)*(n-2), not the
    # N^2 - N that `size - n_matrix` (diagonal-only) would give.
    n_valid = int(residuals.size)
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
    ``auto → averaged`` default under CMA-ES / multistart instead of switching
    the scaling layout — matching the plain dispatch and laminar_flow's
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
        ``shear_weighting='not_applicable_heterodyne'``
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
        # The Stage-1 warm-start delegates to the (inherently per-angle)
        # constant-mode solver. Name the FINAL averaged layout (2 scaling DOF)
        # up front so the per-angle quantile arrays it logs are not misread as
        # ``individual`` mode — the final fit re-optimizes averaged scaling and
        # Stage-1's per-angle scaling is discarded.
        logger.info(
            "L2 Stage-1 warm-start (final mode: averaged): per-angle quantile "
            "scaling frozen for WARM-START ONLY; final fit optimizes 2 averaged "
            "scaling + %d physics = %d params",
            n_physics_varying,
            n_physics_varying + 2,
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
            warm_start_context="L2 Stage-1 -> final mode averaged",
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
    logger.info("Computing per-angle scaling from quantiles")
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
    # in the augmented residual) so behavioural-mode parity across the
    # per-angle scaling modes is preserved.
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
            # degenerate-CV behaviour for the averaged scaling layout.
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

    # L2 keep-better SSR floor (mirrors the individual-mode protection in
    # ``_fit_joint_multi_phi`` — the original C044 RCA fix): revert to the
    # pre-solve warm-start when the plain solve regressed past its own start,
    # BEFORE this (possibly degraded) vector is used as the escape warm-start.
    fitted_all, floor_reverted = _apply_joint_keep_better_floor(
        base_residual_fn, x0, fitted_all, mode_label="averaged mode"
    )

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
        warm_success=bool(joint_result.success) and not floor_reverted,
    )
    # A pure floor-revert (no escape) also loses the covariance / iteration
    # stats tied to the discarded solve — treat it as escape-shaped for the
    # NaN-fill + zeroed-stats bookkeeping below (matches individual-mode
    # convention: a reverted joint_result carries no valid covariance).
    is_escape = global_escape_tag is not None or floor_reverted
    if global_escape_tag == "cmaes_warmstart_auto_skip":
        # Auto-skip kept the CONVERGED warm-start vector UNCHANGED, so its NLSQ
        # covariance / uncertainties / iteration stats are valid — preserve them
        # (laminar parity: ``fit_nlsq_cmaes`` returns ``nlsq_warmstart_cov`` on
        # skip). Build as a plain result for stats; the ``global_escape`` tag is
        # set independently below, so the skip is still recorded.
        is_escape = False

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
    # heterodyne_constant_mode (the canonical chi2-decomposition helper).
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
    # n_total_params is the ACTUAL fitted-parameter count — used below to size
    # the NaN covariance / uncertainties arrays so they match `parameters`.
    n_total_params = int(joint_result.parameters.size)
    # For the compressed averaged layout, popt is [c_avg, o_avg, physics]
    # (n_physics + 2), but the constrained model consumes the EXPANDED
    # 2*n_phi + n_physics scaling DOF. Use the expanded constrained-model DOF
    # for reduced chi^2 ONLY (spec §5 decision 3), matching the large-data result
    # builders (heterodyne_result_builder) so explicit/auto averaged report a
    # consistent metric. SSR / chi2_per_angle / parameters / covariance shape are
    # untouched.
    from xpcsjax.optimization.nlsq.per_angle_mode import effective_constrained_dof

    _eff_dof = effective_constrained_dof("averaged", n_phi=n_phi, n_physical=n_physics_varying)
    _dof_params = int(_eff_dof) if _eff_dof is not None else n_total_params
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
        n_params=_dof_params,
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

    solve_success = bool(joint_result.success) and not floor_reverted
    convergence_status: ConvergenceStatus = "converged" if solve_success else "failed"
    quality_flag: QualityFlag = "good" if solve_success else "marginal"

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
    # (the K-row contract) so behavioural-mode parity across the per-angle
    # scaling modes is preserved.
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
        # PHYSICS-FIRST layout: x0 = [physics | contrast, offset] (see above).
        # The marker disambiguates this legacy averaged path from the engine
        # route's SCALING-FIRST averaged result for downstream readers.
        scaling_first=False,
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
        n_physics=int(n_physics_varying),
    )


# Base seed for the joint CMA-ES escape. ``CMAESWrapperConfig`` (and NLSQ's
# ``CMAESConfig``) default ``seed=None`` → fully non-reproducible; pinning it
# removes the sampler's own RNG as a variable.
#
# IMPORTANT — pinning the seed does NOT make the escape bit-reproducible run to
# run. The objective is a ~3M-point JAX residual whose XLA float-reduction order
# is not stable across runs, and the basin-fragile non-convex search amplifies
# that into different basins (RCA 2026-06-16: C044 ``two_component`` drew SSR
# 5546/beta=-0.41 on one run and SSR 8737/beta=-0.03 on the next, SAME seed,
# 221 vs 354 generations). ``cmaes_n_seeds`` (default 1) runs the escape over
# ``[_JOINT_CMAES_SEED + i]`` and keeps the lowest-SSR draw to raise the chance
# of landing the good basin — strictly keep-better (see
# :func:`_cmaes_keep_best_over_seeds`).
_JOINT_CMAES_SEED = 42

# Per-angle CMA-ES escape seed. Offset by ``angle_idx`` at the call site so each
# angle's stochastic search is individually reproducible yet decorrelated from
# the others (mirrors ``_JOINT_CMAES_SEED``'s pinning; a single shared seed would
# make every angle explore the identical random trajectory). Without this the
# per-angle ``_fit_cmaes`` path left ``CMAESWrapperConfig.seed=None`` →
# non-reproducible, unlike the seed-pinned joint escapes.
_PER_ANGLE_CMAES_SEED = 42


def _cmaes_keep_best_over_seeds(
    *,
    run_one_seed: Callable[[int], Any],
    seeds: Sequence[int],
    data_ssr: Callable[[np.ndarray], float],
) -> tuple[Any, float, int]:
    """Run the joint CMA-ES escape once per seed; keep the lowest-SSR draw.

    The joint global search is NOT bit-reproducible run to run (XLA reduction
    order over the ~3M-point objective; basin-fragile non-convex search), so a
    single seed gambles on the basin. Running ``len(seeds)`` separate draws
    and keeping the one with the smallest DATA-ONLY SSR raises the probability of
    landing the good basin. This is strictly keep-better: a worse draw can never
    displace a better one.

    Parameters
    ----------
    run_one_seed
        Runs the CMA-ES escape with the given seed, returning its
        :class:`OptimizationResult` (``success`` / ``parameters``).
    seeds
        Seeds to try, in order. ``len == 1`` returns that single run's result
        object verbatim (byte-identical to the pre-multiseed path).
    data_ssr
        Data-only SSR at a parameter vector — the keep-better unit (excludes any
        L3 penalty rows), matching the caller's warm-start comparison.

    Returns
    -------
    tuple
        ``(best_result, best_ssr, best_seed)``. A run that did not succeed (or
        has no parameters) scores ``+inf`` so it is only kept when nothing better
        exists (the caller still floors the survivor against the warm-start).
        Ties keep the EARLIER seed, so a single-seed run and the first seed of a
        multi-seed run select identically.
    """
    best_result: Any = None
    best_ssr = float("inf")
    best_seed = int(seeds[0])
    for seed in seeds:
        res = run_one_seed(int(seed))
        if getattr(res, "success", False) and res.parameters is not None:
            ssr = data_ssr(np.asarray(res.parameters, dtype=np.float64))
        else:
            ssr = float("inf")
        # Strict ``<`` ⇒ earlier seed wins ties; first iteration always seeds best.
        if best_result is None or ssr < best_ssr:
            best_result, best_ssr, best_seed = res, ssr, int(seed)
    assert best_result is not None, "seeds is non-empty so the loop ran at least once"
    return best_result, best_ssr, best_seed


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

    Notes
    -----
    This path is reached only for ``individual`` mode (the escape gate was
    narrowed in Task 6 to ``effective_mode == "individual"``).  The escape
    builds the :class:`JointProblem` via :func:`_build_joint_problem` which
    returns a scaling-first vector ``[scaling_head | physics]`` — the same
    layout the plain joint fit produces.

    When CMA-ES is kept, the returned :class:`OptimizationResult` is tagged via
    ``nlsq_diagnostics["global_escape"]`` and, by construction, carries NaN
    covariance / uncertainties and ``n_iterations == 0`` (no covariance solve is
    run on the kept vector). Read ``global_escape`` to detect an escape result.
    """
    try:
        _t_escape_start = time.perf_counter()
        prob = _build_joint_problem(model, c2_data, phi_angles, config, weights)

        # Phase 1: warm-start via the plain joint fit over the SAME problem.
        # This is a PROBE: on a degenerate Jacobian (e.g. C044 two_component) the
        # trust-region solve is EXPECTED to not converge — the keep-better floor
        # reverts it to x0 and Phase 2's global search refines from there, so the
        # run is healthy. The sub-solver ERROR/WARNING barrage for that expected,
        # recovered probe is demoted to keep the log honest (numerics untouched).
        logger.info(
            "Joint CMA-ES escape: Phase 1/2 — warm-start probe (NLSQ trust-region). "
            "Non-convergence here is EXPECTED on a degenerate Jacobian and harmless: "
            "it reverts to x0 and the Phase 2 global search refines from there. "
            "Sub-solver failure diagnostics for this probe are demoted to DEBUG."
        )
        with _quiet_warm_start_probe_logging():
            warm = _fit_joint_multi_phi(
                model=model,
                c2_data=c2_data,
                phi_angles=np.asarray(phi_angles),
                config=config,
                weights=weights,
                prob=prob,
                warm_start_probe=True,
            )
        x_warm = np.asarray(warm.parameters, dtype=np.float64)
        ssr_warm = float(warm.chi_squared)

        # Data-only SSR (excludes any L3 penalty rows) so the keep-better
        # comparison is apples-to-apples with ``warm.chi_squared``.
        base_residual_fn = prob.meta["base_residual_fn"]

        def _data_ssr(x: np.ndarray) -> float:
            return float(np.sum(np.asarray(base_residual_fn(x), dtype=np.float64) ** 2))

        # Phase 2 auto-skip — parity with laminar core.py:2354-2382 and the
        # heterodyne per-angle escape (:func:`_fit_cmaes`): when the warm-start
        # already lands a good fit (reduced χ² below threshold), skip the
        # expensive global search and keep the warm-start.
        _n_data_warm = int(np.asarray(base_residual_fn(x_warm)).size)
        _skip, _reduced = _warmstart_auto_skip_decision(
            config, "cmaes", ssr_warm, _n_data_warm, int(x_warm.size), bool(warm.success)
        )
        if _skip:
            logger.info(
                "Joint CMA-ES escape auto-skip: warm-start reduced χ²=%.4f < "
                "threshold=%.1f; skipping global search (parity with laminar "
                "core.py auto-skip).",
                _reduced,
                float(getattr(config, "cmaes_warmstart_skip_threshold", 5.0)),
            )
            # Auto-skip kept the CONVERGED warm-start UNCHANGED — return it
            # verbatim so its real covariance / uncertainties / n_iterations are
            # preserved (laminar parity: ``fit_nlsq_cmaes`` returns
            # ``nlsq_warmstart_cov`` on skip), only re-tagged so callers see the
            # auto-skip. Rebuilding via ``_build_joint_result(joint_result=None)``
            # would NaN-fill the covariance the warm-start already solved.
            _diag = dict(warm.nlsq_diagnostics) if warm.nlsq_diagnostics else {}
            _diag["global_escape"] = "cmaes_warmstart_auto_skip"
            warm.nlsq_diagnostics = _diag
            return warm

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

        assert fit_with_cmaes is not None, "HAS_CMAES guards entry to the escape"

        # Multi-seed keep-best. The global search is NOT bit-reproducible run to
        # run (XLA float-reduction order over the ~3M-point objective varies, and
        # the basin-fragile non-convex search amplifies it into different basins —
        # RCA 2026-06-16, C044 ``two_component``: SSR 5546/beta=-0.41 vs SSR
        # 8737/beta=-0.03, SAME seed, 221 vs 354 generations). Running
        # ``cmaes_n_seeds`` separate draws and keeping the lowest data-only SSR
        # raises the chance of landing the good basin. ``cmaes_n_seeds`` defaults
        # to 1 ⇒ a single ``seed=_JOINT_CMAES_SEED`` draw, byte-identical to the
        # pre-multiseed path; draw ``i`` is seeded ``_JOINT_CMAES_SEED + i``.
        n_seeds = max(1, int(getattr(config, "cmaes_n_seeds", 1)))
        seeds = [_JOINT_CMAES_SEED + i for i in range(n_seeds)]

        def _run_one_seed(seed: int) -> Any:
            # Build the wrapper config by hand (NOT ``from_nlsq_config``): that
            # helper expects the *homodyne* NLSQConfig (different field names —
            # heterodyne uses ``cmaes_max_iterations`` / ``cmaes_tolx`` /
            # ``cmaes_tolfun``). Same rationale as the per-angle ``_fit_cmaes``.
            cfg_cmaes = CMAESWrapperConfig(
                seed=seed,
                refine_with_nlsq=True,
                # Honor the configured CMA-ES initial step size (config field,
                # NOT the sigma= arg to fit_with_cmaes). cmaes_sigma0 was
                # previously dropped → wrapper used its 0.5 default.
                sigma=float(getattr(config, "cmaes_sigma0", 0.5)),
                max_generations=getattr(config, "cmaes_max_iterations", None),
                popsize=getattr(config, "cmaes_population_size", None),
                tol_x=float(getattr(config, "cmaes_tolx", 1e-8)),
                tol_fun=float(getattr(config, "cmaes_tolfun", 1e-8)),
                restart_strategy=str(getattr(config, "cmaes_restart_strategy", "bipop")),
                max_restarts=int(getattr(config, "cmaes_max_restarts", 9)),
            )
            logger.info(
                "Joint CMA-ES escape: Phase 2/2 — global search over %d-dim "
                "residual (seed=%d, warm SSR=%.6e; this is the long, silent step "
                "— minutes are normal, not a hang)",
                rdim,
                seed,
                ssr_warm,
            )
            return fit_with_cmaes(
                model_func=model_func,
                xdata=np.arange(rdim, dtype=np.float64),
                ydata=np.zeros(rdim, dtype=np.float64),
                p0=x_warm,
                bounds=(prob.lb, prob.ub),
                sigma=None,
                config=cfg_cmaes,
            )

        if n_seeds > 1:
            logger.info(
                "Joint CMA-ES escape: multi-seed keep-best over %d seeds %s — the "
                "non-convex objective is not run-to-run reproducible; keeping the "
                "lowest data-only SSR draw (strictly keep-better).",
                n_seeds,
                seeds,
            )

        # Phase 3: keep-better. ``_cmaes_keep_best_over_seeds`` already computes
        # the DATA-ONLY SSR (``_data_ssr``, excluding any L3 penalty rows) at each
        # successful draw — the same unit as the warm-start's ``chi_squared`` — so
        # ``cmaes_ssr`` is directly comparable below.
        cres, cmaes_ssr, best_seed = _cmaes_keep_best_over_seeds(
            run_one_seed=_run_one_seed,
            seeds=seeds,
            data_ssr=_data_ssr,
        )
        if n_seeds > 1:
            logger.info(
                "Joint CMA-ES escape: kept seed=%d (data-only SSR=%.6e) as the best of %d draws.",
                best_seed,
                cmaes_ssr,
                n_seeds,
            )
        x_cmaes = (
            np.asarray(cres.parameters, dtype=np.float64)
            if (cres.success and cres.parameters is not None)
            else x_warm
        )

        if cres.success and cmaes_ssr <= ssr_warm * (1.0 + 1e-12):
            x_final, escape = x_cmaes, "cmaes"
        else:
            x_final, escape = x_warm, "cmaes_warmstart_kept"

        logger.info(
            "Joint CMA-ES escape: warm SSR=%.6e, cmaes SSR=%.6e → kept %s (%.1fs total)",
            ssr_warm,
            cmaes_ssr,
            escape,
            time.perf_counter() - _t_escape_start,
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
    # Phase-2: intentionally left — implements the keep-better/fallback contract; conversion would risk parity.
    except Exception as exc:  # noqa: BLE001 - best-effort escape, fall back to plain fit
        logger.warning(
            "Joint CMA-ES escape failed (%s: %s); falling back to plain joint fit",
            type(exc).__name__,
            exc,
        )
        return _fit_joint_multi_phi(
            model=model,
            c2_data=c2_data,
            phi_angles=np.asarray(phi_angles),
            config=config,
            weights=weights,
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

    Notes
    -----
    This path is reached only for ``individual`` mode (same gate as the CMA-ES
    escape: Task 6 narrowed it to ``effective_mode == "individual"``).  The
    :class:`JointProblem` is built via :func:`_build_joint_problem` which
    returns a scaling-first ``[scaling_head | physics]`` vector.  A kept
    multistart result is tagged ``nlsq_diagnostics["global_escape"]`` and, by
    construction, carries NaN covariance / uncertainties and ``n_iterations == 0``
    (no covariance solve on the kept vector).
    """
    try:
        prob = _build_joint_problem(model, c2_data, phi_angles, config, weights)
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
                x0_override=np.asarray(x_start, dtype=np.float64),
                warm_start_probe=True,
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
        # Each start + the default baseline are warm-start PROBES: non-convergence
        # on a degenerate Jacobian is EXPECTED and recovered (revert to x0 / the
        # keep-better below), so demote the per-start sub-solver failure noise.
        logger.info(
            "Joint multistart escape: running %d warm-start probes (non-convergence "
            "on a degenerate Jacobian is EXPECTED and recovered; per-probe failure "
            "diagnostics are demoted to DEBUG).",
            ms_cfg.n_starts,
        )
        with _quiet_warm_start_probe_logging():
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
                warm_start_probe=True,
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
    # Phase-2: intentionally left — implements the keep-better/fallback contract; conversion would risk parity.
    except Exception as exc:  # noqa: BLE001 - best-effort escape, fall back to plain fit
        logger.warning(
            "Joint multistart escape failed (%s: %s); falling back to plain joint fit",
            type(exc).__name__,
            exc,
        )
        return _fit_joint_multi_phi(
            model=model,
            c2_data=c2_data,
            phi_angles=np.asarray(phi_angles),
            config=config,
            weights=weights,
        )


# ---------------------------------------------------------------------------
# Shared joint global-escape machinery for the AVERAGED / CONSTANT layouts.
#
# The individual escape (``_fit_joint_cmaes_multi_phi`` /
# ``_fit_joint_multistart``) optimizes the scaling-first joint vector via
# ``_build_joint_problem``. The averaged (2 scaling params) and constant
# (frozen scaling) layouts have their OWN ``base_residual_fn`` + ``[physics |
# scaling]`` vector built inline by ``_fit_joint_averaged_multi_phi`` /
# ``_fit_joint_constant_multi_phi``. To honour the ``auto → averaged`` default
# (and explicit ``constant``) under CMA-ES / multistart — matching the plain
# path AND laminar_flow's CMA-ES, which honours ``use_averaged_scaling`` — those
# two solvers accept a ``global_escape_kind`` and run the search over their own
# data residual via the helpers below. Keep-better (escape kept only if it does
# not increase the data-only SSR) and the NaN-covariance / n_iterations=0 escape
# contract are applied by the solver, exactly like the individual-mode escape.
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
    but over the averaged/constant data residual; ``ydata=zeros`` ⇒ CMA-ES
    minimises ``||residual||²`` directly.
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
        # Honor the configured CMA-ES initial step size (config field, NOT the
        # sigma= arg to fit_with_cmaes). cmaes_sigma0 was previously dropped →
        # wrapper used its 0.5 default.
        sigma=float(getattr(config, "cmaes_sigma0", 0.5)),
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
    refine of the averaged/constant data residual (``_solve_residual_nlsq``).
    The keep-better vs the warm-start is applied by the caller
    (``_apply_global_escape``).
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
            base_residual_fn,
            np.asarray(x_start, dtype=np.float64),
            lb,
            ub,
            solver_config,
            param_names,
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


def _apply_joint_keep_better_floor(
    base_residual_fn: Any,
    x0: np.ndarray,
    solved_params: np.ndarray,
    *,
    floor_fallback_x0: np.ndarray | None = None,
    warm_start_probe: bool = False,
    mode_label: str = "",
) -> tuple[np.ndarray, bool]:
    """L2 keep-better SSR floor, shared by every joint per-angle-mode solve.

    A trust-region solve must never return worse than its own start. Compares
    the DATA-ONLY SSR (``base_residual_fn`` — excludes any L3 penalty rows) at
    ``solved_params`` vs the pre-solve warm-start ``x0`` via the NaN-safe
    :func:`_escape_keeps_candidate`; when the solve regressed (or failed and
    yielded a degraded vector), reverts to the best feasible floor point —
    ``floor_fallback_x0`` when supplied and strictly better than ``x0`` (the
    individual-mode Stage-1 per-angle scaling fallback), else ``x0`` itself.

    Extracted from :func:`_fit_joint_multi_phi` (the original C044 RCA fix,
    individual per-angle mode) so the sibling averaged/constant joint solves
    — which previously had NO such protection — get the same floor.

    Returns ``(final_params, reverted)``. When ``reverted`` is True the
    caller is responsible for treating any NLSQ result tied to the discarded
    ``solved_params`` as invalid (NaN covariance/uncertainties, mirroring the
    global-escape contract).
    """

    def _data_ssr(x: np.ndarray) -> float:
        return float(np.sum(np.asarray(base_residual_fn(x), dtype=np.float64) ** 2))

    ssr_solved = _data_ssr(solved_params)
    ssr_x0 = _data_ssr(x0)
    if _escape_keeps_candidate(ssr_warm=ssr_x0, ssr_cand=ssr_solved):
        return np.asarray(solved_params, dtype=np.float64), False

    ssr_fallback = (
        _data_ssr(np.asarray(floor_fallback_x0, dtype=np.float64))
        if floor_fallback_x0 is not None
        else np.inf
    )
    if np.isfinite(ssr_fallback) and ssr_fallback < ssr_x0:
        final_params = np.asarray(floor_fallback_x0, dtype=np.float64)
        revert_ssr = ssr_fallback
        revert_label = "Stage-1 per-angle floor fallback"
    else:
        final_params = np.asarray(x0, dtype=np.float64)
        revert_ssr = ssr_x0
        revert_label = "x0 (feasible warm-start)"

    logger.log(
        logging.DEBUG if warm_start_probe else logging.WARNING,
        "L2 joint solve degraded data-only SSR vs warm-start x0%s "
        "(%.6e > %.6e); reverting to %s (SSR %.6e) to preserve "
        "the keep-better floor",
        f" ({mode_label})" if mode_label else "",
        ssr_solved,
        ssr_x0,
        revert_label,
        revert_ssr,
    )
    return final_params, True


def _warmstart_auto_skip_decision(
    config: NLSQConfig,
    escape_kind: str | None,
    ssr_warm: float,
    n_data: int,
    n_params: int,
    warm_success: bool = True,
) -> tuple[bool, float]:
    """CMA-ES warm-start auto-skip decision — parity with laminar core.py:2308-2382.

    When the NLSQ warm-start CONVERGED and already lands a good fit (the
    sigma-normalized reduced χ² ``SSR/dof`` is below
    ``cmaes_warmstart_skip_threshold``), a warm-started global search is a local
    refinement that rarely improves on it, so the expensive CMA-ES phase is
    skipped. This mirrors laminar_flow's ``fit_nlsq_cmaes`` and the heterodyne
    per-angle escape (:func:`_fit_cmaes`), which both honor the same knob; the
    JOINT escapes previously dropped it.

    ``warm_success`` is the deciding gate alongside the SSR threshold. Laminar
    sets its warm-start params/chi2 to finite values ONLY on
    ``warmstart_result["success"]`` (core.py:2316), so its auto-skip gate
    (``params is not None and chi2 < inf``) fires ONLY for a CONVERGED
    warm-start; the per-angle :func:`_fit_cmaes` likewise checks
    ``nlsq_result.success``. This matters because XPCS ``C2`` data is normalized
    (≈1), so ``SSR/dof`` is a tiny MSE that is essentially always below the
    threshold — a DEGENERATE warm-start that does not converge but reverts to a
    low-SSR ``x0`` (e.g. C044 ``two_component``) would otherwise be auto-skipped,
    defeating the very CMA-ES escape that exists to rescue it. Gating on
    ``warm_success`` keeps the global search running in exactly that case.

    The decision is CMA-ES-specific (matches the knob name and laminar — a
    ``multistart`` escape is never auto-skipped) and ``dof <= 0`` (more params
    than data) never skips (a meaningless χ²/dof). Returns ``(skip, reduced_chi2)``
    where ``reduced_chi2`` is ``inf`` whenever the decision short-circuits to
    no-skip.
    """
    if escape_kind != "cmaes":
        return False, float("inf")
    if not warm_success:
        # Non-converged warm-start: never auto-skip (parity with laminar +
        # per-angle _fit_cmaes). The global search is what rescues it.
        return False, float("inf")
    if not bool(getattr(config, "cmaes_warmstart_auto_skip", True)):
        return False, float("inf")
    n_dof = n_data - n_params
    if n_dof <= 0:
        return False, float("inf")
    reduced = ssr_warm / n_dof
    threshold = float(getattr(config, "cmaes_warmstart_skip_threshold", 5.0))
    return bool(np.isfinite(reduced) and reduced < threshold), reduced


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
    *,
    warm_success: bool = True,
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

    # Warm-start residual (one eval): SSR for keep-better + row count for the
    # CMA-ES warm-start auto-skip decision below.
    _r_warm = np.asarray(base_residual_fn(x_warm), dtype=np.float64)
    ssr_warm = float(np.sum(_r_warm**2))

    # Parity with laminar's CMA-ES auto-skip (core.py:2354-2382): when the NLSQ
    # warm-start already lands a good fit, skip the expensive global search.
    _skip, _reduced = _warmstart_auto_skip_decision(
        config, escape_kind, ssr_warm, int(_r_warm.size), int(x_warm.size), warm_success
    )
    if _skip:
        logger.info(
            "Joint %s escape auto-skip: warm-start reduced χ²=%.4f < threshold=%.1f; "
            "skipping global search (parity with laminar core.py auto-skip).",
            escape_kind,
            _reduced,
            float(getattr(config, "cmaes_warmstart_skip_threshold", 5.0)),
        )
        return x_warm, "cmaes_warmstart_auto_skip"
    try:
        if escape_kind == "cmaes":
            cand = _cmaes_joint_candidate(base_residual_fn, x_warm, lb, ub, config)
        elif escape_kind == "multistart":
            cand = _multistart_joint_candidate(
                base_residual_fn,
                x_warm,
                lb,
                ub,
                solver_config,
                param_names,
                config,
                multistart_data,
            )
        else:  # pragma: no cover — unknown kind treated as no escape
            return x_warm, None
    # Phase-2: intentionally left — implements the keep-better/fallback contract; conversion would risk parity.
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


def _resolve_effective_mode(config: NLSQConfig, n_phi: int) -> ResolvedPerAngleMode:
    """Map ``config.per_angle_mode`` + ``n_phi`` to a canonical dispatch token.

    Returns one of:

    * ``"constant"`` — frozen per-angle (β, ō) from diagonal-quantile estimator;
      optimizer dimension is ``n_physics_varying`` only.
    * ``"averaged"`` — one (β̄, ō̄) pair optimized jointly with physics. This
      is the homodyne ``auto``-averaged anti-degeneracy path.
    * ``"individual"`` — ``n_phi`` per-angle ``(contrast, offset)`` optimized
      JOINTLY with physics via :func:`_fit_joint_multi_phi`, matching
      ``laminar_flow`` and upstream heterodyne. (The sequential per-angle
      aggregate survives only as the ``config is None`` / single-angle
      fallback inside :func:`fit_nlsq_multi_phi`.)

    ``auto`` resolution is unified with the homodyne
    :class:`AntiDegeneracyController` — ``auto`` only ever selects
    ``individual`` or ``averaged``::

        n_phi <  constant_scaling_threshold (3) -> "individual"
        n_phi >= constant_scaling_threshold (3) -> "averaged"

    ``constant`` is NEVER auto-selected; the user must request it explicitly via
    ``anti_degeneracy.per_angle_mode`` in the config (or a CLI option).

    Explicit resolved modes (``"constant"``, ``"averaged"``, ``"individual"``)
    pass through unchanged. Any other token is rejected with ``ValueError``.
    """
    requested = config.per_angle_mode
    if requested == "constant":
        return "constant"
    if requested == "averaged":
        return "averaged"
    if requested == "individual":
        return "individual"
    if requested == "auto":
        # Unified with the homodyne AntiDegeneracyController: few angles ->
        # per-angle individual scaling; otherwise the averaged single-pair
        # scaling. constant is never auto-selected.
        constant_threshold = max(int(config.constant_scaling_threshold), 1)
        if n_phi < constant_threshold:
            return "individual"
        return "averaged"
    raise ValueError(
        f"unknown per_angle_mode {requested!r}; valid: constant, averaged, individual, auto"
    )


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
    meta: dict[str, Any]


def _joint_param_names_scaling_first(
    *, mode: str, physics_names: list[str], n_phi: int
) -> list[str]:
    """Canonical scaling-first joint parameter-name list ``[scaling_head | physics]``.

    Parameters
    ----------
    mode : str
        Resolved per-angle mode token: ``"constant"``, ``"averaged"``, or
        ``"individual"``.  Any other token raises ``ValueError`` via
        ``n_optimized``'s own contract — names are only ever built for a
        RESOLVED mode.
    physics_names : list[str]
        Ordered physics parameter names (e.g. ``["D0", "alpha"]``).
    n_phi : int
        Number of angles in the joint fit.

    Returns
    -------
    list[str]
        ``constant``   → ``[*physics]``
        ``averaged``   → ``["contrast_avg", "offset_avg", *physics]``
        ``individual`` → ``[contrast_0..N-1, offset_0..N-1, *physics]``
    """
    from xpcsjax.optimization.nlsq.per_angle_mode import n_optimized

    # n_optimized rejects non-canonical tokens via the same ValueError the
    # resolver raises (matched by the "unknown per_angle_mode" test).
    _ = n_optimized(mode, n_phi)  # type: ignore[arg-type]  # validates token
    if mode == "constant":
        return list(physics_names)
    if mode == "averaged":
        return ["contrast_avg", "offset_avg", *physics_names]
    contrast = [f"contrast_{i}" for i in range(n_phi)]
    offset = [f"offset_{i}" for i in range(n_phi)]
    return [*contrast, *offset, *physics_names]


def _split_scaling_first_joint(
    x_final: np.ndarray,
    *,
    mode: str,
    n_phi: int,
    n_physics: int,
    frozen_contrast: np.ndarray | None = None,
    frozen_offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a canonical scaling-first joint vector into dense per-angle scaling.

    Returns ``(physics[n_physics], contrast[n_phi], offset[n_phi])``. The physics
    block is the TAIL; the scaling head is expanded to dense per-angle:

    - ``constant``   → broadcast ``frozen_*`` (scaling absent from the vector);
    - ``averaged``   → broadcast the 2 head scalars to ``n_phi``;
    - ``individual`` → identity reshape of the ``2*n_phi`` head.

    Mirrors ``PerAngleScalingPlan.expand_tail`` but reads the head-then-tail layout
    so it can be used inside ``_build_joint_result`` without a Plan instance.

    Parameters
    ----------
    x_final : np.ndarray
        Canonical scaling-first joint optimizer vector.
    mode : str
        Resolved per-angle mode: ``"constant"``, ``"averaged"``, or
        ``"individual"``.
    n_phi : int
        Number of angles.
    n_physics : int
        Number of physics parameters (length of the tail).
    frozen_contrast : np.ndarray or None
        Required when ``mode == "constant"``; frozen per-angle contrast values.
    frozen_offset : np.ndarray or None
        Required when ``mode == "constant"``; frozen per-angle offset values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(physics[n_physics], contrast[n_phi], offset[n_phi])``
    """
    x = np.asarray(x_final, dtype=np.float64)
    if mode == "constant":
        assert frozen_contrast is not None and frozen_offset is not None
        physics = x[:n_physics]
        return (
            physics,
            np.asarray(frozen_contrast, dtype=np.float64),
            np.asarray(frozen_offset, dtype=np.float64),
        )
    if mode == "averaged":
        c_avg = float(x[0])
        o_avg = float(x[1])
        physics = x[2:]
        return (
            physics,
            np.full(n_phi, c_avg, dtype=np.float64),
            np.full(n_phi, o_avg, dtype=np.float64),
        )
    # individual
    contrast = x[:n_phi]
    offset = x[n_phi : 2 * n_phi]
    physics = x[2 * n_phi :]
    return (
        np.asarray(physics, dtype=np.float64),
        np.asarray(contrast, dtype=np.float64),
        np.asarray(offset, dtype=np.float64),
    )


def _build_floor_fallback_x0(
    *,
    resolved_mode: str,
    n_phi: int,
    n_physics_varying: int,
    per_angle_contrast: np.ndarray,
    per_angle_offset: np.ndarray,
    physics_initial: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    x0: np.ndarray,
) -> np.ndarray:
    """Reconstruct a keep-better floor fallback that PRESERVES Stage-1's per-angle scaling.

    ``_build_joint_problem`` seeds the ``individual`` joint ``x0`` scaling head by
    broadcasting the SCALAR ``model.scaling.contrast[0]`` (and ``offset[0]``) across
    all angles — a deliberate legacy mirror. When Stage-1's constant-mode quantile
    fit froze DISTINCT per-angle scaling (real multi-angle data, e.g. C044), that
    broadcast DISCARDS it, so the warm-start ``x0`` carries a degraded uniform
    scaling head and can be as bad as a failed joint solve. The keep-better floor
    (:func:`_fit_joint_multi_phi`) then has nothing genuinely better than the failed
    vector to revert to, and finalizes the worse Stage-2 result (RCA: C044 reverted
    to SSR 25663 while Stage-1 sat at 6796).

    This returns the un-collapsed ``[per-angle scaling | physics_initial]`` so the
    floor can revert to Stage-1's actual fit. The SOLVER START stays ``x0`` (this
    feeds the floor ONLY), so the success path and the homodyne/heterodyne basin
    behavior are unchanged. Returns ``x0`` for modes without a scalar-collapsed
    per-angle head (``constant`` / ``averaged``) or when the per-angle scaling is
    already uniform (the fallback would equal ``x0``).
    """
    if resolved_mode != "individual":
        return x0
    pac = np.asarray(per_angle_contrast, dtype=np.float64)
    pao = np.asarray(per_angle_offset, dtype=np.float64)
    if pac.shape != (n_phi,) or pao.shape != (n_phi,):
        return x0
    # Uniform per-angle scaling → the scalar broadcast loses nothing; fallback == x0.
    if float(np.ptp(pac)) == 0.0 and float(np.ptp(pao)) == 0.0:
        return x0

    from xpcsjax.optimization.nlsq.per_angle_mode import PerAngleScalingPlan

    # The guard above proves ``resolved_mode == "individual"`` here; pass the literal
    # so mypy can satisfy ``PerAngleScalingPlan.mode``'s ``Literal[...]`` annotation
    # (it cannot narrow a broad ``str`` through a single ``!=`` comparison).
    fallback_plan = PerAngleScalingPlan(
        mode="individual",
        n_phi=n_phi,
        n_physics=n_physics_varying,
        quantile_scaling=(pac, pao),
    )
    fallback = np.concatenate(
        [fallback_plan.seed_tail(), np.asarray(physics_initial, dtype=np.float64)]
    )
    return np.clip(fallback, lb, ub)


def _build_joint_problem(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: np.ndarray,
    config: NLSQConfig,
    weights: np.ndarray | None,
) -> JointProblem:
    """Build the joint heterodyne LSQ problem (residual + x0 + bounds), scaling-first.

    Builds the canonical scaling-first joint vector ``[scaling_head | physics]``
    via :class:`PerAngleScalingPlan`, parameterized by the resolved per-angle
    mode (``resolve_per_angle_mode(config.per_angle_mode, n_phi,
    config.constant_scaling_threshold)``). The scaling head is
    ``[]`` (constant) / ``[c, o]`` (averaged) / ``[c_0..N-1, o_0..N-1]``
    (individual); the physics block is the TAIL. Shared by the plain joint fit
    (:func:`_fit_joint_multi_phi`) and the global escapes so all three optimize
    the SAME objective. The scaling-first layout is always built from ``config``.

    Includes the L2 hierarchical Stage 1 physics-only solve (run when
    ``config.enable_hierarchical`` is True) — its converged physics vector
    warm-starts ``x0`` and its chi^2 is surfaced via ``meta``.

    Returns
    -------
    JointProblem
        ``joint_residual_fn`` is the L3-augmented residual (identity to
        ``base_residual_fn`` when ``config.regularization_mode == "none"``).
        ``meta`` carries ``base_residual_fn``, ``n_physics_varying``,
        ``scaling_tail_size`` (``plan.n_scaling``), ``regularization_active``,
        ``n_penalty_rows``, ``hierarchical_stage1_chi2`` (``None`` when L2 is
        disabled), and the scaling-first bookkeeping ``scaling_first=True``,
        ``resolved_mode``, ``plan`` (:class:`PerAngleScalingPlan`), and
        ``joint_param_names``.
    """
    from xpcsjax.optimization.nlsq.per_angle_mode import (
        PerAngleScalingPlan,
        resolve_per_angle_mode,
    )

    param_manager = model.param_manager
    varying_names = param_manager.varying_names
    n_physics_varying = param_manager.n_varying
    n_phi = len(phi_angles)

    # Resolve the user/config token to a canonical scaling-first variant. The
    # native in-memory joint solve only ever builds for a RESOLVED mode
    # (``constant`` / ``averaged`` / ``individual``).
    resolved_mode = resolve_per_angle_mode(
        config.per_angle_mode, n_phi, config.constant_scaling_threshold
    )

    # Physics parameter initial values and bounds
    physics_initial = param_manager.get_initial_values()
    physics_lower, physics_upper = param_manager.get_bounds()
    physics_initial = np.clip(physics_initial, physics_lower, physics_upper)

    # ------------------------------------------------------------------
    # L2 hierarchical two-stage: Stage 1 — physics-only solve with
    # quantile-fixed scaling (delegates to the constant-mode solver).
    # When `config.enable_hierarchical` is True we run the constant-mode
    # solver first to converge the physics block with scaling frozen,
    # then warm-start the joint scaling-first solve below by overriding
    # `physics_initial` with the converged physics vector. The scaling head
    # keeps its deterministic quantile seed (`plan.seed_tail`).
    # ------------------------------------------------------------------
    hierarchical_stage1_chi2: float | None = None
    if config.enable_hierarchical:
        from xpcsjax.optimization.nlsq.per_angle_mode import n_optimized as _n_opt

        # The Stage-1 warm-start delegates to the constant-mode solver, which is
        # inherently per-angle (it freezes per-angle quantile scaling to converge
        # physics cheaply). Name the resolved FINAL mode + its scaling DOF up
        # front so the per-angle arrays the constant solver logs are not misread
        # as the fit's scaling mode — the final fit re-optimizes under
        # ``resolved_mode`` and Stage-1's per-angle scaling is discarded.
        _final_scaling_dof = _n_opt(resolved_mode, n_phi)
        logger.info(
            "L2 Stage-1 warm-start (final mode: %s): per-angle quantile scaling "
            "frozen for WARM-START ONLY; final fit optimizes %d %s scaling + "
            "%d physics = %d params",
            resolved_mode,
            _final_scaling_dof,
            resolved_mode,
            n_physics_varying,
            _final_scaling_dof + n_physics_varying,
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
            warm_start_context=f"L2 Stage-1 -> final mode {resolved_mode}",
        )
        stage1_physics = np.asarray(stage1_result.parameters, dtype=np.float64)
        hierarchical_stage1_chi2 = float(stage1_result.chi_squared)
        physics_initial = np.clip(stage1_physics, physics_lower, physics_upper)
        logger.info(
            "L2 hierarchical (scaling-first) — Stage 1 done: chi2=%.6f, "
            "warm-starting stage 2 joint refine",
            hierarchical_stage1_chi2,
        )

    # Per-angle scaling seed. The legacy individual x0 broadcast the SCALAR
    # ``scaling.contrast[0]`` / ``scaling.offset[0]`` to all angles; mirror that
    # by building per-angle quantile arrays of length ``n_phi`` from those
    # scalars (the plan's constructor requires ``(n_phi,)`` arrays and the
    # individual seed is identical to the legacy broadcast).
    scaling = model.scaling
    contrast_init = float(scaling.contrast[0]) if len(scaling.contrast) > 0 else 0.5
    offset_init = float(scaling.offset[0]) if len(scaling.offset) > 0 else 1.0
    quantile_contrast = np.full(n_phi, contrast_init, dtype=np.float64)
    quantile_offset = np.full(n_phi, offset_init, dtype=np.float64)

    plan = PerAngleScalingPlan(
        mode=resolved_mode,
        n_phi=n_phi,
        n_physics=n_physics_varying,
        quantile_scaling=(quantile_contrast, quantile_offset),
    )

    # Scaling-first x0/bounds: ``[scaling_head | physics]``. Source contrast/
    # offset bounds from the parameter registry (single source of truth,
    # matches every sibling per-angle-mode solver — averaged, constant, and
    # the engine route's ``_scaling_first_bounds``) instead of hardcoded
    # literals, so the feasible box is the physically valid one, not a
    # stale copy-pasted range.
    from xpcsjax.config.parameter_registry import SCALING_PARAMS

    contrast_bounds = (
        SCALING_PARAMS["contrast"].min_bound,
        SCALING_PARAMS["contrast"].max_bound,
    )
    offset_bounds = (
        SCALING_PARAMS["offset"].min_bound,
        SCALING_PARAMS["offset"].max_bound,
    )
    scaling_head_x0 = plan.seed_tail()
    scaling_head_lb, scaling_head_ub = plan.seed_bounds(
        contrast_bounds=contrast_bounds, offset_bounds=offset_bounds
    )

    x0 = np.concatenate([scaling_head_x0, physics_initial])
    lb = np.concatenate([scaling_head_lb, physics_lower])
    ub = np.concatenate([scaling_head_ub, physics_upper])

    # Keep-better floor fallback (failure path ONLY): a second feasible seed that
    # preserves Stage-1's DISTINCT per-angle frozen scaling instead of the scalar
    # ``contrast[0]`` broadcast ``x0`` uses above. The solver still starts at ``x0``
    # (parity-preserving); on a degenerate joint solve the floor reverts to the BEST
    # of {x0, this fallback}, so it lands on Stage-1's actual fit rather than the
    # degraded uniform-scaling x0. See :func:`_build_floor_fallback_x0`.
    floor_fallback_x0 = _build_floor_fallback_x0(
        resolved_mode=resolved_mode,
        n_phi=n_phi,
        n_physics_varying=n_physics_varying,
        per_angle_contrast=np.asarray(scaling.contrast, dtype=np.float64),
        per_angle_offset=np.asarray(scaling.offset, dtype=np.float64),
        physics_initial=physics_initial,
        lb=lb,
        ub=ub,
        x0=x0,
    )

    logger.info(
        "Joint multi-angle fit (scaling-first, mode=%s): %d scaling + %d physics = "
        "%d total params, %d angles",
        resolved_mode,
        plan.n_scaling,
        n_physics_varying,
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
    n_scaling = plan.n_scaling

    def base_residual_fn(x: np.ndarray) -> Any:  # type: ignore[return-value]
        """Compute concatenated residuals across all angles via vmap.

        Routes through ``compute_multi_angle_residuals`` (jit + vmap) to
        replace the previous n_phi serial kernel dispatches with a single
        batched XLA call.  The combined vector is canonical scaling-first
        ``[scaling_head | physics]``: the scaling head is expanded to per-angle
        contrast/offset via the JIT-safe :meth:`PerAngleScalingPlan.expand_tail_jax`
        before the batched residual call.
        """
        # Split scaling-first combined vector: scaling head, physics tail.
        scaling_head = x[:n_scaling]
        physics_varying = x[n_scaling:]

        # Reconstruct full physics parameter array (immutable JAX scatter)
        varying_jax = jnp.asarray(physics_varying, dtype=jnp.float64)
        full_jax = fixed_values_jax.at[varying_indices_jax].set(varying_jax)

        # Expand the scaling head → per-angle contrast/offset. MUST use the
        # JIT-safe ``expand_tail_jax``: the numpy ``expand_tail`` calls
        # np.asarray on the traced head slice, raising TracerArrayConversionError
        # inside NLSQ's JIT-compiled residual.
        contrasts_jax, offsets_jax = plan.expand_tail_jax(scaling_head)

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
    # offset) and append penalty rows to the residual vector. NLSQ's
    # trust-region solver minimises ``||r||²``, so K appended rows with values
    # ``sqrt(lambda) * CV_g`` yield an extra ``lambda * sum_g(CV_g^2)`` penalty
    # term — the JIT-traceable variant of the CV-based regularizer documented in
    # ``adaptive_regularization.AdaptiveRegularizer``. Penalty rows operate
    # on the *per-angle scaling arrays* (contrast + offset), the natural target
    # for the per-angle CV that actually matters.
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
        # L3 group indices come from the mapper-backed ``plan.group_indices``
        # (the scaling head's contrast / offset groups), the single boundary
        # authority for the canonical scaling-first layout.
        n_penalty_rows = len(plan.group_indices)

        # JAX-traceable penalty rows. We compute CV directly from the per-angle
        # scaling arrays expanded from the scaling-first head
        # (``plan.expand_tail_jax``), so the penalty targets the same per-angle
        # contrast/offset variance the data residual sees.
        sqrt_lambda = float(np.sqrt(float(regularizer.lambda_value)))

        def joint_residual_fn(x: np.ndarray) -> Any:  # type: ignore[return-value]
            r = base_residual_fn(x)
            # Per-angle contrast/offset via the shared JIT-safe expansion of the
            # scaling head (the numpy ``expand_tail`` calls np.asarray on the
            # traced slice and crashes inside the JIT-compiled residual).
            contrasts, offsets = plan.expand_tail_jax(x[:n_scaling])
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
        "scaling_tail_size": int(plan.n_scaling),
        "regularization_active": regularization_active,
        "n_penalty_rows": int(n_penalty_rows),
        "hierarchical_stage1_chi2": hierarchical_stage1_chi2,
        "varying_names": list(varying_names),
        # Keep-better floor fallback preserving Stage-1's per-angle scaling
        # (== x0 unless the individual mode collapsed distinct per-angle scaling).
        "floor_fallback_x0": floor_fallback_x0,
        # Scaling-first bookkeeping (Phase 1+2).
        "scaling_first": True,
        "resolved_mode": resolved_mode,
        "plan": plan,
        "joint_param_names": _joint_param_names_scaling_first(
            mode=resolved_mode, physics_names=list(varying_names), n_phi=n_phi
        ),
    }

    return JointProblem(
        joint_residual_fn=joint_residual_fn,
        x0=x0,
        lb=lb,
        ub=ub,
        meta=meta,
    )


def _fit_joint_multi_phi(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: np.ndarray,
    config: NLSQConfig,
    weights: np.ndarray | None,
    x0_override: np.ndarray | None = None,
    prob: JointProblem | None = None,
    warm_start_probe: bool = False,
) -> OptimizationResult:
    """Joint multi-angle fit over the canonical scaling-first vector.

    Parameters
    ----------
    warm_start_probe
        When True, this fit is a CMA-ES / multistart warm-start probe whose
        non-convergence on a degenerate Jacobian is EXPECTED and recovered (the
        keep-better floor reverts to x0; the global search refines). In that mode
        the wrapper fallback runs a SINGLE deterministic attempt (no 3x retry —
        the retries reproduce the identical degenerate result) and the
        adapter-fallback / floor-revert messages drop to DEBUG. Numerics are
        unchanged; only the retry count (deterministically identical) and log
        levels differ. The caller additionally wraps the probe in
        :func:`_quiet_warm_start_probe_logging` to demote the sub-solver noise.

    Builds and solves the canonical scaling-first joint problem
    ``[scaling_head | physics]`` via :func:`_build_joint_problem`, which resolves
    the per-angle mode (``constant`` / ``averaged`` / ``individual``) from
    ``config`` itself. The optimizer vector layout is:

    * ``individual`` — per-angle scaling is free; vector is
      ``[contrast_0..N | offset_0..N | physics_varying]`` (``2*n_phi + physics``),
      matching xpcsjax ``laminar_flow`` and upstream heterodyne.
    * ``averaged`` — one shared ``(contrast, offset)`` pair:
      ``[contrast_avg | offset_avg | physics_varying]`` (``2 + physics``).
    * ``constant`` — frozen scaling; vector is physics-only.

    The scaling-first builder resolves the layout from ``config``.

    This is the heterodyne equivalent of homodyne's AntiDegeneracyController
    joint-fit path.

    Returns
    -------
    OptimizationResult
        One result for the entire joint solve.  ``parameters`` has the canonical
        scaling-first ``[scaling_head | physics]`` layout. Per-angle diagnostics
        — ``chi2_per_angle``,
        ``per_angle_mode`` (the resolved variant), ``scaling_source='fitted'``,
        ``shear_weighting='not_applicable_heterodyne'`` — live in
        ``nlsq_diagnostics``.  Mirrors the contract of
        :func:`xpcsjax.optimization.nlsq.heterodyne_constant_mode._fit_joint_constant_multi_phi`
        (Sub-PR B2).
    """
    t_start = time.perf_counter()

    n_phi = len(phi_angles)

    # Construct the joint LSQ problem (residual + x0 + bounds) via the shared
    # helper so the plain fit and the global escapes optimize the SAME objective.
    # The builder builds the canonical scaling-first layout from ``config`` for
    # ``constant`` / ``averaged`` / ``individual``. The result-tail bookkeeping
    # (scaling, base residual, regularization/hierarchical meta) is consumed by
    # ``_build_joint_result``, not here.
    #
    # ``prob`` may be supplied by a caller that already built it (the CMA-ES
    # escape builds it once for the residual closures + bounds) so the expensive
    # L2 Stage-1 constant-mode solve inside ``_build_joint_problem`` runs ONCE
    # per escaped fit instead of twice. Default ``None`` ⇒ build here, keeping
    # every existing caller byte-identical.
    if prob is None:
        prob = _build_joint_problem(model, c2_data, phi_angles, config, weights)
    joint_residual_fn = prob.joint_residual_fn
    # ``x0_override`` (Task 3) lets the joint multistart escape seed the solver at
    # an arbitrary LHS start. Default ``None`` ⇒ ``prob.x0`` (the scaling-first
    # warm-start built from ``model.scaling`` + the physics initial values), so
    # the plain fit and every existing caller are byte-identical. The override is
    # clipped to the problem bounds so an out-of-range LHS draw cannot escape the
    # feasible box.
    if x0_override is not None:
        x0 = np.clip(np.asarray(x0_override, dtype=np.float64), prob.lb, prob.ub)
    else:
        x0 = prob.x0
    lb = prob.lb
    ub = prob.ub

    # Run optimization via NLSQAdapter (primary) with NLSQWrapper fallback.
    # max_nfev is multiplied by n_phi here because the joint multi-angle solve
    # packs all angles into a single residual vector; the per-angle budget
    # documented on NLSQConfig.max_nfev is preserved by scaling the
    # combined cap. See NLSQConfig.max_nfev docstring for the contract.
    joint_config = NLSQConfig(
        method=config.method if config.method != "lm" else "trf",
        ftol=config.ftol,
        xtol=config.xtol,
        gtol=config.gtol,
        max_nfev=(config.max_nfev * n_phi if config.max_nfev is not None else None),
        # Inherit the robust-loss kernel from the caller so the individual
        # joint solve minimizes the SAME objective as the averaged/constant joint
        # paths. Omitting these let loss fall to the dataclass default ("soft_l1"),
        # making the solver objective silently mode-dependent.
        loss=config.loss,
        use_nlsq_library=config.use_nlsq_library,
        n_params=len(x0),
    )

    joint_result: NLSQResult | None = None
    # Canonical scaling-first joint parameter names ``[scaling_head | physics]``,
    # already built mode-aware by ``_build_joint_problem``.
    joint_param_names = list(prob.meta["joint_param_names"])

    # L4: per-iteration gradient-collapse monitor (strictly observational).
    # Joint layout is canonical scaling-first [scaling_head | physics] — the
    # scaling head is the per-angle (scaling) block. See ``_build_l4_callback``;
    # pass scaling_first=True so the monitor partitions physical/per-angle
    # indices for THIS layout (the default is physics-first for the averaged path).
    _monitor, _l4_callback = _build_l4_callback(
        model, x0, joint_residual_fn, config, scaling_first=True
    )

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
            # A warm-start probe failing here is EXPECTED + recovered (DEBUG);
            # a standalone joint fit failing is a real fallback event (WARNING).
            logger.log(
                logging.DEBUG if warm_start_probe else logging.WARNING,
                "Joint NLSQAdapter failed, falling back to NLSQWrapper: %s",
                adapter_exc,
            )
            joint_result = None

    if joint_result is None and NLSQWrapper is not None:
        # A warm-start probe runs ONE deterministic wrapper attempt: on the
        # degenerate Jacobian every retry reproduces the identical failed result
        # (verified: all 3 tiers report the same final cost), so the extra
        # retries are pure wasted compute + noise. Standalone fits keep 3 retries.
        joint_wrapper = NLSQWrapper(
            parameter_names=joint_param_names,
            max_retries=1 if warm_start_probe else 3,
        )
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

    # L2 keep-better floor. A degenerate joint solve — e.g. a nan-gradient
    # trust-region step on the near-singular C044 ``two_component`` Jacobian —
    # can return parameters whose data-only SSR is HIGHER than the warm-start
    # ``x0`` (the feasible point the solve began at: ``[scaling | stage-1
    # physics]``). A trust-region method must never return worse than its start;
    # when the backend does (or fails and yields a degraded vector), revert to
    # ``x0``. The comparison uses the DATA-ONLY residual (excludes any L3 penalty
    # rows) via the NaN-safe ``_escape_keeps_candidate``, so it is SSR-monotone
    # and parity-safe under the ``two_component`` no-worse contract. Without this
    # floor the degraded vector poisoned the CMA-ES warm-start (RCA: C044 fit
    # finalized SSR 8737 after discarding a 6796 Stage-1 point).
    base_residual_fn = prob.meta["base_residual_fn"]

    solved_params = np.asarray(joint_result.parameters, dtype=np.float64)
    final_params, reverted = _apply_joint_keep_better_floor(
        base_residual_fn,
        x0,
        solved_params,
        floor_fallback_x0=prob.meta.get("floor_fallback_x0"),
        warm_start_probe=warm_start_probe,
        mode_label="individual mode",
    )
    if reverted:
        # The covariance / uncertainties / iteration counts / L4-monitor data in
        # ``joint_result`` describe the DISCARDED degraded vector, not x0. Drop
        # them so ``_build_joint_result`` NaN-fills uncertainty for the accepted
        # x0 instead of copying evidence from a rejected solve (it recomputes
        # parameters and chi^2 at x0 regardless). Mirrors the global-escape
        # contract, which also carries NaN covariance on a kept warm vector.
        joint_result = None
        used_monitored_backend = False

    return _build_joint_result(
        model,
        prob,
        c2_data,
        final_params,
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
    n_physics_varying = prob.meta["n_physics_varying"]
    regularization_active = prob.meta["regularization_active"]
    n_penalty_rows = prob.meta["n_penalty_rows"]
    hierarchical_stage1_chi2 = prob.meta["hierarchical_stage1_chi2"]
    resolved_mode = prob.meta["resolved_mode"]
    plan = prob.meta["plan"]

    if joint_param_names is None:
        joint_param_names = list(prob.meta["joint_param_names"])

    fitted_params_full = np.asarray(x_final, dtype=np.float64)
    # Canonical scaling-first vector ``[scaling_head | physics]``: physics is the
    # TAIL, the scaling head expands to dense per-angle contrast/offset.
    # ``_split_scaling_first_joint`` is the host-side (NumPy) reader — POST-fit
    # only, never inside the JIT residual. (The joint problem is always
    # scaling-first.)
    fitted_physics, fitted_contrast, fitted_offset = _split_scaling_first_joint(
        fitted_params_full,
        mode=resolved_mode,
        n_phi=n_phi,
        n_physics=n_physics_varying,
        frozen_contrast=plan.frozen_contrast if resolved_mode == "constant" else None,
        frozen_offset=plan.frozen_offset if resolved_mode == "constant" else None,
    )

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

    # When no NLSQ result backs this assembly, the verdict depends on WHY
    # ``joint_result`` is None. A kept GLOBAL ESCAPE returns a pre-accepted
    # vector it has already compared and kept — that is a success. But a
    # REVERTED warm-start (keep-better floor reverted to x0 and dropped
    # ``joint_result``) is NOT a converged solve and must report failure, so the
    # CMA-ES auto-skip gate (which reads ``warm.success``) does not skip the
    # escape on a degenerate fit (parity with laminar core.py:2320). Key on
    # ``global_escape`` to distinguish the two.
    solve_success = (
        joint_result.success if joint_result is not None else (global_escape is not None)
    )
    convergence_status: ConvergenceStatus = "converged" if solve_success else "failed"
    quality_flag: QualityFlag = "good" if solve_success else "marginal"

    # ------------------------------------------------------------------
    # L2 anti-degeneracy: hierarchical two-stage solve.
    #
    # Stage 1 (physics-only with quantile-fixed scaling) ran above — before
    # the joint solve — when `config.enable_hierarchical` was True, producing
    # `hierarchical_stage1_chi2` and a warm-started `physics_initial`. Stage 2
    # is the joint refine the surrounding code already executed (scaling tail
    # unfrozen, jointly fit with physics): the scaling-first path optimizes
    # `[scaling_head | physics]`.
    #
    # The SSR conservation invariant (`chi2_per_angle.sum() == chi_squared`)
    # still holds for stage 2 because the joint solve uses the canonical
    # multi-angle residual decomposition.
    # ------------------------------------------------------------------
    mode_label = f"mode={resolved_mode}"
    hierarchical_extras: dict[str, Any] = {}
    if config.enable_hierarchical and hierarchical_stage1_chi2 is not None:
        logger.info(
            "L2 hierarchical (%s) — Stage 2 done: chi2=%.6f (stage1=%.6f)",
            mode_label,
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
            "L3 adaptive regularization enabled (%s): "
            "mode=%s, lambda=%.6g, target_cv=%.3f, penalty_rows=%d.",
            mode_label,
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
    # The joint solve fits ``[scaling_head | physics]`` jointly; gradient
    # collapse here typically indicates a near-degenerate physics-vs-scaling
    # subspace. The monitor
    # records the per-iteration gradient ratio via NLSQ's curve_fit callback
    # (built in ``_build_l4_callback``); when it recorded zero observations,
    # ``_assemble_l4_extras`` falls back to the post-solve covariance-condition
    # block (tagged ``mechanism="post_solve_fallback"``).
    # ------------------------------------------------------------------
    # Reported ``per_angle_mode``: the scaling-first path reports the resolved
    # variant (``constant`` / ``averaged`` / ``individual``).
    per_angle_mode_label = resolved_mode
    fourier_extras: dict[str, Any] = {}

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

    diagnostics = _build_heterodyne_diagnostics(
        per_angle_mode=per_angle_mode_label,
        chi2_per_angle=chi2_per_angle,
        scaling_source="fitted",
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
        # Mirror the engine route's ``n_physics_field`` rule: the constant
        # (frozen-scaling) layout reports ``None`` (physics-only vector, no
        # scaling tail to disambiguate); averaged/individual report the physics
        # count so the scaling-first ``[scaling_head | physics]`` tail is read.
        n_physics=None if resolved_mode == "constant" else int(n_physics_varying),
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
      branch. Fixed here so the package actually delivers on the "CMA-ES global
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
    warmstart_skip_threshold = float(getattr(config, "cmaes_warmstart_skip_threshold", 5.0))
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
            log_exception(
                logger,
                exc,
                context={
                    "operation": "phase3_cmaes_cost_recompute",
                    "note": (
                        "treating as inf so the NLSQ result wins by default; "
                        "inspect the off-diagonal residual block if this recurs"
                    ),
                    "fallback_cmaes_cost": "inf",
                },
                level=logging.WARNING,
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
        # Residual mask excludes the t=0 boundary row/column AND the diagonal,
        # so the valid count is (n-1)*(n-2), matching _compute_per_angle_chi2 —
        # not the N^2 - N that `size - n_matrix` (diagonal-only) would give.
        n_valid = (n_matrix - 1) * (n_matrix - 2)
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
    result.metadata["global_escape"] = "cmaes" if winner == "cmaes" else "cmaes_warmstart_kept"

    _log_result(result)
    return result


def _cmaes_to_nlsq_result(
    cmaes_result: Any,
    final_cost: float,
    *,
    parameter_names: list[str],
) -> NLSQResult:
    """Pack a :class:`CMAESResult` into the :class:`NLSQResult` shape.

    Downstream consumers (DOF correction, post-fit logging, multi-phi joining)
    then see a uniform structure regardless of which optimizer won Phase 3.

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


# Phase-6 minimal stub: delegates to the standard joint fit so the
# return shape is ``OptimizationResult``.  A real multistart implementation
# (LHS sampling over physics priors + perturbation + best-by-chi-squared
# selection) wired against ``run_multistart_nlsq`` lands in a later phase
# alongside a heterodyne-shaped ``single_fit_func`` adapter.
#
# Note this entry is per-angle (signature parallels ``_fit_local`` /
# ``_fit_cmaes`` — scalar ``phi_angle``, single ``(N, N)`` ``c2_data``).
# The dispatcher at ~line 1175 is also gated behind ``HAS_MULTISTART`` which
# is hard-coded ``False`` at module import, so this body is currently
# unreachable; the conversion is purely about getting the return shape right so the
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

    Currently delegates to the standard joint fit
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
    c2_array = np.asarray(_c2_data)
    if c2_array.ndim == 2:
        c2_batch = c2_array[np.newaxis, ...]
    else:
        c2_batch = c2_array
    phi_angles_array = np.asarray([_phi_angle], dtype=np.float64)

    return _fit_joint_multi_phi(
        model=_model,
        c2_data=c2_batch,
        phi_angles=phi_angles_array,
        config=_config,
        weights=_weights if _weights is None else np.asarray(_weights),
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

    # The adapter (NLSQ ``CurveFit``) sizes its ``ydata`` zero-target by
    # ``n_data``, so it MUST equal the length of the residual vector
    # ``compute_residuals`` actually returns — the off-diagonal, t=0-boundary
    # -excluded vector of length ``(n_time - 1) * (n_time - 2)``, NOT the full
    # ``c2_jax.size`` (= n_time²). Using ``c2_jax.size`` makes NLSQ broadcast a
    # 256-long ``ydata`` against a 210-long residual and raise; the adapter then
    # silently falls back to the wrapper. Derive ``n_data`` from the residual
    # function itself so the adapter path is correct and stays in sync with the
    # kernel's masking convention (matches ``n_per_angle`` at heterodyne_core
    # line ~1361).
    n_data = int(np.asarray(jax_residual_fn(jnp.arange(c2_jax.size), *initial_varying)).size)

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
        # Residual mask excludes the t=0 boundary row/column AND the diagonal,
        # so the valid count is (n-1)*(n-2), matching _compute_per_angle_chi2 —
        # not the N^2 - N that `size - n_matrix` (diagonal-only) would give.
        n_valid = (n_matrix - 1) * (n_matrix - 2)
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
# (averaged / constant / individual / hybrid_streaming) returned an
# ``OptimizationResult`` without ever emitting that block, so the two_component
# log jumped straight from the adapter fit to the CLI results table. The helper
# below restores parity: it is invoked once per analysis from the CLI-facing
# dispatch ``optimization.nlsq._fit_nlsq_heterodyne`` (the heterodyne analog of
# ``fit_nlsq_jax``), not from the per-angle / per-trial sub-fits.
# ---------------------------------------------------------------------------

# Per-angle scaling lives in ``nlsq_diagnostics`` under a mode-specific suffix
# (individual: plain, averaged: ``_quantile``, constant: ``_fixed``). The
# completion logger reads whichever is present so the mean-scaling line stays
# mode-agnostic.
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
    """Return ``(mean_contrast, mean_offset)`` from the per-angle scaling.

    Reads whichever per-angle scaling key the active mode populated, returning
    ``(None, None)`` when none is present.
    """
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

    The canonical joint vector is scaling-first ``[scaling_head | physics_tail]``
    for every mode, so the physical parameters are read from the TAIL
    (``params[-n_physics:]``). ``constant`` is physics-only (empty scaling head),
    so its tail equals the full vector. ``hybrid_streaming`` is likewise
    scaling-first and shares the same tail slice.
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
        # Layout is authoritative when the producer emits an explicit
        # ``scaling_first`` marker (audit 2026-06-17 #1): the averaged token is
        # NOT a reliable layout signal because two producers emit it with
        # OPPOSITE orderings — the legacy `_fit_joint_averaged_multi_phi` is
        # PHYSICS-FIRST while the engine route is SCALING-FIRST. Honour the
        # marker when present; otherwise fall back to the mode/covariance
        # heuristic (averaged + sequential-individual aggregate are physics-first).
        scaling_first_marker = diag.get("scaling_first")
        if scaling_first_marker is not None:
            physics_first = not bool(scaling_first_marker)
        else:
            physics_first = mode == "averaged" or (
                diag.get("covariance_structure") == "block_diagonal_sequential"
            )
        if physics_first:
            phys_vals = params[:n_physics]
            phys_unc = unc[:n_physics] if unc is not None and unc.size >= n_physics else None
        else:
            phys_vals = params[-n_physics:]
            phys_unc = unc[-n_physics:] if unc is not None and unc.size >= n_physics else None

        logger.info("Fitted parameters (%d physical, %d angles):", n_physics, n_phi)
        logger.info("  Physical parameters:")
        for i, name in enumerate(varying_names[:n_physics]):
            unc_val = float(phys_unc[i]) if phys_unc is not None else 0.0
            logger.info("    %s: %.6g +/- %.6g", name, float(phys_vals[i]), unc_val)

    mean_contrast, mean_offset = _mean_scaling_from_diagnostics(diag)
    if mean_contrast is not None and mean_offset is not None:
        logger.info("  Mean scaling: contrast=%.4f, offset=%.4f", mean_contrast, mean_offset)

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
