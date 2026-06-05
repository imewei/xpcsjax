"""NLSQ adapters: NLSQAdapter (JAX-traced) and NLSQWrapper (memory-aware fallback).

Import order: nlsq imports appear before JAX so that nlsq can configure
JAX x64 mode before JAX is initialised.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

# ---------------------------------------------------------------------------
# nlsq imports — MUST precede any JAX import so nlsq can set x64 mode
# ---------------------------------------------------------------------------
from nlsq import CurveFit, curve_fit, curve_fit_large  # noqa: E402

try:
    from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig

    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    AdaptiveHybridStreamingOptimizer = None  # type: ignore[assignment,misc]
    HybridStreamingConfig = None  # type: ignore[assignment,misc]

import jax.numpy as jnp  # noqa: E402 — must follow nlsq to preserve x64 init order

from xpcsjax.optimization.nlsq.gradient_monitor import _get_debug_curvefit_callback
from xpcsjax.optimization.nlsq.heterodyne_adapter_base import NLSQAdapterBase
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_memory import NLSQStrategy, select_nlsq_strategy
from xpcsjax.optimization.nlsq.heterodyne_result_builder import (
    build_failed_result,
    build_result_from_nlsq,
)
from xpcsjax.optimization.nlsq.heterodyne_results import NLSQResult
from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Model cache — avoids re-JIT-compiling CurveFit for identical problem shapes
# ---------------------------------------------------------------------------

_MODEL_CACHE_MAX_SIZE = 64


def _optimizer_kwargs(config: NLSQConfig, method: str) -> dict:
    """Build the kwargs dict passed to every nlsq CurveFit.curve_fit call.

    Centralises all config→optimizer parameter mapping so every call site
    stays in sync.  ``max_nfev=None`` (unlimited) is omitted so nlsq keeps
    its own default; an explicit int is passed through.
    """
    kw: dict = {
        "method": method,
        "loss": config.loss,
        "ftol": config.ftol,
        "xtol": config.xtol,
        "gtol": config.gtol,
        "x_scale": config.x_scale,
    }
    if config.max_nfev is not None:
        kw["max_nfev"] = config.max_nfev
    return kw


@dataclass(frozen=True)
class ModelCacheKey:
    """Cache key for CurveFit instances.

    Includes phi_angles and scaling_mode so that different multi-angle
    or scaling configurations do not share the same compiled fitter.
    """

    n_data: int
    n_params: int
    phi_angles: tuple[float, ...] | None
    scaling_mode: str
    callable_scope: object | None = None


@dataclass
class CachedModel:
    """A cached CurveFit instance with usage stats."""

    fitter: object  # nlsq.CurveFit
    created_at: float = field(default_factory=time.monotonic)
    last_accessed: float = field(default_factory=time.monotonic)
    n_hits: int = 0


_model_cache: dict[ModelCacheKey, CachedModel] = {}
_cache_stats: dict[str, int] = {"hits": 0, "misses": 0}


def get_or_create_fitter(
    n_data: int,
    n_params: int,
    phi_angles: tuple[float, ...] | None = None,
    scaling_mode: str = "auto",
    callable_scope: object | None = None,
) -> tuple[object, bool]:
    """Get a CurveFit instance from cache or create a new one.

    Parameters
    ----------
    n_data : int
        Number of data points (CurveFit ``flength``).
    n_params : int
        Number of parameters.
    phi_angles : tuple of float, optional
        Azimuthal angles (distinguishes multi-angle configs).
    scaling_mode : str
        Contrast/offset scaling mode (e.g. ``"auto"``, ``"individual"``).
    callable_scope : object, optional
        Residual/model callable that must not share a stateful fitter with
        different residual closures.

    Returns
    -------
    tuple of (object, bool)
        The ``CurveFit`` fitter and whether it was a cache hit.
    """
    key = ModelCacheKey(
        n_data=n_data,
        n_params=n_params,
        phi_angles=phi_angles,
        scaling_mode=scaling_mode,
        callable_scope=callable_scope,
    )

    if key in _model_cache:
        _model_cache[key].last_accessed = time.monotonic()
        _model_cache[key].n_hits += 1
        _cache_stats["hits"] += 1
        logger.debug(
            "CurveFit cache hit: n_data=%d n_params=%d phi=%s scaling=%s",
            n_data,
            n_params,
            phi_angles,
            scaling_mode,
        )
        return _model_cache[key].fitter, True

    _cache_stats["misses"] += 1

    # Evict oldest entry if cache is full
    if len(_model_cache) >= _MODEL_CACHE_MAX_SIZE:
        oldest_key = min(_model_cache, key=lambda k: _model_cache[k].last_accessed)
        logger.debug("CurveFit cache eviction: removing oldest entry %s", oldest_key)
        del _model_cache[oldest_key]

    fitter = CurveFit(flength=int(n_data))
    _model_cache[key] = CachedModel(fitter=fitter)
    return fitter, False


def clear_model_cache() -> None:
    """Clear the CurveFit model cache and reset hit/miss counters."""
    _model_cache.clear()
    _cache_stats["hits"] = 0
    _cache_stats["misses"] = 0


def get_cache_stats() -> dict[str, int]:
    """Return cache hit/miss/size statistics."""
    return {**_cache_stats, "size": len(_model_cache)}


# ---------------------------------------------------------------------------
# Shared convergence assessment
# ---------------------------------------------------------------------------


def _assess_convergence(
    fitted_params: np.ndarray,
    initial_params: np.ndarray,
    reduced_chi2: float | None,
) -> tuple[bool, str, str]:
    """Apply post-fit convergence heuristics.

    Flags non-finite parameters, an extreme reduced chi-squared (poor fit), and
    a solution unchanged from the initial guess (no progress).

    Parameters
    ----------
    fitted_params : np.ndarray
        Parameter vector returned by the optimizer.
    initial_params : np.ndarray
        Starting parameter values.
    reduced_chi2 : float or None
        Reduced chi-squared of the fit, if available.

    Returns
    -------
    tuple of (bool, str, str)
        ``(success, message, convergence_reason)``.
    """
    if not np.all(np.isfinite(fitted_params)):
        return False, "Non-finite parameters in result", "failed"

    if reduced_chi2 is not None and reduced_chi2 > 1e6:
        return (
            False,
            f"Poor fit quality (reduced chi-squared = {reduced_chi2:.2e})",
            "poor_fit",
        )

    if np.allclose(fitted_params, initial_params, rtol=1e-12, atol=1e-12):
        return False, "Optimizer made no progress from initial values", "no_progress"

    return True, "Optimization converged", "tolerance"


# ---------------------------------------------------------------------------
# NLSQAdapter — primary JAX-traced adapter
# ---------------------------------------------------------------------------


class NLSQAdapter(NLSQAdapterBase):
    """Adapter for the nlsq library's CurveFit optimizer.

    Uses JAX-accelerated nonlinear least squares from the nlsq package.
    The ``fit()`` method calls ``nlsq.CurveFit`` directly — no scipy delegation.
    For pure-JAX residual functions, prefer ``fit_jax()`` which passes a
    JAX-traceable function to ``CurveFit.curve_fit()``.
    """

    def __init__(self, parameter_names: list[str]) -> None:
        """Initialise the adapter.

        Parameters
        ----------
        parameter_names : list of str
            Names of parameters being optimised, in order.
        """
        self._parameter_names = parameter_names

    @property
    def name(self) -> str:
        """Adapter name (``"nlsq.CurveFit"``)."""
        return "nlsq.CurveFit"

    def supports_bounds(self) -> bool:
        """Return whether the adapter supports box bounds (always ``True``)."""
        return True

    def supports_jacobian(self) -> bool:
        """Return whether the adapter accepts an analytic Jacobian (always ``True``)."""
        return True

    def fit(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        config: NLSQConfig,
        jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        callback: Callable[..., Any] | None = None,
    ) -> NLSQResult:
        """Run NLSQ optimisation using nlsq.CurveFit.

        Wraps the residual function into the ``(xdata, *params)`` signature
        expected by ``CurveFit.curve_fit`` and normalises the result via
        ``build_result_from_nlsq``.

        Parameters
        ----------
        residual_fn : callable
            ``(params: ndarray) -> residuals: ndarray``.
        initial_params : np.ndarray
            Starting parameter values.
        bounds : tuple of np.ndarray
            ``(lower, upper)`` bound arrays.
        config : NLSQConfig
            Optimisation configuration.
        jacobian_fn : callable, optional
            Analytic Jacobian (unused by CurveFit; kept for API compatibility).
        callback : callable, optional
            Per-iteration ``curve_fit`` callback
            ``(iteration, cost, params, info=None, **kwargs) -> None``. Strictly
            observational — must not mutate solve state. Forwarded to
            ``CurveFit.curve_fit`` when provided; takes precedence over the
            Task-0 debug seam, which remains as a secondary fallback.

        Returns
        -------
        NLSQResult
            Normalised fit result.
        """
        start_time = time.perf_counter()

        lower_bounds, upper_bounds = bounds
        initial_params = np.clip(initial_params, lower_bounds, upper_bounds)
        n_params = len(initial_params)

        logger.info("NLSQAdapter.fit: %d parameters", n_params)

        try:
            # Probe residual length
            probe = residual_fn(initial_params)
            n_data = len(np.asarray(probe))

            # Create xdata/ydata for CurveFit API (target = zero residuals)
            xdata = np.arange(n_data, dtype=np.float64)
            ydata = np.zeros(n_data, dtype=np.float64)

            # Wrap residual_fn into (xdata, *params) signature.
            # jnp.array is required: nlsq 0.6.12 calls func(xdata, *args) inside
            # @jit, so *params are traced JAX scalars — np.array would raise
            # TracerArrayConversionError.
            def _wrapped(x: np.ndarray, *params: Any) -> Any:
                # jnp.Array satisfies the ndarray protocol at runtime; ignore static mismatch.
                return residual_fn(jnp.array(params, dtype=jnp.float64))  # type: ignore[arg-type]

            fitter, cache_hit = get_or_create_fitter(
                n_data=n_data,
                n_params=n_params,
                phi_angles=None,
                scaling_mode="auto",
                callable_scope=residual_fn,
            )
            if cache_hit:
                logger.debug("CurveFit cache hit for shape (%d, %d)", n_data, n_params)

            # Resolve method — dogbox is not supported by CurveFit
            method = config.method
            if method == "dogbox":
                logger.warning("Method 'dogbox' not supported by CurveFit; using 'trf'")
                method = "trf"

            logger.info(
                "NLSQAdapter settings: method=%s loss=%s gtol=%.2e max_nfev=%s x_scale=%s",
                method,
                config.loss,
                config.gtol,
                config.max_nfev if config.max_nfev is not None else f"auto({100 * n_params})",
                config.x_scale,
            )
            fit_kwargs = _optimizer_kwargs(config, method)
            if callback is not None and "callback" not in fit_kwargs:
                fit_kwargs["callback"] = callback
            _dbg_cb = _get_debug_curvefit_callback()
            if _dbg_cb is not None and "callback" not in fit_kwargs:
                fit_kwargs["callback"] = _dbg_cb
            nlsq_result = fitter.curve_fit(  # type: ignore[attr-defined,union-attr]
                f=_wrapped,
                xdata=xdata,
                ydata=ydata,
                p0=initial_params,
                bounds=(lower_bounds, upper_bounds),
                **fit_kwargs,
            )

            wall_time = time.perf_counter() - start_time

            # Normalise via result_builder (handles tuple / object / dict formats)
            result = build_result_from_nlsq(
                nlsq_result=nlsq_result,
                parameter_names=self._parameter_names,
                n_data=n_data,
                wall_time=wall_time,
            )

            # Apply convergence heuristics on top of build_result_from_nlsq
            success, message, reason = _assess_convergence(
                fitted_params=result.parameters,
                initial_params=initial_params,
                reduced_chi2=result.reduced_chi_squared,
            )
            if not success:
                logger.warning("NLSQAdapter convergence check failed: %s", message)
                # Return a corrected result with success=False
                return NLSQResult(
                    parameters=result.parameters,
                    parameter_names=self._parameter_names,
                    success=False,
                    message=message,
                    uncertainties=result.uncertainties,
                    covariance=result.covariance,
                    final_cost=result.final_cost,
                    reduced_chi_squared=result.reduced_chi_squared,
                    n_iterations=result.n_iterations,
                    n_function_evals=result.n_function_evals,
                    convergence_reason=reason,
                    residuals=result.residuals,
                    jacobian=result.jacobian,
                    wall_time_seconds=wall_time,
                    metadata=result.metadata,
                )

            return result

        except (RuntimeError, ValueError, TypeError) as exc:
            logger.error("NLSQAdapter.fit failed: %s", exc)
            wall_time = time.perf_counter() - start_time
            return build_failed_result(
                parameter_names=self._parameter_names,
                message=str(exc),
                initial_params=initial_params,
                wall_time=wall_time,
            )

    def fit_jax(
        self,
        jax_residual_fn: Callable[..., Any],
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        config: NLSQConfig,
        n_data: int,
        callback: Callable[..., Any] | None = None,
    ) -> NLSQResult:
        """Run NLSQ optimisation using a pure JAX-traceable residual function.

        This method accepts a function with the signature
        ``(xdata, *params) -> residuals`` that nlsq can trace through JAX.

        Parameters
        ----------
        jax_residual_fn : callable
            JAX-compatible ``(x, *params) -> residuals``.
        initial_params : np.ndarray
            Starting parameter values.
        bounds : tuple of np.ndarray
            ``(lower, upper)`` bound arrays.
        config : NLSQConfig
            Optimisation configuration.
        n_data : int
            Number of data points (used as CurveFit ``flength``).
        callback : callable, optional
            Per-iteration ``curve_fit`` callback (strictly observational).
            Forwarded to ``CurveFit.curve_fit`` when provided; takes precedence
            over the Task-0 debug seam.

        Returns
        -------
        NLSQResult
            Normalised fit result.
        """
        start_time = time.perf_counter()

        lower_bounds, upper_bounds = bounds
        initial_params = np.clip(initial_params, lower_bounds, upper_bounds)
        n_params = len(initial_params)

        logger.info("NLSQAdapter.fit_jax: %d parameters, %d data points", n_params, n_data)

        try:
            xdata = np.arange(n_data, dtype=np.float64)
            ydata = np.zeros(n_data, dtype=np.float64)

            fitter, cache_hit = get_or_create_fitter(
                n_data=n_data,
                n_params=n_params,
                phi_angles=None,
                scaling_mode="auto",
                callable_scope=jax_residual_fn,
            )
            if cache_hit:
                logger.debug("CurveFit cache hit for shape (%d, %d)", n_data, n_params)

            method = config.method
            if method == "dogbox":
                logger.warning("Method 'dogbox' not supported by CurveFit; using 'trf'")
                method = "trf"

            logger.info(
                "NLSQAdapter.fit_jax settings: method=%s loss=%s gtol=%.2e max_nfev=%s x_scale=%s",
                method,
                config.loss,
                config.gtol,
                config.max_nfev if config.max_nfev is not None else f"auto({100 * n_params})",
                config.x_scale,
            )
            fit_kwargs = _optimizer_kwargs(config, method)
            if callback is not None and "callback" not in fit_kwargs:
                fit_kwargs["callback"] = callback
            _dbg_cb = _get_debug_curvefit_callback()
            if _dbg_cb is not None and "callback" not in fit_kwargs:
                fit_kwargs["callback"] = _dbg_cb
            nlsq_result = fitter.curve_fit(  # type: ignore[attr-defined,union-attr]
                f=jax_residual_fn,
                xdata=xdata,
                ydata=ydata,
                p0=initial_params,
                bounds=(lower_bounds, upper_bounds),
                **fit_kwargs,
            )

            wall_time = time.perf_counter() - start_time

            # Normalise result via build_result_from_nlsq (single source of truth)
            base = build_result_from_nlsq(
                nlsq_result=nlsq_result,
                parameter_names=self._parameter_names,
                n_data=n_data,
                wall_time=wall_time,
            )

            # Use residuals already stored in the optimizer result.
            # ``build_result_from_nlsq`` extracts them from ``nlsq_result.fun``
            # which is the final residual vector the optimizer converged to —
            # re-evaluating at the same point is numerically identical and wastes
            # one full N×N forward pass.  Fall back to re-evaluation only when
            # the optimizer did not expose residuals (e.g. streaming backends).
            if base.residuals is not None:
                final_residuals = base.residuals
            else:
                logger.debug(
                    "fit_jax: optimizer did not expose final residuals; "
                    "re-evaluating residual function"
                )
                final_residuals_jax = jax_residual_fn(jnp.arange(n_data), *base.parameters)
                final_residuals = np.asarray(final_residuals_jax)
            final_cost = 0.5 * float(np.sum(final_residuals**2))
            n_dof = n_data - n_params
            reduced_chi2: float | None = 2.0 * final_cost / n_dof if n_dof > 0 else None

            success, message, reason = _assess_convergence(
                fitted_params=base.parameters,
                initial_params=initial_params,
                reduced_chi2=reduced_chi2,
            )
            if not success:
                logger.warning("fit_jax convergence check failed: %s", message)

            return NLSQResult(
                parameters=base.parameters,
                parameter_names=self._parameter_names,
                success=success,
                message=message,
                uncertainties=base.uncertainties,
                covariance=base.covariance,
                final_cost=final_cost,
                reduced_chi_squared=reduced_chi2,
                n_iterations=base.n_iterations,
                n_function_evals=base.n_function_evals,
                convergence_reason=reason,
                residuals=final_residuals,
                jacobian=base.jacobian,
                wall_time_seconds=wall_time,
                metadata=base.metadata,
            )

        except (RuntimeError, ValueError, TypeError) as exc:
            logger.error("NLSQAdapter.fit_jax failed: %s", exc)
            wall_time = time.perf_counter() - start_time
            return build_failed_result(
                parameter_names=self._parameter_names,
                message=str(exc),
                initial_params=initial_params,
                wall_time=wall_time,
            )


# ---------------------------------------------------------------------------
# NLSQWrapper — stable fallback with memory-aware strategy routing
# ---------------------------------------------------------------------------


class NLSQWrapper(NLSQAdapterBase):
    """Stable fallback adapter with memory-aware strategy routing.

    Selects between STANDARD, LARGE, and STREAMING optimization tiers based
    on the estimated peak memory usage of the Jacobian matrix.  Falls back
    down the tier list if a higher tier fails.

    Fallback order (descending resource intensity):
        STREAMING → LARGE → STANDARD

    Each tier is retried up to ``max_retries`` times before falling back.
    """

    def __init__(
        self,
        parameter_names: list[str],
        enable_large_dataset: bool = True,
        enable_recovery: bool = True,
        max_retries: int = 3,
    ) -> None:
        """Initialise the wrapper.

        Parameters
        ----------
        parameter_names : list of str
            Names of parameters being optimised, in order.
        enable_large_dataset : bool
            Allow the LARGE tier when memory warrants it.
        enable_recovery : bool
            Enable cross-tier fallback on failure.
        max_retries : int
            Maximum per-tier retries before falling back.
        """
        self._parameter_names = parameter_names
        self._enable_large_dataset = enable_large_dataset
        self._enable_recovery = enable_recovery
        self._max_retries = max(1, max_retries)

    @property
    def name(self) -> str:
        """Adapter name (``"nlsq.NLSQWrapper"``)."""
        return "nlsq.NLSQWrapper"

    def supports_bounds(self) -> bool:
        """Return whether the adapter supports box bounds (always ``True``)."""
        return True

    def supports_jacobian(self) -> bool:
        """Return whether the adapter accepts an analytic Jacobian (always ``True``)."""
        return True

    def fit(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        config: NLSQConfig,
        jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> NLSQResult:
        """Run NLSQ optimisation with automatic memory-based strategy routing.

        Parameters
        ----------
        residual_fn : callable
            ``(params: ndarray) -> residuals: ndarray``.
        initial_params : np.ndarray
            Starting parameter values.
        bounds : tuple of np.ndarray
            ``(lower, upper)`` bound arrays.
        config : NLSQConfig
            Optimisation configuration.
        jacobian_fn : callable, optional
            Analytic Jacobian (for API compatibility).

        Returns
        -------
        NLSQResult
            Normalised fit result.
        """
        start_time = time.perf_counter()

        lower_bounds, upper_bounds = bounds
        initial_params = np.clip(initial_params, lower_bounds, upper_bounds)
        n_params = len(initial_params)

        # --- Determine data size via a probe call ---
        try:
            probe = residual_fn(initial_params)
            n_data = len(np.asarray(probe))
        except Exception as exc:  # noqa: BLE001
            logger.error("NLSQWrapper: residual probe failed: %s", exc)
            wall_time = time.perf_counter() - start_time
            return build_failed_result(
                parameter_names=self._parameter_names,
                message=f"Residual probe failed: {exc}",
                initial_params=initial_params,
                wall_time=wall_time,
            )

        # --- Memory-based strategy selection ---
        decision = select_nlsq_strategy(n_points=n_data, n_params=n_params)
        logger.info(
            "NLSQWrapper strategy: %s (%s)",
            decision.strategy.value,
            decision.reason,
        )

        # --- Build xdata/ydata for CurveFit-style API ---
        xdata = np.arange(n_data, dtype=np.float64)
        ydata = np.zeros(n_data, dtype=np.float64)

        # jnp.array required: nlsq 0.6.12 calls func(xdata, *args) inside @jit.
        def _wrapped(x: np.ndarray, *params: Any) -> Any:
            return residual_fn(jnp.array(params, dtype=jnp.float64))  # type: ignore[arg-type]

        method = config.method
        if method == "dogbox":
            logger.warning("Method 'dogbox' unsupported by nlsq; using 'trf'")
            method = "trf"
        loss = config.loss
        logger.info(
            "NLSQWrapper settings: method=%s ftol=%.2e xtol=%.2e gtol=%.2e "
            "max_nfev=%s x_scale=%r (loss=%r not applied; "
            "STREAMING tier ignores tolerances)",
            method,
            config.ftol,
            config.xtol,
            config.gtol,
            config.max_nfev if config.max_nfev is not None else f"auto({100 * n_params})",
            config.x_scale,
            loss,
        )

        # --- Tier ordering: initial strategy → fallback cascade ---
        tiers = self._build_tier_list(decision.strategy)

        last_exc: Exception | None = None
        for tier in tiers:
            result = self._try_tier(
                tier=tier,
                wrapped_fn=_wrapped,
                xdata=xdata,
                ydata=ydata,
                initial_params=initial_params,
                bounds=(lower_bounds, upper_bounds),
                n_data=n_data,
                n_params=n_params,
                method=method,
                loss=loss,
                config=config,
                start_time=start_time,
            )
            if result is not None:
                return result
            # Result is None → this tier exhausted all retries; try next
            last_exc = RuntimeError(f"Tier {tier.value} failed after {self._max_retries} retries")
            if not self._enable_recovery:
                break

        # All tiers failed
        wall_time = time.perf_counter() - start_time
        message = str(last_exc) if last_exc else "All NLSQ tiers failed"
        logger.error("NLSQWrapper: %s", message)
        return build_failed_result(
            parameter_names=self._parameter_names,
            message=message,
            initial_params=initial_params,
            wall_time=wall_time,
        )

    def _build_tier_list(self, initial_strategy: NLSQStrategy) -> list[NLSQStrategy]:
        """Return ordered list of tiers to attempt, starting from initial_strategy."""
        all_tiers = [NLSQStrategy.STREAMING, NLSQStrategy.LARGE, NLSQStrategy.STANDARD]

        # Start from the selected strategy and work downward
        try:
            start_idx = all_tiers.index(initial_strategy)
        except ValueError:
            start_idx = len(all_tiers) - 1  # default to STANDARD

        tiers = all_tiers[start_idx:]

        # Drop LARGE if large-dataset support is disabled
        if not self._enable_large_dataset and NLSQStrategy.LARGE in tiers:
            tiers = [t for t in tiers if t != NLSQStrategy.LARGE]

        return tiers

    def _try_tier(
        self,
        tier: NLSQStrategy,
        wrapped_fn: Callable[..., np.ndarray],
        xdata: np.ndarray,
        ydata: np.ndarray,
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        n_data: int,
        n_params: int,
        method: str,
        loss: str,
        start_time: float,
        config: NLSQConfig | None = None,
    ) -> NLSQResult | None:
        """Attempt a single tier up to ``max_retries`` times.

        Returns
        -------
        NLSQResult or None
            The result on success, or ``None`` if all retries failed.
        """
        lower_bounds, upper_bounds = bounds
        for attempt in range(self._max_retries):
            try:
                raw_result = self._call_tier(
                    tier=tier,
                    wrapped_fn=wrapped_fn,
                    xdata=xdata,
                    ydata=ydata,
                    p0=initial_params,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                    n_data=n_data,
                    n_params=n_params,
                    method=method,
                    loss=loss,
                    config=config,
                )
                wall_time = time.perf_counter() - start_time
                result = build_result_from_nlsq(
                    nlsq_result=raw_result,
                    parameter_names=self._parameter_names,
                    n_data=n_data,
                    wall_time=wall_time,
                    metadata={"strategy": tier.value, "attempt": attempt},
                )

                # Apply convergence heuristics (same as NLSQAdapter)
                success, message, reason = _assess_convergence(
                    fitted_params=result.parameters,
                    initial_params=initial_params,
                    reduced_chi2=result.reduced_chi_squared,
                )
                if not success:
                    logger.warning(
                        "NLSQWrapper: tier %s convergence check failed: %s",
                        tier.value,
                        message,
                    )
                    result = NLSQResult(
                        parameters=result.parameters,
                        parameter_names=self._parameter_names,
                        success=False,
                        message=message,
                        uncertainties=result.uncertainties,
                        covariance=result.covariance,
                        final_cost=result.final_cost,
                        reduced_chi_squared=result.reduced_chi_squared,
                        n_iterations=result.n_iterations,
                        n_function_evals=result.n_function_evals,
                        convergence_reason=reason,
                        residuals=result.residuals,
                        jacobian=result.jacobian,
                        wall_time_seconds=wall_time,
                        metadata=result.metadata,
                    )

                logger.info(
                    "NLSQWrapper: tier %s succeeded on attempt %d/%d",
                    tier.value,
                    attempt + 1,
                    self._max_retries,
                )
                return result

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "NLSQWrapper: tier %s attempt %d/%d failed: %s",
                    tier.value,
                    attempt + 1,
                    self._max_retries,
                    exc,
                )

        return None

    def _call_tier(
        self,
        tier: NLSQStrategy,
        wrapped_fn: Callable[..., np.ndarray],
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        n_data: int,
        n_params: int,
        method: str,
        loss: str,
        config: NLSQConfig | None = None,
    ) -> Any:
        """Dispatch a single call to the appropriate nlsq function/class.

        ``ftol``/``xtol``/``gtol``/``x_scale``/``max_nfev`` are propagated to
        STANDARD and LARGE tiers via ``**kwargs`` (both ``nlsq.curve_fit`` and
        ``curve_fit_large`` forward unknown kwargs to scipy ``least_squares``).
        ``loss`` is intentionally omitted on all tiers: the NLSQWrapper path
        wraps residuals as a plain numpy function, so robust-loss kernels would
        re-enter JAX tracing and raise ``TracerArrayConversionError``.
        STREAMING tier ignores tolerances (fixed ``AdaptiveHybridStreamingOptimizer``
        signature).
        """
        # Solver kwargs propagated to STANDARD and LARGE tiers.
        solver_kwargs: dict[str, Any] = {}
        if config is not None:
            solver_kwargs["ftol"] = config.ftol
            solver_kwargs["xtol"] = config.xtol
            solver_kwargs["gtol"] = config.gtol
            solver_kwargs["x_scale"] = config.x_scale
            if config.max_nfev is not None:
                solver_kwargs["max_nfev"] = config.max_nfev

        if tier == NLSQStrategy.STREAMING:
            if not STREAMING_AVAILABLE or AdaptiveHybridStreamingOptimizer is None:
                raise RuntimeError(
                    "AdaptiveHybridStreamingOptimizer not available in this nlsq build"
                )
            optimizer = AdaptiveHybridStreamingOptimizer()
            return optimizer.fit(
                data_source=(xdata, ydata),
                func=wrapped_fn,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
            )

        if tier == NLSQStrategy.LARGE:
            return curve_fit_large(
                f=wrapped_fn,
                xdata=xdata,
                ydata=ydata,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                **solver_kwargs,
            )

        # STANDARD tier.  loss intentionally omitted — see docstring.
        _ = loss
        return curve_fit(  # type: ignore[call-arg, arg-type]
            f=wrapped_fn,
            xdata=xdata,
            ydata=ydata,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            method=method,  # type: ignore[arg-type]
            **solver_kwargs,
        )


__all__ = [
    "CachedModel",
    "ModelCacheKey",
    "NLSQAdapter",
    "NLSQWrapper",
    "STREAMING_AVAILABLE",
    "clear_model_cache",
    "get_cache_stats",
    "get_or_create_fitter",
]
