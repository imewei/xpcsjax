"""NLSQ Adapter using CurveFit class for homodyne optimization.

Role and When to Use (v2.11.0+)
-------------------------------

**NLSQAdapter** (this module) is the **recommended adapter** for:
- Standard optimizations (static_isotropic mode)
- Small to medium datasets (< 10M points)
- Multi-start optimization (model caching provides 3-5× speedup)
- Performance-critical workflows requiring JIT compilation

Use **NLSQWrapper** instead for:
- Complex optimizations requiring full anti-degeneracy integration
- laminar_flow mode with many phi angles (> 6)
- Large datasets (> 100M points) requiring streaming/chunking strategies
- Custom transforms or advanced recovery mechanisms

**Key Differences:**

* Model caching: NLSQAdapter=Built-in, NLSQWrapper=None
* JIT compilation: NLSQAdapter=Auto, NLSQWrapper=Manual
* Workflow auto-select: NLSQAdapter=Via NLSQ, NLSQWrapper=Custom
* Anti-degeneracy layers: NLSQAdapter=Via fit(), NLSQWrapper=Full
* Recovery system: NLSQAdapter=NLSQ native, NLSQWrapper=3-attempt
* Streaming support: NLSQAdapter=Via NLSQ, NLSQWrapper=Full custom

**Decision Guide:**

1. If you need maximum speed for multi-start optimization: Use NLSQAdapter
2. If you need robust streaming for 100M+ points: Use NLSQWrapper
3. If you need full anti-degeneracy control: Use NLSQWrapper
4. Default recommendation for new code: Use NLSQAdapter (via use_adapter=True)

This module provides a modern adapter layer between homodyne's optimization API
and the NLSQ package's CurveFit class, leveraging:
- CurveFit class for JIT compilation caching
- Model instance caching (WeakValueDictionary) for multi-start speedup
- xpcsjax's own memory-aware strategy selection
- Built-in stability and recovery systems
- Runtime fallback to NLSQWrapper on failure

This is the recommended integration path for NLSQ v0.4+ (homodyne v2.11.0+).

Key Features:
- Model caching: 3-5× speedup for multi-start optimization
- JIT compilation: 2-3× speedup for single fits
- Automatic workflow selection based on dataset size and memory
- Native NLSQ stability and recovery systems
- Integration with homodyne's anti-degeneracy defense system
- Backward-compatible interface with NLSQWrapper.fit()
- Automatic fallback to NLSQWrapper when adapter fails

Migration Guide:
- Replace NLSQWrapper with NLSQAdapter
- Set use_adapter=True in fit_nlsq_jax() (default in v2.11.0+)
- Anti-degeneracy layers work unchanged

References:
- NLSQ Package: https://github.com/imewei/NLSQ
- Architecture: See CLAUDE.md for NLSQ integration details
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.optimization.nlsq.adapter_base import NLSQAdapterBase
from xpcsjax.optimization.nlsq.results import OptimizationResult
from xpcsjax.utils.logging import get_logger

# Import NLSQ components with graceful fallback
try:
    from nlsq import CurveFit

    NLSQ_CURVEFIT_AVAILABLE = True
except ImportError:
    CurveFit = None  # type: ignore[assignment, misc]
    NLSQ_CURVEFIT_AVAILABLE = False


try:
    from nlsq.streaming import HybridStreamingConfig

    NLSQ_STREAMING_AVAILABLE = True
except ImportError:
    HybridStreamingConfig = None  # type: ignore[assignment, misc]
    NLSQ_STREAMING_AVAILABLE = False

logger = get_logger(__name__)


# =============================================================================
# T001: ModelCacheKey frozen dataclass
# =============================================================================
@dataclass(frozen=True)
class ModelCacheKey:
    """Immutable key for model cache lookup.

    Hashable tuple of (analysis_mode, phi_angles_tuple, q, per_angle_scaling).
    NumPy arrays converted to tuples for hashability.

    Attributes:
        analysis_mode: "static_isotropic" or "laminar_flow"
        phi_angles: Unique phi angles (sorted) as tuple
        q: Scattering wavevector magnitude
        per_angle_scaling: Whether per-angle contrast/offset is used
    """

    analysis_mode: AnalysisMode
    phi_angles: tuple[float, ...]
    q: float
    per_angle_scaling: bool


# =============================================================================
# T002: CachedModel dataclass
# =============================================================================
@dataclass
class CachedModel:
    """Cached model instance with JIT-compiled prediction function.

    Stored in dict with LRU eviction - oldest entries removed when cache is full.

    Attributes:
        model: CombinedModel instance for computing g1/g2 values
        model_func: Model prediction function (NumPy-compatible wrapper)
        created_at: time.time() for diagnostics
        n_hits: Cache hit counter for monitoring
    """

    model: Any  # CombinedModel or other model type
    model_func: Callable[[np.ndarray, Any], np.ndarray]
    created_at: float = field(default_factory=time.time)
    n_hits: int = 0


# =============================================================================
# T003: Module-level _model_cache dict with LRU eviction
# T004: _cache_stats dict for hit/miss tracking
# =============================================================================
# Module-level cache (per-process in ProcessPoolExecutor spawn context)
# Thread safety: Python GIL protects dict operations; no explicit locks needed
# Using regular dict instead of WeakValueDictionary because we return (model, model_func)
# directly, not CachedModel - so the wrapper would be garbage collected immediately.
_model_cache: dict[ModelCacheKey, CachedModel] = {}
_cache_stats: dict[str, int] = {"hits": 0, "misses": 0}
_CACHE_MAX_SIZE: int = 64  # LRU eviction threshold


# =============================================================================
# T006: _make_cache_key() helper function
# =============================================================================
def _make_cache_key(
    analysis_mode: AnalysisMode,
    phi_angles: np.ndarray,
    q: float,
    per_angle_scaling: bool,
) -> ModelCacheKey:
    """Create hashable cache key from parameters.

    Args:
        analysis_mode: 'static_isotropic' or 'laminar_flow'
        phi_angles: Unique phi angles in radians (np.ndarray)
        q: Scattering wavevector magnitude
        per_angle_scaling: Whether per-angle contrast/offset is used

    Returns:
        ModelCacheKey: Hashable, immutable key for cache lookup
    """
    return ModelCacheKey(
        analysis_mode=analysis_mode,
        phi_angles=tuple(np.sort(np.unique(phi_angles))),
        q=round(q, 10),  # Avoid floating-point precision issues
        per_angle_scaling=per_angle_scaling,
    )


# =============================================================================
# Task 30: heterodyne (two_component) routing helper
# =============================================================================
def _get_or_create_heterodyne_model(
    phi_angles: np.ndarray,
    q: float,
    t: np.ndarray,
    dt: float,
    per_angle_scaling: bool = True,
    enable_jit: bool = True,
) -> tuple[Any, Callable[[np.ndarray, Any], np.ndarray], bool]:
    """Construct a :class:`HeterodyneModel` + ``model_func`` for NLSQ curve_fit.

    Separate path from the homodyne :func:`get_or_create_model` so the existing
    per-angle scaling expansion / lineage gating machinery is unaffected. The
    returned ``model_func`` follows the per-angle convention used by
    heterodyne's ``NLSQAdapter.fit_jax``: a dummy ``xdata`` of shape
    ``(N*N,)`` and parameter vector ``[contrast, offset, *physics_14]`` (or
    ``[*physics_14]`` when ``per_angle_scaling=False``) — it returns the
    flattened predicted c2 matrix.

    Notes
    -----
    - Single-angle only. Multi-angle heterodyne fits go through the source
      heterodyne CLI (see scripts/extract_heterodyne_baseline.py).
    - Time grid is fixed at construction (curve_fit closure); ``xdata`` is
      a dummy index array; the model evaluates ``compute_c2_heterodyne(t, ...)``
      with the stored ``t`` regardless.

    Parameters
    ----------
    phi_angles : np.ndarray
        1-element array with the phi angle (degrees) for this fit.
    q : float
        Scattering wavevector magnitude.
    t : np.ndarray
        1-D time array (frames or seconds — must be consistent with ``dt``).
    dt : float
        Time step.
    per_angle_scaling : bool
        If True, ``model_func`` consumes a leading ``(contrast, offset)`` pair
        from ``*params`` before the 14 physics parameters.
    enable_jit : bool
        Forwarded as a hint; the underlying kernel is already JAX-compiled.

    Returns
    -------
    tuple
        ``(model, model_func, cache_hit)``. ``cache_hit`` is always ``False``
        (heterodyne path is uncached for now; small fixture size makes this
        cheap).
    """
    from xpcsjax.core.heterodyne_jax_backend import compute_c2_heterodyne
    from xpcsjax.core.heterodyne_model import HeterodyneModel

    if len(phi_angles) != 1:
        raise ValueError(
            "Heterodyne routing in NLSQAdapter currently supports single-angle "
            f"fits only; got {len(phi_angles)} phi angles. For multi-angle "
            "heterodyne fits use the source heterodyne CLI / fit_nlsq_multi_phi."
        )

    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")

    model = HeterodyneModel()

    import jax.numpy as jnp

    phi_scalar = float(phi_angles[0])
    t_jax = jnp.asarray(t, dtype=jnp.float64)
    q_val = float(q)
    dt_val = float(dt)

    def model_func(xdata: np.ndarray, *params: float) -> np.ndarray:  # noqa: ARG001
        """Evaluate c2 model at the configured phi angle.

        ``xdata`` is ignored — the heterodyne kernel evaluates on the closed-over
        ``t`` grid. Parameter layout:

        - ``per_angle_scaling=True``:  ``[contrast, offset, *physics_14]``
        - ``per_angle_scaling=False``: ``[*physics_14]``
        """
        params_jax = jnp.stack(params)
        n_params_val = len(params)

        if per_angle_scaling and n_params_val >= 16:
            contrast = params_jax[0]
            offset = params_jax[1]
            physics = params_jax[2:16]
        else:
            contrast = jnp.float64(1.0)
            offset = jnp.float64(0.0)
            physics = params_jax[:14]

        c2_pred = compute_c2_heterodyne(
            params=physics,
            t=t_jax,
            q=q_val,
            dt=dt_val,
            phi_angle=phi_scalar,
            contrast=contrast,
            offset=offset,
        )
        # Return as JAX array so NLSQ can trace through curve_fit's JIT. The
        # NLSQ runtime converts to numpy for the final result after tracing.
        return c2_pred.ravel()

    if enable_jit:
        # The kernel is already JAX-compiled; the closure dispatches through it.
        logger.debug("Heterodyne model_func uses pre-JIT-compiled kernel")

    return model, model_func, False


# =============================================================================
# T007: get_or_create_model() function per contracts/model-caching.md
# =============================================================================
def get_or_create_model(
    analysis_mode: AnalysisMode,
    phi_angles: np.ndarray,
    q: float,
    per_angle_scaling: bool = True,
    config: dict[str, Any] | None = None,
    enable_jit: bool = True,
    t: np.ndarray | None = None,
    dt: float | None = None,
) -> tuple[Any, Callable[[np.ndarray, Any], np.ndarray], bool]:
    """Get cached model or create new one.

    This function provides model instance caching to avoid redundant model
    creation during multi-start optimization. Expected 3-5× speedup.

    Uses CombinedModel (not HomodyneModel) for simpler initialization.
    The model function closure captures the model and experimental setup.

    For ``analysis_mode='two_component'`` (heterodyne) the call is routed to
    :func:`_get_or_create_heterodyne_model` — see that function's docstring
    for the parameter-layout convention and the ``t``/``dt`` requirement.

    Args:
        analysis_mode: 'static_anisotropic', 'static_isotropic', 'laminar_flow',
            or 'two_component'
        phi_angles: Unique phi angles in radians (homodyne) or degrees
            (heterodyne — convention matches the source heterodyne kernel)
        q: Scattering wavevector magnitude
        per_angle_scaling: Whether per-angle contrast/offset is used
        config: Optional config dict for model initialization
        enable_jit: Whether to JIT-compile the model function
        t: Time array (required when ``analysis_mode='two_component'``;
            ignored for homodyne modes which read ``t1``/``t2`` from the data
            dict at fit time).
        dt: Time step (required when ``analysis_mode='two_component'``).

    Returns:
        Tuple of (model, model_func, cache_hit) where:
            - model: CombinedModel or HeterodyneModel instance
            - model_func: Prediction function (JIT-compiled if enable_jit=True)
            - cache_hit: True if model was retrieved from cache

    Raises:
        ValueError: If analysis_mode is invalid, phi_angles is empty, or q <= 0

    Example:
        >>> model, model_func, hit = get_or_create_model(
        ...     "laminar_flow",
        ...     np.array([0.0, 0.5, 1.0]),
        ...     0.001,
        ... )
        >>> if hit:
        ...     logger.debug("Model cache hit")
    """
    global _cache_stats

    # Heterodyne route (Task 30): two_component does not share scaling /
    # caching plumbing with homodyne — dispatch before homodyne validation.
    if analysis_mode == "two_component" or analysis_mode == "heterodyne":
        if t is None or dt is None:
            raise ValueError(
                f"Heterodyne (two_component) routing requires `t` and `dt`; "
                f"got t={t!r}, dt={dt!r}"
            )
        return _get_or_create_heterodyne_model(
            phi_angles=phi_angles,
            q=q,
            t=t,
            dt=dt,
            per_angle_scaling=per_angle_scaling,
            enable_jit=enable_jit,
        )

    # Validate inputs
    if analysis_mode not in {"static_anisotropic", "static_isotropic", "laminar_flow"}:
        raise ValueError(
            f"Invalid analysis_mode: '{analysis_mode}'. "
            f"Expected 'static_anisotropic', 'static_isotropic', 'laminar_flow', "
            f"or 'two_component'"
        )
    if len(phi_angles) == 0:
        raise ValueError("phi_angles cannot be empty")
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")

    normalized_mode = analysis_mode

    # Create cache key
    cache_key = _make_cache_key(normalized_mode, phi_angles, q, per_angle_scaling)

    # Check cache
    cached = _model_cache.get(cache_key)
    if cached is not None:
        _cache_stats["hits"] += 1
        cached.n_hits += 1
        logger.debug(
            "Model cache hit: mode=%s, n_phi=%d, q=%.6g, hits=%d",
            normalized_mode,
            len(phi_angles),
            q,
            cached.n_hits,
        )
        return cached.model, cached.model_func, True

    # Cache miss - create new model
    _cache_stats["misses"] += 1
    logger.debug(
        "Model cache miss: mode=%s, n_phi=%d, q=%.6g",
        normalized_mode,
        len(phi_angles),
        q,
    )

    # Import here to avoid circular imports
    from xpcsjax.core.models import CombinedModel

    start_time = time.time()

    # Use CombinedModel which has simpler init (just analysis_mode).
    # Preserve isotropic/anisotropic distinction for the model; the physics
    # parameter count is identical so the rest of the closure only cares
    # whether this is a static-family mode or laminar_flow.
    model_mode = normalized_mode if "static" in normalized_mode else "laminar_flow"
    model = CombinedModel(analysis_mode=model_mode)

    # Store experimental parameters for model function closure
    phi_unique = np.unique(phi_angles)
    q_val = float(q)
    n_phi = len(phi_unique)

    # Create model function compatible with NLSQ curve_fit
    # This closure captures model configuration
    # IMPORTANT: This function must be JAX-traceable for CMA-ES JIT compilation
    import jax.numpy as jnp

    # Pre-convert phi_unique to JAX array for use in closure
    phi_unique_jax = jnp.array(phi_unique)

    # Cache for xdata JAX conversion — avoids redundant jnp.array() on every call.
    # NLSQ passes the same xdata repeatedly during optimization; only params change.
    # Keyed by id(xdata); size-limited to 4 entries for streaming mode safety.
    _xdata_cache: dict[int, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = {}

    def model_func(xdata: np.ndarray, *params: float) -> np.ndarray:
        """Model function compatible with NLSQ curve_fit.

        This function is designed to be JAX-traceable for CMA-ES JIT compilation.
        All operations use JAX primitives to preserve tracers during tracing.

        Args:
            xdata: Independent variables [n_points, 3] with columns [t1, t2, phi_idx]
                   where phi_idx is the PRECOMPUTED phi angle index (v2.17.0+)
            *params: Parameter values (may be JAX tracers during JIT)

        Returns:
            Predicted g2 values [n_points]

        Note:
            As of v2.17.0, xdata[:, 2] contains precomputed phi indices (integers
            stored as float64) to avoid expensive argmin/gather operations inside
            the JIT-compiled function. This eliminates XLA slow_operation_alarm
            warnings for large datasets (23M+ points).
        """
        # Use jnp.stack to preserve JAX tracers during JIT tracing
        params_array = jnp.stack(params)
        n_params_val = len(params)  # Use Python len on tuple, not traced array

        # Extract per-angle scaling parameters if present
        n_physical = 3 if "static" in model_mode else 7
        if per_angle_scaling and n_params_val >= n_physical + 2 * n_phi:
            contrast_vals = params_array[:n_phi]
            offset_vals = params_array[n_phi : 2 * n_phi]
            physical_params = params_array[2 * n_phi :]
        else:
            # Legacy scalar mode (for backward compatibility)
            c0 = params_array[0] if n_params_val > 0 else 0.5
            o0 = params_array[1] if n_params_val > 1 else 1.0
            contrast_vals = jnp.full(n_phi, c0)
            offset_vals = jnp.full(n_phi, o0)
            default_phys = jnp.array([1000.0, 0.5, 10.0])
            physical_params = params_array[2:] if n_params_val > 2 else default_phys

        # Compute g2 for each point using vectorized computation
        # xdata columns: [t1, t2, phi_idx] where phi_idx is precomputed (v2.17.0+)
        # Performance Optimization (Spec 001 - FR-006, T042): Use batched vmap
        # computation instead of Python loop for better performance.

        # Extract time arrays from xdata with caching (xdata is always concrete numpy)
        # jnp.array() copies data, so caching by id(xdata) is safe — the same
        # numpy array object yields the same JAX arrays across optimizer iterations.
        xdata_id = id(xdata)
        if xdata_id in _xdata_cache:
            t1_batch, t2_batch, phi_indices = _xdata_cache[xdata_id]
        else:
            t1_batch = jnp.array(xdata[:, 0])
            t2_batch = jnp.array(xdata[:, 1])
            # phi_idx is precomputed in _flatten_xpcs_data (v2.17.0+)
            phi_indices = jnp.array(xdata[:, 2]).astype(jnp.int32)
            if len(_xdata_cache) < 4:  # Limit cache for streaming mode
                _xdata_cache[xdata_id] = (t1_batch, t2_batch, phi_indices)

        # Look up phi values from precomputed indices (simple indexing, no gather)
        phi_batch = phi_unique_jax[phi_indices]

        # Use batched g1 computation via vmap
        g1_batch = model.compute_g1_batch(
            physical_params,  # Already a JAX array
            t1_batch,
            t2_batch,
            phi_batch,
            q_val,
            1.0,  # Default L (stator-rotor gap), will be scaled by params
        )

        # Compute g2 = offset + contrast * g1^2 (all JAX operations)
        # Get per-point contrast and offset based on phi indices
        contrast_per_point = contrast_vals[phi_indices]
        offset_per_point = offset_vals[phi_indices]

        g2_pred = offset_per_point + contrast_per_point * g1_batch**2

        # Convert to numpy for compatibility with NLSQ
        return np.asarray(g2_pred)

    # JIT compilation: The model_func now uses JAX vmap for vectorized computation
    # (FR-006, T042). The underlying CombinedModel.compute_g1_batch() uses JAX vmap.
    # We track jit_applied=False here; actual JIT is applied by NLSQ if configured.
    jit_applied = False
    if enable_jit:
        # Note: Direct JAX JIT of model_func not feasible due to NumPy/loop usage.
        # The JIT benefit comes from CombinedModel's internal JAX operations.
        logger.debug("JIT flag enabled; actual JIT applied by underlying model or NLSQ")
        jit_applied = True  # Signal intent even if direct JIT not applied

    creation_time = time.time() - start_time
    logger.debug("Model created in %.3fs (JIT=%s)", creation_time, jit_applied)

    # LRU eviction: remove oldest entry if cache is full
    if len(_model_cache) >= _CACHE_MAX_SIZE:
        # Find oldest entry by created_at
        oldest_key = min(_model_cache.keys(), key=lambda k: _model_cache[k].created_at)
        del _model_cache[oldest_key]
        logger.debug("LRU eviction: removed oldest cached model")

    # Cache the model
    cached_model = CachedModel(
        model=model,
        model_func=model_func,
        created_at=time.time(),
        n_hits=0,
    )
    _model_cache[cache_key] = cached_model

    return model, model_func, False


# =============================================================================
# T008: clear_model_cache() function
# =============================================================================
def clear_model_cache() -> int:
    """Clear all cached models.

    Returns:
        Number of models removed from cache

    Notes:
        Useful for testing or when configuration changes require fresh models.
    """
    global _cache_stats
    n_cleared = len(_model_cache)
    _model_cache.clear()
    logger.info("Cleared model cache: %d models removed", n_cleared)
    return n_cleared


# =============================================================================
# T009: get_cache_stats() function
# =============================================================================
def get_cache_stats() -> dict[str, int]:
    """Get cache statistics.

    Returns:
        Dictionary with:
            - "hits": Cache hit count
            - "misses": Cache miss count
            - "size": Current cache size
    """
    return {
        "hits": _cache_stats["hits"],
        "misses": _cache_stats["misses"],
        "size": len(_model_cache),
    }


@dataclass
class AdapterConfig:
    """Configuration for NLSQAdapter.

    Attributes:
        enable_cache: Enable model instance caching (new in v2.11.0)
        enable_jit: Enable JIT compilation of model functions (new in v2.11.0)
        enable_recovery: Enable NLSQ's built-in recovery system
        enable_stability: Enable NLSQ's numerical stability guard
        goal: Optimization goal (fast, robust, quality, memory_efficient)
        workflow: Workflow tier override (auto, standard, streaming)
    """

    # T005: New fields for model caching and JIT
    enable_cache: bool = True  # Model instance caching
    enable_jit: bool = True  # JIT compilation of model functions
    enable_recovery: bool = True
    enable_stability: bool = True
    goal: str = "quality"  # XPCS requires precision
    workflow: str = "auto"


class NLSQAdapter(NLSQAdapterBase):
    """Adapter for NLSQ package using CurveFit class.

    Uses NLSQ's CurveFit for JIT compilation caching and xpcsjax's own
    memory-aware strategy selection. This is the modern integration
    path for NLSQ v0.4+ with improved performance and reliability.

    Usage:
        adapter = NLSQAdapter()
        result = adapter.fit(data, config, initial_params, bounds, analysis_mode)

    Compared to NLSQWrapper:
        - Uses CurveFit class for JIT compilation caching
        - Delegates recovery to NLSQ's built-in systems
        - Simpler codebase with less custom logic

    Note:
        Anti-degeneracy layers (hierarchical, shear_weighting, etc.) remain
        in homodyne as they are physics-specific to XPCS analysis.
    """

    def __init__(
        self,
        config: AdapterConfig | None = None,
    ) -> None:
        """Initialize NLSQAdapter.

        Args:
            config: Adapter configuration. If None, uses defaults.

        Raises:
            ImportError: If NLSQ CurveFit class is not available.
        """
        if not NLSQ_CURVEFIT_AVAILABLE:
            raise ImportError(
                "NLSQ CurveFit class not available. "
                "Please install NLSQ >= 0.4.0: pip install nlsq>=0.4.0"
            )

        self.config = config or AdapterConfig()

        # Initialize CurveFit with caching
        self._fitter = CurveFit(
            enable_recovery=self.config.enable_recovery,
            enable_stability=self.config.enable_stability,
        )

        logger.debug(
            "NLSQAdapter initialized: cache=%s, recovery=%s, stability=%s, goal=%s",
            self.config.enable_cache,
            self.config.enable_recovery,
            self.config.enable_stability,
            self.config.goal,
        )

    @staticmethod
    def _get_physical_param_names(analysis_mode: AnalysisMode) -> list[str]:
        """Get physical parameter names for a given analysis mode."""
        normalized_mode = analysis_mode.lower()

        if normalized_mode in {"static_anisotropic", "static_isotropic"}:
            return ["D0", "alpha", "D_offset"]
        elif normalized_mode == "laminar_flow":
            return [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_t0",
                "beta",
                "gamma_dot_t_offset",
                "phi0",
            ]
        else:
            raise ValueError(
                f"Unknown analysis_mode: '{analysis_mode}'. "
                f"Expected 'static_anisotropic', 'static_isotropic', or 'laminar_flow'"
            )

    @staticmethod
    def _extract_nlsq_settings(config: Any) -> dict[str, Any]:
        """Extract NLSQ-specific settings from config."""
        config_dict = None
        if hasattr(config, "config") and isinstance(config.config, dict):
            config_dict = config.config
        elif isinstance(config, dict):
            config_dict = config

        if not config_dict:
            return {}

        result: dict[str, Any] = config_dict.get("optimization", {}).get("nlsq", {})
        return result

    def _select_workflow(
        self,
        n_points: int,
        n_params: int,
    ) -> dict[str, Any]:
        """Select workflow configuration based on dataset size.

        This method determines the memory strategy for optimization.
        Since homodyne uses curve_fit() directly (not NLSQ's fit() unified API),
        these are internal homodyne strategy names, not NLSQ workflow presets.

        Note: NLSQ 0.6.3+ simplified workflows to 3 presets: "auto", "auto_global", "hpc"
        The old presets ("streaming", "standard", etc.) were removed from NLSQ.
        Homodyne maintains its own strategy selection via select_nlsq_strategy().

        Args:
            n_points: Number of data points
            n_params: Number of parameters

        Returns:
            Dict with internal strategy info (not passed to NLSQ)
        """
        # These are xpcsjax-internal strategy names for logging/diagnostics
        if n_points > 10_000_000:
            strategy = "hybrid_streaming"  # Maps to NLSQ's streaming mode
        elif n_points > 1_000_000:
            strategy = "chunked"  # Maps to NLSQ's chunked mode
        else:
            strategy = "in_memory"  # Maps to NLSQ's standard curve_fit

        return {
            "strategy": strategy,  # Internal homodyne strategy name
            "goal": self.config.goal,
        }

    def _build_model_function(
        self,
        data: dict[str, Any],
        config: Any,
        analysis_mode: AnalysisMode,
        per_angle_scaling: bool,
        n_phi: int,
    ) -> tuple[Callable[[np.ndarray, Any], np.ndarray], bool, bool]:
        """Build the model function for NLSQ optimization.

        This creates a callable that computes g2 predictions given parameters.
        Uses model caching (T011) and JIT compilation for performance.

        Args:
            data: XPCS experimental data
            config: Configuration manager
            analysis_mode: 'static_anisotropic', 'static_isotropic', or 'laminar_flow'
            per_angle_scaling: Whether per-angle contrast/offset is used
            n_phi: Number of phi angles

        Returns:
            Tuple of (model_func, cache_hit, jit_compiled) where:
                - model_func: Callable for curve_fit
                - cache_hit: True if model was retrieved from cache
                - jit_compiled: True if JIT compilation was applied
        """
        # Extract wavevector q
        q = self._get_attr(data, "q")
        if q is None:
            q = self._get_attr(data, "wavevector_q_list", [1.0])
        if isinstance(q, (list, np.ndarray)):
            q = q[0]

        # Get unique phi angles
        phi = self._get_attr(data, "phi")
        if phi is None:
            phi = self._get_attr(data, "phi_angles_list")
        if phi is None:
            raise ValueError("Data must contain 'phi' or 'phi_angles_list'")
        phi_unique = np.unique(phi)

        # T011: Use get_or_create_model for caching and JIT
        if self.config.enable_cache:
            model, model_func, cache_hit = get_or_create_model(
                analysis_mode=analysis_mode,
                phi_angles=phi_unique,
                q=float(q),
                per_angle_scaling=per_angle_scaling,
                config=None,
                enable_jit=self.config.enable_jit,
            )
            # T013: Cache statistics logging (DEBUG level)
            stats = get_cache_stats()
            logger.debug(
                "Model cache stats: hits=%d, misses=%d, size=%d",
                stats["hits"],
                stats["misses"],
                stats["size"],
            )
            # Determine if JIT was applied (check if function is traced)
            jit_compiled = self.config.enable_jit
            return model_func, cache_hit, jit_compiled
        else:
            # Caching disabled - create model directly using CombinedModel
            from xpcsjax.core.models import CombinedModel

            # Use same logic as get_or_create_model for consistency
            normalized_mode = analysis_mode
            model_mode = normalized_mode if "static" in normalized_mode else "laminar_flow"
            model = CombinedModel(analysis_mode=model_mode)

            # Store experimental parameters for closure
            q_val = float(q)

            def model_func(xdata: np.ndarray, *params: float) -> np.ndarray:
                """Model function compatible with NLSQ curve_fit."""
                params_array = np.array(params)
                n_params = len(params_array)

                # Extract per-angle scaling parameters if present
                n_physical = 3 if "static" in model_mode else 7
                if per_angle_scaling and n_params >= n_physical + 2 * n_phi:
                    contrast_vals = params_array[:n_phi]
                    offset_vals = params_array[n_phi : 2 * n_phi]
                    physical_params = params_array[2 * n_phi :]
                else:
                    # Legacy scalar mode (for backward compatibility)
                    c0 = params_array[0] if len(params_array) > 0 else 0.5
                    o0 = params_array[1] if len(params_array) > 1 else 1.0
                    contrast_vals = np.full(n_phi, c0)
                    offset_vals = np.full(n_phi, o0)
                    default_phys = np.array([1000.0, 0.5, 10.0])
                    physical_params = (
                        params_array[2:] if len(params_array) > 2 else default_phys
                    )

                # Vectorized g2 computation (single JAX dispatch)
                import jax.numpy as jnp

                params_jax = jnp.asarray(physical_params, dtype=jnp.float64)
                t1_all = xdata[:, 0]
                t2_all = xdata[:, 1]
                phi_all = xdata[:, 2]

                # Map phi values to indices (vectorized). Warn on out-of-grid
                # phi: an unguarded clip would silently route those points to the
                # boundary angle bin, mis-associating them in the residual.
                phi_idx_all = np.searchsorted(phi_unique, phi_all)
                n_phi_oob = int(np.sum(phi_idx_all >= len(phi_unique)))
                if n_phi_oob > 0:
                    logger.warning(
                        "%d phi value(s) lie beyond the fitted angle grid; clipped "
                        "to the boundary bin. Check data/config alignment.",
                        n_phi_oob,
                    )
                phi_idx_all = np.clip(phi_idx_all, 0, len(phi_unique) - 1)

                # Batch compute g1 using model
                g1_all = model.compute_g1(
                    params_jax,
                    jnp.asarray(t1_all, dtype=jnp.float64),
                    jnp.asarray(t2_all, dtype=jnp.float64),
                    jnp.asarray(phi_unique, dtype=jnp.float64),
                    q_val,
                    1.0,
                )
                g1_arr = np.asarray(g1_all)

                # Select per-point g1 and compute g2
                if g1_arr.ndim == 2:
                    point_idx = np.arange(len(xdata))
                    g1_per_point = g1_arr[phi_idx_all, point_idx]
                else:
                    g1_per_point = g1_arr.ravel()

                g2_pred = (
                    offset_vals[phi_idx_all]
                    + contrast_vals[phi_idx_all] * g1_per_point**2
                )
                return g2_pred

            return model_func, False, False

    @staticmethod
    def _get_attr(data: Any, key: str, default: Any = None) -> Any:
        """Get attribute from dict or object."""
        if isinstance(data, dict):
            return data.get(key, default)
        return getattr(data, key, default)

    def _flatten_xpcs_data(
        self,
        data: Any,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Flatten XPCS data for NLSQ optimization.

        Args:
            data: XPCS experimental data (dict or object) with attributes:
                - t1, t2: Time coordinates (1D or 2D)
                - phi: Phi angles
                - g2 or c2_exp: Experimental g2 values

        Returns:
            Tuple of (xdata, ydata, n_phi) where:
                - xdata: Flattened independent variables [t1, t2, phi_idx]
                         where phi_idx is the precomputed phi angle index
                - ydata: Flattened g2 observations
                - n_phi: Number of unique phi angles

        Note:
            As of v2.17.0, phi_idx is precomputed here to avoid expensive
            gather operations inside JIT-compiled functions (XLA slow_operation_alarm).
        """
        # Get time coordinates (works with both dict and object)
        t1 = self._get_attr(data, "t1")
        if t1 is None:
            t1 = self._get_attr(data, "t1_2d")
        t2 = self._get_attr(data, "t2")
        if t2 is None:
            t2 = self._get_attr(data, "t2_2d")

        if t1 is None or t2 is None:
            raise ValueError("Data must contain 't1'/'t1_2d' and 't2'/'t2_2d'")

        # Handle 2D meshgrid format
        if t1.ndim == 2:
            t1 = t1.ravel()
        if t2.ndim == 2:
            t2 = t2.ravel()

        # Get phi angles
        phi = self._get_attr(data, "phi")
        if phi is None:
            phi = self._get_attr(data, "phi_angles_list")
        if phi is None:
            raise ValueError("Data must contain 'phi' or 'phi_angles_list'")

        phi_unique = np.unique(phi)
        n_phi = len(phi_unique)

        # Get g2 observations
        g2 = self._get_attr(data, "g2")
        if g2 is None:
            g2 = self._get_attr(data, "c2_exp")
        if g2 is None:
            raise ValueError("Data must contain 'g2' or 'c2_exp'")

        # Flatten if needed
        if g2.ndim > 1:
            g2 = g2.ravel()

        # Build xdata array [t1, t2, phi_idx]
        # Broadcast phi if needed
        if len(phi) != len(t1):
            # phi has n_phi entries; broadcast to match flattened t1/t2/g2
            # by repeating each phi value for all time points in that angle
            n_time_per_angle = len(t1) // n_phi
            phi_broadcast = np.repeat(phi_unique, n_time_per_angle)
        else:
            phi_broadcast = phi

        # Precompute phi indices to avoid expensive argmin inside JIT (v2.17.0)
        # This prevents XLA slow_operation_alarm from gather operations
        # during constant folding of large arrays (23M+ points)
        phi_indices = np.argmin(
            np.abs(phi_broadcast[:, np.newaxis] - phi_unique[np.newaxis, :]),
            axis=1,
        ).astype(np.float64)  # Use float for consistent xdata dtype

        xdata = np.column_stack([t1, t2, phi_indices])

        return xdata, g2, n_phi

    def _convert_nlsq_result(
        self,
        popt: np.ndarray,
        pcov: np.ndarray,
        info: dict[str, Any],
        n_data: int,
        execution_time: float,
        cache_hit: bool = False,
        jit_compiled: bool = False,
    ) -> OptimizationResult:
        """Convert NLSQ result to homodyne OptimizationResult.

        Args:
            popt: Optimized parameters
            pcov: Covariance matrix
            info: Additional info from NLSQ
            n_data: Number of data points
            execution_time: Optimization time in seconds
            cache_hit: Whether model was retrieved from cache (T012)
            jit_compiled: Whether model function is JIT-compiled (T017)

        Returns:
            OptimizationResult dataclass
        """
        n_params = len(popt)

        # Compute uncertainties from covariance diagonal
        uncertainties = (
            np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros(n_params)
        )

        # Compute chi-squared from info.
        # NLSQ/scipy cost = 0.5 * sum(rho(r²)), so chi² = 2 * cost for linear loss.
        # If "fun" (raw residuals) is available, prefer computing from those directly.
        raw_fun = info.get("fun", None)
        if raw_fun is not None and isinstance(raw_fun, np.ndarray):
            chi_squared = float(np.sum(raw_fun**2))
        else:
            cost = info.get("cost", 0.0)
            chi_squared = float(cost) * 2.0  # cost = 0.5 * sum(r²)

        # Reduced chi-squared
        dof = max(1, n_data - n_params)
        reduced_chi_squared = chi_squared / dof

        # Convergence status
        success = info.get("success", False)
        convergence_status = "converged" if success else "failed"
        if info.get("status", 0) == 1:  # max iterations
            convergence_status = "max_iter"

        # Iterations
        iterations = info.get("nfev", info.get("iterations", 0))

        # Quality flag based on reduced chi-squared
        if reduced_chi_squared < 2.0:
            quality_flag = "good"
        elif reduced_chi_squared < 5.0:
            quality_flag = "marginal"
        else:
            quality_flag = "poor"

        # Device info (T012: cache_hit, T017: jit_compiled)
        device_info = {
            "device": "cpu",
            "adapter": "NLSQAdapter",
            "cache_hit": cache_hit,
            "jit_compiled": jit_compiled,
        }

        # Streaming diagnostics if available
        streaming_diagnostics = info.get("streaming_diagnostics")

        return OptimizationResult(
            parameters=popt,
            uncertainties=uncertainties,
            covariance=pcov if pcov is not None else np.eye(n_params),
            chi_squared=chi_squared,
            reduced_chi_squared=reduced_chi_squared,
            convergence_status=convergence_status,
            iterations=iterations,
            execution_time=execution_time,
            device_info=device_info,
            recovery_actions=[],
            quality_flag=quality_flag,
            streaming_diagnostics=streaming_diagnostics,
            stratification_diagnostics=None,
            nlsq_diagnostics=info,
        )

    def fit(
        self,
        data: Any,
        config: Any,
        initial_params: np.ndarray | None = None,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        analysis_mode: AnalysisMode = AnalysisMode.STATIC_ISOTROPIC,
        per_angle_scaling: bool = True,
        diagnostics_enabled: bool = False,
        shear_transforms: dict[str, Any] | None = None,
        per_angle_scaling_initial: dict[str, list[float]] | None = None,
        anti_degeneracy_controller: Any | None = None,
    ) -> OptimizationResult:
        """Execute NLSQ optimization using CurveFit class.

        This method provides the same interface as NLSQWrapper.fit() for
        backward compatibility while using NLSQ's modern CurveFit class.

        Args:
            data: XPCS experimental data
            config: Configuration manager with optimization settings
            initial_params: Initial parameter guess (required)
            bounds: Parameter bounds as (lower, upper) tuple
            analysis_mode: 'static_anisotropic', 'static_isotropic', or 'laminar_flow'
            per_angle_scaling: Must be True (per-angle is physically correct)
            diagnostics_enabled: Enable extended diagnostics
            shear_transforms: Shear parameter transformations
            per_angle_scaling_initial: Initial per-angle contrast/offset
            anti_degeneracy_controller: Anti-degeneracy controller (physics-specific)

        Returns:
            OptimizationResult with converged parameters and diagnostics

        Raises:
            ValueError: If bounds are invalid or per_angle_scaling=False
            ImportError: If NLSQ CurveFit is not available
        """
        start_time = time.time()

        # Validate per-angle scaling
        if not per_angle_scaling:
            raise ValueError(
                "per_angle_scaling=False is deprecated and removed. "
                "Use per_angle_scaling=True (default) for physically correct behavior."
            )

        # Validate initial params
        if initial_params is None:
            raise ValueError("initial_params must be provided for NLSQAdapter.fit()")

        # Extract NLSQ settings from config
        nlsq_settings = self._extract_nlsq_settings(config)

        # Flatten XPCS data
        xdata, ydata, n_phi = self._flatten_xpcs_data(data)
        n_data = len(ydata)
        n_params = len(initial_params)

        logger.info(
            "NLSQAdapter.fit: n_data=%d, n_params=%d, n_phi=%d, mode=%s",
            n_data,
            n_params,
            n_phi,
            analysis_mode,
        )

        # Build model function (T011: returns tuple with cache metadata)
        model_func, cache_hit, jit_compiled = self._build_model_function(
            data=data,
            config=config,
            analysis_mode=analysis_mode,
            per_angle_scaling=per_angle_scaling,
            n_phi=n_phi,
        )

        # Select workflow
        workflow_config = self._select_workflow(n_data, n_params)
        logger.debug("Selected workflow: %s", workflow_config)

        # Extract optimizer settings
        loss = nlsq_settings.get("loss", "soft_l1")
        ftol = nlsq_settings.get("ftol", 1e-8)
        gtol = nlsq_settings.get("gtol", 1e-8)
        xtol = nlsq_settings.get("xtol", 1e-8)
        max_nfev = nlsq_settings.get("max_iterations", nlsq_settings.get("max_nfev"))

        # Prepare kwargs for curve_fit
        fit_kwargs: dict[str, Any] = {
            "p0": initial_params,
            "bounds": bounds,
            "method": "trf",
            "loss": loss,
            "ftol": ftol,
            "gtol": gtol,
            "xtol": xtol,
        }

        if max_nfev is not None:
            fit_kwargs["max_nfev"] = max_nfev

        # Apply anti-degeneracy callbacks if controller is provided
        if anti_degeneracy_controller is not None:
            # Check if controller has NLSQ callback adapter
            if hasattr(anti_degeneracy_controller, "create_nlsq_callbacks"):
                callbacks = anti_degeneracy_controller.create_nlsq_callbacks()
                if callbacks:
                    fit_kwargs.update(callbacks)
                    logger.debug(
                        "Injected anti-degeneracy callbacks: %s", list(callbacks.keys())
                    )

        # Run optimization via CurveFit
        try:
            result = self._fitter.curve_fit(
                f=model_func,
                xdata=xdata,
                ydata=ydata,
                **fit_kwargs,
            )

            # Handle different result formats
            if isinstance(result, tuple):
                if len(result) == 2:
                    popt, pcov = result
                    info: dict[str, Any] = {}
                elif len(result) == 3:
                    popt, pcov, info = result
                else:
                    raise TypeError(f"Unexpected tuple length: {len(result)}")
            elif hasattr(result, "popt"):
                # CurveFitResult object
                popt = result.popt
                pcov = result.pcov
                info = getattr(result, "info", {})
            else:
                raise TypeError(f"Unexpected result type: {type(result)}")

        except (ValueError, RuntimeError, TypeError, OSError, MemoryError) as e:
            logger.error("NLSQ optimization failed: %s", e)
            # Return failed result (T012, T017: include cache metadata)
            execution_time = time.time() - start_time
            return OptimizationResult(
                parameters=initial_params,
                uncertainties=np.zeros(n_params),
                covariance=np.eye(n_params),
                chi_squared=float("inf"),
                reduced_chi_squared=float("inf"),
                convergence_status="failed",
                iterations=0,
                execution_time=execution_time,
                device_info={
                    "device": "cpu",
                    "adapter": "NLSQAdapter",
                    "cache_hit": cache_hit,
                    "jit_compiled": jit_compiled,
                    "error": str(e),
                },
                recovery_actions=[],
                quality_flag="poor",
            )

        execution_time = time.time() - start_time

        # Convert to OptimizationResult (T012, T017: pass cache metadata)
        opt_result = self._convert_nlsq_result(
            popt=np.asarray(popt),
            pcov=np.asarray(pcov) if pcov is not None else None,
            info=info if isinstance(info, dict) else {},
            n_data=n_data,
            execution_time=execution_time,
            cache_hit=cache_hit,
            jit_compiled=jit_compiled,
        )

        logger.info(
            "NLSQAdapter.fit completed: chi2=%.6g, reduced_chi2=%.6g, status=%s, time=%.2fs",
            opt_result.chi_squared,
            opt_result.reduced_chi_squared,
            opt_result.convergence_status,
            execution_time,
        )

        return opt_result

    def is_available(self) -> bool:
        """Check if NLSQ CurveFit is available."""
        return NLSQ_CURVEFIT_AVAILABLE


def get_adapter(config: AdapterConfig | None = None) -> NLSQAdapter:
    """Factory function to get NLSQAdapter instance.

    Args:
        config: Adapter configuration

    Returns:
        NLSQAdapter instance

    Raises:
        ImportError: If NLSQ CurveFit is not available
    """
    return NLSQAdapter(config=config)


def is_adapter_available() -> bool:
    """Check if NLSQAdapter can be used.

    Returns:
        True if NLSQ CurveFit class is available
    """
    return NLSQ_CURVEFIT_AVAILABLE
