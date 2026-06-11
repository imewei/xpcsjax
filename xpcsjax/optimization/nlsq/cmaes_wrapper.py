"""CMA-ES global optimization wrapper for homodyne.

Provides CMA-ES integration using NLSQ's CMAESOptimizer with:
- Automatic memory configuration for large datasets
- BIPOP restart strategy for robust convergence
- Scale-ratio based method selection
- Integration with homodyne's model caching

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is particularly
beneficial for XPCS laminar_flow mode where parameters have vastly different
scales (e.g., D₀ ~ 1e4 vs γ̇₀ ~ 1e-3, scale ratio > 1e7).

NLSQ Features:
- evosax backend for JAX-accelerated evolution
- BIPOP restart strategy (alternating large/small populations)
- Memory batching: population_batch_size, data_chunk_size
- MethodSelector for auto-selection based on scale ratio

Usage
-----
>>> from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESWrapper
>>> wrapper = CMAESWrapper()
>>> if wrapper.should_use_cmaes(bounds):
...     result = wrapper.fit(model_func, xdata, ydata, p0, bounds)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from xpcsjax.utils.logging import get_logger, log_exception, log_phase

if TYPE_CHECKING:
    from xpcsjax.optimization.nlsq.config import NLSQConfig

logger = get_logger(__name__)


def _format_bounds_summary(bounds: tuple[np.ndarray, np.ndarray]) -> str:
    """Format bounds summary for logging.

    Parameters
    ----------
    bounds : tuple[np.ndarray, np.ndarray]
        Lower and upper bounds as (lower, upper) arrays.

    Returns
    -------
    str
        Human-readable bounds summary.
    """
    lower, upper = bounds
    ranges = upper - lower
    n_params = len(lower)

    # Find min/max ranges for summary
    valid_ranges = ranges[ranges > 0]
    if len(valid_ranges) == 0:
        return f"{n_params} params (no valid ranges)"

    min_range = np.min(valid_ranges)
    max_range = np.max(valid_ranges)

    return f"{n_params} params, range=[{min_range:.2e}, {max_range:.2e}]"


# =============================================================================
# Parameter Normalization Utilities
# =============================================================================
# These functions implement bounds-based normalization to improve CMA-ES
# convergence for multi-scale problems (e.g., D₀ ~ 1e4 vs γ̇₀ ~ 1e-3).


def _compute_normalization_factors(
    bounds: tuple[np.ndarray, np.ndarray],
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute normalization factors for bounds-based normalization.

    Normalization: x_norm = (x - lower) / (upper - lower)

    Parameters
    ----------
    bounds : tuple[np.ndarray, np.ndarray]
        Lower and upper bounds as (lower, upper) arrays.
    epsilon : float
        Small value to prevent division by zero for fixed parameters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (scale, offset, range) where:
        - scale = 1 / (upper - lower + epsilon) for safe division
        - offset = lower (subtracted before scaling)
        - range = upper - lower (for covariance adjustment)
    """
    lower, upper = bounds
    param_range = upper - lower

    # Prevent division by zero for fixed parameters (where upper == lower)
    safe_range = np.where(param_range > epsilon, param_range, 1.0)
    scale = 1.0 / safe_range

    return scale, lower, param_range


def _normalize_params(
    params: np.ndarray,
    scale: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """Normalize parameters from physical space to [0, 1] space.

    Parameters
    ----------
    params : np.ndarray
        Parameters in physical space.
    scale : np.ndarray
        Scale factors (1 / range).
    offset : np.ndarray
        Offset values (lower bounds).

    Returns
    -------
    np.ndarray
        Parameters normalized to [0, 1] space.
    """
    return (params - offset) * scale


def _denormalize_params(
    params_norm: np.ndarray,
    scale: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """Denormalize parameters from [0, 1] space back to physical space.

    Parameters
    ----------
    params_norm : np.ndarray
        Parameters in normalized [0, 1] space.
    scale : np.ndarray
        Scale factors (1 / range).
    offset : np.ndarray
        Offset values (lower bounds).

    Returns
    -------
    np.ndarray
        Parameters in physical space.
    """
    return params_norm / scale + offset


def _normalize_bounds(
    bounds: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize bounds to [0, 1] space.

    Parameters
    ----------
    bounds : tuple[np.ndarray, np.ndarray]
        Physical bounds as (lower, upper) arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Normalized bounds (zeros, ones).
    """
    n_params = len(bounds[0])
    return (np.zeros(n_params), np.ones(n_params))


def _adjust_covariance_for_normalization(
    covariance: np.ndarray | None,
    param_range: np.ndarray,
) -> np.ndarray | None:
    """Transform covariance from normalized to physical space.

    Uses Jacobian scaling: Cov_phys = J @ Cov_norm @ J.T
    where J is diagonal with elements = param_range.

    Parameters
    ----------
    covariance : np.ndarray | None
        Covariance matrix in normalized space.
    param_range : np.ndarray
        Parameter ranges (upper - lower bounds).

    Returns
    -------
    np.ndarray | None
        Covariance matrix in physical space, or None if input is None.
    """
    if covariance is None:
        return None

    # Jacobian is diagonal with elements = param_range
    # Cov_phys[i,j] = range_i * Cov_norm[i,j] * range_j
    jacobian_outer = np.outer(param_range, param_range)
    return covariance * jacobian_outer


def _is_cmaes_available() -> bool:
    """Check if CMA-ES is available via NLSQ."""
    try:
        from nlsq.global_optimization import is_evosax_available

        return is_evosax_available()
    except ImportError:
        return False


CMAES_AVAILABLE = _is_cmaes_available()

# Skip L-M refinement when CMA-ES chi2 exceeds this multiple of the
# warm-start chi2 — the comparison in core.py will discard it anyway.
# 10× chosen empirically: large enough that L-M local refinement cannot
# recover the gap (trust-region step is bounded), small enough to catch
# near-degenerate CMA-ES runs before they waste L-M budget.
_REFINEMENT_SKIP_CHI2_RATIO = 10.0


@dataclass
class CMAESWrapperConfig:
    """Configuration for CMA-ES wrapper.

    Attributes
    ----------
    preset : str
        CMA-ES preset: "cmaes-fast" (50 gen), "cmaes" (100 gen), "cmaes-global" (200 gen).
    max_generations : int | None
        Maximum CMA-ES generations. None = use preset default + adaptive scaling.
    sigma : float
        Initial step size as fraction of search range (0, 1].
    tol_fun : float
        Function value tolerance for convergence.
    tol_x : float
        Parameter tolerance for convergence.
    restart_strategy : str
        Restart strategy: "none" or "bipop".
    max_restarts : int
        Maximum number of BIPOP restarts.
    population_batch_size : int | None
        Batch size for population evaluation (None = auto).
    data_chunk_size : int | None
        Chunk size for data streaming (None = auto).
    refine_with_nlsq : bool
        Whether to refine CMA-ES solution with NLSQ TRF.
    auto_memory : bool
        Whether to auto-configure memory parameters.
    memory_limit_gb : float
        Memory limit for auto-configuration in GB.
    refinement_workflow : str
        NLSQ workflow for refinement: "auto" (recommended), "standard", "streaming".
    refinement_ftol : float
        Function tolerance for NLSQ refinement.
    refinement_xtol : float
        Parameter tolerance for NLSQ refinement.
    refinement_gtol : float
        Gradient tolerance for NLSQ refinement.
    refinement_max_nfev : int
        Maximum function evaluations for NLSQ refinement.
    refinement_loss : str
        Loss function for NLSQ refinement: "linear", "soft_l1", "huber", etc.
    """

    # CMA-ES global search settings
    preset: str = "cmaes"
    max_generations: int | None = None  # None = use preset + adaptive scaling
    popsize: int | None = None  # None = auto from 4+3*ln(n)
    seed: int | None = None  # Deterministic RNG seed for CMA-ES (None = NLSQ default)
    sigma: float = 0.5
    sigma_warmstart: float = 0.05  # Reduced sigma for warm-start (local refinement)
    tol_fun: float = 1e-8
    tol_x: float = 1e-8
    restart_strategy: str = "bipop"
    max_restarts: int = 9
    population_batch_size: int | None = None
    data_chunk_size: int | None = None
    auto_memory: bool = True
    memory_limit_gb: float = 8.0

    # NLSQ TRF refinement settings (post-CMA-ES)
    refine_with_nlsq: bool = True
    refinement_workflow: str = "auto"  # "auto" auto-selects memory strategy
    refinement_ftol: float = 1e-10  # Tighter than CMA-ES for local refinement
    refinement_xtol: float = 1e-10
    refinement_gtol: float = 1e-10
    refinement_max_nfev: int = 500  # Refinement shouldn't need many iterations
    refinement_loss: str = "linear"  # Linear loss for final refinement

    # Parameter normalization
    # Normalizes parameters to [0,1] based on bounds for better scale handling
    normalize: bool = True  # Enable bounds-based normalization
    normalization_epsilon: float = 1e-12  # Prevent division by zero

    @classmethod
    def from_nlsq_config(cls, config: NLSQConfig) -> CMAESWrapperConfig:
        """Create CMAESWrapperConfig from NLSQConfig.

        Parameters
        ----------
        config : NLSQConfig
            NLSQ configuration object.

        Returns
        -------
        CMAESWrapperConfig
            CMA-ES wrapper configuration.
        """
        # NLSQConfig might not have all fields if it's an older version or partial
        # Use getattr with defaults where appropriate or access directly if we are sure
        _pop_batch = getattr(config, "cmaes_population_batch_size", None)
        _data_chunk = getattr(config, "cmaes_data_chunk_size", None)
        return cls(
            # CMA-ES global search settings
            preset=getattr(config, "cmaes_preset", "cmaes"),
            max_generations=getattr(config, "cmaes_max_generations", None),
            popsize=getattr(config, "cmaes_popsize", None),
            seed=getattr(config, "cmaes_seed", None),
            sigma=getattr(config, "cmaes_sigma", 0.5),
            sigma_warmstart=getattr(config, "cmaes_sigma_warmstart", 0.05),
            tol_fun=getattr(config, "cmaes_tol_fun", 1e-8),
            tol_x=getattr(config, "cmaes_tol_x", 1e-8),
            restart_strategy=getattr(config, "cmaes_restart_strategy", "bipop"),
            max_restarts=getattr(config, "cmaes_max_restarts", 9),
            population_batch_size=_pop_batch,
            data_chunk_size=_data_chunk,
            auto_memory=_pop_batch is None and _data_chunk is None,
            memory_limit_gb=getattr(config, "cmaes_memory_limit_gb", 8.0),
            # NLSQ TRF refinement settings
            refine_with_nlsq=getattr(config, "cmaes_refine_with_nlsq", True),
            refinement_workflow=getattr(config, "cmaes_refinement_workflow", "auto"),
            refinement_ftol=getattr(config, "cmaes_refinement_ftol", 1e-10),
            refinement_xtol=getattr(config, "cmaes_refinement_xtol", 1e-10),
            refinement_gtol=getattr(config, "cmaes_refinement_gtol", 1e-10),
            refinement_max_nfev=getattr(config, "cmaes_refinement_max_nfev", 500),
            refinement_loss=getattr(config, "cmaes_refinement_loss", "linear"),
            # Parameter normalization
            normalize=getattr(config, "cmaes_normalize", True),
            normalization_epsilon=getattr(config, "cmaes_normalization_epsilon", 1e-12),
        )

    def to_cmaes_config(self, n_params: int, *, sigma_override: float | None = None) -> Any:
        """Convert to NLSQ CMAESConfig.

        Parameters
        ----------
        n_params : int
            Number of parameters for popsize calculation.
        sigma_override : float or None
            If provided, override the default sigma value. Used to apply
            warm-start sigma when NLSQ warm-start is active.

        Returns
        -------
        CMAESConfig
            NLSQ CMAESConfig object.

        Raises
        ------
        ImportError
            If NLSQ CMA-ES is not available.
        """
        if not CMAES_AVAILABLE:
            raise ImportError(
                "CMA-ES requires NLSQ with evosax backend. "
                "Install with: pip install nlsq[evosax]"
            )

        from nlsq.global_optimization import CMAESConfig, compute_default_popsize

        # Use configured popsize if specified, otherwise compute default
        if self.popsize is not None:
            popsize = self.popsize
        else:
            popsize = compute_default_popsize(n_params)

        # Map preset to max_generations if using preset
        preset_generations = {
            "cmaes-fast": 50,
            "cmaes": 100,
            "cmaes-global": 200,
        }
        # An explicitly configured max_generations overrides the preset default
        # (config semantics: None = use preset + adaptive scaling). Putting
        # self.max_generations in the .get() default slot would only consult it
        # when the preset key is absent, silently ignoring it for every valid
        # preset.
        if self.max_generations is not None:
            max_gen = self.max_generations
        else:
            max_gen = preset_generations.get(self.preset, 100)

        effective_sigma = sigma_override if sigma_override is not None else self.sigma

        # ``restart_strategy`` is Literal['none','bipop'] in NLSQ's API but
        # comes from a free-text config field. The validator in this class's
        # __post_init__ checks the value against {'none','bipop'} so the
        # cast is safe; mypy can't see the runtime guard.
        from typing import Literal, cast

        restart_strategy_literal = cast(Literal["none", "bipop"], self.restart_strategy)
        # Forward the deterministic seed only when set. NLSQ's CMAESConfig
        # defaults ``seed=None`` (non-deterministic); pinning it makes the
        # CMA-ES search reproducible (used by the joint global escape).
        cmaes_kwargs: dict[str, Any] = {}
        if self.seed is not None:
            cmaes_kwargs["seed"] = self.seed

        return CMAESConfig(
            popsize=popsize,
            max_generations=max_gen,
            sigma=effective_sigma,
            tol_fun=self.tol_fun,
            tol_x=self.tol_x,
            restart_strategy=restart_strategy_literal,
            max_restarts=self.max_restarts,
            population_batch_size=self.population_batch_size,
            data_chunk_size=self.data_chunk_size,
            # Disable NLSQ's internal refinement - we do it explicitly in homodyne
            refine_with_nlsq=False,
            **cmaes_kwargs,
        )


@dataclass
class CMAESResult:
    """Result from CMA-ES optimization.

    Attributes
    ----------
    parameters : np.ndarray
        Optimized parameter values.
    covariance : np.ndarray | None
        Parameter covariance matrix (if computed).
    chi_squared : float
        Final chi-squared value.
    success : bool
        Whether optimization converged successfully.
    diagnostics : dict
        CMA-ES diagnostics (generations, evaluations, etc.).
    method_used : str
        Method used: "cmaes" or "multi-start".
    nlsq_refined : bool
        Whether result was refined with NLSQ L-M.
    message : str
        Convergence message.
    """

    parameters: np.ndarray
    covariance: np.ndarray | None
    chi_squared: float
    success: bool
    diagnostics: dict = field(default_factory=dict)
    method_used: str = "cmaes"
    nlsq_refined: bool = False
    message: str = ""


class CMAESWrapper:
    """Wrapper around NLSQ's CMAESOptimizer for homodyne integration.

    This wrapper provides:
    - Scale-ratio based method selection (CMA-ES vs multi-start)
    - Automatic memory configuration for large datasets
    - BIPOP restart strategy for robust global optimization
    - Optional L-M refinement of CMA-ES solutions

    Parameters
    ----------
    config : CMAESWrapperConfig | None
        Configuration for CMA-ES wrapper. If None, uses defaults.

    Examples
    --------
    >>> wrapper = CMAESWrapper()
    >>> if wrapper.should_use_cmaes(bounds, scale_threshold=1000):
    ...     result = wrapper.fit(model_func, xdata, ydata, p0, bounds)
    """

    def __init__(self, config: CMAESWrapperConfig | None = None) -> None:
        """Initialize CMA-ES wrapper.

        Parameters
        ----------
        config : CMAESWrapperConfig | None
            Configuration for wrapper. Uses defaults if None.
        """
        self.config = config or CMAESWrapperConfig()
        self._optimizer = None
        self._restarter = None

    @property
    def is_available(self) -> bool:
        """Check if CMA-ES is available."""
        return CMAES_AVAILABLE

    def compute_scale_ratio(
        self,
        bounds: tuple[np.ndarray, np.ndarray],
    ) -> float:
        """Compute scale ratio from parameter bounds.

        The scale ratio is the ratio of the largest to smallest parameter
        range. High scale ratios (> 1000) indicate multi-scale problems
        where CMA-ES excels.

        Parameters
        ----------
        bounds : tuple[np.ndarray, np.ndarray]
            Lower and upper bounds as (lower, upper) arrays.

        Returns
        -------
        float
            Scale ratio (max_range / min_range).

        Examples
        --------
        >>> lower = np.array([0, 0.001, 100])
        >>> upper = np.array([1, 0.01, 10000])
        >>> wrapper.compute_scale_ratio((lower, upper))
        11000.0  # (10000-100) / (0.01-0.001)
        """
        lower, upper = bounds
        ranges = upper - lower

        # Avoid division by zero
        valid_ranges = ranges[ranges > 0]
        if len(valid_ranges) < 2:
            return 1.0

        min_range = np.min(valid_ranges)
        max_range = np.max(valid_ranges)

        return max_range / min_range if min_range > 0 else 1.0

    def should_use_cmaes(
        self,
        bounds: tuple[np.ndarray, np.ndarray],
        scale_threshold: float = 1000.0,  # XPCS: Γ (~1e-6–1e0 s) vs β (~0.1–1.0) → scale ratios routinely exceed 1e3
    ) -> bool:
        """Determine if CMA-ES should be used based on scale ratio.

        CMA-ES adapts its covariance matrix to different parameter scales,
        making it ideal for multi-scale optimization problems. This method
        checks if the scale ratio exceeds the threshold.

        Parameters
        ----------
        bounds : tuple[np.ndarray, np.ndarray]
            Parameter bounds as (lower, upper) arrays.
        scale_threshold : float
            Scale ratio threshold for CMA-ES selection. Default: 1000.

        Returns
        -------
        bool
            True if CMA-ES should be used.

        Notes
        -----
        XPCS laminar_flow mode typically has scale ratios > 1e7:
        - D₀ ~ 1e4 (diffusion coefficient)
        - γ̇₀ ~ 1e-3 (shear rate)
        """
        if not self.is_available:
            logger.info(
                "[CMA-ES] Method unavailable: evosax not installed. "
                "Install with: pip install nlsq[evosax]"
            )
            return False

        scale_ratio = self.compute_scale_ratio(bounds)
        should_use = scale_ratio >= scale_threshold
        bounds_summary = _format_bounds_summary(bounds)

        if should_use:
            # Log at INFO when CMA-ES is selected - this is an important decision
            logger.info(
                f"[CMA-ES] Auto-selected: scale_ratio={scale_ratio:.2e} "
                f"(threshold={scale_threshold:.2e}), {bounds_summary}"
            )
        else:
            # Log at DEBUG when not selected - less important
            logger.debug(
                f"[CMA-ES] Not selected: scale_ratio={scale_ratio:.2e} "
                f"< threshold={scale_threshold:.2e}"
            )

        return should_use

    def _run_nlsq_refinement(
        self,
        model_func: Callable,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        sigma: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run NLSQ TRF refinement on CMA-ES solution.

        Uses NLSQ's curve_fit with workflow="auto" for memory-aware strategy
        selection, similar to NLSQ's "auto_global" workflow.

        Parameters
        ----------
        model_func : Callable
            Model function: ``y = f(x, *params)``.
        xdata : np.ndarray
            Independent variable data.
        ydata : np.ndarray
            Dependent variable data.
        p0 : np.ndarray
            Initial parameters (CMA-ES solution).
        bounds : tuple[np.ndarray, np.ndarray]
            Parameter bounds as (lower, upper).
        sigma : np.ndarray | None
            Data uncertainties (optional).

        Returns
        -------
        dict[str, Any]
            Refinement result with keys: popt, pcov, infodict, mesg, ier.
        """
        from nlsq import curve_fit

        n_data = len(ydata)
        logger.info(
            f"[CMA-ES] Refinement starting: workflow={self.config.refinement_workflow}, "
            f"n_data={n_data:,}, ftol={self.config.refinement_ftol:.1e}, "
            f"max_nfev={self.config.refinement_max_nfev}"
        )

        try:
            with log_phase("CMA-ES refinement", logger, track_memory=True) as phase:
                # Build refinement kwargs for NLSQ curve_fit
                # Note: NLSQ curve_fit uses 'workflow' instead of full scipy interface
                refinement_kwargs: dict[str, Any] = {
                    "ftol": self.config.refinement_ftol,
                    "xtol": self.config.refinement_xtol,
                    "gtol": self.config.refinement_gtol,
                    "max_nfev": self.config.refinement_max_nfev,
                    "loss": self.config.refinement_loss,
                }

                # Run NLSQ curve_fit with workflow="auto" for memory-aware selection
                # NLSQ returns (popt, pcov), not scipy's 5-tuple with full_output
                popt, pcov = curve_fit(
                    f=model_func,
                    xdata=xdata,
                    ydata=ydata,
                    p0=p0,
                    sigma=sigma,
                    bounds=bounds,
                    workflow=self.config.refinement_workflow,
                    tr_solver="exact",  # Model uses closure data, not xdata
                    **refinement_kwargs,
                )

            # Compute refined chi-squared
            pred = model_func(xdata, *popt)
            residuals = ydata - pred
            if sigma is not None:
                residuals = residuals / sigma
            chi_squared = float(np.sum(residuals**2))

            logger.info(
                f"[CMA-ES] Refinement completed: chi2={chi_squared:.4e}, time={phase.duration:.1f}s"
            )

            return {
                "popt": np.asarray(popt),
                "pcov": np.asarray(pcov) if pcov is not None else None,
                "chi_squared": chi_squared,
                "infodict": {},  # NLSQ doesn't return infodict
                "mesg": "NLSQ TRF refinement converged",
                "ier": 1,  # Success
                "success": True,
                "duration_s": phase.duration,
            }

        except (ValueError, RuntimeError, TypeError, OSError, MemoryError) as e:
            log_exception(
                logger,
                e,
                context={
                    "phase": "NLSQ refinement",
                    "workflow": self.config.refinement_workflow,
                    "n_data": n_data,
                },
                level=30,  # WARNING
                include_traceback=False,  # Keep it concise
            )
            # Return original parameters on refinement failure
            return {
                "popt": p0,
                "pcov": None,
                "chi_squared": None,
                "infodict": {},
                "mesg": f"Refinement failed: {e}",
                "ier": -1,
                "success": False,
            }

    def _configure_memory(
        self,
        n_data: int,
        n_params: int,
    ) -> tuple[int | None, int | None]:
        """Auto-configure memory parameters for large datasets.

        Parameters
        ----------
        n_data : int
            Number of data points.
        n_params : int
            Number of parameters.

        Returns
        -------
        tuple[int | None, int | None]
            (population_batch_size, data_chunk_size) or (None, None) if
            auto-configuration is disabled.
        """
        if not self.config.auto_memory:
            return self.config.population_batch_size, self.config.data_chunk_size

        if not CMAES_AVAILABLE:
            return None, None

        try:
            from nlsq.global_optimization import (
                auto_configure_cmaes_memory,
                compute_default_popsize,
            )

            # Use configured popsize if specified, otherwise compute default
            if self.config.popsize is not None:
                popsize = self.config.popsize
            else:
                popsize = compute_default_popsize(n_params)
            pop_batch, data_chunk = auto_configure_cmaes_memory(
                n_data=n_data,
                popsize=popsize,
                available_memory_gb=self.config.memory_limit_gb,
            )

            # Calculate estimated memory usage for logging
            # Each individual evaluation: n_data * 8 bytes (float64)
            est_memory_mb = (pop_batch * n_data * 8) / (1024 * 1024) if pop_batch else 0

            # Format data_chunk safely (may be None if auto-configured)
            data_chunk_str = f"{data_chunk:,}" if data_chunk is not None else "auto"

            logger.info(
                f"[CMA-ES] Memory configured: population_batch={pop_batch}, "
                f"data_chunk={data_chunk_str}, popsize={popsize}, "
                f"limit={self.config.memory_limit_gb:.1f}GB, "
                f"est_batch_memory={est_memory_mb:.0f}MB"
            )

            return pop_batch, data_chunk

        except (ValueError, RuntimeError, TypeError, OSError, MemoryError) as e:
            log_exception(
                logger,
                e,
                context={
                    "phase": "memory auto-configuration",
                    "n_data": n_data,
                    "n_params": n_params,
                    "memory_limit_gb": self.config.memory_limit_gb,
                },
                level=30,  # WARNING
                include_traceback=False,
            )
            return None, None

    def fit(
        self,
        model_func: Callable,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        sigma: np.ndarray | None = None,
        warmstart_chi2: float | None = None,
    ) -> CMAESResult:
        """Run CMA-ES global optimization.

        Parameters
        ----------
        model_func : Callable
            Model function: ``y = f(x, *params)``.
        xdata : np.ndarray
            Independent variable data.
        ydata : np.ndarray
            Dependent variable data to fit.
        p0 : np.ndarray
            Initial parameter guess.
        bounds : tuple[np.ndarray, np.ndarray]
            Parameter bounds as (lower, upper).
        sigma : np.ndarray | None
            Data uncertainties (optional).
        warmstart_chi2 : float | None
            Chi-squared from NLSQ warm-start. If provided and CMA-ES chi2 exceeds
            10x this value, refinement is skipped (the comparison in core.py will
            discard the CMA-ES result anyway). Also triggers use of
            ``sigma_warmstart`` instead of ``sigma`` for the CMA-ES search.

        Returns
        -------
        CMAESResult
            Optimization result with parameters, covariance, diagnostics.

        Raises
        ------
        ImportError
            If CMA-ES is not available.
        RuntimeError
            If optimization fails.
        """
        if not CMAES_AVAILABLE:
            raise ImportError(
                "CMA-ES requires NLSQ with evosax backend. "
                "Install with: pip install nlsq[evosax]"
            )

        from nlsq.global_optimization import CMAESOptimizer

        n_params = len(p0)
        n_data = len(ydata)
        scale_ratio = self.compute_scale_ratio(bounds)
        bounds_summary = _format_bounds_summary(bounds)

        # Log comprehensive configuration summary
        logger.info(
            f"[CMA-ES] Optimization starting: n_params={n_params}, "
            f"n_data={n_data:,}, preset={self.config.preset}"
        )
        logger.info(
            f"[CMA-ES] Problem characteristics: scale_ratio={scale_ratio:.2e}, {bounds_summary}"
        )

        # Log bounds for debugging
        # Bounds order follows _bounds_to_arrays canonical order:
        # [contrast, offset, D0, alpha, D_offset, gamma_dot_t0, beta,
        #  gamma_dot_t_offset, phi0] (may differ from config parameter order)
        lower, upper = bounds
        logger.debug(
            "[CMA-ES] Parameter bounds (canonical order): lower=%s, upper=%s",
            np.array2string(lower, precision=4, separator=", "),
            np.array2string(upper, precision=4, separator=", "),
        )

        # Select sigma based on warm-start state
        # When warm-start provides a near-optimal starting point, use a smaller
        # sigma for local refinement instead of global exploration.
        warmstart_active = warmstart_chi2 is not None and warmstart_chi2 < float("inf")
        if warmstart_active:
            effective_sigma = self.config.sigma_warmstart
            logger.info(
                f"[CMA-ES] Warm-start active: using sigma_warmstart="
                f"{effective_sigma:.3f} (default sigma={self.config.sigma:.3f})"
            )
        else:
            effective_sigma = None  # Use config default

        # Configure memory batching
        pop_batch, data_chunk = self._configure_memory(n_data, n_params)

        # Build CMAESConfig with memory settings
        cmaes_config = self.config.to_cmaes_config(n_params, sigma_override=effective_sigma)

        # When warm-start is active, override BIPOP -> none
        # BIPOP large-population restarts are designed for global exploration,
        # but with sigma_warmstart (small sigma), the large populations sample
        # densely in a small neighborhood without actually exploring broadly.
        # A single focused run with the full generation budget is more coherent
        # for local refinement around the warm-start solution.
        if warmstart_active and cmaes_config.restart_strategy == "bipop":
            cmaes_config.restart_strategy = "none"
            cmaes_config.max_restarts = 0
            logger.info(
                "[CMA-ES] Warm-start: overriding restart_strategy='bipop' -> 'none' "
                "(BIPOP large-population restarts are incoherent with small sigma_warmstart)"
            )

        # Adaptive population sizing for high scale-ratio problems
        # Default popsize (4+3*ln(9) ~ 11) is too small for multi-scale problems.
        # Scale up popsize and generations when scale ratio is large, unless
        # the user explicitly configured a popsize.
        if scale_ratio > 1e3:
            from nlsq.global_optimization import compute_default_popsize

            default_pop = compute_default_popsize(n_params)

            if self.config.popsize is not None:
                # Warn when explicit popsize may be too small for the scale ratio
                if scale_ratio > 1e6:
                    recommended = max(200, default_pop * 10)
                elif scale_ratio > 1e4:
                    recommended = max(100, default_pop * 5)
                else:
                    recommended = max(50, default_pop * 3)
                if self.config.popsize < recommended:
                    logger.warning(
                        f"[CMA-ES] Explicit popsize={self.config.popsize} may be too small "
                        f"for scale_ratio={scale_ratio:.2e}. Adaptive scaling recommends "
                        f"popsize={recommended}. Set popsize: null in config to enable "
                        f"adaptive sizing."
                    )
            else:
                # Auto-scale popsize and generations
                if scale_ratio > 1e6:
                    adaptive_pop = max(200, default_pop * 10)
                    adaptive_gen = max(500, cmaes_config.max_generations * 3)
                elif scale_ratio > 1e4:
                    adaptive_pop = max(100, default_pop * 5)
                    adaptive_gen = max(300, cmaes_config.max_generations * 2)
                else:
                    adaptive_pop = max(50, default_pop * 3)
                    adaptive_gen = max(200, cmaes_config.max_generations)

                logger.info(
                    f"[CMA-ES] Adaptive scaling: scale_ratio={scale_ratio:.2e} -> "
                    f"popsize {cmaes_config.popsize}->{adaptive_pop}, "
                    f"max_gen {cmaes_config.max_generations}->{adaptive_gen}"
                )
                cmaes_config.popsize = adaptive_pop
                cmaes_config.max_generations = adaptive_gen

        # Override with auto-configured memory settings
        if pop_batch is not None:
            cmaes_config.population_batch_size = pop_batch
        if data_chunk is not None:
            cmaes_config.data_chunk_size = data_chunk

        # Log algorithm configuration
        logger.info(
            f"[CMA-ES] Algorithm settings: max_generations={cmaes_config.max_generations}, "
            f"popsize={cmaes_config.popsize}, sigma={cmaes_config.sigma:.3f}"
            f"{' (warm-start)' if warmstart_active else ''}, "
            f"restart={cmaes_config.restart_strategy}, max_restarts={cmaes_config.max_restarts}"
        )
        if self.config.refine_with_nlsq:
            logger.info(
                f"[CMA-ES] Post-refinement enabled: workflow={self.config.refinement_workflow}, "
                f"ftol={self.config.refinement_ftol:.1e}"
            )

        # Create optimizer with config
        optimizer = CMAESOptimizer(config=cmaes_config)

        # Set up parameter normalization if enabled
        # Normalizes parameters to [0, 1] space for better CMA-ES covariance adaptation
        norm_state: dict[str, Any] | None = None
        if self.config.normalize:
            norm_scale, norm_offset, param_range = _compute_normalization_factors(
                bounds, self.config.normalization_epsilon
            )
            norm_state = {
                "scale": norm_scale,
                "offset": norm_offset,
                "range": param_range,
            }

            # Normalize initial parameters and bounds
            fit_p0 = _normalize_params(p0, norm_scale, norm_offset)
            fit_bounds = _normalize_bounds(bounds)

            # Wrap model function to denormalize params before evaluation
            # IMPORTANT: Use JAX operations to preserve tracers during JIT compilation
            original_model_func = model_func
            import jax.numpy as jnp  # noqa: E402 - lazy import, CMA-ES is optional

            # Pre-convert normalization factors to JAX arrays (captured in closure)
            norm_scale_jax = jnp.array(norm_scale)
            norm_offset_jax = jnp.array(norm_offset)

            def normalized_model_func(xdata: np.ndarray, *params_norm: float) -> np.ndarray:
                # Use jnp.stack to preserve JAX tracers during JIT tracing
                params_norm_jax = jnp.stack(params_norm)
                # Denormalize: x = x_norm / scale + offset = x_norm * range + lower
                params_phys = params_norm_jax / norm_scale_jax + norm_offset_jax
                return original_model_func(xdata, *params_phys)

            fit_model_func = normalized_model_func

            logger.info(
                f"[CMA-ES] Parameter normalization enabled: "
                f"range=[{np.min(param_range):.2e}, {np.max(param_range):.2e}]"
            )
        else:
            fit_p0 = p0
            fit_bounds = bounds
            fit_model_func = model_func

        # Run CMA-ES global search (NLSQ internal refinement is disabled)
        # - CMA-ES global search with covariance adaptation
        # - Optional BIPOP restarts (configured via cmaes_config.restart_strategy)
        logger.info("[CMA-ES] Global search phase starting...")

        with log_phase("CMA-ES global search", logger, track_memory=True) as search_phase:
            result = optimizer.fit(
                f=fit_model_func,
                xdata=xdata,
                ydata=ydata,
                p0=fit_p0,
                bounds=fit_bounds,
                sigma=sigma,
            )

        # Extract CMA-ES results
        cmaes_params = np.asarray(result.get("popt", fit_p0))
        cmaes_covariance = result.get("pcov", None)
        if cmaes_covariance is not None:
            cmaes_covariance = np.asarray(cmaes_covariance)

        # Denormalize results if normalization was applied
        if norm_state is not None:
            cmaes_params = _denormalize_params(
                cmaes_params, norm_state["scale"], norm_state["offset"]
            )
            cmaes_covariance = _adjust_covariance_for_normalization(
                cmaes_covariance, norm_state["range"]
            )
            logger.debug("[CMA-ES] Parameters denormalized from [0,1] to physical space")

        # Build CMA-ES diagnostics
        # NLSQ stores diagnostics under 'cmaes_diagnostics' dict
        cmaes_diag = result.get("cmaes_diagnostics", {})
        generations = cmaes_diag.get("total_generations", 0)
        restarts = cmaes_diag.get("total_restarts", 0)
        convergence_reason = cmaes_diag.get("convergence_reason", "unknown")

        # Calculate evaluations from restart history (each generation evaluates popsize candidates)
        restart_history = cmaes_diag.get("restart_history", [])
        evaluations = (
            sum(r.get("generations", 0) * r.get("popsize", 0) for r in restart_history)
            if restart_history
            else generations * cmaes_config.popsize
        )

        diagnostics = {
            "generations": generations,
            "evaluations": evaluations,
            "restarts": restarts,
            "convergence_reason": convergence_reason,
            "global_search_time_s": search_phase.duration,
            "normalization_enabled": norm_state is not None,
        }
        if search_phase.memory_peak_gb is not None:
            diagnostics["global_search_memory_gb"] = search_phase.memory_peak_gb

        # Compute CMA-ES chi-squared
        pred = model_func(xdata, *cmaes_params)
        residuals = ydata - pred
        if sigma is not None:
            residuals = residuals / sigma
        cmaes_chi_squared = float(np.sum(residuals**2))

        # Calculate evaluations per second for performance insight
        evals_per_sec = evaluations / search_phase.duration if search_phase.duration > 0 else 0

        logger.info(
            f"[CMA-ES] Global search completed: chi2={cmaes_chi_squared:.4e}, "
            f"generations={generations}, restarts={restarts}"
        )
        logger.info(
            f"[CMA-ES] Performance: {evaluations:,} evals in {search_phase.duration:.1f}s "
            f"({evals_per_sec:.0f} evals/s), reason={convergence_reason}"
        )

        # Early-exit: skip refinement if CMA-ES chi2 is much worse than warm-start
        # The comparison in core.py will discard this result anyway, so refinement
        # from a bad starting point wastes compute (can take 400+ seconds).
        skip_refinement = False
        if (
            warmstart_chi2 is not None
            and cmaes_chi_squared > _REFINEMENT_SKIP_CHI2_RATIO * warmstart_chi2
        ):
            skip_refinement = True
            logger.warning(
                f"[CMA-ES] Skipping refinement: CMA-ES chi2={cmaes_chi_squared:.4e} "
                f"is >{_REFINEMENT_SKIP_CHI2_RATIO:.0f}x worse than warm-start chi2={warmstart_chi2:.4e}. "
                f"Warm-start result will be selected in comparison phase."
            )

        # Run explicit NLSQ TRF refinement if enabled
        if self.config.refine_with_nlsq and not skip_refinement:
            refinement_result = self._run_nlsq_refinement(
                model_func=model_func,
                xdata=xdata,
                ydata=ydata,
                p0=cmaes_params,  # Start from CMA-ES solution
                bounds=bounds,
                sigma=sigma,
            )

            if refinement_result["success"]:
                # Use refined parameters
                best_params = refinement_result["popt"]
                best_covariance = refinement_result["pcov"]
                best_chi_squared = refinement_result["chi_squared"]
                nlsq_refined = True

                # Update diagnostics with refinement info.
                # NLSQ curve_fit returns only (popt, pcov) — no nfev is available
                # (infodict is always empty), so we do not emit a misleading
                # always-zero refinement_nfev key here.
                diagnostics["refinement_message"] = refinement_result["mesg"]
                diagnostics["cmaes_chi_squared"] = cmaes_chi_squared
                diagnostics["refined_chi_squared"] = best_chi_squared

                # Calculate improvement percentage
                if cmaes_chi_squared > 0.0:
                    improvement = (cmaes_chi_squared - best_chi_squared) / cmaes_chi_squared
                else:
                    improvement = 0.0
                diagnostics["chi_squared_improvement"] = improvement

                # Add refinement timing if available
                if "duration_s" in refinement_result:
                    diagnostics["refinement_time_s"] = refinement_result["duration_s"]

                logger.info(
                    f"[CMA-ES] Refinement improved chi2: "
                    f"{cmaes_chi_squared:.4e} -> {best_chi_squared:.4e} "
                    f"({improvement:.2%} improvement)"
                )
            else:
                # Refinement failed, use CMA-ES result
                logger.warning(
                    f"[CMA-ES] Refinement failed, using CMA-ES result. "
                    f"Reason: {refinement_result['mesg']}"
                )
                best_params = cmaes_params
                best_covariance = cmaes_covariance
                best_chi_squared = cmaes_chi_squared
                nlsq_refined = False
                diagnostics["refinement_failed"] = True
                diagnostics["refinement_message"] = refinement_result["mesg"]
        else:
            # No refinement requested
            best_params = cmaes_params
            best_covariance = cmaes_covariance
            best_chi_squared = cmaes_chi_squared
            nlsq_refined = False
            logger.debug("[CMA-ES] Post-refinement disabled, using global search result")

        # Calculate total time
        total_time = search_phase.duration
        if nlsq_refined and "refinement_time_s" in diagnostics:
            total_time += diagnostics["refinement_time_s"]
        diagnostics["total_time_s"] = total_time

        # Log final summary
        refined_str = " (refined)" if nlsq_refined else ""
        logger.info(
            f"[CMA-ES] Optimization completed{refined_str}: "
            f"chi2={best_chi_squared:.4e}, total_time={total_time:.1f}s"
        )

        converged_reasons = {"tol_fun", "tol_x", "tol_fun_hist", "ftarget"}
        cmaes_converged = convergence_reason in converged_reasons
        # CR-5: NLSQ refinement can polish a point but cannot make an unconverged
        # global search "successful". Keep the two notions distinct and warn when
        # the success flag rests on refinement alone, so the caller in
        # heterodyne_core (which picks CMA-ES vs NLSQ off this flag) is not misled.
        success = cmaes_converged or nlsq_refined
        if nlsq_refined and not cmaes_converged:
            logger.warning(
                "CMA-ES did not meet a convergence criterion (reason=%s); success "
                "set via NLSQ refinement only — the global search may not have "
                "found the correct basin.",
                convergence_reason,
            )

        if cmaes_converged:
            message = f"CMA-ES converged: {convergence_reason}"
        elif nlsq_refined:
            message = (
                f"CMA-ES did not converge (reason={convergence_reason}); result refined by NLSQ"
            )
        else:
            message = f"CMA-ES did not converge (reason={convergence_reason})"

        return CMAESResult(
            parameters=best_params,
            covariance=best_covariance,
            chi_squared=best_chi_squared,
            success=success,
            diagnostics=diagnostics,
            method_used="cmaes",
            nlsq_refined=nlsq_refined,
            message=message,
        )


def fit_with_cmaes(
    model_func: Callable,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    sigma: np.ndarray | None = None,
    config: CMAESWrapperConfig | None = None,
) -> CMAESResult:
    """Run CMA-ES optimization with default wrapper configuration.

    Parameters
    ----------
    model_func : Callable
        Model function: ``y = f(x, *params)``.
    xdata : np.ndarray
        Independent variable data.
    ydata : np.ndarray
        Dependent variable data to fit.
    p0 : np.ndarray
        Initial parameter guess.
    bounds : tuple[np.ndarray, np.ndarray]
        Parameter bounds as (lower, upper).
    sigma : np.ndarray | None
        Data uncertainties (optional).
    config : CMAESWrapperConfig | None
        Configuration. Uses defaults if None.

    Returns
    -------
    CMAESResult
        Optimization result.

    Examples
    --------
    >>> result = fit_with_cmaes(model, x, y, p0, bounds)
    >>> print(f"Best params: {result.parameters}")
    """
    wrapper = CMAESWrapper(config)
    return wrapper.fit(model_func, xdata, ydata, p0, bounds, sigma)
