"""Multi-start NLSQ optimization with Latin Hypercube Sampling.

This module implements multi-start optimization to explore the parameter space
and avoid local minima. All datasets use the FULL strategy (N complete fits).

NOTE: Subsampling is explicitly NOT supported per project requirements.
Numerical precision and reproducibility take priority over computational speed.

Part of homodyne v2.6.0 architecture.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import qmc

from xpcsjax.optimization.nlsq.progress import MultiStartProgressTracker
from xpcsjax.utils.logging import get_logger

# Timeout for individual worker results (seconds)
# If a worker doesn't return within this time, we fall back to sequential
_WORKER_TIMEOUT = 1800  # 30 minutes per worker

# Maximum data points for parallel execution
# Beyond this, sequential execution is used to avoid memory/serialization issues
# 5 workers × 1M points × ~100 bytes/point = ~500MB serialization overhead per worker
_MAX_POINTS_FOR_PARALLEL = 500_000  # 500K points

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from xpcsjax.optimization.nlsq.results import ConvergenceStatus, OptimizationResult

logger = get_logger(__name__)


# =============================================================================
# Configuration Dataclass
# =============================================================================


@dataclass
class MultiStartConfig:
    """Configuration for multi-start optimization.

    Attributes
    ----------
    enable : bool
        Whether to use multi-start optimization. Default: False.
    n_starts : int
        Number of starting points to generate. Default: 10.
    seed : int
        Random seed for reproducibility. Default: 42.
    sampling_strategy : str
        Method for generating starting points. "latin_hypercube" or "random".
    custom_starts : list[list[float]] | None
        User-provided custom starting points to include alongside generated starts.
    n_workers : int
        Number of parallel workers. 0 = auto (min of n_starts, cpu_count).
    use_screening : bool
        Whether to pre-filter starting points by initial cost.
    screen_keep_fraction : float
        Fraction of starting points to keep after screening.
    refine_top_k : int
        Number of top solutions to refine with tighter tolerance.
    refinement_ftol : float
        Function tolerance for refinement phase.
    degeneracy_threshold : float
        Chi-squared similarity threshold for degeneracy detection.
    """

    enable: bool = False
    # 10 satisfies LHS coverage heuristic (n_starts ≥ n_params) for typical XPCS physics fits
    # (4–8 physics params). Raise for fourier/individual heterodyne modes with large n_phi.
    n_starts: int = 10
    seed: int = 42
    sampling_strategy: str = "latin_hypercube"
    custom_starts: list[list[float]] | None = None
    n_workers: int = 0
    use_screening: bool = True
    screen_keep_fraction: float = 0.5
    refine_top_k: int = 3
    refinement_ftol: float = 1e-12
    degeneracy_threshold: float = 0.1
    # M-7: when every start fails, the "best" result is a failed one carrying
    # initial-guess parameters. Callers that only check chi_squared (not
    # .success) could silently proceed with it. Opt-in to a hard RuntimeError.
    raise_on_total_failure: bool = False

    @classmethod
    def from_nlsq_config(cls, nlsq_config: Any) -> MultiStartConfig:
        """Create MultiStartConfig from NLSQConfig.

        Parameters
        ----------
        nlsq_config : NLSQConfig
            NLSQ configuration object.

        Returns
        -------
        MultiStartConfig
            Multi-start configuration.
        """
        # Handle custom_starts if present in NLSQConfig
        custom_starts = getattr(nlsq_config, "multi_start_custom_starts", None)

        return cls(
            enable=nlsq_config.enable_multi_start,
            n_starts=nlsq_config.multi_start_n_starts,
            seed=nlsq_config.multi_start_seed,
            sampling_strategy=nlsq_config.multi_start_sampling_strategy,
            custom_starts=custom_starts,
            n_workers=nlsq_config.multi_start_n_workers,
            use_screening=nlsq_config.multi_start_use_screening,
            screen_keep_fraction=nlsq_config.multi_start_screen_keep_fraction,
            refine_top_k=nlsq_config.multi_start_refine_top_k,
            refinement_ftol=nlsq_config.multi_start_refinement_ftol,
            degeneracy_threshold=nlsq_config.multi_start_degeneracy_threshold,
        )

    def to_nlsq_global_config(self) -> Any:
        """Convert to NLSQ's GlobalOptimizationConfig.

        Returns
        -------
        GlobalOptimizationConfig
            NLSQ global optimization configuration.

        Raises
        ------
        ImportError
            If NLSQ GlobalOptimizationConfig is not available.

        Notes
        -----
        Maps homodyne's multi-start configuration to NLSQ's GlobalOptimizationConfig:
        - sampling_strategy -> sampler (lhs, sobol, halton)
        - use_screening -> elimination_rounds (0 if disabled)
        - screen_keep_fraction -> elimination_fraction (inverted)
        """
        try:
            from nlsq.global_optimization import GlobalOptimizationConfig
        except ImportError as e:
            raise ImportError(
                "NLSQ GlobalOptimizationConfig not available. "
                "Please install NLSQ >= 0.4.0: pip install nlsq>=0.4.0"
            ) from e

        # Map sampling strategy to NLSQ sampler
        sampler_map = {
            "latin_hypercube": "lhs",
            "lhs": "lhs",
            "sobol": "sobol",
            "halton": "halton",
            "random": "lhs",  # Fallback random to LHS
        }
        sampler = sampler_map.get(self.sampling_strategy, "lhs")

        # Map screening to elimination rounds
        # screen_keep_fraction=0.5 means 50% kept = 50% eliminated
        elimination_fraction = 1.0 - self.screen_keep_fraction
        elimination_rounds = 3 if self.use_screening else 0

        # ``sampler`` is a Literal['lhs','sobol','halton'] in NLSQ's API but
        # comes from a free-text config field here. Validation upstream
        # guarantees it's one of the three; cast to keep mypy quiet without
        # adding a redundant runtime ``assert sampler in {...}``.
        from typing import Literal, cast

        return GlobalOptimizationConfig(
            n_starts=self.n_starts,
            sampler=cast(Literal["lhs", "sobol", "halton"], sampler),
            elimination_rounds=elimination_rounds,
            elimination_fraction=elimination_fraction,
        )


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass
class SingleStartResult:
    """Result from a single starting point optimization.

    Attributes
    ----------
    start_idx : int
        Index of the starting point in the LHS sequence.
    initial_params : NDArray[np.float64]
        Initial parameter values used.
    final_params : NDArray[np.float64]
        Optimized parameter values.
    chi_squared : float
        Final chi-squared value.
    reduced_chi_squared : float
        Chi-squared divided by degrees of freedom.
    success : bool
        Whether optimization converged successfully.
    status : int
        Optimizer status code.
    message : str
        Optimizer status message.
    n_iterations : int
        Number of optimization iterations.
    n_fev : int
        Number of function evaluations.
    wall_time : float
        Execution time in seconds.
    hessian : NDArray[np.float64] | None
        Hessian matrix at solution.
    covariance : NDArray[np.float64] | None
        Parameter covariance matrix.
    jacobian : NDArray[np.float64] | None
        Final Jacobian matrix.
    """

    start_idx: int
    initial_params: NDArray[np.float64]
    final_params: NDArray[np.float64]
    chi_squared: float
    reduced_chi_squared: float = float("inf")
    success: bool = False
    status: int = 0
    message: str = ""
    n_iterations: int = 0
    n_fev: int = 0
    wall_time: float = 0.0
    hessian: NDArray[np.float64] | None = None
    covariance: NDArray[np.float64] | None = None
    jacobian: NDArray[np.float64] | None = None


@dataclass
class MultiStartResult:
    """Aggregated results from multi-start optimization.

    Attributes
    ----------
    best : SingleStartResult
        Best result by chi-squared.
    all_results : list[SingleStartResult]
        All optimization results.
    config : MultiStartConfig
        Configuration used.
    strategy_used : str
        Strategy that was used (always "full").
    n_successful : int
        Number of successful optimizations.
    n_unique_basins : int
        Number of distinct local minima found.
    degeneracy_detected : bool
        Whether parameter degeneracy was detected.
    total_wall_time : float
        Total execution time in seconds.
    screening_costs : NDArray[np.float64] | None
        Initial costs from screening phase.
    basin_labels : NDArray[np.int64] | None
        Cluster labels for each result.
    """

    best: SingleStartResult
    all_results: list[SingleStartResult]
    config: MultiStartConfig
    strategy_used: str
    n_successful: int = 0
    n_unique_basins: int = 1
    degeneracy_detected: bool = False
    total_wall_time: float = 0.0
    screening_costs: NDArray[np.float64] | None = None
    basin_labels: NDArray[np.int64] | None = None

    def to_optimization_result(self) -> OptimizationResult:
        """Convert MultiStartResult to OptimizationResult for CLI compatibility.

        Returns
        -------
        OptimizationResult
            Optimization result object containing the best solution with
            multi-start metadata in nlsq_diagnostics.
        """
        from xpcsjax.optimization.nlsq.results import OptimizationResult

        best = self.best
        n_params = len(best.final_params)

        # Determine convergence status
        convergence_status: ConvergenceStatus
        if best.success:
            convergence_status = "converged"
        else:
            convergence_status = "failed"

        # Determine quality flag based on chi-squared
        if best.reduced_chi_squared < 2.0:
            quality_flag = "good"
        elif best.reduced_chi_squared < 10.0:
            quality_flag = "marginal"
        else:
            quality_flag = "poor"

        # Build multi-start diagnostics
        multistart_diagnostics = {
            "strategy_used": self.strategy_used,
            "n_starts": len(self.all_results),
            "n_successful": self.n_successful,
            "n_unique_basins": self.n_unique_basins,
            "degeneracy_detected": self.degeneracy_detected,
            "total_wall_time": self.total_wall_time,
            "best_start_idx": best.start_idx,
        }

        return OptimizationResult(
            parameters=best.final_params,
            uncertainties=(
                np.sqrt(np.diag(best.covariance))
                if best.covariance is not None
                else np.zeros(n_params)
            ),
            covariance=(
                best.covariance if best.covariance is not None else np.eye(n_params)
            ),
            chi_squared=best.chi_squared,
            reduced_chi_squared=best.reduced_chi_squared,
            convergence_status=convergence_status,
            iterations=best.n_iterations,
            execution_time=self.total_wall_time,
            device_info={"type": "cpu", "multistart": True},
            recovery_actions=[],
            quality_flag=quality_flag,
            nlsq_diagnostics=multistart_diagnostics,
        )


# =============================================================================
# Helper Functions
# =============================================================================


def _get_phi_from_data(data: dict[str, Any]) -> NDArray[np.float64] | None:
    """Extract phi array from data dictionary, handling numpy array truthiness.

    Parameters
    ----------
    data : dict
        Data dictionary that may contain 'phi' or 'phi_angles_list'.

    Returns
    -------
    NDArray | None
        Phi array if found, None otherwise.
    """
    phi = data.get("phi")
    if phi is not None:
        return np.asarray(phi)
    phi = data.get("phi_angles_list")
    if phi is not None:
        return np.asarray(phi)
    return None


def _get_dataset_size(data: dict[str, Any]) -> int:
    """Calculate total number of data points from data dictionary.

    This function handles both test fixtures (where 'phi' is flattened with
    repeated angles) and actual XPCS data (where 'phi_angles_list' contains
    only unique angles and 'g2'/'c2_exp' contains the actual 3D data).

    Parameters
    ----------
    data : dict
        Data dictionary that may contain:
        - 'g2' or 'c2_exp': Experimental data array (n_phi, n_t1, n_t2)
        - 'phi' or 'phi_angles_list': Phi angles

    Returns
    -------
    int
        Total number of data points.

    Raises
    ------
    ValueError
        If no valid data array is found.
    """
    # First, try to get size from actual data arrays (most reliable)
    for key in ("g2", "c2_exp"):
        arr = data.get(key)
        if arr is not None:
            arr = np.asarray(arr)
            return int(arr.size)

    # Fallback: calculate from phi array
    # This handles test fixtures where phi is already flattened
    phi = _get_phi_from_data(data)
    if phi is not None:
        return len(np.asarray(phi).ravel())

    raise ValueError("Cannot determine dataset size: no 'g2', 'c2_exp', or 'phi' found")


# =============================================================================
# Core Functions: Bounds Validation
# =============================================================================


def check_zero_volume_bounds(bounds: NDArray[np.float64]) -> bool:
    """Check if parameter bounds have zero volume (all lower == upper).

    Parameters
    ----------
    bounds : NDArray[np.float64]
        Parameter bounds as (n_params, 2) array with [lower, upper] for each.

    Returns
    -------
    bool
        True if bounds have zero volume (all parameters fixed).
    """
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    widths = upper - lower

    # Zero volume if all widths are effectively zero.
    # ``np.all`` returns ``np.bool_`` rather than Python ``bool``; coerce so
    # callers using the result in a Python boolean context type-check.
    return bool(np.all(np.abs(widths) < 1e-15))


def validate_n_starts_for_lhs(
    n_starts: int,
    n_params: int,
    warn: bool = True,
) -> int:
    """Validate n_starts for Latin Hypercube Sampling coverage.

    For LHS to provide meaningful coverage, n_starts should be at least n_params.
    Very large n_starts relative to parameter space may produce redundant samples.

    Parameters
    ----------
    n_starts : int
        Requested number of starting points.
    n_params : int
        Number of parameters (dimensions).
    warn : bool
        Whether to emit warnings for suboptimal settings.

    Returns
    -------
    int
        Validated n_starts (unchanged if valid).
    """
    # Minimum recommended: at least n_params for basic coverage
    if n_starts < n_params and warn:
        logger.warning(
            f"n_starts ({n_starts}) < n_params ({n_params}): "
            f"LHS coverage may be inadequate. Consider n_starts >= {n_params}."
        )

    # Very large n_starts warning (heuristic: >1000 per dimension is likely excessive)
    max_meaningful = n_params * 1000
    if n_starts > max_meaningful and warn:
        logger.warning(
            f"n_starts ({n_starts}) is very large for {n_params} parameters. "
            f"This may produce redundant samples with diminishing returns. "
            f"Consider n_starts <= {max_meaningful}."
        )

    return n_starts


# =============================================================================
# Core Functions: LHS Generation
# =============================================================================


def generate_lhs_starts(
    bounds: NDArray[np.float64],
    n_starts: int,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Generate starting points via Latin Hypercube Sampling.

    Parameters
    ----------
    bounds : NDArray[np.float64]
        Parameter bounds as (n_params, 2) array with [lower, upper] for each.
    n_starts : int
        Number of starting points to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    NDArray[np.float64]
        Starting points as (n_starts, n_params) array.
    """
    n_params = bounds.shape[0]

    # Use scipy's Latin Hypercube Sampling
    sampler = qmc.LatinHypercube(d=n_params, seed=seed, optimization="random-cd")
    unit_samples = sampler.random(n=n_starts)  # Samples in [0, 1]^d

    # Scale to parameter bounds
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    scaled_samples = qmc.scale(unit_samples, lower, upper)

    logger.debug(f"Generated {n_starts} LHS starting points for {n_params} parameters")
    return scaled_samples


def generate_random_starts(
    bounds: NDArray[np.float64],
    n_starts: int,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Generate starting points via random uniform sampling.

    Parameters
    ----------
    bounds : NDArray[np.float64]
        Parameter bounds as (n_params, 2) array.
    n_starts : int
        Number of starting points to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    NDArray[np.float64]
        Starting points as (n_starts, n_params) array.
    """
    rng = np.random.default_rng(seed)
    n_params = bounds.shape[0]

    lower = bounds[:, 0]
    upper = bounds[:, 1]
    samples = rng.uniform(lower, upper, size=(n_starts, n_params))

    logger.debug(
        f"Generated {n_starts} random starting points for {n_params} parameters"
    )
    return samples


def include_custom_starts(
    generated_starts: NDArray[np.float64],
    custom_starts: list[list[float]] | NDArray[np.float64] | None,
    bounds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Include user-provided custom starting points alongside generated starts.

    Custom starting points are prepended to the generated starts so they are
    always included (not filtered by screening).

    Parameters
    ----------
    generated_starts : NDArray[np.float64]
        Starting points generated by LHS or random sampling.
    custom_starts : list[list[float]] | NDArray[np.float64] | None
        User-provided custom starting points.
    bounds : NDArray[np.float64]
        Parameter bounds for validation.

    Returns
    -------
    NDArray[np.float64]
        Combined starting points with custom starts first.
    """
    if custom_starts is None or len(custom_starts) == 0:
        return generated_starts

    custom_array = np.asarray(custom_starts, dtype=np.float64)

    # Validate dimensions
    n_params = bounds.shape[0]
    if custom_array.ndim == 1:
        custom_array = custom_array.reshape(1, -1)

    if custom_array.shape[1] != n_params:
        logger.warning(
            f"Custom starts have wrong dimension: {custom_array.shape[1]} != {n_params}. "
            f"Ignoring custom starts."
        )
        return generated_starts

    # Validate bounds
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    n_custom = len(custom_array)
    valid_mask = np.all((custom_array >= lower) & (custom_array <= upper), axis=1)
    n_valid = np.sum(valid_mask)

    if n_valid < n_custom:
        n_invalid = n_custom - n_valid
        logger.warning(
            f"{n_invalid} custom starting point(s) are outside bounds and will be skipped."
        )
        custom_array = custom_array[valid_mask]

    if len(custom_array) == 0:
        return generated_starts

    logger.info(f"Including {len(custom_array)} custom starting point(s)")

    # Prepend custom starts so they're always included
    combined = np.vstack([custom_array, generated_starts])
    return combined


# =============================================================================
# Core Functions: Screening
# =============================================================================


def screen_starts(
    cost_func: Callable[[NDArray[np.float64]], float],
    starts: NDArray[np.float64],
    keep_fraction: float = 0.5,
    min_keep: int = 3,
    n_workers: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pre-filter starting points by initial cost.

    Parameters
    ----------
    cost_func : Callable
        Function that computes cost (chi-squared) for a parameter vector.
    starts : NDArray[np.float64]
        Starting points as (n_starts, n_params) array.
    keep_fraction : float
        Fraction of starting points to keep (0, 1].
    min_keep : int
        Minimum number of starting points to keep.
    n_workers : int
        Number of parallel workers for cost evaluation. 0 = auto (cpu_count - 1).

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        Filtered starting points and their initial costs.
    """
    n_starts = len(starts)
    n_keep = max(min_keep, int(n_starts * keep_fraction))
    n_keep = min(n_keep, n_starts)  # Don't keep more than we have

    # Evaluate initial cost for each starting point
    # Use parallel execution for faster screening (4x speedup typical)
    if n_workers == 0:
        n_workers = max(1, (os.cpu_count() or 1) - 1)

    # Only parallelize if we have enough starts to benefit
    if n_starts >= 4 and n_workers > 1:
        try:
            from concurrent.futures import ThreadPoolExecutor

            # Use threads since cost_func likely releases GIL (JAX/NumPy)
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                costs = np.array(list(executor.map(cost_func, starts)))
        except (RuntimeError, OSError, ValueError) as e:
            logger.warning(
                f"Parallel screening failed, falling back to sequential: {e}"
            )
            costs = np.array([cost_func(start) for start in starts])
    else:
        costs = np.array([cost_func(start) for start in starts])

    # Sort by cost and keep top n_keep
    sorted_indices = np.argsort(costs)
    keep_indices = sorted_indices[:n_keep]

    filtered_starts = starts[keep_indices]
    filtered_costs = costs[keep_indices]

    logger.info(
        f"Screening: kept {n_keep}/{n_starts} starts "
        f"(best cost: {filtered_costs[0]:.4g}, worst kept: {filtered_costs[-1]:.4g})"
    )

    return filtered_starts, costs


# =============================================================================
# Core Functions: Degeneracy Detection
# =============================================================================


def detect_degeneracy(
    results: list[SingleStartResult],
    chi_sq_threshold: float = 0.1,
    param_threshold: float = 0.2,
) -> tuple[bool, int, NDArray[np.int64] | None]:
    """Detect parameter degeneracy from multiple optimization results.

    Parameters
    ----------
    results : list[SingleStartResult]
        List of optimization results.
    chi_sq_threshold : float
        Maximum relative chi-squared difference to consider similar.
    param_threshold : float
        Maximum relative parameter distance to consider same basin.

    Returns
    -------
    tuple[bool, int, NDArray[np.int64] | None]
        (degeneracy_detected, n_unique_basins, basin_labels)
    """
    # Filter successful results
    successful = [r for r in results if r.success]
    if len(successful) < 2:
        return False, 1, None

    # Sort by chi-squared
    successful.sort(key=lambda r: r.chi_squared)
    best_chi_sq = successful[0].chi_squared

    # Cluster into basins
    basins: list[list[SingleStartResult]] = []
    basin_assignments: list[int] = []

    for r in successful:
        # Check chi-squared similarity
        chi_sq_diff = abs(r.chi_squared - best_chi_sq) / (best_chi_sq + 1e-10)
        if chi_sq_diff > chi_sq_threshold:
            # Not similar enough, assign to "other" basin
            basin_assignments.append(-1)
            continue

        # Check parameter distance to existing basins
        r_params = r.final_params
        found_basin = False

        for basin_idx, basin in enumerate(basins):
            basin_center = basin[0].final_params
            param_dist = np.linalg.norm(r_params - basin_center) / (
                np.linalg.norm(basin_center) + 1e-10
            )
            if param_dist < param_threshold:
                basin.append(r)
                basin_assignments.append(basin_idx)
                found_basin = True
                break

        if not found_basin:
            basins.append([r])
            basin_assignments.append(len(basins) - 1)

    n_unique_basins = len(basins)
    degeneracy_detected = n_unique_basins > 1

    # Create labels array
    labels = np.array(basin_assignments, dtype=np.int64)

    if degeneracy_detected:
        logger.warning(
            f"Parameter degeneracy detected: {n_unique_basins} distinct basins "
            f"with similar chi-squared values"
        )

    return degeneracy_detected, n_unique_basins, labels


# =============================================================================
# Core Functions: Parallel Execution
# =============================================================================


def get_n_workers(config: MultiStartConfig, n_starts: int) -> int:
    """Determine number of parallel workers.

    Parameters
    ----------
    config : MultiStartConfig
        Multi-start configuration.
    n_starts : int
        Number of starting points.

    Returns
    -------
    int
        Number of workers to use.
    """
    if config.n_workers > 0:
        n_workers = config.n_workers
    else:
        n_workers = os.cpu_count() or 4

    # Don't use more workers than starts
    n_workers = min(n_workers, n_starts)

    logger.debug(f"Using {n_workers} parallel workers for {n_starts} starts")
    return n_workers


def _run_sequential(
    optimize_func: Callable[[int, NDArray[np.float64]], SingleStartResult],
    starts: NDArray[np.float64],
) -> list[SingleStartResult]:
    """Run optimizations sequentially."""
    results: list[SingleStartResult] = []
    for idx, start in enumerate(starts):
        try:
            result = optimize_func(idx, start)
            results.append(result)
        except (ValueError, RuntimeError, OSError) as e:
            logger.warning(f"Start {idx} failed: {e}")
            results.append(
                SingleStartResult(
                    start_idx=idx,
                    initial_params=start,
                    final_params=start,
                    chi_squared=np.inf,
                    success=False,
                    message=str(e),
                )
            )
    return results


def _is_pickle_error(error_msg: str) -> bool:
    """Check if an error message indicates a pickle/serialization issue."""
    pickle_indicators = [
        "pickle",
        "local object",
        "can't get local",
        "cannot serialize",
        "attributeerror",
    ]
    error_lower = error_msg.lower()
    return any(indicator in error_lower for indicator in pickle_indicators)


# =============================================================================
# Main Orchestration Function
# =============================================================================


def run_multistart_nlsq(
    data: dict[str, Any],
    bounds: NDArray[np.float64],
    config: MultiStartConfig,
    single_fit_func: Callable[[dict[str, Any], NDArray[np.float64]], SingleStartResult],
    cost_func: Callable[[NDArray[np.float64]], float] | None = None,
    custom_starts: list[list[float]] | NDArray[np.float64] | None = None,
) -> MultiStartResult:
    """Run multi-start NLSQ optimization with FULL strategy.

    NOTE: Only FULL strategy is supported. Subsampling is explicitly NOT used
    per project requirements - numerical precision takes priority over speed.

    Parameters
    ----------
    data : dict
        XPCS data dictionary.
    bounds : NDArray[np.float64]
        Parameter bounds as (n_params, 2) array.
    config : MultiStartConfig
        Multi-start configuration.
    single_fit_func : Callable
        Function that runs a single NLSQ fit.
        Signature: (data, initial_params) -> SingleStartResult
    cost_func : Callable, optional
        Function that computes cost for screening.
        Signature: (params) -> float
    custom_starts : list[list[float]] | NDArray, optional
        User-provided custom starting points (overrides config.custom_starts).

    Returns
    -------
    MultiStartResult
        Aggregated results from all starting points.
    """
    start_time = time.perf_counter()

    # Log configuration
    logger.info("=" * 60)
    logger.info("MULTI-START NLSQ OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(
        f"Configuration: n_starts={config.n_starts}, "
        f"sampling={config.sampling_strategy}, seed={config.seed}"
    )
    logger.info(
        f"Options: screening={'ON' if config.use_screening else 'OFF'}, "
        f"keep_fraction={config.screen_keep_fraction:.0%}, "
        f"n_workers={config.n_workers if config.n_workers > 0 else 'auto'}"
    )

    # Check for zero-volume bounds (all parameters fixed)
    if check_zero_volume_bounds(bounds):
        logger.warning(
            "Parameter bounds have zero volume (all lower == upper). "
            "Falling back to single-start at bounds center."
        )
        center = (bounds[:, 0] + bounds[:, 1]) / 2
        result = single_fit_func(data, center)
        result.start_idx = 0
        result.initial_params = center

        return MultiStartResult(
            best=result,
            all_results=[result],
            config=config,
            strategy_used="single_start_fallback",
            n_successful=1 if result.success else 0,
            n_unique_basins=1,
            degeneracy_detected=False,
            total_wall_time=time.perf_counter() - start_time,
        )

    # Determine dataset size from data arrays (not just phi array length)
    n_points = _get_dataset_size(data)
    n_params = bounds.shape[0]

    # Log dataset and parameter info
    logger.info(f"Dataset: {n_points:,} total data points")
    logger.info(f"Parameters: {n_params} free parameters")
    logger.debug(f"Parameter bounds:\n{bounds}")

    # Always use FULL strategy - no subsampling
    logger.info("Strategy: FULL (all starting points run complete optimization)")

    # Validate n_starts for LHS
    validate_n_starts_for_lhs(config.n_starts, n_params)

    # Generate starting points
    logger.info("-" * 40)
    logger.info("PHASE 1: Generating starting points")
    logger.info("-" * 40)
    if config.sampling_strategy == "latin_hypercube":
        logger.info(
            f"Using Latin Hypercube Sampling (n={config.n_starts}, seed={config.seed})"
        )
        starts = generate_lhs_starts(bounds, config.n_starts, config.seed)
    else:
        logger.info(
            f"Using random uniform sampling (n={config.n_starts}, seed={config.seed})"
        )
        starts = generate_random_starts(bounds, config.n_starts, config.seed)
    logger.info(f"Generated {len(starts)} starting points")

    # Include custom starting points (from argument or config)
    custom = custom_starts if custom_starts is not None else config.custom_starts
    n_before = len(starts)
    starts = include_custom_starts(starts, custom, bounds)
    n_added = len(starts) - n_before
    if n_added > 0:
        logger.info(f"Added {n_added} custom starting point(s), total: {len(starts)}")

    # Screening phase (optional)
    screening_costs = None
    if config.use_screening and cost_func is not None:
        logger.info("-" * 40)
        logger.info("PHASE 2: Screening starting points")
        logger.info("-" * 40)
        n_before_screen = len(starts)
        starts, screening_costs = screen_starts(
            cost_func, starts, config.screen_keep_fraction
        )
        n_filtered = n_before_screen - len(starts)
        logger.info(f"Screening filtered {n_filtered} starts, keeping {len(starts)}")
    else:
        logger.debug("Screening disabled, proceeding with all starting points")

    # Get worker count
    n_workers = get_n_workers(config, len(starts))
    cpu_count = os.cpu_count() or 1

    # Execute FULL strategy (N complete fits)
    logger.info("-" * 40)
    logger.info("PHASE 3: Running optimizations")
    logger.info("-" * 40)
    logger.info(
        f"Starting {len(starts)} optimizations with "
        f"{n_workers} worker(s) (CPUs available: {cpu_count})"
    )
    if n_points > _MAX_POINTS_FOR_PARALLEL:
        logger.info(
            f"Note: Large dataset ({n_points:,} > {_MAX_POINTS_FOR_PARALLEL:,}), "
            f"forcing sequential execution to avoid serialization overhead"
        )

    results = _run_full_strategy(
        data,
        starts,
        single_fit_func,
        n_workers,
        enable_progress_bar=True,  # Always show progress for multi-start
        verbose=1,
    )

    # Find best result
    logger.info("-" * 40)
    logger.info("PHASE 4: Analyzing results")
    logger.info("-" * 40)

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if not successful:
        logger.error("All multi-start optimizations failed!")
        for r in failed[:5]:  # Show first 5 failures
            logger.error(f"  Start {r.start_idx}: {r.message}")
        if len(failed) > 5:
            logger.error(f"  ... and {len(failed) - 5} more failures")

        if config.raise_on_total_failure:
            raise RuntimeError(
                f"All {len(results)} multi-start optimizations failed; no "
                "converged result is available (raise_on_total_failure=True)."
            )

        best = (
            results[0]
            if results
            else SingleStartResult(
                start_idx=0,
                initial_params=starts[0],
                final_params=starts[0],
                chi_squared=np.inf,
                success=False,
                message="All optimizations failed",
            )
        )
    else:
        best = min(successful, key=lambda r: r.chi_squared)
        logger.info(f"Successful optimizations: {len(successful)}/{len(results)}")
        if failed:
            logger.warning(f"Failed optimizations: {len(failed)}")
            for r in failed:
                logger.debug(f"  Start {r.start_idx} failed: {r.message}")

    # Degeneracy detection
    degeneracy_detected, n_unique_basins, basin_labels = detect_degeneracy(
        results, config.degeneracy_threshold
    )

    total_time = time.perf_counter() - start_time

    # Final summary
    logger.info("=" * 60)
    logger.info("MULTI-START OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best result: chi2={best.chi_squared:.6e} (start {best.start_idx})")
    logger.info(f"Best reduced chi2: {best.reduced_chi_squared:.6f}")
    logger.info(f"Unique basins found: {n_unique_basins}")
    if degeneracy_detected:
        logger.warning(
            f"DEGENERACY DETECTED: {n_unique_basins} distinct minima with similar chi2"
        )
    logger.info(f"Total wall time: {total_time:.1f}s")
    logger.info("=" * 60)

    return MultiStartResult(
        best=best,
        all_results=results,
        config=config,
        strategy_used="full",
        n_successful=len(successful),
        n_unique_basins=n_unique_basins,
        degeneracy_detected=degeneracy_detected,
        total_wall_time=total_time,
        screening_costs=screening_costs,
        basin_labels=basin_labels,
    )


# =============================================================================
# Strategy Implementation
# =============================================================================


class _OptimizeWorker:
    """Picklable worker class for parallel optimization.

    This class wraps the single_fit_func and data, making them picklable
    for use with ProcessPoolExecutor. Unlike nested closures, class instances
    with __call__ can be pickled as long as their attributes are picklable.
    """

    def __init__(
        self,
        data: dict[str, Any],
        single_fit_func: Callable[
            [dict[str, Any], NDArray[np.float64]], SingleStartResult
        ],
    ) -> None:
        self.data = data
        self.single_fit_func = single_fit_func

    def __call__(self, idx: int, start: NDArray[np.float64]) -> SingleStartResult:
        """Run optimization for a single starting point."""
        start_time = time.perf_counter()
        result = self.single_fit_func(self.data, start)
        result.start_idx = idx
        result.initial_params = start
        result.wall_time = time.perf_counter() - start_time
        return result


def _run_full_strategy(
    data: dict[str, Any],
    starts: NDArray[np.float64],
    single_fit_func: Callable[[dict[str, Any], NDArray[np.float64]], SingleStartResult],
    n_workers: int,
    enable_progress_bar: bool = True,
    verbose: int = 1,
) -> list[SingleStartResult]:
    """Full multi-start: run N complete fits in parallel.

    This is the ONLY supported strategy. No subsampling is performed.

    For large datasets (>500K points), sequential execution is used to avoid
    memory/serialization overhead from sending data to worker processes.

    Parameters
    ----------
    data : dict
        XPCS data dictionary.
    starts : NDArray
        Starting points as (n_starts, n_params) array.
    single_fit_func : Callable
        Function to run single NLSQ fit.
    n_workers : int
        Number of parallel workers.
    enable_progress_bar : bool
        Whether to show progress bar.
    verbose : int
        Verbosity level.

    Returns
    -------
    list[SingleStartResult]
        Results from all starting points.
    """
    n_starts = len(starts)
    execution_mode = "parallel"

    # Check dataset size - force sequential for large datasets
    # Parallel execution with large data causes serialization overhead and hangs
    n_points = _get_dataset_size(data)
    if n_points > _MAX_POINTS_FOR_PARALLEL and n_workers > 1:
        # Note: The caller (run_multistart_nlsq) already logs this decision
        n_workers = 1
        execution_mode = "sequential (large dataset)"
    elif n_workers == 1:
        execution_mode = "sequential"

    logger.debug(f"Execution mode: {execution_mode}, workers: {n_workers}")

    # Use a picklable worker class instead of a closure
    worker = _OptimizeWorker(data, single_fit_func)

    # Sequential mode with progress tracking
    if n_workers == 1:
        logger.info(f"Running {n_starts} optimizations sequentially")
        results: list[SingleStartResult] = []
        with MultiStartProgressTracker(
            n_starts=n_starts,
            enable_progress_bar=enable_progress_bar,
            verbose=verbose,
        ) as progress:
            for idx, start in enumerate(starts):
                logger.debug(f"Starting optimization {idx + 1}/{n_starts}")
                try:
                    result = worker(idx, start)
                    results.append(result)
                    progress.update(
                        start_idx=idx,
                        success=result.success,
                        chi_squared=result.chi_squared,
                        message=result.message,
                        wall_time=result.wall_time,
                    )
                except (ValueError, RuntimeError, OSError) as e:
                    logger.debug(f"Optimization {idx + 1} raised exception: {e}")
                    failed_result = SingleStartResult(
                        start_idx=idx,
                        initial_params=start,
                        final_params=start,
                        chi_squared=np.inf,
                        success=False,
                        message=str(e),
                    )
                    results.append(failed_result)
                    progress.update(
                        start_idx=idx,
                        success=False,
                        chi_squared=np.inf,
                        message=str(e),
                    )
        return results

    # Parallel mode - progress bar updated as results complete
    logger.info(
        f"Running {n_starts} optimizations in parallel with {n_workers} workers"
    )
    with MultiStartProgressTracker(
        n_starts=n_starts,
        enable_progress_bar=enable_progress_bar,
        verbose=verbose,
    ) as progress:
        results = _run_parallel_with_progress(worker, starts, n_workers, progress)

    return results


def _run_parallel_with_progress(
    optimize_func: Callable[[int, NDArray[np.float64]], SingleStartResult],
    starts: NDArray[np.float64],
    n_workers: int,
    progress: MultiStartProgressTracker,
) -> list[SingleStartResult]:
    """Run optimizations in parallel with progress tracking.

    Uses 'spawn' multiprocessing context to avoid JAX JIT compilation deadlocks.
    Falls back to sequential execution if parallel execution fails or hangs.

    Parameters
    ----------
    optimize_func : Callable
        Function that takes (start_idx, initial_params) and returns SingleStartResult.
    starts : NDArray[np.float64]
        Starting points as (n_starts, n_params) array.
    n_workers : int
        Number of parallel workers.
    progress : MultiStartProgressTracker
        Progress tracker to update.

    Returns
    -------
    list[SingleStartResult]
        Results from all starting points.
    """
    results: list[SingleStartResult] = []
    fallback_to_sequential = False
    fallback_reason = ""

    # Use 'spawn' context to avoid JAX/XLA deadlocks
    # 'fork' can cause issues with JAX's XLA compilation locks
    mp_context = multiprocessing.get_context("spawn")

    try:
        logger.info(
            f"Launching parallel execution: {n_workers} workers, "
            f"{len(starts)} tasks, spawn context"
        )
        logger.debug(f"Worker timeout: {_WORKER_TIMEOUT}s per task")

        parallel_start_time = time.perf_counter()

        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=mp_context
        ) as executor:
            futures = {
                executor.submit(optimize_func, idx, start): idx
                for idx, start in enumerate(starts)
            }
            logger.debug(f"Submitted {len(futures)} tasks to executor")

            # Track completion with timeout
            completed_count = 0
            total_count = len(futures)

            for future in as_completed(futures, timeout=_WORKER_TIMEOUT):
                idx = futures[future]
                try:
                    # Timeout for individual result retrieval
                    result = future.result(timeout=60)
                    results.append(result)
                    completed_count += 1
                    progress.update(
                        start_idx=idx,
                        success=result.success,
                        chi_squared=result.chi_squared,
                        message=result.message,
                        wall_time=result.wall_time,
                    )
                    logger.debug(
                        f"Worker {idx} completed: chi2={result.chi_squared:.4e}, "
                        f"time={result.wall_time:.1f}s ({completed_count}/{total_count})"
                    )
                except TimeoutError:
                    logger.warning(
                        f"Worker {idx} timed out after 60s waiting for result"
                    )
                    logger.info("Falling back to sequential execution")
                    fallback_to_sequential = True
                    fallback_reason = f"Worker {idx} timeout"
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break
                except (
                    ValueError,
                    RuntimeError,
                    TypeError,
                    OSError,
                    AttributeError,
                ) as e:
                    error_msg = str(e)
                    if _is_pickle_error(error_msg):
                        fallback_to_sequential = True
                        fallback_reason = f"Pickle error: {error_msg[:100]}"
                        logger.warning(f"Pickle/serialization error detected: {e}")
                        logger.info("Falling back to sequential execution")
                        for f in futures:
                            f.cancel()
                        break
                    else:
                        # Non-fatal error for this worker
                        logger.warning(f"Worker {idx} failed: {e}")
                        failed_result = SingleStartResult(
                            start_idx=idx,
                            initial_params=starts[idx],
                            final_params=starts[idx],
                            chi_squared=np.inf,
                            success=False,
                            message=str(e),
                        )
                        results.append(failed_result)
                        completed_count += 1
                        progress.update(
                            start_idx=idx,
                            success=False,
                            chi_squared=np.inf,
                            message=str(e),
                        )

        if not fallback_to_sequential:
            parallel_time = time.perf_counter() - parallel_start_time
            logger.info(
                f"Parallel execution complete: {completed_count}/{total_count} tasks "
                f"in {parallel_time:.1f}s"
            )

    except TimeoutError:
        logger.warning(f"Parallel execution timed out after {_WORKER_TIMEOUT}s")
        logger.info("Falling back to sequential execution")
        fallback_to_sequential = True
        fallback_reason = f"Overall timeout ({_WORKER_TIMEOUT}s)"

    except (ValueError, RuntimeError, TypeError, OSError, AttributeError) as e:
        error_msg = str(e)
        if _is_pickle_error(error_msg):
            fallback_to_sequential = True
            fallback_reason = f"Pickle error: {error_msg[:100]}"
            logger.warning(f"ProcessPoolExecutor pickle error: {e}")
        else:
            fallback_to_sequential = True
            fallback_reason = f"Executor error: {error_msg[:100]}"
            logger.warning(f"ProcessPoolExecutor failed: {e}")
        logger.info("Falling back to sequential execution")

    # If parallel failed, fall back to sequential
    if fallback_to_sequential:
        logger.info(f"Sequential fallback reason: {fallback_reason}")
        logger.info(f"Running {len(starts)} optimizations sequentially")
        # Clear any partial results
        results = []
        for idx, start in enumerate(starts):
            logger.debug(f"Starting sequential optimization {idx + 1}/{len(starts)}")
            try:
                result = optimize_func(idx, start)
                results.append(result)
                progress.update(
                    start_idx=idx,
                    success=result.success,
                    chi_squared=result.chi_squared,
                    message=result.message,
                    wall_time=result.wall_time,
                )
            except (ValueError, RuntimeError, OSError) as e:
                logger.debug(f"Sequential optimization {idx + 1} failed: {e}")
                failed_result = SingleStartResult(
                    start_idx=idx,
                    initial_params=start,
                    final_params=start,
                    chi_squared=np.inf,
                    success=False,
                    message=str(e),
                )
                results.append(failed_result)
                progress.update(
                    start_idx=idx,
                    success=False,
                    chi_squared=np.inf,
                    message=str(e),
                )

    # Sort by start_idx for consistent ordering
    results.sort(key=lambda r: r.start_idx)
    return results
