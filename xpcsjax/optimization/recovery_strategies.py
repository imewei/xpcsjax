"""Error recovery strategies for NLSQ optimization failures.

This module defines error-specific recovery strategies that can be applied
when optimization encounters failures. Each error type has a prioritized
list of recovery actions to attempt.
"""

from typing import Any

import numpy as np

from xpcsjax.optimization.exceptions import NLSQConvergenceError, NLSQNumericalError

# Error-specific recovery strategies
# Each error type maps to a list of (strategy_name, strategy_param) tuples
ERROR_RECOVERY_STRATEGIES: dict[type[Exception], list[tuple[str, Any]]] = {
    NLSQConvergenceError: [
        ("perturb_parameters", 0.05),  # 5% random perturbation
        ("increase_iterations", 1.5),  # 50% more iterations
        ("relax_tolerance", 10.0),  # 10x tolerance relaxation
    ],
    NLSQNumericalError: [
        ("reduce_step_size", 0.5),  # Halve step size
        ("tighten_bounds", 0.9),  # 10% tighter bounds
        ("rescale_data", "normalize"),  # Normalize to [0, 1]
    ],
}


class RecoveryStrategyApplicator:
    """Apply recovery strategies for optimization failures.

    This class implements various recovery strategies that can be applied
    when optimization fails. Strategies are error-type specific and are
    applied in a prioritized order.

    Parameters
    ----------
    max_retries : int, optional
        Maximum number of retry attempts per batch, by default 2

    Examples
    --------
    >>> applicator = RecoveryStrategyApplicator(max_retries=2)
    >>> error = NLSQConvergenceError("Failed to converge")
    >>> strategy_name, modified_params = applicator.get_recovery_strategy(
    ...     error, params, attempt=0
    ... )
    >>> # strategy_name is "perturb_parameters"
    >>> # modified_params has 5% random noise added
    """

    def __init__(self, max_retries: int = 2, seed: int = 42):
        """Initialize recovery strategy applicator.

        Parameters
        ----------
        max_retries : int, optional
            Maximum retry attempts, by default 2
        seed : int, optional
            RNG seed for reproducible perturbations, by default 42
        """
        self.max_retries = max_retries
        self._rng = np.random.default_rng(seed)

    def get_recovery_strategy(
        self,
        error: Exception,
        params: np.ndarray,
        attempt: int,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[str, np.ndarray] | None:
        """Get recovery strategy for the given error and attempt.

        Parameters
        ----------
        error : Exception
            The exception that was raised
        params : np.ndarray
            Current parameter values
        attempt : int
            Retry attempt number (0-indexed)
        bounds : tuple of np.ndarray, optional
            Parameter bounds (lower, upper), by default None

        Returns
        -------
        tuple of (str, np.ndarray) or None
            (strategy_name, modified_params) if strategy available, else None
        """
        error_type = type(error)
        strategies = ERROR_RECOVERY_STRATEGIES.get(error_type, [])

        if attempt >= len(strategies):
            return None  # No more strategies available

        strategy_name, strategy_param = strategies[attempt]

        # Apply the strategy
        modified_params = self._apply_strategy(
            strategy_name,
            params,
            strategy_param,
            bounds,
        )

        return strategy_name, modified_params

    def _apply_strategy(
        self,
        strategy_name: str,
        params: np.ndarray,
        strategy_param: Any,
        bounds: tuple[np.ndarray, np.ndarray] | None,
    ) -> np.ndarray:
        """Apply the specified recovery strategy.

        Parameters
        ----------
        strategy_name : str
            Name of strategy to apply
        params : np.ndarray
            Current parameter values
        strategy_param : Any
            Strategy-specific parameter
        bounds : tuple of np.ndarray or None
            Parameter bounds

        Returns
        -------
        np.ndarray
            Modified parameters after applying strategy
        """
        if strategy_name == "perturb_parameters":
            return self._perturb_parameters(params, strategy_param)

        elif strategy_name == "increase_iterations":
            # This strategy doesn't modify params, just a flag
            # Return params unchanged; caller should increase max_iter
            return params.copy()

        elif strategy_name == "relax_tolerance":
            # This strategy doesn't modify params
            # Return params unchanged; caller should relax tolerance
            return params.copy()

        elif strategy_name == "reduce_step_size":
            # This strategy affects optimizer settings, not params
            return params.copy()

        elif strategy_name == "tighten_bounds":
            if bounds is not None:
                # Ensure params stay within tightened bounds
                lower, upper = bounds
                center = (upper + lower) / 2
                range_width = (upper - lower) * strategy_param
                new_lower = center - range_width / 2
                new_upper = center + range_width / 2
                result: np.ndarray = np.clip(params, new_lower, new_upper)
                return result
            return params.copy()

        elif strategy_name == "rescale_data":
            # This strategy affects data, not params
            return params.copy()

        else:
            # Unknown strategy, return params unchanged
            return params.copy()

    def _perturb_parameters(
        self,
        params: np.ndarray,
        perturbation_fraction: float,
    ) -> np.ndarray:
        """Add random perturbation to parameters.

        Parameters
        ----------
        params : np.ndarray
            Current parameter values
        perturbation_fraction : float
            Fraction of parameter value to use as perturbation scale

        Returns
        -------
        np.ndarray
            Perturbed parameters
        """
        perturbation = self._rng.standard_normal(params.shape) * perturbation_fraction
        # Additive fallback for zero-valued params (multiplicative would leave them at zero)
        scale = np.where(np.abs(params) > 1e-30, np.abs(params), 1.0)
        perturbed = params + perturbation * scale
        return np.asarray(perturbed)

    def should_retry(self, attempt: int) -> bool:
        """Check if another retry attempt should be made.

        Parameters
        ----------
        attempt : int
            Current attempt number (0-indexed)

        Returns
        -------
        bool
            True if should retry, False if max retries exhausted
        """
        return attempt < self.max_retries
