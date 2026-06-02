"""Custom exceptions for NLSQ optimization.

This module defines a comprehensive exception hierarchy for handling
errors specific to NLSQ optimization, including convergence failures
and numerical instabilities.

The exception hierarchy enables fine-grained error handling and recovery
strategies tailored to specific failure modes.

Exception Hierarchy:
    NLSQOptimizationError (base)
    ├── NLSQConvergenceError (convergence failures)
    └── NLSQNumericalError (NaN/Inf issues)

Examples
--------
Catching specific errors for targeted recovery:

>>> try:
...     result = optimizer.fit(data, model, p0)
... except NLSQNumericalError as e:
...     # Handle NaN/Inf with learning rate reduction
...     result = optimizer.fit(data, model, p0, learning_rate=0.5*lr)
... except NLSQConvergenceError as e:
...     # Handle convergence failure with perturbation
...     p0_perturbed = p0 * (1 + 0.01 * np.random.randn(*p0.shape))
...     result = optimizer.fit(data, model, p0_perturbed)

Using base exception for generic handling:

>>> try:
...     result = optimizer.fit(data, model, p0)
... except NLSQOptimizationError as e:
...     logger.error(f"Optimization failed: {e}")
...     # Fallback to simpler strategy
...     result = use_fallback_strategy()

Notes
-----
All exceptions inherit from `NLSQOptimizationError`, enabling catch-all
error handling while also supporting fine-grained recovery strategies.

The exception messages are designed to be actionable, providing specific
guidance on how to address each type of failure.

See Also
--------
NLSQWrapper : Main optimization wrapper using these exceptions
xpcsjax.optimization.strategy : Strategy selection and fallback logic
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class NLSQOptimizationError(Exception):
    """Base exception for all NLSQ optimization errors.

    This is the base class for all NLSQ-related exceptions. Catching this
    exception will catch all optimization failures regardless of their specific
    cause.

    Attributes
    ----------
    message : str
        Detailed error message
    error_context : dict
        Additional context about the error (parameters, data characteristics, etc.)

    Examples
    --------
    >>> try:
    ...     result = optimizer.fit(data, model, p0)
    ... except NLSQOptimizationError as e:
    ...     print(f"Optimization failed: {e}")
    ...     print(f"Context: {e.error_context}")
    """

    def __init__(self, message: str, error_context: dict | None = None):
        """Initialize base optimization error.

        Parameters
        ----------
        message : str
            Detailed error message
        error_context : dict, optional
            Additional context about the error
        """
        super().__init__(message)
        self.error_context = error_context or {}

    def __str__(self) -> str:
        """Return formatted error message with context."""
        base_msg = super().__str__()
        if self.error_context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.error_context.items())
            return f"{base_msg} (context: {context_str})"
        return base_msg


class NLSQConvergenceError(NLSQOptimizationError):
    """Raised when NLSQ optimization fails to converge.

    This exception indicates that the optimizer could not find a satisfactory
    solution within the specified constraints (maximum iterations, tolerance, etc.).

    Common Causes
    -------------
    - Poor initial guess (p0 too far from optimum)
    - Overly restrictive parameter bounds
    - Insufficient maximum iterations
    - Model function incompatible with data
    - Local minimum trap

    Recovery Strategies
    -------------------
    1. Perturb initial guess: `p0 * (1 + 0.05 * np.random.randn(*p0.shape))`
    2. Relax bounds: Increase parameter search space
    3. Increase max iterations: Allow more optimization steps
    4. Try different optimization method: Switch between 'trf' and 'lm'
    5. Simplify model: Use fewer parameters

    Attributes
    ----------
    iteration_count : int
        Number of iterations completed before failure
    final_loss : float
        Final loss value at termination
    parameters : np.ndarray
        Parameter values at termination

    Examples
    --------
    >>> try:
    ...     result = optimizer.fit(data, model, p0, max_iter=100)
    ... except NLSQConvergenceError as e:
    ...     print(f"Failed after {e.iteration_count} iterations")
    ...     print(f"Final loss: {e.final_loss}")
    ...     # Retry with more iterations
    ...     result = optimizer.fit(data, model, p0, max_iter=500)
    """

    def __init__(
        self,
        message: str,
        iteration_count: int | None = None,
        final_loss: float | None = None,
        parameters: np.ndarray | None = None,
        error_context: dict | None = None,
    ):
        """Initialize convergence error.

        Parameters
        ----------
        message : str
            Detailed error message
        iteration_count : int, optional
            Number of iterations completed
        final_loss : float, optional
            Final loss value
        parameters : np.ndarray, optional
            Parameter values at termination
        error_context : dict, optional
            Additional context
        """
        context = error_context or {}
        if iteration_count is not None:
            context["iteration_count"] = iteration_count
        if final_loss is not None:
            context["final_loss"] = final_loss
        if parameters is not None:
            context["n_params"] = len(parameters)

        super().__init__(message, context)
        self.iteration_count = iteration_count
        self.final_loss = final_loss
        self.parameters = parameters


class NLSQNumericalError(NLSQOptimizationError):
    """Raised for NaN/Inf numerical stability issues.

    This exception indicates that the optimization encountered numerical
    instabilities such as NaN (Not a Number) or Inf (Infinity) values during
    computation.

    Common Causes
    -------------
    - Gradient overflow/underflow
    - Division by zero in model function
    - Exponential overflow in parameters
    - Ill-conditioned Jacobian matrix
    - Learning rate too large

    Detection Points
    ----------------
    1. After gradient computation: `jnp.isfinite(gradients).all()`
    2. After parameter update: `jnp.isfinite(new_params).all()`
    3. After loss calculation: `jnp.isfinite(loss_value)`

    Recovery Strategies
    -------------------
    1. Reduce learning rate: `lr = 0.5 * lr`
    2. Scale data: Normalize inputs to [0, 1] range
    3. Add numerical stability: Use log-transform for exponentials
    4. Check model function: Ensure JAX-compatible operations
    5. Adjust parameter bounds: Prevent extreme values

    Attributes
    ----------
    detection_point : str
        Where NaN/Inf was detected ('gradient', 'parameter', 'loss')
    invalid_values : list
        Description of invalid values found

    Examples
    --------
    >>> try:
    ...     result = optimizer.fit(data, model, p0)
    ... except NLSQNumericalError as e:
    ...     if e.detection_point == 'gradient':
    ...         # Reduce learning rate
    ...         result = optimizer.fit(data, model, p0, learning_rate=0.01)
    ...     elif e.detection_point == 'parameter':
    ...         # Tighten bounds
    ...         bounds = (lower * 0.8, upper * 0.8)
    ...         result = optimizer.fit(data, model, p0, bounds=bounds)
    """

    def __init__(
        self,
        message: str,
        detection_point: str | None = None,
        invalid_values: list | None = None,
        error_context: dict | None = None,
    ):
        """Initialize numerical error.

        Parameters
        ----------
        message : str
            Detailed error message
        detection_point : str, optional
            Where NaN/Inf was detected
        invalid_values : list, optional
            Description of invalid values
        error_context : dict, optional
            Additional context
        """
        context = error_context or {}
        if detection_point:
            context["detection_point"] = detection_point
        if invalid_values:
            context["n_invalid"] = len(invalid_values)

        super().__init__(message, context)
        self.detection_point = detection_point
        self.invalid_values = invalid_values or []
