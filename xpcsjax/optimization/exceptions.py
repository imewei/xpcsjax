"""Custom exceptions for NLSQ optimization.

This module defines a comprehensive exception hierarchy for handling
errors specific to NLSQ optimization, including numerical instabilities.

The exception hierarchy enables fine-grained error handling and recovery
strategies tailored to specific failure modes.

Exception Hierarchy:
    NLSQOptimizationError (base)
    └── NLSQNumericalError (NaN/Inf issues)

Examples
--------
Catching specific errors for targeted recovery:

>>> try:
...     result = optimizer.fit(data, model, p0)
... except NLSQNumericalError as e:
...     # Handle NaN/Inf with learning rate reduction
...     result = optimizer.fit(data, model, p0, learning_rate=0.5*lr)

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
