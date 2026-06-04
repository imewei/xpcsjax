"""Numerical validation for optimization at critical points.

This module provides validation functions to detect numerical issues
(NaN, Inf, bounds violations) at three critical points during optimization:
1. After gradient computation
2. After parameter update
3. After loss calculation

These validations help catch numerical instabilities early and enable
targeted recovery strategies.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np

from xpcsjax.optimization.exceptions import NLSQNumericalError


class NumericalValidator:
    """Validator for numerical stability at critical optimization points.

    This class provides methods to validate numerical values at three
    critical points: gradients, parameters, and loss values. Detection
    of NaN/Inf enables targeted recovery strategies.

    Attributes
    ----------
    enable_validation : bool
        Whether to perform validation (can disable for speed)
    bounds : tuple of np.ndarray or None
        Parameter bounds (lower, upper) for bounds checking

    Examples
    --------
    >>> validator = NumericalValidator(enable_validation=True)
    >>> try:
    ...     validator.validate_gradients(gradients)
    ...     validator.validate_parameters(params, bounds)
    ...     validator.validate_loss(loss_value)
    ... except NLSQNumericalError as e:
    ...     print(f"Numerical error at {e.detection_point}")
    """

    def __init__(
        self,
        enable_validation: bool = True,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        """Initialize numerical validator.

        Parameters
        ----------
        enable_validation : bool, optional
            Whether to perform validation, by default True
        bounds : tuple of np.ndarray, optional
            Parameter bounds (lower, upper), by default None
        """
        self.enable_validation = enable_validation
        self.bounds = bounds

    def validate_gradients(self, gradients: Any) -> None:
        """Validate gradients for NaN/Inf after Jacobian computation.

        This is validation point 1: Gradients can become non-finite due to
        overflow in the model function or ill-conditioned Jacobian.

        Parameters
        ----------
        gradients : array-like
            Gradient values to validate

        Raises
        ------
        NLSQNumericalError
            If gradients contain NaN or Inf values
        """
        if not self.enable_validation:
            return

        # Convert to JAX array for consistent finite check
        grad_array = jnp.asarray(gradients)

        if not jnp.isfinite(grad_array).all():
            # Find indices of invalid values
            invalid_mask = ~jnp.isfinite(grad_array)
            invalid_indices = jnp.where(invalid_mask)[0]

            invalid_values = [
                f"grad[{int(idx)}]={float(grad_array[idx])}"
                for idx in invalid_indices[:5]  # Report first 5
            ]

            raise NLSQNumericalError(
                f"Non-finite gradients detected at {len(invalid_indices)} locations. "
                f"This typically indicates:\n"
                f"  1. Overflow in model function evaluation\n"
                f"  2. Ill-conditioned Jacobian matrix\n"
                f"  3. Learning rate too large\n"
                f"First few invalid values: {invalid_values}",
                detection_point="gradient",
                invalid_values=invalid_values,
                error_context={"n_invalid": int(len(invalid_indices))},
            )

    def validate_parameters(
        self,
        parameters: Any,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Validate parameters for NaN/Inf and bounds violations after update.

        This is validation point 2: Parameters can become non-finite after
        update steps, especially with aggressive step sizes.

        Parameters
        ----------
        parameters : array-like
            Parameter values to validate
        bounds : tuple of np.ndarray, optional
            Parameter bounds (lower, upper), overrides instance bounds

        Raises
        ------
        NLSQNumericalError
            If parameters contain NaN or Inf values
        """
        if not self.enable_validation:
            return

        # Convert to JAX array
        param_array = jnp.asarray(parameters)

        # Check for NaN/Inf
        if not jnp.isfinite(param_array).all():
            invalid_mask = ~jnp.isfinite(param_array)
            invalid_indices = jnp.where(invalid_mask)[0]

            invalid_values = [
                f"param[{int(idx)}]={float(param_array[idx])}" for idx in invalid_indices[:5]
            ]

            raise NLSQNumericalError(
                f"Non-finite parameters detected at {len(invalid_indices)} locations. "
                f"This typically indicates:\n"
                f"  1. Step size too large\n"
                f"  2. Unbounded parameter growth\n"
                f"  3. Numerical overflow in parameter update\n"
                f"First few invalid values: {invalid_values}",
                detection_point="parameter",
                invalid_values=invalid_values,
                error_context={"n_invalid": int(len(invalid_indices))},
            )

        # Check bounds violations if bounds provided
        bounds_to_check = bounds or self.bounds
        if bounds_to_check is not None:
            lower, upper = bounds_to_check
            lower = jnp.asarray(lower)
            upper = jnp.asarray(upper)

            violations_lower = param_array < lower
            violations_upper = param_array > upper

            if jnp.any(violations_lower) or jnp.any(violations_upper):
                n_violations = int(jnp.sum(violations_lower) + jnp.sum(violations_upper))

                # Collect violation indices via vectorized jnp.where, then do a
                # single device-to-host transfer rather than one per element.
                lower_indices = np.asarray(jnp.where(violations_lower)[0])
                upper_indices = np.asarray(jnp.where(violations_upper)[0])
                param_np = np.asarray(param_array)
                lower_np = np.asarray(lower)
                upper_np = np.asarray(upper)

                violation_info = [
                    f"param[{i}]={param_np[i]:.6g} < lower={lower_np[i]:.6g}"
                    for i in lower_indices[:5]
                ] + [
                    f"param[{i}]={param_np[i]:.6g} > upper={upper_np[i]:.6g}"
                    for i in upper_indices[: max(0, 5 - len(lower_indices))]
                ]

                # Note: Bounds violations are often clipped automatically by optimizer
                # This is more of a warning condition than a hard error
                # Could log instead of raising, but raising allows recovery strategies
                raise NLSQNumericalError(
                    f"Parameter bounds violations detected at {n_violations} locations.\n"
                    f"Violations: {violation_info[:5]}",
                    detection_point="parameter_bounds",
                    invalid_values=violation_info[:10],
                    error_context={"n_violations": n_violations},
                )

    def validate_loss(self, loss_value: Any) -> None:
        """Validate loss value for NaN/Inf after loss computation.

        This is validation point 3: Loss can become non-finite due to
        overflow in residual computation or invalid parameter values.

        Parameters
        ----------
        loss_value : scalar
            Loss value to validate

        Raises
        ------
        NLSQNumericalError
            If loss is NaN or Inf
        """
        if not self.enable_validation:
            return

        # Convert to scalar
        loss_scalar = float(loss_value)

        if not np.isfinite(loss_scalar):
            raise NLSQNumericalError(
                f"Non-finite loss value detected: {loss_scalar}. "
                f"This typically indicates:\n"
                f"  1. Overflow in residual computation\n"
                f"  2. Invalid parameter values\n"
                f"  3. Model function returning NaN/Inf\n"
                f"Consider:\n"
                f"  - Reducing step size\n"
                f"  - Tightening parameter bounds\n"
                f"  - Normalizing data\n"
                f"  - Checking model function for numerical issues",
                detection_point="loss",
                invalid_values=[f"loss={loss_scalar}"],
                error_context={"loss_value": loss_scalar},
            )

    def set_bounds(self, bounds: tuple[np.ndarray, np.ndarray]) -> None:
        """Update parameter bounds for validation.

        Parameters
        ----------
        bounds : tuple of np.ndarray
            New parameter bounds (lower, upper)
        """
        self.bounds = bounds

    def disable(self) -> None:
        """Disable validation for performance-critical sections."""
        self.enable_validation = False

    def enable(self) -> None:
        """Re-enable validation after disabling."""
        self.enable_validation = True
