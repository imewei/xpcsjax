"""Abstract base class for NLSQ adapters (FR-012).

Provides shared methods for NLSQAdapter and NLSQWrapper to reduce code duplication.

Created as part of architecture refactoring (T059-T061).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np

from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


class NLSQAdapterBase(ABC):
    """Abstract base class for NLSQ optimization adapters.

    Provides shared methods for data preparation, validation, result building,
    error handling, bounds setup, and covariance computation.

    Subclasses must implement the `fit()` method.
    """

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        """Fit the model to data.

        Must be implemented by subclasses.
        """
        ...

    def _prepare_data(
        self,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        g2: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Prepare input data for optimization.

        Parameters
        ----------
        t1 : np.ndarray
            First time coordinates
        t2 : np.ndarray
            Second time coordinates
        phi : np.ndarray
            Angle coordinates
        g2 : np.ndarray
            g2 correlation values
        weights : np.ndarray | None, optional
            Optional weights for weighted least squares

        Returns
        -------
        dict[str, Any]
            Prepared data structure with keys:
            - 't1': validated t1 array
            - 't2': validated t2 array
            - 'phi': validated phi array
            - 'g2': validated g2 array
            - 'weights': weights or None
            - 'n_points': number of data points
            - 'phi_unique': unique phi values
            - 'n_phi': number of unique phi values
        """
        # Convert to numpy arrays
        t1 = np.asarray(t1, dtype=np.float64)
        t2 = np.asarray(t2, dtype=np.float64)
        phi = np.asarray(phi, dtype=np.float64)
        g2 = np.asarray(g2, dtype=np.float64)

        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)

        # Get unique phi values
        phi_unique = np.unique(phi)
        n_phi = len(phi_unique)

        return {
            "t1": t1,
            "t2": t2,
            "phi": phi,
            "g2": g2,
            "weights": weights,
            "n_points": len(t1),
            "phi_unique": phi_unique,
            "n_phi": n_phi,
        }

    def _validate_input(
        self,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        g2: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        """Validate input arrays for consistency.

        Parameters
        ----------
        t1 : np.ndarray
            First time coordinates
        t2 : np.ndarray
            Second time coordinates
        phi : np.ndarray
            Angle coordinates
        g2 : np.ndarray
            g2 correlation values
        weights : np.ndarray | None, optional
            Optional weights for weighted least squares

        Raises
        ------
        ValueError
            If arrays have inconsistent shapes or invalid values
        """
        # Check array lengths match
        n = len(t1)
        if len(t2) != n:
            raise ValueError(f"t1 and t2 must have same length: {n} vs {len(t2)}")
        if len(phi) != n:
            raise ValueError(f"t1 and phi must have same length: {n} vs {len(phi)}")
        if len(g2) != n:
            raise ValueError(f"t1 and g2 must have same length: {n} vs {len(g2)}")

        if weights is not None and len(weights) != n:
            raise ValueError(f"weights must have same length as data: {n} vs {len(weights)}")

        # Check for NaN/Inf values
        if np.any(~np.isfinite(t1)):
            raise ValueError("t1 contains NaN or Inf values")
        if np.any(~np.isfinite(t2)):
            raise ValueError("t2 contains NaN or Inf values")
        if np.any(~np.isfinite(phi)):
            raise ValueError("phi contains NaN or Inf values")
        if np.any(~np.isfinite(g2)):
            raise ValueError("g2 contains NaN or Inf values")

        if weights is not None and np.any(~np.isfinite(weights)):
            raise ValueError("weights contains NaN or Inf values")

        # Check for empty arrays
        if n == 0:
            raise ValueError("Input arrays cannot be empty")

        logger.debug(f"Input validation passed: {n} points")

    def _build_result(
        self,
        params: np.ndarray,
        chi_squared: float,
        covariance: np.ndarray | None,
        param_names: list[str],
        n_iter: int,
        success: bool,
        message: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build standardized result dictionary.

        Parameters
        ----------
        params : np.ndarray
            Optimized parameter values
        chi_squared : float
            Final chi-squared value
        covariance : np.ndarray | None
            Parameter covariance matrix
        param_names : list[str]
            Parameter names
        n_iter : int
            Number of iterations
        success : bool
            Whether optimization succeeded
        message : str
            Status message
        diagnostics : dict[str, Any] | None, optional
            Additional diagnostics

        Returns
        -------
        dict[str, Any]
            Standardized result dictionary
        """
        result = {
            "params": params,
            "chi_squared": chi_squared,
            "covariance": covariance,
            "param_names": param_names,
            "n_iter": n_iter,
            "success": success,
            "message": message,
        }

        if diagnostics is not None:
            result["diagnostics"] = diagnostics

        # Compute parameter uncertainties from covariance
        if covariance is not None:
            try:
                uncertainties = np.sqrt(np.diag(covariance))
                result["uncertainties"] = uncertainties
            except (ValueError, np.linalg.LinAlgError):
                result["uncertainties"] = None
        else:
            result["uncertainties"] = None

        return result

    def _handle_error(
        self,
        error: Exception,
        context: str,
        params: np.ndarray | None = None,
        raise_on_error: bool = False,
    ) -> dict[str, Any] | None:
        """Handle optimization errors gracefully.

        Parameters
        ----------
        error : Exception
            The exception that occurred
        context : str
            Context description for logging
        params : np.ndarray | None, optional
            Current parameter values at time of error
        raise_on_error : bool, optional
            Whether to re-raise the error after logging

        Returns
        -------
        dict[str, Any] | None
            Error result dictionary if not raising, None otherwise

        Raises
        ------
        Exception
            Re-raises original error if raise_on_error is True
        """
        logger.error(f"Error in {context}: {error}")

        if raise_on_error:
            raise error

        return {
            "params": params,
            "chi_squared": np.inf,
            "covariance": None,
            "param_names": [],
            "n_iter": 0,
            "success": False,
            "message": f"Error: {error}",
            "error": str(error),
        }

    def _setup_bounds(
        self,
        param_names: list[str],
        bounds_dict: dict[str, tuple[float, float]] | None = None,
        default_bounds: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Set up parameter bounds arrays.

        Parameters
        ----------
        param_names : list[str]
            Parameter names
        bounds_dict : dict[str, tuple[float, float]] | None, optional
            Dictionary mapping parameter names to (lower, upper) bounds
        default_bounds : tuple[float, float] | None, optional
            Default bounds if not specified (defaults to (-inf, inf))

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (lower_bounds, upper_bounds) arrays
        """
        n_params = len(param_names)

        if default_bounds is None:
            default_bounds = (-np.inf, np.inf)

        lower = np.full(n_params, default_bounds[0])
        upper = np.full(n_params, default_bounds[1])

        if bounds_dict is not None:
            for i, name in enumerate(param_names):
                if name in bounds_dict:
                    lb, ub = bounds_dict[name]
                    lower[i] = lb
                    upper[i] = ub

        logger.debug(f"Bounds setup: {n_params} parameters")
        return lower, upper

    def _compute_covariance(
        self,
        jacobian: np.ndarray,
        residuals: np.ndarray,
        n_params: int,
    ) -> np.ndarray | None:
        """Compute parameter covariance matrix from Jacobian.

        Uses the standard formula: cov = (J^T J)^{-1} * s^2
        where s^2 = sum(residuals^2) / (n - p)

        Parameters
        ----------
        jacobian : np.ndarray
            Jacobian matrix (n_points x n_params)
        residuals : np.ndarray
            Residual vector
        n_params : int
            Number of parameters

        Returns
        -------
        np.ndarray | None
            Covariance matrix or None if computation fails
        """
        try:
            n_points = len(residuals)
            dof = n_points - n_params

            if dof <= 0:
                logger.warning(
                    f"Insufficient degrees of freedom: {n_points} points, {n_params} params"
                )
                return None

            # Compute J^T J
            jtj = jacobian.T @ jacobian

            # Check condition number for numerical stability
            cond = np.linalg.cond(jtj)
            if cond > 1e12:
                logger.warning(f"J^T J ill-conditioned (cond={cond:.2e}), using SVD")
                # Use pseudo-inverse for ill-conditioned case
                jtj_inv = np.linalg.pinv(jtj)
            else:
                jtj_inv = np.linalg.inv(jtj)

            # Compute variance estimate
            s2 = np.sum(residuals**2) / dof

            # Covariance matrix
            covariance = jtj_inv * s2

            return cast(np.ndarray, covariance)

        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Covariance computation failed: {e}")
            return None


__all__ = ["NLSQAdapterBase"]
