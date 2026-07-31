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

# Shared by NLSQAdapter.fit() (adapter.py) and NLSQWrapper.fit() (wrapper.py) — both reject
# the removed per_angle_scaling=False legacy mode with this exact message.
PER_ANGLE_SCALING_REMOVED_MSG = (
    "per_angle_scaling=False is deprecated and removed. "
    "Use per_angle_scaling=True (default) for physically correct behavior."
)


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
