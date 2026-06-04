"""Result Building Utilities for NLSQ Optimization.

This module provides utilities for building and processing optimization results,
extracted from wrapper.py to improve code organization.

Extracted from wrapper.py as part of refactoring (Dec 2025).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for optimization results.

    Attributes:
        chi_squared: Sum of squared residuals
        reduced_chi_squared: chi_squared / degrees of freedom
        quality_flag: 'good', 'marginal', or 'poor'
        n_at_bounds: Number of parameters at bounds
    """

    chi_squared: float
    reduced_chi_squared: float
    quality_flag: str
    n_at_bounds: int = 0


def compute_quality_metrics(
    residuals: np.ndarray,
    n_data: int,
    n_params: int,
    parameter_status: list[str] | None = None,
) -> QualityMetrics:
    """Compute quality metrics from residuals.

    Args:
        residuals: Array of residuals
        n_data: Number of data points
        n_params: Number of parameters
        parameter_status: List of parameter statuses (optional)

    Returns:
        QualityMetrics with computed values
    """
    chi_squared = float(np.sum(residuals**2))
    dof = max(n_data - n_params, 1)  # Avoid division by zero
    reduced_chi_squared = chi_squared / dof

    # Count parameters at bounds
    n_at_bounds = 0
    if parameter_status:
        n_at_bounds = sum(1 for s in parameter_status if s in ("at_lower_bound", "at_upper_bound"))

    # Determine quality flag
    if reduced_chi_squared < 2.0 and n_at_bounds == 0:
        quality_flag = "good"
    elif reduced_chi_squared < 5.0 and n_at_bounds <= 2:
        quality_flag = "marginal"
    else:
        quality_flag = "poor"

    return QualityMetrics(
        chi_squared=chi_squared,
        reduced_chi_squared=reduced_chi_squared,
        quality_flag=quality_flag,
        n_at_bounds=n_at_bounds,
    )


def compute_uncertainties(covariance: np.ndarray) -> np.ndarray:
    """Extract parameter uncertainties from covariance matrix.

    Args:
        covariance: Covariance matrix

    Returns:
        Array of standard deviations (square root of diagonal)
    """
    if covariance is None or covariance.size == 0:
        return np.array([])

    diagonal = np.asarray(np.diag(covariance), dtype=float)

    # Reject non-finite variances (NaN/inf from a singular or failed solve);
    # np.maximum leaves NaN as NaN, so zero them out explicitly first.
    diagonal = np.where(np.isfinite(diagonal), diagonal, 0.0)

    # Handle negative diagonal elements (numerical issues)
    diagonal = np.maximum(diagonal, 0.0)

    return np.asarray(np.sqrt(diagonal))


def normalize_nlsq_result(
    result: Any,
    strategy_name: str = "unknown",
    logger: Any = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Normalize various NLSQ result formats to standard format.

    NLSQ can return results in different formats depending on the function
    and version used. This normalizes them to (popt, pcov, info).

    Args:
        result: NLSQ result in any format
        strategy_name: Name of strategy for logging
        logger: Optional logger

    Returns:
        Tuple of (popt, pcov, info)

    Raises:
        TypeError: If result format is unrecognized
    """
    # Case 1: Dict (from StreamingOptimizer or advanced functions)
    if isinstance(result, dict):
        popt_raw = result.get("x", result.get("popt"))
        if popt_raw is None:
            raise KeyError(
                f"Result dict has neither 'x' nor 'popt' key. Available keys: {list(result.keys())}"
            )
        popt = np.asarray(popt_raw)
        pcov = np.asarray(result.get("pcov", np.eye(len(popt))))
        info = {
            "streaming_diagnostics": result.get("streaming_diagnostics", {}),
            "success": result.get("success", False),
            "message": result.get("message", ""),
            "best_loss": result.get("best_loss", None),
            "final_epoch": result.get("final_epoch", None),
        }
        if logger:
            logger.debug(f"Normalized dict result (strategy: {strategy_name})")
        return popt, pcov, info

    # Case 2: Tuple with 2 or 3 elements
    if isinstance(result, tuple):
        if len(result) == 2:
            popt, pcov = result
            info = {}
            if logger:
                logger.debug(f"Normalized (popt, pcov) tuple (strategy: {strategy_name})")
        elif len(result) == 3:
            popt, pcov, info = result
            if not isinstance(info, dict):
                if logger:
                    logger.warning(f"Info object is not a dict: {type(info)}. Converting to dict.")
                info = {"raw_info": info}
            if logger:
                logger.debug(f"Normalized (popt, pcov, info) tuple (strategy: {strategy_name})")
        else:
            raise TypeError(
                f"Unexpected tuple length: {len(result)}. "
                f"Expected 2 (popt, pcov) or 3 (popt, pcov, info). "
            )
        return np.asarray(popt), np.asarray(pcov), info

    # Case 3: Object with attributes (CurveFitResult, OptimizeResult, etc.)
    if hasattr(result, "x") or hasattr(result, "popt"):
        popt_raw = getattr(result, "x", getattr(result, "popt", None))
        if popt_raw is None:
            raise AttributeError(
                f"Result object has neither 'x' nor 'popt' attribute. "
                f"Available attributes: {dir(result)}"
            )
        popt = np.asarray(popt_raw)

        pcov_raw = getattr(result, "pcov", None)
        if pcov_raw is None:
            _logger = logger or get_logger(__name__)
            _logger.warning("No pcov attribute in result object. Using identity matrix.")
            pcov = np.eye(len(popt))
        else:
            pcov = np.asarray(pcov_raw)

        info = {}
        for attr in ["message", "success", "nfev", "njev", "fun", "jac", "optimality"]:
            if hasattr(result, attr):
                info[attr] = getattr(result, attr)

        if hasattr(result, "info") and isinstance(result.info, dict):
            info.update(result.info)

        if logger:
            logger.debug(
                f"Normalized object result (type: {type(result).__name__}, "
                f"strategy: {strategy_name})"
            )
        return np.asarray(popt), np.asarray(pcov), info

    # Case 4: Unrecognized format
    raise TypeError(
        f"Unrecognized NLSQ result format: {type(result)}. "
        f"Expected tuple, dict, or object with 'x'/'popt' attributes."
    )


def determine_convergence_status(
    info: dict[str, Any],
    quality_metrics: QualityMetrics,
) -> str:
    """Determine convergence status from optimization info.

    Args:
        info: Optimization info dict
        quality_metrics: Quality metrics

    Returns:
        Convergence status: 'converged', 'max_iter', or 'failed'
    """
    # Check explicit success flag
    if "success" in info:
        if info["success"]:
            return "converged"
        # Check for max iterations
        message = str(info.get("message", "")).lower()
        if "max" in message and ("iter" in message or "fev" in message):
            return "max_iter"
        return "failed"

    # Infer from quality (no explicit success flag available)
    if quality_metrics.reduced_chi_squared < 5.0:
        logger.warning(
            f"Convergence inferred from reduced_chi_squared="
            f"{quality_metrics.reduced_chi_squared:.4f} < 5.0 "
            f"(no explicit success flag in optimizer info)"
        )
        return "converged"

    return "failed"


@dataclass
class ResultBuilder:
    """Builder for constructing OptimizationResult objects.

    Provides a fluent interface for building results with proper validation.
    """

    parameters: np.ndarray | None = None
    covariance: np.ndarray | None = None
    n_data: int = 0
    start_time: float = field(default_factory=time.time)
    recovery_actions: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    stratification_diagnostics: Any = None
    nlsq_diagnostics: dict[str, Any] | None = None

    def with_parameters(self, params: np.ndarray) -> ResultBuilder:
        """Set optimized parameters."""
        self.parameters = np.asarray(params)
        return self

    def with_covariance(self, cov: np.ndarray) -> ResultBuilder:
        """Set parameter covariance matrix."""
        self.covariance = np.asarray(cov)
        return self

    def with_data_size(self, n_data: int) -> ResultBuilder:
        """Set number of data points."""
        self.n_data = n_data
        return self

    def with_start_time(self, start_time: float) -> ResultBuilder:
        """Set optimization start time."""
        self.start_time = start_time
        return self

    def with_recovery_actions(self, actions: list[str]) -> ResultBuilder:
        """Set recovery actions taken."""
        self.recovery_actions = actions
        return self

    def with_info(self, info: dict[str, Any]) -> ResultBuilder:
        """Set optimization info dict."""
        self.info = info
        return self

    def with_stratification_diagnostics(self, diags: Any) -> ResultBuilder:
        """Set stratification diagnostics."""
        self.stratification_diagnostics = diags
        return self

    def with_nlsq_diagnostics(self, diags: dict[str, Any]) -> ResultBuilder:
        """Set NLSQ solver diagnostics."""
        self.nlsq_diagnostics = diags
        return self

    def with_fourier_covariance_transform(
        self,
        fourier_reparameterizer: Any,
        n_phi: int,
        n_physical: int,
    ) -> ResultBuilder:
        """Transform covariance from Fourier to per-angle space.

        T037-T039: Implements Fourier→per-angle covariance transformation.

        The transformation uses the Jacobian of the Fourier→per-angle mapping:
            Cov_per_angle = J @ Cov_fourier @ J.T

        Physical parameter covariance is preserved (not transformed).

        Parameters
        ----------
        fourier_reparameterizer : FourierReparameterizer
            The Fourier reparameterizer used during optimization.
        n_phi : int
            Number of phi angles.
        n_physical : int
            Number of physical parameters.

        Returns
        -------
        ResultBuilder
            Self for method chaining.

        Notes
        -----
        If covariance is None or fourier_reparameterizer is None,
        this method is a no-op.
        """
        if self.covariance is None or fourier_reparameterizer is None:
            return self

        if not fourier_reparameterizer.use_fourier:
            return self

        # Get Jacobian for Fourier→per-angle transformation
        # J has shape (n_per_angle, n_fourier_coeffs)
        try:
            jacobian = fourier_reparameterizer.get_jacobian_transform()
        except AttributeError:
            # FourierReparameterizer doesn't have get_jacobian_transform
            # Fall back to no transformation
            return self

        n_fourier_coeffs = fourier_reparameterizer.n_coeffs_per_param
        n_fourier_total = 2 * n_fourier_coeffs
        n_per_angle_total = 2 * n_phi

        # T039: Validate dimensions
        cov_shape = self.covariance.shape
        expected_fourier_dim = n_fourier_total + n_physical
        if cov_shape[0] != expected_fourier_dim:
            # Covariance dimensions don't match expected Fourier space
            return self

        # Build block-diagonal Jacobian for full parameter vector
        # [J_contrast  0        0      ]
        # [0          J_offset  0      ]
        # [0          0         I_phys ]
        full_jacobian = np.zeros((n_per_angle_total + n_physical, n_fourier_total + n_physical))

        # Contrast block: J (n_phi × n_fourier_coeffs)
        full_jacobian[:n_phi, :n_fourier_coeffs] = jacobian

        # Offset block: J (n_phi × n_fourier_coeffs)
        full_jacobian[n_phi:n_per_angle_total, n_fourier_coeffs:n_fourier_total] = jacobian

        # T038: Physical block: Identity (preserve physical covariance)
        full_jacobian[n_per_angle_total:, n_fourier_total:] = np.eye(n_physical)

        # Transform: Cov_per_angle = J @ Cov_fourier @ J.T
        transformed_cov = full_jacobian @ self.covariance @ full_jacobian.T

        # T039: Validate output dimensions
        expected_dim = n_per_angle_total + n_physical
        if transformed_cov.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Transformed covariance has wrong dimensions: "
                f"{transformed_cov.shape}, expected ({expected_dim}, {expected_dim})"
            )

        self.covariance = transformed_cov
        return self

    def build(
        self,
        residual_fn: Any = None,
        xdata: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Build the result dictionary.

        Args:
            residual_fn: Residual function for computing chi-squared
            xdata: X data for residual computation

        Returns:
            Dictionary with all result fields

        Raises:
            ValueError: If required fields are missing
        """
        if self.parameters is None:
            raise ValueError("Parameters must be set before building result")

        n_params = len(self.parameters)
        execution_time = time.time() - self.start_time

        # Compute uncertainties
        if self.covariance is not None:
            uncertainties = compute_uncertainties(self.covariance)
        else:
            uncertainties = np.zeros(n_params)

        # Compute quality metrics
        if residual_fn is not None and xdata is not None:
            try:
                residuals = residual_fn(xdata, *self.parameters)
                quality = compute_quality_metrics(residuals, self.n_data, n_params)
            except (ValueError, RuntimeError, TypeError):
                # Fallback if residual computation fails.
                # NLSQ/scipy least_squares stores cost = 0.5*RSS as "fun"
                # (internal convention).  Multiply by 2 to get chi-squared.
                # NOTE: OOC and hierarchical paths use different keys
                # ("chi_squared", "final_cost") and don't set "fun", so this
                # defaults to 0.0 for those callers (acceptable — they supply
                # residual_fn and hit the primary path above).
                fun_val = float(self.info.get("fun", 0.0))
                chi_sq_fallback = fun_val * 2.0
                quality = QualityMetrics(
                    chi_squared=chi_sq_fallback,
                    reduced_chi_squared=chi_sq_fallback / max(self.n_data - n_params, 1),
                    quality_flag="unknown",
                )
        else:
            # Use info from optimizer.
            # NLSQ/scipy least_squares stores cost = 0.5*RSS as "fun".
            # Multiply by 2 to get chi-squared.  See note above re: OOC/hier.
            chi_sq = float(self.info.get("fun", 0.0)) * 2.0
            quality = QualityMetrics(
                chi_squared=chi_sq,
                reduced_chi_squared=chi_sq / max(self.n_data - n_params, 1),
                quality_flag="unknown",
            )

        # Determine convergence status
        convergence_status = determine_convergence_status(self.info, quality)

        return {
            "parameters": self.parameters,
            "uncertainties": uncertainties,
            "covariance": self.covariance if self.covariance is not None else np.eye(n_params),
            "chi_squared": quality.chi_squared,
            "reduced_chi_squared": quality.reduced_chi_squared,
            "convergence_status": convergence_status,
            "iterations": int(self.info.get("nfev", 0)),
            "execution_time": execution_time,
            "device_info": {"type": "cpu", "name": "CPU"},
            "recovery_actions": self.recovery_actions,
            "quality_flag": quality.quality_flag,
            "stratification_diagnostics": self.stratification_diagnostics,
            "nlsq_diagnostics": self.nlsq_diagnostics,
        }
