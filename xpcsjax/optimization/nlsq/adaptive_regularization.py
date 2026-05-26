"""Adaptive Relative Regularization for Anti-Degeneracy Defense.

This module implements CV-based (Coefficient of Variation) regularization
that scales properly with data, replacing the ineffective absolute variance
regularization.

Part of Anti-Degeneracy Defense System v2.9.0.
See: docs/specs/anti-degeneracy-defense-v2.9.0.md

Mathematical Formulation::

    Current (ineffective):
        L_reg = lambda * Var(params) * n_points

    Proposed (CV-based):
        CV = std(params) / abs(mean(params))
        L_reg = lambda * CV^2 * MSE * n_points

    Auto-tuned lambda:
        lambda = target_contribution / target_cv^2

        Example: Allow 10% variation (CV=0.1), contribute 10% to loss
        lambda = 0.1 / 0.01 = 10
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import jax.numpy as jnp
import numpy as np

from xpcsjax.optimization.nlsq.config import safe_float
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AdaptiveRegularizationConfig:
    """Configuration for adaptive relative regularization.

    Attributes
    ----------
    enable : bool
        Whether to enable regularization. Default True.
    mode : str
        Regularization mode: "absolute", "relative", or "auto".
        - "absolute": Original variance-based (L_reg = λ × Var × n)
        - "relative": CV-based (L_reg = λ × CV² × MSE × n)
        - "auto": Use relative for n_phi > 5, absolute otherwise
    lambda_base : float
        Base regularization strength. Default 1.0 (100× stronger than v2.8).
    target_cv : float
        Target coefficient of variation. Default 0.10 (10% variation allowed).
    target_contribution : float
        Target fraction of MSE to contribute. Default 0.10 (10% of loss).
    auto_tune_lambda : bool
        Whether to auto-compute λ from target_cv and target_contribution.
    max_cv : float
        Maximum allowed CV before hard constraint warning. Default 0.20.
    group_indices : list of tuple, optional
        Parameter group indices [(start, end), ...]. Auto-computed if None.
    """

    enable: bool = True
    mode: Literal["absolute", "relative", "auto"] = "relative"

    # Relative (CV-based) mode settings
    lambda_base: float = 1.0  # 100× stronger than v2.8 default of 0.01
    target_cv: float = 0.10  # 10% variation target
    target_contribution: float = 0.10  # 10% of MSE contribution

    # Auto-tune λ based on target_cv and target_contribution
    auto_tune_lambda: bool = True

    # Maximum allowed CV before hard constraint kicks in
    max_cv: float = 0.20  # 20% max variation

    # Group indices (auto-computed if None)
    group_indices: list[tuple[int, int]] | None = None

    @classmethod
    def from_dict(cls, config_dict: dict) -> AdaptiveRegularizationConfig:
        """Create config from dictionary with safe type conversion."""
        return cls(
            enable=bool(config_dict.get("enable", True)),
            mode=cast(
                Literal["absolute", "relative", "auto"],
                config_dict.get("mode", "relative"),
            ),
            lambda_base=safe_float(config_dict.get("lambda"), 1.0),
            target_cv=safe_float(config_dict.get("target_cv"), 0.10),
            target_contribution=safe_float(
                config_dict.get("target_contribution"), 0.10
            ),
            auto_tune_lambda=bool(config_dict.get("auto_tune_lambda", True)),
            max_cv=safe_float(config_dict.get("max_cv"), 0.20),
            group_indices=config_dict.get("group_indices"),
        )


class AdaptiveRegularizer:
    """CV-based adaptive regularization for per-angle parameters.

    This regularizer addresses the fundamental problem where absolute variance
    regularization (λ=0.01) contributed only ~0.05% to total loss, providing
    no effective constraint on per-angle parameter variation.

    The CV-based approach ensures regularization scales properly:
    - CV is dimensionless (ratio of std to mean)
    - Auto-tuned λ makes regularization ~10% of MSE
    - Prevents per-angle parameters from absorbing physical signals

    Parameters
    ----------
    config : AdaptiveRegularizationConfig
        Regularization configuration.
    n_phi : int
        Number of unique phi angles.

    Attributes
    ----------
    lambda_value : float
        Effective regularization strength (auto-tuned or from config).
    group_indices : list of tuple
        Parameter groups to regularize.

    Examples
    --------
    >>> config = AdaptiveRegularizationConfig(target_cv=0.10, target_contribution=0.10)
    >>> regularizer = AdaptiveRegularizer(config, n_phi=23)
    >>> reg_term = regularizer.compute_regularization(
    ...     params, mse=0.04, n_points=23_000_000
    ... )
    """

    def __init__(
        self,
        config: AdaptiveRegularizationConfig,
        n_phi: int,
        n_params: int | None = None,
    ):
        """Initialize adaptive regularizer.

        Parameters
        ----------
        config : AdaptiveRegularizationConfig
            Regularization configuration.
        n_phi : int
            Number of unique phi angles.
        n_params : int, optional
            Actual parameter vector length. When provided and less than
            2 * n_phi + n_physical, auto_averaged mode is assumed
            (2 scaling params instead of 2 * n_phi).
        """
        self.config = config
        self.n_phi = n_phi

        # Auto-compute group indices if not provided
        if config.group_indices is None:
            # For auto_averaged mode, the parameter vector has only 2 scaling
            # params (1 contrast_avg + 1 offset_avg) instead of 2*n_phi.
            # Detect this by checking if n_params < 2*n_phi.
            if n_params is not None and n_params < 2 * n_phi:
                # auto_averaged: indices [0,1) for contrast_avg, [1,2) for offset_avg
                self.group_indices = [
                    (0, 1),  # contrast_avg group
                    (1, 2),  # offset_avg group
                ]
            else:
                self.group_indices = [
                    (0, n_phi),  # contrast group
                    (n_phi, 2 * n_phi),  # offset group
                ]
        else:
            self.group_indices = list(config.group_indices)

        # Auto-tune λ if enabled
        if config.auto_tune_lambda and config.target_cv > 0:
            self.lambda_value = config.target_contribution / (config.target_cv**2)
            logger.debug(
                f"Auto-tuned lambda = {self.lambda_value:.2f} "
                f"(target_cv={config.target_cv}, "
                f"contribution={config.target_contribution})"
            )
        else:
            self.lambda_value = config.lambda_base

        # Diagnostics
        self._last_cv_values: dict[int, float] = {}
        self._last_reg_contribution: float = 0.0

    def compute_regularization(
        self, params: np.ndarray, mse: float, n_points: int
    ) -> float:
        """Compute regularization term to add to loss.

        Parameters
        ----------
        params : np.ndarray
            Full parameter vector.
        mse : float
            Current mean squared error.
        n_points : int
            Number of data points.

        Returns
        -------
        float
            Regularization term to add to loss (SSE scale).
        """
        if not self.config.enable:
            return 0.0

        # Guard against NaN/inf params from a failed solver step. Returning 0.0
        # here would *remove* the stabilizing penalty at exactly the moment it
        # is most needed and let the diverged step pass through silently.
        # Returning +inf instead makes the augmented loss unambiguously bad so
        # the trust-region step is rejected (loss_augmentation is a scalar
        # penalty added to the loss — see anti_degeneracy_controller).
        if not np.all(np.isfinite(params)):
            logger.warning(
                "compute_regularization received non-finite params (diverged "
                "solver step); returning +inf to force trust-region step rejection."
            )
            return np.inf

        total_reg = 0.0
        self._last_cv_values = {}

        for group_idx, (start, end) in enumerate(self.group_indices):
            if start >= len(params) or end > len(params):
                logger.warning(
                    f"Group indices ({start}, {end}) out of bounds "
                    f"for params length {len(params)}"
                )
                continue

            group_params = params[start:end]

            # Security: Validate n_group to prevent division by zero
            n_group = end - start
            if n_group < 2:
                logger.warning(
                    f"Group ({start}, {end}) has fewer than 2 elements, skipping regularization"
                )
                continue

            if self.config.mode == "relative" or (
                self.config.mode == "auto" and self.n_phi > 5
            ):
                # CV-based regularization (NaN-safe: degenerate optimizer steps can yield NaN params)
                mean_val = np.nanmean(group_params)
                std_val = np.nanstd(group_params)

                if abs(mean_val) > 1e-10:
                    cv = std_val / abs(mean_val)
                else:
                    cv = std_val  # Fallback to absolute std

                self._last_cv_values[group_idx] = cv

                # L_reg = lambda x CV^2 x MSE x n_points
                group_reg = self.lambda_value * (cv**2) * mse * n_points

            else:
                # Original absolute variance
                var_val = np.nanvar(group_params)
                group_reg = self.lambda_value * var_val * n_points

                # Still compute CV for diagnostics
                mean_val = np.nanmean(group_params)
                std_val = np.nanstd(group_params)
                if abs(mean_val) > 1e-10:
                    self._last_cv_values[group_idx] = std_val / abs(mean_val)

            total_reg += group_reg

        self._last_reg_contribution = total_reg
        return total_reg

    def compute_regularization_jax(
        self, params: jnp.ndarray, mse: jnp.ndarray, n_points: int
    ) -> jnp.ndarray:
        """Compute regularization term using JAX for autodiff compatibility.

        This method uses JAX operations (jnp) instead of NumPy, making it
        compatible with JAX's JIT compilation and autodiff (jax.grad).

        Use this method when the regularization needs to be part of a
        differentiable loss function.

        Parameters
        ----------
        params : jnp.ndarray
            Full parameter vector (JAX array, possibly traced).
        mse : jnp.ndarray
            Current mean squared error (JAX scalar, possibly traced).
        n_points : int
            Number of data points.

        Returns
        -------
        jnp.ndarray
            Regularization term to add to loss (SSE scale, JAX scalar).
        """
        if not self.config.enable:
            return jnp.array(0.0)

        total_reg = jnp.array(0.0)

        for _group_idx, (start, end) in enumerate(self.group_indices):
            if start >= len(params) or end > len(params):
                continue

            group_params = params[start:end]

            # Security: Validate n_group to prevent division by zero
            n_group = end - start
            if n_group < 2:
                continue

            if self.config.mode == "relative" or (
                self.config.mode == "auto" and self.n_phi > 5
            ):
                # CV-based regularization using JAX operations
                mean_val = jnp.mean(group_params)
                std_val = jnp.std(group_params)

                # Use jnp.where for safe division (avoids conditional on traced value)
                cv = jnp.where(
                    jnp.abs(mean_val) > 1e-10,
                    std_val / jnp.abs(mean_val),
                    std_val,  # Fallback to absolute std
                )

                # L_reg = λ × CV² × MSE × n_points
                group_reg = self.lambda_value * (cv**2) * mse * n_points

            else:
                # Original absolute variance using JAX operations
                var_val = jnp.var(group_params)
                group_reg = self.lambda_value * var_val * n_points

            total_reg = total_reg + group_reg

        return total_reg

    def compute_regularization_gradient(
        self, params: np.ndarray, mse: float, n_points: int
    ) -> np.ndarray:
        """Compute gradient of regularization term.

        Parameters
        ----------
        params : np.ndarray
            Full parameter vector.
        mse : float
            Current mean squared error.
        n_points : int
            Number of data points.

        Returns
        -------
        np.ndarray
            Gradient w.r.t. all parameters (zeros for non-regularized params).
        """
        grad = np.zeros_like(params, dtype=np.float64)

        if not self.config.enable:
            return grad

        for start, end in self.group_indices:
            if start >= len(params) or end > len(params):
                continue

            group_params = params[start:end]
            n_group = end - start

            # Security: Validate n_group to prevent division by zero
            if n_group < 2:
                continue

            mean_val = np.nanmean(group_params)
            std_val = np.nanstd(group_params)

            if self.config.mode == "relative" or (
                self.config.mode == "auto" and self.n_phi > 5
            ):
                # CV-based gradient
                if abs(mean_val) > 1e-10 and std_val > 1e-10:
                    cv = std_val / abs(mean_val)

                    # ∂CV²/∂p_i = 2CV × ∂CV/∂p_i
                    # ∂CV/∂p_i = ∂(std/mean)/∂p_i
                    #          = (∂std/∂p_i × mean - std × ∂mean/∂p_i) / mean²
                    # ∂std/∂p_i = (p_i - mean) / (n × std)
                    # ∂mean/∂p_i = 1/n

                    for i, p_i in enumerate(group_params):
                        d_std = (p_i - mean_val) / (n_group * std_val)
                        d_mean = 1.0 / n_group

                        # Handle sign of mean
                        if mean_val >= 0:
                            d_cv = (d_std * mean_val - std_val * d_mean) / (mean_val**2)
                        else:
                            d_cv = (d_std * (-mean_val) - std_val * (-d_mean)) / (
                                mean_val**2
                            )

                        d_cv_sq = 2 * cv * d_cv

                        grad[start + i] = self.lambda_value * d_cv_sq * mse * n_points

            else:
                # Absolute variance gradient
                # ∂Var/∂p_i = 2(p_i - mean) / n
                for i, p_i in enumerate(group_params):
                    grad[start + i] = (
                        self.lambda_value * 2 * (p_i - mean_val) / n_group * n_points
                    )

        return grad

    def check_constraint_violation(self, params: np.ndarray) -> dict[str, dict]:
        """Check if CV exceeds max_cv threshold.

        Parameters
        ----------
        params : np.ndarray
            Full parameter vector.

        Returns
        -------
        dict
            Dictionary of violations, empty if none.
        """
        violations = {}

        for group_idx, (start, end) in enumerate(self.group_indices):
            if start >= len(params) or end > len(params):
                continue

            group_params = params[start:end]
            mean_val = np.nanmean(group_params)
            std_val = np.nanstd(group_params)
            cv = std_val / (abs(mean_val) + 1e-10)

            if cv > self.config.max_cv:
                group_name = "contrast" if group_idx == 0 else "offset"
                violations[f"group_{group_idx}_{group_name}"] = {
                    "cv": float(cv),
                    "max_cv": self.config.max_cv,
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "min": float(np.nanmin(group_params)),
                    "max": float(np.nanmax(group_params)),
                }

        return violations

    def get_diagnostics(self) -> dict:
        """Get regularization diagnostics for logging.

        Returns
        -------
        dict
            Diagnostic information including CV values and contribution.
        """
        return {
            "enabled": self.config.enable,
            "mode": self.config.mode,
            "lambda": self.lambda_value,
            "n_phi": self.n_phi,
            "group_indices": self.group_indices,
            "last_cv_values": self._last_cv_values,
            "last_reg_contribution": self._last_reg_contribution,
            "target_cv": self.config.target_cv,
            "max_cv": self.config.max_cv,
        }

    def log_summary(self, params: np.ndarray, mse: float, n_points: int) -> None:
        """Log regularization summary.

        Parameters
        ----------
        params : np.ndarray
            Full parameter vector.
        mse : float
            Current mean squared error.
        n_points : int
            Number of data points.
        """
        if not self.config.enable:
            logger.info("Adaptive regularization: DISABLED")
            return

        # Compute regularization
        reg_term = self.compute_regularization(params, mse, n_points)
        sse = mse * n_points
        contribution_pct = 100 * reg_term / (sse + reg_term) if sse > 0 else 0

        logger.info("Adaptive Regularization Summary:")
        logger.info(f"  Mode: {self.config.mode}")
        logger.info(f"  Lambda: {self.lambda_value:.2f}")
        logger.info(f"  Regularization term: {reg_term:.2e}")
        logger.info(f"  SSE: {sse:.2e}")
        logger.info(f"  Contribution: {contribution_pct:.2f}%")

        for group_idx, cv in self._last_cv_values.items():
            group_name = "contrast" if group_idx == 0 else "offset"
            status = "OK" if cv <= self.config.max_cv else "VIOLATION"
            logger.info(f"  Group {group_idx} ({group_name}): CV={cv:.4f} [{status}]")

        violations = self.check_constraint_violation(params)
        if violations:
            logger.warning(f"  CV violations detected: {list(violations.keys())}")
