"""Result container for NLSQ optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class NLSQResult:
    """Result of a heterodyne NLSQ optimization.

    Container for fitted parameters, their uncertainties, and fit-quality
    metrics produced by the heterodyne ``two_component`` solver adapters.

    Notes
    -----
    For heterodyne fits the ``parameters`` array is physics-first, laid out as
    ``[physics | contrast | offset]`` — the reverse of the homodyne-side
    :class:`xpcsjax.optimization.nlsq.results.OptimizationResult`, whose vector
    is scaling-first. Consumers that index by position must account for this.
    """

    # Core results
    parameters: np.ndarray  # Fitted parameter values
    parameter_names: list[str]  # Names in order
    success: bool  # Whether optimization succeeded
    message: str  # Status message

    # Uncertainties (from covariance matrix)
    uncertainties: np.ndarray | None = None
    covariance: np.ndarray | None = None

    # Fit quality metrics
    final_cost: float | None = None
    reduced_chi_squared: float | None = None
    n_iterations: int = 0
    n_function_evals: int = 0
    convergence_reason: str = ""

    # Residuals and Jacobian (optional, can be large)
    residuals: np.ndarray | None = None
    jacobian: np.ndarray | None = None
    fitted_correlation: np.ndarray | None = None

    # Timing
    wall_time_seconds: float | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        """Number of fitted parameters."""
        return len(self.parameters)

    @property
    def params_dict(self) -> dict[str, float]:
        """Parameters as dictionary."""
        return {name: float(self.parameters[i]) for i, name in enumerate(self.parameter_names)}

    def get_param(self, name: str) -> float:
        """Return a fitted parameter value by name.

        Parameters
        ----------
        name
            Parameter name to look up in :attr:`parameter_names`.

        Returns
        -------
        float
            The fitted value of the named parameter.

        Raises
        ------
        KeyError
            If *name* is not among :attr:`parameter_names`.
        """
        try:
            idx = self.parameter_names.index(name)
            return float(self.parameters[idx])
        except ValueError:
            raise KeyError(f"Parameter '{name}' not found") from None

    def get_uncertainty(self, name: str) -> float | None:
        """Return the 1-sigma uncertainty for a parameter by name.

        Parameters
        ----------
        name
            Parameter name to look up in :attr:`parameter_names`.

        Returns
        -------
        float or None
            The standard uncertainty, or ``None`` when uncertainties were not
            computed or *name* is not found.
        """
        if self.uncertainties is None:
            return None
        try:
            idx = self.parameter_names.index(name)
            return float(self.uncertainties[idx])
        except ValueError:
            return None

    def get_correlation_matrix(self) -> np.ndarray | None:
        """Compute the parameter correlation matrix from the covariance.

        Returns
        -------
        numpy.ndarray or None
            The correlation matrix (covariance normalised by the outer product
            of the marginal standard deviations), or ``None`` when
            :attr:`covariance` is unavailable.
        """
        if self.covariance is None:
            return None

        std = np.sqrt(np.diag(self.covariance))
        std_outer = np.outer(std, std)
        # Avoid division by zero
        std_outer = np.where(std_outer > 0, std_outer, 1.0)
        return self.covariance / std_outer

    def validate(self) -> list[str]:
        """Inspect the result and collect fit-quality warnings.

        Flags a failed solve, reduced chi-squared outside the
        ``[0.5, 2.0]`` band (possible overfit / poor fit), parameters whose
        relative uncertainty exceeds 100%, and pairs of highly correlated
        parameters (``|r| > 0.95``).

        Returns
        -------
        list of str
            Human-readable warning messages; empty when no issues are found.
        """
        warnings = []

        if not self.success:
            warnings.append(f"Optimization failed: {self.message}")

        if self.reduced_chi_squared is not None:
            if self.reduced_chi_squared > 2.0:
                warnings.append(f"Poor fit: χ²_red = {self.reduced_chi_squared:.2f} > 2")
            elif self.reduced_chi_squared < 0.5:
                warnings.append(f"Possible overfit: χ²_red = {self.reduced_chi_squared:.2f} < 0.5")

        if self.uncertainties is not None:
            for name, val, unc in zip(
                self.parameter_names, self.parameters, self.uncertainties, strict=True
            ):
                if val != 0 and abs(unc / val) > 1.0:
                    warnings.append(f"Large uncertainty: {name} = {val:.3e} ± {unc:.3e}")

        # Check for highly correlated parameters
        corr = self.get_correlation_matrix()
        if corr is not None:
            n = len(self.parameter_names)
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(corr[i, j]) > 0.95:
                        warnings.append(
                            f"Highly correlated: {self.parameter_names[i]} and "
                            f"{self.parameter_names[j]} (r = {corr[i, j]:.3f})"
                        )

        return warnings

    def summary(self) -> str:
        """Render a human-readable multi-line summary of the fit.

        Returns
        -------
        str
            A formatted block listing success status, each parameter with its
            uncertainty (when available), and the fit statistics.
        """
        lines = [
            "NLSQ Fit Result",
            "=" * 50,
            f"Success: {self.success}",
            f"Message: {self.message}",
            "",
            "Parameters:",
            "-" * 50,
        ]

        for i, name in enumerate(self.parameter_names):
            val = self.parameters[i]
            if self.uncertainties is not None:
                unc = self.uncertainties[i]
                lines.append(f"  {name:18s}: {val:12.4e} ± {unc:.2e}")
            else:
                lines.append(f"  {name:18s}: {val:12.4e}")

        lines.append("")
        lines.append("Statistics:")
        lines.append("-" * 50)

        if self.final_cost is not None:
            lines.append(f"  Final cost: {self.final_cost:.6e}")
        if self.reduced_chi_squared is not None:
            lines.append(f"  Reduced χ²: {self.reduced_chi_squared:.4f}")
        lines.append(f"  Iterations: {self.n_iterations}")
        lines.append(f"  Function evals: {self.n_function_evals}")
        if self.wall_time_seconds is not None:
            lines.append(f"  Wall time: {self.wall_time_seconds:.2f} s")

        return "\n".join(lines)
