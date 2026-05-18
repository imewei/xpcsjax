"""Abstract base class for NLSQ adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_results import NLSQResult


class NLSQAdapterBase(ABC):
    """Abstract base class for NLSQ optimization adapters.

    Adapters wrap different optimization backends (scipy, nlsq library, etc.)
    with a consistent interface.
    """

    @abstractmethod
    def fit(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        config: NLSQConfig,
        jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> NLSQResult:
        """Run optimization.

        Args:
            residual_fn: Function that computes residuals given parameters
            initial_params: Initial parameter values
            bounds: (lower_bounds, upper_bounds) arrays
            config: Optimization configuration
            jacobian_fn: Optional function that computes Jacobian

        Returns:
            NLSQResult with optimization results
        """
        ...

    @abstractmethod
    def supports_bounds(self) -> bool:
        """Whether this adapter supports bounded optimization."""
        ...

    @abstractmethod
    def supports_jacobian(self) -> bool:
        """Whether this adapter supports analytic Jacobian."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the optimization backend."""
        ...
