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
    """Abstract base class for heterodyne NLSQ optimization adapters.

    Adapters wrap different optimization backends (the ``nlsq`` library, a
    scipy fallback, etc.) behind a single ``fit`` interface so the heterodyne
    ``two_component`` solver paths can dispatch to a backend without depending
    on its concrete API.

    Notes
    -----
    This is the heterodyne-side base class returning an
    :class:`~xpcsjax.optimization.nlsq.heterodyne_results.NLSQResult`; it is
    distinct from the homodyne-side
    :class:`xpcsjax.optimization.nlsq.adapter_base.NLSQAdapterBase`, which
    provides shared data-preparation / covariance helpers instead.
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
        """Run the optimization and return a populated result.

        Parameters
        ----------
        residual_fn
            Callable that computes the residual vector for a given parameter
            vector.
        initial_params
            Initial parameter values used to seed the solve.
        bounds
            ``(lower_bounds, upper_bounds)`` arrays defining the box constraints.
        config
            Optimization configuration controlling the solver and tolerances.
        jacobian_fn
            Optional callable that computes the analytic Jacobian; ``None``
            falls back to a finite-difference approximation.

        Returns
        -------
        NLSQResult
            The optimization result with fitted parameters and diagnostics.
        """
        ...

    @abstractmethod
    def supports_bounds(self) -> bool:
        """Return whether this adapter supports bounded (box-constrained) optimization."""
        ...

    @abstractmethod
    def supports_jacobian(self) -> bool:
        """Return whether this adapter can consume an analytic Jacobian."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the underlying optimization backend."""
        ...
