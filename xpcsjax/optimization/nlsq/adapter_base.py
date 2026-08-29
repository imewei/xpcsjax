"""Abstract base class for NLSQ adapters (FR-012).

Shared ABC for NLSQAdapter and NLSQWrapper; enforces the `fit()` contract
and holds the shared per_angle_scaling=False rejection message.

Created as part of architecture refactoring (T059-T061).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# Shared by NLSQAdapter.fit() (adapter.py) and NLSQWrapper.fit() (wrapper.py) — both reject
# the removed per_angle_scaling=False legacy mode with this exact message.
PER_ANGLE_SCALING_REMOVED_MSG = (
    "per_angle_scaling=False is deprecated and removed. "
    "Use per_angle_scaling=True (default) for physically correct behavior."
)


class NLSQAdapterBase(ABC):
    """Abstract base class for NLSQ optimization adapters.

    Subclasses must implement the `fit()` method.
    """

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        """Fit the model to data.

        Must be implemented by subclasses.
        """
        ...


__all__ = ["NLSQAdapterBase", "PER_ANGLE_SCALING_REMOVED_MSG"]
