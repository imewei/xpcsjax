"""Centralized index mapping for anti-degeneracy layers.

This module provides the ParameterIndexMapper class which ensures consistent
index ranges regardless of whether Fourier reparameterization is active.
This is the single source of truth for parameter group boundaries.

Created: 2025-12-31
Feature: 001-fix-nlsq-anti-degeneracy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xpcsjax.optimization.nlsq.fourier_reparam import FourierReparameterizer


@dataclass
class ParameterIndexMapper:
    """Centralized index mapping for anti-degeneracy layers.

    Provides consistent index ranges regardless of whether Fourier
    reparameterization or constant scaling is active. This class is the
    single source of truth for parameter group boundaries.

    Parameters
    ----------
    n_phi : int
        Number of unique phi angles.
    n_physical : int
        Number of physical parameters (typically 7 for laminar_flow mode).
    fourier : FourierReparameterizer | None
        Reference to Fourier reparameterizer if Layer 1 is active.
    use_constant : bool
        Whether constant scaling mode is active (single contrast/offset
        shared across all angles).

    Attributes
    ----------
    n_per_angle_total : int
        Total number of per-angle parameters (Fourier coefficients, raw, or 2).
    n_per_group : int
        Number of parameters per group (contrast or offset).
    use_fourier : bool
        Whether Fourier reparameterization is active.
    use_constant : bool
        Whether constant scaling mode is active.
    total_params : int
        Total number of parameters.
    mode_name : str
        Human-readable name of current mode ("constant", "fourier", or "individual").

    Examples
    --------
    >>> # Constant mode (23 phi angles)
    >>> mapper = ParameterIndexMapper(n_phi=23, n_physical=7, use_constant=True)
    >>> mapper.get_group_indices()
    [(0, 1), (1, 2)]
    >>> mapper.n_per_angle_total
    2
    >>> mapper.mode_name
    'constant'

    >>> # Non-Fourier mode (23 phi angles)
    >>> mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=None)
    >>> mapper.get_group_indices()
    [(0, 23), (23, 46)]
    >>> mapper.n_per_angle_total
    46

    >>> # Fourier mode (23 phi angles, order=2)
    >>> mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=fourier_obj)
    >>> mapper.get_group_indices()
    [(0, 5), (5, 10)]
    >>> mapper.n_per_angle_total
    10
    """

    n_phi: int
    n_physical: int
    fourier: FourierReparameterizer | None = None
    use_constant: bool = False

    def __post_init__(self) -> None:
        """Validate inputs and cache computed values."""
        if self.n_phi < 1:
            raise ValueError(f"n_phi must be >= 1, got {self.n_phi}")
        if self.n_physical < 1:
            raise ValueError(f"n_physical must be >= 1, got {self.n_physical}")
        # T011: Mutual exclusion - cannot use both Fourier and constant mode
        if self.use_constant and self.fourier is not None and self.fourier.use_fourier:
            raise ValueError(
                "Cannot use both Fourier reparameterization and constant scaling mode. "
                "Choose one: set use_constant=False or fourier=None."
            )

    @property
    def use_fourier(self) -> bool:
        """Check if Fourier reparameterization is active."""
        return self.fourier is not None and self.fourier.use_fourier

    @property
    def n_per_group(self) -> int:
        """Get number of parameters per group (contrast or offset).

        Returns
        -------
        int
            1 for constant mode, n_coeffs for Fourier, n_phi for individual.
        """
        # T010: Return 1 for constant mode (single value per group)
        if self.use_constant:
            return 1
        if self.use_fourier:
            return self.fourier.n_coeffs_per_param
        return self.n_phi

    @property
    def mode_name(self) -> str:
        """Get human-readable name of current mode.

        Returns
        -------
        str
            "constant", "fourier", or "individual"
        """
        if self.use_constant:
            return "constant"
        if self.use_fourier:
            return "fourier"
        return "individual"

    @property
    def n_per_angle_total(self) -> int:
        """Get total number of per-angle parameters (scaling params)."""
        if self.use_constant:
            return 2  # One contrast + one offset
        if self.use_fourier:
            return self.fourier.n_coeffs
        return 2 * self.n_phi

    @property
    def total_params(self) -> int:
        """Get total number of parameters."""
        return self.n_per_angle_total + self.n_physical

    def get_group_indices(self) -> list[tuple[int, int]]:
        """Get (start, end) tuples for contrast and offset parameter groups.

        Returns
        -------
        list[tuple[int, int]]
            Two tuples: [(contrast_start, contrast_end), (offset_start, offset_end)]

        Notes
        -----
        - Contrast group: indices [0, n_per_group)
        - Offset group: indices [n_per_group, 2*n_per_group)
        """
        n = self.n_per_group
        return [(0, n), (n, 2 * n)]

    def get_physical_indices(self) -> list[int]:
        """Get indices of physical parameters.

        Returns
        -------
        list[int]
            Indices of physical parameters in the full parameter vector.
        """
        start = self.n_per_angle_total
        return list(range(start, start + self.n_physical))

    def get_per_angle_indices(self) -> list[int]:
        """Get indices of all per-angle parameters.

        Returns
        -------
        list[int]
            Indices of per-angle parameters (contrast + offset).
        """
        return list(range(self.n_per_angle_total))

    def validate_indices(self, params: np.ndarray) -> bool:
        """Validate that group indices are within parameter vector bounds.

        Parameters
        ----------
        params : np.ndarray
            Full parameter vector.

        Returns
        -------
        bool
            True if all indices are valid, False otherwise.

        Raises
        ------
        ValueError
            If indices are out of bounds (with descriptive message).
        """
        n_params = len(params)

        for i, (start, end) in enumerate(self.get_group_indices()):
            if start < 0:
                raise ValueError(f"Group {i} start index {start} is negative")
            if end > n_params:
                raise ValueError(
                    f"Group {i} end index {end} exceeds parameter count {n_params}. "
                    f"This may indicate a Fourier/regularization mode mismatch."
                )
            if start >= end:
                raise ValueError(f"Group {i} has invalid range [{start}, {end})")

        return True

    def get_diagnostics(self) -> dict:
        """Get diagnostic information for logging.

        Returns
        -------
        dict
            Diagnostic information including mode, counts, and indices.
        """
        return {
            "mode_name": self.mode_name,
            "use_constant": self.use_constant,
            "use_fourier": self.use_fourier,
            "n_phi": self.n_phi,
            "n_physical": self.n_physical,
            "n_per_group": self.n_per_group,
            "n_per_angle_total": self.n_per_angle_total,
            "total_params": self.total_params,
            "group_indices": self.get_group_indices(),
            "physical_indices": self.get_physical_indices(),
        }

    def get_covariance_slice_indices(self) -> tuple[slice, slice]:
        """Get slice indices for covariance matrix transformation.

        Returns slices for extracting per-angle and physical parameter
        blocks from a covariance matrix.

        Returns
        -------
        tuple[slice, slice]
            (per_angle_slice, physical_slice) for indexing covariance matrices.
        """
        per_angle_slice = slice(0, self.n_per_angle_total)
        physical_slice = slice(self.n_per_angle_total, self.total_params)
        return per_angle_slice, physical_slice
