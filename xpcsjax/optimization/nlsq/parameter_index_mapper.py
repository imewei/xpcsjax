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

from xpcsjax.optimization.nlsq.per_angle_mode import PerAngleMode
from xpcsjax.optimization.nlsq.per_angle_mode import n_optimized as _n_optimized

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
            # Invariant: ``use_fourier`` is derived from
            # ``self.fourier is not None and self.fourier.use_fourier`` (see
            # the ``use_fourier`` property). Narrow for mypy.
            assert self.fourier is not None
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
            assert self.fourier is not None  # use_fourier invariant — see above
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

    @staticmethod
    def canonical(mode: str, n_phi: int, n_physics: int) -> CanonicalIndexMapper:
        """Scaling-first canonical layout authority (spec §4 Seam 2).

        The SOLE constructor for the new ``[scaling_head | physics]`` layout. Rejects
        unresolved tokens (``auto``/``fourier``/``independent``) — resolve via
        :func:`~xpcsjax.optimization.nlsq.per_angle_mode.resolve_per_angle_mode` first.
        There is no ``from_resolved``/``from_mode`` alias. Returns a
        :class:`CanonicalIndexMapper`.
        """
        return _canonical_index_mapper(mode, n_phi, n_physics)


@dataclass(frozen=True)
class CanonicalIndexMapper:
    """Scaling-first canonical layout authority (spec §4 Seam 2).

    Optimizer vector is ``[scaling_head | physics]``: the scaling head occupies
    indices ``[0, n_optimized)`` and physics the tail ``[n_optimized, vector_length)``.
    This is the single source of truth for vector length, block slices, the L3
    group indices, and the ``freeze`` flag across every execution path. Built via
    :meth:`ParameterIndexMapper.canonical`.

    Attributes
    ----------
    mode : PerAngleMode
        Resolved variant: ``"constant"``, ``"averaged"``, or ``"individual"``.
    n_phi : int
        Number of unique phi angles.
    n_physics : int
        Number of physical parameters (7 homodyne laminar_flow, 14 heterodyne).
    n_optimized : int
        Optimized scaling params: ``0`` (constant), ``2`` (averaged),
        ``2 * n_phi`` (individual).
    vector_length : int
        ``n_physics + n_optimized``.
    scaling_block : slice
        Head slice ``slice(0, n_optimized)`` (empty for ``constant``).
    physics_block : slice
        Tail slice ``slice(n_optimized, n_optimized + n_physics)``.
    group_indices : list[tuple[int, int]]
        L3 regularization groups within the scaling head:
        ``[(c_start, c_end), (o_start, o_end)]``; empty for ``constant``.
    freeze : bool
        ``True`` iff ``mode == "constant"`` (scaling frozen, not optimized).
    """

    mode: PerAngleMode
    n_phi: int
    n_physics: int
    n_optimized: int
    vector_length: int
    scaling_block: slice
    physics_block: slice
    group_indices: list[tuple[int, int]]
    freeze: bool


def _canonical_index_mapper(
    mode: str, n_phi: int, n_physics: int
) -> CanonicalIndexMapper:
    if n_phi < 1:
        raise ValueError(f"n_phi must be >= 1, got {n_phi}")
    if n_physics < 1:
        raise ValueError(f"n_physics must be >= 1, got {n_physics}")
    # Reject auto/fourier/independent: canonical layout requires a RESOLVED mode.
    if mode not in ("constant", "averaged", "individual"):
        raise ValueError(
            f"unknown per_angle_mode {mode!r}; valid: constant, averaged, individual"
        )
    resolved: PerAngleMode = mode  # type: ignore[assignment]
    n_opt = _n_optimized(resolved, n_phi)
    scaling_block = slice(0, n_opt)
    physics_block = slice(n_opt, n_opt + n_physics)
    # group_indices: contrast then offset within the scaling head.
    if n_opt == 0:
        group_indices: list[tuple[int, int]] = []
    else:
        half = n_opt // 2
        group_indices = [(0, half), (half, n_opt)]
    return CanonicalIndexMapper(
        mode=resolved,
        n_phi=n_phi,
        n_physics=n_physics,
        n_optimized=n_opt,
        vector_length=n_physics + n_opt,
        scaling_block=scaling_block,
        physics_block=physics_block,
        group_indices=group_indices,
        freeze=(resolved == "constant"),
    )


# NOTE: do NOT dynamically attach `ParameterIndexMapper.canonical = staticmethod(...)` here.
# The `canonical` staticmethod is defined IN the class body above (so mypy/Pyright/IDE
# inference resolves `.canonical(...)`); this module-level helper is what it delegates to.
# The legacy fourier/use_constant dataclass FIELDS and METHODS stay byte-untouched (Risk 2);
# adding one staticmethod is additive and does not alter them.
